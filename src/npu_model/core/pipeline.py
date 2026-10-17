from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from npu_model.core.cache import ConversionCache, compute_cache_key
from npu_model.core.errors import NpuModelError
from npu_model.core.manifest import collect_files, write_manifest
from npu_model.core.registry import Registry
from npu_model.core.types import ConvertMode, ConvertResult, ExplainPlan, GraphBundle
from npu_model.inspect.hf_inspector import inspect_hf_style_dir
from npu_model.sources.local import materialize_local
from npu_model.sources.hf import parse_hf_spec, materialize_hf
from npu_model.adapters.auto import select_adapter


# ---------------------------------------------------------------------------
# Source materialization (single call, reused across explain + convert)
# ---------------------------------------------------------------------------

def _materialize(input_spec: str, cache_dir: Optional[Path]) -> tuple[Path, dict[str, Any]]:
    if input_spec.startswith("hf:"):
        spec = parse_hf_spec(input_spec)
        model_dir = materialize_hf(spec, cache_dir)
        return model_dir, {"type": "hf", "repo_id": spec.repo_id, "revision": spec.revision}
    else:
        model_dir = materialize_local(Path(input_spec))
        return model_dir, {"type": "local", "path": str(model_dir)}


def _resolve_mode(mode: str) -> ConvertMode:
    try:
        return ConvertMode(mode)
    except ValueError:
        raise NpuModelError(
            stage="pipeline",
            reason_code="UNKNOWN_MODE",
            message=f"Unknown convert mode: {mode}",
            hint=f"Supported modes: {[m.value for m in ConvertMode]}",
        )


# ---------------------------------------------------------------------------
# explain_plan
# ---------------------------------------------------------------------------

def explain_plan(
    input_spec: str,
    backend_id: str,
    target: str,
    runtime_format_id: str,
    cache_dir: Optional[Path],
    registry: Registry,
    quantizer_id: str = "passthrough",
    mode: str = "export",
) -> ExplainPlan:
    convert_mode = _resolve_mode(mode)
    model_dir, source = _materialize(input_spec, cache_dir)
    mi = inspect_hf_style_dir(model_dir, source=source)

    adapter_id = select_adapter(registry, mi)

    backend = registry.backends.get(backend_id)
    if backend is None:
        raise NpuModelError(
            stage="backend",
            reason_code="UNKNOWN_BACKEND",
            message=f"Unknown backend: {backend_id}",
            hint=f"Available: {sorted(registry.backends.keys())}",
        )

    runtime = registry.runtime_formats.get(runtime_format_id)
    if runtime is None:
        raise NpuModelError(
            stage="runtime",
            reason_code="UNKNOWN_RUNTIME_FORMAT",
            message=f"Unknown runtime format: {runtime_format_id}",
            hint=f"Available: {sorted(registry.runtime_formats.keys())}",
        )

    if quantizer_id not in registry.quantizers:
        raise NpuModelError(
            stage="quant",
            reason_code="UNKNOWN_QUANTIZER",
            message=f"Unknown quantizer: {quantizer_id}",
            hint=f"Available: {sorted(registry.quantizers.keys())}",
        )

    ts = backend.resolve_target(target, env=dict(os.environ))

    return ExplainPlan(
        input_spec=input_spec,
        materialized_dir=model_dir,
        model_type=mi.model_type,
        architectures=mi.architectures,
        adapter_id=adapter_id,
        backend_id=backend_id,
        target_name=ts.name,
        runtime_format_id=runtime_format_id,
        quantizer_id=quantizer_id,
        convert_mode=convert_mode.value,
    )


# ---------------------------------------------------------------------------
# convert_model
# ---------------------------------------------------------------------------

def convert_model(
    input_spec: str,
    out_dir: Path,
    backend_id: str,
    target: str,
    runtime_format_id: str,
    quantizer_id: str | None,
    cache_dir: Optional[Path],
    registry: Registry,
    mode: str = "export",
    compile_config: Optional[dict[str, Any]] = None,
    export_options: Optional[dict[str, Any]] = None,
    pack_ollama_name: Optional[str] = None,
    pack_ollama_opts: Optional[dict[str, Any]] = None,
    use_cache: bool = True,
    keep_work: bool = False,
    stop_after: Optional[str] = None,
    quantizer_was_auto: bool = False,
) -> ConvertResult:
    convert_mode = _resolve_mode(mode)
    if stop_after not in (None, "export", "quantize"):
        raise NpuModelError(
            stage="pipeline", reason_code="UNKNOWN_STOP_STAGE",
            message=f"Unknown stop-after stage: {stop_after}",
            hint="Use export or quantize.",
        )
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Materialize (once) ----
    model_dir, source = _materialize(input_spec, cache_dir)
    mi = inspect_hf_style_dir(model_dir, source=source)

    # ---- 2. Select plugins ----
    adapter_id = select_adapter(registry, mi)
    adapter = registry.adapters[adapter_id]

    backend = registry.backends.get(backend_id)
    if backend is None:
        raise NpuModelError(
            stage="backend", reason_code="UNKNOWN_BACKEND",
            message=f"Unknown backend: {backend_id}",
            hint=f"Available: {sorted(registry.backends.keys())}",
        )
    runtime = registry.runtime_formats.get(runtime_format_id)
    if runtime is None:
        raise NpuModelError(
            stage="runtime", reason_code="UNKNOWN_RUNTIME_FORMAT",
            message=f"Unknown runtime format: {runtime_format_id}",
            hint=f"Available: {sorted(registry.runtime_formats.keys())}",
        )
    cc = compile_config or {}
    strategy = cc.get("strategy", "passthrough")
    _SUPPORTED_OLIVE_FAMILIES = {"phi", "phi3", "llama"}
    _model_family = (mi.model_type or "").lower()
    effective_quantizer_id = quantizer_id or "passthrough"
    if (
        quantizer_was_auto
        and backend_id == "qnn"
        and strategy == "context-cache"
    ):
        effective_quantizer_id = (
            "olive-qnn-llm" if _model_family in _SUPPORTED_OLIVE_FAMILIES else "qnn-qdq"
        )
        import logging
        logging.getLogger("npu_model").info(
            "Auto-selected quantizer for model_family='%s': %s",
            _model_family or "unknown",
            effective_quantizer_id,
        )

    quantizer = registry.quantizers.get(effective_quantizer_id)
    if quantizer is None:
        raise NpuModelError(
            stage="quant", reason_code="UNKNOWN_QUANTIZER",
            message=f"Unknown quantizer: {effective_quantizer_id}",
            hint=f"Available: {sorted(registry.quantizers.keys())}",
        )

    # Resolve backend target early (needed for cache key)
    ts = backend.resolve_target(target, env=dict(os.environ))
    is_npu_target = (
        backend_id == "qnn"
        and ts.params.get("backend_type", "").lower() in ("htp", "")
    )
    effective_export_options = dict(export_options or {})
    if convert_mode == ConvertMode.EXPORT:
        from npu_model.exporters.ort_genai_builder import resolve_export_precision

        effective_export_options["precision"] = resolve_export_precision(
            effective_export_options.get("precision"), effective_quantizer_id,
        )

    # ---- 2b. Check cache ----
    import npu_model
    cache = ConversionCache(out_dir)
    cache_key = compute_cache_key(
        input_spec=input_spec,
        input_revision=source.get("revision"),
        adapter_id=adapter_id,
        mode=convert_mode.value,
        backend_id=backend_id,
        target_name=ts.name,
        target_params=ts.params,
        compile_strategy=strategy,
        compile_config=cc,
        quantizer_id=effective_quantizer_id,
        runtime_format_id=runtime_format_id,
        tool_version=npu_model.__version__,
        export_options=effective_export_options,
        shape_options={"num_ctx": (pack_ollama_opts or {}).get("num_ctx", 512)},
    )

    if use_cache and stop_after is None:
        cached = cache.get(cache_key)
        if cached is not None:
            bundle_dir, manifest_path = cache.restore(cached, out_dir)
            runtime.validate_layout(bundle_dir)
            if pack_ollama_name and is_npu_target and strategy == "context-cache":
                _validate_npu_bundle(bundle_dir)
            pack_dir = _maybe_pack_ollama(
                bundle_dir, pack_ollama_name, pack_ollama_opts, out_dir,
            )
            return ConvertResult(
                bundle_dir=bundle_dir,
                manifest_path=manifest_path,
                pack_dir=pack_dir,
            )

    # ---- 3. Adapter export / import ----
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    adapter_out = work / f"adapter_{adapter.id}"
    export_config: dict[str, Any] = {"mode": convert_mode.value, "model_info": mi}
    export_config.update(effective_export_options)

    try:
        graphs = adapter.export(model_dir, adapter_out, export_config=export_config)
    except NpuModelError:
        raise
    except Exception as e:
        raise NpuModelError(
            stage="export",
            reason_code="ADAPTER_EXPORT_FAILED",
            message=f"Adapter '{adapter_id}' export failed: {e}",
            hint=(
                f"Mode: {convert_mode.value}\n"
                f"Input: {model_dir}\n"
                "If using export mode, ensure deps are installed: pip install npu-model[export]\n"
                "For prebuilt import: --mode prebuilt-ort-genai"
            ),
            cause=e,
        ) from e

    # ---- 4. Bridge + normalize tokenizer for runtime compatibility ----
    import logging
    _log = logging.getLogger("npu_model")

    # 4a. Bridge: copy missing tokenizer files from HF source into export output
    from npu_model.core.tokenizer_bridge import bridge_tokenizer_files

    bridged = bridge_tokenizer_files(
        src_model_dir=model_dir, dst_tokenizer_dir=graphs.tokenizer_dir,
    )
    if bridged:
        _log.info("Tokenizer files bridged from source: %s", ", ".join(bridged))

    # 4b. Normalize: rewrite unsupported tokenizer_class if possible
    from npu_model.core.tokenizer_norm import normalize_tokenizer_config

    tok_norm = normalize_tokenizer_config(
        graphs.tokenizer_dir, model_type=mi.model_type,
    )
    if tok_norm.changed:
        _log.info(
            "Tokenizer normalized: %s -> %s", tok_norm.original_class, tok_norm.new_class,
        )
    for w in tok_norm.warnings:
        _log.warning("Tokenizer: %s", w)

    # 4c. If packing for Ollama, fail hard on unsupported tokenizer
    if pack_ollama_name and not tok_norm.changed and tok_norm.original_class:
        from npu_model.core.tokenizer_norm import _is_unsupported
        if _is_unsupported(tok_norm.original_class):
            raise NpuModelError(
                stage="tokenizer",
                reason_code="TOKENIZER_UNSUPPORTED_FOR_OLLAMA",
                message=(
                    f"tokenizer_class='{tok_norm.original_class}' is not supported "
                    f"by ORT GenAI on this platform, and normalization could not fix it."
                ),
                hint=(
                    "Ensure tokenizer.model (SentencePiece) is present in the source model.\n"
                    "The tokenizer bridge attempted to copy it from the HF snapshot but "
                    "it was not found. Check that the source model includes tokenizer.model."
                ),
            )

    # ---- 4d. Stop after export (for staged workflows) ----
    if stop_after == "export":
        from npu_model.core.handoff import create_handoff_bundle
        handoff = create_handoff_bundle(
            graphs=graphs,
            out_dir=out_dir / "handoff_bundle",
            stopped_after="export",
            metadata={
                "input_spec": input_spec,
                "adapter_id": adapter_id,
                "model_type": mi.model_type,
                "model_family": (mi.model_type or "").lower(),
                "quantization_format": None,
                "split_count": len(graphs.graphs),
                "layout": "monolith" if len(graphs.graphs) == 1 else "split",
                "graph_metadata": graphs.metadata,
            },
        )
        _log.info("Stopped after export. Handoff bundle: %s", handoff.bundle_dir)
        return ConvertResult(
            bundle_dir=handoff.bundle_dir,
            manifest_path=handoff.manifest_path,
        )

    # ---- 4e. Policy: QNN/HTP + context-cache requires proper quantization ----
    if is_npu_target and strategy == "context-cache" and effective_quantizer_id == "passthrough":
        if pack_ollama_name:
            raise NpuModelError(
                stage="pipeline",
                reason_code="NPU_REQUIRES_QUANTIZATION",
                message=(
                    "QNN HTP context-cache compilation requires QDQ quantization. "
                    "Passthrough quantizer cannot produce NPU-ready artifacts."
                ),
                hint=(
                    "Use --quant qnn-qdq with calibration data:\n"
                    "  npu-model convert --quant qnn-qdq --calib-prompts prompts.txt --compile-strategy context-cache ..."
                ),
            )
        else:
            _log.warning(
                "QNN HTP context-cache compilation with passthrough quantizer. "
                "The model may not be quantized — HTP requires QDQ quantization."
            )

    # ---- 4f. Policy: warn LLM families using generic QDQ for context-cache ----
    # The generic qnn-qdq path is experimental for decoder LLMs.  Phi, Phi-3,
    # and Llama-family models need LLM-specific splitting/shaping (Olive).
    _LLM_FAMILY_TYPES = {"phi", "phi3", "llama", "llama2", "llama3", "mistral", "gemma"}
    if (
        is_npu_target
        and strategy == "context-cache"
        and effective_quantizer_id == "qnn-qdq"
        and _model_family in _LLM_FAMILY_TYPES
    ):
        _log.warning(
            "Model family '%s' is using the generic qnn-qdq quantizer with "
            "context-cache compilation.  This path is EXPERIMENTAL for LLMs and "
            "may produce oversized or invalid context artifacts.\n"
            "For production deployment, use: --quant olive-qnn-llm",
            _model_family,
        )

    # ---- 5. Shape fixing (required for QNN HTP) ----
    num_ctx_val = (pack_ollama_opts or {}).get("num_ctx", 512)

    if is_npu_target and strategy == "context-cache":
        from npu_model.core.shapes import has_dynamic_shapes, fix_dynamic_shapes

        num_ctx = num_ctx_val
        shape_fix_dir = work / "shape_fixed"
        shape_fix_dir.mkdir(exist_ok=True)
        new_graphs: dict[str, Path] = {}
        for gname, gpath in graphs.graphs.items():
            try:
                dynamic = has_dynamic_shapes(gpath)
            except (ImportError, OSError) as e:
                raise NpuModelError(
                    stage="shapes", reason_code="MISSING_SHAPE_DEPS",
                    message=f"Cannot check mandatory QNN input shapes: {e}",
                    hint="Install a compatible onnx build before context-cache compilation.",
                    cause=e,
                ) from e
            if dynamic:
                _log.info("Fixing dynamic shapes in %s: %s", gpath.name, dynamic)
                result = fix_dynamic_shapes(
                    gpath, shape_fix_dir / gpath.name,
                    batch_size=1, sequence_length=num_ctx,
                )
                if result.fixed and result.output_path:
                    new_graphs[gname] = result.output_path
                    for w in result.warnings:
                        _log.warning("Shape fix: %s", w)
                else:
                    raise NpuModelError(
                        stage="shapes", reason_code="SHAPE_FIX_FAILED",
                        message=f"Could not fix shapes for {gpath.name}.",
                        hint="\n".join(result.errors),
                    )
                remaining = has_dynamic_shapes(new_graphs[gname])
                if remaining:
                    raise NpuModelError(
                        stage="shapes", reason_code="DYNAMIC_SHAPES_REMAIN",
                        message=f"Unfixed input dimensions in {gpath.name}: {remaining}",
                        hint="Provide an already-shaped graph; no static-shape claim can be made.",
                    )
            else:
                new_graphs[gname] = gpath
        graphs = GraphBundle(
            graphs=new_graphs,
            tokenizer_dir=graphs.tokenizer_dir,
            extra_files=graphs.extra_files,
            metadata=graphs.metadata,
        )

    # ---- 6. Auto-calibration + Quantize ----
    quant_config: dict[str, Any] = {}
    quant_config["model_family"] = _model_family
    quant_config["model_type"] = mi.model_type
    quant_config["input_spec"] = input_spec
    quant_config["model_dir"] = str(model_dir)
    quant_config["model_info"] = {
        "source": dict(mi.source),
        "model_type": mi.model_type,
        "architectures": list(mi.architectures),
    }
    if export_options and export_options.get("olive_python"):
        quant_config["olive_python"] = export_options.get("olive_python")

    if effective_quantizer_id != "passthrough":
        # Check model size — quantization loads the entire model into RAM.
        # On memory-constrained devices, this silently crashes (SIGSEGV / OOM).
        total_model_bytes = sum(
            p.stat().st_size for p in graphs.graphs.values() if p.exists()
        )
        for gpath in graphs.graphs.values():
            data_file = gpath.parent / f"{gpath.name}.data"
            if data_file.exists():
                total_model_bytes += data_file.stat().st_size

        import os as _os
        try:
            sysconf = getattr(_os, "sysconf")
            mem_info = sysconf("SC_PAGE_SIZE") * sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError):
            # Windows — use ctypes to get total physical memory
            mem_info = None
            try:
                import ctypes
                kernel32 = getattr(ctypes, "windll").kernel32

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                mem_status = MEMORYSTATUSEX()
                mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                    mem_info = mem_status.ullTotalPhys
            except Exception:
                pass

        model_gb = total_model_bytes / (1024**3)
        required_ram_gb = model_gb * 2.5
        system_ram_gb = mem_info / (1024**3) if mem_info else None

        # Only block if we can detect RAM and model won't fit
        if system_ram_gb is not None and required_ram_gb > system_ram_gb:
            _log.warning(
                "Model size is %.1f GB (needs ~%.0f GB RAM). "
                "System has %.0f GB total RAM.",
                model_gb, required_ram_gb, system_ram_gb,
            )
            _input_escaped = input_spec.replace("'", "\\'")
            raise NpuModelError(
                stage="quant",
                reason_code="MODEL_TOO_LARGE_FOR_QUANTIZATION",
                message=(
                    f"Model is {model_gb:.1f} GB — quantization requires ~{required_ram_gb:.0f} GB RAM "
                    f"but this system has {system_ram_gb:.0f} GB."
                ),
                hint=(
                    "Use a staged workflow across two machines:\n"
                    "\n"
                    "Step 1 — On a machine with enough RAM, export + quantize:\n"
                    f"  npu-model convert --input '{_input_escaped}' "
                    f"--out .\\handoff --stop-after quantize --keep-work\n"
                    "\n"
                    "Step 2 — Copy the handoff_bundle/ folder to the target device.\n"
                    "\n"
                    "Step 3 — On the target device (with QNN EP), compile + pack:\n"
                    f"  npu-model compile-context --input .\\handoff\\handoff_bundle "
                    f"--out .\\compiled\n"
                    f"  npu-model pack-ollama --input .\\compiled "
                    f"--name <ollama-tag> --out .\\publish\n"
                    "\n"
                    "Or use a prebuilt model: --mode prebuilt-ort-genai"
                ),
            )
        elif model_gb > 4.0:
            _log.warning(
                "Model size is %.1f GB. Quantization requires loading the entire model "
                "into RAM (typically 2-3x model size = ~%.0f GB). "
                "If this crashes silently, use --stop-after export for staged workflow.",
                model_gb, required_ram_gb,
            )

        # Check if the quantizer needs calibration data
        requires_calib = getattr(quantizer, "requires_calibration", True)
        if requires_calib:
            from npu_model.calib.prompt_source import get_prompt_source
            from npu_model.calib.data_reader import build_calibration_reader

            calib_source = (export_options or {}).get("calib_source", "builtin:mixed_small")
            calib_prompts_file = (export_options or {}).get("calib_prompts_file")
            calib_samples = int((export_options or {}).get("calib_samples", 64))
            calib_maxlen = int((export_options or {}).get("calib_maxlen", min(num_ctx_val, 256)))

            prompt_src = get_prompt_source(calib_source, calib_prompts_file)
            prompts = prompt_src.load()

            # Pick the first graph to inspect input names/dtypes
            first_graph_path = next(iter(graphs.graphs.values()))

            _log.info(
                "Building calibration data: %d samples, maxlen=%d, source=%s",
                calib_samples, calib_maxlen,
                calib_prompts_file or calib_source,
            )
            reader = build_calibration_reader(
                prompts=prompts,
                tokenizer_dir=graphs.tokenizer_dir,
                onnx_path=first_graph_path,
                num_samples=calib_samples,
                max_seq_len=calib_maxlen,
            )
            quant_config["calibration_data_reader"] = reader

    graphs = quantizer.apply(graphs, quant_config=quant_config)

    # ---- 6b. Stop after quantize (for staged workflows) ----
    if stop_after == "quantize":
        quant_format = (
            "qdq"
            if effective_quantizer_id in {"qnn-qdq", "olive-qnn-llm"}
            else effective_quantizer_id
        )
        split_count = int(graphs.metadata.get("split_count", len(graphs.graphs)))
        layout = str(graphs.metadata.get("layout", "split" if split_count > 1 else "monolith"))
        from npu_model.core.handoff import create_handoff_bundle
        handoff = create_handoff_bundle(
            graphs=graphs,
            out_dir=out_dir / "handoff_bundle",
            stopped_after="quantize",
            metadata={
                "input_spec": input_spec,
                "adapter_id": adapter_id,
                "model_type": mi.model_type,
                "model_family": (mi.model_type or "").lower(),
                "quantizer_id": effective_quantizer_id,
                "quantization_format": quant_format,
                "split_count": split_count,
                "layout": layout,
                "graph_metadata": graphs.metadata,
            },
        )
        _log.info("Stopped after quantize. Handoff bundle: %s", handoff.bundle_dir)
        _log.info(
            "Next: transfer handoff_bundle/ to target device and run:\n"
            "  npu-model compile-context --input %s --out <compiled_dir>",
            handoff.bundle_dir,
        )
        return ConvertResult(
            bundle_dir=handoff.bundle_dir,
            manifest_path=handoff.manifest_path,
        )

    # ---- 7. Backend resolve + compile/prepare ----
    backend_out = work / f"backend_{backend_id}"
    try:
        if strategy != "passthrough" and hasattr(backend, "compile"):
            prepared = backend.compile(graphs, backend_out, target=ts, compile_config=cc)
        else:
            prepared = backend.prepare(graphs, backend_out, target=ts, backend_config=cc)
    except NpuModelError:
        raise
    except Exception as e:
        # Enhanced error with diagnostics
        diag_lines = [
            f"Backend: {backend_id}",
            f"Target: {ts.normalized_repr()}",
            f"Strategy: {strategy}",
        ]
        if hasattr(backend, "detect_environment"):
            caps = backend.detect_environment()
            diag_lines.append(f"Compile available: {caps.compile_available}")
            for d in caps.diagnostics:
                diag_lines.append(f"  {d}")
        raise NpuModelError(
            stage="backend",
            reason_code="BACKEND_PREPARE_FAILED",
            message=f"Backend '{backend_id}' preparation failed: {e}",
            hint="\n".join(diag_lines),
            cause=e,
        ) from e

    # ---- 6. Assemble runtime bundle ----
    bundle_dir = runtime.assemble(prepared, graphs.tokenizer_dir, out_dir, format_config={})

    # ---- 7. Manifest ----
    plan = ExplainPlan(
        input_spec=input_spec,
        materialized_dir=model_dir,
        model_type=mi.model_type,
        architectures=mi.architectures,
        adapter_id=adapter_id,
        backend_id=backend_id,
        target_name=ts.name,
        runtime_format_id=runtime_format_id,
        quantizer_id=effective_quantizer_id,
        convert_mode=convert_mode.value,
    )
    manifest = {
        "tool": {"name": "npu-model", "version": npu_model.__version__},
        "input": source,
        "plan": asdict(plan),
        "backend_metadata": prepared.backend_metadata,
        "bundle": {"format": runtime_format_id, "path": str(bundle_dir)},
        "files": collect_files(bundle_dir),
    }
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    # Cached results must not bypass required device preflight.
    if pack_ollama_name and is_npu_target and strategy == "context-cache":
        _validate_npu_bundle(bundle_dir)

    # ---- 7b. Store in cache ----
    if use_cache:
        cache.put(cache_key, bundle_dir, manifest_path, meta={
            "input_spec": input_spec,
            "adapter_id": adapter_id,
            "backend_id": backend_id,
            "target": ts.name,
            "strategy": strategy,
        })

    # ---- 9. Optional: pack for Ollama ----
    pack_dir = _maybe_pack_ollama(bundle_dir, pack_ollama_name, pack_ollama_opts, out_dir)

    # ---- 9. Clean up work dir (unless --keep-work) ----
    if not keep_work and work.exists():
        shutil.rmtree(work)

    return ConvertResult(bundle_dir=bundle_dir, manifest_path=manifest_path, pack_dir=pack_dir)


def _validate_npu_bundle(bundle_dir: Path) -> None:
    from npu_model.validate.npu_strict import validate_npu_strict

    result = validate_npu_strict(bundle_dir)
    if not result.passed:
        raise NpuModelError(
            stage="validate", reason_code="NPU_VALIDATION_FAILED",
            message="Strict QNN preflight failed or incomplete; NPU execution is not established.",
            hint="\n".join(
                f"  [{c.status}] {c.name}: {c.detail}"
                for c in result.checks if c.status in ("FAIL", "SKIP")
            ),
        )


def _maybe_pack_ollama(
    bundle_dir: Path,
    name: Optional[str],
    opts: Optional[dict[str, Any]],
    out_dir: Path,
) -> Path | None:
    if not name:
        return None
    from npu_model.packagers.ollama import pack_for_ollama
    opts = opts or {}
    result = pack_for_ollama(
        bundle_dir=bundle_dir,
        model_name=name,
        out_dir=out_dir / "ollama_publish",
        num_ctx=opts.get("num_ctx", 512),
        num_predict=opts.get("num_predict", 128),
    )
    return result.pack_dir
