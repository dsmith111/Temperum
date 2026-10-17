from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.backends.base import Backend
from npu_model.core.errors import NpuModelError
from npu_model.core.types import BackendCapabilities, GraphBundle, BackendPreparedBundle, TargetSpec

_log = logging.getLogger("npu_model.backend.qnn")


@dataclass
class QnnBackend(Backend):
    """
    Backend plugin: QNN.

    All QNN/HTP/Snapdragon-specific knobs live here behind TargetSpec.params.
    Core pipeline treats params as opaque.

    Compilation strategies (selected via compile_config["strategy"]):
      - "ort-ep-context": use ORT QNN EP context caching to generate compiled artifacts
      - "passthrough":    copy graphs as-is (default / fallback)
    """
    id: str = "qnn"

    # -------------------------------------------------------------------
    # resolve_target
    # -------------------------------------------------------------------

    def resolve_target(self, target: str, env: dict[str, str]) -> TargetSpec:
        name = target or "auto"
        params = {
            "backend_type": env.get("NPU_QNN_BACKEND_TYPE", "htp"),
            "backend_path": env.get("NPU_QNN_BACKEND_PATH", "QnnHtp.dll"),
        }
        return TargetSpec(backend_id=self.id, name=name, params=params)

    # -------------------------------------------------------------------
    # detect_environment
    # -------------------------------------------------------------------

    def detect_environment(self) -> BackendCapabilities:
        diag: list[str] = []
        compile_ok = False
        runtime_ok = False

        # Check if onnxruntime-qnn is importable
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "QNNExecutionProvider" in providers:
                compile_ok = True
                runtime_ok = True
                diag.append("QNNExecutionProvider available via onnxruntime")
            else:
                diag.append(
                    f"onnxruntime installed but QNNExecutionProvider not in providers: {providers}"
                )
        except ImportError:
            diag.append("onnxruntime not installed")

        # Check for QNN SDK env
        qnn_sdk = os.environ.get("QNN_SDK_ROOT")
        if qnn_sdk and Path(qnn_sdk).is_dir():
            diag.append(f"QNN_SDK_ROOT set: {qnn_sdk}")
        else:
            diag.append("QNN_SDK_ROOT not set or not a directory")

        return BackendCapabilities(
            backend_id=self.id,
            compile_available=compile_ok,
            runtime_available=runtime_ok,
            toolchain_info={
                "qnn_sdk_root": qnn_sdk,
            },
            diagnostics=diag,
        )

    # -------------------------------------------------------------------
    # prepare (copy-based passthrough, always works)
    # -------------------------------------------------------------------

    def prepare(
        self,
        graphs: GraphBundle,
        out_dir: Path,
        *,
        target: TargetSpec,
        backend_config: dict[str, Any],
    ) -> BackendPreparedBundle:
        out_dir.mkdir(parents=True, exist_ok=True)

        graphs_out = out_dir / "graphs"
        graphs_out.mkdir(exist_ok=True)
        prepared_graphs: dict[str, Path] = {}
        for name, p in graphs.graphs.items():
            dst = graphs_out / p.name
            shutil.copy2(p, dst)
            prepared_graphs[name] = dst
            # Copy co-located ONNX external data file if present
            data_file = p.parent / f"{p.name}.data"
            if data_file.exists():
                shutil.copy2(data_file, graphs_out / data_file.name)

        artifacts_dir = out_dir / "backend_artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for p in graphs.extra_files:
            if p.exists() and p.is_file():
                shutil.copy2(p, artifacts_dir / p.name)

        backend_metadata = {
            "backend": self.id,
            "target": target.name,
            "params": dict(target.params),
            "compile_strategy": "passthrough",
        }

        return BackendPreparedBundle(
            graphs=prepared_graphs,
            artifacts_dir=artifacts_dir,
            backend_metadata=backend_metadata,
        )

    # -------------------------------------------------------------------
    # QNN op compatibility check
    # -------------------------------------------------------------------

    # Ops that are structurally incompatible (wrong quant format).
    _INCOMPATIBLE_OPS = frozenset({"MatMulNBits"})

    # Ops known to be unsupported or problematic on QNN HTP in monolithic
    # decoder graphs.  A graph dominated by these will not compile into a
    # compact context binary.
    _HTP_RISKY_OPS = frozenset({
        "LayerNormalization", "SkipLayerNormalization",
        "FastGelu", "BiasGelu",
        "GroupQueryAttention", "MultiHeadAttention",
        "RotaryEmbedding",
    })

    def _check_qnn_compatible_ops(self, graphs: GraphBundle) -> None:
        """Check ONNX graphs for ops that are incompatible with QNN HTP.

        MatMulNBits (INT4 weight-only from ORT GenAI builder) is NOT the
        same as QDQ quantization. QNN HTP needs QuantizeLinear/DequantizeLinear
        nodes, not MatMulNBits.
        """
        try:
            import onnx
        except ImportError:
            return  # can't check without onnx

        for name, graph_path in graphs.graphs.items():
            try:
                model = onnx.load(str(graph_path), load_external_data=False)
            except Exception:
                continue

            incompatible = set()
            for node in model.graph.node:
                if node.op_type in self._INCOMPATIBLE_OPS:
                    incompatible.add(node.op_type)

            if incompatible:
                raise NpuModelError(
                    stage="backend",
                    reason_code="INCOMPATIBLE_QUANTIZATION",
                    message=(
                        f"Graph '{name}' contains ops incompatible with QNN HTP: "
                        f"{sorted(incompatible)}"
                    ),
                    hint=(
                        "MatMulNBits is INT4 weight-only quantization (for CPU/DML), "
                        "NOT QDQ quantization (for QNN HTP).\n\n"
                        "QNN HTP requires QDQ models with QuantizeLinear/DequantizeLinear nodes.\n"
                        "These are produced by: qnn_preprocess_model → get_qnn_qdq_config → quantize\n\n"
                        "To fix:\n"
                        "  1. Re-export from HF at FP32: --precision fp32\n"
                        "  2. Quantize with QNN QDQ: --quant qnn-qdq\n"
                        "  3. Then compile context-cache\n\n"
                        "The ORT GenAI builder's --precision int4 produces MatMulNBits,\n"
                        "which is a different format not supported by QNN HTP."
                    ),
                )

    def _audit_htp_op_coverage(self, graphs: GraphBundle) -> dict[str, set[str]]:
        """Return per-graph sets of ops known to be risky on QNN HTP.

        Does NOT raise — callers decide the policy.
        """
        try:
            import onnx
        except ImportError:
            return {}

        results: dict[str, set[str]] = {}
        for name, graph_path in graphs.graphs.items():
            try:
                model = onnx.load(str(graph_path), load_external_data=False)
            except Exception:
                continue
            risky = {n.op_type for n in model.graph.node if n.op_type in self._HTP_RISKY_OPS}
            if risky:
                results[name] = risky
        return results

    # -------------------------------------------------------------------
    # HTP eligibility probe (no-fallback tiny run with synthetic feed)
    # -------------------------------------------------------------------

    @staticmethod
    def _build_synthetic_feed(graph_path: Path) -> dict[str, Any]:
        """Build a minimal all-zeros feed dict for every input in the graph.

        Used by the HTP eligibility probe so it can call ``sess.run()``
        instead of relying solely on session creation (which may not
        surface per-op failures on some ORT/QNN builds).
        """
        import numpy as np

        try:
            import onnx
            from onnx import TensorProto
        except ImportError:
            return {}

        _ONNX_DTYPE_MAP = {
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.FLOAT: np.float32,
            TensorProto.FLOAT16: np.float16,
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
        }

        try:
            model = onnx.load(str(graph_path), load_external_data=False)
        except Exception:
            return {}

        feed: dict[str, Any] = {}
        for inp in model.graph.input:
            if not inp.type.HasField("tensor_type"):
                continue
            dtype = _ONNX_DTYPE_MAP.get(inp.type.tensor_type.elem_type, np.float32)
            shape_proto = inp.type.tensor_type.shape
            if shape_proto is None:
                continue
            dims: list[int] = []
            for dim in shape_proto.dim:
                if dim.dim_value > 0:
                    dims.append(dim.dim_value)
                else:
                    dims.append(1)  # symbolic → 1 for probe
            feed[inp.name] = np.zeros(dims, dtype=dtype)
        return feed

    def _probe_htp_eligibility(
        self,
        graph_path: Path,
        backend_path: str,
        graph_name: str,
    ) -> None:
        """Run a no-fallback probe session **with a real synthetic feed** to
        verify the graph can execute entirely on QNN HTP before attempting
        context-cache generation.

        The probe:
          1. Creates a session with ``session.disable_cpu_ep_fallback=1`` and
             ep.context_enable left OFF.
          2. Builds an all-zeros feed for every graph input.
          3. Calls ``sess.run(None, feed)`` — which forces ORT to actually
             partition and execute every op on QNN.  Session creation alone
             is not sufficient because some ORT/QNN combinations defer op
             validation until run time.

        If either session creation or the run fails, a clear
        ``HTP_PROBE_FAILED`` error is raised directing the user to the
        Olive-backed LLM path.
        """
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        # Strict NPU-only — fails if any op falls back to CPU
        sess_opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        qnn_opts: dict[str, str] = {"backend_path": backend_path}

        try:
            sess = ort.InferenceSession(
                str(graph_path),
                sess_opts,
                providers=["QNNExecutionProvider"],
                provider_options=[qnn_opts],
            )
        except Exception as e:
            err_str = str(e)
            raise NpuModelError(
                stage="backend",
                reason_code="HTP_PROBE_FAILED",
                message=(
                    f"Graph '{graph_name}' cannot be loaded on QNN HTP "
                    f"(session creation failed).\n"
                    f"Probe error: {err_str[:500]}"
                ),
                hint=(
                    "This typically means the QDQ graph still contains ops that QNN HTP "
                    "cannot execute.\n\n"
                    "The generic QDQ quantization path is experimental for LLMs.\n"
                    "For production deployment of Phi / Phi-3 / Llama models on QNN HTP, "
                    "use the Olive-backed pipeline:\n"
                    "  npu-model convert --quant olive-qnn-llm ...\n\n"
                    "For details, see docs/plugin_dev.md"
                ),
                cause=e,
            ) from e

        # Build a synthetic feed and run inference to validate ops at
        # execution time, not just at graph-load time.
        feed = self._build_synthetic_feed(graph_path)
        if feed:
            try:
                sess.run(None, feed)
            except Exception as e:
                err_str = str(e)
                raise NpuModelError(
                    stage="backend",
                    reason_code="HTP_PROBE_FAILED",
                    message=(
                        f"Graph '{graph_name}' loaded on QNN HTP but inference failed "
                        f"with a synthetic feed.\n"
                        f"Probe error: {err_str[:500]}"
                    ),
                    hint=(
                        "Session creation succeeded but running the graph failed.  "
                        "This usually means some ops are partially supported (loadable "
                        "but not executable) on HTP.\n\n"
                        "The generic QDQ quantization path is experimental for LLMs.\n"
                        "For production deployment of Phi / Phi-3 / Llama models, "
                        "use:\n"
                        "  npu-model convert --quant olive-qnn-llm ...\n\n"
                        "For details, see docs/plugin_dev.md"
                    ),
                    cause=e,
                ) from e
        else:
            _log.warning(
                "Could not build synthetic feed for HTP probe of '%s' — "
                "probe relied on session creation only.",
                graph_name,
            )

        del sess

    def _validate_compiled_model_controls(
        self,
        *,
        ctx_graphs: dict[str, Path],
        backend_path: str,
        require_fail_on_suboptimal: bool = True,
    ) -> None:
        """Load compiled wrappers with model compilation disabled.

        This ensures deployable artifacts do not rely on implicit recompilation
        at runtime.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            return

        for _name, graph_path in ctx_graphs.items():
            qnn_provider_opts: dict[str, str] = {"backend_path": backend_path}

            def _make_opts(*, with_fail_on_suboptimal: bool) -> Any:
                sess_opts = ort.SessionOptions()
                sess_opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                sess_opts.add_session_config_entry("session.disable_model_compile", "1")
                if with_fail_on_suboptimal:
                    sess_opts.add_session_config_entry("session.fail_on_suboptimal_compiled_model", "1")
                return sess_opts

            try:
                sess = ort.InferenceSession(
                    str(graph_path),
                    _make_opts(with_fail_on_suboptimal=require_fail_on_suboptimal),
                    providers=["QNNExecutionProvider"],
                    provider_options=[qnn_provider_opts],
                )
                del sess
            except Exception as e:
                msg = str(e)
                if (
                    require_fail_on_suboptimal
                    and "fail_on_suboptimal_compiled_model" in msg.lower()
                ):
                    # Some ORT builds may not support this key; retry without it.
                    try:
                        sess = ort.InferenceSession(
                            str(graph_path),
                            _make_opts(with_fail_on_suboptimal=False),
                            providers=["QNNExecutionProvider"],
                            provider_options=[qnn_provider_opts],
                        )
                        del sess
                        continue
                    except Exception as e2:
                        raise NpuModelError(
                            stage="backend",
                            reason_code="COMPILED_MODEL_VALIDATION_FAILED",
                            message=(
                                f"Compiled wrapper '{graph_path.name}' failed strict load "
                                "(disable_model_compile=1)."
                            ),
                            hint=str(e2)[:800],
                            cause=e2,
                        ) from e2

                raise NpuModelError(
                    stage="backend",
                    reason_code="COMPILED_MODEL_VALIDATION_FAILED",
                    message=(
                        f"Compiled wrapper '{graph_path.name}' failed strict load "
                        "(disable_model_compile=1)."
                    ),
                    hint=msg[:800],
                    cause=e,
                ) from e

    # -------------------------------------------------------------------
    # compile (real compilation via ORT QNN EP context caching)
    # -------------------------------------------------------------------

    def compile(
        self,
        graphs: GraphBundle,
        out_dir: Path,
        *,
        target: TargetSpec,
        compile_config: dict[str, Any],
    ) -> BackendPreparedBundle:
        strategy = compile_config.get("strategy", "passthrough")

        if strategy == "passthrough":
            return self.prepare(graphs, out_dir, target=target, backend_config=compile_config)

        if strategy in ("ort-ep-context", "context-cache"):
            return self._compile_context_cache(graphs, out_dir, target=target, config=compile_config)

        raise NpuModelError(
            stage="backend",
            reason_code="UNKNOWN_COMPILE_STRATEGY",
            message=f"QNN backend: unknown compile strategy '{strategy}'",
            hint="Supported strategies: passthrough, context-cache",
        )

    def _compile_context_cache(
        self,
        graphs: GraphBundle,
        out_dir: Path,
        *,
        target: TargetSpec,
        config: dict[str, Any],
    ) -> BackendPreparedBundle:
        """Generate QNN context-cache artifacts via ORT QNN EP.

        Uses the documented session config entries:
          - session.disable_cpu_ep_fallback = 1  (NPU-only, no silent CPU fallback)
          - ep.context_enable = 1
          - ep.context_embed_mode = 0  (separate .bin file)
          - ep.context_file_path = <output path>

        Produces:
          - <model>_ctx.onnx   (lightweight wrapper graph)
          - <model>_qnn.bin    (compiled QNN context binary)

        The _ctx.onnx + _qnn.bin pair is what the HTP backend uses at runtime.
        The original model.onnx (+ .data) is NOT needed after this step.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise NpuModelError(
                stage="backend",
                reason_code="MISSING_ORT",
                message="onnxruntime is required for context-cache compilation",
                hint="pip install onnxruntime-qnn",
            )

        providers = ort.get_available_providers()
        if "QNNExecutionProvider" not in providers:
            raise NpuModelError(
                stage="backend",
                reason_code="NO_QNN_EP",
                message="QNNExecutionProvider not available in this onnxruntime build",
                hint="Install onnxruntime-qnn or ensure QNN libraries are on PATH.",
            )

        from npu_model.core.npu_invariant import (
            apply_npu_only_session_options,
            apply_context_cache_session_options,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        compile_dir = out_dir / "context_cache_compile"
        compile_dir.mkdir(exist_ok=True)

        # Pre-flight: check model size vs available memory
        total_bytes = 0
        for name, src_path in graphs.graphs.items():
            if src_path.exists():
                total_bytes += src_path.stat().st_size
            data = src_path.parent / f"{src_path.name}.data"
            if data.exists():
                total_bytes += data.stat().st_size
        model_gb = total_bytes / (1024**3)
        required_ram_gb = model_gb * 2.0

        # Detect system RAM
        system_ram_gb = None
        try:
            import ctypes
            kernel32 = getattr(ctypes, "windll").kernel32

            class _MEMSTATEX(ctypes.Structure):
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

            ms = _MEMSTATEX()
            ms.dwLength = ctypes.sizeof(_MEMSTATEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                system_ram_gb = ms.ullTotalPhys / (1024**3)
        except Exception:
            try:
                import os as _os2
                sysconf = getattr(_os2, "sysconf")
                system_ram_gb = (sysconf("SC_PAGE_SIZE") * sysconf("SC_PHYS_PAGES")) / (1024**3)
            except Exception:
                pass

        if system_ram_gb is not None and required_ram_gb > system_ram_gb:
            raise NpuModelError(
                stage="backend",
                reason_code="MODEL_TOO_LARGE_FOR_CONTEXT_CACHE",
                message=(
                    f"QDQ model is {model_gb:.1f} GB — context-cache compilation needs "
                    f"~{required_ram_gb:.0f} GB RAM but this system has {system_ram_gb:.0f} GB."
                ),
                hint=(
                    "Options:\n"
                    "  1. Compile context-cache on a machine with more RAM\n"
                    "  2. Use a smaller model\n"
                    "  3. Use a prebuilt model: --mode prebuilt-ort-genai"
                ),
            )
        elif model_gb > 4.0:
            import logging
            logging.getLogger("npu_model").warning(
                "QDQ model is %.1f GB. Context-cache compilation loads the entire model "
                "into RAM (~%.0f GB needed). If this crashes, compile on a bigger machine.",
                model_gb, required_ram_gb,
            )

        # Pre-flight: check for incompatible quantization formats
        self._check_qnn_compatible_ops(graphs)

        # Pre-flight: audit op coverage and warn about risky ops
        risky_ops = self._audit_htp_op_coverage(graphs)
        if risky_ops:
            for gname, ops in risky_ops.items():
                _log.warning(
                    "Graph '%s' contains ops that are often unsupported on QNN HTP: %s. "
                    "Context-cache compilation may fail or produce oversized artifacts.",
                    gname, sorted(ops),
                )

        backend_path = target.params.get("backend_path", "QnnHtp.dll")
        embed_mode = config.get("ep_context_embed_mode", "0")

        # Copy graphs + co-located .data files into compile dir
        for name, src_path in graphs.graphs.items():
            dst = compile_dir / src_path.name
            if not dst.exists():
                shutil.copy2(src_path, dst)
            data = src_path.parent / f"{src_path.name}.data"
            if data.exists():
                dst_data = compile_dir / data.name
                if not dst_data.exists():
                    shutil.copy2(data, dst_data)

        # Build session options using documented session config entries
        # (NOT provider_options — those are for QNN EP-specific knobs only)
        context_errors: list[str] = []
        for name, src_path in graphs.graphs.items():
            graph_in_compile = compile_dir / src_path.name

            # ---- HTP eligibility probe ----
            # Run a no-context session with disable_cpu_ep_fallback to verify
            # the graph can execute entirely on QNN HTP before dumping context.
            self._probe_htp_eligibility(graph_in_compile, backend_path, name)

            # ep.context_file_path: ORT uses this as the output path for the
            # generated context model. Pass the compile directory — ORT will
            # create <model>_ctx.onnx and <model>_ctx_qnn.bin alongside it.
            # Some ORT versions expect a file path, others a directory.
            ctx_output_path = str(compile_dir / f"{graph_in_compile.stem}_ctx.onnx")

            sess_options = ort.SessionOptions()
            apply_npu_only_session_options(sess_options)
            apply_context_cache_session_options(
                sess_options,
                context_file_path=ctx_output_path,
                embed_mode=embed_mode,
            )

            # provider_options contains ONLY QNN EP knobs (backend_path, perf settings)
            qnn_provider_opts: dict[str, str] = {
                "backend_path": backend_path,
            }
            # Pass through any extra QNN EP options from compile config
            for k, v in config.items():
                if k.startswith("qnn_"):
                    qnn_provider_opts[k[4:]] = str(v)

            try:
                sess = ort.InferenceSession(
                    str(graph_in_compile),
                    sess_options=sess_options,
                    providers=["QNNExecutionProvider"],
                    provider_options=[qnn_provider_opts],
                )
                # Explicitly close the session to ensure context files are flushed
                del sess
                import gc
                gc.collect()
            except Exception as e:
                context_errors.append(f"{name} ({src_path.name}): {e}")

        if context_errors:
            raise NpuModelError(
                stage="backend",
                reason_code="CONTEXT_CACHE_FAILED",
                message="QNN context-cache generation failed:\n  "
                + "\n  ".join(context_errors),
                hint=(
                    "Common causes:\n"
                    "  - Model not properly QDQ-quantized (use --quant qnn-qdq with calibration data)\n"
                    "  - Dynamic shapes present (should be fixed automatically)\n"
                    "  - Unsupported ops for QNN HTP\n"
                    "  - QNN SDK / runtime version mismatch"
                ),
            )

        # Collect generated artifacts
        artifacts_dir = out_dir / "backend_artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        ctx_graphs: dict[str, Path] = {}
        for p in sorted(compile_dir.glob("*")):
            if not p.is_file():
                continue
            if "_ctx" in p.stem and p.suffix == ".onnx":
                if p.stat().st_size == 0:
                    continue  # skip empty ctx files (generation failed silently)
                dst = artifacts_dir / p.name
                shutil.copy2(p, dst)
                key = p.stem.replace("_ctx", "")
                ctx_graphs[key] = dst
            elif p.suffix == ".bin" and p.stat().st_size > 0:
                shutil.copy2(p, artifacts_dir / p.name)

        # Copy extra files (genai_config, etc.) from adapter
        for p in graphs.extra_files:
            if p.exists() and p.is_file():
                dst = artifacts_dir / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)

        # HARD REQUIREMENT: context-cache must produce valid ctx artifacts
        if not ctx_graphs:
            # Diagnose what went wrong
            all_ctx_files = list(compile_dir.glob("*_ctx*"))
            all_bins = list(compile_dir.glob("*.bin"))
            diag = []
            for f in all_ctx_files:
                diag.append(f"  {f.name}: {f.stat().st_size} bytes")
            for f in all_bins:
                diag.append(f"  {f.name}: {f.stat().st_size} bytes")

            raise NpuModelError(
                stage="backend",
                reason_code="NO_CTX_ARTIFACTS",
                message=(
                    "Context-cache compilation completed but no valid *_ctx.onnx "
                    "artifacts were generated (files may be 0 bytes)."
                ),
                hint=(
                    "Generated files in compile dir:\n"
                    + ("\n".join(diag) if diag else "  (none)")
                    + "\n\n"
                    "A 0-byte _ctx.onnx typically means ORT created the file but the "
                    "context serialization failed (e.g. protobuf >2GB limit).\n\n"
                    "Possible fixes:\n"
                    "  - Try ep_context_embed_mode=1 (embeds context in the ONNX)\n"
                    "  - Ensure the QDQ model's external data (.onnx.data) is co-located\n"
                    "  - Check ORT/QNN version compatibility"
                ),
            )

        qnn_bins = list(artifacts_dir.glob("*.bin"))
        if not qnn_bins:
            raise NpuModelError(
                stage="backend",
                reason_code="NO_QNN_BINS",
                message="Context-cache compilation produced _ctx.onnx but no .bin artifacts.",
                hint="Expected QNN compiled binary alongside context ONNX wrapper.",
            )

        # .bin size floor: a real compiled QNN binary is at least several KB.
        # An implausibly tiny .bin (< 1 KB) means the compiler produced a stub
        # rather than a real context binary.
        _MIN_QNN_BIN_BYTES = 1024  # 1 KB
        for bin_path in qnn_bins:
            bin_size = bin_path.stat().st_size
            if bin_size < _MIN_QNN_BIN_BYTES:
                raise NpuModelError(
                    stage="backend",
                    reason_code="QNN_BIN_TOO_SMALL",
                    message=(
                        f"Compiled binary '{bin_path.name}' is only {bin_size} bytes — "
                        f"expected at least {_MIN_QNN_BIN_BYTES} bytes for a real context binary."
                    ),
                    hint=(
                        "An implausibly small .bin usually means QNN compilation produced "
                        "a stub or empty context.  The compiled graph may be too simple or "
                        "the QNN SDK could not compile the graph into HTP instructions.\n\n"
                        "Check QNN SDK version compatibility and ensure the model is properly "
                        "QDQ-quantized."
                    ),
                )

        # Output size guard: a valid context wrapper should be compact (< 10 MB).
        # If the _ctx.onnx is enormous, context serialization likely embedded the
        # full weights instead of producing a separate .bin — this is not a usable
        # deployment artifact.
        _MAX_CTX_WRAPPER_BYTES = 10 * 1024 * 1024  # 10 MB
        for gname, ctx_path in ctx_graphs.items():
            ctx_size = ctx_path.stat().st_size
            if ctx_size > _MAX_CTX_WRAPPER_BYTES:
                ctx_mb = ctx_size / (1024 * 1024)
                raise NpuModelError(
                    stage="backend",
                    reason_code="CTX_WRAPPER_OVERSIZED",
                    message=(
                        f"Context wrapper '{ctx_path.name}' is {ctx_mb:.0f} MB — "
                        f"expected a compact wrapper (< 10 MB)."
                    ),
                    hint=(
                        "An oversized _ctx.onnx means context serialization embedded the full "
                        "model weights instead of producing a separate .bin binary.\n\n"
                        "This usually happens when QNN HTP could not compile all ops, causing "
                        "ORT to fall back to embedding raw weights.\n\n"
                        "For LLMs (Phi, Llama), use the Olive-backed pipeline:\n"
                        "  npu-model convert --quant olive-qnn-llm ...\n"
                        "Or try ep_context_embed_mode=0 (separate .bin)."
                    ),
                )

        self._validate_compiled_model_controls(
            ctx_graphs=ctx_graphs,
            backend_path=backend_path,
            require_fail_on_suboptimal=bool(config.get("fail_on_suboptimal_compiled_model", True)),
        )

        backend_metadata = {
            "backend": self.id,
            "target": target.name,
            "params": dict(target.params),
            "compile_strategy": "context-cache",
            "ep_context_embed_mode": embed_mode,
            "ctx_graphs": [p.name for p in ctx_graphs.values()],
            "qnn_bins": [p.name for p in qnn_bins],
            "compiled_model_validation": "strict",
        }

        return BackendPreparedBundle(
            graphs=ctx_graphs,
            artifacts_dir=artifacts_dir,
            backend_metadata=backend_metadata,
        )
