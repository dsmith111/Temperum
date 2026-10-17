from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle, ModelInfo
from npu_model.exporters.base import Exporter


# Model families supported by ORT GenAI model builder
_SUPPORTED_MODEL_TYPES: set[str] = {
    "phi3", "phi", "llama", "mistral", "gemma", "qwen2",
}


def resolve_export_precision(requested: str | None, quantizer_id: str = "passthrough") -> str:
    """Choose a default, but never replace an explicitly requested precision."""
    precision = requested if requested is not None else (
        "int4" if quantizer_id == "passthrough" else "fp16"
    )
    if precision not in {"int4", "fp16", "fp32"}:
        raise NpuModelError(
            stage="export", reason_code="UNSUPPORTED_EXPORT_PRECISION",
            message=f"Unsupported export precision: {precision!r}",
            hint="The ORT GenAI builder path supports int4, fp16, fp32; bf16 is not supported here.",
        )
    if precision == "int4" and quantizer_id != "passthrough":
        raise NpuModelError(
            stage="export", reason_code="INCOMPATIBLE_EXPORT_PRECISION",
            message=f"Explicit int4 export is incompatible with quantizer '{quantizer_id}'.",
            hint="Use --precision fp16 or fp32 before QDQ quantization, or omit --precision for fp16.",
        )
    return precision


@dataclass
class OrtGenaiBuilderExporter(Exporter):
    """Export HF models to ORT GenAI format using onnxruntime-genai model builder.

    Wraps the ``onnxruntime_genai.models.builder`` module which produces:
      - ONNX graph(s)
      - genai_config.json
      - tokenizer files

    Heavy dependencies (onnxruntime-genai, torch, transformers) are imported
    lazily inside methods so the tool starts fast even if they aren't installed.
    """

    id: str = "ort-genai-builder"

    def can_export(self, model: ModelInfo) -> bool:
        mt = (model.model_type or "").lower()
        return mt in _SUPPORTED_MODEL_TYPES

    def check_dependencies(self) -> list[str]:
        missing: list[str] = []
        for pkg in ("onnxruntime_genai", "torch", "transformers"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        return missing

    def export(
        self,
        model_dir: Path,
        out_dir: Path,
        model: ModelInfo,
        *,
        export_config: dict[str, Any],
    ) -> GraphBundle:
        precision = resolve_export_precision(export_config.get("precision"))
        missing = self.check_dependencies()
        if missing:
            raise NpuModelError(
                stage="export",
                reason_code="MISSING_EXPORT_DEPS",
                message=f"Missing dependencies for ORT GenAI builder: {missing}",
                hint="pip install onnxruntime-genai torch transformers",
            )

        out_dir.mkdir(parents=True, exist_ok=True)

        execution_provider = export_config.get("execution_provider", "cpu")
        extra_args = export_config.get("extra_args", [])

        # If the model's config.json has "auto_map", it will force transformers
        # to load custom modeling code from the HF repo.  That custom code is
        # often stale and incompatible with newer transformers (e.g. Phi-3's
        # rope_scaling["type"] KeyError with transformers >=5.0).
        #
        # Since transformers already has built-in support for Phi-3, Llama,
        # Mistral, etc., we temporarily remove "auto_map" from config.json
        # so the builder uses the built-in implementation.
        config_patched = False
        config_backup: bytes | None = None
        config_path = model_dir / "config.json"
        if not export_config.get("trust_remote_code", False) and config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                if "auto_map" in cfg:
                    config_backup = config_path.read_bytes()
                    patched = {k: v for k, v in cfg.items() if k != "auto_map"}
                    config_path.write_text(
                        json.dumps(patched, indent=2), encoding="utf-8",
                    )
                    config_patched = True
            except Exception:
                pass  # if patching fails, proceed with original

        cmd = [
            sys.executable, "-m", "onnxruntime_genai.models.builder",
            "-i", str(model_dir),
            "-o", str(out_dir),
            "-p", precision,
            "-e", execution_provider,
        ]
        cmd.extend(str(a) for a in extra_args)

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=export_config.get("timeout", 3600),
            )
        except subprocess.CalledProcessError as e:
            stderr_tail = (e.stderr or "")[-2000:]
            raise NpuModelError(
                stage="export",
                reason_code="BUILDER_FAILED",
                message=f"ORT GenAI model builder failed (exit {e.returncode})",
                hint=f"stderr:\n{stderr_tail}" if stderr_tail else "Check model compatibility.",
                cause=e,
            ) from e
        except subprocess.TimeoutExpired as e:
            raise NpuModelError(
                stage="export",
                reason_code="BUILDER_TIMEOUT",
                message="ORT GenAI model builder timed out",
                hint="Increase timeout or use a smaller model.",
                cause=e,
            ) from e
        finally:
            # Always restore original config.json if we patched it
            if config_patched and config_backup is not None:
                try:
                    config_path.write_bytes(config_backup)
                except Exception:
                    pass

        # Collect exported artifacts — separate graphs, tokenizer, and extras
        # into distinct directories so downstream pipeline stages don't mix them.
        graphs_dir = out_dir / "_graphs"
        graphs_dir.mkdir(exist_ok=True)
        tok_dir = out_dir / "_tokenizer"
        tok_dir.mkdir(exist_ok=True)

        graphs: dict[str, Path] = {}
        for p in sorted(out_dir.rglob("*.onnx")):
            if p.parent.name.startswith("_"):
                continue  # skip our own staging dirs
            dst = graphs_dir / p.name
            shutil.copy2(p, dst)
            graphs[p.stem] = dst
            # Co-located .data file
            data = p.parent / f"{p.name}.data"
            if data.exists():
                shutil.copy2(data, graphs_dir / data.name)

        if not graphs:
            raise NpuModelError(
                stage="export",
                reason_code="NO_EXPORT_OUTPUT",
                message=f"Builder completed but no .onnx files found in {out_dir}",
                hint="Check builder output for warnings.",
            )

        # Tokenizer assets → _tokenizer/
        _TOK_NAMES = {
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
        }
        for name in _TOK_NAMES:
            src = out_dir / name
            if src.exists():
                shutil.copy2(src, tok_dir / name)

        # Extra files (genai_config, generation_config, bins, etc.)
        extra_files: list[Path] = []
        for p in sorted(out_dir.glob("*.json")):
            if p.name not in {"config.json"} and p.name not in _TOK_NAMES:
                extra_files.append(p)
        for p in sorted(out_dir.glob("*.bin")):
            extra_files.append(p)
        for p in sorted(out_dir.glob("*.jinja")):
            if p.name not in _TOK_NAMES:
                extra_files.append(p)

        metadata: dict[str, Any] = {
            "exporter": self.id,
            "precision": precision,
            "execution_provider": execution_provider,
            "model_type": model.model_type,
        }

        return GraphBundle(
            graphs=graphs,
            tokenizer_dir=tok_dir,
            extra_files=extra_files,
            metadata=metadata,
        )
