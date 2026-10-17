from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from npu_model.core.errors import NpuModelError
from npu_model.core.manifest import collect_files, write_manifest
from npu_model.runtime_formats.ort_genai_folder import (
    collect_ollama_files,
    validate_ollama_ortgenai_dir,
    write_modelfile,
)

import npu_model


@dataclass(frozen=True)
class PackResult:
    pack_dir: Path
    file_count: int


def pack_for_ollama(
    bundle_dir: Path,
    model_name: str,
    out_dir: Path,
    *,
    num_ctx: int = 512,
    num_predict: int = 128,
) -> PackResult:
    """Assemble an Ollama input directory; does not install, execute, or upload.

    Uses atomic staging: builds into a temp directory, validates, then
    replaces the destination to avoid leaving an empty output on failure.
    """
    bundle_dir = bundle_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if not bundle_dir.exists():
        raise NpuModelError(
            stage="pack",
            reason_code="BUNDLE_NOT_FOUND",
            message=f"Bundle not found: {bundle_dir}",
        )

    # --- collect allowlisted files ---
    files = collect_ollama_files(bundle_dir)
    if not files:
        raise NpuModelError(
            stage="pack",
            reason_code="NO_FILES_COLLECTED",
            message="No allowlisted files found in bundle directory.",
            hint="Ensure the input directory contains *.onnx, genai_config.json, and tokenizer files.",
        )

    # --- NPU-only guard: compiled-only packaging contract ---
    collected_names = {f.name for f in files}
    has_data_files = any(n.endswith(".onnx.data") for n in collected_names)
    has_ctx_graphs = any("_ctx" in n and n.endswith(".onnx") for n in collected_names)
    has_qnn_bins = any(n.endswith(".bin") for n in collected_names)
    non_ctx_onnx = sorted(n for n in collected_names if n.endswith(".onnx") and "_ctx" not in n)

    if has_data_files:
        raise NpuModelError(
            stage="pack",
            reason_code="ONNX_DATA_NOT_ALLOWED",
            message=(
                "Bundle contains .onnx.data (raw float weights). "
                "Deployable NPU bundles must not contain external ONNX data files."
            ),
            hint=(
                "Run compile-context first and pack the resulting compiled bundle "
                "(containing *_ctx.onnx + .bin)."
            ),
        )
    if non_ctx_onnx:
        raise NpuModelError(
            stage="pack",
            reason_code="NON_CTX_ONNX_NOT_ALLOWED",
            message=(
                "Bundle contains non-context ONNX files. "
                f"Only compiled wrappers are allowed: {non_ctx_onnx}"
            ),
            hint="Pack only the output of compile-context (compiled_bundle).",
        )
    if not has_ctx_graphs:
        raise NpuModelError(
            stage="pack",
            reason_code="CTX_ONNX_REQUIRED",
            message="No *_ctx.onnx context wrapper files found in bundle.",
            hint="Run npu-model compile-context before pack-ollama.",
        )
    if not has_qnn_bins:
        raise NpuModelError(
            stage="pack",
            reason_code="QNN_BIN_REQUIRED",
            message="No .bin QNN context binaries found in bundle.",
            hint="Run npu-model compile-context before pack-ollama.",
        )

    # --- atomic staging ---
    staging = out_dir.parent / f"{out_dir.name}.staging.{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        # Copy collected files flat into staging root
        for src in files:
            dst = staging / src.name
            # Handle name collisions from nested dirs by preferring shallower files
            if not dst.exists():
                shutil.copy2(src, dst)

        # Write Modelfile
        write_modelfile(staging, num_ctx=num_ctx, num_predict=num_predict)

        # Write publish manifest
        manifest_payload = {
            "tool": {"name": "npu-model", "version": npu_model.__version__},
            "pack_type": "ollama-ortgenai",
            "model_name": model_name,
            "params": {"num_ctx": num_ctx, "num_predict": num_predict},
            "files": collect_files(staging),
        }
        write_manifest(staging / "_publish_manifest.json", manifest_payload)

        # Validate staging before promoting
        validation = validate_ollama_ortgenai_dir(staging)
        if validation.errors:
            msg = "Staged output failed validation:\n  " + "\n  ".join(validation.errors)
            raise NpuModelError(
                stage="pack",
                reason_code="STAGING_VALIDATION_FAILED",
                message=msg,
            )

        # Count files (excluding Modelfile and manifest for the "copied" count)
        copied_count = len([p for p in staging.iterdir() if p.is_file()])

        # Atomic swap: remove existing destination, rename staging into place
        if out_dir.exists():
            shutil.rmtree(out_dir)
        staging.rename(out_dir)

    except NpuModelError:
        # Clean up staging on our errors
        if staging.exists():
            shutil.rmtree(staging)
        raise
    except Exception as e:
        if staging.exists():
            shutil.rmtree(staging)
        raise NpuModelError(
            stage="pack",
            reason_code="PACK_FAILED",
            message=f"Failed to build Ollama pack: {e}",
            cause=e,
        ) from e

    return PackResult(pack_dir=out_dir, file_count=copied_count)
