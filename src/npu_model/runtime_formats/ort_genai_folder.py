from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.types import BackendPreparedBundle
from npu_model.runtime_formats.base import RuntimeFormat


# ---------------------------------------------------------------------------
# Ollama ORT-GenAI packaging contract
# ---------------------------------------------------------------------------

# Extensions that Ollama will pick up when the primary detection is ONNX.
_OLLAMA_ALLOW_EXTENSIONS: frozenset[str] = frozenset({
    ".onnx",
    ".bin",
    ".json",
    ".jinja",
})

# Exact filenames always collected regardless of extension.
_OLLAMA_ALLOW_NAMES: frozenset[str] = frozenset({
    "tokenizer.model",
})

# Extensions whose presence in the publish dir would cause Ollama to
# mis-detect the model format. Must never appear in output.
_OLLAMA_DENY_EXTENSIONS: frozenset[str] = frozenset({
    ".gguf",
    ".safetensors",
    ".lib",
    ".pdb",
})


def collect_ollama_files(src_dir: Path) -> list[Path]:
    """Return allowlisted files from *src_dir* (recursive) for Ollama publish.

    Walks the full tree so callers can point at a nested ORT GenAI bundle
    (which may have ``onnx/``, ``tokenizer/``, ``backend_artifacts/`` sub-dirs).
    """
    collected: list[Path] = []
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in _OLLAMA_DENY_EXTENSIONS:
            continue
        if p.name in _OLLAMA_ALLOW_NAMES or p.suffix.lower() in _OLLAMA_ALLOW_EXTENSIONS:
            collected.append(p)
    return collected


@dataclass
class OllamaOrtGenaiValidation:
    errors: list[str]
    warnings: list[str]


def validate_ollama_ortgenai_dir(path: Path) -> OllamaOrtGenaiValidation:
    """Validate a directory for Ollama-compatible ORT GenAI packaging.

    Returns structured errors (hard failures) and warnings (soft issues).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not path.is_dir():
        errors.append(f"Path is not a directory: {path}")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)

    all_files = {p.name: p for p in path.rglob("*") if p.is_file()}
    all_names = set(all_files.keys())
    all_suffixes = {p.suffix.lower() for p in all_files.values()}

    # --- hard requirements ---
    onnx_files = [n for n in all_names if n.lower().endswith(".onnx")]
    if not onnx_files:
        errors.append("No *.onnx files found (at least 1 required)")
    elif any("_ctx" not in n.lower() for n in onnx_files):
        errors.append(
            "Pack input must contain compiled context wrappers only (*_ctx.onnx). "
            f"Found non-ctx ONNX files: {[n for n in onnx_files if '_ctx' not in n.lower()]}"
        )

    if "genai_config.json" not in all_names:
        errors.append("genai_config.json not found (required)")

    has_tokenizer = "tokenizer.model" in all_names or "tokenizer.json" in all_names
    if not has_tokenizer:
        errors.append("Neither tokenizer.model nor tokenizer.json found (at least 1 required)")

    # --- mis-detection hazards ---
    for ext in _OLLAMA_DENY_EXTENSIONS:
        if ext in all_suffixes:
            bad = [n for n in all_names if n.lower().endswith(ext)]
            errors.append(f"Files with extension {ext} present (would cause mis-detection): {bad}")

    # --- soft warnings ---
    bin_files = [n for n in all_names if n.lower().endswith(".bin")]
    if not bin_files:
        errors.append("No *.bin files found (compiled QNN context binaries are required)")

    # Warn about ONNX external data files — Ollama may not materialize these
    data_files = [n for n in all_names if n.lower().endswith(".onnx.data")]
    if data_files:
        errors.append(
            f"ONNX external data files found: {data_files}. "
            "Deployable NPU bundles must not include .onnx.data."
        )

    if "chat_template.jinja" not in all_names:
        warnings.append("chat_template.jinja not found (chat mode may not work)")

    # --- tokenizer class compatibility ---
    if "tokenizer_config.json" in all_files:
        try:
            tok_cfg = json.loads(all_files["tokenizer_config.json"].read_text(encoding="utf-8"))
            tok_class = tok_cfg.get("tokenizer_class", "")
            _BAD_TOK_CLASSES = {"TokenizersBackend"}
            if tok_class in _BAD_TOK_CLASSES:
                errors.append(
                    f"tokenizer_class='{tok_class}' is not supported by ORT GenAI on "
                    f"Windows ARM64. Re-run convert (which normalizes the tokenizer) "
                    f"or manually set tokenizer_class to 'LlamaTokenizer' in tokenizer_config.json."
                )
            elif tok_class.endswith("Fast"):
                warnings.append(
                    f"tokenizer_class='{tok_class}' may not be supported on all ORT GenAI "
                    f"builds. Consider using a non-Fast class (e.g. 'LlamaTokenizer')."
                )
        except Exception:
            warnings.append("Could not parse tokenizer_config.json for tokenizer class check.")

    for optional in ("config.json", "generation_config.json", "special_tokens_map.json"):
        if optional not in all_names:
            warnings.append(f"Optional file missing: {optional}")

    return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)


def write_modelfile(out_dir: Path, *, num_ctx: int = 512, num_predict: int = 128) -> Path:
    """Write a minimal Modelfile suitable for ``ollama create -f Modelfile``."""
    mf = out_dir / "Modelfile"
    mf.write_text(
        f"FROM .\n"
        f"\n"
        f"# Safe defaults for ORT GenAI KV cache\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        f"PARAMETER num_predict {num_predict}\n",
        encoding="utf-8",
    )
    return mf


# ---------------------------------------------------------------------------
# genai_config.json sanity validation
# ---------------------------------------------------------------------------

def validate_genai_config(path: Path) -> OllamaOrtGenaiValidation:
    """Validate genai_config.json content for sanity.

    Checks critical fields that affect KV cache allocation and generation behavior.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Find genai_config.json
    if path.is_file() and path.name == "genai_config.json":
        config_path = path
    elif path.is_dir():
        config_path = path / "genai_config.json"
    else:
        errors.append(f"Cannot find genai_config.json at: {path}")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)

    if not config_path.exists():
        errors.append(f"genai_config.json not found: {config_path}")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        errors.append(f"Failed to parse genai_config.json: {e}")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)

    if not isinstance(cfg, dict):
        errors.append("genai_config.json must contain a JSON object")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)

    # Check for model section
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        errors.append("model must be a JSON object")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)
    if not model_cfg:
        warnings.append("genai_config.json has no 'model' section")

    # Check decoder section
    decoder = model_cfg.get("decoder", {})
    if not isinstance(decoder, dict):
        errors.append("model.decoder must be a JSON object")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)

    # Check search section (generation params)
    search = cfg.get("search", {})
    if not isinstance(search, dict):
        errors.append("search must be a JSON object")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)

    # Validate context_length / max_length relationship
    context_length = decoder.get("context_length")
    max_length = search.get("max_length")
    num_layers = decoder.get("num_hidden_layers")
    for name, value in (
        ("decoder.context_length", context_length),
        ("search.max_length", max_length),
        ("decoder.num_hidden_layers", num_layers),
    ):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            errors.append(f"{name} must be a positive integer, got {value!r}")
    if errors:
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)

    if context_length is not None and max_length is not None:
        if max_length > context_length:
            warnings.append(
                f"search.max_length ({max_length}) > decoder.context_length ({context_length}). "
                "This may cause KV cache overflow at runtime."
            )

    # Check for session options (EP config)
    session = decoder.get("session_options", {})
    if not isinstance(session, dict):
        errors.append("decoder.session_options must be a JSON object")
        return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)
    if not session:
        warnings.append(
            "No session_options in decoder config. "
            "Runtime may use default execution provider."
        )

    return OllamaOrtGenaiValidation(errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# ORT GenAI folder runtime format (used by convert pipeline)
# ---------------------------------------------------------------------------

@dataclass
class OrtGenaiFolderFormat(RuntimeFormat):
    id: str = "ort-genai-folder"

    def assemble(
        self,
        prepared: BackendPreparedBundle,
        tokenizer_dir: Path,
        out_dir: Path,
        *,
        format_config: dict[str, Any],
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir = out_dir / "ort_genai"
        bundle_dir.mkdir(exist_ok=True)

        # Graphs
        onnx_dir = bundle_dir / "onnx"
        onnx_dir.mkdir(exist_ok=True)
        for _, p in prepared.graphs.items():
            shutil.copy2(p, onnx_dir / p.name)
            # Copy co-located ONNX external data file if present
            data_file = p.parent / f"{p.name}.data"
            if data_file.exists():
                shutil.copy2(data_file, onnx_dir / data_file.name)

        # Tokenizer
        tok_out = bundle_dir / "tokenizer"
        tok_out.mkdir(exist_ok=True)
        if tokenizer_dir.exists():
            for p in sorted(tokenizer_dir.glob("*")):
                if p.is_file():
                    shutil.copy2(p, tok_out / p.name)

        # Backend artifacts
        ba_out = bundle_dir / "backend_artifacts"
        ba_out.mkdir(exist_ok=True)
        if prepared.artifacts_dir.exists():
            for p in sorted(prepared.artifacts_dir.glob("*")):
                if p.is_file():
                    shutil.copy2(p, ba_out / p.name)

        # If adapter provided genai_config.json, keep it at bundle root
        provided = list(ba_out.glob("genai_config.json"))
        if provided:
            shutil.copy2(provided[0], bundle_dir / "genai_config.json")

        # Backend metadata file for debugging
        (bundle_dir / "backend_metadata.json").write_text(
            json.dumps(prepared.backend_metadata, indent=2),
            encoding="utf-8",
        )

        self.validate_layout(bundle_dir)
        return bundle_dir

    def validate_layout(self, bundle_dir: Path) -> None:
        if not (bundle_dir / "onnx").is_dir():
            raise NpuModelError(
                stage="runtime",
                reason_code="MISSING_ONNX_DIR",
                message=f"Missing: {bundle_dir / 'onnx'}",
            )
        any_onnx = any(p.is_file() for p in (bundle_dir / "onnx").glob("*.onnx"))
        if not any_onnx:
            raise NpuModelError(
                stage="runtime",
                reason_code="NO_ONNX_FILES",
                message="No .onnx files found in bundle/onnx",
                hint="In MVP, the adapter copies existing .onnx files from the input model directory.",
            )
