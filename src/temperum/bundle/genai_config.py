"""genai_config.json sanity checks shared by bundle inspection and assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigCheck:
    errors: list[str]
    warnings: list[str]


def validate_genai_config(path: Path) -> ConfigCheck:
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
        return ConfigCheck(errors=errors, warnings=warnings)

    if not config_path.exists():
        errors.append(f"genai_config.json not found: {config_path}")
        return ConfigCheck(errors=errors, warnings=warnings)

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        errors.append(f"Failed to parse genai_config.json: {e}")
        return ConfigCheck(errors=errors, warnings=warnings)

    if not isinstance(cfg, dict):
        errors.append("genai_config.json must contain a JSON object")
        return ConfigCheck(errors=errors, warnings=warnings)

    # Check for model section
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        errors.append("model must be a JSON object")
        return ConfigCheck(errors=errors, warnings=warnings)
    if not model_cfg:
        warnings.append("genai_config.json has no 'model' section")

    # Check decoder section
    decoder = model_cfg.get("decoder", {})
    if not isinstance(decoder, dict):
        errors.append("model.decoder must be a JSON object")
        return ConfigCheck(errors=errors, warnings=warnings)

    # Check search section (generation params)
    search = cfg.get("search", {})
    if not isinstance(search, dict):
        errors.append("search must be a JSON object")
        return ConfigCheck(errors=errors, warnings=warnings)

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
        return ConfigCheck(errors=errors, warnings=warnings)

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
        return ConfigCheck(errors=errors, warnings=warnings)
    if not session:
        warnings.append(
            "No session_options in decoder config. "
            "Runtime may use default execution provider."
        )

    return ConfigCheck(errors=errors, warnings=warnings)
