from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle

_log = logging.getLogger("npu_model.olive")

SUPPORTED_OLIVE_LLM_FAMILIES: frozenset[str] = frozenset({"phi", "phi3", "llama"})

# Legacy template map — kept for reference / advanced override scenarios.
_TEMPLATE_BY_FAMILY: dict[str, str] = {
    "phi": "phi2_qnn.json.j2",
    "phi3": "phi_qnn.json.j2",
    "llama": "llama_qnn.json.j2",
}


@dataclass(frozen=True)
class OliveConfigPlan:
    family: str
    template_path: Path
    config_path: Path
    output_dir: Path
    primary_graph: Path


def _normalize_family(raw: str | None) -> str:
    text = (raw or "").strip().lower()
    if text.startswith("phi3"):
        return "phi3"
    if text.startswith("phi"):
        return "phi"
    if "llama" in text:
        return "llama"
    return text


def detect_supported_family(
    *,
    graphs: GraphBundle,
    quant_config: dict[str, Any],
) -> str:
    """Resolve model family and enforce initial Olive support scope."""
    hints = [
        quant_config.get("model_family"),
        graphs.metadata.get("model_family"),
        quant_config.get("model_type"),
        graphs.metadata.get("model_type"),
        quant_config.get("input_spec"),
    ]
    family = ""
    for hint in hints:
        family = _normalize_family(str(hint) if hint is not None else "")
        if family:
            break

    if family not in SUPPORTED_OLIVE_LLM_FAMILIES:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_MODEL_FAMILY_UNSUPPORTED",
            message=(
                "olive-qnn-llm currently supports only model families "
                f"{sorted(SUPPORTED_OLIVE_LLM_FAMILIES)}; got '{family or 'unknown'}'."
            ),
            hint=(
                "For unsupported families, use the experimental path:\n"
                "  --quant qnn-qdq --calib-prompts <prompts.txt>"
            ),
        )
    return family


# ---------------------------------------------------------------------------
# Legacy template rendering — retained for advanced/override use.
# ---------------------------------------------------------------------------

def _render_template(template_path: Path, values: dict[str, Any]) -> str:
    text = template_path.read_text(encoding="utf-8")
    for key, value in values.items():
        token_pattern = re.compile(r"{{\s*" + re.escape(key) + r"\s*}}")
        replacement = str(value)
        text = token_pattern.sub(lambda _m: replacement, text)
    return text


# ---------------------------------------------------------------------------
# Dict-based config generation (primary path)
# ---------------------------------------------------------------------------

def _resolve_input_model(
    quant_config: dict[str, Any],
    primary_graph: Path,
) -> dict[str, Any]:
    """Determine Olive input_model dict based on source info.

    For HF sources with a local materialized snapshot, uses ``HfModel``
    so Olive handles export/conversion with its own LLM-aware logic.
    Otherwise falls back to ``OnnxModel`` with the adapter-exported graph.
    """
    model_info = quant_config.get("model_info") or {}
    source = model_info.get("source") or {}
    model_dir = quant_config.get("model_dir")

    if source.get("type") == "hf" and model_dir:
        _log.info(
            "Using HfModel input for Olive (source: %s, local: %s)",
            source.get("repo_id", "unknown"),
            model_dir,
        )
        return {
            "type": "HfModel",
            "model_path": str(model_dir),
        }

    explicit_source = quant_config.get("olive_model_source")
    model_path = str(explicit_source) if explicit_source else str(primary_graph)
    return {
        "type": "OnnxModel",
        "model_path": model_path,
    }


def _build_data_configs(
    *,
    context_length: int = 1024,
    max_samples_weight: int = 128,
    max_samples_act: int = 256,
) -> list[dict[str, Any]]:
    """Build calibration data configs using wikitext-2.

    Returns two configs following Olive's public QNN LLM recipe pattern:
      - joined: for weight-related calibration
      - line-by-line: for activation calibration (OnnxStaticQuantization)
    """
    return [
        {
            "name": "wikitext2_train_joined",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {
                "data_name": "wikitext",
                "subset": "wikitext-2-raw-v1",
                "split": "train",
            },
            "pre_process_data_config": {
                "strategy": "join",
                "add_special_tokens": False,
                "max_seq_len": context_length,
                "max_samples": max_samples_weight,
            },
        },
        {
            "name": "wikitext2_train_act",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {
                "data_name": "wikitext",
                "subset": "wikitext-2-raw-v1",
                "split": "train",
            },
            "pre_process_data_config": {
                "strategy": "line-by-line",
                "add_special_tokens": True,
                "max_samples": max_samples_act,
                "max_seq_len": context_length,
            },
        },
    ]


def _build_quantize_passes(family: str) -> dict[str, dict[str, Any]]:
    """Build a quantize-stage pass chain for a supported LLM family.

    This produces a **split / static / QDQ handoff**, not a final deployable.
    EPContextBinaryGenerator and ComposeOnnxModels belong in the compile step
    and are intentionally excluded here.

    Pass ordering follows Olive's public Phi-3.5 QNN recipe pattern:
      CaptureSplitInfo → preprocessing → quantization → split → static.
    """
    passes: dict[str, dict[str, Any]] = {}

    # 1. Identify split points before any graph transformation.
    passes["capture_split_info"] = {
        "type": "CaptureSplitInfo",
        "num_splits": 4,
        "unique_embeds_lm_head_splits": True,
    }

    # 2. Preprocess graph for quantization compatibility.
    passes["preprocess"] = {
        "type": "OnnxQuantizationPreprocess",
    }

    # 3. Static quantization (CPU calibration — no CUDA required).
    passes["static_quant"] = {
        "type": "OnnxStaticQuantization",
        "data_config": "wikitext2_train_act",
        "activation_type": "uint16",
        "precision": "uint8",
        "calibration_providers": ["CPUExecutionProvider"],
        "quant_preprocess": True,
        "op_types_to_exclude": [
            "GatherBlockQuantized",
            "GroupQueryAttention",
            "MatMulNBits",
        ],
        "save_as_external_data": True,
    }

    # 4. Split into sub-graphs at the captured split points.
    passes["split_model"] = {
        "type": "SplitModel",
    }

    # 5. Freeze shapes to static batch / sequence length for QNN HTP.
    passes["static_llm"] = {
        "type": "StaticLLM",
        "batch_size": 1,
        "context_length": 64,
    }

    return passes


def _build_quantize_config(
    *,
    family: str,
    input_model: dict[str, Any],
    output_dir: str,
    cache_dir: str,
    context_length: int = 1024,
) -> dict[str, Any]:
    """Build a complete Olive quantize-stage config dict.

    Follows the RunConfig schema used by ``olive.run(config_dict)`` and
    ``python -m olive run --config``.  The structure mirrors Olive's public
    QNN LLM recipes (Phi-3.5, DeepSeek) but scoped to quantize-stage only.
    """
    return {
        "input_model": input_model,
        "systems": {
            "qnn_system": {
                "type": "LocalSystem",
                "accelerators": [
                    {
                        "device": "npu",
                        "execution_providers": ["QNNExecutionProvider"],
                    }
                ],
            }
        },
        "data_configs": _build_data_configs(context_length=context_length),
        "passes": _build_quantize_passes(family),
        "target": "qnn_system",
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "log_severity_level": 1,
        "no_artifacts": False,
        "evaluate_input_model": False,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_olive_config(
    *,
    graphs: GraphBundle,
    quant_config: dict[str, Any],
    work_dir: Path,
) -> OliveConfigPlan:
    """Create a family-specific Olive quantize-stage config.

    Uses dict-based config generation aligned with Olive's RunConfig schema.
    For HF model sources, emits ``HfModel`` input so Olive handles
    export/conversion with its own LLM-aware logic.
    """
    family = detect_supported_family(graphs=graphs, quant_config=quant_config)

    if not graphs.graphs:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_INPUT_EMPTY",
            message="No ONNX graphs were provided to olive-qnn-llm.",
        )

    primary_graph = next(iter(graphs.graphs.values()))

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "olive_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "olive_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / f"olive_{family}_qnn_config.json"

    input_model = _resolve_input_model(quant_config, primary_graph)
    context_length = int(quant_config.get("context_length", 1024))

    config = _build_quantize_config(
        family=family,
        input_model=input_model,
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
        context_length=context_length,
    )

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    _log.info(
        "Generated Olive config: family=%s, input_type=%s, passes=%s, config=%s",
        family,
        input_model["type"],
        list(config["passes"].keys()),
        config_path,
    )

    return OliveConfigPlan(
        family=family,
        template_path=config_path,
        config_path=config_path,
        output_dir=output_dir,
        primary_graph=primary_graph,
    )
