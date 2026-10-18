"""Maintained Olive QNN recipes (vendored from microsoft/olive-recipes) and how we adapt them.

Upstream recipes assume an x64 + NVIDIA host, GPTQModel and the `datasets` package. The
renderer normalizes their variations (three GPTQ pass types, old and new ModelBuilder
option names, `qnn_system` vs `target_system`, rotation inside GPTQModel vs a QuaRot pass)
and applies the quality mode, quantization device, target SoC, calibration loader and the
compile environment whose QNN build matches the runtime.

Quality modes:
  standard  QuaRot + round-to-nearest 8-bit weights. Calibration-free and near-lossless;
            the tested CPU default (M0).
  recipe    The recipe's own GPTQ settings (usually 4-bit), via Olive's native GPTQ.
            Needs CUDA.
  auto      `recipe` when quantizing on CUDA, otherwise `standard`.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.profiles import TargetProfile

UPSTREAM = Path(__file__).resolve().parent / "upstream"

# model_type -> curated recipe used as the template for other models of that family. The
# Qwen2, Qwen3 and Llama templates are the ones exercised in M0.
FAMILY_TEMPLATES: dict[str, str] = {
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "phi3": "microsoft/Phi-3.5-mini-instruct",
}

QUALITIES = ("auto", "standard", "recipe")
GPTQ_PASSES = ("GptqModel", "GptqQuantizer", "Gptq")
MODEL_BUILDER_RENAMES = {"block_size": "int4_block_size", "accuracy_level": "int4_accuracy_level",
                         "op_types_to_quantize": "int4_op_types_to_quantize"}
# Calibration names by upstream pre-processing strategy: GPTQ uses joined text, activation
# calibration uses individual lines.
DATA_NAMES = {"join": "calib_joined", "line-by-line": "calib_act"}
# Activation calibration sees text the way inference does (with BOS where the tokenizer adds
# one); GPTQ's joined text does not. Fixed rather than per recipe: upstream's Qwen3-0.6B
# recipe swaps the two, which is a no-op for Qwen tokenizers (no BOS) and wrong for Llama.
ADD_SPECIAL_TOKENS = {"join": False, "line-by-line": True}
# CPU calibration budget (samples, tokens per sample); CUDA keeps the recipe's own budget.
CPU_BUDGET = {"join": (32, 1024), "line-by-line": (64, 512)}
# Size budget for one compiled NPU context binary (one graph split). With the pinned QNN 2.42,
# contexts hold about 2 bytes per block weight at 8 bits and 1 at 4 bits. On the X1E78100 a
# 0.9 GB context loads (four of them together too), but a single 2.65 GB one fails with QNN
# error 1002 (memory allocation), so stay at about 1 GB each.
SPLIT_BUDGET_BYTES = 1024**3
CONTEXT_BYTES_PER_WEIGHT_BIT = 2 / 8


@dataclass(frozen=True)
class Recipe:
    repo: str
    file: Path
    upstream_path: str
    sha256: str

    def load(self) -> dict[str, Any]:
        data = json.loads(self.file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        return data


@dataclass(frozen=True)
class RecipeChoice:
    recipe: Recipe
    kind: str  # "curated" | "family"
    note: str


@lru_cache(maxsize=1)
def catalog_index() -> dict[str, Any]:
    data = json.loads((UPSTREAM / "index.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def catalog() -> dict[str, Recipe]:
    index = catalog_index()
    return {repo: Recipe(repo, UPSTREAM / item["file"], item["upstream_path"], item["sha256"])
            for repo, item in index["recipes"].items()}


def choose(repo: str | None, model_type: str | None, architecture: str | None) -> RecipeChoice:
    """The exact model's recipe if one is maintained, else its family's template recipe."""
    recipes = catalog()
    by_lower = {name.lower(): recipe for name, recipe in recipes.items()}
    if repo and repo.lower() in by_lower:
        return RecipeChoice(by_lower[repo.lower()], "curated", "maintained recipe for this model")
    family = (model_type or "").lower()
    if family in FAMILY_TEMPLATES and (architecture or "").endswith("ForCausalLM"):
        template = recipes[FAMILY_TEMPLATES[family]]
        return RecipeChoice(template, "family",
                            f"no recipe for this exact model; using the {family} family recipe "
                            f"({template.repo}); verification decides whether the result is usable")
    raise NpuModelError(
        "recipe", "UNSUPPORTED_ARCHITECTURE",
        f"No NPU recipe for {architecture or model_type or 'this model'}.",
        hint=f"Supported causal-LM families: {', '.join(sorted(FAMILY_TEMPLATES))}.")


def resolve_quality(quality: str, device: str) -> str:
    if quality not in QUALITIES:
        raise NpuModelError("recipe", "UNKNOWN_QUALITY", f"Unknown quality mode: {quality}",
                            hint=f"Use one of: {', '.join(QUALITIES)}")
    if quality == "auto":
        return "recipe" if device == "cuda" else "standard"
    if quality == "recipe" and device != "cuda":
        raise NpuModelError("recipe", "QUALITY_NEEDS_CUDA",
                            "--quality recipe runs GPTQ, which needs an NVIDIA GPU.",
                            hint="Use --quality standard (8-bit weights) on this machine.")
    return quality


@dataclass(frozen=True)
class RenderOptions:
    model_dir: Path
    target: TargetProfile
    quality: str  # resolved: "standard" | "recipe"
    device: str  # "cpu" | "cuda"
    corpus: Path
    olive_data_script: Path
    compile_python_dir: Path
    output_dir: Path
    cache_dir: Path
    family: bool = False  # the recipe is a family template, not this model's own
    modules: frozenset[str] = frozenset()  # checkpoint module names, for per-module overrides
    block_params: int | None = None  # parameters in transformer blocks, for split sizing
    num_splits: int | None = None  # explicit override


def _unsupported(recipe: Recipe, detail: str) -> NpuModelError:
    return NpuModelError("recipe", "UNSUPPORTED_RECIPE", f"Recipe for {recipe.repo} uses {detail}.",
                         hint="Re-vendor the recipes or extend the renderer.")


def _data_configs(recipe: Recipe, upstream: dict[str, Any], opts: RenderOptions
                  ) -> tuple[list[dict[str, Any]], dict[str, str]]:
    configs: list[dict[str, Any]] = []
    renames: dict[str, str] = {}
    for item in upstream.get("data_configs", []):
        pre = item.get("pre_process_data_config", {})
        strategy = pre.get("strategy")
        if strategy not in DATA_NAMES:
            raise _unsupported(recipe, f"calibration strategy {strategy!r}")
        name = DATA_NAMES[strategy]
        renames[item["name"]] = name
        if any(c["name"] == name for c in configs):
            continue
        samples, seq_len = (CPU_BUDGET[strategy] if opts.device != "cuda"
                            else (pre.get("max_samples", 128), pre.get("max_seq_len", 2048)))
        configs.append({
            "name": name,
            "type": "DataContainer",
            "user_script": str(opts.olive_data_script),
            "load_dataset_config": {"type": "npu_jsonl_text", "params": {"data_files": str(opts.corpus)}},
            "pre_process_data_config": {"type": "npu_text_gen", "params": {
                "model_name": str(opts.model_dir), "strategy": strategy, "max_seq_len": seq_len,
                "max_samples": samples, "add_special_tokens": ADD_SPECIAL_TOKENS[strategy]}},
            "dataloader_config": {"type": "default_dataloader", "params": {"batch_size": 1}},
        })
    return configs, renames


def _overrides(recipe: Recipe, dynamic: dict[str, Any] | None, modules: frozenset[str],
               base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """GPTQModel `dynamic` regex rules -> Olive per-module overrides (first match wins)."""
    result: dict[str, dict[str, Any]] = {}
    for key, spec in (dynamic or {}).items():
        if key.startswith("-:"):
            raise _unsupported(recipe, f"a module exclusion rule ({key})")
        pattern = re.compile(key[2:] if key.startswith("+:") else key)
        override = {"bits": int(spec.get("bits", base["bits"])),
                    "symmetric": bool(spec.get("sym", base["sym"])),
                    "group_size": int(spec.get("group_size", base["group_size"]))}
        if (override["bits"], override["symmetric"], override["group_size"]) == (
                base["bits"], base["sym"], base["group_size"]):
            continue
        for name in sorted(modules | {"lm_head"}):
            if pattern.match(name):
                result.setdefault(name, override)
    return result


def _weight_quantizer(recipe: Recipe, p: dict[str, Any], opts: RenderOptions,
                      renames: dict[str, str]) -> dict[str, Any]:
    base = {"bits": 8 if opts.quality == "standard" else int(p.get("bits", 4)),
            "sym": bool(p.get("sym", True)), "group_size": int(p.get("group_size", -1))}
    quantizer: dict[str, Any] = {"type": "Rtn" if opts.quality == "standard" else "Gptq", **base,
                                 "lm_head": bool(p.get("lm_head", False))}
    overrides = _overrides(recipe, p.get("dynamic"), opts.modules, base)
    if overrides:
        quantizer["overrides"] = overrides
    if opts.quality == "recipe":
        if "desc_act" in p:
            quantizer["desc_act"] = bool(p["desc_act"])
        quantizer["data_config"] = renames.get(p.get("data_config", ""), "calib_joined")
    return quantizer


def split_count(upstream_splits: int, opts: RenderOptions, bits: int) -> int:
    if opts.num_splits:
        return opts.num_splits
    if opts.block_params is None:
        return upstream_splits
    needed = opts.block_params * bits * CONTEXT_BYTES_PER_WEIGHT_BIT / SPLIT_BUDGET_BYTES
    splits = 1
    while splits < needed:  # powers of two, like the upstream recipes (1 or 4)
        splits *= 2
    return max(1 if opts.family else upstream_splits, splits)


def render(recipe: Recipe, opts: RenderOptions) -> dict[str, Any]:
    """Olive run config for `recipe`, adapted to this machine, target, quality and runtime."""
    if opts.quality not in ("standard", "recipe"):
        raise NpuModelError("recipe", "UNKNOWN_QUALITY", f"Unresolved quality mode: {opts.quality}")
    upstream = recipe.load()
    cfg = copy.deepcopy(upstream)
    cfg["input_model"] = {"type": "HfModel", "model_path": str(opts.model_dir)}
    compile_systems = 0
    for system in cfg.get("systems", {}).values():
        providers = [ep for acc in system.get("accelerators", []) for ep in acc.get("execution_providers", [])]
        if "QNNExecutionProvider" in providers:
            system["python_environment_path"] = str(opts.compile_python_dir)
            compile_systems += 1
    if compile_systems != 1:
        raise _unsupported(recipe, f"{compile_systems} QNN systems (expected one)")
    cfg["data_configs"], renames = _data_configs(recipe, upstream, opts)

    rotation: tuple[str, dict[str, Any]] | None = None
    passes: dict[str, Any] = {}
    bits = 8
    for key, value in upstream["passes"].items():
        p = copy.deepcopy(value)
        kind = p["type"]
        if kind == "QuaRot":
            rotation = (key, p)
            continue
        if kind in GPTQ_PASSES:
            if p.get("rotation") and rotation is None:
                rotation = ("q", {"type": "QuaRot", "rotate_mode": str(p["rotation"])})
            passes[key] = _weight_quantizer(recipe, p, opts, renames)
            bits = int(passes[key]["bits"])
            continue
        if "data_config" in p:
            if p["data_config"] not in renames:
                raise _unsupported(recipe, f"unknown data config {p['data_config']!r} in pass {key}")
            p["data_config"] = renames[p["data_config"]]
        if kind == "ModelBuilder":
            p = {MODEL_BUILDER_RENAMES.get(k, k): v for k, v in p.items()}
        elif kind == "OnnxStaticQuantization":
            p["calibration_providers"] = ["CUDAExecutionProvider" if opts.device == "cuda" else "CPUExecutionProvider"]
        elif kind == "EPContextBinaryGenerator":
            p.setdefault("provider_options", {})["soc_model"] = opts.target.soc_model
        passes[key] = p
    for p in passes.values():  # sized after the quantizer's bit width is known
        if p["type"] == "CaptureSplitInfo":
            p["num_splits"] = split_count(int(p.get("num_splits", 1)), opts, bits)
    if rotation is not None:  # rotate before splitting and quantizing, as the Qwen3 recipes do
        passes = {rotation[0]: rotation[1], **passes}
    cfg["passes"] = passes
    cfg["output_dir"] = str(opts.output_dir)
    cfg["cache_dir"] = str(opts.cache_dir)
    cfg["log_severity_level"] = 1
    cfg["no_artifacts"] = True
    return cfg


def verified_conversions() -> list[dict[str, Any]]:
    """Conversions that were checked end to end on hardware (see verified.json)."""
    data = json.loads((Path(__file__).resolve().parent / "verified.json").read_text(encoding="utf-8"))
    return list(data["entries"])


def precision_label(config: dict[str, Any]) -> str:
    """Weight/activation precision of a rendered config, e.g. w8a16-rtn or w4a16-gptq."""
    for p in config.get("passes", {}).values():
        if p.get("type") in ("Rtn", "Gptq"):
            return f"w{p['bits']}a16-{p['type'].lower()}"
    return "unknown"
