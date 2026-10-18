from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.profiles import get_target
from npu_model.recipes import (
    FAMILY_TEMPLATES,
    RenderOptions,
    catalog,
    catalog_index,
    choose,
    precision_label,
    render,
    resolve_quality,
)

M0 = Path(__file__).parent / "fixtures" / "m0_configs"
QWEN_MODULES = frozenset({"model.embed_tokens", "model.layers.0.self_attn.v_proj",
                          "model.layers.0.mlp.down_proj", "model.norm"})


def _opts(**overrides: Any) -> RenderOptions:
    values: dict[str, Any] = dict(
        model_dir=Path("<MODEL>"), target=get_target("qnn-x-elite"), quality="standard", device="cpu",
        corpus=Path("<CORPUS>"), olive_data_script=Path("<SCRIPT>"), compile_python_dir=Path("<COMPILE>"),
        output_dir=Path("<OUT>"), cache_dir=Path("<CACHE>"), modules=QWEN_MODULES, block_params=358_000_000)
    values.update(overrides)
    return RenderOptions(**values)


def _expected(run: str) -> dict[str, Any]:
    data = json.loads((M0 / f"{run}.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_vendored_recipes_match_their_recorded_hashes() -> None:
    index = catalog_index()
    assert index["license"] == "MIT" and len(index["commit"]) == 40
    for recipe in catalog().values():
        assert hashlib.sha256(recipe.file.read_bytes()).hexdigest() == recipe.sha256, recipe.repo
    for template in FAMILY_TEMPLATES.values():
        assert template in catalog()


@pytest.mark.parametrize("run,repo,model_type,family", [
    ("q05-w8rtn", "Qwen/Qwen2.5-0.5B-Instruct", "qwen2", False),
    ("qwen3-w8rtn", "Qwen/Qwen3-0.6B", "qwen3", False),
    ("smol-w8rtn", "HuggingFaceTB/SmolLM2-360M-Instruct", "llama", True),
])
def test_standard_quality_reproduces_the_m0_configs(run: str, repo: str, model_type: str, family: bool) -> None:
    """The configs that converted, compiled and served correctly on the X Elite in M0."""
    arch = {"qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM", "llama": "LlamaForCausalLM"}[model_type]
    choice = choose(repo, model_type, arch)
    assert choice.kind == ("family" if family else "curated")
    rendered = render(choice.recipe, _opts(family=family))
    expected = _expected(run)
    assert list(rendered["passes"]) == list(expected["passes"])
    assert rendered == expected
    assert precision_label(rendered) == "w8a16-rtn"


def test_every_vendored_recipe_renders_for_both_devices() -> None:
    for recipe in catalog().values():
        for quality, device in (("standard", "cpu"), ("recipe", "cuda")):
            cfg = render(recipe, _opts(quality=quality, device=device, block_params=None))
            kinds = [p["type"] for p in cfg["passes"].values()]
            assert kinds[0] == "QuaRot" and kinds[-1] == "ComposeOnnxModels", recipe.repo
            assert not {"GptqModel", "GptqQuantizer"} & set(kinds), recipe.repo
            assert ("Rtn" if quality == "standard" else "Gptq") in kinds, recipe.repo
            assert {d["name"] for d in cfg["data_configs"]} == {"calib_joined", "calib_act"}
            assert all(p.get("data_config") in (None, "calib_joined", "calib_act") for p in cfg["passes"].values())
            for p in cfg["passes"].values():
                if p["type"] == "ModelBuilder":
                    assert not {"block_size", "accuracy_level", "op_types_to_quantize"} & set(p)
                if p["type"] == "OnnxStaticQuantization":
                    expected_ep = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
                    assert p["calibration_providers"] == [expected_ep]


def test_recipe_quality_keeps_gptq_settings_and_mixed_precision() -> None:
    recipe = catalog()["meta-llama/Llama-3.1-8B-Instruct"]
    modules = frozenset({f"model.layers.{i}.self_attn.{proj}" for i in range(2) for proj in ("q_proj", "v_proj")})
    cfg = render(recipe, _opts(quality="recipe", device="cuda", modules=modules, block_params=None))
    gptq = cfg["passes"]["g"]
    assert (gptq["type"], gptq["bits"], gptq["data_config"]) == ("Gptq", 4, "calib_joined")
    assert set(gptq["overrides"]) == {"model.layers.0.self_attn.v_proj", "model.layers.1.self_attn.v_proj",
                                      "lm_head"}
    assert gptq["overrides"]["model.layers.0.self_attn.v_proj"] == {"bits": 8, "symmetric": True, "group_size": -1}
    assert gptq["overrides"]["lm_head"] == {"bits": 4, "symmetric": True, "group_size": 32}
    joined = next(d for d in cfg["data_configs"] if d["name"] == "calib_joined")
    assert joined["pre_process_data_config"]["params"]["max_samples"] == 128  # recipe budget on CUDA
    assert precision_label(cfg) == "w4a16-gptq"


def test_standard_quality_drops_overrides_that_equal_the_base() -> None:
    recipe = catalog()["meta-llama/Llama-3.1-8B-Instruct"]
    cfg = render(recipe, _opts(modules=frozenset({"model.layers.0.self_attn.v_proj"}), block_params=None))
    assert cfg["passes"]["g"]["overrides"] == {"lm_head": {"bits": 4, "symmetric": True, "group_size": 32}}


def test_split_count_follows_model_size() -> None:
    recipe = catalog()["Qwen/Qwen2.5-0.5B-Instruct"]
    small = render(recipe, _opts(family=True, block_params=440_000_000))  # Qwen3-0.6B: 0.9 GB context
    middle = render(recipe, _opts(family=True, block_params=1_310_000_000))  # Qwen2.5-1.5B: 2.6 GB
    large = render(recipe, _opts(family=True, block_params=6_500_000_000))
    forced = render(recipe, _opts(num_splits=3))
    assert small["passes"]["cs"]["num_splits"] == 1
    assert middle["passes"]["cs"]["num_splits"] == 4  # one 2.65 GB context failed to load
    assert large["passes"]["cs"]["num_splits"] == 16  # ~13 GB of contexts at 8 bits
    assert forced["passes"]["cs"]["num_splits"] == 3
    curated = render(catalog()["Qwen/Qwen2.5-7B-Instruct"], _opts(block_params=None))
    assert curated["passes"]["cs"]["num_splits"] == 4  # recipe value kept when size is unknown


def test_target_soc_and_compile_env_are_applied() -> None:
    cfg = render(catalog()["Qwen/Qwen3-0.6B"], _opts(target=get_target("qnn-x2-elite")))
    assert cfg["passes"]["cb"]["provider_options"]["soc_model"] == "87"
    assert cfg["systems"]["qnn_system"]["python_environment_path"] == "<COMPILE>"


def test_choose_matches_repo_ids_case_insensitively() -> None:
    assert choose("qwen/qwen2.5-0.5b-instruct", "qwen2", "Qwen2ForCausalLM").kind == "curated"


@pytest.mark.parametrize("model_type,arch", [("gpt2", "GPT2LMHeadModel"), ("llama", "LlamaForSequenceClassification"),
                                             (None, None)])
def test_unsupported_models_are_rejected(model_type: str | None, arch: str | None) -> None:
    with pytest.raises(NpuModelError) as exc:
        choose("org/model", model_type, arch)
    assert exc.value.reason_code == "UNSUPPORTED_ARCHITECTURE"


@pytest.mark.parametrize("quality,device,expected", [
    ("auto", "cpu", "standard"), ("auto", "cuda", "recipe"), ("standard", "cuda", "standard"),
])
def test_quality_resolution(quality: str, device: str, expected: str) -> None:
    assert resolve_quality(quality, device) == expected


def test_recipe_quality_needs_cuda() -> None:
    with pytest.raises(NpuModelError) as exc:
        resolve_quality("recipe", "cpu")
    assert exc.value.reason_code == "QUALITY_NEEDS_CUDA"
    with pytest.raises(NpuModelError) as exc:
        resolve_quality("best", "cpu")
    assert exc.value.reason_code == "UNKNOWN_QUALITY"
