from __future__ import annotations

import json
from pathlib import Path

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle
from npu_model.olive import config_builder


# ---------------------------------------------------------------------------
# Legacy template renderer tests (retained — _render_template still exists)
# ---------------------------------------------------------------------------

def test_render_template_preserves_windows_backslashes(tmp_path: Path) -> None:
    tpl = tmp_path / "tpl.j2"
    tpl.write_text('{"model_path": {{ input_model_path }}}', encoding="utf-8")

    windows_path = r"C:\Users\dsmith111\models\phi2\model.onnx"
    rendered = config_builder._render_template(
        tpl,
        {"input_model_path": json.dumps(windows_path)},
    )
    parsed = json.loads(rendered)
    assert parsed["model_path"] == windows_path


# ---------------------------------------------------------------------------
# Dict-based config builder tests
# ---------------------------------------------------------------------------

def _make_bundle(tmp_path: Path, *, family_hint: str = "phi") -> GraphBundle:
    graph = tmp_path / "model.onnx"
    graph.write_bytes(b"\x00" * 16)
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()
    return GraphBundle(
        graphs={"decoder": graph},
        tokenizer_dir=tok_dir,
        extra_files=[],
        metadata={},
    )


def test_build_olive_config_uses_target_not_engine(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={"model_family": "phi"},
        work_dir=tmp_path / "work",
    )
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))

    assert "engine" not in config
    assert config["target"] == "qnn_system"
    assert "cache_dir" in config
    assert config.get("evaluate_input_model") is False
    assert config.get("no_artifacts") is False


def test_build_olive_config_has_data_configs(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={"model_family": "phi"},
        work_dir=tmp_path / "work",
    )
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))

    assert "data_configs" in config
    data_configs = config["data_configs"]
    assert len(data_configs) == 2
    names = {dc["name"] for dc in data_configs}
    assert "wikitext2_train_joined" in names
    assert "wikitext2_train_act" in names

    # Verify joined config uses join strategy
    joined = next(dc for dc in data_configs if dc["name"] == "wikitext2_train_joined")
    assert joined["pre_process_data_config"]["strategy"] == "join"

    # Verify act config uses line-by-line strategy
    act = next(dc for dc in data_configs if dc["name"] == "wikitext2_train_act")
    assert act["pre_process_data_config"]["strategy"] == "line-by-line"


def test_build_olive_config_has_recipe_passes(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={"model_family": "phi"},
        work_dir=tmp_path / "work",
    )
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))

    passes = config["passes"]
    assert "capture_split_info" in passes
    assert passes["capture_split_info"]["type"] == "CaptureSplitInfo"
    assert passes["capture_split_info"]["num_splits"] == 4

    assert "preprocess" in passes
    assert passes["preprocess"]["type"] == "OnnxQuantizationPreprocess"

    assert "static_quant" in passes
    assert passes["static_quant"]["type"] == "OnnxStaticQuantization"
    assert passes["static_quant"]["data_config"] == "wikitext2_train_act"
    assert "CPUExecutionProvider" in passes["static_quant"]["calibration_providers"]

    assert "split_model" in passes
    assert passes["split_model"]["type"] == "SplitModel"

    assert "static_llm" in passes
    assert passes["static_llm"]["type"] == "StaticLLM"


def test_build_olive_config_hfmodel_when_hf_source(tmp_path: Path) -> None:
    """When source is HF with a local model_dir, config uses HfModel."""
    bundle = _make_bundle(tmp_path)
    model_dir = tmp_path / "hf_snapshot"
    model_dir.mkdir()

    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={
            "model_family": "phi",
            "model_dir": str(model_dir),
            "model_info": {
                "source": {"type": "hf", "repo_id": "microsoft/phi-2"},
                "model_type": "phi",
                "architectures": ["PhiForCausalLM"],
            },
        },
        work_dir=tmp_path / "work",
    )
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))

    assert config["input_model"]["type"] == "HfModel"
    assert config["input_model"]["model_path"] == str(model_dir)


def test_build_olive_config_onnxmodel_when_no_hf_source(tmp_path: Path) -> None:
    """When no HF source info, config uses OnnxModel with graph path."""
    bundle = _make_bundle(tmp_path)
    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={"model_family": "phi"},
        work_dir=tmp_path / "work",
    )
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))

    assert config["input_model"]["type"] == "OnnxModel"
    assert "model_path" in config["input_model"]


def test_build_olive_config_context_length_propagates(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={"model_family": "phi", "context_length": 2048},
        work_dir=tmp_path / "work",
    )
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))

    for dc in config["data_configs"]:
        assert dc["pre_process_data_config"]["max_seq_len"] == 2048


def test_build_olive_config_no_ep_context_binary_generator(tmp_path: Path) -> None:
    """Quantize-stage config must NOT include compile-time passes."""
    bundle = _make_bundle(tmp_path)
    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={"model_family": "phi"},
        work_dir=tmp_path / "work",
    )
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))

    pass_types = {p["type"] for p in config["passes"].values()}
    assert "EPContextBinaryGenerator" not in pass_types
    assert "ComposeOnnxModels" not in pass_types


def test_build_olive_config_rejects_unsupported_family(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    with pytest.raises(NpuModelError) as exc_info:
        config_builder.build_olive_config(
            graphs=bundle,
            quant_config={"model_family": "falcon"},
            work_dir=tmp_path / "work",
        )
    assert exc_info.value.reason_code == "OLIVE_MODEL_FAMILY_UNSUPPORTED"


def test_build_olive_config_windows_paths_in_json(tmp_path: Path) -> None:
    """Config with Windows-style paths must produce valid JSON."""
    graph = tmp_path / "C_Users_test_model.onnx"
    graph.write_bytes(b"\x00" * 16)
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()
    bundle = GraphBundle(
        graphs={"decoder": graph},
        tokenizer_dir=tok_dir,
        extra_files=[],
        metadata={},
    )
    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={
            "model_family": "phi",
            "model_dir": r"C:\Users\dsmith111\models\phi2",
            "model_info": {
                "source": {"type": "hf", "repo_id": "microsoft/phi-2"},
            },
        },
        work_dir=tmp_path / "work",
    )
    # Must parse without error
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))
    assert config["input_model"]["model_path"] == r"C:\Users\dsmith111\models\phi2"


def test_build_olive_config_systems_has_accelerators(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    plan = config_builder.build_olive_config(
        graphs=bundle,
        quant_config={"model_family": "llama"},
        work_dir=tmp_path / "work",
    )
    config = json.loads(plan.config_path.read_text(encoding="utf-8"))

    system = config["systems"]["qnn_system"]
    assert system["type"] == "LocalSystem"
    assert len(system["accelerators"]) == 1
    acc = system["accelerators"][0]
    assert acc["device"] == "npu"
    assert "QNNExecutionProvider" in acc["execution_providers"]
