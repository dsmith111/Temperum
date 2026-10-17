from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_bundle_inspect import fake_onnx as fake_onnx

from npu_model.assemble import AssembleInputs, assemble, prompt_token_ids, safe_pad_token
from npu_model.core.errors import NpuModelError
from npu_model.profiles import get_runtime, get_target
from npu_model.templates import from_jinja

FIXTURES = Path(__file__).parent / "fixtures" / "templates"
CPU = "Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU"
SMOL_TOKENS = {"0": {"content": "<|endoftext|>", "special": True}, "1": {"content": "<|im_start|>", "special": True},
               "2": {"content": "<|im_end|>", "special": True}}


def _olive_output(root: Path, context: int = 8192, pad: int = 2, window: int | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "model_ctx.onnx").write_bytes(b"stub-graph")
    (root / "context.qnn").write_bytes(b"stub-context")
    (root / "tokenizer.json").write_text('{"version":"1.0","model":{"type":"BPE","vocab":{}}}', encoding="utf-8")
    decoder: dict[str, Any] = {"filename": "model_ctx.onnx",
                               "session_options": {"provider_options": [{"qnn": {"backend_type": "htp"}}]}}
    if window:
        decoder["sliding_window"] = {"window_size": window, "pad_value": 0, "alignment": "left"}
    config = {"model": {"type": "llama", "context_length": context, "vocab_size": 10, "eos_token_id": 2,
                        "pad_token_id": pad, "decoder": decoder}, "search": {"max_length": context}}
    (root / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def _snapshot(root: Path) -> Path:
    root.mkdir()
    jinja = (FIXTURES / "smollm2-instruct.jinja").read_text(encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": SMOL_TOKENS,
                                                            "chat_template": jinja}), encoding="utf-8")
    (root / "LICENSE").write_text("Apache License 2.0", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def _inputs(tmp_path: Path, **overrides: Any) -> AssembleInputs:
    snapshot = tmp_path / "snapshot"
    values: dict[str, Any] = dict(
        olive_output=tmp_path / "olive", snapshot=snapshot, destination=tmp_path / "bundle",
        target=get_target("qnn-x-elite"), runtime=get_runtime("fork-qnn-1.24.1"),
        chat=from_jinja((FIXTURES / "smollm2-instruct.jinja").read_text(encoding="utf-8")), context=None,
        precision="w8a16-rtn", processor_name=CPU, source={"repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
                                                           "commit": "c" * 40, "license": "apache-2.0"},
        provenance={"recipe": "meta-llama/Llama-3.2-1B-Instruct"})
    values.update(overrides)
    return AssembleInputs(**values)


def test_bundle_gets_declaration_template_manifest_and_safe_pad(tmp_path: Path, fake_onnx: SimpleNamespace) -> None:
    _olive_output(tmp_path / "olive")
    _snapshot(tmp_path / "snapshot")
    final, inspection = assemble(_inputs(tmp_path))
    assert inspection.importable and inspection.target["status"] == "declared-match"
    cfg = json.loads((final / "genai_config.json").read_text(encoding="utf-8"))
    assert cfg["model"]["context_length"] == 4096 and cfg["search"]["max_length"] == 4096
    assert cfg["model"]["pad_token_id"] == 0  # 2 is <|im_end|>, which every chat prompt contains
    decl = json.loads((final / "ollama_npu.json").read_text(encoding="utf-8"))
    assert decl["stages"] == [{"filename": "model_ctx.onnx", "provider": "qnn"}]
    assert decl["processor_names"] == [CPU] and decl["context_length"] == 4096
    assert decl["source"] == {"repository": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
                              "revision": "c" * 40}
    assert (final / "Modelfile").read_text(encoding="utf-8").startswith("FROM .\nTEMPLATE")
    assert (final / "LICENSE").is_file() and not (final / "model.safetensors").exists()
    manifest = json.loads((final / "npu_model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["genai_config_changes"]["pad_token_id"] == 0
    assert manifest["chat_template"]["family"] == "chatml" and "genai_config.json" in manifest["files"]
    assert not (tmp_path / "bundle.partial").exists()
    olive_cfg = json.loads((tmp_path / "olive" / "genai_config.json").read_text(encoding="utf-8"))
    assert olive_cfg["model"]["context_length"] == 8192 and olive_cfg["model"]["pad_token_id"] == 2


def test_existing_destination_needs_overwrite(tmp_path: Path, fake_onnx: SimpleNamespace) -> None:
    _olive_output(tmp_path / "olive")
    _snapshot(tmp_path / "snapshot")
    (tmp_path / "bundle").mkdir()
    with pytest.raises(NpuModelError) as exc:
        assemble(_inputs(tmp_path))
    assert exc.value.reason_code == "DESTINATION_EXISTS"
    final, inspection = assemble(_inputs(tmp_path, overwrite=True))
    assert inspection.importable and (final / "ollama_npu.json").is_file()


def test_context_is_whole_windows_within_the_model_maximum(tmp_path: Path, fake_onnx: SimpleNamespace) -> None:
    _olive_output(tmp_path / "olive", context=2048, window=64)
    _snapshot(tmp_path / "snapshot")
    final, _ = assemble(_inputs(tmp_path, context=1000))
    assert json.loads((final / "genai_config.json").read_text(encoding="utf-8"))["model"]["context_length"] == 960
    for bad in (100, 4096):
        with pytest.raises(NpuModelError) as exc:
            assemble(_inputs(tmp_path, context=bad, overwrite=True))
        assert exc.value.reason_code == "BAD_CONTEXT"


def test_without_a_processor_name_no_declaration_is_written(tmp_path: Path, fake_onnx: SimpleNamespace) -> None:
    _olive_output(tmp_path / "olive")
    _snapshot(tmp_path / "snapshot")
    final, inspection = assemble(_inputs(tmp_path, processor_name=None))
    assert inspection.importable and inspection.declaration_status == "absent"
    assert not (final / "ollama_npu.json").exists()


def test_pad_token_rules() -> None:
    jinja = "{{ bos_token }}<|start_header_id|>user<|end_header_id|>{{ content }}<|eot_id|>"
    tokens = {"128000": {"content": "<|begin_of_text|>"}, "128004": {"content": "<|finetune_right_pad_id|>"},
              "128006": {"content": "<|start_header_id|>"}, "128007": {"content": "<|end_header_id|>"},
              "128009": {"content": "<|eot_id|>"}}
    config = {"added_tokens_decoder": tokens, "bos_token": "<|begin_of_text|>"}
    assert prompt_token_ids(config, jinja) == {128000, 128006, 128007, 128009}
    assert safe_pad_token(128009, config, jinja) == 128004  # a dedicated pad token wins
    assert safe_pad_token(128001, config, jinja) == 128001  # unused in prompts: kept
    assert safe_pad_token(5, {}, None) == 5  # no chat template: nothing to protect
    with pytest.raises(NpuModelError) as exc:
        safe_pad_token(2, {"added_tokens_decoder": {"2": {"content": "<|im_end|>"}}}, "<|im_end|>")
    assert exc.value.reason_code == "PAD_TOKEN_CONFLICT"
