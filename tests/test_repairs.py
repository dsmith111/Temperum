"""Known checkpoint quirks are repaired in a hard-linked view of the snapshot, never in the snapshot."""
from __future__ import annotations

import json
import os
from pathlib import Path

from temperum import repairs


def test_extra_special_tokens_list_moves_to_additional_special_tokens() -> None:
    config = {"eos_token": "<|im_end|>", "extra_special_tokens": ["<|im_start|>", "<|im_end|>", "<|box_start|>"],
              "additional_special_tokens": ["<|im_end|>"]}
    fixed = repairs.extra_special_tokens_as_list(config)
    assert fixed == {"eos_token": "<|im_end|>",
                     "additional_special_tokens": ["<|im_end|>", "<|im_start|>", "<|box_start|>"]}
    assert config["extra_special_tokens"]  # the input is not modified
    assert repairs.extra_special_tokens_as_list({"extra_special_tokens": {"image_token": "<image>"}}) is None
    assert repairs.extra_special_tokens_as_list({"eos_token": "x"}) is None


def test_prepare_source_returns_the_snapshot_when_nothing_needs_repair(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "tokenizer_config.json").write_text(json.dumps({"eos_token": "x"}), encoding="utf-8")
    said: list[str] = []
    assert repairs.prepare_source(snapshot, tmp_path / "view", said.append) == (snapshot, [])
    assert not (tmp_path / "view").exists() and said == []


def test_prepare_source_links_everything_and_rewrites_the_repaired_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    (snapshot / ".cache" / "huggingface").mkdir(parents=True)
    (snapshot / ".cache" / "huggingface" / "meta").write_text("x", encoding="utf-8")
    (snapshot / "sub").mkdir()
    (snapshot / "sub" / "note.txt").write_text("n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"w" * 1000)
    (snapshot / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}), encoding="utf-8")
    bad = {"tokenizer_class": "Qwen2Tokenizer", "extra_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    (snapshot / "tokenizer_config.json").write_text(json.dumps(bad), encoding="utf-8")
    said: list[str] = []
    view, applied = repairs.prepare_source(snapshot, tmp_path / "job" / "source", said.append)
    assert view == tmp_path / "job" / "source" and len(applied) == 1 and "tokenizer_config.json" in applied[0]
    assert said and "repaired the source" in said[0] and "untouched" in said[0]
    assert (view / "model.safetensors").read_bytes() == b"w" * 1000
    assert os.stat(view / "model.safetensors").st_nlink == 2  # a hard link, not a copy
    assert (view / "sub" / "note.txt").read_text(encoding="utf-8") == "n"
    assert not (view / ".cache").exists()
    fixed = json.loads((view / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert fixed == {"tokenizer_class": "Qwen2Tokenizer", "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    assert json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8")) == bad  # untouched
    view2, _ = repairs.prepare_source(snapshot, tmp_path / "job" / "source", said.append)  # rerun replaces the view
    assert view2 == view and (view / "config.json").is_file()
