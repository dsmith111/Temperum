"""Speculative-decoding draft checkpoints are refused when resolved, with their target model named."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from temperum import sources
from temperum.core.errors import TemperumError

DFLASH = {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "num_hidden_layers": 6,
          "dflash_config": {"block_size": 16}, "num_target_layers": 48}


def test_draft_models_are_recognized_by_architecture_keys_or_tags() -> None:
    assert sources.draft_model_reason(DFLASH) == "architecture DFlash2DraftModel"
    assert sources.draft_model_reason({"architectures": ["LlamaForCausalLMEagle3"], "draft_vocab_size": 32000,
                                       "target_hidden_size": 4096}) == "config has draft_vocab_size, target_hidden_size"
    assert sources.draft_model_reason({"architectures": ["LlamaForCausalLM"]}, ["draft-model"]) == \
        "tagged draft-model on Hugging Face"
    assert sources.draft_model_reason({"architectures": ["LlamaForCausalLM"]}, ["text-generation"]) is None
    assert sources.base_model_tag(["base_model:finetune:org/big", "base_model:org/big", "x"]) == "org/big"
    assert sources.base_model_tag(["base_model:finetune:org/big"]) is None


def test_resolve_refuses_a_hub_draft_model_and_names_its_target(tmp_path: Path) -> None:
    class Api:
        def model_info(self, repo: str, revision: Any = None, files_metadata: bool = False) -> Any:
            return SimpleNamespace(sha="a" * 40, gated=False, card_data={"license": "apache-2.0"},
                                   siblings=[SimpleNamespace(rfilename="config.json", size=10),
                                             SimpleNamespace(rfilename="model.safetensors", size=1000)],
                                   tags=["qwen3", "dflash2", "draft-model", "base_model:ornith-ai/Ornith-1.5-122B",
                                         "base_model:finetune:ornith-ai/Ornith-1.5-122B"])

    def download(repo: str, filename: str, revision: Any = None) -> str:
        path = tmp_path / filename
        path.write_text(json.dumps(DFLASH), encoding="utf-8")
        return str(path)

    with pytest.raises(TemperumError) as exc:
        sources.resolve(sources.parse_ref("incoai/Ornith-1.5-122B-DFlash2"), api=Api(), download=download)
    assert exc.value.reason_code == "DRAFT_MODEL"
    assert "DFlash2DraftModel" in str(exc.value) and "cannot answer on its own" in str(exc.value)
    assert exc.value.hint and "ornith-ai/Ornith-1.5-122B" in exc.value.hint and "--plan" in exc.value.hint


def test_resolve_refuses_a_local_draft_model(tmp_path: Path) -> None:
    folder = tmp_path / "draft"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "medusa_num_heads": 4}),
                                        encoding="utf-8")
    with pytest.raises(TemperumError) as exc:
        sources.resolve(sources.parse_ref(str(folder)))
    assert exc.value.reason_code == "DRAFT_MODEL" and "medusa_num_heads" in str(exc.value)
    assert exc.value.hint and "trained for" in exc.value.hint
