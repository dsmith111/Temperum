"""Speculative-decoding draft checkpoints are recognized when resolved, with the target their card names; the
GGUF route pairs them (test_draft_pair.py), the QNN route refuses them with the pairing command."""
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
CARD = ("# Qwen3-4B-DFlash-b16\n\nThis model is the **drafter** component. It must be used in conjunction with the "
        "target model `Qwen/Qwen3-4B`.\n")


def test_draft_models_are_recognized_by_architecture_keys_or_tags() -> None:
    assert sources.draft_model_reason(DFLASH) == "architecture DFlash2DraftModel"
    assert sources.draft_model_reason({"architectures": ["LlamaForCausalLMEagle3"], "draft_vocab_size": 32000,
                                       "target_hidden_size": 4096}) == "config has draft_vocab_size, target_hidden_size"
    assert sources.draft_model_reason({"architectures": ["LlamaForCausalLM"]}, ["draft-model"]) == \
        "tagged draft-model on Hugging Face"
    assert sources.draft_model_reason({"architectures": ["LlamaForCausalLM"]}, ["text-generation"]) is None
    assert sources.base_model_tag(["base_model:finetune:org/big", "base_model:org/big", "x"]) == "org/big"
    assert sources.base_model_tag(["base_model:finetune:org/big"]) is None


def test_block_size_pairability_and_the_target_named_in_a_card() -> None:
    assert sources.draft_block_size(DFLASH) == 16
    assert sources.draft_block_size({"block_size": 8}) == 8
    assert sources.draft_block_size({"block_size": True}) is None and sources.draft_block_size({}) is None
    assert sources.pairable_draft(DFLASH) and sources.pairable_draft({"architectures": ["DFlashDraftModel"]})
    assert not sources.pairable_draft({"architectures": ["LlamaForCausalLMEagle3"], "draft_vocab_size": 1})
    assert sources.target_in_card(CARD) == "Qwen/Qwen3-4B"
    assert sources.target_in_card("No pairing here.") is None


class _Api:
    def __init__(self, tags: list[str], files: list[str]) -> None:
        self.tags, self.files = tags, files

    def model_info(self, repo: str, revision: Any = None, files_metadata: bool = False) -> Any:
        return SimpleNamespace(sha="a" * 40, gated=False, card_data={"license": "mit"},
                               siblings=[SimpleNamespace(rfilename=f, size=10) for f in self.files], tags=self.tags)


def _download(tmp_path: Path, config: dict[str, Any], card: str | None = None) -> Any:
    def download(repo: str, filename: str, revision: Any = None) -> str:
        path = tmp_path / filename
        path.write_text(json.dumps(config) if filename == "config.json" else (card or ""), encoding="utf-8")
        return str(path)
    return download


def test_resolve_marks_a_hub_draft_and_takes_the_target_from_its_tags(tmp_path: Path) -> None:
    api = _Api(["qwen3", "dflash2", "draft-model", "base_model:ornith-ai/Ornith-1.5-122B",
                "base_model:finetune:ornith-ai/Ornith-1.5-122B"], ["config.json", "model.safetensors"])
    model = sources.resolve(sources.parse_ref("incoai/Ornith-1.5-122B-DFlash2"), api=api,
                            download=_download(tmp_path, DFLASH))
    assert model.draft_reason == "architecture DFlash2DraftModel" and model.draft_target == "ornith-ai/Ornith-1.5-122B"
    assert model.to_dict()["draft"] == model.draft_reason and model.to_dict()["draft_target"] == model.draft_target


def test_resolve_reads_the_target_from_the_card_when_no_tag_names_it(tmp_path: Path) -> None:
    api = _Api(["dflash", "speculative-decoding"], ["config.json", "model.safetensors", "README.md"])
    model = sources.resolve(sources.parse_ref("z-lab/Qwen3-4B-DFlash-b16"), api=api,
                            download=_download(tmp_path, DFLASH, CARD))
    assert model.draft_reason and model.draft_target == "Qwen/Qwen3-4B"
    plain = sources.resolve(sources.parse_ref("Qwen/Qwen3-4B"), api=api,
                            download=_download(tmp_path, {"architectures": ["Qwen3ForCausalLM"]}, CARD))
    assert plain.draft_reason is None and plain.draft_target is None  # a card is read for drafts only


def test_resolve_marks_a_local_draft(tmp_path: Path) -> None:
    folder = tmp_path / "draft"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "medusa_num_heads": 4}),
                                        encoding="utf-8")
    (folder / "README.md").write_text(CARD, encoding="utf-8")
    model = sources.resolve(sources.parse_ref(str(folder)))
    assert model.draft_reason == "config has medusa_num_heads" and model.draft_target == "Qwen/Qwen3-4B"


def test_the_qnn_route_refuses_a_draft_with_the_pairing_command(tmp_path: Path) -> None:
    folder = tmp_path / "draft"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(DFLASH), encoding="utf-8")
    model = sources.resolve(sources.parse_ref(str(folder)))
    with pytest.raises(TemperumError) as exc:
        sources.refuse_draft(model, "QNN")
    assert exc.value.reason_code == "DRAFT_MODEL" and "QNN route does not pair" in str(exc.value)
    assert exc.value.hint and "--draft" in exc.value.hint and "<target model>" in exc.value.hint
    sources.refuse_draft(sources.resolve(sources.parse_ref(str(folder))).__class__(  # a plain model passes
        model.ref, None, False, None, {"architectures": ["LlamaForCausalLM"]}, None), "QNN")
