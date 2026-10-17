from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from npu_model.inspect.hf_inspector import inspect_hf_style_dir


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def phi3_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "phi3_model"
    d.mkdir()
    shutil.copy2(FIXTURES / "hf_phi3_config.json", d / "config.json")
    # Add a dummy tokenizer file
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    return d


@pytest.fixture()
def llama_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "llama_model"
    d.mkdir()
    shutil.copy2(FIXTURES / "hf_llama_config.json", d / "config.json")
    (d / "tokenizer.model").write_bytes(b"\x00")
    return d


def test_inspect_phi3(phi3_model_dir: Path) -> None:
    mi = inspect_hf_style_dir(phi3_model_dir, source={"type": "local", "path": str(phi3_model_dir)})
    assert mi.model_type == "phi3"
    assert "Phi3ForCausalLM" in mi.architectures
    assert "tokenizer.json" in mi.tokenizer_files


def test_inspect_llama(llama_model_dir: Path) -> None:
    mi = inspect_hf_style_dir(llama_model_dir, source={"type": "local", "path": str(llama_model_dir)})
    assert mi.model_type == "llama"
    assert "LlamaForCausalLM" in mi.architectures
    assert "tokenizer.model" in mi.tokenizer_files
