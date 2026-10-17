from __future__ import annotations

import json
from pathlib import Path

import pytest

from npu_model.bundle.genai_config import validate_genai_config


class TestValidateGenaiConfig:

    def test_valid_config(self, tmp_path: Path) -> None:
        cfg = {
            "model": {
                "decoder": {
                    "context_length": 4096,
                    "num_hidden_layers": 32,
                    "session_options": {"provider": "QNN"},
                }
            },
            "search": {"max_length": 2048},
        }
        (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_genai_config(tmp_path)
        assert result.errors == []

    def test_missing_file(self, tmp_path: Path) -> None:
        result = validate_genai_config(tmp_path)
        assert any("not found" in e for e in result.errors)

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "genai_config.json").write_text("{bad json", encoding="utf-8")
        result = validate_genai_config(tmp_path)
        assert any("parse" in e.lower() for e in result.errors)

    def test_warns_max_length_gt_context(self, tmp_path: Path) -> None:
        cfg = {
            "model": {"decoder": {"context_length": 512}},
            "search": {"max_length": 1024},
        }
        (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_genai_config(tmp_path)
        assert any("overflow" in w.lower() for w in result.warnings)

    def test_errors_on_negative_context(self, tmp_path: Path) -> None:
        cfg = {
            "model": {"decoder": {"context_length": -1}},
            "search": {},
        }
        (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_genai_config(tmp_path)
        assert any("positive" in e for e in result.errors)

    def test_warns_no_session_options(self, tmp_path: Path) -> None:
        cfg = {"model": {"decoder": {"context_length": 4096}}, "search": {}}
        (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_genai_config(tmp_path)
        assert any("session_options" in w for w in result.warnings)

    def test_warns_no_model_section(self, tmp_path: Path) -> None:
        cfg = {"search": {"max_length": 128}}
        (tmp_path / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_genai_config(tmp_path)
        assert any("model" in w.lower() for w in result.warnings)

    def test_works_with_file_path(self, tmp_path: Path) -> None:
        cfg = {"model": {"decoder": {}}, "search": {}}
        p = tmp_path / "genai_config.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_genai_config(p)
        assert result.errors == []


@pytest.mark.parametrize("config", [
    [], None, {"model": None}, {"model": {"decoder": []}}, {"search": []},
    {"model": {"decoder": {"session_options": []}}},
    {"model": {"decoder": {"context_length": "512"}}},
    {"model": {"decoder": {"context_length": True}}},
    {"search": {"max_length": 2.5}},
])
def test_malformed_structure_or_numeric_fields_fail_cleanly(tmp_path: Path, config: object) -> None:
    (tmp_path / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")
    result = validate_genai_config(tmp_path)
    assert result.errors
