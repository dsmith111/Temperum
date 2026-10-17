from __future__ import annotations

import json
from pathlib import Path


from npu_model.core.tokenizer_bridge import bridge_tokenizer_files
from npu_model.core.tokenizer_norm import normalize_tokenizer_config
from npu_model.runtime_formats.ort_genai_folder import validate_ollama_ortgenai_dir


# ---------------------------------------------------------------------------
# bridge_tokenizer_files
# ---------------------------------------------------------------------------


class TestBridgeTokenizerFiles:

    def test_copies_missing_tokenizer_model(self, tmp_path: Path) -> None:
        src = tmp_path / "hf_snapshot"
        src.mkdir()
        (src / "tokenizer.model").write_bytes(b"\x00SP")

        dst = tmp_path / "export_tok"
        dst.mkdir()

        copied = bridge_tokenizer_files(src_model_dir=src, dst_tokenizer_dir=dst)
        assert "tokenizer.model" in copied
        assert (dst / "tokenizer.model").exists()

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        src = tmp_path / "hf_snapshot"
        src.mkdir()
        (src / "tokenizer.json").write_text('{"src": true}', encoding="utf-8")

        dst = tmp_path / "export_tok"
        dst.mkdir()
        (dst / "tokenizer.json").write_text('{"dst": true}', encoding="utf-8")

        copied = bridge_tokenizer_files(src_model_dir=src, dst_tokenizer_dir=dst)
        assert "tokenizer.json" not in copied
        # Existing file should be untouched
        assert json.loads((dst / "tokenizer.json").read_text())["dst"] is True

    def test_copies_multiple_files(self, tmp_path: Path) -> None:
        src = tmp_path / "hf"
        src.mkdir()
        (src / "tokenizer.model").write_bytes(b"\x00")
        (src / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (src / "special_tokens_map.json").write_text("{}", encoding="utf-8")

        dst = tmp_path / "tok"
        dst.mkdir()

        copied = bridge_tokenizer_files(src_model_dir=src, dst_tokenizer_dir=dst)
        assert set(copied) == {"tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"}

    def test_creates_dst_dir(self, tmp_path: Path) -> None:
        src = tmp_path / "hf"
        src.mkdir()
        (src / "tokenizer.model").write_bytes(b"\x00")

        dst = tmp_path / "new_tok_dir"
        copied = bridge_tokenizer_files(src_model_dir=src, dst_tokenizer_dir=dst)
        assert "tokenizer.model" in copied
        assert dst.is_dir()

    def test_returns_empty_when_nothing_to_copy(self, tmp_path: Path) -> None:
        src = tmp_path / "hf"
        src.mkdir()

        dst = tmp_path / "tok"
        dst.mkdir()

        copied = bridge_tokenizer_files(src_model_dir=src, dst_tokenizer_dir=dst)
        assert copied == []


# ---------------------------------------------------------------------------
# Bridge + normalize end-to-end
# ---------------------------------------------------------------------------


class TestBridgeThenNormalize:

    def test_bridge_enables_normalization(self, tmp_path: Path) -> None:
        """TokenizersBackend + tokenizer.model in source → bridge + normalize succeeds."""
        src = tmp_path / "hf_snapshot"
        src.mkdir()
        (src / "tokenizer.model").write_bytes(b"\x00SP")

        tok_dir = tmp_path / "exported_tok"
        tok_dir.mkdir()
        cfg = {"tokenizer_class": "TokenizersBackend"}
        (tok_dir / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")

        # Without bridge: normalize can't fix it
        result_before = normalize_tokenizer_config(tok_dir, model_type="phi3")
        assert result_before.changed is False

        # Bridge first
        copied = bridge_tokenizer_files(src_model_dir=src, dst_tokenizer_dir=tok_dir)
        assert "tokenizer.model" in copied

        # Now normalize succeeds
        result_after = normalize_tokenizer_config(tok_dir, model_type="phi3")
        assert result_after.changed is True
        assert result_after.new_class == "LlamaTokenizer"


# ---------------------------------------------------------------------------
# normalize_tokenizer_config
# ---------------------------------------------------------------------------


class TestNormalizeTokenizerConfig:

    def test_no_config_returns_unchanged(self, tmp_path: Path) -> None:
        result = normalize_tokenizer_config(tmp_path)
        assert result.changed is False
        assert result.warnings == []

    def test_supported_class_unchanged(self, tmp_path: Path) -> None:
        cfg = {"tokenizer_class": "LlamaTokenizer"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = normalize_tokenizer_config(tmp_path)
        assert result.changed is False
        assert result.original_class == "LlamaTokenizer"

    def test_tokenizers_backend_with_sp(self, tmp_path: Path) -> None:
        """TokenizersBackend + tokenizer.model → rewrite to LlamaTokenizer."""
        cfg = {"tokenizer_class": "TokenizersBackend", "other": "keep"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (tmp_path / "tokenizer.model").write_bytes(b"\x00")

        result = normalize_tokenizer_config(tmp_path, model_type="phi3")
        assert result.changed is True
        assert result.original_class == "TokenizersBackend"
        assert result.new_class == "LlamaTokenizer"
        assert result.warnings == []

        # Verify file was actually rewritten
        rewritten = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert rewritten["tokenizer_class"] == "LlamaTokenizer"
        assert rewritten["other"] == "keep"  # other fields preserved

    def test_fast_class_with_sp(self, tmp_path: Path) -> None:
        """LlamaTokenizerFast + tokenizer.model → rewrite to LlamaTokenizer."""
        cfg = {"tokenizer_class": "LlamaTokenizerFast"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (tmp_path / "tokenizer.model").write_bytes(b"\x00")

        result = normalize_tokenizer_config(tmp_path, model_type="llama")
        assert result.changed is True
        assert result.original_class == "LlamaTokenizerFast"
        assert result.new_class == "LlamaTokenizer"

    def test_tokenizers_backend_json_only(self, tmp_path: Path) -> None:
        """TokenizersBackend + tokenizer.json (no .model) → rewrite to BPE class."""
        cfg = {"tokenizer_class": "TokenizersBackend"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

        result = normalize_tokenizer_config(tmp_path, model_type="phi")
        assert result.changed is True
        assert result.new_class == "CodeGenTokenizer"

    def test_tokenizers_backend_no_tokenizer_files(self, tmp_path: Path) -> None:
        """TokenizersBackend but no tokenizer files at all → not changed, warning."""
        cfg = {"tokenizer_class": "TokenizersBackend"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")

        result = normalize_tokenizer_config(tmp_path)
        assert result.changed is False
        assert len(result.warnings) == 1
        assert "fallback" in result.warnings[0].lower()

    def test_model_type_selects_fallback(self, tmp_path: Path) -> None:
        cfg = {"tokenizer_class": "TokenizersBackend"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (tmp_path / "tokenizer.model").write_bytes(b"\x00")

        # gemma → GemmaTokenizer
        result = normalize_tokenizer_config(tmp_path, model_type="gemma")
        assert result.new_class == "GemmaTokenizer"

    def test_default_fallback_for_unknown_type(self, tmp_path: Path) -> None:
        cfg = {"tokenizer_class": "TokenizersBackend"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (tmp_path / "tokenizer.model").write_bytes(b"\x00")

        result = normalize_tokenizer_config(tmp_path, model_type="unknown_model")
        assert result.new_class == "LlamaTokenizer"  # default fallback

    def test_invalid_json_warns(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer_config.json").write_text("{bad json", encoding="utf-8")
        result = normalize_tokenizer_config(tmp_path)
        assert result.changed is False
        assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# validate_ollama_ortgenai_dir — tokenizer class checks
# ---------------------------------------------------------------------------


class TestValidateTokenizerClass:

    def _make_valid_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "bundle"
        d.mkdir()
        (d / "model.onnx").write_bytes(b"\x00")
        (d / "genai_config.json").write_text("{}", encoding="utf-8")
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
        return d

    def test_tokenizers_backend_is_error(self, tmp_path: Path) -> None:
        d = self._make_valid_dir(tmp_path)
        cfg = {"tokenizer_class": "TokenizersBackend"}
        (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_ollama_ortgenai_dir(d)
        assert any("TokenizersBackend" in e for e in result.errors)

    def test_fast_class_is_warning(self, tmp_path: Path) -> None:
        d = self._make_valid_dir(tmp_path)
        cfg = {"tokenizer_class": "LlamaTokenizerFast"}
        (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_ollama_ortgenai_dir(d)
        assert result.errors == [] or not any("LlamaTokenizerFast" in e for e in result.errors)
        assert any("Fast" in w for w in result.warnings)

    def test_supported_class_no_error(self, tmp_path: Path) -> None:
        d = self._make_valid_dir(tmp_path)
        cfg = {"tokenizer_class": "LlamaTokenizer"}
        (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        result = validate_ollama_ortgenai_dir(d)
        tok_errors = [e for e in result.errors if "tokenizer_class" in e.lower()]
        assert tok_errors == []

    def test_normalized_dir_passes(self, tmp_path: Path) -> None:
        """After normalization, the directory should pass validation."""
        d = self._make_valid_dir(tmp_path)
        cfg = {"tokenizer_class": "TokenizersBackend"}
        (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (d / "tokenizer.model").write_bytes(b"\x00")

        # Normalize
        norm = normalize_tokenizer_config(d, model_type="phi3")
        assert norm.changed is True

        # Now validate
        result = validate_ollama_ortgenai_dir(d)
        tok_errors = [e for e in result.errors if "tokenizer_class" in e.lower() or "TokenizersBackend" in e]
        assert tok_errors == []
