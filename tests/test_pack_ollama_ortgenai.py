from __future__ import annotations

from pathlib import Path

import pytest

from npu_model.packagers.ollama import pack_for_ollama
from npu_model.runtime_formats.ort_genai_folder import (
    collect_ollama_files,
    validate_ollama_ortgenai_dir,
)


@pytest.fixture()
def minimal_ortgenai_dir(tmp_path: Path) -> Path:
    """Create a minimal valid compiled ORT GenAI directory."""
    d = tmp_path / "ortgenai_src"
    d.mkdir()
    (d / "decoder_ctx.onnx").write_bytes(b"\x00" * 16)
    (d / "decoder_qnn.bin").write_bytes(b"\x00" * 2048)
    (d / "genai_config.json").write_text('{"model": {}}', encoding="utf-8")
    (d / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# collect_ollama_files
# ---------------------------------------------------------------------------


class TestCollectOllamaFiles:
    def test_picks_up_onnx_json_tokenizer(self, minimal_ortgenai_dir: Path) -> None:
        files = collect_ollama_files(minimal_ortgenai_dir)
        names = {f.name for f in files}
        assert "decoder_ctx.onnx" in names
        assert "genai_config.json" in names
        assert "tokenizer.json" in names

    def test_picks_up_bin_and_jinja(self, minimal_ortgenai_dir: Path) -> None:
        (minimal_ortgenai_dir / "weights.bin").write_bytes(b"\x00")
        (minimal_ortgenai_dir / "chat_template.jinja").write_text("hi", encoding="utf-8")
        files = collect_ollama_files(minimal_ortgenai_dir)
        names = {f.name for f in files}
        assert "weights.bin" in names
        assert "chat_template.jinja" in names

    def test_picks_up_tokenizer_model(self, minimal_ortgenai_dir: Path) -> None:
        (minimal_ortgenai_dir / "tokenizer.model").write_bytes(b"\x00")
        files = collect_ollama_files(minimal_ortgenai_dir)
        names = {f.name for f in files}
        assert "tokenizer.model" in names

    def test_excludes_gguf(self, minimal_ortgenai_dir: Path) -> None:
        (minimal_ortgenai_dir / "model.gguf").write_bytes(b"\x00")
        files = collect_ollama_files(minimal_ortgenai_dir)
        names = {f.name for f in files}
        assert "model.gguf" not in names

    def test_excludes_safetensors(self, minimal_ortgenai_dir: Path) -> None:
        (minimal_ortgenai_dir / "model.safetensors").write_bytes(b"\x00")
        files = collect_ollama_files(minimal_ortgenai_dir)
        names = {f.name for f in files}
        assert "model.safetensors" not in names

    def test_recurses_subdirs(self, tmp_path: Path) -> None:
        root = tmp_path / "nested"
        sub = root / "onnx"
        sub.mkdir(parents=True)
        (sub / "decoder.onnx").write_bytes(b"\x00")
        (root / "genai_config.json").write_text("{}", encoding="utf-8")
        files = collect_ollama_files(root)
        names = {f.name for f in files}
        assert "decoder.onnx" in names
        assert "genai_config.json" in names


# ---------------------------------------------------------------------------
# validate_ollama_ortgenai_dir
# ---------------------------------------------------------------------------


class TestValidateOllamaOrtgenai:
    def test_valid_dir_no_errors(self, minimal_ortgenai_dir: Path) -> None:
        result = validate_ollama_ortgenai_dir(minimal_ortgenai_dir)
        assert result.errors == []

    def test_missing_onnx(self, tmp_path: Path) -> None:
        d = tmp_path / "no_onnx"
        d.mkdir()
        (d / "genai_config.json").write_text("{}", encoding="utf-8")
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
        result = validate_ollama_ortgenai_dir(d)
        assert any("onnx" in e.lower() for e in result.errors)

    def test_missing_genai_config(self, tmp_path: Path) -> None:
        d = tmp_path / "no_config"
        d.mkdir()
        (d / "model.onnx").write_bytes(b"\x00")
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
        result = validate_ollama_ortgenai_dir(d)
        assert any("genai_config.json" in e for e in result.errors)

    def test_missing_tokenizer(self, tmp_path: Path) -> None:
        d = tmp_path / "no_tok"
        d.mkdir()
        (d / "model.onnx").write_bytes(b"\x00")
        (d / "genai_config.json").write_text("{}", encoding="utf-8")
        result = validate_ollama_ortgenai_dir(d)
        assert any("tokenizer" in e.lower() for e in result.errors)

    def test_gguf_causes_error(self, minimal_ortgenai_dir: Path) -> None:
        (minimal_ortgenai_dir / "bad.gguf").write_bytes(b"\x00")
        result = validate_ollama_ortgenai_dir(minimal_ortgenai_dir)
        assert any(".gguf" in e for e in result.errors)

    def test_safetensors_causes_error(self, minimal_ortgenai_dir: Path) -> None:
        (minimal_ortgenai_dir / "bad.safetensors").write_bytes(b"\x00")
        result = validate_ollama_ortgenai_dir(minimal_ortgenai_dir)
        assert any(".safetensors" in e for e in result.errors)

    def test_missing_bin_is_error(self, minimal_ortgenai_dir: Path) -> None:
        (minimal_ortgenai_dir / "decoder_qnn.bin").unlink()
        result = validate_ollama_ortgenai_dir(minimal_ortgenai_dir)
        assert any("bin" in e.lower() for e in result.errors)

    def test_non_ctx_onnx_is_error(self, minimal_ortgenai_dir: Path) -> None:
        (minimal_ortgenai_dir / "model.onnx").write_bytes(b"\x00")
        result = validate_ollama_ortgenai_dir(minimal_ortgenai_dir)
        assert any("non-ctx" in e.lower() for e in result.errors)

    def test_warns_no_chat_template(self, minimal_ortgenai_dir: Path) -> None:
        result = validate_ollama_ortgenai_dir(minimal_ortgenai_dir)
        assert any("chat_template.jinja" in w for w in result.warnings)

    def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        result = validate_ollama_ortgenai_dir(f)
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# pack_for_ollama (end-to-end)
# ---------------------------------------------------------------------------


class TestPackForOllama:
    def test_produces_flat_output(self, minimal_ortgenai_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "publish"
        result = pack_for_ollama(
            bundle_dir=minimal_ortgenai_dir,
            model_name="test/model:tag",
            out_dir=out,
        )
        assert result.pack_dir == out
        assert result.pack_dir.exists()
        names = {p.name for p in out.iterdir() if p.is_file()}
        assert "Modelfile" in names
        assert "decoder_ctx.onnx" in names
        assert "decoder_qnn.bin" in names
        assert "genai_config.json" in names
        assert "tokenizer.json" in names
        assert "_publish_manifest.json" in names
        # No subdirectories in output (flat)
        subdirs = [p for p in out.iterdir() if p.is_dir()]
        assert subdirs == []

    def test_modelfile_content(self, minimal_ortgenai_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "publish"
        pack_for_ollama(
            bundle_dir=minimal_ortgenai_dir,
            model_name="test/m:t",
            out_dir=out,
            num_ctx=1024,
            num_predict=256,
        )
        mf = (out / "Modelfile").read_text(encoding="utf-8")
        assert "FROM ." in mf
        assert "num_ctx 1024" in mf
        assert "num_predict 256" in mf

    def test_excludes_gguf_from_output(self, minimal_ortgenai_dir: Path, tmp_path: Path) -> None:
        (minimal_ortgenai_dir / "bad.gguf").write_bytes(b"\x00")
        out = tmp_path / "publish"
        # gguf is excluded from collect, but the input dir has it —
        # the staged output should not contain it and should fail validation
        # because gguf is excluded from collection (never copied).
        # Actually, since collect excludes it, it won't be in staging,
        # so validation should pass (no gguf in output).
        result = pack_for_ollama(
            bundle_dir=minimal_ortgenai_dir,
            model_name="test/m:t",
            out_dir=out,
        )
        assert result.pack_dir == out.resolve()
        names = {p.name for p in out.iterdir()}
        assert "bad.gguf" not in names

    def test_rejects_non_ctx_input(self, tmp_path: Path) -> None:
        from npu_model.core.errors import NpuModelError

        src = tmp_path / "src"
        src.mkdir()
        (src / "model.onnx").write_bytes(b"\x00")
        (src / "decoder_qnn.bin").write_bytes(b"\x00" * 2048)
        (src / "genai_config.json").write_text("{}", encoding="utf-8")
        (src / "tokenizer.json").write_text("{}", encoding="utf-8")
        with pytest.raises(NpuModelError) as exc_info:
            pack_for_ollama(bundle_dir=src, model_name="x/y:z", out_dir=tmp_path / "out")
        assert exc_info.value.reason_code == "NON_CTX_ONNX_NOT_ALLOWED"

    def test_custom_params(self, minimal_ortgenai_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "pub"
        pack_for_ollama(
            bundle_dir=minimal_ortgenai_dir,
            model_name="x/y:z",
            out_dir=out,
            num_ctx=2048,
            num_predict=64,
        )
        mf = (out / "Modelfile").read_text(encoding="utf-8")
        assert "num_ctx 2048" in mf
        assert "num_predict 64" in mf

    def test_file_count(self, minimal_ortgenai_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "pub"
        result = pack_for_ollama(
            bundle_dir=minimal_ortgenai_dir,
            model_name="x/y:z",
            out_dir=out,
        )
        actual = len([p for p in out.iterdir() if p.is_file()])
        assert result.file_count == actual

    def test_validate_passes_on_output(self, minimal_ortgenai_dir: Path, tmp_path: Path) -> None:
        """pack-ollama output should pass ollama-ortgenai validation."""
        out = tmp_path / "pub"
        pack_for_ollama(
            bundle_dir=minimal_ortgenai_dir,
            model_name="x/y:z",
            out_dir=out,
        )
        result = validate_ollama_ortgenai_dir(out)
        assert result.errors == []

    def test_validate_fails_with_gguf_injected(self, minimal_ortgenai_dir: Path, tmp_path: Path) -> None:
        """If a gguf file is added to the output dir, validation must fail."""
        out = tmp_path / "pub"
        pack_for_ollama(
            bundle_dir=minimal_ortgenai_dir,
            model_name="x/y:z",
            out_dir=out,
        )
        # Inject a gguf into the output
        (out / "model.gguf").write_bytes(b"\x00")
        result = validate_ollama_ortgenai_dir(out)
        assert any(".gguf" in e for e in result.errors)

    def test_atomic_staging_cleans_up(self, tmp_path: Path) -> None:
        """If input is missing required files, staging dir should not remain."""
        empty = tmp_path / "empty_src"
        empty.mkdir()
        out = tmp_path / "out"
        from npu_model.core.errors import NpuModelError

        with pytest.raises(NpuModelError):
            pack_for_ollama(bundle_dir=empty, model_name="x/y:z", out_dir=out)
        # Staging dir should have been cleaned up
        staging_dirs = list(tmp_path.glob("*.staging.*"))
        assert staging_dirs == []

    def test_replaces_existing_output(self, minimal_ortgenai_dir: Path, tmp_path: Path) -> None:
        """Running pack twice into the same output should work (atomic replace)."""
        out = tmp_path / "pub"
        pack_for_ollama(bundle_dir=minimal_ortgenai_dir, model_name="a/b:c", out_dir=out)
        # Add an extra file that shouldn't survive a re-pack
        (out / "stale.txt").write_text("old", encoding="utf-8")
        pack_for_ollama(bundle_dir=minimal_ortgenai_dir, model_name="a/b:c", out_dir=out)
        assert not (out / "stale.txt").exists()
