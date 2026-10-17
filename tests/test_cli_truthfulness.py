from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from npu_model.cli import main
from npu_model.core.types import ConvertResult


runner = CliRunner()


def test_unknown_validation_mode_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"not-a-model")
    result = runner.invoke(main.app, ["validate", "--input", str(tmp_path), "--as", "typo"])
    assert result.exit_code == 2
    assert "UNKNOWN_VALIDATION_MODE" in result.output
    assert "passed" not in result.output.lower()


def test_validation_requires_runtime_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "registry", SimpleNamespace(runtime_formats={}))
    result = runner.invoke(main.app, ["validate", "--input", str(tmp_path)])
    assert result.exit_code == 2
    assert "MISSING_RUNTIME_PLUGIN" in result.output


def test_layout_requires_onnx_file_not_named_directory(tmp_path: Path) -> None:
    (tmp_path / "onnx" / "model.onnx").mkdir(parents=True)
    result = runner.invoke(main.app, ["validate", "--input", str(tmp_path)])
    assert result.exit_code == 2
    assert "NO_ONNX_FILES" in result.output


def test_deprecated_bundle_alias_still_works(tmp_path: Path) -> None:
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"layout-only")
    result = runner.invoke(main.app, ["validate", "--bundle", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.output.lower()
    assert "layout" in result.output.lower()


def test_conflicting_validation_paths_rejected(tmp_path: Path) -> None:
    result = runner.invoke(
        main.app, ["validate", "--input", str(tmp_path), "--bundle", str(tmp_path / "other")]
    )
    assert result.exit_code == 2
    assert "CONFLICTING_INPUT" in result.output


def test_runtime_validation_missing_dependency_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)
    result = runner.invoke(main.app, ["validate", "--input", str(tmp_path), "--as", "runtime-load"])
    assert result.exit_code == 2
    assert "MISSING_VALIDATE_DEPS" in result.output
    assert "succeeded" not in result.output


def test_runtime_validation_generates_with_prompt_plus_token_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    og = MagicMock()
    og.Tokenizer.return_value.encode.return_value = [1, 2, 3]
    og.Generator.return_value.is_done.side_effect = [False, True]
    og.Generator.return_value.get_sequence.return_value = [1, 2, 3, 4]
    og.Tokenizer.return_value.decode.return_value = "Hello!"
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", og)
    result = runner.invoke(
        main.app,
        ["validate", "--input", str(tmp_path), "--as", "runtime-load", "--max-tokens", "1"],
    )
    assert result.exit_code == 0, result.output
    og.GeneratorParams.return_value.set_search_options.assert_called_once_with(max_length=4)
    og.Generator.return_value.append_tokens.assert_called_once_with([1, 2, 3])
    og.Generator.return_value.generate_next_token.assert_called_once()
    assert "not NPU certification" in result.output


def test_runtime_validation_no_generation_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    og = MagicMock()
    og.Tokenizer.return_value.encode.return_value = [1, 2]
    og.Generator.return_value.is_done.return_value = True
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", og)
    result = runner.invoke(main.app, ["validate", "--input", str(tmp_path), "--as", "runtime-load"])
    assert result.exit_code == 2
    assert "NO_GENERATION" in result.output
    assert "succeeded" not in result.output


@pytest.mark.parametrize("command", ["publish", "convert", "pack-ollama"])
def test_assembly_commands_do_not_claim_remote_publication(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assembled = tmp_path / "assembled"
    monkeypatch.setattr(
        main, "convert_model",
        lambda **kwargs: ConvertResult(tmp_path, tmp_path / "manifest.json", assembled),
    )
    from npu_model.packagers import ollama

    monkeypatch.setattr(
        ollama, "pack_for_ollama", lambda **kwargs: SimpleNamespace(pack_dir=assembled, file_count=4)
    )
    args = [command, "--input", str(tmp_path), "--out", str(assembled)]
    args += ["--pack-ollama", "test/model:c0"] if command == "convert" else ["--name", "test/model:c0"]
    result = runner.invoke(main.app, args)
    assert result.exit_code == 0, result.output
    assert "assembled" in result.output.lower()
    assert "no Ollama install or remote upload" in result.output
    assert "Ready to publish" not in result.output
    assert "ollama push" not in result.output


def test_strict_npu_cli_cannot_pass_without_optional_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model_ctx.onnx").write_bytes(b"not-a-real-model")
    (tmp_path / "model_qnn.bin").write_bytes(b"not-a-real-context")
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"session_options": {
            "provider_options": [{"qnn": {}}], "disable_cpu_ep_fallback": "1",
        }}}}), encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    result = runner.invoke(main.app, ["validate", "--input", str(tmp_path), "--as", "strict-npu"])
    assert result.exit_code == 2
    assert "SKIP" in result.output
    assert "VALIDATION PASSED" not in result.output
