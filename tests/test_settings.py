"""Stored settings: the fork's executable, the default target and the endpoint, and their precedence."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from temperum import settings
from temperum.cli.main import _fork_status, app
from temperum.core.errors import TemperumError
from temperum.install import fork

runner = CliRunner()


@pytest.fixture(autouse=True)
def _own_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMPERUM_HOME", str(tmp_path / "home"))
    for name in ("TEMPERUM_OLLAMA_EXE", "TEMPERUM_TARGET", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)


def test_precedence_is_option_then_environment_then_file(monkeypatch: pytest.MonkeyPatch) -> None:
    assert settings.resolve("ollama-exe") == settings.Setting(None, "default")
    settings.set_value("ollama-exe", "C:/stored/ollama.exe")
    assert settings.resolve("ollama-exe") == settings.Setting("C:/stored/ollama.exe", "config")
    monkeypatch.setenv("TEMPERUM_OLLAMA_EXE", "C:/env/ollama.exe")
    assert settings.resolve("ollama-exe") == settings.Setting("C:/env/ollama.exe", "TEMPERUM_OLLAMA_EXE")
    assert settings.resolve("ollama-exe", "C:/opt/ollama.exe") == settings.Setting("C:/opt/ollama.exe", "option")
    monkeypatch.delenv("TEMPERUM_OLLAMA_EXE")
    settings.unset("ollama-exe")
    assert settings.load() == {} and settings.resolve("target", None, "auto") == settings.Setting("auto", "default")
    with pytest.raises(TemperumError):
        settings.set_value("colour", "blue")


def test_an_unreadable_file_is_reported() -> None:
    settings.path().parent.mkdir(parents=True)
    settings.path().write_text("{not json", encoding="utf-8")
    with pytest.raises(TemperumError) as exc:
        settings.load()
    assert exc.value.reason_code == "CONFIG_UNREADABLE"


def test_config_command_stores_shows_and_forgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    result = runner.invoke(app, ["config", "ollama-exe", str(exe)])
    assert result.exit_code == 0, result.output
    assert settings.load()["ollama-exe"] == str(exe.resolve())
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0 and "ollama-exe" in result.stdout and "config" in result.stdout
    result = runner.invoke(app, ["config", "ollama-exe"])
    assert result.exit_code == 0 and result.stdout.strip() == str(exe.resolve())
    result = runner.invoke(app, ["config", "ollama-exe", str(tmp_path / "missing.exe")])
    assert result.exit_code == 2 and "EXE_MISSING" in result.output
    monkeypatch.setenv("TEMPERUM_OLLAMA_EXE", "C:/stale/ollama.exe")  # a setx value an old terminal still carries
    result = runner.invoke(app, ["config", "ollama-exe", str(exe)])
    assert result.exit_code == 0 and "takes precedence" in result.output
    monkeypatch.delenv("TEMPERUM_OLLAMA_EXE")
    result = runner.invoke(app, ["config", "host", "127.0.0.1:18767"])
    assert result.exit_code == 0 and settings.load()["host"] == "http://127.0.0.1:18767"
    assert fork.endpoint_for(None) == "http://127.0.0.1:18767"
    result = runner.invoke(app, ["config", "target", "qnn-x-elite"])
    assert result.exit_code == 0 and settings.load()["target"] == "qnn-x-elite"
    result = runner.invoke(app, ["config", "target", "nonsense"])
    assert result.exit_code == 2
    result = runner.invoke(app, ["config", "host", "--unset"])
    assert result.exit_code == 0 and "host" not in settings.load()
    result = runner.invoke(app, ["config", "nonsense", "x"])
    assert result.exit_code == 2 and "UNKNOWN_SETTING" in result.output


def test_the_stored_fork_is_used_everywhere(tmp_path: Path) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    settings.set_value("ollama-exe", str(exe))
    assert fork.executable(None) == exe
    status = _fork_status()
    assert status["exe"] == str(exe) and status["source"] == "config" and status["state"] != "not set"


def test_setup_remembers_the_fork_and_shows_it(tmp_path: Path) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    result = runner.invoke(app, ["setup", "--check", "--ollama-exe", str(exe)])
    assert result.exit_code == 0, result.output
    assert settings.load()["ollama-exe"] == str(exe.resolve())
    assert "remembered" in result.output and "fork" in result.stdout
    result = runner.invoke(app, ["setup", "--check", "--ollama-exe", str(tmp_path / "missing.exe")])
    assert result.exit_code == 2 and "EXE_MISSING" in result.output
