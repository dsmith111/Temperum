"""The flow after a conversion: install, validate and push, by flag (scripts) and as commands."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from temperum.cli.main import app
from temperum.install import fork
from temperum.install import push as pushing

runner = CliRunner()


def _running(endpoint: str, exe: Any, say: Any, log_dir: Any) -> fork.ServerInfo:
    return fork.ServerInfo(endpoint, "0.0.0-npu.dev.test", started=False)


def test_validate_and_push_apply_to_the_installed_model() -> None:
    result = runner.invoke(app, ["convert", "org/model", "--validate"])
    assert result.exit_code == 2 and "INSTALL_REQUIRED" in result.output
    result = runner.invoke(app, ["convert", "org/model", "--push"])
    assert result.exit_code == 2 and "INSTALL_REQUIRED" in result.output


def test_validate_command_reports_through_the_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fork, "ensure", _running)
    monkeypatch.setattr("temperum.install.validate.validate", lambda endpoint, tag, *, fork_dir, say: {
        "tag": tag, "endpoint": endpoint, "placement": {"where": "npu", "npu_mib": 900},
        "speed": {"decode_tokens_per_s": 30.0, "prompt_tokens_per_s": 900.0, "tokens": 120},
        "cpu": {"cores": 0.1, "processes": 2}})
    result = runner.invoke(app, ["validate", "m:npu", "--host", "http://127.0.0.1:18799", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["placement"]["where"] == "npu"
    result = runner.invoke(app, ["validate", "m:npu", "--host", "http://127.0.0.1:18799"])
    assert result.exit_code == 0 and "whole on the NPU" in result.output and "0.1 cores" in result.output


def test_validate_command_starts_the_fork_when_asked_to(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    seen: dict[str, Any] = {}

    def started(endpoint: str, found: Any, say: Any, log_dir: Any) -> fork.ServerInfo:
        seen.update(exe=found, log_dir=log_dir)
        return fork.ServerInfo(endpoint, "0.0.0-npu.dev.test", started=True, pid=7, log=log_dir / "serve.log")

    monkeypatch.setattr(fork, "ensure", started)
    monkeypatch.setattr("temperum.install.validate.validate",
                        lambda endpoint, tag, *, fork_dir, say: {"tag": tag, "fork_dir": str(fork_dir)})
    result = runner.invoke(app, ["validate", "m:npu", "--ollama-exe", str(exe), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert seen["exe"] == exe and report["server"]["pid"] == 7 and report["fork_dir"] == str(tmp_path)


def test_push_command_without_signin_shows_the_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fork, "ensure", _running)

    def not_linked(endpoint: str, timeout: float = 60.0, request: Any = None) -> str:
        raise pushing.SigninRequired("https://ollama.com/connect?name=pc&key=k")

    monkeypatch.setattr(pushing, "account", not_linked)
    result = runner.invoke(app, ["push", "m:npu", "--host", "http://127.0.0.1:18799", "--yes"])
    assert result.exit_code == 2 and "SIGNIN_REQUIRED" in result.output
    assert "https://ollama.com/connect" in result.output


def test_push_command_names_and_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fork, "ensure", _running)
    monkeypatch.setattr(pushing, "account", lambda endpoint, timeout=60.0, request=None: "dsmith")
    copied: list[tuple[str, str]] = []
    monkeypatch.setattr(pushing, "copy", lambda endpoint, source, destination, timeout=120.0, request=None:
                        copied.append((source, destination)))

    def push(endpoint: str, name: str, *, progress: Any, timeout: float = 3600.0) -> dict[str, Any]:
        progress("retrieving manifest", 0, 0)
        progress("pushing abc", 50, 100)
        progress("success", 0, 0)
        return {"name": name, "status": "success", "url": "https://ollama.com/" + name.rsplit(":", 1)[0]}

    monkeypatch.setattr(pushing, "push", push)
    result = runner.invoke(app, ["push", "thing:npu", "--host", "http://127.0.0.1:18799", "--yes", "--json"])
    assert result.exit_code == 0, result.output
    assert copied == [("thing:npu", "dsmith/thing:npu")]
    assert json.loads(result.stdout)["url"] == "https://ollama.com/dsmith/thing"
    assert "retrieving manifest" in result.stderr and "dsmith/thing:npu" in result.stderr
    copied.clear()
    result = runner.invoke(app, ["push", "dsmith/thing:npu", "--as", "dsmith/thing:npu", "--host",
                                 "http://127.0.0.1:18799", "--yes"])
    assert result.exit_code == 0 and copied == []  # already named as it will be published
    result = runner.invoke(app, ["push", "thing:npu", "--as", "nonamespace", "--host", "http://127.0.0.1:18799",
                                 "--yes"])
    assert result.exit_code == 2 and "INVALID_NAME" in result.output
