"""The flows shared by the command line and the TUI, driven through a scripted session."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from rich.console import Console

from temperum import flows
from temperum.install import fork
from temperum.install import push as pushing


class FakeSession:
    """Answers questions from a script; records everything said, noted and shown."""

    def __init__(self, answers: list[Any] | None = None, interactive: bool = True) -> None:
        self.interactive = interactive
        self.answers = list(answers or [])
        self.lines: list[str] = []
        self.shown: list[Any] = []
        self.urls: list[str] = []

    def say(self, message: str) -> None:
        self.lines.append(message)

    def note(self, text: str, warning: bool = False) -> None:
        self.lines.append(("! " if warning else "") + text)

    def show(self, renderable: Any) -> None:
        self.shown.append(renderable)

    def ready(self, summary: dict[str, Any]) -> None:
        self.shown.append(summary)

    def confirm(self, question: str, default: bool = True) -> bool:
        self.lines.append("? " + question)
        answer = self.answers.pop(0)
        return bool(answer) if isinstance(answer, bool) else default

    def ask(self, prompt: str, default: str) -> str:
        self.lines.append("? " + prompt)
        answer = self.answers.pop(0)
        return str(answer) if isinstance(answer, str) else default

    def open_url(self, url: str) -> None:
        self.urls.append(url)

    @contextlib.contextmanager
    def stages(self, estimate_minutes: float, probe: Any = None, shares: Any = None) -> Iterator[Any]:
        self.lines.append(f"stages {estimate_minutes}")
        self.shares = shares
        yield self.say

    def transfer(self, label: str) -> flows.Transfer:
        self.lines.append(f"transfer {label}")
        return (lambda status, completed, total: self.lines.append(f"{status} {completed}/{total}"),
                lambda: self.lines.append("transfer stop"))


def _server(endpoint: str, exe: Any, say: Any, log_dir: Any, **_: Any) -> fork.ServerInfo:
    return fork.ServerInfo(endpoint, "0.0.0-npu.dev.test", started=False)


def test_after_convert_asks_installs_offers_the_switch_and_validates(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    exe = tmp_path / "fork" / "ollama.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(fork, "version", lambda endpoint: "0.0.0-npu.dev.test")
    monkeypatch.setattr(fork, "ensure", _server)
    monkeypatch.setattr(fork, "listener", lambda endpoint: (4242, exe))
    restarted: list[str] = []
    monkeypatch.setattr(fork, "restart_with_switch", lambda endpoint, found, say, *, log_dir, **kw:
                        restarted.append(endpoint))
    monkeypatch.setattr("temperum.install.ollama_import.npu_placement",
                        lambda endpoint, tag: {"where": "npu", "npu_mib": 900})
    monkeypatch.setattr("temperum.install.validate.validate", lambda endpoint, tag, *, fork_dir, say, **kw: {
        "placement": {"where": "npu", "npu_mib": 900}, "speed": {"decode_tokens_per_s": 30.0, "tokens": 120},
        "cpu": {"cores": 0.1, "processes": 2}})
    session = FakeSession(answers=[True, True, True, False])  # install, restart, validate, upload
    summary: dict[str, Any] = {"bundle": "C:/b", "verdict": "passed"}
    installed = {"name": "m:npu", "placement": {"where": "cpu"}, "warnings": ["ran on the CPU"]}
    flows.after_convert(session, summary, tag="m:npu", installer=lambda endpoint: dict(installed), install=False,
                        validate=False, push=False, push_as=None, exe=exe, host="http://127.0.0.1:18799")
    assert restarted == ["http://127.0.0.1:18799"]
    assert summary["install"]["placement"]["where"] == "npu" and summary["validation"]["cpu"]["cores"] == 0.1
    questions = [line for line in session.lines if line.startswith("? ")]
    assert [q[:12] for q in questions] == ["? Install it", "? That serve", "? Validate i", "? Upload it "]
    assert any("installed as m:npu; the fork runs it on the CPU" in line for line in session.lines)
    assert any(line == "! ran on the CPU" for line in session.lines)
    assert any("the fork now runs it whole on the NPU (900 MiB)" in line for line in session.lines)
    assert session.shown and "validation" not in session.shown[0].__class__.__name__.lower()


def test_after_convert_without_a_fork_executable_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession()
    flows.after_convert(session, {"bundle": "C:/b"}, tag="m:npu", installer=lambda endpoint: {}, install=False,
                        validate=False, push=False, push_as=None, exe=None, host=None)
    assert any("temperum install --input C:/b --name m:npu --execute" in line for line in session.lines)


def test_after_convert_by_flag_never_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fork, "ensure", _server)
    session = FakeSession(interactive=False)
    summary: dict[str, Any] = {"bundle": "C:/b"}
    flows.after_convert(session, summary, tag="m:npu", installer=lambda endpoint: {"name": "m:npu", "placement": {}},
                        install=True, validate=False, push=False, push_as=None, exe=None, host="http://127.0.0.1:1")
    assert summary["install"]["name"] == "m:npu" and not [line for line in session.lines if line.startswith("? ")]


def test_push_flow_signs_in_names_copies_and_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    accounts = iter([pushing.SigninRequired("https://ollama.com/connect?name=pc&key=k"), "dsmith"])

    def account(endpoint: str, timeout: float = 60.0, request: Any = None) -> str:
        outcome = next(accounts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(pushing, "account", account)
    copied: list[tuple[str, str]] = []
    monkeypatch.setattr(pushing, "copy", lambda endpoint, source, destination, timeout=120.0, request=None:
                        copied.append((source, destination)))

    def push(endpoint: str, name: str, *, progress: Any, timeout: float = 3600.0) -> dict[str, Any]:
        progress("pushing abc", 50, 100)
        return {"name": name, "status": "success", "url": "https://ollama.com/dsmith/thing"}

    monkeypatch.setattr(pushing, "push", push)
    session = FakeSession(answers=[True, "dsmith/thing:npu"])  # linked? ; upload as
    result = flows.push_flow(session, "http://127.0.0.1:1", "thing:npu", None)
    assert result["url"] == "https://ollama.com/dsmith/thing" and copied == [("thing:npu", "dsmith/thing:npu")]
    assert session.urls == ["https://ollama.com/connect?name=pc&key=k"]
    assert "transfer uploading" in session.lines and "pushing abc 50/100" in session.lines
    assert session.lines[-1].startswith("uploaded: https://ollama.com/dsmith/thing")


def test_push_flow_declined_signin_is_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_linked(endpoint: str, timeout: float = 60.0, request: Any = None) -> str:
        raise pushing.SigninRequired("https://ollama.com/connect?name=pc&key=k")

    monkeypatch.setattr(pushing, "account", not_linked)
    with pytest.raises(flows.Cancelled):
        flows.push_flow(FakeSession(answers=[False]), "http://127.0.0.1:1", "thing:npu", None)
    with pytest.raises(pushing.SigninRequired):
        flows.push_flow(FakeSession(interactive=False), "http://127.0.0.1:1", "thing:npu", None)


def test_run_gguf_uses_the_sessions_stages_and_shows_the_result(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    plan = SimpleNamespace(model=SimpleNamespace(ref=SimpleNamespace(repo="org/model"), commit="abc"),
                           estimates={"minutes": 12, "download_gb": 0.5})
    result = SimpleNamespace(bundle=tmp_path / "bundle", job_dir=tmp_path / "job", verification={
        "verdict": "warning", "reasons": ["answers not compared"],
        "npu_placement": {"layers_offloaded": 29, "layers_total": 29, "htp_weight_mib": 1841},
        "quality": {"perplexity": {"ratio": 1.06}}, "speed": {"median_tokens_per_s": 17.0}})

    def run(plan_in: Any, progress: Any, verifier: Any) -> Any:
        progress("fetching org/model")
        progress("converting to GGUF")
        return result

    monkeypatch.setattr("temperum.gguf.run", run)
    session = FakeSession()
    summary, got = flows.run_gguf(session, plan, start=0.0, verifier=object())
    assert got is result and summary["verdict"] == "warning" and summary["perplexity_ratio"] == 1.06
    assert session.lines[:3] == ["stages 12.0", "fetching org/model", "converting to GGUF"]
    assert session.shown == [summary]
    lines = flows.ready_lines(summary)
    assert lines[0].startswith("[bold yellow]warning") and "29/29 layers" in lines[2] and "1.06x" in lines[3]


def test_run_gguf_weights_the_stage_bar_for_a_recipe_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from temperum import optimize, ui

    plain = SimpleNamespace(model=SimpleNamespace(ref=SimpleNamespace(repo="org/model"), commit="abc"),
                            estimates={"minutes": 12, "download_gb": 0.5, "params_billions": 1.5}, recipe=None)
    searched = SimpleNamespace(model=plain.model, recipe="auto",
                               estimates={"minutes": 12 + optimize.estimate_minutes(1.5), "download_gb": 0.5,
                                          "params_billions": 1.5})
    result = SimpleNamespace(bundle=tmp_path / "b", job_dir=tmp_path / "j",
                             verification={"verdict": "passed", "reasons": []})
    monkeypatch.setattr("temperum.gguf.run", lambda plan_in, progress, verifier: result)
    session = FakeSession()
    flows.run_gguf(session, plain, start=0.0, verifier=object())
    assert session.shares is None
    flows.run_gguf(session, searched, start=0.0, verifier=object())
    expected = ui.shares_with_search(float(searched.estimates["minutes"]), optimize.estimate_minutes(1.5))
    assert session.shares == expected and expected[2] > 0.26 and abs(sum(expected) - 1.0) < 1e-9


def test_plain_session_prints_the_result_as_lines_and_dedups_transfer_statuses() -> None:
    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    said: list[str] = []
    session = flows.PlainSession(said.append, console)
    session.ready({"verdict": "passed", "bundle": "C:/b", "minutes": 2.0, "reasons": [], "npu_placement": {},
                   "speed": {}})
    assert "passed C:/b (2.0 min)" in console.file.getvalue()  # type: ignore[attr-defined]
    update, stop = session.transfer("uploading")
    update("pushing", 1, 2)
    update("pushing", 2, 2)
    update("success", 0, 0)
    stop()
    assert said == ["pushing", "success"]
    assert session.confirm("x?", default=False) is False and session.ask("y", "d") == "d"
    quiet = flows.PlainSession(said.append, Console(file=io.StringIO(), force_terminal=False), quiet=True)
    quiet.ready({"verdict": "passed", "bundle": "C:/b", "minutes": 2.0})
    assert quiet.console.file.getvalue() == ""  # type: ignore[attr-defined]


def test_console_session_stamps_progress_lines() -> None:
    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    session = flows.ConsoleSession(console, start=0.0)
    session.say("fetching org/model")
    session.note("careful", warning=True)
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert "fetching org/model" in text and "! careful" in text and session.interactive


def test_gguf_installer_replaces_the_tag_when_the_conversion_was_an_overwrite(monkeypatch: pytest.MonkeyPatch,
                                                                            tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr("temperum.install.ollama_import.install_gguf",
                        lambda bundle, tag, **kw: seen.append({"tag": tag, **kw}) or {"name": tag})
    result = SimpleNamespace(bundle=tmp_path / "bundle", job_dir=tmp_path / "job")
    flows.gguf_installer(FakeSession(), result, "gpt2:npu", tmp_path / "ollama.exe", overwrite=True)("http://h")
    assert seen[0]["overwrite"] is True and seen[0]["host"] == "http://h"
    flows.gguf_installer(FakeSession(), result, "gpt2:npu", tmp_path / "ollama.exe")("http://h")
    assert seen[1]["overwrite"] is False
