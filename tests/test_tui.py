"""The full-screen interface: commands in the prompt, questions answered there, the stage bar."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest

from temperum import flows
from temperum.core.errors import TemperumError
from temperum.tui.app import TemperumApp, TuiSession, _options

pytestmark = pytest.mark.usefixtures("_own_home")


@pytest.fixture
def _own_home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMPERUM_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TEMPERUM_OLLAMA_EXE", raising=False)


def _run(test: Callable[[TemperumApp, Any], Awaitable[None]]) -> None:
    async def go() -> None:
        app = TemperumApp()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await test(app, pilot)
    asyncio.run(go())


async def _until(pilot: Any, condition: Callable[[], bool], seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition not met in time")


def test_options_parse_flags_values_and_quoted_paths() -> None:
    positional, options = _options(['"C:\\my models\\b"', "--name", "x:npu", "--overwrite"], {"--overwrite"},
                                   {"--name"})
    assert positional == ["C:\\my models\\b"] and options == {"--name": "x:npu", "--overwrite": "1"}
    with pytest.raises(Exception, match="needs a value"):
        _options(["--name"], set(), {"--name"})
    with pytest.raises(Exception, match="unknown option"):
        _options(["--bogus"], set(), set())


def test_help_and_unknown_commands() -> None:
    async def test(app: TemperumApp, pilot: Any) -> None:
        await pilot.press(*"/help", "enter")
        await pilot.pause()
        assert any("/convert MODEL" in line and "/push TAG" in line for line in app.history)
        await pilot.press(*"/bogus", "enter")
        await pilot.pause()
        assert any("unknown command /bogus" in line for line in app.history)
    _run(test)


def test_a_job_asks_in_the_prompt_and_drives_the_stage_bar() -> None:
    seen: dict[str, Any] = {}

    def job(session: TuiSession) -> None:
        with session.stages(10) as say:
            say("fetching org/model")
            seen["confirm"] = session.confirm("Proceed?")
            seen["again"] = session.confirm("Sure?", default=False)
            seen["name"] = session.ask("Upload as", default="me/thing:npu")
        update, stop = session.transfer("uploading")
        update("pushing abc", 50, 100)
        stop()
        session.note("all done")

    async def test(app: TemperumApp, pilot: Any) -> None:
        app.run_job("testing", job)
        await _until(pilot, lambda: app.pending is not None)
        assert app.busy == "testing" and app._tracker is not None and app._tracker.stage_name == "downloading"
        assert app.query_one("#stage").display
        await pilot.press(*"maybe", "enter")  # not an answer: asked again
        await _until(pilot, lambda: any("answer y or n" in line for line in app.history))
        await _until(pilot, lambda: app.pending is not None)
        await pilot.press("enter")  # the default: yes
        await _until(pilot, lambda: app.pending is not None and app.pending.text == "Sure?")
        await pilot.press(*"n", "enter")
        await _until(pilot, lambda: app.pending is not None and app.pending.kind == "ask")
        await pilot.press("enter")  # the default name
        await _until(pilot, lambda: app.busy is None)
        assert seen == {"confirm": True, "again": False, "name": "me/thing:npu"}
        assert any("all done" in line for line in app.history) and app._tracker is None
        assert any("? Proceed?" in line for line in app.history)
    _run(test)


def test_a_bare_model_id_converts_through_the_shared_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    plan = SimpleNamespace(model=SimpleNamespace(name="model"), warnings=["be careful"], estimates={"minutes": 1})
    monkeypatch.setattr(flows, "plan_gguf", lambda model, **kw: calls.append(("plan", model, kw["quant"])) or (plan, None))
    monkeypatch.setattr(flows, "gguf_plan_table", lambda plan_in: "the plan")
    monkeypatch.setattr(flows, "run_gguf", lambda session, plan_in, *, start, verifier=None:
                        calls.append("run") or ({"verdict": "passed", "bundle": "C:/b"}, object()))
    monkeypatch.setattr(flows, "gguf_installer", lambda session, result, tag, exe: (lambda endpoint: {}))
    monkeypatch.setattr(flows, "after_convert", lambda session, summary, **kw: calls.append(("after", kw["tag"])))

    async def test(app: TemperumApp, pilot: Any) -> None:
        await pilot.press(*"org/model --target gguf-hexagon --quant Q8_0", "enter")
        await _until(pilot, lambda: app.pending is not None)
        assert app.pending is not None and app.pending.text == "Proceed?"
        assert any("the plan" in line for line in app.history) and any("! be careful" in line for line in app.history)
        await pilot.press("enter")
        await _until(pilot, lambda: app.busy is None)
        assert calls == [("plan", "org/model", "Q8_0"), "run", ("after", "model:npu")]
        await pilot.press(*"org/model --target gguf-hexagon", "enter")
        await _until(pilot, lambda: app.pending is not None)
        await pilot.press(*"n", "enter")
        await _until(pilot, lambda: app.busy is None)
        assert any(line.strip() == "cancelled" for line in app.history)
    _run(test)


def test_errors_in_a_job_are_shown_not_fatal() -> None:
    def job(session: TuiSession) -> None:
        raise TemperumError("import", "EXE_REQUIRED", "Installing needs the fork executable.", hint="set it")

    async def test(app: TemperumApp, pilot: Any) -> None:
        app.run_job("failing", job)
        await _until(pilot, lambda: app.busy is None)
        assert any("[import/EXE_REQUIRED]" in line and "Installing needs" in line for line in app.history)
        assert any("Hint: set it" in line for line in app.history)
        await pilot.press(*"/config", "enter")
        await _until(pilot, lambda: app.busy is None and any("ollama-exe" in line for line in app.history))
    _run(test)


def test_leaving_abandons_a_pending_question() -> None:
    outcome: dict[str, Any] = {}

    def job(session: TuiSession) -> None:
        try:
            session.confirm("Install?")
        except flows.Cancelled:
            outcome["cancelled"] = True

    async def test(app: TemperumApp, pilot: Any) -> None:
        app.run_job("asking", job)
        await _until(pilot, lambda: app.pending is not None)
        app.action_leave(force=True)
        await pilot.pause()
    _run(test)
    assert outcome == {"cancelled": True}
