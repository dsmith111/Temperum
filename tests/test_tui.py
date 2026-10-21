"""The full-screen interface: home, model, work and done screens; questions as buttons; copying."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest
from rich.table import Table

from temperum import flows
from temperum.core.errors import TemperumError
from temperum.tui import app as tui
from temperum.tui.app import DoneScreen, HomeScreen, ModelScreen, TemperumApp, TuiSession, WorkScreen, _options
from temperum.tui.widgets import ChoiceBar, Gauge, ResourceGraph, StageList
from textual.widgets import Input, OptionList

pytestmark = pytest.mark.usefixtures("_own_home")


@pytest.fixture
def _own_home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMPERUM_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TEMPERUM_OLLAMA_EXE", raising=False)
    # the home screen looks at this machine in a worker; keep the tests quick and deterministic
    monkeypatch.setattr(TemperumApp, "_machine_job", lambda self: None)


def _run(test: Callable[[TemperumApp, Any], Awaitable[None]]) -> None:
    async def go() -> None:
        app = TemperumApp(monitor=None)
        async with app.run_test(size=(110, 36)) as pilot:
            await pilot.pause()
            await test(app, pilot)
    asyncio.run(go())


async def _until(pilot: Any, condition: Callable[[], bool], seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition not met in time")


def _fake_plan(quant: str, params: float = 0.49) -> Any:
    model = SimpleNamespace(name="qwen2.5-0.5b-instruct", license="apache-2.0", architecture="Qwen2ForCausalLM",
                            config={"max_position_embeddings": 32768}, gated=False,
                            ref=SimpleNamespace(repo="Qwen/Qwen2.5-0.5B-Instruct", path=None,
                                                display="Qwen/Qwen2.5-0.5B-Instruct"),
                            chat_template="{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
                                          "<|im_end|>{% endfor %}")
    return SimpleNamespace(quant=quant, model=model, context=4096, warnings=["be careful"],
                           destination=Path("C:/nowhere/bundle"),
                           estimates={"gguf_gb": {"Q4_0": 0.3, "Q8_0": 0.6, "Q4_K_M": 0.35}[quant], "ram_gb": 2.0,
                                      "minutes": 3, "params_billions": params, "download_gb": 1.0, "disk_gb": 2.5})


def _fake_plan_job(self: TemperumApp, screen: ModelScreen, model: str, opts: dict[str, str]) -> None:
    plans = {quant: _fake_plan(quant) for quant in ("Q4_0", "Q8_0", "Q4_K_M")}
    machine = SimpleNamespace(total_ram_gb=15.4, available_ram_gb=6.0)
    recommended, why = tui.recommend(plans)
    self.call_from_thread(screen.show, plans["Q4_0"].model, plans, opts.get("--quant") or recommended, recommended,
                          why, None, machine)


def _fake_run(session: Any, plan: Any, *, start: float, verifier: Any = None) -> tuple[dict[str, Any], Any]:
    with session.stages(1.0) as say:
        say("fetching the model")
        say("converting to GGUF (Q8_0)")
        say("assembling the bundle")
        say("loading the bundle on the NPU with llama-server")
        say("on the NPU: 24/24 layers, 400 MiB of weights")
    summary = {"bundle": "C:/b", "verdict": "passed", "reasons": [], "minutes": 0.1,
               "npu_placement": {"layers_offloaded": 24, "layers_total": 24, "htp_weight_mib": 400},
               "speed": {"median_tokens_per_s": 40.0}, "perplexity_ratio": 1.04, "base_model": False}
    session.ready(summary)
    return summary, object()


def _fake_after(session: Any, summary: dict[str, Any], *, tag: str, installer: Any, **_kw: Any) -> None:
    if session.confirm("Install it into the fork?"):
        session.say("installing into the fork as " + tag)
        session.say("installed as " + tag)
        summary["install"] = {"name": tag, "placement": {"where": "npu", "npu_mib": 400}, "warnings": []}
        session.say(f"loading {tag} through the fork")
        summary["validation"] = {"placement": {"where": "npu", "npu_mib": 400}, "load_s": 1.2,
                                 "chat": {"correct": 2, "of": 2, "answers": ["Paris.", "Berlin."]},
                                 "speed": {"decode_tokens_per_s": 30.0}, "cpu": {"cores": 0.05, "processes": 2}}


def test_options_parse_flags_values_and_quoted_paths() -> None:
    positional, opts = _options(['"C:\\models\\my bundle"', "--name", "x:npu", "--overwrite"], {"--overwrite"},
                                {"--name"})
    assert positional == ["C:\\models\\my bundle"] and opts == {"--name": "x:npu", "--overwrite": "1"}
    with pytest.raises(tui.UsageError):
        _options(["--name"], set(), {"--name"})
    with pytest.raises(tui.UsageError):
        _options(["--what"], set(), set())


def test_a_typed_model_goes_home_to_model_to_work_to_done(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TemperumApp, "_plan_job", _fake_plan_job)
    monkeypatch.setattr(flows, "gguf_plan_table", lambda plan: "the plan")
    monkeypatch.setattr(flows, "run_gguf", _fake_run)
    monkeypatch.setattr(flows, "after_convert", _fake_after)
    monkeypatch.setattr(flows, "gguf_installer", lambda session, result, tag, exe, **kw: (lambda endpoint: {}))
    copied: list[tuple[str, str]] = []
    monkeypatch.setattr(TemperumApp, "copy_text", lambda self, text, what: copied.append((what, text)))

    async def test(app: TemperumApp, pilot: Any) -> None:
        assert isinstance(app.screen, HomeScreen)
        assert app.screen.query_one("#verified", OptionList).option_count >= 1  # the verified models
        await pilot.press(*"Qwen/Qwen2.5-0.5B-Instruct", "enter")
        await _until(pilot, lambda: isinstance(app.screen, ModelScreen) and app.screen.chosen is not None)
        screen = app.screen
        assert isinstance(screen, ModelScreen)
        assert screen.chosen == "Q8_0"  # a 0.49B model: the recommendation is 8 bits
        pressed = screen.quants.pressed_button
        assert pressed is not None and "★ recommended" in str(pressed.label) and pressed.id == "q-Q8_0"
        assert screen.query_one("#g-ram", Gauge).value_text == "2.0 GB needed, 6.0 GB free of 15.4"
        assert "0.6 GB of the NPU's 5.5 GB window" in screen.query_one("#g-npu", Gauge).value_text
        assert str(pressed.label).startswith("Q8_0  best quality  ·  0.6 GB  ·  ~3 min  ·  whole on the NPU")
        assert "be careful" in str(screen.query_one("#warnings").render())
        await pilot.press("enter")  # convert
        try:
            await _until(pilot, lambda: isinstance(app.screen, WorkScreen) and app.pending is not None)
        except AssertionError:
            raise AssertionError(f"no question; screen {type(app.screen).__name__}; history {app.history[-4:]}")
        work = app.screen
        assert isinstance(work, WorkScreen)
        stages = work.query_one("#stages", StageList)
        assert stages.states["download"] == "done" and stages.states["verify on the NPU"] == "done"
        assert work.query_one("#choice", ChoiceBar).display and "Install it" in app.history[-1]
        graph = work.query_one("#graph", ResourceGraph)
        assert str(graph.query_one("#value-npu").render()) == "24 of 24 layers (400 MiB)"
        await pilot.press("y")
        await _until(pilot, lambda: isinstance(app.screen, DoneScreen))
        done = app.screen
        assert isinstance(done, DoneScreen)
        assert stages.states["install"] == "done" and stages.states["validate"] == "done"
        assert done.run_command() == "ollama run qwen2.5-0.5b-instruct:npu"
        assert "24 of 24 layers" in done.query_one("#g-npu", Gauge).value_text
        assert done.query_one("#g-quality", Gauge).value_text.startswith("perplexity 1.04x")
        assert "0.05 of" in done.query_one("#g-cpu", Gauge).value_text
        await pilot.press("c")
        await pilot.pause()
        assert copied[-1] == ("the run command", "ollama run qwen2.5-0.5b-instruct:npu")
        await pilot.press("enter")  # convert another: back home
        await _until(pilot, lambda: isinstance(app.screen, HomeScreen))
        assert app.busy is None
    _run(test)


def test_a_lookup_that_fails_is_shown_on_the_model_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(self: TemperumApp, screen: ModelScreen, model: str, opts: dict[str, str]) -> None:
        self.call_from_thread(screen.show_error, "No such model", "Check the id on huggingface.co")
    monkeypatch.setattr(TemperumApp, "_plan_job", failing)

    async def test(app: TemperumApp, pilot: Any) -> None:
        await pilot.press(*"org/missing", "enter")
        await _until(pilot, lambda: isinstance(app.screen, ModelScreen))
        await pilot.pause(0.2)
        text = str(app.screen.query_one("#card").render())
        assert "Could not plan this model" in text and "No such model" in text and "Check the id" in text
        await pilot.press("escape")
        await _until(pilot, lambda: isinstance(app.screen, HomeScreen))
    _run(test)


def test_a_job_asks_with_buttons_and_drives_the_stage_bar() -> None:
    answers: list[Any] = []

    def job(session: TuiSession) -> None:
        with session.stages(2.0) as say:
            say("fetching the model")
            answers.append(session.confirm("Carry on?", default=False))
            say("converting to GGUF")
        answers.append(session.ask("Name?", "default-name"))

    async def test(app: TemperumApp, pilot: Any) -> None:
        app.run_job("testing", job, title="Testing", stages=["download", "convert"])
        await _until(pilot, lambda: app.pending is not None)
        assert isinstance(app.screen, WorkScreen)
        assert app.screen.query_one("#stage").display and app.screen.query_one("#stages", StageList).running == "download"
        await pilot.click("#no")
        await _until(pilot, lambda: app.pending is not None and app.pending.kind == "ask")
        assert answers == [False]
        await pilot.press("enter")  # the default name
        await _until(pilot, lambda: app.busy is None)
        assert answers == [False, "default-name"]
        assert app.screen.query_one("#stages", StageList).states == {"download": "done", "convert": "done"}
        assert any("answer: default-name" in line for line in app.history)
        await pilot.press("escape")
        await _until(pilot, lambda: isinstance(app.screen, HomeScreen))
    _run(test)


def test_errors_in_a_job_are_shown_not_fatal() -> None:
    def job(session: TuiSession) -> None:
        raise TemperumError("import", "EXE_REQUIRED", "Installing needs the fork executable.", hint="set it")

    async def test(app: TemperumApp, pilot: Any) -> None:
        app.run_job("installing", job)
        await _until(pilot, lambda: app.busy is None)
        assert any("Error [import/EXE_REQUIRED]: Installing needs the fork executable." in line for line in app.history)
        assert any("Hint: set it" in line for line in app.history) and app.is_running
        assert not isinstance(app.screen, DoneScreen)
    _run(test)


def test_up_and_down_walk_the_history_and_it_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TemperumApp, "start_model", lambda self, model, opts: None)

    async def first(app: TemperumApp, pilot: Any) -> None:
        await pilot.press(*"org/one", "enter")
        await pilot.press(*"org/two", "enter")
        await pilot.press(*"dra")
        prompt = app.screen.query_one("#prompt", Input)
        await pilot.press("up")
        assert prompt.value == "org/two"
        await pilot.press("up")
        assert prompt.value == "org/one"
        await pilot.press("up")
        assert prompt.value == "org/one"  # the oldest
        await pilot.press("down", "down")
        assert prompt.value == "dra"  # back to what was being typed
    _run(first)

    async def second(app: TemperumApp, pilot: Any) -> None:
        assert app.commands == ["org/one", "org/two"]
        await pilot.press("up")
        assert app.screen.query_one("#prompt", Input).value == "org/two"
    _run(second)


def test_selected_text_in_the_details_is_copied_with_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    copied: list[tuple[str, str]] = []
    monkeypatch.setattr(TemperumApp, "copy_text", lambda self, text, what: copied.append((what, text)))

    def job(session: TuiSession) -> None:
        session.say("the first line of the details")
        session.say("the second line of the details")

    async def test(app: TemperumApp, pilot: Any) -> None:
        app.run_job("saying", job)
        await _until(pilot, lambda: app.busy is None)
        await pilot.mouse_down("#log", offset=(2, 1))  # inside the content, under the top border
        await pilot.hover("#log", offset=(40, 2))
        await pilot.mouse_up("#log", offset=(40, 2))
        await pilot.pause()
        assert app.screen.get_selected_text()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert copied[-1][0] == "the selection" and "first line" in copied[-1][1]
        assert app.is_running and app.screen.get_selected_text() in (None, "")
        app.action_mouse()  # F2: the terminal gets the mouse (no driver methods headlessly, no crash)
        assert app.native_mouse
        app.action_mouse()
        assert not app.native_mouse
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


def test_commands_still_run_from_the_home_box(monkeypatch: pytest.MonkeyPatch) -> None:
    table = Table("check", "state", box=None)
    table.add_row("fork", "ready")
    monkeypatch.setattr("temperum.cli.main._doctor_report", lambda: ({}, table))

    async def test(app: TemperumApp, pilot: Any) -> None:
        await pilot.press(*"/doctor", "enter")
        await _until(pilot, lambda: isinstance(app.screen, WorkScreen) and app.busy is None)
        assert any("fork" in line and "ready" in line for line in app.history)
        await pilot.press("escape")
        await _until(pilot, lambda: isinstance(app.screen, HomeScreen))
        await pilot.press(*"/nonsense", "enter")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)  # unknown commands only notify
    _run(test)
