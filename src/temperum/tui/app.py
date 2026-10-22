"""The full-screen interface, for someone who wants a model on the NPU without learning the command line.

Four screens: Home (what this does, the models verified here, a box for a model id), Model (what the model
is, what the conversion takes, which quantization to use), Work (the stages, a progress bar, the machine's
load, the details, questions as buttons) and Done (what came out and how to use it). `temperum` with no
arguments opens it in a terminal; the command line keeps its brief output for people who prefer that.

A job runs in a worker thread through a TuiSession, the flows.Session of this front end. Widgets are
touched only from the app's thread, so every update crosses over with call_from_thread, and a question
blocks the worker until the person answers.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.selection import Selection
from textual.strip import Strip
from textual.timer import Timer
from textual.widgets import (Button, Checkbox, Input, Label, OptionList, ProgressBar, RadioButton, RadioSet, RichLog,
                             Static)
from textual.widgets.option_list import Option

from temperum import __version__, flows, profiles, settings, templates, ui
from temperum.core.errors import TemperumError
from temperum.tui.monitor import Monitor, Sample
from temperum.tui.widgets import ChoiceBar, Gauge, Question, ResourceGraph, StageList

_SCALE = 1000
PROMPT_HINT = "Type a Hugging Face model id (for example Qwen/Qwen2.5-0.5B-Instruct), or /help"
HISTORY_FILE = "tui-history.txt"
HISTORY_KEEP = 200
NPU_WINDOW_GB = profiles.NPU_WINDOW_GB  # the whole model must map into the backend's buffer at once
QUANT_WORDS = {"Q4_0": "fastest and smallest", "Q8_0": "best quality",
               "Q4_K_M": "in between (Q4_0 is better supported)"}
PASS_RATIO = 1.20  # verification's perplexity pass line
CONVERT_STAGES = ["prepare tools", "download", "convert", "assemble", "verify on the NPU", "install", "validate"]
TRACKER_ROWS = {"preparing": "prepare tools", "downloading": "download", "converting": "convert",
                "assembling": "assemble", "verifying": "verify on the NPU", "installing": "install"}
COMMANDS: tuple[tuple[str, str], ...] = (
    ("MODEL  or  /convert MODEL [--quant Q4_0|Q8_0|Q4_K_M] [--recipe auto] [--context N] [--name TAG] "
     "[--verify LEVEL]",
     "look a model up, choose a quantization (or search the best mix on the NPU), convert it, then install, "
     "validate, upload"),
    ("/install BUNDLE [--name TAG]", "install a converted bundle into the fork, then validate, upload"),
    ("/validate TAG", "load a model through the fork: where it runs, speed, CPU use"),
    ("/push TAG [--as NAME]", "upload an installed model to ollama.com through the fork"),
    ("/doctor", "this machine, the fork, the toolchains"),
    ("/models", "conversions verified on hardware"),
    ("/config [KEY [VALUE]]", "what temperum remembers: ollama-exe, target, host"),
    ("/help", "this list"),
    ("/quit", "leave (Ctrl+Q); a running job stops at its current step"),
)
HELP_NOTES = ("Up and Down recall earlier entries (kept between sessions). Drag the mouse over the details to select "
              "text and press Ctrl+C to copy it. F2 hands the mouse to the terminal, so text is selected and "
              "copied the terminal's own way (F2 again takes it back).")
INTRO = ("Temperum takes a model from Hugging Face, converts it for this machine's NPU, checks that the result "
         "answers like the original and runs on the NPU, and installs it into Ollama so any app can use it.\n"
         "Pick a model below, or type any Hugging Face model id.")

Job = Callable[["TuiSession"], None]


# -- small shared widgets -----------------------------------------------------------------------------
class Transcript(RichLog):
    """A RichLog whose text can be selected with the mouse and copied (Textual gives that to Log, not RichLog)."""

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        return selection.extract("\n".join(strip.text for strip in self.lines)), "\n"

    def selection_updated(self, selection: Selection | None) -> None:
        self.refresh()

    def _render_line(self, y: int, scroll_x: int, width: int) -> Strip:
        selection = self.text_selection
        span = selection.get_span(y) if selection is not None and y < len(self.lines) else None
        if span is None:
            return super()._render_line(y, scroll_x, width)
        start, end = span
        text = Text(no_wrap=True)
        for segment in self.lines[y]:
            text.append(segment.text, segment.style)
        text.stylize(self.screen.get_component_rich_style("screen--selection"), start,
                     len(text) if end == -1 else end)
        return Strip(text.render(self.app.console), text.cell_len).crop_extend(scroll_x, scroll_x + width,
                                                                               self.rich_style)


class PromptInput(Input):
    """The box on the home screen: Up and Down walk through earlier entries, like a shell."""

    BINDINGS = [Binding("up", "history(-1)", "Earlier entry", show=False),
                Binding("down", "history(1)", "Later entry", show=False)]

    def action_history(self, step: int) -> None:
        app = self.app
        if isinstance(app, TemperumApp):
            app.history_step(step)


def _windows_clipboard(text: str) -> None:
    """Set the Windows clipboard through PowerShell, for terminals that ignore the OSC 52 sequence."""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
            handle.write(text)
            path = handle.name
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Get-Content -Raw -Encoding UTF8 -LiteralPath '{path}' | Set-Clipboard"],
                       capture_output=True, timeout=20)
        os.unlink(path)
    except Exception:  # noqa: BLE001  (best effort: the OSC 52 copy may have worked)
        pass


class UsageError(Exception):
    """A command was typed wrongly; the message is the usage."""


def _options(args: list[str], flags: set[str], valued: set[str]) -> tuple[list[str], dict[str, str]]:
    positional: list[str] = []
    options: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in flags:
            options[arg] = "1"
        elif arg in valued:
            if i + 1 >= len(args):
                raise UsageError(f"{arg} needs a value")
            options[arg] = args[i + 1].strip('"')
            i += 1
        elif arg.startswith("--"):
            raise UsageError(f"unknown option {arg}")
        else:
            positional.append(arg.strip('"'))
        i += 1
    return positional, options


# -- the plan in words: which quantization, and why ---------------------------------------------------
def fits_npu(plan: Any) -> bool:
    return float(plan.estimates["gguf_gb"]) <= NPU_WINDOW_GB


def recommend(plans: dict[str, Any]) -> tuple[str, str]:
    """The quantization to suggest, with the reason in words. Plans that failed are TemperumErrors."""
    usable = {quant: plan for quant, plan in plans.items() if not isinstance(plan, Exception)}
    if not usable:
        return "Q4_0", "no quantization could be planned"
    q8, q4 = usable.get("Q8_0"), usable.get("Q4_0")
    params = float(next(iter(usable.values())).estimates["params_billions"])
    if q8 is not None and params < 1.0 and fits_npu(q8):
        return "Q8_0", ("a model this small loses noticeable quality at 4 bits, and at 8 bits it is still small and "
                        "fast")
    if q4 is not None:
        why = ("at this size 4 bits keeps the quality within a few percent of the original, and it is the NPU's "
               "fastest type")
        if q8 is not None and not fits_npu(q8):
            why += "; Q8_0 would not fit in the NPU's memory window"
        return "Q4_0", why
    quant = next(iter(usable))
    return quant, "the only type that can be planned here"


def quant_label(quant: str, plan: Any, recommended: str) -> str:
    if isinstance(plan, Exception):
        return f"{quant}: not possible here ({plan})"
    est = plan.estimates
    fit = "whole on the NPU" if fits_npu(plan) else "too big for the NPU: runs split, slowly"
    star = "  ★ recommended" if quant == recommended else ""
    return (f"{quant}  {QUANT_WORDS.get(quant, quant)}  ·  {est['gguf_gb']} GB  ·  ~{est['minutes']:.0f} min  ·  "
            f"{fit}{star}")


def model_card(model: Any, plan: Any, machine: Any) -> Table:
    from temperum.convert import disk_free_gb
    from temperum.gguf import trained_context
    from temperum.paths import home

    est = plan.estimates
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    repo = model.ref.repo or str(model.ref.path)
    publisher = f"  (by {repo.split('/')[0]})" if "/" in repo else ""
    grid.add_row("Model", escape(repo) + escape(publisher))
    grid.add_row("What it is", escape(f"{model.architecture or 'unknown architecture'}, "
                                      f"{est['params_billions']} billion parameters"))
    grid.add_row("Chat", escape(templates.describe(getattr(model, "chat_template", None))))
    grid.add_row("Licence", escape(model.license or "unknown") + (" (gated: accepted on Hugging Face)"
                                                                  if getattr(model, "gated", False) else ""))
    trained = trained_context(model.config)
    grid.add_row("Context", escape(f"{plan.context} tokens" + (f" (trained for {trained})"
                                                               if trained and trained != plan.context else "")))
    grid.add_row("Download", escape(f"{est['download_gb']} GB" if est.get("download_gb") else
                                    "already on this machine"))
    total, free = getattr(machine, "total_ram_gb", None), getattr(machine, "available_ram_gb", None)
    disk = disk_free_gb(home())
    grid.add_row("This machine", escape(f"{total} GB of RAM, {free} GB free now; {os.cpu_count() or '?'} cores; "
                                        + (f"{disk:.0f} GB of disk free" if disk is not None else "disk unknown")))
    return grid


_LAYERS = re.compile(r"on the NPU: (\d+)/(\d+) layers(?:, ([\d.]+) MiB of weights)?")
_WHOLE = re.compile(r"whole on the NPU(?: \(([\d.]+) MiB\))?")
_SPLIT = re.compile(r"(\d+) of (\d+) MiB on the NPU")


def npu_share(message: str) -> Optional[tuple[float, str]]:
    """The share of the model on the NPU that a progress line reports, with words for it: verification's
    layer count, the install's and validation's bytes; None for any other line."""
    layers = _LAYERS.search(message)
    if layers:
        on, total = int(layers.group(1)), int(layers.group(2))
        mib = f" ({layers.group(3)} MiB)" if layers.group(3) else ""
        return (on / total if total else 0.0), f"{on} of {total} layers{mib}"
    if message.startswith("on the NPU: no layers"):
        return 0.0, "no layers: the server put nothing on the NPU"
    whole = _WHOLE.search(message)
    if whole:
        return 1.0, "the whole model" + (f" ({whole.group(1)} MiB)" if whole.group(1) else "")
    split = _SPLIT.search(message)
    if split:
        on, total = int(split.group(1)), int(split.group(2))
        return (on / total if total else 0.0), f"{on} of {total} MiB, the rest on the CPU"
    if "the fork runs it on the CPU" in message:
        return 0.0, "nothing: the model runs on the CPU"
    return None


def _reasons_text(summary: dict[str, Any]) -> str:
    return "\n".join(f"  • {escape(str(reason))}" for reason in summary.get("reasons") or [])


# -- the session the jobs talk to ----------------------------------------------------------------------
class TuiSession:
    """flows.Session for the app, used from the worker thread."""

    interactive = True

    def __init__(self, app: TemperumApp) -> None:
        self.app = app
        self.start = time.time()

    def _ui(self, callback: Callable[..., Any], *args: Any) -> Any:
        try:
            return self.app.call_from_thread(callback, *args)
        except RuntimeError:  # the app has left; the job finishes its current step on its own
            return None

    def say(self, message: str) -> None:
        minutes, seconds = divmod(int(time.time() - self.start), 60)
        self._ui(self.app.write, f"[dim]{minutes:3d}:{seconds:02d}[/dim] {escape(message)}")
        self._ui(self.app.note_progress, message)

    def note(self, text: str, warning: bool = False) -> None:
        self._ui(self.app.write, ("  [yellow]![/yellow] " if warning else "  ") + escape(text))
        self._ui(self.app.note_progress, text)

    def show(self, renderable: Any) -> None:
        self._ui(self.app.write, renderable)

    def ready(self, summary: dict[str, Any]) -> None:
        self._ui(self.app.write, ui.ready_panel(summary))
        self._ui(self.app.set_summary, summary)

    def confirm(self, question: str, default: bool = True) -> bool:
        while True:
            answer = self._pose(Question(question, "yes" if default else "no", "confirm")).strip().lower()
            if answer == "":
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            self.note("answer y or n (Enter takes the default)", warning=True)

    def ask(self, prompt: str, default: str) -> str:
        return self._pose(Question(prompt, default, "ask")).strip() or default

    def _pose(self, question: Question) -> str:
        self._ui(self.app.pose, question)
        while not question.answered.wait(0.25):
            if not self.app.is_running:  # the app left without answering: the job must not wait forever
                question.abandoned = True
                break
        if question.abandoned:
            raise flows.Cancelled()
        return question.answer

    def open_url(self, url: str) -> None:
        webbrowser.open(url)

    @contextlib.contextmanager
    def stages(self, estimate_minutes: float, probe: ui.BytesProbe | None = None,
               shares: Sequence[float] | None = None) -> Iterator[Callable[[str], None]]:
        tracker = ui.StageTracker(estimate_minutes, probe, shares=shares)
        self._ui(self.app.begin_stages, tracker)
        ok = False
        try:
            def say(message: str) -> None:
                tracker.say(message)
                self.say(message)
            yield say
            ok = True
        finally:
            self._ui(self.app.end_stages, tracker, ok)

    def transfer(self, label: str) -> flows.Transfer:
        self._ui(self.app.begin_transfer, label)

        def update(status: str, completed: int, total: int) -> None:
            self._ui(self.app.update_transfer, status, completed, total)

        def stop() -> None:
            self._ui(self.app.end_transfer)
        return update, stop


# -- screens --------------------------------------------------------------------------------------------
class HomeScreen(Screen[None]):
    """What this does, the models verified on this kind of machine, and a box for any model id."""

    def compose(self) -> ComposeResult:
        yield Static(f" Temperum {__version__}  ·  models for this machine's NPU", id="title")
        yield Static(INTRO, id="intro")
        yield Static("[dim]looking at this machine…[/dim]", id="machine")
        yield Label("Verified on this kind of machine (Enter converts the highlighted one):", id="verified-label")
        yield OptionList(id="verified")
        yield PromptInput(placeholder=PROMPT_HINT, id="prompt")
        yield Static(" Enter: convert  ·  ↑↓: history, or the list  ·  F1: help  ·  F2: mouse to the terminal  ·  "
                     "Ctrl+Q: quit", id="hint")

    def on_mount(self) -> None:
        from temperum.recipes import verified_conversions

        options = self.query_one("#verified", OptionList)
        entries = [e for e in verified_conversions() if e.get("target") == "gguf-hexagon"]
        for entry in sorted(entries, key=lambda e: str(e["repo"]).lower()):
            options.add_option(Option(f"{entry['repo']}   ·   {entry.get('decode_tokens_per_s', '?')} tokens/s on the "
                                      f"NPU, quality x{entry.get('ppl_ratio', '?')}", id=str(entry["repo"])))
        self.query_one("#prompt", Input).focus()

    @on(OptionList.OptionSelected, "#verified")
    def _picked(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.app.submit(event.option.id)  # type: ignore[attr-defined]

    @on(Input.Submitted, "#prompt")
    def _typed(self, event: Input.Submitted) -> None:
        text = event.value
        event.input.value = ""
        self.app.submit(text)  # type: ignore[attr-defined]


class ModelScreen(Screen[None]):
    """What the model is, what the conversion will take, and which quantization to use."""

    BINDINGS = [Binding("escape", "back", "Back", show=False),
                Binding("enter", "convert", "Convert", priority=True, show=False)]

    def __init__(self, model: str, opts: dict[str, str]) -> None:
        super().__init__()
        self.model_id = model
        self.opts = opts
        self.plans: dict[str, Any] = {}
        self.exe: Optional[Path] = None
        self.machine: Any = None
        self.chosen: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Static(f" {escape(self.model_id)}", id="title")
        with VerticalScroll(id="body"):
            yield Static(f"[dim]looking up {escape(self.model_id)} on Hugging Face…[/dim]", id="card")
            with Vertical(id="fit"):
                yield Gauge("RAM needed", id="g-ram")
                yield Gauge("NPU memory", id="g-npu")
                yield Static("", id="time")
            yield Static("", id="quants-label")
            self.quants = RadioSet(classes="quants")
            yield self.quants
            yield Checkbox("Search the quantization mix on the NPU (Q4_0 base; each candidate built, scored and "
                           "timed; adds the minutes shown)", value=self.opts.get("--recipe") == "auto",
                           id="recipe")
            yield Static("", id="warnings")
            with Horizontal(id="buttons"):
                yield Button("Convert", id="convert", variant="primary", disabled=True)
                yield Button("Back", id="back")
        yield Static(" Enter: convert  ·  ↑↓ Space: pick a quantization  ·  Esc: back", id="hint")

    def show(self, model: Any, plans: dict[str, Any], chosen: str, recommended: str, reason: str,
             exe: Optional[Path], machine: Any) -> None:
        self.plans, self.exe, self.machine = plans, exe, machine
        usable = {quant: plan for quant, plan in plans.items() if not isinstance(plan, Exception)}
        if chosen not in usable:
            chosen = recommended if recommended in usable else next(iter(usable))
        self.chosen = chosen
        self.query_one("#card", Static).update(model_card(model, usable[chosen], machine))
        self.query_one("#quants-label", Static).update(
            f"\n[bold]Quantization[/bold]  [dim]recommended: {recommended}, {escape(reason)}[/dim]")
        old = self.quants
        buttons = [RadioButton(quant_label(quant, plan, recommended), value=quant == chosen, id=f"q-{quant}",
                               disabled=isinstance(plan, Exception)) for quant, plan in plans.items()]
        fresh = RadioSet(*buttons, classes="quants")
        self.quants = fresh
        old.remove()
        self.query_one("#body", VerticalScroll).mount(fresh, before="#warnings")
        self._fit(chosen)
        self.query_one("#convert", Button).disabled = False
        self.call_after_refresh(fresh.focus)

    def show_error(self, message: str, hint: Optional[str]) -> None:
        self.query_one("#card", Static).update(f"[bold red]Could not plan this model:[/bold red] {escape(message)}"
                                               + (f"\n[yellow]Hint:[/yellow] {escape(hint)}" if hint else "")
                                               + "\n\n[dim]Esc goes back.[/dim]")

    def _fit(self, quant: str) -> None:
        plan = self.plans.get(quant)
        if plan is None or isinstance(plan, Exception):
            return
        est = plan.estimates
        total = float(getattr(self.machine, "total_ram_gb", None) or 0) or None
        free = getattr(self.machine, "available_ram_gb", None)
        ram = float(est["ram_gb"])
        colour = "green" if free is None or ram <= float(free) else ("yellow" if total and ram <= total * 0.9 else "red")
        self.query_one("#g-ram", Gauge).set(ram / total if total else 0.0,
                                            f"{ram} GB needed, {free} GB free of {total}", colour)
        gguf = float(est["gguf_gb"])
        self.query_one("#g-npu", Gauge).set(gguf / NPU_WINDOW_GB, f"{gguf} GB of the NPU's {NPU_WINDOW_GB} GB window"
                                            + ("" if fits_npu(plan) else ": too big, would run split with the CPU"),
                                            "green" if fits_npu(plan) else "red")
        from temperum import optimize

        search = self.query_one("#recipe", Checkbox)
        search.disabled = quant != "Q4_0"
        extra = optimize.estimate_minutes(float(est["params_billions"])) if (search.value and quant == "Q4_0") else 0
        self.query_one("#time", Static).update(f"[dim]{'Time':<18}[/dim] about {est['minutes'] + extra:.0f} min"
                                               + (f" (of which {extra:.0f} searching mixes)" if extra else "")
                                               + f"; {est['disk_gb']} GB of disk while converting")
        warnings = [str(w) for w in getattr(plan, "warnings", [])]
        if plan.destination.exists():
            warnings.append(f"converted before: {plan.destination} is rebuilt (finished steps are reused)")
        self.query_one("#warnings", Static).update("\n".join(f"[yellow]![/yellow] {escape(w)}" for w in warnings))

    @on(RadioSet.Changed)
    def _changed(self, event: RadioSet.Changed) -> None:
        quant = str(event.pressed.id or "")[2:]
        if quant in self.plans:
            self.chosen = quant
            self._fit(quant)

    @on(Checkbox.Changed, "#recipe")
    def _recipe_changed(self) -> None:
        if self.chosen:
            self._fit(self.chosen)

    @on(Button.Pressed, "#convert")
    def _convert_button(self) -> None:
        self.action_convert()

    @on(Button.Pressed, "#back")
    def _back_button(self) -> None:
        self.action_back()

    def action_convert(self) -> None:
        plan = self.plans.get(self.chosen or "")
        if plan is None or isinstance(plan, Exception):
            return
        if self.query_one("#recipe", Checkbox).value and plan.quant == "Q4_0" and not plan.recipe:
            from temperum import optimize

            plan.recipe = "auto"  # the search runs between the Q8_0 conversion and assembly
            plan.estimates["minutes"] += optimize.estimate_minutes(float(plan.estimates["params_billions"]))
        self.app.convert(plan, self.exe, self.opts)  # type: ignore[attr-defined]

    def action_back(self) -> None:
        self.app.pop_screen()


class WorkScreen(Screen[None]):
    """A job as it runs: the stages, a progress bar, the machine's load, the details, questions as buttons."""

    BINDINGS = [Binding("escape", "back", "Back", show=False),
                Binding("pageup", "scroll_log(-1)", "Earlier details", show=False),
                Binding("pagedown", "scroll_log(1)", "Later details", show=False)]

    def __init__(self, title: str, stages: list[str], total_gb: Optional[float],
                 on_ready: Optional[Callable[[], None]] = None) -> None:
        super().__init__()
        self.title_text = title
        self.stage_names = stages
        self.total_gb = total_gb
        self.on_ready = on_ready  # the job starts once the screen can show it

    def compose(self) -> ComposeResult:
        yield Static(f" {escape(self.title_text)}", id="title")
        with Horizontal(id="top"):
            yield StageList(self.stage_names, id="stages")
            yield ResourceGraph(self.total_gb, os.cpu_count() or 1, id="graph")
        with Horizontal(id="stage"):
            yield Label("", id="stage-label")
            yield ProgressBar(total=_SCALE, show_eta=False, id="bar")
            yield Label("", id="stage-eta")
        yield ChoiceBar(id="choice")
        yield Transcript(id="log", markup=True, wrap=True, highlight=False)
        yield Static("", id="status")

    def on_mount(self) -> None:
        self.query_one("#stages", StageList).display = bool(self.stage_names)
        self.query_one("#log", RichLog).focus()
        if self.on_ready is not None:
            self.on_ready()

    def action_scroll_log(self, direction: int) -> None:
        log = self.query_one("#log", RichLog)
        if direction < 0:
            log.scroll_page_up()
        else:
            log.scroll_page_down()

    def action_back(self) -> None:
        app = self.app
        if getattr(app, "busy", None):
            app.notify(f"{app.busy} is still running; Ctrl+C twice stops it", timeout=4)  # type: ignore[attr-defined]
            return
        app.pop_screen()


class DoneScreen(Screen[None]):
    """What came out of a conversion and how to use it."""

    BINDINGS = [Binding("enter", "another", "Convert another", show=False),
                Binding("escape", "another", "Convert another", show=False),
                Binding("c", "copy_command", "Copy the run command", show=False),
                Binding("p", "copy_path", "Copy the bundle path", show=False)]

    def __init__(self, summary: dict[str, Any], tag: Optional[str]) -> None:
        super().__init__()
        self.summary = summary
        self.tag = tag

    def compose(self) -> ComposeResult:
        verdict = str(self.summary.get("verdict"))
        title = {"passed": "Model ready", "warning": "Model ready, with notes"}.get(verdict, "Conversion failed")
        yield Static(f" {title}", id="title")
        with VerticalScroll(id="body"):
            yield Static("", id="verdict")
            yield Gauge("On the NPU", id="g-npu")
            yield Gauge("Quality loss", id="g-quality")
            yield Static("", id="speed")
            yield Static("", id="recipe")
            yield Static("", id="install")
            yield Gauge("CPU during chat", id="g-cpu")
            yield Static("", id="use")
            with Horizontal(id="buttons"):
                yield Button("Convert another", id="another", variant="primary")
                yield Button("Quit", id="quit")
        yield Static(" Enter: convert another  ·  c: copy the run command  ·  p: copy the bundle path  ·  Ctrl+Q: "
                     "quit", id="hint")

    def on_mount(self) -> None:
        s = self.summary
        verdict = str(s.get("verdict"))
        colour = {"passed": "green", "warning": "yellow"}.get(verdict, "red")
        words = {"passed": "Verified: answers like the original and runs on the NPU.",
                 "warning": "Usable, with notes:", "failed": "Not usable:"}.get(verdict, verdict)
        self.query_one("#verdict", Static).update(
            f"[bold {colour}]{escape(words)}[/bold {colour}]\n{_reasons_text(s)}\n"
            f"[dim]{'Bundle':<18}[/dim] {escape(str(s.get('bundle')))}   [dim]took {s.get('minutes')} min[/dim]")
        placed = s.get("npu_placement") or {}
        total_layers = placed.get("layers_total")
        if total_layers:
            on_npu = int(placed.get("layers_offloaded") or 0)
            self.query_one("#g-npu", Gauge).set(on_npu / int(total_layers),
                                                f"{on_npu} of {total_layers} layers, {placed.get('htp_weight_mib')} MiB "
                                                "of weights", "green" if on_npu == int(total_layers) else "yellow")
        else:
            self.query_one("#g-npu", Gauge).display = False
        ratio = s.get("perplexity_ratio")
        if ratio is not None:
            loss = max(0.0, (float(ratio) - 1.0) / (PASS_RATIO - 1.0))
            self.query_one("#g-quality", Gauge).set(loss, f"perplexity {ratio}x the reference (the pass line is "
                                                    f"{PASS_RATIO}x)", "green" if float(ratio) <= PASS_RATIO else "red")
        else:
            self.query_one("#g-quality", Gauge).display = False
        speed = s.get("speed") or {}
        if speed.get("median_tokens_per_s"):
            self.query_one("#speed", Static).update(f"[dim]{'Speed':<18}[/dim] {speed['median_tokens_per_s']:.0f} "
                                                    "tokens per second while verifying")
        recipe = s.get("recipe") or {}
        if recipe.get("label"):
            self.query_one("#recipe", Static).update(f"[dim]{'Recipe':<18}[/dim] {escape(str(recipe['label']))} "
                                                     f"after {recipe.get('trials')} trials on the NPU\n"
                                                     f"[dim]{'':<18}[/dim] {escape(str(recipe.get('why')))}")
        install = s.get("install") or {}
        validation = s.get("validation") or {}
        lines = []
        if install:
            lines.append(f"[dim]{'Installed as':<18}[/dim] [bold]{escape(str(install.get('name')))}[/bold]; "
                         + escape(flows.placement_text(install.get("placement") or {})))
            for warning in install.get("warnings") or []:
                lines.append(f"  [yellow]![/yellow] {escape(str(warning))}")
        if validation:
            chat = validation.get("chat") or {}
            if chat:
                what = "continuations on topic" if chat.get("mode") == "completion" else "answers as expected"
                lines.append(f"[dim]{'Validation':<18}[/dim] {chat.get('correct')}/{chat.get('of')} {what}; loaded in "
                             f"{validation.get('load_s')} s; {(validation.get('speed') or {}).get('decode_tokens_per_s')} "
                             "tokens per second through the fork")
        self.query_one("#install", Static).update("\n".join(lines))
        cpu = validation.get("cpu") or {}
        if cpu.get("cores") is not None:
            cores = float(cpu["cores"])
            machine_cores = os.cpu_count() or 1
            self.query_one("#g-cpu", Gauge).set(cores / machine_cores, f"{cores:.2f} of {machine_cores} cores while "
                                                "the model answered (low: the NPU did the work)",
                                                "green" if cores < 1.0 else "yellow")
        else:
            self.query_one("#g-cpu", Gauge).display = False
        tag = str(install.get("name") or self.tag or "")
        use = []
        if tag and install:
            use.append(f"[dim]{'Use it':<18}[/dim] [bold]{escape(self.run_command())}[/bold]")
            use.append("[dim]Any app that talks to Ollama can use it by that name. The fork's server must run with "
                       "OLLAMA_HEXAGON=1 for the NPU (temperum started one that way if none was running).[/dim]")
        elif verdict != "failed":
            use.append("[dim]Not installed: run /install with the bundle path to install it later.[/dim]")
        self.query_one("#use", Static).update("\n".join(use))

    def run_command(self) -> str:
        install = self.summary.get("install") or {}
        return f"ollama run {install.get('name') or self.tag or ''}"

    @on(Button.Pressed, "#another")
    def _another(self) -> None:
        self.action_another()

    @on(Button.Pressed, "#quit")
    def _quit(self) -> None:
        self.app.action_leave(force=True)  # type: ignore[attr-defined]

    def action_another(self) -> None:
        self.app.go_home()  # type: ignore[attr-defined]

    def action_copy_command(self) -> None:
        self.app.copy_text(self.run_command(), "the run command")  # type: ignore[attr-defined]

    def action_copy_path(self) -> None:
        self.app.copy_text(str(self.summary.get("bundle")), "the bundle path")  # type: ignore[attr-defined]


# -- the app ----------------------------------------------------------------------------------------------
class TemperumApp(App[None]):
    TITLE = "Temperum"
    CSS = """
    Screen { layout: vertical; }
    #title { height: 1; background: $primary; color: $text; padding: 0 1; text-style: bold; }
    #hint { height: 1; color: $text-muted; padding: 0 1; }
    #status { height: 1; color: $text-muted; padding: 0 1; }
    #intro { padding: 1 2 0 2; height: auto; }
    #machine { padding: 1 2 0 2; height: auto; color: $text-muted; }
    #verified-label { padding: 1 2 0 2; color: $text-muted; }
    #verified { height: 1fr; margin: 0 2; }
    #prompt { margin: 0 1; border: tall $primary; }
    ModelScreen #body { padding: 1 2; }
    ModelScreen #card { height: auto; }
    ModelScreen #fit { height: auto; margin-top: 1; }
    ModelScreen #time { height: 1; }
    ModelScreen #quants-label { height: auto; }
    ModelScreen .quants { width: 1fr; height: auto; }
    ModelScreen #warnings { height: auto; margin-top: 1; }
    ModelScreen #buttons { height: 3; margin-top: 1; }
    ModelScreen Button { margin-right: 2; }
    WorkScreen #top { height: 8; padding: 1 1 0 1; }
    WorkScreen #stage { height: 1; padding: 0 1; display: none; }
    #stage-label { width: auto; }
    #bar { width: 1fr; }
    #stage-eta { width: auto; color: $text-muted; }
    WorkScreen #log { height: 1fr; padding: 0 1; border-top: solid $panel; }
    DoneScreen #body { padding: 1 2; }
    DoneScreen Static { height: auto; }
    DoneScreen Gauge { height: 1; }
    DoneScreen #buttons { height: 3; margin-top: 1; }
    DoneScreen Button { margin-right: 2; }
    """
    BINDINGS = [Binding("ctrl+c", "leave", "Copy or quit", priority=True, show=False),
                Binding("ctrl+q", "leave(True)", "Quit", priority=True, show=False),
                Binding("f1", "help", "Help", show=False),
                Binding("f2", "mouse", "Mouse to the terminal", priority=True, show=False)]

    def __init__(self, monitor: Optional[Callable[[Callable[[Sample], None]], Any]] = Monitor) -> None:
        super().__init__()
        self.monitor_factory = monitor
        self.monitor: Any = None
        self.pending: Optional[Question] = None
        self.busy: Optional[str] = None  # the running job's label
        self.work: Optional[WorkScreen] = None  # the screen the running (or last) job writes to
        self.summary: Optional[dict[str, Any]] = None
        self.tag: Optional[str] = None
        self.history: list[str] = []  # plain text of everything written to the details
        self.commands: list[str] = []  # earlier entries, oldest first
        self.fork_state = "checking"
        self.machine: Any = None
        self.native_mouse = False
        self._tracker: Optional[ui.StageTracker] = None
        self._timer: Optional[Timer] = None
        self._leave_asked = 0.0
        self._hist: Optional[int] = None  # index into commands while walking the history, else None
        self._draft = ""
        self._history_value: Optional[str] = None

    def get_default_screen(self) -> Screen[None]:
        return HomeScreen()

    def on_mount(self) -> None:
        self.commands = self._load_history()
        self.run_worker(self._machine_job, thread=True, group="status", exit_on_error=False)

    @property
    def home(self) -> HomeScreen:
        return self.screen_stack[0]  # type: ignore[return-value]

    def go_home(self) -> None:
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.query_one("#prompt", Input).focus()

    # -- this machine --------------------------------------------------------------------------------
    def _machine_job(self) -> None:
        from temperum.cli.main import _fork_status
        from temperum.host import detect_host
        from temperum.install import fork

        status = _fork_status()
        machine = detect_host()
        endpoint = fork.endpoint_for(None)
        server = fork.version(endpoint, timeout=1.5)
        runtime = "ready" if status["state"] == "ready" else f"{status['state']}: {status.get('hint', '')}"
        parts = [f"NPU runtime {runtime}", f"fork server {('running, ' + server) if server else 'not running'}",
                 f"{machine.available_ram_gb} GB of {machine.total_ram_gb} GB RAM free"]
        self.call_from_thread(self._set_machine, machine, status["state"], "  ·  ".join(parts))

    def _set_machine(self, machine: Any, fork_state: str, text: str) -> None:
        self.machine = machine
        self.fork_state = fork_state
        with contextlib.suppress(Exception):
            self.home.query_one("#machine", Static).update(f"[dim]This machine:[/dim] {escape(text)}")

    # -- entries and their history --------------------------------------------------------------------
    def _history_path(self) -> Path:
        return home_dir() / HISTORY_FILE

    def _load_history(self) -> list[str]:
        try:
            lines = self._history_path().read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [line for line in lines if line.strip()][-HISTORY_KEEP:]

    def remember(self, command: str) -> None:
        if not self.commands or self.commands[-1] != command:
            self.commands.append(command)
            del self.commands[:-HISTORY_KEEP]
            try:
                path = self._history_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(self.commands) + "\n", encoding="utf-8")
            except OSError:
                pass
        self._hist = None

    def history_step(self, step: int) -> None:
        """Up (-1) walks to earlier entries, Down (+1) back towards what was being typed."""
        if not self.commands:
            return
        prompt = self.query_one("#prompt", Input)
        if self._hist is None:
            self._draft = prompt.value
            index = len(self.commands)
        else:
            index = self._hist
        index = max(0, min(len(self.commands), index + step))
        self._hist = None if index == len(self.commands) else index
        value = self._draft if self._hist is None else self.commands[index]
        self._history_value = value
        prompt.value = value
        prompt.cursor_position = len(value)

    @on(Input.Changed, "#prompt")
    def typed(self, event: Input.Changed) -> None:
        if event.value != self._history_value:  # typing leaves the history walk
            self._hist = None
            self._history_value = None

    def submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.remember(text)
        if self.busy:
            self.notify(f"{self.busy} is running; wait for it to finish", severity="warning", timeout=4)
            return
        if text.startswith("/"):
            self.dispatch(text)
        else:
            self.start_model(text, {})

    # -- the clipboard and the mouse ------------------------------------------------------------------
    def copy_text(self, text: str, what: str) -> None:
        self.copy_to_clipboard(text)  # OSC 52, for terminals that honour it (Windows Terminal does)
        if sys.platform == "win32":
            threading.Thread(target=_windows_clipboard, args=(text,), daemon=True).start()
        self.notify(f"Copied {what} ({len(text)} characters)", timeout=3)

    def action_mouse(self) -> None:
        """F2: hand the mouse to the terminal, so text is selected and copied its way; F2 again takes it back."""
        driver = self._driver
        self.native_mouse = not self.native_mouse
        method = "_disable_mouse_support" if self.native_mouse else "_enable_mouse_support"
        if driver is not None and hasattr(driver, method):
            getattr(driver, method)()
        self.notify("The terminal has the mouse: drag to select text and copy it the terminal's way. F2 gives it "
                    "back to Temperum." if self.native_mouse else
                    "Temperum has the mouse again: click, scroll, drag to select, Ctrl+C copies.", timeout=6)

    # -- the details log ------------------------------------------------------------------------------
    def write(self, renderable: Any) -> None:
        if isinstance(renderable, str):
            self.history.append(Text.from_markup(renderable).plain)
        else:
            console = Console(file=io.StringIO(), width=100, force_terminal=False)
            console.print(renderable)
            self.history.append(str(console.file.getvalue()))  # type: ignore[attr-defined]
        if self.work is not None and self.work.is_mounted:
            self.work.query_one("#log", RichLog).write(renderable)

    def set_summary(self, summary: dict[str, Any]) -> None:
        self.summary = summary

    def note_progress(self, message: str) -> None:
        """Move the stage checklist along with the pipeline's messages."""
        if self.work is None or not self.work.is_mounted:
            return
        share = npu_share(message)
        if share is not None:
            self.work.query_one("#graph", ResourceGraph).set_npu(*share)
        stages = self.work.query_one("#stages", StageList)
        index = ui.stage_of(message)
        lowered = message.strip().lower()
        if index is not None:
            stages.reach(TRACKER_ROWS[ui.STAGES[index][0]])
        elif lowered.startswith("loading ") and "through the fork" in lowered:
            stages.reach("validate")
        elif lowered.startswith("installed as"):
            stages.mark("install", "done")

    # -- questions ------------------------------------------------------------------------------------
    def pose(self, question: Question) -> None:
        self.pending = question
        self.write(f"[bold cyan]?[/bold cyan] {question.text}")
        if self.work is not None and self.work.is_mounted:
            self.work.query_one("#choice", ChoiceBar).ask(question)
            self.refresh_status()

    @on(ChoiceBar.Answered)
    def answered(self, event: ChoiceBar.Answered) -> None:
        question, self.pending = self.pending, None
        if question is None:
            return
        text = event.text.strip()
        self.write(f"  [dim]answer:[/dim] {escape(text) if text else escape(question.hint) + ' [dim](default)[/dim]'}")
        if self.work is not None and self.work.is_mounted:
            self.work.query_one("#choice", ChoiceBar).hide()
            self.work.query_one("#log", RichLog).focus()
        question.answer = text
        question.answered.set()
        self.refresh_status()

    def refresh_status(self) -> None:
        if self.work is None or not self.work.is_mounted:
            return
        if self.pending is not None:
            text = "waiting for your answer (y / n, or click)"
        elif self.busy:
            text = f"{self.busy}  ·  Ctrl+C copies selected text, twice stops the job"
        else:
            text = "finished  ·  Esc: back  ·  Ctrl+C copies selected text"
        self.work.query_one("#status", Static).update(f" {escape(text)}")

    # -- looking a model up -----------------------------------------------------------------------------
    def start_model(self, model: str, opts: dict[str, str]) -> None:
        screen = ModelScreen(model, opts)
        self.push_screen(screen)
        self.run_worker(lambda: self._plan_job(screen, model, opts), thread=True, group="plan", exclusive=True,
                        exit_on_error=False)

    def _plan_job(self, screen: ModelScreen, model: str, opts: dict[str, str]) -> None:
        try:
            from temperum import gguf
            from temperum.host import detect_host, resolve_target
            from temperum.profiles import get_target
            from temperum.sources import parse_ref, resolve

            machine = self.machine or detect_host()
            target_id = str(settings.resolve("target", opts.get("--target"), "auto").value)
            if target_id == "auto":
                target_id = resolve_target(machine).id
            target = get_target(target_id)
            if target.format != "gguf":
                raise TemperumError("plan", "NOT_GGUF", f"this interface converts GGUF targets; for a QNN bundle run: "
                                                        f"temperum convert {model} --target {target_id}")
            exe, _source = settings.ollama_exe(None)
            resolved = resolve(parse_ref(model))
            context = int(opts["--context"]) if "--context" in opts else None
            plans: dict[str, Any] = {}
            for quant in gguf.QUANT_TYPES:
                try:
                    plans[quant] = gguf.plan(resolved, machine, target, ollama_exe=exe, quant=quant, context=context,
                                             out=Path(opts["--out"]) if "--out" in opts else None,
                                             verify=opts.get("--verify", "standard"), overwrite=True, python=None)
                except TemperumError as exc:
                    plans[quant] = exc
            if all(isinstance(plan, Exception) for plan in plans.values()):
                raise next(plan for plan in plans.values())
            recommended, reason = recommend(plans)
            self.call_from_thread(screen.show, resolved, plans, opts.get("--quant") or recommended, recommended,
                                  reason, exe, machine)
        except TemperumError as exc:
            self.call_from_thread(screen.show_error, str(exc), exc.hint)
        except Exception as exc:  # noqa: BLE001  (a lookup must never take the interface down)
            self.call_from_thread(screen.show_error, f"{type(exc).__name__}: {exc}", None)

    # -- jobs --------------------------------------------------------------------------------------------
    def convert(self, plan: Any, exe: Optional[Path], opts: dict[str, str]) -> None:
        model = plan.model
        display = str(getattr(model.ref, "display", None) or model.name)
        tag = opts.get("--name") or f"{model.name}:npu"

        def job(session: TuiSession) -> None:
            start = time.time()
            session.show(flows.gguf_plan_table(plan))
            for warning in plan.warnings:
                session.note(str(warning), warning=True)
            summary, result = flows.run_gguf(session, plan, start=start)
            if summary["verdict"] == "failed":
                return
            flows.after_convert(session, summary, tag=tag, install=False, validate=False, push=False, push_as=None,
                                exe=exe, host=opts.get("--host"),
                                installer=flows.gguf_installer(session, result, tag, exe, overwrite=True))
        self.run_job(f"converting {display}", job, title=f"Converting {display}", stages=CONVERT_STAGES, tag=tag)

    def run_job(self, label: str, job: Job, *, title: Optional[str] = None, stages: Optional[list[str]] = None,
                tag: Optional[str] = None) -> None:
        self.busy = label
        self.summary = None
        self.tag = tag
        self.work = WorkScreen(title or label, stages or [], getattr(self.machine, "total_ram_gb", None),
                               on_ready=lambda: self._launch(job))
        self.push_screen(self.work)

    def _launch(self, job: Job) -> None:  # the work screen is mounted: nothing it is told gets lost
        self._start_monitor()
        self.refresh_status()
        self.run_worker(lambda: self._job(job), thread=True, exclusive=True, group="job", exit_on_error=False)

    def _job(self, job: Job) -> None:  # the worker thread
        session = TuiSession(self)
        ok = False
        try:
            job(session)
            ok = True
        except flows.Cancelled:
            session._ui(self.write, "[dim]cancelled[/dim]")
        except TemperumError as exc:
            code = escape(f"[{exc.stage}/{exc.reason_code}]")
            session._ui(self.write, f"[bold red]Error[/bold red] {code}: {escape(str(exc))}"
                        + (f"\n[yellow]Hint:[/yellow] {escape(exc.hint)}" if exc.hint else ""))
        except Exception as exc:  # noqa: BLE001  (a job must never take the interface down)
            session._ui(self.write, f"[bold red]{escape(type(exc).__name__)}[/bold red]: {escape(str(exc))}")
        finally:
            session._ui(self._job_done, ok)

    def _job_done(self, ok: bool) -> None:
        self.busy = None
        self._stop_monitor()
        if self.work is not None and self.work.is_mounted:
            self.work.query_one("#stages", StageList).finish(ok)
        self.refresh_status()
        if ok and self.summary is not None:
            self.push_screen(DoneScreen(self.summary, self.tag))

    def _start_monitor(self) -> None:
        self._stop_monitor()
        if self.monitor_factory is None:
            return
        self.monitor = self.monitor_factory(self._sample)
        if not self.monitor.start():
            self.monitor = None
            with contextlib.suppress(Exception):
                self.work.query_one("#graph", ResourceGraph).unavailable("no load readings on this system")  # type: ignore[union-attr]

    def _sample(self, sample: Sample) -> None:  # the monitor's reader thread
        with contextlib.suppress(RuntimeError):
            self.call_from_thread(self._show_sample, sample)

    def _show_sample(self, sample: Sample) -> None:
        if self.work is not None and self.work.is_mounted:
            self.work.query_one("#graph", ResourceGraph).push(sample)

    def _stop_monitor(self) -> None:
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor = None

    # -- commands typed on the home screen --------------------------------------------------------------
    def dispatch(self, text: str) -> None:
        try:
            words = shlex.split(text, posix=False)  # keeps backslashes in Windows paths
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        if not words:
            return
        command, args = words[0].lower(), words[1:]
        if command in ("/quit", "/exit"):
            self.action_leave(force=True)
            return
        if command == "/help":
            self.action_help()
            return
        if command == "/convert":
            try:
                positional, opts = _options(args, {"--overwrite"},
                                            {"--quant", "--context", "--name", "--verify", "--out", "--target",
                                             "--host", "--recipe"})
            except UsageError as exc:
                self.notify(f"usage: {exc}", severity="warning", timeout=6)
                return
            if len(positional) != 1:
                self.notify(f"usage: {COMMANDS[0][0]}", severity="warning", timeout=6)
                return
            self.start_model(positional[0], opts)
            return
        handler = {"/install": self.cmd_install, "/validate": self.cmd_validate, "/push": self.cmd_push,
                   "/doctor": self.cmd_doctor, "/models": self.cmd_models, "/config": self.cmd_config}.get(command)
        if handler is None:
            self.notify(f"unknown command {command}; /help lists them", severity="warning", timeout=6)
            return
        try:
            label, job = handler(args)
        except UsageError as exc:
            self.notify(f"usage: {exc}", severity="warning", timeout=6)
            return
        self.run_job(label, job)

    def cmd_install(self, args: list[str]) -> tuple[str, Job]:
        positional, opts = _options(args, set(), {"--name", "--host"})
        if len(positional) != 1:
            raise UsageError(COMMANDS[1][0])
        bundle = Path(positional[0]).expanduser()

        def job(session: TuiSession) -> None:
            from temperum import paths
            from temperum.install import fork
            from temperum.install.ollama_import import install_gguf

            if not (bundle / "model.gguf").is_file():
                raise TemperumError("import", "BUNDLE_MISSING", f"{bundle} holds no model.gguf.",
                                    hint="Name the bundle folder a conversion produced.")
            exe = fork.executable(None)
            if exe is None:
                raise TemperumError("import", "EXE_REQUIRED", "Installing needs the fork executable.",
                                    hint="/config ollama-exe C:\\path\\to\\fork\\ollama.exe")
            tag = opts.get("--name") or f"{bundle.name.split('-gguf')[0]}:npu"
            work = paths.jobs_dir() / "install" / bundle.name

            def installer(endpoint: str) -> dict[str, Any]:
                return install_gguf(bundle, tag, executable=exe, host=endpoint, work_dir=work, say=session.say)
            flows.after_convert(session, {"bundle": str(bundle)}, tag=tag, install=True, validate=False, push=False,
                                push_as=None, exe=exe, host=opts.get("--host"), installer=installer)
        return f"installing {bundle.name}", job

    def cmd_validate(self, args: list[str]) -> tuple[str, Job]:
        positional, opts = _options(args, set(), {"--host"})
        if len(positional) != 1:
            raise UsageError(COMMANDS[2][0])
        tag = positional[0]

        def job(session: TuiSession) -> None:
            from temperum import paths
            from temperum.install import fork
            from temperum.install.validate import validate as validate_model

            exe = fork.executable(None)
            endpoint = fork.endpoint_for(opts.get("--host"))
            info = fork.ensure(endpoint, exe, session.say, log_dir=paths.home() / "logs")
            if info.started:
                session.note(f"started the fork's server at {endpoint} (pid {info.pid}); it keeps running")
            session.show(ui.validation_table(validate_model(endpoint, tag, fork_dir=exe.parent if exe else None,
                                                            say=session.say)))
        return f"validating {tag}", job

    def cmd_push(self, args: list[str]) -> tuple[str, Job]:
        positional, opts = _options(args, set(), {"--as", "--host"})
        if len(positional) != 1:
            raise UsageError(COMMANDS[3][0])
        tag = positional[0]

        def job(session: TuiSession) -> None:
            from temperum import paths
            from temperum.install import fork

            endpoint = fork.endpoint_for(opts.get("--host"))
            info = fork.ensure(endpoint, fork.executable(None), session.say, log_dir=paths.home() / "logs")
            if info.started:
                session.note(f"started the fork's server at {endpoint} (pid {info.pid}); it keeps running")
            flows.push_flow(session, endpoint, tag, opts.get("--as"))
        return f"uploading {tag}", job

    def cmd_doctor(self, args: list[str]) -> tuple[str, Job]:
        def job(session: TuiSession) -> None:
            from temperum.cli.main import _doctor_report

            _report, table = _doctor_report()
            session.show(table)
        return "doctor", job

    def cmd_models(self, args: list[str]) -> tuple[str, Job]:
        def job(session: TuiSession) -> None:
            from temperum.recipes import verified_conversions

            table = Table("model", "target", "type", "perplexity", "decode", box=None)
            for entry in sorted(verified_conversions(), key=lambda e: (e["target"] != "gguf-hexagon", e["repo"])):
                table.add_row(escape(entry["repo"]), entry["target"], entry["quality"], f"x{entry['ppl_ratio']}",
                              f"{entry['decode_tokens_per_s']} tok/s")
            session.show(table)
            session.note("details in docs/models.md; any architecture llama.cpp's converter handles can be tried")
        return "models", job

    def cmd_config(self, args: list[str]) -> tuple[str, Job]:
        positional, _opts = _options(args, set(), set())
        if len(positional) > 2:
            raise UsageError(COMMANDS[6][0])

        def job(session: TuiSession) -> None:
            from temperum.cli.main import _store_setting

            if len(positional) == 2:
                value = _store_setting(positional[0], positional[1])
                session.note(f"{positional[0]} = {value} (stored in {settings.path()})")
                self.call_from_thread(self.run_worker, self._machine_job, thread=True, group="status",
                                      exit_on_error=False)
                return
            if len(positional) == 1:
                settings.check_key(positional[0])
            table = Table("setting", "in effect", "from", "meaning", box=None)
            for name in settings.KEYS:
                if positional and name != positional[0]:
                    continue
                current = settings.resolve(name)
                table.add_row(name, escape(current.value or ""), current.source if current.value else "",
                              settings.DESCRIPTIONS[name])
            session.show(table)
            session.note(f"stored in {settings.path()}; /config KEY VALUE changes one")
        return "config", job

    # -- actions ---------------------------------------------------------------------------------------
    def action_help(self) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        for usage, meaning in COMMANDS:
            table.add_row(escape(usage), meaning)
        if self.work is not None and self.work.is_mounted and self.screen is self.work:
            self.write(table)
            self.write(f"[dim]{HELP_NOTES}[/dim]")
        else:
            self.notify("\n".join(f"{usage}\n    {meaning}" for usage, meaning in COMMANDS) + "\n\n" + HELP_NOTES,
                        title="Help", timeout=20)

    def action_leave(self, force: bool = False) -> None:
        """Ctrl+C: copy a selection when there is one, else quit (asking first while a job runs)."""
        if not force:
            focused = self.focused
            selected = self.screen.get_selected_text() or (focused.selected_text
                                                           if isinstance(focused, Input) else "")
            if selected:
                self.copy_text(selected, "the selection")
                self.screen.clear_selection()
                return
        if self.busy and not force and time.time() - self._leave_asked > 3:
            self._leave_asked = time.time()
            self.notify(f"{self.busy} is running. Press Ctrl+C again within 3 s to quit anyway; the job stops at "
                        "its current step.", severity="warning", timeout=4)
            return
        if self.pending is not None:
            self.pending.abandoned = True
            self.pending.answered.set()
        self._stop_monitor()
        self.exit()

    # -- the stage bar ---------------------------------------------------------------------------------
    def _stage_widget(self, selector: str, kind: Any) -> Any:
        if self.work is None or not self.work.is_mounted:
            return None
        return self.work.query_one(selector, kind)

    def begin_stages(self, tracker: ui.StageTracker) -> None:
        self._tracker = tracker
        stage = self._stage_widget("#stage", Horizontal)
        if stage is not None:
            stage.display = True
        self._tick()
        self._timer = self.set_interval(0.5, self._tick)

    def _tick(self) -> None:
        tracker = self._tracker
        if tracker is None:
            return
        bar, label, eta = (self._stage_widget("#bar", ProgressBar), self._stage_widget("#stage-label", Label),
                           self._stage_widget("#stage-eta", Label))
        if bar is None:
            return
        self._stage_widget("#stage", Horizontal).display = True
        bar.update(total=_SCALE, progress=tracker.completed() * _SCALE)
        label.update(f"[bold]{tracker.stage_name}[/bold] {escape(tracker.short())} ")
        eta.update(f" {tracker.eta_text()}")
        stages = self._stage_widget("#stages", StageList)
        running = stages.running if stages is not None else None
        if stages is not None and running is not None:
            stages.mark(running, "running", tracker.elapsed_text())

    def end_stages(self, tracker: ui.StageTracker, ok: bool) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._tracker = None
        bar, label, eta = (self._stage_widget("#bar", ProgressBar), self._stage_widget("#stage-label", Label),
                           self._stage_widget("#stage-eta", Label))
        if bar is None:
            return
        if ok:
            bar.update(total=_SCALE, progress=_SCALE)
            label.update(f"[bold]done[/bold] in {tracker.elapsed_text()} ")
            stages = self._stage_widget("#stages", StageList)
            if stages is not None and stages.running is not None:
                stages.mark(stages.running, "done", tracker.elapsed_text())
        else:
            label.update(f"[bold red]stopped[/bold red] {escape(tracker.short())} ")
        eta.update("")

    def begin_transfer(self, label: str) -> None:
        stage = self._stage_widget("#stage", Horizontal)
        if stage is None:
            return
        stage.display = True
        self._stage_widget("#stage-label", Label).update(f"[bold]{escape(label)}[/bold] ")
        self._stage_widget("#stage-eta", Label).update("")
        self._stage_widget("#bar", ProgressBar).update(total=None)

    def update_transfer(self, status: str, completed: int, total: int) -> None:
        label = self._stage_widget("#stage-label", Label)
        if label is None:
            return
        label.update(f"[bold]{escape(status[:48])}[/bold] ")
        if total:
            self._stage_widget("#bar", ProgressBar).update(total=total, progress=completed)
            self._stage_widget("#stage-eta", Label).update(f" {completed / 2**20:.0f} / {total / 2**20:.0f} MiB")

    def end_transfer(self) -> None:
        stage = self._stage_widget("#stage", Horizontal)
        if stage is None:
            return
        stage.display = False
        self._stage_widget("#bar", ProgressBar).update(total=_SCALE, progress=0)


def home_dir() -> Path:
    from temperum import paths

    return paths.home()


def run() -> None:
    TemperumApp().run()
