"""Pieces of the full-screen interface: a gauge, the stage checklist, the live load graphs, the question bar."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Label, Sparkline, Static

from temperum.tui.monitor import Sample

FILLED, EMPTY = "█", "░"


def bar(fraction: float, width: int = 24) -> str:
    """A bar of `width` cells, filled in proportion (clamped to 0..1)."""
    cells = max(0, min(width, round(max(0.0, min(1.0, fraction)) * width)))
    return FILLED * cells + EMPTY * (width - cells)


class Gauge(Static):
    """One labelled bar with its value in words: `RAM needed   ████████░░░░  5.8 GB, 2.1 free of 15.4`."""

    DEFAULT_CSS = "Gauge { height: 1; }"

    def __init__(self, label: str, *, fraction: float = 0.0, text: str = "", colour: str = "green",
                 width: int = 24, label_width: int = 18, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.label_text = label
        self.bar_width = width
        self.label_width = label_width
        self.fraction = fraction
        self.value_text = text
        self.colour = colour
        self.set(fraction, text, colour)

    def set(self, fraction: float, text: str, colour: str = "green") -> None:
        self.fraction, self.value_text, self.colour = fraction, text, colour
        self.update(f"[dim]{escape(self.label_text):<{self.label_width}}[/dim] [{colour}]{bar(fraction, self.bar_width)}"
                    f"[/{colour}]  {escape(text)}")


STATE_ICON = {"pending": "[dim]○[/dim]", "running": "[cyan]◐[/cyan]", "done": "[green]✓[/green]",
              "failed": "[red]✗[/red]", "skipped": "[dim]–[/dim]"}


class StageList(Static):
    """The pipeline as a checklist: what is done, what runs now, what is still to come."""

    DEFAULT_CSS = "StageList { width: 28; height: auto; }"

    def __init__(self, names: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.names = list(names)
        self.states = {name: "pending" for name in self.names}
        self.details = {name: "" for name in self.names}
        self.render_all()

    def mark(self, name: str, state: str, detail: str = "") -> None:
        if name in self.states:
            self.states[name] = state
            self.details[name] = detail
            self.render_all()

    def reach(self, name: str) -> None:
        """Everything before `name` is done, `name` runs now."""
        if name not in self.states:
            return
        for earlier in self.names:
            if earlier == name:
                break
            if self.states[earlier] in ("pending", "running"):
                self.states[earlier] = "done"
        if self.states[name] != "done":
            self.states[name] = "running"
        self.render_all()

    def finish(self, ok: bool) -> None:
        """The job ended: the running stage is done or failed; stages never reached were skipped."""
        for name in self.names:
            if self.states[name] == "running":
                self.states[name] = "done" if ok else "failed"
            elif self.states[name] == "pending":
                self.states[name] = "skipped"
        self.render_all()

    @property
    def running(self) -> Optional[str]:
        return next((name for name in self.names if self.states[name] == "running"), None)

    def render_all(self) -> None:
        lines = []
        for name in self.names:
            detail = f"  [dim]{escape(self.details[name])}[/dim]" if self.details[name] else ""
            lines.append(f"{STATE_ICON[self.states[name]]} {escape(name)}{detail}")
        self.update("\n".join(lines))


class ResourceGraph(Vertical):
    """Three live graphs, one sample a second: the machine's CPU, memory in use, and the CPU the fork's own
    processes use (near zero while a model answers on the NPU)."""

    DEFAULT_CSS = """
    ResourceGraph { height: 4; width: 1fr; }
    ResourceGraph Horizontal { height: 1; }
    ResourceGraph .name { width: 10; color: $text-muted; }
    ResourceGraph Sparkline { width: 1fr; }
    ResourceGraph #npu-bar { width: 1fr; }
    ResourceGraph .value { width: 30; padding-left: 1; }
    """
    KEEP = 90
    SEGMENTS = 10  # the NPU row: lit segments are the share of the model's layers (or bytes) on the NPU
    PURPLE, GRAY = "#a970ff", "grey35"
    ROWS = (("cpu", "CPU"), ("memory", "memory"), ("fork", "fork CPU"))

    def __init__(self, total_gb: Optional[float], cores: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.total_gb = total_gb
        self.cores = max(1, cores)
        self.series: dict[str, list[float]] = {key: [] for key, _name in self.ROWS}
        self.last: Optional[tuple[float, Sample]] = None

    def compose(self) -> ComposeResult:
        for key, name in self.ROWS:
            with Horizontal():
                yield Label(name, classes="name")
                yield Sparkline([], summary_function=max, id=f"spark-{key}")
                yield Label("", classes="value", id=f"value-{key}")
        with Horizontal():
            yield Label("NPU load", classes="name")
            yield Static(self.segments(None), id="npu-bar")
            yield Label("no model loaded yet", classes="value", id="value-npu")

    @classmethod
    def segments(cls, fraction: Optional[float]) -> str:
        """Ten segments, purple for the share on the NPU (at least one when any is), grey for the rest."""
        lit = 0
        if fraction is not None and fraction > 0:
            lit = max(1, round(min(1.0, fraction) * cls.SEGMENTS))
        return " ".join([f"[{cls.PURPLE}]██[/]"] * lit + [f"[{cls.GRAY}]██[/]"] * (cls.SEGMENTS - lit))

    def set_npu(self, fraction: Optional[float], text: str) -> None:
        self.query_one("#npu-bar", Static).update(self.segments(fraction))
        self.query_one("#value-npu", Label).update(escape(text))

    def push(self, sample: Sample, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._add("cpu", sample.cpu_percent, f"{sample.cpu_percent:.0f}%")
        if self.total_gb:
            used = max(0.0, self.total_gb - sample.available_mb / 1024)
            self._add("memory", used, f"{used:.1f} of {self.total_gb:.1f} GB in use")
        else:
            self._add("memory", sample.available_mb / 1024, f"{sample.available_mb / 1024:.1f} GB free")
        cores = 0.0
        if self.last is not None:
            then, previous = self.last
            if now > then:
                cores = max(0.0, (sample.fork_seconds - previous.fork_seconds) / (now - then))
        self.last = (now, sample)
        who = f", {sample.fork_processes} procs" if sample.fork_processes else ", not running"
        self._add("fork", cores, f"{cores:.2f} of {self.cores} cores{who}")

    def unavailable(self, why: str) -> None:
        for key, _name in self.ROWS:
            self.query_one(f"#value-{key}", Label).update(f"[dim]{escape(why)}[/dim]")

    def _add(self, key: str, value: float, text: str) -> None:
        series = self.series[key]
        series.append(value)
        del series[:-self.KEEP]
        self.query_one(f"#spark-{key}", Sparkline).data = list(series)
        self.query_one(f"#value-{key}", Label).update(escape(text))


@dataclass
class Question:
    text: str
    hint: str  # what Enter alone means
    kind: str  # confirm | ask
    answered: threading.Event = field(default_factory=threading.Event)
    answer: str = ""
    abandoned: bool = False


class ChoiceBar(Vertical):
    """A question from the job, answered with a click or a key: Yes / No (Enter takes the default), or a text."""

    DEFAULT_CSS = """
    ChoiceBar { height: auto; display: none; border: round $accent; padding: 0 1; margin: 0 1; }
    ChoiceBar #question { height: auto; }
    ChoiceBar #choices { height: 3; }
    ChoiceBar Button { margin-right: 2; min-width: 8; }
    ChoiceBar #answer { width: 1fr; display: none; }
    """
    BINDINGS = [Binding("y", "answer('y')", "Yes", show=False), Binding("n", "answer('n')", "No", show=False)]

    class Answered(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.question: Optional[Question] = None

    def compose(self) -> ComposeResult:
        yield Static("", id="question")
        with Horizontal(id="choices"):
            yield Button("Yes", id="yes", variant="success")
            yield Button("No", id="no")
            yield Input(id="answer")

    def ask(self, question: Question) -> None:
        self.question = question
        confirm = question.kind == "confirm"
        self.query_one("#question", Static).update(
            f"[bold cyan]?[/bold cyan] {question.text}  [dim](Enter: {escape(question.hint)})[/dim]")
        for button_id in ("yes", "no"):
            self.query_one(f"#{button_id}", Button).display = confirm
        answer = self.query_one("#answer", Input)
        answer.display = not confirm
        answer.value = ""
        answer.placeholder = f"Enter = {question.hint}"
        self.display = True
        if confirm:
            self.query_one("#yes" if question.hint == "yes" else "#no", Button).focus()
        else:
            answer.focus()

    def hide(self) -> None:
        self.question = None
        self.display = False

    def action_answer(self, text: str) -> None:
        if self.question is not None and self.question.kind == "confirm":
            self.post_message(self.Answered(text))

    @on(Button.Pressed, "#yes")
    def _yes(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Answered("y"))

    @on(Button.Pressed, "#no")
    def _no(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Answered("n"))

    @on(Input.Submitted, "#answer")
    def _typed(self, event: Input.Submitted) -> None:
        event.stop()
        self.post_message(self.Answered(event.value))
