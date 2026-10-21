"""Console pieces for an interactive run: a stage bar against the plan's estimate, panels and questions.

Scripts keep the timestamped lines of cli.main._progress (and `--json` its events); these are used only
when the run is interactive: a terminal on both ends and neither `--yes` nor `--json`.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Stages in the order the pipeline reaches them, the message prefixes that announce each, and the share of
# the estimated time each usually takes (the GGUF route; QNN messages map onto the same stages). The bar
# advances by stage and, within a stage, by elapsed time against that share, so it moves steadily without
# claiming more than the pipeline has done.
STAGES: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("preparing", ("creating ", "setting up", "installing python", "checking ", "first use"), 0.06),
    ("downloading", ("fetching ", "downloading"), 0.18),
    ("converting", ("converting", "quantizing", "pass ", "running olive", "compiling", "exporting"), 0.26),
    ("assembling", ("assembling",), 0.02),
    ("verifying", ("reading the source", "loading the bundle", "prompt ", "long prompt", "measuring",
                   "running the source", "comparing", "verifying"), 0.42),
    ("installing", ("installing into", "loading it once"), 0.06),
)
_SCALE = 1000
_MESSAGE_WIDTH = 64

BytesProbe = Callable[[], Optional[tuple[int, int]]]


def stage_of(message: str) -> int | None:
    """The index of the stage a progress message announces, or None for a message within the current stage."""
    lowered = message.strip().lower()
    for index, (_name, prefixes, _share) in enumerate(STAGES):
        if lowered.startswith(prefixes):
            return index
    return None


def directory_bytes(path: Path) -> int:
    """Bytes under a folder (a download in progress), ignoring what disappears while counting."""
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def snapshot_probe(snapshot: Path, expected_bytes: int) -> BytesProbe:
    """Download progress for the stage bar: bytes landed in the snapshot folder against the plan's estimate."""
    def probe() -> tuple[int, int] | None:
        if expected_bytes <= 0:
            return None
        return min(directory_bytes(snapshot), expected_bytes), expected_bytes
    return probe


class StageTracker:
    """The arithmetic behind a stage display: feed it the pipeline's messages, read a fraction and an ETA.
    Front ends (the Rich bar below, the TUI's bar) render it."""

    def __init__(self, estimate_minutes: float, bytes_probe: BytesProbe | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.estimate_s = max(60.0, float(estimate_minutes) * 60)
        self.bytes_probe = bytes_probe
        self.clock = clock
        self.started = clock()
        self.stage = -1
        self.stage_started = self.started
        self.message = "starting"
        self._lock = threading.Lock()

    # -- the callback -----------------------------------------------------------------------------
    def say(self, message: str) -> None:
        with self._lock:
            index = stage_of(message)
            if index is not None and index > self.stage:
                self.stage, self.stage_started = index, self.clock()
            self.message = message

    @property
    def stage_name(self) -> str:
        with self._lock:
            return STAGES[self.stage][0] if self.stage >= 0 else "starting"

    # -- arithmetic ------------------------------------------------------------------------------
    def completed(self) -> float:
        """Fraction done: the stages passed, plus the current stage's share scaled by its elapsed time."""
        with self._lock:
            stage, stage_started = self.stage, self.stage_started
        if stage < 0:
            return 0.0
        base = sum(share for _n, _p, share in STAGES[:stage])
        name, _prefixes, share = STAGES[stage]
        within = None
        if name == "downloading" and self.bytes_probe is not None:
            probed = self.bytes_probe()
            if probed and probed[1] > 0:
                within = probed[0] / probed[1]
        if within is None:
            budget = max(1.0, share * self.estimate_s)
            within = (self.clock() - stage_started) / budget
        return min(1.0, base + share * max(0.0, min(within, 0.95)))

    def elapsed_text(self) -> str:
        minutes, seconds = divmod(int(self.clock() - self.started), 60)
        return f"{minutes}:{seconds:02d}"

    def eta_text(self) -> str:
        remaining = self.estimate_s - (self.clock() - self.started)
        if remaining > 90:
            return f"{self.elapsed_text()}, about {math.ceil(remaining / 60)} min left"
        if remaining > 0:
            return f"{self.elapsed_text()}, about a minute left"
        return f"{self.elapsed_text()}, longer than the estimate"

    def short(self) -> str:
        message = self.message.strip()
        return message if len(message) <= _MESSAGE_WIDTH else message[:_MESSAGE_WIDTH - 1] + "…"


class StageProgress(StageTracker):
    """A `say(message)` progress callback that drives a Rich bar; use as a context manager around the run."""

    def __init__(self, console: Console, estimate_minutes: float, bytes_probe: BytesProbe | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(estimate_minutes, bytes_probe, clock)
        self.console = console
        self._stop = threading.Event()
        self._progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(bar_width=None),
                                  TextColumn("[dim]{task.fields[eta]}[/dim]"), console=console, transient=False)
        self._task: TaskID | None = None
        self._ticker: threading.Thread | None = None

    def __enter__(self) -> StageProgress:
        self._progress.start()
        self._task = self._progress.add_task("starting", total=_SCALE, eta="")
        self._ticker = threading.Thread(target=self._tick, name="temperum-progress", daemon=True)
        self._ticker.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=2)
        if self._task is not None:
            if exc_type is None:
                self._progress.update(self._task, completed=_SCALE, description="[bold]done[/bold]",
                                      eta=f"{self.elapsed_text()}")
            else:
                self._progress.update(self._task, completed=self.completed() * _SCALE,
                                      description=f"[bold red]stopped[/bold red] {escape(self.short())}", eta="")
        self._progress.stop()

    def say(self, message: str) -> None:
        super().say(message)
        self._render()

    # -- rendering -------------------------------------------------------------------------------
    def _render(self) -> None:
        if self._task is None:
            return
        self._progress.update(self._task, completed=self.completed() * _SCALE,
                              description=f"[bold]{self.stage_name}[/bold] {escape(self.short())}",
                              eta=self.eta_text())

    def _tick(self) -> None:
        while not self._stop.wait(0.5):
            self._render()


# -- panels and tables ---------------------------------------------------------------------------
def ready_panel(summary: dict[str, Any]) -> Panel:
    """The result of a conversion: verdict, where the bundle is, how it did on the NPU."""
    verdict = str(summary.get("verdict"))
    colour = {"passed": "green", "warning": "yellow"}.get(verdict, "red")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("Bundle", escape(str(summary.get("bundle"))))
    reasons = [str(r) for r in summary.get("reasons") or []]
    grid.add_row("Verdict", f"[{colour}]{escape(verdict)}[/{colour}]"
                 + (": " + escape("; ".join(reasons)) if reasons else ""))
    placed = summary.get("npu_placement") or {}
    if placed.get("layers_total"):
        grid.add_row("NPU", escape(f"{placed.get('layers_offloaded')}/{placed.get('layers_total')} layers, "
                                   f"{placed.get('htp_weight_mib')} MiB of weights on HTP"))
    if summary.get("perplexity_ratio") is not None:
        grid.add_row("Quality", escape(f"perplexity {summary['perplexity_ratio']}x the reference"))
    speed = summary.get("speed") or {}
    if speed.get("median_tokens_per_s"):
        prompt = (f", prompt {speed['long_prompt_tokens_per_s']:.0f} tok/s ({speed['long_prompt_tokens']} tokens)"
                  if speed.get("long_prompt_tokens_per_s") else "")
        grid.add_row("Speed", escape(f"decode {speed['median_tokens_per_s']:.0f} tok/s{prompt}"))
    if summary.get("base_model"):
        grid.add_row("Chat", "none: a base model, installed as a completion model (the prompt is sent as it is)")
    if summary.get("minutes") is not None:
        grid.add_row("Took", escape(f"{summary['minutes']} min"))
    title = {"passed": "Model ready", "warning": "Model ready, with warnings"}.get(verdict, "Conversion failed")
    return Panel(grid, title=title, border_style=colour, expand=False)


def validation_table(report: dict[str, Any]) -> Table:
    """What `validate` measured through the fork."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    placed = report.get("placement") or {}
    where = {"npu": f"whole on the NPU ({placed.get('npu_mib')} MiB)",
             "split": f"{placed.get('npu_mib')} of {placed.get('size_mib')} MiB on the NPU, the rest on the CPU",
             "cpu": "on the CPU (the server runs without the NPU switch, or the model has no NPU route)",
             }.get(str(placed.get("where")), "unknown")
    table.add_row("Runs", escape(where))
    if report.get("load_s") is not None:
        table.add_row("Load", escape(f"{report['load_s']} s"))
    chat = report.get("chat") or {}
    if chat:
        answers = ", ".join(repr(a.strip()[:40]) for a in chat.get("answers", []))
        if chat.get("mode") == "completion":  # a base model: scored on how it continues a sentence
            ran_on = (f" ({chat['ran_on']} of {chat.get('of')} ran to the {chat.get('limit')}-token limit: a base "
                      "model does not stop by itself)" if chat.get("ran_on") else "")
            table.add_row("Completion", escape(f"{chat.get('correct')}/{chat.get('of')} continuations on topic: "
                                               f"{answers}{ran_on}"))
        else:
            ran_on = (f" ({chat['ran_on']} of {chat.get('of')} hit the {chat.get('limit')}-token limit: the model "
                      "does not stop by itself, so its answers run on)" if chat.get("ran_on") else "")
            table.add_row("Chat", escape(f"{chat.get('correct')}/{chat.get('of')} answers as expected: "
                                         f"{answers}{ran_on}"))
    speed = report.get("speed") or {}
    if speed.get("decode_tokens_per_s"):
        table.add_row("Speed", escape(f"decode {speed['decode_tokens_per_s']} tok/s, prompt "
                                      f"{speed.get('prompt_tokens_per_s') or 0} tok/s ({speed.get('tokens')} tokens)"))
    cpu = report.get("cpu") or {}
    if cpu.get("cores") is not None:
        table.add_row("CPU", escape(f"{cpu['cores']} cores busy while generating ({cpu.get('processes')} fork "
                                    "processes)"))
    elif cpu.get("skipped"):
        table.add_row("CPU", escape(f"not measured: {cpu['skipped']}"))
    return table


def confirm(console: Console, question: str, default: bool = True) -> bool:
    return bool(Confirm.ask(question, console=console, default=default))


def ask(console: Console, prompt: str, default: str) -> str:
    return str(Prompt.ask(prompt, console=console, default=default))
