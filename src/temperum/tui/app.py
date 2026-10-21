"""The full-screen interface: a transcript, a prompt that takes a model id or a /command, questions
answered in the prompt, and a stage bar. `temperum` with no arguments opens it in a terminal.

A job (a conversion, a validation, an upload) runs in a worker thread through a TuiSession, the
flows.Session of this front end. Widgets are touched only from the app's thread, so every update
crosses over with call_from_thread, and a question blocks the worker until the person answers.
"""
from __future__ import annotations

import contextlib
import io
import shlex
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.timer import Timer
from textual.widgets import Input, Label, ProgressBar, RichLog, Static

from temperum import __version__, flows, settings, ui
from temperum.core.errors import TemperumError

_SCALE = 1000
PROMPT_HINT = "Hugging Face model id to convert, or /help"
COMMANDS: tuple[tuple[str, str], ...] = (
    ("/convert MODEL [--quant Q4_0|Q8_0|Q4_K_M] [--context N] [--name TAG] [--verify LEVEL] [--overwrite]",
     "convert a Hugging Face model (a bare model id does the same), then install, validate, upload"),
    ("/install BUNDLE [--name TAG]", "install a converted bundle into the fork, then validate, upload"),
    ("/validate TAG", "load a model through the fork: where it runs, speed, CPU use"),
    ("/push TAG [--as NAME]", "upload an installed model to ollama.com through the fork"),
    ("/doctor", "this machine, the fork, the toolchains"),
    ("/models", "conversions verified on hardware"),
    ("/config [KEY [VALUE]]", "what temperum remembers: ollama-exe, target, host"),
    ("/help", "this list"),
    ("/quit", "leave (Ctrl+C); a running job stops at its current step"),
)

Job = Callable[["TuiSession"], None]


class UsageError(Exception):
    """A command was typed wrongly; the message is the usage."""


@dataclass
class Question:
    text: str
    hint: str  # what Enter alone means
    kind: str  # confirm | ask
    answered: threading.Event = field(default_factory=threading.Event)
    answer: str = ""
    abandoned: bool = False


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

    def note(self, text: str, warning: bool = False) -> None:
        self._ui(self.app.write, ("  [yellow]![/yellow] " if warning else "  ") + escape(text))

    def show(self, renderable: Any) -> None:
        self._ui(self.app.write, renderable)

    def ready(self, summary: dict[str, Any]) -> None:
        self._ui(self.app.write, ui.ready_panel(summary))

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
        question.answered.wait()
        if question.abandoned:
            raise flows.Cancelled()
        return question.answer

    def open_url(self, url: str) -> None:
        webbrowser.open(url)

    @contextlib.contextmanager
    def stages(self, estimate_minutes: float, probe: ui.BytesProbe | None = None) -> Iterator[Callable[[str], None]]:
        tracker = ui.StageTracker(estimate_minutes, probe)
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


class TemperumApp(App[None]):
    TITLE = "Temperum"
    CSS = """
    Screen { layout: vertical; }
    #title { height: 1; background: $primary; color: $text; padding: 0 1; }
    #transcript { height: 1fr; padding: 0 1; }
    #stage { height: 1; display: none; padding: 0 1; }
    #stage-label { width: auto; }
    #bar { width: 1fr; }
    #stage-eta { width: auto; color: $text-muted; }
    #prompt { border: tall $primary; }
    #status { height: 1; color: $text-muted; padding: 0 1; }
    """
    BINDINGS = [Binding("ctrl+c", "leave", "Quit", priority=True, show=False),
                Binding("ctrl+q", "leave", "Quit", priority=True, show=False),
                Binding("f1", "help", "Help", show=False)]

    def __init__(self) -> None:
        super().__init__()
        self.pending: Optional[Question] = None
        self.busy: Optional[str] = None  # the running job's label
        self.history: list[str] = []  # plain text of everything in the transcript
        self.fork_state = "checking"
        self._tracker: Optional[ui.StageTracker] = None
        self._timer: Optional[Timer] = None
        self._leave_asked = 0.0

    # -- layout ------------------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static(f" Temperum {__version__}", id="title")
        yield RichLog(id="transcript", markup=True, wrap=True, highlight=False)
        with Horizontal(id="stage"):
            yield Label("", id="stage-label")
            yield ProgressBar(total=_SCALE, show_eta=False, id="bar")
            yield Label("", id="stage-eta")
        yield Input(placeholder=PROMPT_HINT, id="prompt")
        yield Static("", id="status")

    def on_mount(self) -> None:
        self.write(f"[bold]Temperum[/bold] {__version__}: convert a Hugging Face model for this machine's NPU, install "
                   "it into the Ollama NPU fork, validate it there, share it on ollama.com.")
        self.write("Type a model id (for example [bold]Qwen/Qwen2.5-0.5B-Instruct[/bold]) and press Enter, or /help.")
        self.refresh_status()
        self.query_one("#prompt", Input).focus()
        self.run_worker(self._status_job, thread=True, group="status", exit_on_error=False)

    def _status_job(self) -> None:
        """The fork, the target and the switch, found in a worker so the prompt is usable at once."""
        from temperum.cli.main import _fork_status
        from temperum.host import HEXAGON_OPT_IN, hexagon_opted_in

        status = _fork_status()
        target = settings.resolve("target", None, "auto").value
        host = settings.resolve("host", None, flows_default_host()).value
        lines = [("fork: " + (f"{escape(str(status['exe']))}: {escape(status['state'])}" if status.get("exe")
                              else "not set") + (f" ({escape(str(status['hint']))})" if status.get("hint") else "")),
                 f"target: {escape(str(target))}, server: {escape(str(host))}, home: {escape(str(home_dir()))}",
                 f"NPU switch: {HEXAGON_OPT_IN} is " + ("set here" if hexagon_opted_in() else
                                                        "not set here (temperum sets it when it starts the server)")]
        self.fork_state = status["state"]
        self.call_from_thread(self.write, "[dim]" + "\n".join(lines) + "[/dim]")
        self.call_from_thread(self.refresh_status)

    def refresh_status(self) -> None:
        running = f"[b]{escape(self.busy)}[/b]  " if self.busy else ""
        self.query_one("#status", Static).update(f" {running}fork {escape(self.fork_state)}  ·  F1 help  ·  Ctrl+C quit")

    # -- the transcript ----------------------------------------------------------------------------
    def write(self, renderable: Any) -> None:
        self.query_one("#transcript", RichLog).write(renderable)
        if isinstance(renderable, str):
            self.history.append(Text.from_markup(renderable).plain)
        else:
            console = Console(file=io.StringIO(), width=100, force_terminal=False)
            console.print(renderable)
            self.history.append(str(console.file.getvalue()))  # type: ignore[attr-defined]

    # -- the prompt --------------------------------------------------------------------------------
    @on(Input.Submitted, "#prompt")
    def submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if self.pending is not None:
            question, self.pending = self.pending, None
            event.input.placeholder = PROMPT_HINT
            self.write(f"  [dim]answer:[/dim] {escape(text) if text else escape(question.hint) + ' [dim](default)[/dim]'}")
            question.answer = text
            question.answered.set()
            return
        if not text:
            return
        self.write(f"[bold green]>[/bold green] {escape(text)}")
        self.dispatch(text)

    def pose(self, question: Question) -> None:
        self.pending = question
        self.write(f"[bold cyan]?[/bold cyan] {question.text}")
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = (f"y / n, Enter = {question.hint}" if question.kind == "confirm"
                              else f"Enter = {question.hint}")
        prompt.focus()

    def dispatch(self, text: str) -> None:
        try:
            words = shlex.split(text, posix=False)  # keeps backslashes in Windows paths
        except ValueError as exc:
            self.write(f"[red]{escape(str(exc))}[/red]")
            return
        if not words:
            return
        command, args = (words[0].lower(), words[1:]) if words[0].startswith("/") else ("/convert", words)
        if command in ("/quit", "/exit"):
            self.action_leave(force=True)
            return
        if command == "/help":
            self.action_help()
            return
        handler = {"/convert": self.cmd_convert, "/install": self.cmd_install, "/validate": self.cmd_validate,
                   "/push": self.cmd_push, "/doctor": self.cmd_doctor, "/models": self.cmd_models,
                   "/config": self.cmd_config}.get(command)
        if handler is None:
            self.write(f"[red]unknown command {escape(command)}[/red]; /help lists them")
            return
        if self.busy:
            self.write(f"[yellow]{escape(self.busy)} is running; wait for it to finish.[/yellow]")
            return
        try:
            label, job = handler(args)
        except UsageError as exc:
            self.write(f"[yellow]usage: {escape(str(exc))}[/yellow]")
            return
        self.run_job(label, job)

    # -- jobs --------------------------------------------------------------------------------------
    def run_job(self, label: str, job: Job) -> None:
        self.busy = label
        self.refresh_status()
        self.run_worker(lambda: self._job(job), thread=True, exclusive=True, group="job", exit_on_error=False)

    def _job(self, job: Job) -> None:  # the worker thread
        session = TuiSession(self)
        try:
            job(session)
        except flows.Cancelled:
            session._ui(self.write, "[dim]cancelled[/dim]")
        except TemperumError as exc:
            code = escape(f"[{exc.stage}/{exc.reason_code}]")
            session._ui(self.write, f"[bold red]Error[/bold red] {code}: {escape(str(exc))}"
                        + (f"\n[yellow]Hint:[/yellow] {escape(exc.hint)}" if exc.hint else ""))
        except Exception as exc:  # noqa: BLE001  (a job must never take the interface down)
            session._ui(self.write, f"[bold red]{escape(type(exc).__name__)}[/bold red]: {escape(str(exc))}")
        finally:
            session._ui(self._job_done)

    def _job_done(self) -> None:
        self.busy = None
        self.refresh_status()

    # -- commands: each returns a label and the job to run in the worker ---------------------------
    def cmd_convert(self, args: list[str]) -> tuple[str, Job]:
        positional, opts = _options(args, {"--overwrite"},
                                    {"--quant", "--context", "--name", "--verify", "--out", "--target", "--host"})
        if len(positional) != 1:
            raise UsageError(COMMANDS[0][0])
        model = positional[0]

        def job(session: TuiSession) -> None:
            from temperum.host import detect_host, resolve_target
            from temperum.profiles import get_target

            target = str(settings.resolve("target", opts.get("--target"), "auto").value)
            machine = None
            if target == "auto":
                machine = detect_host()
                target = resolve_target(machine).id
            if get_target(target).format != "gguf":
                session.note(f"this interface converts GGUF targets; for a QNN bundle run: temperum convert {model} "
                             f"--target {target}", warning=True)
                return
            start = time.time()
            plan, exe = flows.plan_gguf(model, target_id=target, quant=opts.get("--quant", "Q4_0"),
                                        context=int(opts["--context"]) if "--context" in opts else None,
                                        out_dir=Path(opts["--out"]) if "--out" in opts else None,
                                        verify_level=opts.get("--verify", "standard"),
                                        overwrite="--overwrite" in opts, python=None, machine=machine)
            session.show(flows.gguf_plan_table(plan))
            for warning in plan.warnings:
                session.note(str(warning), warning=True)
            if not session.confirm("Proceed?"):
                raise flows.Cancelled()
            summary, result = flows.run_gguf(session, plan, start=start)
            if summary["verdict"] == "failed":
                return
            tag = opts.get("--name") or f"{plan.model.name}:npu"
            flows.after_convert(session, summary, tag=tag, install=False, validate=False, push=False, push_as=None,
                                exe=exe, host=opts.get("--host"),
                                installer=flows.gguf_installer(session, result, tag, exe))
        return f"converting {model}", job

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
                self.call_from_thread(self.run_worker, self._status_job, thread=True, group="status",
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

    # -- actions -----------------------------------------------------------------------------------
    def action_help(self) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        for usage, meaning in COMMANDS:
            table.add_row(escape(usage), meaning)
        self.write(table)

    def action_leave(self, force: bool = False) -> None:
        if self.busy and not force and time.time() - self._leave_asked > 3:
            self._leave_asked = time.time()
            self.write(f"[yellow]{escape(self.busy)} is running. Press Ctrl+C again within 3 s to quit anyway; the "
                       "job stops at its current step.[/yellow]")
            return
        if self.pending is not None:
            self.pending.abandoned = True
            self.pending.answered.set()
        self.exit()

    # -- the stage bar -----------------------------------------------------------------------------
    def begin_stages(self, tracker: ui.StageTracker) -> None:
        self._tracker = tracker
        self.query_one("#stage").display = True
        self._tick()
        self._timer = self.set_interval(0.5, self._tick)

    def _tick(self) -> None:
        tracker = self._tracker
        if tracker is None:
            return
        self.query_one("#bar", ProgressBar).update(total=_SCALE, progress=tracker.completed() * _SCALE)
        self.query_one("#stage-label", Label).update(f"[bold]{tracker.stage_name}[/bold] {escape(tracker.short())} ")
        self.query_one("#stage-eta", Label).update(f" {tracker.eta_text()}")

    def end_stages(self, tracker: ui.StageTracker, ok: bool) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._tracker = None
        if ok:
            self.query_one("#bar", ProgressBar).update(total=_SCALE, progress=_SCALE)
            self.query_one("#stage-label", Label).update(f"[bold]done[/bold] in {tracker.elapsed_text()} ")
        else:
            self.query_one("#stage-label", Label).update(f"[bold red]stopped[/bold red] {escape(tracker.short())} ")
        self.query_one("#stage-eta", Label).update("")
        self.set_timer(8, self._hide_stage)

    def _hide_stage(self) -> None:
        if self._tracker is None:
            self.query_one("#stage").display = False

    def begin_transfer(self, label: str) -> None:
        self.query_one("#stage").display = True
        self.query_one("#stage-label", Label).update(f"[bold]{escape(label)}[/bold] ")
        self.query_one("#stage-eta", Label).update("")
        self.query_one("#bar", ProgressBar).update(total=None)

    def update_transfer(self, status: str, completed: int, total: int) -> None:
        self.query_one("#stage-label", Label).update(f"[bold]{escape(status[:48])}[/bold] ")
        if total:
            self.query_one("#bar", ProgressBar).update(total=total, progress=completed)
            self.query_one("#stage-eta", Label).update(f" {completed / 2**20:.0f} / {total / 2**20:.0f} MiB")

    def end_transfer(self) -> None:
        self.query_one("#stage").display = False
        self.query_one("#bar", ProgressBar).update(total=_SCALE, progress=0)


def flows_default_host() -> str:
    from temperum.install import fork

    return fork.DEFAULT_HOST


def home_dir() -> Path:
    from temperum import paths

    return paths.home()


def run() -> None:
    TemperumApp().run()
