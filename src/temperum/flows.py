"""Flows shared by the command line and the TUI: run a GGUF conversion, then install the model into the
Ollama NPU fork, validate it through the fork, and upload it to ollama.com.

A Session is how a flow talks to the person: progress lines, things to show, questions. The command
line's interactive path uses ConsoleSession (Rich on stderr); scripts and `--json` use PlainSession,
which never asks (flags decide); the TUI brings its own (temperum.tui).
"""
from __future__ import annotations

import contextlib
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol, Sequence

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from temperum import settings, templates, ui
from temperum.core.errors import TemperumError

Transfer = tuple[Callable[[str, int, int], None], Callable[[], None]]


class Cancelled(Exception):
    """The person declined to go on."""


class Session(Protocol):
    """What a flow needs from its front end."""

    interactive: bool

    def say(self, message: str) -> None:
        """A progress line from the pipeline."""

    def note(self, text: str, warning: bool = False) -> None:
        """A line from the flow itself (plain text)."""

    def show(self, renderable: Any) -> None:
        """A Rich renderable: a plan table, a validation table."""

    def ready(self, summary: dict[str, Any]) -> None:
        """The result of a conversion."""

    def confirm(self, question: str, default: bool = True) -> bool: ...

    def ask(self, prompt: str, default: str) -> str: ...

    def open_url(self, url: str) -> None: ...

    def stages(self, estimate_minutes: float, probe: ui.BytesProbe | None = None,
               shares: Sequence[float] | None = None) -> contextlib.AbstractContextManager[Callable[[str], None]]:
        """A stage display for a conversion; yields the progress callback to give the pipeline. `shares`: this
        run's stage weights when they differ from the usual ones (a recipe search)."""

    def transfer(self, label: str) -> Transfer:
        """A byte-counting display for an upload: (update(status, completed, total), stop)."""


# -- sessions ---------------------------------------------------------------------------------------
class ConsoleSession:
    """Rich on a console (stderr): timestamped lines, panels, questions, a stage bar."""

    interactive = True

    def __init__(self, console: Console, start: float | None = None) -> None:
        self.console = console
        self.start = time.time() if start is None else start

    def say(self, message: str) -> None:
        minutes, seconds = divmod(int(time.time() - self.start), 60)
        self.console.print(f"[dim]{minutes:3d}:{seconds:02d}[/dim] {escape(message)}")

    def note(self, text: str, warning: bool = False) -> None:
        self.console.print(("  [yellow]![/yellow] " if warning else "  ") + escape(text))

    def show(self, renderable: Any) -> None:
        self.console.print(renderable)

    def ready(self, summary: dict[str, Any]) -> None:
        self.console.print(ui.ready_panel(summary))

    def confirm(self, question: str, default: bool = True) -> bool:
        return ui.confirm(self.console, question, default)

    def ask(self, prompt: str, default: str) -> str:
        return ui.ask(self.console, prompt, default)

    def open_url(self, url: str) -> None:
        webbrowser.open(url)

    @contextlib.contextmanager
    def stages(self, estimate_minutes: float, probe: ui.BytesProbe | None = None,
               shares: Sequence[float] | None = None) -> Iterator[Callable[[str], None]]:
        with ui.StageProgress(self.console, estimate_minutes, probe, shares=shares) as bar:
            yield bar.say

    def transfer(self, label: str) -> Transfer:
        from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

        progress = Progress(TextColumn("{task.description}"), BarColumn(bar_width=None), DownloadColumn(),
                            TransferSpeedColumn(), console=self.console, transient=True)
        progress.start()
        task = progress.add_task(label, total=None)

        def update(status: str, completed: int, total: int) -> None:
            progress.update(task, description=escape(status[:48]), completed=completed, total=total or None)
        return update, progress.stop


class PlainSession:
    """Scripts and `--json`: progress through a callback, no questions, results as lines on stderr."""

    interactive = False

    def __init__(self, say: Callable[[str], None], console: Console, quiet: bool = False) -> None:
        self._say = say
        self.console = console
        self.quiet = quiet  # --json: the document on stdout carries the result

    def say(self, message: str) -> None:
        self._say(message)

    def note(self, text: str, warning: bool = False) -> None:
        self.console.print(("  [yellow]![/yellow] " if warning else "  ") + escape(text))

    def show(self, renderable: Any) -> None:
        self.console.print(renderable)

    def ready(self, summary: dict[str, Any]) -> None:
        if self.quiet:
            return
        for line in ready_lines(summary):
            self.console.print(line)

    def confirm(self, question: str, default: bool = True) -> bool:
        return default

    def ask(self, prompt: str, default: str) -> str:
        return default

    def open_url(self, url: str) -> None:
        pass

    @contextlib.contextmanager
    def stages(self, estimate_minutes: float, probe: ui.BytesProbe | None = None,
               shares: Sequence[float] | None = None) -> Iterator[Callable[[str], None]]:
        yield self._say

    def transfer(self, label: str) -> Transfer:
        seen: set[str] = set()

        def update(status: str, _completed: int, _total: int) -> None:
            if status not in seen:
                seen.add(status)
                self._say(status)
        return update, lambda: None


# -- the GGUF conversion ------------------------------------------------------------------------------
def plan_gguf(model: str, *, target_id: str, quant: str, context: Optional[int], out_dir: Optional[Path],
              verify_level: str, overwrite: bool, python: Optional[Path], machine: Any = None,
              ollama_exe: Optional[Path] = None, recipe: Optional[str] = None) -> tuple[Any, Optional[Path]]:
    """Resolve the model and plan its GGUF conversion; also returns the fork executable in effect."""
    from temperum import gguf
    from temperum.host import detect_host
    from temperum.profiles import get_target
    from temperum.sources import parse_ref, resolve

    exe, _source = settings.ollama_exe(ollama_exe)
    plan = gguf.plan(resolve(parse_ref(model)), machine or detect_host(), get_target(target_id), ollama_exe=exe,
                     quant=quant, context=context, out=out_dir, verify=verify_level, overwrite=overwrite,
                     python=python, recipe=recipe)
    return plan, exe


def gguf_plan_table(plan: Any) -> Table:
    est = plan.estimates
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("Model", escape(f"{plan.model.ref.display}" + (f" @ {plan.model.commit[:12]}"
                                                                  if plan.model.commit else "")))
    table.add_row("Architecture", escape(f"{plan.model.architecture}, ~{est['params_billions']}B params"))
    table.add_row("Licence", escape(plan.model.license or "unknown") + (" (gated)" if plan.model.gated else ""))
    table.add_row("Target", escape(f"{plan.target.id}: {plan.target.description}"))
    table.add_row("Runtime", escape(f"{plan.runtime.description} ({plan.payload.root})"))
    table.add_row("NPU libraries", escape(str(plan.payload.htp)))
    table.add_row("Format", escape(f"GGUF {plan.quant}, context {plan.context}"
                                   + (", recipe searched on the NPU" if plan.recipe else "")))
    table.add_row("Chat", escape(templates.describe(plan.model.chat_template)))
    table.add_row("Verification", escape(plan.verify_level))
    table.add_row("Output", escape(str(plan.destination)))
    table.add_row("Estimate", escape(f"~{est['minutes']:.0f} min, ~{est['ram_gb']} GB RAM, ~{est['disk_gb']} GB "
                                     f"disk, {est['gguf_gb']} GB file"))
    return table


def run_gguf(session: Session, plan: Any, *, start: float, verifier: Any = None) -> tuple[dict[str, Any], Any]:
    """Run a planned GGUF conversion with the session's stage display; shows the result and returns it."""
    from temperum import gguf, paths
    from temperum.gguf_verify import verify_gguf

    probe = None
    if session.interactive:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()  # the stage display shows the download instead of the hub's own bars
        snapshot = paths.sources_dir() / str(plan.model.ref.repo).replace("/", "--") / str(plan.model.commit)
        probe = ui.snapshot_probe(snapshot, int(float(plan.estimates.get("download_gb") or 0) * 1e9))
    shares = None
    if getattr(plan, "recipe", None):
        from temperum import optimize

        search = optimize.estimate_minutes(float(plan.estimates.get("params_billions") or 0))
        shares = ui.shares_with_search(float(plan.estimates["minutes"]), search)
    with session.stages(float(plan.estimates["minutes"]), probe, shares) as say:
        result = gguf.run(plan, progress=say, verifier=verifier or verify_gguf)
    summary = gguf_summary(result, start)
    session.ready(summary)
    return summary, result


def gguf_summary(result: Any, start: float) -> dict[str, Any]:
    report = result.verification
    summary: dict[str, Any] = {"bundle": str(result.bundle), "verdict": report.get("verdict"),
                               "reasons": report.get("reasons", []), "job": str(result.job_dir),
                               "minutes": round((time.time() - start) / 60, 1),
                               "npu_placement": report.get("npu_placement", {}), "speed": report.get("speed", {})}
    ppl = report.get("quality", {}).get("perplexity", {})
    if ppl:
        summary["perplexity_ratio"] = ppl.get("ratio")
    summary["base_model"] = report.get("template", {}).get("modelfile") == "raw"
    recipe = getattr(result, "recipe", None)
    if recipe:
        chosen = recipe.get("chosen", {})
        summary["recipe"] = {"label": chosen.get("label"), "name": chosen.get("name"), "why": chosen.get("why"),
                             "trials": len(recipe.get("trials", []))}
    return summary


def ready_lines(summary: dict[str, Any]) -> list[str]:
    """The result of a GGUF conversion as log lines (Rich markup)."""
    colour = {"passed": "green", "warning": "yellow"}.get(str(summary["verdict"]), "red")
    lines = [f"[bold {colour}]{escape(str(summary['verdict']))}[/bold {colour}] "
             f"{escape(str(summary['bundle']))} ({summary['minutes']} min)"]
    lines += [f"  - {escape(str(reason))}" for reason in summary.get("reasons", [])]
    placed = summary.get("npu_placement") or {}
    if placed.get("layers_total"):
        lines.append(f"  NPU: {placed.get('layers_offloaded')}/{placed.get('layers_total')} layers, "
                     f"{placed.get('htp_weight_mib')} MiB of weights on HTP")
        if placed.get("ops"):
            from temperum import schedlog

            lines.append(f"  ops: {escape(schedlog.describe(placed['ops']))}")
    if "perplexity_ratio" in summary:
        lines.append(f"  perplexity vs the Q8_0 conversion on the CPU: {summary['perplexity_ratio']}x")
    speed = summary.get("speed") or {}
    if speed.get("median_tokens_per_s"):
        prompt = (f"{speed['long_prompt_tokens_per_s']:.0f} tokens/s ({speed['long_prompt_tokens']}-token prompt)"
                  if speed.get("long_prompt_tokens_per_s") else
                  f"{speed.get('median_prompt_tokens_per_s') or 0:.0f} tokens/s")
        lines.append(f"  decode: {speed['median_tokens_per_s']:.0f} tokens/s, prompt: {prompt}")
    if summary.get("base_model"):
        lines.append("  a base model with no chat template: installed as a completion model, the prompt is sent as "
                     "it is")
    recipe = summary.get("recipe")
    if recipe:
        lines.append(f"  recipe: {escape(str(recipe.get('label')))} after {recipe.get('trials')} trials on the NPU: "
                     f"{escape(str(recipe.get('why')))}")
    return lines


def gguf_installer(session: Session, result: Any, tag: str, exe: Optional[Path],
                   overwrite: bool = False) -> Callable[[str], dict[str, Any]]:
    """What `after_convert` calls to install a GGUF bundle into the fork at an endpoint. `overwrite`: the
    conversion replaced its bundle, so the tag of the same name is replaced too (a rebuild, not a clash)."""
    from temperum.install.ollama_import import install_gguf

    def installer(endpoint: str) -> dict[str, Any]:
        if exe is None:
            raise TemperumError("import", "EXE_REQUIRED", "Installing needs the fork executable.",
                                hint="Run `temperum setup --ollama-exe PATH` once, or pass --ollama-exe PATH.")
        return install_gguf(result.bundle, tag, executable=exe, host=endpoint, overwrite=overwrite,
                            work_dir=result.job_dir / "install", say=session.say)
    return installer


# -- after the conversion: install, validate, upload --------------------------------------------------
def placement_text(placed: dict[str, Any], cpu: str = "on the CPU") -> str:
    return {"npu": f"whole on the NPU ({placed.get('npu_mib')} MiB)",
            "split": f"{placed.get('npu_mib')} of {placed.get('size_mib')} MiB on the NPU",
            "cpu": cpu}.get(str(placed.get("where")), "")


def note_install(session: Session, installed: dict[str, Any]) -> None:
    where = placement_text(installed.get("placement", {}))
    session.note(f"installed as {installed['name']}" + (f"; the fork runs it {where}" if where else "")
                 + f": ollama run {installed['name']}")
    for warning in installed.get("warnings", []):
        session.note(str(warning), warning=True)


def after_convert(session: Session, summary: dict[str, Any], *, tag: str,
                  installer: Callable[[str], dict[str, Any]], install: bool, validate: bool, push: bool,
                  push_as: Optional[str], exe: Optional[Path], host: Optional[str]) -> None:
    """Install into the fork, validate through it, upload to ollama.com: asked when the session is
    interactive, else by flag."""
    from temperum import paths
    from temperum.install import fork
    from temperum.install.validate import validate as validate_model

    interactive = session.interactive
    if interactive and not install:
        if exe is None:
            session.note(f"To install it into the fork, set {fork.EXE_SETTING} to the fork's ollama.exe and run: "
                         f"temperum install --input {summary['bundle']} --name {tag} --execute")
            return
        install = session.confirm(f"Install it into the Ollama NPU fork as [bold]{escape(tag)}[/bold]?", default=True)
    if not (install or validate or push):
        return
    endpoint = fork.endpoint_for(host)
    if interactive and fork.version(endpoint) is None:
        if exe is None or not session.confirm(f"No server answers at {endpoint}. Start the fork's server now, with "
                                              "the NPU switch on?", default=True):
            session.note(f"Start the fork's server (OLLAMA_HEXAGON=1, then ollama serve) and run: temperum install "
                         f"--input {summary['bundle']} --name {tag} --execute")
            return
    info = fork.ensure(endpoint, exe, session.say, log_dir=paths.home() / "logs")
    if info.started:
        summary["server"] = {"endpoint": endpoint, "pid": info.pid, "log": str(info.log), "started": True}
        session.note(f"started the fork's server at {endpoint} (pid {info.pid}); it keeps running after temperum "
                     "exits")
    elif not info.is_fork:
        session.note(f"the server at {endpoint} is version {info.version}, which does not look like the NPU fork",
                     warning=True)
    if install:
        summary["install"] = installer(endpoint)
        note_install(session, summary["install"])
        if interactive and exe is not None and summary["install"].get("placement", {}).get("where") == "cpu":
            moved = offer_switch(session, endpoint, exe, tag)
            if moved is not None:
                summary["install"]["placement"] = moved
    if interactive and not validate:
        validate = session.confirm("Validate it through the fork (load it, time a generation, watch the CPU)?",
                                   default=True)
    if validate:
        report = validate_model(endpoint, tag, fork_dir=exe.parent if exe else None, say=session.say,
                                completion=summary.get("base_model"))
        summary["validation"] = report
        session.show(ui.validation_table(report))
    if interactive and not push:
        push = session.confirm("Upload it to ollama.com, to share it and pull it on your other devices?",
                               default=False)
    if push:
        summary["push"] = push_flow(session, endpoint, tag, push_as)


def offer_switch(session: Session, endpoint: str, exe: Path, tag: str) -> Optional[dict[str, Any]]:
    """The running server put the model on the CPU: when it is the fork's own, offer to restart it with the
    NPU switch on, then look again."""
    from temperum import paths
    from temperum.install import fork
    from temperum.install.ollama_import import npu_placement

    found = fork.listener(endpoint)
    if found is None or not fork.same_folder(found[1], exe):
        session.note(f"The server at {endpoint} was not started from the fork at {exe}, so temperum leaves it "
                     f"alone. Stop it, then run `temperum validate {tag}`: temperum starts the fork's server with "
                     "the NPU switch on.")
        return None
    if not session.confirm(f"That server (pid {found[0]}, the fork at {escape(str(found[1]))}) runs without the NPU "
                           "switch. Stop it and start one with the switch on?", default=True):
        return None
    fork.restart_with_switch(endpoint, exe, session.say, log_dir=paths.home() / "logs")
    session.say("loading it again to see where the fork runs it now")
    placed = npu_placement(endpoint, tag)
    where = placement_text(placed, cpu="still on the CPU: check the NPU libraries with `temperum doctor`") or "unknown"
    session.note(f"the fork now runs it {where}")
    return placed


def push_flow(session: Session, endpoint: str, tag: str, push_as: Optional[str]) -> dict[str, Any]:
    """Upload through the fork: sign in if needed, choose the ollama.com name, copy, push with progress."""
    from temperum.install import push as pushing
    from temperum.install.ollama_import import _canonical_tag

    while True:
        try:
            owner = pushing.account(endpoint)
            break
        except pushing.SigninRequired as exc:
            if not session.interactive:
                raise
            session.note(f"The fork's server is not linked to an ollama.com account yet. Sign in at: {exc.url}")
            session.open_url(exc.url)
            if not session.confirm("Linked? Check again", default=True):
                raise Cancelled() from None
    remote = pushing.check_remote_name(push_as or pushing.remote_name(owner, tag))
    if session.interactive and push_as is None:
        remote = pushing.check_remote_name(session.ask("Upload as", default=remote))
    if remote != _canonical_tag(tag):
        session.say(f"naming it {remote}")
        pushing.copy(endpoint, tag, remote)
    session.say(f"uploading {remote} to ollama.com through the fork")
    update, stop = session.transfer("uploading")
    try:
        result = pushing.push(endpoint, remote, progress=update)
    finally:
        stop()
    session.note(f"uploaded: {result['url']}; pull it elsewhere with: ollama pull {remote}")
    return result
