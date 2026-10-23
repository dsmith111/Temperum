from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from temperum import flows, settings, ui
from temperum.core.errors import TemperumError

if TYPE_CHECKING:
    from temperum.host import HostInfo

# The hub's cache works without symlinks on Windows; its warning about that is just noise here.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

app = typer.Typer(add_completion=False, invoke_without_command=True,
                  help="Convert Hugging Face models to run on your machine's NPU, with Ollama integration. "
                       "With no command, in a terminal, opens the interactive interface.")
out = Console()
err = Console(stderr=True)


@app.callback()
def main(ctx: typer.Context) -> None:
    """`temperum` alone opens the full-screen interface (a terminal on both ends); otherwise the help."""
    if ctx.invoked_subcommand is not None:
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        from temperum.tui.app import run

        run()
        return
    typer.echo(ctx.get_help())


def _handle_err(e: Exception) -> None:
    if isinstance(e, TemperumError):
        code = escape(f"[{e.stage}/{e.reason_code}]")
        err.print(f"[bold red]Error[/bold red] {code}: {escape(str(e))}")
        if e.hint:
            err.print(f"[yellow]Hint:[/yellow] {escape(e.hint)}")
        raise typer.Exit(code=2)
    raise e


def _progress(start: float, as_json: bool = False) -> Any:
    def say(message: str) -> None:
        elapsed = time.time() - start
        if as_json:  # one JSON event per line on stderr; stdout carries the final document
            typer.echo(json.dumps({"event": "progress", "seconds": round(elapsed, 1), "message": message}), err=True)
            return
        minutes, seconds = divmod(int(elapsed), 60)
        err.print(f"[dim]{minutes:3d}:{seconds:02d}[/dim] {escape(message)}")
    return say


def _interactive(yes: bool, as_json: bool) -> bool:
    """Questions and the stage bar need a terminal on both ends and neither --yes nor --json."""
    return not yes and not as_json and sys.stdin.isatty() and sys.stderr.isatty()


def _session(start: float, yes: bool, as_json: bool) -> flows.Session:
    """How a flow talks to the person: Rich with questions in a terminal, plain lines (or JSON events) otherwise."""
    if _interactive(yes, as_json):
        return flows.ConsoleSession(err, start)
    return flows.PlainSession(_progress(start, as_json), err, quiet=as_json)


def _print_plan(plan: Any) -> None:
    model, est = plan.model, plan.estimates
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("Model", escape(f"{model.ref.display}" + (f" @ {model.commit[:12]}" if model.commit else "")))
    table.add_row("Architecture", escape(f"{model.architecture} ({model.model_type}), ~{est['params_billions']}B params"))
    table.add_row("Licence", escape(model.license or "unknown") + (" (gated)" if model.gated else ""))
    table.add_row("Recipe", escape(f"{plan.recipe.recipe.repo} ({plan.recipe.kind})"))
    table.add_row("Quantization", escape(f"{plan.quality} on {plan.device}"))
    table.add_row("Target", escape(f"{plan.target.id}: {plan.target.description}"))
    table.add_row("Runtime", escape(plan.runtime.description))
    table.add_row("Verification", escape(plan.verify_level))
    table.add_row("Output", escape(str(plan.destination)))
    if plan.resumed:
        table.add_row("Resuming", escape(f"already done: {', '.join(plan.resumed)}"))
    if "olive" not in plan.resumed:
        table.add_row("Estimate", escape(f"~{est['minutes']:.0f} min, ~{est['ram_gb']} GB RAM, ~{est['disk_gb']} GB "
                                         f"disk (download {est['download_gb']} GB)"))
    err.print(table)
    for warning in plan.warnings:
        err.print(f"[yellow]![/yellow] {escape(warning)}")


@app.command("convert")
def convert_command(
    model: str = typer.Argument(..., help="Hugging Face id (org/name[@revision]) or a local model folder"),
    target: Optional[str] = typer.Option(None, "--target", envvar="TEMPERUM_TARGET",
                               help="NPU target profile (auto = a GGUF for this machine's NPU; qnn-x-elite for a "
                                    "QNN bundle)"),
    quality: str = typer.Option("auto", "--quality", help="QNN: auto | standard (8-bit RTN) | recipe (GPTQ, CUDA)"),
    device: str = typer.Option("auto", "--device", help="QNN quantization device: auto | cpu | cuda"),
    context: Optional[int] = typer.Option(None, "--context", help="Declared context length (default: min(model, 4096))"),
    out_dir: Optional[Path] = typer.Option(None, "--out", help="Bundle folder (default under the temperum home)"),
    verify_level: str = typer.Option("standard", "--verify", help="none | quick | standard | full"),
    template: Optional[Path] = typer.Option(None, "--template", help="Ollama Go template file (overrides detection)"),
    stop: list[str] = typer.Option([], "--stop", help="Stop sequence for --template (repeatable)"),
    num_splits: Optional[int] = typer.Option(None, "--num-splits", help="Advanced: NPU graph splits"),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Keep Olive's intermediate cache"),
    overwrite: bool = typer.Option(False, "--overwrite",
                                   help="Replace an existing bundle folder, and the installed tag of that name"),
    python: Optional[Path] = typer.Option(None, "--python", help="Native Python for toolchains (default: uv)"),
    plan_only: bool = typer.Option(False, "--plan", help="Show the plan and stop"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Run without asking (accepts the plan and licence notice)"),
    as_json: bool = typer.Option(False, "--json", help="Print the plan/result as JSON on stdout"),
    install: bool = typer.Option(False, "--install", help="Also install into the Ollama fork (its server is started "
                                                        "if none runs)"),
    validate: bool = typer.Option(False, "--validate", help="With --install: load it through the fork and measure "
                                                          "where it runs, speed and CPU use"),
    push: bool = typer.Option(False, "--push", help="With --install: upload it to ollama.com through the fork"),
    push_as: Optional[str] = typer.Option(None, "--as", help="ollama.com name for --push (default: your account/"
                                                            "<name>:<tag>)"),
    name: Optional[str] = typer.Option(None, "--name", help="Ollama tag for --install (default: <model>:npu; "
                                                                    "<model>:npu-qnn for a QNN bundle)"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="Fork executable for --install"),
    host: Optional[str] = typer.Option(None, "--host", help="Fork endpoint http://127.0.0.1:PORT for --install "
                                                                "(or OLLAMA_HOST)"),
    quant: str = typer.Option("Q4_0", "--quant", help="GGUF type for --target gguf-hexagon: Q4_0 | Q8_0 | Q4_K_M"),
    recipe: Optional[str] = typer.Option(None, "--recipe", help="auto: search per-tensor quantization mixes on "
                                                                "the NPU and keep the best by the rule (GGUF)"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate",
                                   help="GGUF: quantize with an importance matrix computed first on the CPU (on by "
                                        "default when the fork's runtime ships llama-imatrix)"),
    draft: Optional[str] = typer.Option(None, "--draft", help="GGUF: a speculative-decoding draft (DFlash) to pair "
                                                             "with the model: converted with its metadata, bundled "
                                                             "and named in the Modelfile"),
    target_model: Optional[str] = typer.Option(None, "--target-model",
                                               help="GGUF, when MODEL is a draft: the target model it pairs with "
                                                    "(asked for otherwise)"),
) -> None:
    """Convert a model for this machine's NPU (resolve, convert, quantize, verify, optionally install)."""
    from temperum import convert
    from temperum.host import detect_host, resolve_target
    from temperum.profiles import get_target

    start = time.time()
    try:
        if (validate or push) and not install:
            raise TemperumError("import", "INSTALL_REQUIRED", "--validate and --push apply to the installed model.",
                                hint="Add --install.")
        if install:  # fail now, not after a long conversion
            _check_install_options(ollama_exe, host)
        target = str(settings.resolve("target", target, "auto").value)
        machine = None
        if target == "auto":
            machine = detect_host()
            target = resolve_target(machine).id
        if get_target(target).format == "gguf":
            qnn_only = [flag for flag, used in (("--template", template), ("--stop", stop), ("--num-splits", num_splits),
                                                ("--keep-cache", keep_cache), ("--quality", quality != "auto"),
                                                ("--device", device != "auto")) if used]
            if qnn_only:
                err.print(f"[yellow]![/yellow] {', '.join(qnn_only)} only apply to QNN bundles (--target qnn-x-elite); "
                          "ignored for a GGUF")
            _convert_gguf(model, target, quant=quant, context=context, out_dir=out_dir, verify_level=verify_level,
                          overwrite=overwrite, python=python, plan_only=plan_only, yes=yes, as_json=as_json,
                          install=install, name=name, ollama_exe=ollama_exe, host=host, start=start, machine=machine,
                          validate=validate, push=push, push_as=push_as, recipe=recipe, calibrate=calibrate,
                          draft=draft, target_model=target_model)
            return
        if draft or target_model:
            err.print("[yellow]![/yellow] --draft and --target-model pair a draft on the GGUF route only; ignored "
                      "for a QNN bundle")
        options = convert.ConvertOptions(model=model, target=target, quality=quality, device=device,
                                         context=context, out=out_dir, verify=verify_level, overwrite=overwrite,
                                         template=template, stops=list(stop), num_splits=num_splits,
                                         keep_cache=keep_cache, python=python)
        plan = convert.plan(options)
        if as_json:
            typer.echo(json.dumps({"event": "plan", "plan": plan.to_dict()}, default=str), err=True)
        else:
            _print_plan(plan)
        if plan_only:
            if as_json:
                typer.echo(json.dumps(plan.to_dict(), indent=2, default=str))
            return
        if not yes and not typer.confirm("Proceed?", default=True, err=True):
            raise typer.Exit(code=1)
        interactive = _interactive(yes, as_json)
        if interactive:
            with ui.StageProgress(err, float(plan.estimates["minutes"])) as stages:
                result = convert.run(plan, progress=stages.say)
        else:
            result = convert.run(plan, progress=_progress(start, as_json))
        report = result.verification
        summary: dict[str, Any] = {"bundle": str(result.bundle), "verdict": report.get("verdict"),
                                   "reasons": report.get("reasons", []), "job": str(result.job_dir),
                                   "minutes": round((time.time() - start) / 60, 1)}
        quality_report = report.get("quality", {})
        if quality_report.get("perplexity"):
            summary["perplexity_ratio"] = quality_report["perplexity"].get("ratio")
        summary["speed"] = report.get("speed", {})
        tag = name or f"{plan.model.name}:npu-qnn"
        if interactive:
            err.print(ui.ready_panel(summary))
        elif not as_json:
            colour = {"passed": "green", "warning": "yellow"}.get(str(summary["verdict"]), "red")
            err.print(f"[bold {colour}]{escape(str(summary['verdict']))}[/bold {colour}] "
                      f"{escape(str(result.bundle))} ({summary['minutes']} min)")
            for reason in summary["reasons"]:
                err.print(f"  - {escape(reason)}")
            if "perplexity_ratio" in summary:
                err.print(f"  perplexity vs source model: {summary['perplexity_ratio']}x")
            speed = summary["speed"]
            if speed.get("median_tokens_per_s"):
                err.print(f"  decode: {speed['median_tokens_per_s']:.0f} tokens/s, "
                          f"first token: {speed.get('median_ttft_s') or 0:.2f} s")
            if not install and summary["verdict"] != "failed":
                err.print(f"  install: temperum install --input {escape(str(result.bundle))} "
                          f"--name {escape(tag)} --ollama-exe FORK_EXE --host http://127.0.0.1:11434 "
                          "(review the plan, then add --execute --expect-fingerprint FP)")
        if summary["verdict"] == "failed":  # never install a bundle that failed verification
            if not as_json:
                err.print(f"  Olive's cache is kept in {escape(str(result.job_dir))}, so rerunning with --overwrite "
                          "(for example with --num-splits) reuses the quantization")
        else:
            flows.after_convert(_session(start, yes, as_json), summary, tag=tag, install=install, validate=validate,
                                push=push, push_as=push_as, exe=ollama_exe, host=host,
                                installer=lambda endpoint: _install(result.bundle, tag, ollama_exe, endpoint,
                                                                    result.job_dir, plan.host.processor_name))
        if as_json:
            typer.echo(json.dumps(summary, indent=2, default=str))
        if summary["verdict"] == "failed":
            raise typer.Exit(code=3)
    except flows.Cancelled:
        raise typer.Exit(code=1) from None
    except TemperumError as exc:
        _handle_err(exc)


def _convert_gguf(model: str, target_id: str, *, quant: str, context: Optional[int], out_dir: Optional[Path],
                  verify_level: str, overwrite: bool, python: Optional[Path], plan_only: bool, yes: bool,
                  as_json: bool, install: bool, name: Optional[str], ollama_exe: Optional[Path],
                  host: Optional[str], start: float, machine: Optional[HostInfo] = None,
                  validate: bool = False, push: bool = False, push_as: Optional[str] = None,
                  recipe: Optional[str] = None, calibrate: bool = True, draft: Optional[str] = None,
                  target_model: Optional[str] = None) -> None:
    """The GGUF target: llama.cpp's converter, the fork's llama-quantize, verified on the NPU."""
    ask = None if yes or as_json else (lambda question, default: str(typer.prompt(question, default=default,
                                                                                   show_default=bool(default), err=True)))
    plan, exe = flows.plan_gguf(model, target_id=target_id, quant=quant, context=context, out_dir=out_dir,
                                verify_level=verify_level, overwrite=overwrite, python=python, machine=machine,
                                ollama_exe=ollama_exe, recipe=recipe, calibrate=calibrate, draft=draft,
                                target_model=target_model, ask=ask)
    if as_json:
        typer.echo(json.dumps({"event": "plan", "plan": plan.to_dict()}, default=str), err=True)
    else:
        err.print(flows.gguf_plan_table(plan))
        for warning in plan.warnings:
            err.print(f"[yellow]![/yellow] {escape(warning)}")
    if plan_only:
        if as_json:
            typer.echo(json.dumps(plan.to_dict(), indent=2, default=str))
        return
    if not yes and not typer.confirm("Proceed?", default=True, err=True):
        raise typer.Exit(code=1)
    session = _session(start, yes, as_json)
    summary, result = flows.run_gguf(session, plan, start=start)
    tag = name or f"{plan.model.name}:npu"
    try:
        if summary["verdict"] != "failed":
            flows.after_convert(session, summary, tag=tag, install=install, validate=validate, push=push,
                                push_as=push_as, exe=exe, host=host,
                                installer=flows.gguf_installer(session, result, tag, exe, overwrite=overwrite))
    except flows.Cancelled:
        raise typer.Exit(code=1) from None
    if as_json:
        typer.echo(json.dumps(summary, indent=2, default=str))
    if summary["verdict"] == "failed":
        raise typer.Exit(code=3)


def _check_install_options(ollama_exe: Optional[Path], host: Optional[str]) -> None:
    from temperum.install import fork

    fork.endpoint_for(host)
    exe = fork.executable(ollama_exe)
    if exe is None:
        raise TemperumError("import", "EXE_REQUIRED", "--install needs the fork executable.",
                            hint="Run `temperum setup --ollama-exe PATH` once, or pass --ollama-exe PATH.")


def _install(bundle: Path, name: str, ollama_exe: Optional[Path], host: Optional[str], job_dir: Path,
             processor_name: Optional[str]) -> dict[str, Any]:
    from temperum.install.ollama_import import execute_import, plan_import

    plan = plan_import(bundle, name, executable=ollama_exe, host=host, processor_name=processor_name)
    work = job_dir / "install"
    work.mkdir(parents=True, exist_ok=True)
    return execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=work)


@app.command("setup")
def setup_command(
    target: Optional[str] = typer.Option(None, "--target", envvar="TEMPERUM_TARGET",
                                         help="What to prepare for: auto (a GGUF for this machine's NPU) or a "
                                              "target id such as qnn-x-elite"),
    device: str = typer.Option("auto", "--device", help="QNN: quantization device the quantize toolchain targets"),
    python: Optional[Path] = typer.Option(None, "--python", help="Native Python for toolchains (default: uv)"),
    check: bool = typer.Option(False, "--check", help="Only report status (deep check)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild environments even if they look ready"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="The fork's ollama.exe, remembered for "
                                                                        "every later command"),
) -> None:
    """Create (or check) what conversions need: for a GGUF the conversion toolchain and llama.cpp's converter;
    for QNN bundles also the compile and verify toolchains. Remembers the fork's executable."""
    from temperum.host import detect_host, resolve_target
    from temperum.profiles import DEFAULT_LLAMA_RUNTIME, DEFAULT_RUNTIME, LLAMA_RUNTIMES, get_runtime
    from temperum.toolchains import Toolchains
    from temperum.toolchains.llama_cpp import ConverterSource

    try:
        _remember_fork(ollama_exe, ask=not check)
        target = str(settings.resolve("target", target, "auto").value)
        host = detect_host()
        gguf = resolve_target(host, target).format == "gguf"
        chosen = "cpu" if gguf or device == "auto" and not host.cuda_gpus else ("cuda" if device == "auto" else device)
        chains = Toolchains(host, get_runtime(DEFAULT_RUNTIME), chosen, python=python,
                            progress=_progress(time.time()))
        roles = ["quantize"] if gguf else ["quantize", "compile"] + (["verify"] if host.native_arch == "arm64" else [])
        table = Table("role", "state", "path", "tested")
        for role in roles:
            status = chains.status(role, deep=True) if check else (
                chains.build(role) if rebuild else chains.ensure(role))
            table.add_row(role, status.state + (f" ({status.detail})" if status.detail else ""), str(status.path),
                          status.tested)
        if gguf:  # llama.cpp's converter, pinned to the fork's llama.cpp and checked file by file
            source = ConverterSource.load(LLAMA_RUNTIMES[DEFAULT_LLAMA_RUNTIME].llama_cpp_tag)
            if not check:
                source.ensure(progress=_progress(time.time()))
            problems = source.problems()
            table.add_row("llama.cpp converter", "ready" if not problems else f"missing ({len(problems)} files)",
                          str(source.directory), source.tag)
        fork_status = _fork_status()
        table.add_row("fork", fork_status["state"] + (f" ({fork_status['hint']})" if fork_status.get("hint") else ""),
                      str(fork_status.get("exe") or ""), str(fork_status.get("llama_cpp") or ""))
        out.print(table)
    except TemperumError as exc:
        _handle_err(exc)


def _remember_fork(option: Optional[Path], ask: bool) -> None:
    """Store the fork executable given to setup; in a terminal, ask for it when none is known."""
    if option is not None:
        exe = option.expanduser()
        if not exe.is_file():
            raise TemperumError("setup", "EXE_MISSING", f"{exe} is not a file.",
                                hint="Point --ollama-exe at the fork's ollama.exe.")
        settings.set_value("ollama-exe", str(exe.resolve()))
        err.print(f"fork: {escape(str(exe.resolve()))} (remembered in {escape(str(settings.path()))})")
        _warn_if_environment_overrides("ollama-exe", str(exe.resolve()))
        return
    known, _source = settings.ollama_exe(None)
    if known is None and ask and sys.stdin.isatty() and sys.stderr.isatty():
        answer = ui.ask(err, "Where is the fork's ollama.exe? (Enter to skip)", default="").strip().strip('"')
        if not answer:
            return
        path = Path(answer).expanduser()
        if path.is_file():
            settings.set_value("ollama-exe", str(path.resolve()))
            err.print(f"fork: {escape(str(path.resolve()))} (remembered in {escape(str(settings.path()))})")
        else:
            err.print(f"  [yellow]![/yellow] {escape(answer)} is not a file; run `temperum setup --ollama-exe PATH` "
                      "when you have the fork")


def _warn_if_environment_overrides(key: str, stored: str) -> None:
    """A value set with setx lingers in terminals started before it changed, and the environment wins."""
    env_name = settings.KEYS[key]
    current = os.environ.get(env_name)
    if current and current != stored:
        err.print(f"  [yellow]![/yellow] {env_name} is set to {escape(current)} in this terminal and takes "
                  "precedence over the remembered value; unset it, or open a terminal started after it changed "
                  "(for an editor's terminals, restart the editor)")


def _fork_status() -> dict[str, Any]:
    """The fork executable in effect (--ollama-exe, TEMPERUM_OLLAMA_EXE or the stored setting) and what its NPU
    runtime is missing."""
    from temperum.gguf import SETUP_COMMAND, LlamaPayload

    exe, source = settings.ollama_exe(None)
    if exe is None:
        return {"exe": None, "source": source, "state": "not set",
                "hint": "run `temperum setup --ollama-exe C:\\path\\to\\fork\\ollama.exe` once (or set "
                        "TEMPERUM_OLLAMA_EXE)"}
    if not exe.is_file():
        return {"exe": str(exe), "source": source, "state": "missing", "hint": f"{exe} ({source}) does not name a file"}
    payload = LlamaPayload.from_executable(exe)
    missing = payload.problems()
    status: dict[str, Any] = {"exe": str(exe), "source": source, "state": "ready" if not missing else "incomplete",
                              "missing": missing, "llama_cpp": payload.version(),
                              "htp": str(payload.htp) if payload.htp else None}
    if missing:
        status["hint"] = (f"run `{SETUP_COMMAND}`" if missing == [m for m in missing if "signed HTP" in m]
                          else "use a fork package built with -HexagonSDKRoot")
    return status


@app.command("doctor")
def doctor_command(as_json: bool = typer.Option(False, "--json", help="Print JSON")) -> None:
    """Report this machine's NPU target, the fork's NPU runtime, toolchains, paths and resources."""
    report, table = _doctor_report()
    if as_json:
        typer.echo(json.dumps(report, indent=2, default=str))
        return
    out.print(table)


def _doctor_report() -> tuple[dict[str, Any], Table]:
    """The doctor's findings as data and as a table (the command line and the TUI show the table)."""
    from temperum import paths
    from temperum.host import HEXAGON_OPT_IN, detect_host, hexagon_opted_in, long_paths_enabled, resolve_target
    from temperum.profiles import DEFAULT_LLAMA_RUNTIME, DEFAULT_RUNTIME, LLAMA_RUNTIMES, get_runtime
    from temperum.toolchains import Toolchains
    from temperum.toolchains.llama_cpp import ConverterSource

    host = detect_host()
    report: dict[str, Any] = {"host": host.to_dict(), "home": str(paths.home()),
                              "long_paths_enabled": long_paths_enabled(), "hexagon_opt_in": hexagon_opted_in()}
    try:
        report["target"] = resolve_target(host, settings.resolve("target", None, "auto").value).id
    except TemperumError as exc:
        report["target"] = None
        report["target_error"] = str(exc)
    report["fork"] = _fork_status()
    usage = shutil.disk_usage(paths.home() if paths.home().exists() else paths.home().anchor)
    report["disk_free_gb"] = round(usage.free / 2**30, 1)
    device = "cuda" if host.cuda_gpus else "cpu"
    report["quantize_device"] = device
    try:
        chains = Toolchains(host, get_runtime(DEFAULT_RUNTIME), device)
        roles = ["quantize", "compile"] + (["verify"] if host.native_arch == "arm64" else [])
        report["toolchains"] = [chains.status(role, deep=True).to_dict() for role in roles]
    except TemperumError as exc:
        report["toolchains_error"] = str(exc)
    source = ConverterSource.load(LLAMA_RUNTIMES[DEFAULT_LLAMA_RUNTIME].llama_cpp_tag)
    report["llama_cpp_converter"] = {"tag": source.tag, "ready": not source.problems()}
    fork = report["fork"]
    table = Table(show_header=False, box=None)
    table.add_row("Processor", escape(str(host.processor_name)))
    table.add_row("Architecture", f"{host.native_arch} (this Python: {host.python_platform})")
    table.add_row("NPU", escape(", ".join(host.npu_devices) or "none found"))
    table.add_row("Target", escape(report["target"] or report.get("target_error", "")) +
                  (" (default; --target qnn-x-elite for a QNN bundle)" if report["target"] == "gguf-hexagon" else ""))
    table.add_row("Fork", escape(f"{fork['exe'] or 'not set'}" + (f" ({fork['source']})" if fork["exe"] else "")
                                 + f": {fork['state']}"
                                 + (f" (llama.cpp {fork['llama_cpp']}, NPU libraries in {fork['htp']})"
                                    if fork["state"] == "ready" else "")
                                 + (f"; missing {', '.join(fork['missing'])}" if fork.get("missing") else "")
                                 + (f"; {fork['hint']}" if fork.get("hint") else "")))
    table.add_row("NPU switch", f"{HEXAGON_OPT_IN}=1 in this shell; the fork's server needs it too"
                  if report["hexagon_opt_in"] else
                  f"{HEXAGON_OPT_IN} is not set here: without it the fork's server runs GGUF models on the CPU")
    table.add_row("Memory", f"{host.total_ram_gb} GB total, {host.available_ram_gb} GB free"
                  + (f", {host.available_commit_gb} GB more can be committed"
                     if host.available_commit_gb is not None else ""))
    table.add_row("Home", escape(report["home"]) + f" ({report['disk_free_gb']} GB free)")
    table.add_row("Long paths", "enabled" if report["long_paths_enabled"] else "disabled (keep the home path short)")
    for status in report.get("toolchains", []):
        qnn_only = " (QNN bundles only)" if status["role"] != "quantize" else ""
        table.add_row(f"Toolchain {status['role']}", escape(status["state"] + (f": {status['detail']}"
                                                                               if status["detail"] else "") + qnn_only))
    if "toolchains_error" in report:
        table.add_row("Toolchains", escape(report["toolchains_error"]))
    table.add_row("llama.cpp converter", escape(f"{source.tag}: " + ("ready" if report["llama_cpp_converter"]["ready"]
                                                                     else "not fetched yet (temperum setup)")))
    table.add_row("QNN quantization", f"{device}" + (f" ({escape(', '.join(host.cuda_gpus))})" if host.cuda_gpus
                                                      else ""))
    return report, table


def _verify_gguf_bundle(bundle: Path, manifest: dict[str, Any], snapshot: Path, level: str,
                        ollama_exe: Optional[Path], host: HostInfo, say: Any) -> dict[str, Any]:
    """Re-run a GGUF bundle's verification with the fork's current runtime."""
    from temperum import gguf, paths
    from temperum.checkpoint import read_checkpoint
    from temperum.gguf_verify import verify_gguf
    from temperum.profiles import DEFAULT_LLAMA_RUNTIME, DEFAULT_RUNTIME, LLAMA_RUNTIMES, get_runtime, get_target
    from temperum.repairs import prepare_source
    from temperum.sources import ModelRef, ResolvedModel
    from temperum.toolchains import Toolchains
    from temperum.toolchains.llama_cpp import ConverterSource

    exe, _source = settings.ollama_exe(ollama_exe)
    if exe is None:
        raise TemperumError("verify", "NEEDS_RUNTIME", "A GGUF bundle is verified with the fork's llama.cpp build.",
                            hint="Run `temperum setup --ollama-exe PATH` once, or pass --ollama-exe PATH.")
    payload = gguf.LlamaPayload.from_executable(exe)
    missing = payload.problems()
    if missing:
        raise TemperumError("verify", "RUNTIME_INCOMPLETE", f"{payload.root} lacks: {', '.join(missing)}",
                            hint=f"Get Qualcomm's signed NPU libraries with `{gguf.SETUP_COMMAND}`.")
    src = manifest.get("source", {})
    model = ResolvedModel(ModelRef("hf", repo=src["repo"]) if src.get("repo") else ModelRef("local", path=snapshot),
                          commit=src.get("commit"), gated=False, license=src.get("license"),
                          config=json.loads((snapshot / "config.json").read_text(encoding="utf-8")), weight_bytes=None)
    runtime = LLAMA_RUNTIMES.get(str(manifest.get("runtime", {}).get("id")), LLAMA_RUNTIMES[DEFAULT_LLAMA_RUNTIME])
    plan = gguf.GgufPlan(model, host, get_target(str(manifest.get("target", {}).get("id", "gguf-hexagon"))), runtime,
                         payload, str(manifest.get("precision", gguf.DEFAULT_QUANT)),
                         int(manifest.get("context_length") or gguf.DEFAULT_CONTEXT), bundle, "verify",
                         {"params_billions": round(read_checkpoint(snapshot).total_params / 1e9, 2)}, level)
    chains = Toolchains(host, get_runtime(DEFAULT_RUNTIME), "cpu")
    chains.ensure("quantize")
    work = paths.jobs_dir() / "verify" / bundle.name
    work.mkdir(parents=True, exist_ok=True)
    snapshot, _repairs = prepare_source(snapshot, work / "source", say)
    reference = None
    try:
        if level in ("standard", "full"):  # perplexity is measured against a Q8_0 conversion on the CPU
            converter = ConverterSource.load(runtime.llama_cpp_tag)
            converter.ensure(progress=say)
            say("converting the source again (Q8_0) for the perplexity reference")
            reference = gguf.convert(chains.python_exe("quantize"), converter, snapshot, work / "reference-Q8_0.gguf",
                                     work / "gguf.log", say)
        return verify_gguf(bundle, snapshot, plan, chains, work, say, reference_gguf=reference,
                           chat=gguf.derived_template(snapshot))
    finally:
        if reference is not None:
            reference.unlink(missing_ok=True)


@app.command("models")
def models_command(as_json: bool = typer.Option(False, "--json", help="Print JSON")) -> None:
    """List maintained recipes, supported families and conversions verified on hardware."""
    from temperum.recipes import FAMILY_TEMPLATES, catalog, catalog_index, verified_conversions

    recipes = catalog()
    data = {"recipes_commit": catalog_index()["commit"], "curated": sorted(recipes),
            "families": FAMILY_TEMPLATES, "verified": verified_conversions()}
    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return
    out.print("[bold]Verified on hardware[/bold] (perplexity ratio, decode tok/s on the NPU; details in "
              "docs/models.md):")
    for entry in sorted(data["verified"], key=lambda e: (e["target"] != "gguf-hexagon", e["repo"])):
        out.print(f"  {escape(entry['repo'])}  {entry['target']}  {entry['quality']}  "
                  f"x{entry['ppl_ratio']}  {entry['decode_tokens_per_s']} tok/s")
    out.print("[bold]GGUF (the default target)[/bold]: any architecture llama.cpp's converter handles (about 300); "
              "verification on the NPU decides whether the result is usable.")
    out.print(f"[bold]QNN bundles[/bold] (--target qnn-x-elite): maintained recipes (microsoft/olive-recipes @ "
              f"{data['recipes_commit'][:8]}):")
    for repo in sorted(recipes):
        out.print(f"  {escape(repo)}")
    out.print("  Other models of these families use their family's recipe:")
    for family, repo in sorted(FAMILY_TEMPLATES.items()):
        out.print(f"  model_type {family}: {escape(repo)} recipe")
    out.print("  Converting a QNN bundle on the CPU needs RAM roughly 8x the parameter count in bytes; "
              "`temperum convert MODEL --plan` shows estimates.")


@app.command("verify")
def verify_command(
    bundle: Path = typer.Argument(..., help="Bundle folder"),
    level: str = typer.Option("standard", "--level", help="quick | standard | full"),
    source: Optional[Path] = typer.Option(None, "--source", help="Source model folder (default: pinned snapshot)"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="GGUF: fork executable (or "
                                                                          "TEMPERUM_OLLAMA_EXE)"),
    as_json: bool = typer.Option(False, "--json", help="Print the full report"),
) -> None:
    """Re-run verification of a bundle on this machine's NPU."""
    from temperum import calib, paths
    from temperum.checkpoint import read_checkpoint
    from temperum.convert import VERIFICATION_FILE
    from temperum.host import detect_host
    from temperum.profiles import DEFAULT_RUNTIME, get_runtime
    from temperum.templates import from_jinja, load_jinja, own
    from temperum.toolchains import Toolchains
    from temperum.verify import verify

    try:
        manifest_file = next((p for p in (bundle / "temperum_manifest.json", bundle / "npu_model_manifest.json")
                              if p.is_file()), bundle / "temperum_manifest.json")  # older bundles keep the old name
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        src = manifest.get("source", {})
        snapshot = source or (paths.sources_dir() / str(src.get("repo", "")).replace("/", "--") / str(src.get("commit")))
        if not (snapshot / "config.json").is_file():
            raise TemperumError("verify", "SOURCE_MISSING", f"Source model not found at {snapshot}",
                                hint="Pass --source with the original model folder.")
        host = detect_host()
        if manifest.get("format") == "gguf":
            report = _verify_gguf_bundle(bundle, manifest, snapshot, level, ollama_exe, host,
                                         _progress(time.time(), as_json))
        else:
            chains = Toolchains(host, get_runtime(DEFAULT_RUNTIME), "cuda" if host.cuda_gpus else "cpu")
            for role in ("quantize", "verify"):
                chains.ensure(role)
            heldout = calib.ensure_corpus(calib.HELDOUT, paths.data_dir()).path if level != "quick" else None
            own_template = manifest.get("chat_template", {}).get("family") == "own"
            report = verify(bundle, snapshot, own() if own_template else from_jinja(load_jinja(snapshot)), level,
                            quantize_python=chains.python_exe("quantize"), verify_python=chains.python_exe("verify"),
                            work=paths.jobs_dir() / "verify" / bundle.name, heldout=heldout,
                            params=read_checkpoint(snapshot).total_params, available_gb=host.available_ram_gb,
                            total_gb=host.total_ram_gb,
                            progress=_progress(time.time(), as_json))
        (bundle / VERIFICATION_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if as_json:
            typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            reasons = "; ".join(report.get("reasons", []))
            err.print(str(report["verdict"]) + (f": {reasons}" if reasons else ""))
        if report["verdict"] == "failed":
            raise typer.Exit(code=3)
    except TemperumError as exc:
        _handle_err(exc)


@app.command("optimize")
def optimize_command(
    source: Path = typer.Argument(..., help="A Q8_0 GGUF: a bundle's model.gguf converted with --quant Q8_0, or a "
                                            "conversion job's model-Q8_0.gguf"),
    out_dir: Optional[Path] = typer.Option(None, "--out", help="Work folder for the trials (default: optimize/ beside "
                                                               "the source)"),
    decode_cap: float = typer.Option(0.10, "--decode-cap", help="Largest decode loss against plain Q4_0 a mix may "
                                                               "cost (fraction)"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="Fork executable whose llama.cpp build "
                                                                         "quantizes and measures"),
    runtime: bool = typer.Option(True, "--runtime/--no-runtime", help="After the mix: measure llama-server's CPU "
                                                                     "threads beside the NPU for the chosen file"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate",
                                   help="Quantize every candidate with an importance matrix computed first on the "
                                        "CPU (needs llama-imatrix in the fork's runtime)"),
    as_json: bool = typer.Option(False, "--json", help="Print the recipe document on stdout"),
) -> None:
    """Search per-tensor quantization mixes of a Q8_0 GGUF on this machine's NPU; keep the best by the rule."""
    from temperum import calib, gguf, optimize, paths
    from temperum.host import detect_host

    try:
        exe, _source = settings.ollama_exe(ollama_exe)
        if exe is None:
            raise TemperumError("optimize", "NEEDS_RUNTIME",
                                "The search quantizes and measures with the Ollama fork's own llama.cpp build.",
                                hint="Run `temperum setup --ollama-exe PATH` once, or pass --ollama-exe PATH.")
        if not source.is_file():
            raise TemperumError("optimize", "SOURCE_MISSING", f"{source} is not a file.",
                                hint="Point at a Q8_0 GGUF (convert with --quant Q8_0 --verify none first).")
        payload = gguf.LlamaPayload.from_executable(exe)
        missing = payload.problems()
        if missing:
            raise TemperumError("optimize", "RUNTIME_INCOMPLETE",
                                f"{payload.root} lacks the llama.cpp Hexagon runtime: {', '.join(missing)}")
        work = out_dir or source.parent / "optimize"
        heldout = calib.ensure_corpus(calib.HELDOUT, paths.data_dir()).path
        objective = optimize.Objective(decode_cap=decode_cap, size_budget_mib=gguf.NPU_WINDOW_GB * 1024)

        def say(message: str) -> None:
            if not as_json:
                err.print(escape(message))

        imatrix: Optional[Path] = None
        calibration: Optional[dict[str, Any]] = None
        if calibrate and payload.has_imatrix:
            work.mkdir(parents=True, exist_ok=True)
            corpus = calib.ensure_corpus(calib.CALIBRATION, paths.data_dir())
            imatrix = work / gguf.IMATRIX
            if not imatrix.is_file():
                say(f"calibrating: an importance matrix from {gguf.CALIBRATION_CHUNKS} chunks of {corpus.spec.dataset} "
                    f"({corpus.spec.split} rows) on the CPU")
                gguf.compute_imatrix(payload, source, gguf.calibration_text(corpus.path, work / "calibration.txt"),
                                     imatrix, work / "calibrate.log", say)
            calibration = gguf.calibration_record(payload, corpus)
        elif calibrate:
            say("the runtime has no llama-imatrix: the candidates are quantized without calibration")
        found = optimize.search(payload, source, work, heldout, say=say, objective=objective, imatrix=imatrix)
        rt = optimize.runtime_search(payload, found.chosen.path, work, heldout, say=say) if runtime else None
        doc = optimize.document(found, found.reference_ppl, objective, chunks=optimize.PPL_CHUNKS,
                                host=detect_host().to_dict(), llama_cpp=payload.version(),
                                source={"path": str(source), "calibration": calibration}, runtime=rt)
        recipe_path = work / gguf.RECIPE
        recipe_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        (work / gguf.RECIPE_MD).write_text(optimize.markdown(doc), encoding="utf-8")
        if as_json:
            typer.echo(json.dumps(doc, indent=2, default=str))
        else:
            for line in optimize.lines(found, objective, rt):
                err.print(escape(line))
            err.print(f"recipe: {escape(str(recipe_path))}; chosen file: {escape(str(found.chosen.path))}")
    except TemperumError as exc:
        _handle_err(exc)


@app.command("inspect")
def inspect_command(
    input: Path = typer.Option(..., "--input", help="Bundle folder to inspect (read-only)"),
    processor_name: Optional[str] = typer.Option(None, "--processor-name", help="Compare an exact declared target"),
) -> None:
    """Emit JSON hashes and structural/declaration/target evidence levels; never executes the model."""
    from temperum.bundle.inspect import inspect_prebuilt

    report = inspect_prebuilt(input, processor_name=processor_name)
    typer.echo(json.dumps(report.to_dict(), indent=2))
    if not report.importable:
        raise typer.Exit(code=2)


@app.command("config")
def config_command(
    key: Optional[str] = typer.Argument(None, help="Setting: ollama-exe | target | host (none: show all)"),
    value: Optional[str] = typer.Argument(None, help="New value (none: show the one in effect)"),
    unset: bool = typer.Option(False, "--unset", help="Forget the stored value"),
) -> None:
    """Show or change what temperum remembers: the fork's ollama.exe, the default target, the fork's endpoint.
    A command-line option wins over these, then the environment (TEMPERUM_OLLAMA_EXE, TEMPERUM_TARGET,
    OLLAMA_HOST), then the stored value."""
    try:
        if key is None:
            stored = settings.load()
            table = Table("setting", "in effect", "from", "stored", "meaning", box=None)
            for name in settings.KEYS:
                current = settings.resolve(name)
                table.add_row(name, escape(current.value or ""), current.source if current.value else "",
                              escape(stored.get(name, "")), settings.DESCRIPTIONS[name])
            out.print(table)
            err.print(f"[dim]{escape(str(settings.path()))}[/dim]")
            return
        settings.check_key(key)
        if unset:
            settings.unset(key)
            err.print(f"{key}: forgotten")
            return
        if value is None:
            current = settings.resolve(key)
            typer.echo(current.value or "")  # plain, for scripts
            err.print(f"[dim]{key}: from {current.source}[/dim]")
            return
        value = _store_setting(key, value)
        err.print(f"{key} = {escape(value)} (stored in {escape(str(settings.path()))})")
        _warn_if_environment_overrides(key, value)
    except TemperumError as exc:
        _handle_err(exc)


def _store_setting(key: str, value: str) -> str:
    """Check a setting's value the way its consumers will, store it, and return the stored form."""
    settings.check_key(key)
    if key == "ollama-exe":
        exe = Path(value).expanduser()
        if not exe.is_file():
            raise TemperumError("config", "EXE_MISSING", f"{exe} is not a file.", hint="Name the fork's ollama.exe.")
        value = str(exe.resolve())
    elif key == "host":
        from temperum.install import fork

        value = fork.endpoint_for(value)
    elif value != "auto":
        from temperum.profiles import get_target

        get_target(value)
    settings.set_value(key, value)
    return value


@app.command("tui")
def tui_command() -> None:
    """Open the full-screen interface (also what `temperum` alone does in a terminal)."""
    from temperum.tui.app import run

    run()


@app.command("validate")
def validate_command(
    tag: str = typer.Argument(..., help="An installed Ollama tag"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="Fork executable (default: "
                                                                        "TEMPERUM_OLLAMA_EXE); its server is "
                                                                        "started if none runs"),
    host: Optional[str] = typer.Option(None, "--host", help="http://127.0.0.1:PORT (default: OLLAMA_HOST, else "
                                                           "port 11434)"),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON"),
) -> None:
    """Load a model through the fork and report where it runs, its speed and the CPU it keeps busy."""
    from temperum import paths
    from temperum.install import fork
    from temperum.install.validate import validate as validate_model

    say = _progress(time.time(), as_json)
    try:
        exe = fork.executable(ollama_exe)
        endpoint = fork.endpoint_for(host)
        info = fork.ensure(endpoint, exe, say, log_dir=paths.home() / "logs")
        if info.started:
            say(f"started the fork's server at {endpoint} (pid {info.pid}); it keeps running")
        report = validate_model(endpoint, tag, fork_dir=exe.parent if exe else None, say=say)
        if info.started:
            report["server"] = {"endpoint": endpoint, "pid": info.pid, "log": str(info.log), "started": True}
        if as_json:
            typer.echo(json.dumps(report, indent=2, default=str))
        else:
            err.print(ui.validation_table(report))
    except TemperumError as exc:
        _handle_err(exc)


@app.command("push")
def push_command(
    tag: str = typer.Argument(..., help="An installed Ollama tag"),
    push_as: Optional[str] = typer.Option(None, "--as", help="ollama.com name, account/name:tag (default: your "
                                                            "account/<name>:<tag>)"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="Fork executable (default: "
                                                                        "TEMPERUM_OLLAMA_EXE); its server is "
                                                                        "started if none runs"),
    host: Optional[str] = typer.Option(None, "--host", help="http://127.0.0.1:PORT (default: OLLAMA_HOST, else "
                                                           "port 11434)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="No questions: fail if the server is not signed in"),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON"),
) -> None:
    """Upload a model to ollama.com through the fork, which holds your account key; temperum never sees it."""
    from temperum import paths
    from temperum.install import fork

    session = _session(time.time(), yes, as_json)
    try:
        endpoint = fork.endpoint_for(host)
        info = fork.ensure(endpoint, fork.executable(ollama_exe), session.say, log_dir=paths.home() / "logs")
        if info.started:
            session.say(f"started the fork's server at {endpoint} (pid {info.pid}); it keeps running")
        result = flows.push_flow(session, endpoint, tag, push_as)
        if as_json:
            typer.echo(json.dumps(result, indent=2))
    except flows.Cancelled:
        raise typer.Exit(code=1) from None
    except TemperumError as exc:
        _handle_err(exc)


@app.command("install")
def install_command(
    input: Path = typer.Option(..., "--input", help="Bundle folder to install"),
    name: str = typer.Option(..., "--name", help="Local Ollama tag, optionally namespace/name:tag"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="Compatible Ollama fork executable"),
    host: Optional[str] = typer.Option(None, "--host", help="Numeric loopback HTTP endpoint; or OLLAMA_HOST"),
    modelfile: Optional[Path] = typer.Option(None, "--modelfile", help="Local FROM + Go TEMPLATE Modelfile"),
    processor_name: Optional[str] = typer.Option(None, "--processor-name", help="Compare an exact declared target"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Explicit consent to replace an existing local tag"),
    execute: bool = typer.Option(False, "--execute", help="Execute the reviewed plan (otherwise plan only)"),
    expect_fingerprint: Optional[str] = typer.Option(None, "--expect-fingerprint", help="Reviewed plan fingerprint"),
    work_dir: Path = typer.Option(Path("."), "--work-dir", help="Existing scratch parent outside the bundle"),
    timeout: float = typer.Option(300, "--timeout", min=1, max=3600, help="Execution time budget in seconds"),
) -> None:
    """Plan (default) or execute a local `ollama create` of a bundle; never downloads or uploads."""
    from temperum import paths
    from temperum.install import fork
    from temperum.install.ollama_import import execute_import, install_gguf, plan_import

    try:
        if (input / "model.gguf").is_file():  # GGUF bundle: FROM + parameters; the GGUF carries its template
            exe = fork.executable(ollama_exe)
            if exe is None:
                raise TemperumError("import", "EXE_REQUIRED", "Installing needs the fork executable.",
                                    hint="Run `temperum setup --ollama-exe PATH` once, or pass --ollama-exe PATH.")
            endpoint = fork.endpoint_for(host)
            if not execute:
                typer.echo(json.dumps({"action": "local-create", "format": "gguf", "bundle": str(input), "name": name,
                                       "executable": str(exe), "endpoint": endpoint, "executed": False,
                                       "next": "Repeat with --execute to run `ollama create` on the fork."}, indent=2))
                return
            say = _progress(time.time())
            info = fork.ensure(endpoint, exe, say, log_dir=paths.home() / "logs")
            result = install_gguf(input, name, executable=exe, host=endpoint, work_dir=work_dir / "temperum-install",
                                  overwrite=overwrite, timeout=timeout, say=say)
            if info.started:
                result["server"] = {"endpoint": endpoint, "pid": info.pid, "log": str(info.log), "started": True}
            typer.echo(json.dumps({"action": "local-create", "format": "gguf", "executed": True, "installed": True,
                                   **result}, indent=2))
            return
        plan = plan_import(input, name, executable=ollama_exe, host=host, modelfile=modelfile,
                           overwrite=overwrite, processor_name=processor_name)
        result = (execute_import(plan, expected_fingerprint=expect_fingerprint, work_dir=work_dir,
                                 timeout=timeout) if execute else plan.to_dict())
        typer.echo(json.dumps(result, indent=2))
    except TemperumError as exc:
        _handle_err(exc)
