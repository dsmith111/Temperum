from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from npu_model.core.errors import NpuModelError

# The hub's cache works without symlinks on Windows; its warning about that is just noise here.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Convert Hugging Face models into NPU-ready bundles for the Ollama fork.")
out = Console()
err = Console(stderr=True)


def _handle_err(e: Exception) -> None:
    if isinstance(e, NpuModelError):
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
    table.add_row("Estimate", escape(f"~{est['minutes']:.0f} min, ~{est['ram_gb']} GB RAM, ~{est['disk_gb']} GB disk "
                                     f"(download {est['download_gb']} GB)"))
    err.print(table)
    for warning in plan.warnings:
        err.print(f"[yellow]![/yellow] {escape(warning)}")


@app.command("convert")
def convert_command(
    model: str = typer.Argument(..., help="Hugging Face id (org/name[@revision]) or a local model folder"),
    target: str = typer.Option("auto", "--target", help="NPU target profile (auto = this machine)"),
    quality: str = typer.Option("auto", "--quality", help="auto | standard (8-bit RTN) | recipe (recipe GPTQ, CUDA)"),
    device: str = typer.Option("auto", "--device", help="Quantization device: auto | cpu | cuda"),
    context: Optional[int] = typer.Option(None, "--context", help="Declared context length (default: min(model, 4096))"),
    out_dir: Optional[Path] = typer.Option(None, "--out", help="Bundle folder (default under the npu-model home)"),
    verify_level: str = typer.Option("standard", "--verify", help="none | quick | standard | full"),
    template: Optional[Path] = typer.Option(None, "--template", help="Ollama Go template file (overrides detection)"),
    stop: list[str] = typer.Option([], "--stop", help="Stop sequence for --template (repeatable)"),
    num_splits: Optional[int] = typer.Option(None, "--num-splits", help="Advanced: NPU graph splits"),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Keep Olive's intermediate cache"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing bundle folder"),
    python: Optional[Path] = typer.Option(None, "--python", help="Native Python for toolchains (default: uv)"),
    plan_only: bool = typer.Option(False, "--plan", help="Show the plan and stop"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Run without asking (accepts the plan and licence notice)"),
    as_json: bool = typer.Option(False, "--json", help="Print the plan/result as JSON on stdout"),
    install: bool = typer.Option(False, "--install", help="Also install into a running Ollama fork"),
    name: Optional[str] = typer.Option(None, "--name", help="Ollama tag for --install (default: <model>:npu)"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="Fork executable for --install"),
    host: Optional[str] = typer.Option(None, "--host", help="Fork endpoint http://127.0.0.1:PORT for --install "
                                                                "(or OLLAMA_HOST)"),
) -> None:
    """Convert a model into an NPU bundle for this machine (resolve, quantize, compile, verify)."""
    from npu_model import convert

    start = time.time()
    try:
        if install:  # fail now, not after a long conversion
            _check_install_options(ollama_exe, host)
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
        result = convert.run(plan, progress=_progress(start, as_json))
        report = result.verification
        summary: dict[str, Any] = {"bundle": str(result.bundle), "verdict": report.get("verdict"),
                                   "reasons": report.get("reasons", []), "job": str(result.job_dir),
                                   "minutes": round((time.time() - start) / 60, 1)}
        quality_report = report.get("quality", {})
        if quality_report.get("perplexity"):
            summary["perplexity_ratio"] = quality_report["perplexity"].get("ratio")
        summary["speed"] = report.get("speed", {})
        install = install and summary["verdict"] != "failed"  # never install a bundle that failed verification
        if install:
            summary["install"] = _install(result.bundle, name or f"{plan.model.name}:npu", ollama_exe, host,
                                          result.job_dir, plan.host.processor_name)
        if as_json:
            typer.echo(json.dumps(summary, indent=2, default=str))
        else:
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
            if install:
                err.print(f"  installed as {escape(str(summary['install'].get('name')))}: "
                          f"ollama run {escape(str(summary['install'].get('name')))}")
            elif summary["verdict"] != "failed":
                err.print(f"  install: npu-model install --input {escape(str(result.bundle))} "
                          f"--name {escape(plan.model.name)}:npu --ollama-exe FORK_EXE --host http://127.0.0.1:11434 "
                          "(review the plan, then add --execute --expect-fingerprint FP)")
        if summary["verdict"] == "failed":
            raise typer.Exit(code=3)
    except NpuModelError as exc:
        _handle_err(exc)


def _check_install_options(ollama_exe: Optional[Path], host: Optional[str]) -> None:
    from npu_model.install.ollama_import import local_endpoint

    local_endpoint(host or os.environ.get("OLLAMA_HOST"))
    exe = ollama_exe or (Path(os.environ["NPU_MODEL_OLLAMA_EXE"]) if os.environ.get("NPU_MODEL_OLLAMA_EXE") else None)
    if exe is None or not exe.is_file():
        raise NpuModelError("import", "EXE_REQUIRED", "--install needs the fork executable.",
                            hint="Pass --ollama-exe PATH or set NPU_MODEL_OLLAMA_EXE.")


def _install(bundle: Path, name: str, ollama_exe: Optional[Path], host: Optional[str], job_dir: Path,
             processor_name: Optional[str]) -> dict[str, Any]:
    from npu_model.install.ollama_import import execute_import, plan_import

    plan = plan_import(bundle, name, executable=ollama_exe, host=host, processor_name=processor_name)
    work = job_dir / "install"
    work.mkdir(parents=True, exist_ok=True)
    return execute_import(plan, expected_fingerprint=plan.fingerprint, work_dir=work)


@app.command("setup")
def setup_command(
    device: str = typer.Option("auto", "--device", help="Quantization device the quantize toolchain targets"),
    python: Optional[Path] = typer.Option(None, "--python", help="Native Python for toolchains (default: uv)"),
    check: bool = typer.Option(False, "--check", help="Only report status (deep check)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild environments even if they look ready"),
) -> None:
    """Create (or check) the quantize, compile and verify toolchains."""
    from npu_model.host import detect_host
    from npu_model.profiles import DEFAULT_RUNTIME, get_runtime
    from npu_model.toolchains import Toolchains

    try:
        host = detect_host()
        chosen = "cuda" if device == "auto" and host.cuda_gpus else ("cpu" if device == "auto" else device)
        chains = Toolchains(host, get_runtime(DEFAULT_RUNTIME), chosen, python=python,
                            progress=_progress(time.time()))
        roles = ["quantize", "compile"] + (["verify"] if host.native_arch == "arm64" else [])
        table = Table("role", "state", "path", "tested")
        for role in roles:
            status = chains.status(role, deep=True) if check else (
                chains.build(role) if rebuild else chains.ensure(role))
            table.add_row(role, status.state + (f" ({status.detail})" if status.detail else ""), str(status.path),
                          status.tested)
        out.print(table)
    except NpuModelError as exc:
        _handle_err(exc)


@app.command("doctor")
def doctor_command(as_json: bool = typer.Option(False, "--json", help="Print JSON")) -> None:
    """Report this machine's NPU target, quantization device, toolchains, paths and resources."""
    from npu_model import paths
    from npu_model.host import detect_host, long_paths_enabled, resolve_target
    from npu_model.profiles import DEFAULT_RUNTIME, get_runtime
    from npu_model.toolchains import Toolchains

    host = detect_host()
    report: dict[str, Any] = {"host": host.to_dict(), "home": str(paths.home()),
                              "long_paths_enabled": long_paths_enabled()}
    try:
        report["target"] = resolve_target(host).id
    except NpuModelError as exc:
        report["target"] = None
        report["target_error"] = str(exc)
    usage = shutil.disk_usage(paths.home() if paths.home().exists() else paths.home().anchor)
    report["disk_free_gb"] = round(usage.free / 2**30, 1)
    device = "cuda" if host.cuda_gpus else "cpu"
    report["quantize_device"] = device
    try:
        chains = Toolchains(host, get_runtime(DEFAULT_RUNTIME), device)
        roles = ["quantize", "compile"] + (["verify"] if host.native_arch == "arm64" else [])
        report["toolchains"] = [chains.status(role, deep=True).to_dict() for role in roles]
    except NpuModelError as exc:
        report["toolchains_error"] = str(exc)
    if as_json:
        typer.echo(json.dumps(report, indent=2, default=str))
        return
    table = Table(show_header=False, box=None)
    table.add_row("Processor", escape(str(host.processor_name)))
    table.add_row("Architecture", f"{host.native_arch} (this Python: {host.python_platform})")
    table.add_row("NPU", escape(", ".join(host.npu_devices) or "none found"))
    table.add_row("Target", escape(report["target"] or report.get("target_error", "")))
    table.add_row("Quantization", f"{device}" + (f" ({escape(', '.join(host.cuda_gpus))})" if host.cuda_gpus else ""))
    table.add_row("Memory", f"{host.total_ram_gb} GB total, {host.available_ram_gb} GB free"
                  + (f", {host.available_commit_gb} GB more can be committed"
                     if host.available_commit_gb is not None else ""))
    table.add_row("Home", escape(report["home"]) + f" ({report['disk_free_gb']} GB free)")
    table.add_row("Long paths", "enabled" if report["long_paths_enabled"] else "disabled (keep the home path short)")
    for status in report.get("toolchains", []):
        table.add_row(f"Toolchain {status['role']}", escape(status["state"] + (f": {status['detail']}"
                                                                               if status["detail"] else "")))
    if "toolchains_error" in report:
        table.add_row("Toolchains", escape(report["toolchains_error"]))
    out.print(table)


@app.command("models")
def models_command(as_json: bool = typer.Option(False, "--json", help="Print JSON")) -> None:
    """List maintained recipes, supported families and conversions verified on hardware."""
    from npu_model.recipes import FAMILY_TEMPLATES, catalog, catalog_index, verified_conversions

    recipes = catalog()
    data = {"recipes_commit": catalog_index()["commit"], "curated": sorted(recipes),
            "families": FAMILY_TEMPLATES, "verified": verified_conversions()}
    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return
    out.print("[bold]Verified on hardware[/bold] (perplexity vs source model, decode speed):")
    for entry in data["verified"]:
        out.print(f"  {escape(entry['repo'])}  {entry['target']}  {entry['quality']}  "
                  f"x{entry['ppl_ratio']}  {entry['decode_tokens_per_s']} tok/s  [dim]{escape(entry['evidence'])}[/dim]")
    out.print(f"[bold]Maintained recipes[/bold] (microsoft/olive-recipes @ {data['recipes_commit'][:8]}):")
    for repo in sorted(recipes):
        out.print(f"  {escape(repo)}")
    out.print("[bold]Other models[/bold] use their family's recipe and are checked by verification:")
    for family, repo in sorted(FAMILY_TEMPLATES.items()):
        out.print(f"  model_type {family}: {escape(repo)} recipe")
    out.print("Converting on CPU needs RAM roughly 8x the parameter count in bytes; "
              "`npu-model convert MODEL --plan` shows estimates.")


@app.command("verify")
def verify_command(
    bundle: Path = typer.Argument(..., help="Bundle folder"),
    level: str = typer.Option("standard", "--level", help="quick | standard | full"),
    source: Optional[Path] = typer.Option(None, "--source", help="Source model folder (default: pinned snapshot)"),
    as_json: bool = typer.Option(False, "--json", help="Print the full report"),
) -> None:
    """Re-run verification of a bundle on this machine's NPU."""
    from npu_model import calib, paths
    from npu_model.checkpoint import read_checkpoint
    from npu_model.convert import VERIFICATION_FILE
    from npu_model.host import detect_host
    from npu_model.profiles import DEFAULT_RUNTIME, get_runtime
    from npu_model.templates import from_jinja, load_jinja
    from npu_model.toolchains import Toolchains
    from npu_model.verify import verify

    try:
        manifest = json.loads((bundle / "npu_model_manifest.json").read_text(encoding="utf-8"))
        src = manifest.get("source", {})
        snapshot = source or (paths.sources_dir() / str(src.get("repo", "")).replace("/", "--") / str(src.get("commit")))
        if not (snapshot / "config.json").is_file():
            raise NpuModelError("verify", "SOURCE_MISSING", f"Source model not found at {snapshot}",
                                hint="Pass --source with the original model folder.")
        host = detect_host()
        chains = Toolchains(host, get_runtime(DEFAULT_RUNTIME), "cuda" if host.cuda_gpus else "cpu")
        for role in ("quantize", "verify"):
            chains.ensure(role)
        heldout = calib.ensure_corpus(calib.HELDOUT, paths.data_dir()).path if level != "quick" else None
        report = verify(bundle, snapshot, from_jinja(load_jinja(snapshot)), level,
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
    except NpuModelError as exc:
        _handle_err(exc)


@app.command("inspect")
def inspect_command(
    input: Path = typer.Option(..., "--input", help="Bundle folder to inspect (read-only)"),
    processor_name: Optional[str] = typer.Option(None, "--processor-name", help="Compare an exact declared target"),
) -> None:
    """Emit JSON hashes and structural/declaration/target evidence levels; never executes the model."""
    from npu_model.bundle.inspect import inspect_prebuilt

    report = inspect_prebuilt(input, processor_name=processor_name)
    typer.echo(json.dumps(report.to_dict(), indent=2))
    if not report.importable:
        raise typer.Exit(code=2)


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
    from npu_model.install.ollama_import import execute_import, plan_import

    try:
        plan = plan_import(input, name, executable=ollama_exe, host=host, modelfile=modelfile,
                           overwrite=overwrite, processor_name=processor_name)
        result = (execute_import(plan, expected_fingerprint=expect_fingerprint, work_dir=work_dir,
                                 timeout=timeout) if execute else plan.to_dict())
        typer.echo(json.dumps(result, indent=2))
    except NpuModelError as exc:
        _handle_err(exc)
