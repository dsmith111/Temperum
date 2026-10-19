from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from npu_model.core.errors import NpuModelError

if TYPE_CHECKING:
    from npu_model.host import HostInfo

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
    target: str = typer.Option("auto", "--target", envvar="NPU_MODEL_TARGET",
                               help="NPU target profile (auto = a GGUF for this machine's NPU; qnn-x-elite for a "
                                    "QNN bundle)"),
    quality: str = typer.Option("auto", "--quality", help="QNN: auto | standard (8-bit RTN) | recipe (GPTQ, CUDA)"),
    device: str = typer.Option("auto", "--device", help="QNN quantization device: auto | cpu | cuda"),
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
    name: Optional[str] = typer.Option(None, "--name", help="Ollama tag for --install (default: <model>:npu; "
                                                                    "<model>:npu-qnn for a QNN bundle)"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="Fork executable for --install"),
    host: Optional[str] = typer.Option(None, "--host", help="Fork endpoint http://127.0.0.1:PORT for --install "
                                                                "(or OLLAMA_HOST)"),
    quant: str = typer.Option("Q4_0", "--quant", help="GGUF type for --target gguf-hexagon: Q4_0 | Q8_0 | Q4_K_M"),
) -> None:
    """Convert a model for this machine's NPU (resolve, convert, quantize, verify, optionally install)."""
    from npu_model import convert
    from npu_model.host import detect_host, resolve_target
    from npu_model.profiles import get_target

    start = time.time()
    try:
        if install:  # fail now, not after a long conversion
            _check_install_options(ollama_exe, host)
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
                          install=install, name=name, ollama_exe=ollama_exe, host=host, start=start, machine=machine)
            return
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
            summary["install"] = _install(result.bundle, name or f"{plan.model.name}:npu-qnn", ollama_exe, host,
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
                          f"--name {escape(plan.model.name)}:npu-qnn --ollama-exe FORK_EXE --host http://127.0.0.1:11434 "
                          "(review the plan, then add --execute --expect-fingerprint FP)")
            else:
                err.print(f"  Olive's cache is kept in {escape(str(result.job_dir))}, so rerunning with --overwrite "
                          "(for example with --num-splits) reuses the quantization")
        if summary["verdict"] == "failed":
            raise typer.Exit(code=3)
    except NpuModelError as exc:
        _handle_err(exc)


def _convert_gguf(model: str, target_id: str, *, quant: str, context: Optional[int], out_dir: Optional[Path],
                  verify_level: str, overwrite: bool, python: Optional[Path], plan_only: bool, yes: bool,
                  as_json: bool, install: bool, name: Optional[str], ollama_exe: Optional[Path],
                  host: Optional[str], start: float, machine: Optional[HostInfo] = None) -> None:
    """The GGUF target: llama.cpp's converter, the fork's llama-quantize, verified on the NPU."""
    from npu_model import gguf
    from npu_model.gguf_verify import verify_gguf
    from npu_model.host import detect_host
    from npu_model.install.ollama_import import install_gguf
    from npu_model.profiles import get_target
    from npu_model.sources import parse_ref, resolve

    exe = ollama_exe or (Path(os.environ["NPU_MODEL_OLLAMA_EXE"]) if os.environ.get("NPU_MODEL_OLLAMA_EXE") else None)
    plan = gguf.plan(resolve(parse_ref(model)), machine or detect_host(), get_target(target_id), ollama_exe=exe,
                     quant=quant, context=context, out=out_dir, verify=verify_level, overwrite=overwrite,
                     python=python)
    if as_json:
        typer.echo(json.dumps({"event": "plan", "plan": plan.to_dict()}, default=str), err=True)
    else:
        est = plan.estimates
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_row("Model", escape(f"{plan.model.ref.display}" + (f" @ {plan.model.commit[:12]}"
                                                                      if plan.model.commit else "")))
        table.add_row("Architecture", escape(f"{plan.model.architecture}, ~{est['params_billions']}B params"))
        table.add_row("Licence", escape(plan.model.license or "unknown") + (" (gated)" if plan.model.gated else ""))
        table.add_row("Target", escape(f"{plan.target.id}: {plan.target.description}"))
        table.add_row("Runtime", escape(f"{plan.runtime.description} ({plan.payload.root})"))
        table.add_row("NPU libraries", escape(str(plan.payload.htp)))
        table.add_row("Format", escape(f"GGUF {plan.quant}, context {plan.context}"))
        table.add_row("Verification", escape(plan.verify_level))
        table.add_row("Output", escape(str(plan.destination)))
        table.add_row("Estimate", escape(f"~{est['minutes']:.0f} min, ~{est['ram_gb']} GB RAM, ~{est['disk_gb']} GB "
                                         f"disk, {est['gguf_gb']} GB file"))
        err.print(table)
        for warning in plan.warnings:
            err.print(f"[yellow]![/yellow] {escape(warning)}")
    if plan_only:
        if as_json:
            typer.echo(json.dumps(plan.to_dict(), indent=2, default=str))
        return
    if not yes and not typer.confirm("Proceed?", default=True, err=True):
        raise typer.Exit(code=1)
    result = gguf.run(plan, progress=_progress(start, as_json), verifier=verify_gguf)
    report = result.verification
    summary: dict[str, Any] = {"bundle": str(result.bundle), "verdict": report.get("verdict"),
                               "reasons": report.get("reasons", []), "job": str(result.job_dir),
                               "minutes": round((time.time() - start) / 60, 1),
                               "npu_placement": report.get("npu_placement", {}), "speed": report.get("speed", {})}
    ppl = report.get("quality", {}).get("perplexity", {})
    if ppl:
        summary["perplexity_ratio"] = ppl.get("ratio")
    if install and summary["verdict"] != "failed" and exe is not None:
        summary["install"] = install_gguf(result.bundle, name or f"{plan.model.name}:npu", executable=exe,
                                          host=host or os.environ.get("OLLAMA_HOST"), work_dir=result.job_dir / "install",
                                          say=_progress(start, as_json))
    if as_json:
        typer.echo(json.dumps(summary, indent=2, default=str))
    else:
        colour = {"passed": "green", "warning": "yellow"}.get(str(summary["verdict"]), "red")
        err.print(f"[bold {colour}]{escape(str(summary['verdict']))}[/bold {colour}] "
                  f"{escape(str(result.bundle))} ({summary['minutes']} min)")
        for reason in summary["reasons"]:
            err.print(f"  - {escape(reason)}")
        placed = summary["npu_placement"]
        if placed.get("layers_total"):
            err.print(f"  NPU: {placed.get('layers_offloaded')}/{placed.get('layers_total')} layers, "
                      f"{placed.get('htp_weight_mib')} MiB of weights on HTP")
        if "perplexity_ratio" in summary:
            err.print(f"  perplexity vs the Q8_0 conversion on the CPU: {summary['perplexity_ratio']}x")
        speed = summary["speed"]
        if speed.get("median_tokens_per_s"):
            prompt = (f"{speed['long_prompt_tokens_per_s']:.0f} tokens/s ({speed['long_prompt_tokens']}-token prompt)"
                      if speed.get("long_prompt_tokens_per_s") else
                      f"{speed.get('median_prompt_tokens_per_s') or 0:.0f} tokens/s")
            err.print(f"  decode: {speed['median_tokens_per_s']:.0f} tokens/s, prompt: {prompt}")
        if "install" in summary:
            installed = summary["install"]
            placed = installed.get("placement", {})
            where = {"npu": f"whole on the NPU ({placed.get('npu_mib')} MiB)",
                     "split": f"{placed.get('npu_mib')} of {placed.get('size_mib')} MiB on the NPU",
                     "cpu": "on the CPU"}.get(str(placed.get("where")), "")
            err.print(f"  installed as {escape(str(installed['name']))}" + (f"; the fork runs it {where}" if where
                                                                             else "")
                      + f": ollama run {escape(str(installed['name']))}")
            for warning in installed.get("warnings", []):
                err.print(f"  [yellow]![/yellow] {escape(str(warning))}")
    if summary["verdict"] == "failed":
        raise typer.Exit(code=3)


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
    target: str = typer.Option("auto", "--target", envvar="NPU_MODEL_TARGET",
                               help="What to prepare for: auto (a GGUF for this machine's NPU) or a target id such "
                                    "as qnn-x-elite"),
    device: str = typer.Option("auto", "--device", help="QNN: quantization device the quantize toolchain targets"),
    python: Optional[Path] = typer.Option(None, "--python", help="Native Python for toolchains (default: uv)"),
    check: bool = typer.Option(False, "--check", help="Only report status (deep check)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild environments even if they look ready"),
) -> None:
    """Create (or check) what conversions need: for a GGUF the conversion toolchain and llama.cpp's converter;
    for QNN bundles also the compile and verify toolchains."""
    from npu_model.host import detect_host, resolve_target
    from npu_model.profiles import DEFAULT_LLAMA_RUNTIME, DEFAULT_RUNTIME, LLAMA_RUNTIMES, get_runtime
    from npu_model.toolchains import Toolchains
    from npu_model.toolchains.llama_cpp import ConverterSource

    try:
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
        out.print(table)
    except NpuModelError as exc:
        _handle_err(exc)


def _fork_status() -> dict[str, Any]:
    """The fork executable conversions use (NPU_MODEL_OLLAMA_EXE) and what its NPU runtime is missing."""
    from npu_model.gguf import SETUP_COMMAND, LlamaPayload

    setting = os.environ.get("NPU_MODEL_OLLAMA_EXE")
    if not setting:
        return {"exe": None, "state": "not set", "hint": "set NPU_MODEL_OLLAMA_EXE to the fork's ollama.exe"}
    exe = Path(setting)
    if not exe.is_file():
        return {"exe": setting, "state": "missing", "hint": "NPU_MODEL_OLLAMA_EXE does not name a file"}
    payload = LlamaPayload.from_executable(exe)
    missing = payload.problems()
    status: dict[str, Any] = {"exe": str(exe), "state": "ready" if not missing else "incomplete", "missing": missing,
                              "llama_cpp": payload.version(), "htp": str(payload.htp) if payload.htp else None}
    if missing:
        status["hint"] = (f"run `{SETUP_COMMAND}`" if missing == [m for m in missing if "signed HTP" in m]
                          else "use a fork package built with -HexagonSDKRoot")
    return status


@app.command("doctor")
def doctor_command(as_json: bool = typer.Option(False, "--json", help="Print JSON")) -> None:
    """Report this machine's NPU target, the fork's NPU runtime, toolchains, paths and resources."""
    from npu_model import paths
    from npu_model.host import HEXAGON_OPT_IN, detect_host, hexagon_opted_in, long_paths_enabled, resolve_target
    from npu_model.profiles import DEFAULT_LLAMA_RUNTIME, DEFAULT_RUNTIME, LLAMA_RUNTIMES, get_runtime
    from npu_model.toolchains import Toolchains
    from npu_model.toolchains.llama_cpp import ConverterSource

    host = detect_host()
    report: dict[str, Any] = {"host": host.to_dict(), "home": str(paths.home()),
                              "long_paths_enabled": long_paths_enabled(), "hexagon_opt_in": hexagon_opted_in()}
    try:
        report["target"] = resolve_target(host, os.environ.get("NPU_MODEL_TARGET")).id
    except NpuModelError as exc:
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
    except NpuModelError as exc:
        report["toolchains_error"] = str(exc)
    source = ConverterSource.load(LLAMA_RUNTIMES[DEFAULT_LLAMA_RUNTIME].llama_cpp_tag)
    report["llama_cpp_converter"] = {"tag": source.tag, "ready": not source.problems()}
    if as_json:
        typer.echo(json.dumps(report, indent=2, default=str))
        return
    fork = report["fork"]
    table = Table(show_header=False, box=None)
    table.add_row("Processor", escape(str(host.processor_name)))
    table.add_row("Architecture", f"{host.native_arch} (this Python: {host.python_platform})")
    table.add_row("NPU", escape(", ".join(host.npu_devices) or "none found"))
    table.add_row("Target", escape(report["target"] or report.get("target_error", "")) +
                  (" (default; --target qnn-x-elite for a QNN bundle)" if report["target"] == "gguf-hexagon" else ""))
    table.add_row("Fork", escape(f"{fork['exe'] or 'NPU_MODEL_OLLAMA_EXE'}: {fork['state']}"
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
                                                                     else "not fetched yet (npu-model setup)")))
    table.add_row("QNN quantization", f"{device}" + (f" ({escape(', '.join(host.cuda_gpus))})" if host.cuda_gpus
                                                      else ""))
    out.print(table)


def _verify_gguf_bundle(bundle: Path, manifest: dict[str, Any], snapshot: Path, level: str,
                        ollama_exe: Optional[Path], host: HostInfo, say: Any) -> dict[str, Any]:
    """Re-run a GGUF bundle's verification with the fork's current runtime."""
    from npu_model import gguf, paths
    from npu_model.checkpoint import read_checkpoint
    from npu_model.gguf_verify import verify_gguf
    from npu_model.profiles import DEFAULT_LLAMA_RUNTIME, DEFAULT_RUNTIME, LLAMA_RUNTIMES, get_runtime, get_target
    from npu_model.sources import ModelRef, ResolvedModel
    from npu_model.toolchains import Toolchains
    from npu_model.toolchains.llama_cpp import ConverterSource

    exe = ollama_exe or (Path(os.environ["NPU_MODEL_OLLAMA_EXE"]) if os.environ.get("NPU_MODEL_OLLAMA_EXE") else None)
    if exe is None:
        raise NpuModelError("verify", "NEEDS_RUNTIME", "A GGUF bundle is verified with the fork's llama.cpp build.",
                            hint="Set NPU_MODEL_OLLAMA_EXE to the fork's ollama.exe (or pass --ollama-exe).")
    payload = gguf.LlamaPayload.from_executable(exe)
    missing = payload.problems()
    if missing:
        raise NpuModelError("verify", "RUNTIME_INCOMPLETE", f"{payload.root} lacks: {', '.join(missing)}",
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
    from npu_model.recipes import FAMILY_TEMPLATES, catalog, catalog_index, verified_conversions

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
              "`npu-model convert MODEL --plan` shows estimates.")


@app.command("verify")
def verify_command(
    bundle: Path = typer.Argument(..., help="Bundle folder"),
    level: str = typer.Option("standard", "--level", help="quick | standard | full"),
    source: Optional[Path] = typer.Option(None, "--source", help="Source model folder (default: pinned snapshot)"),
    ollama_exe: Optional[Path] = typer.Option(None, "--ollama-exe", help="GGUF: fork executable (or "
                                                                          "NPU_MODEL_OLLAMA_EXE)"),
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
        if manifest.get("format") == "gguf":
            report = _verify_gguf_bundle(bundle, manifest, snapshot, level, ollama_exe, host,
                                         _progress(time.time(), as_json))
        else:
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
    from npu_model.install.ollama_import import execute_import, install_gguf, plan_import

    try:
        if (input / "model.gguf").is_file():  # GGUF bundle: FROM + parameters; the GGUF carries its template
            exe = ollama_exe or (Path(os.environ["NPU_MODEL_OLLAMA_EXE"])
                                 if os.environ.get("NPU_MODEL_OLLAMA_EXE") else None)
            if exe is None:
                raise NpuModelError("import", "EXE_REQUIRED", "Installing needs the fork executable.",
                                    hint="Pass --ollama-exe PATH or set NPU_MODEL_OLLAMA_EXE.")
            if not execute:
                typer.echo(json.dumps({"action": "local-create", "format": "gguf", "bundle": str(input), "name": name,
                                       "executable": str(exe), "executed": False,
                                       "next": "Repeat with --execute to run `ollama create` on the fork."}, indent=2))
                return
            result = install_gguf(input, name, executable=exe, host=host or os.environ.get("OLLAMA_HOST"),
                                  work_dir=work_dir / "npu-model-install", overwrite=overwrite, timeout=timeout)
            typer.echo(json.dumps({"action": "local-create", "format": "gguf", "executed": True, "installed": True,
                                   **result}, indent=2))
            return
        plan = plan_import(input, name, executable=ollama_exe, host=host, modelfile=modelfile,
                           overwrite=overwrite, processor_name=processor_name)
        result = (execute_import(plan, expected_fingerprint=expect_fingerprint, work_dir=work_dir,
                                 timeout=timeout) if execute else plan.to_dict())
        typer.echo(json.dumps(result, indent=2))
    except NpuModelError as exc:
        _handle_err(exc)
