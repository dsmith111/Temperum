from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.markup import escape

from npu_model.core.errors import NpuModelError

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Convert Hugging Face models into NPU-ready bundles for the Ollama fork.")


def _handle_err(e: Exception) -> None:
    if isinstance(e, NpuModelError):
        code = escape(f"[{e.stage}/{e.reason_code}]")
        rprint(f"[bold red]Error[/bold red] {code}: {escape(str(e))}")
        if e.hint:
            rprint(f"[yellow]Hint:[/yellow] {escape(e.hint)}")
        raise typer.Exit(code=2)
    raise e


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
