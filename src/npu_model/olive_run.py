"""Run a rendered Olive config in the quantize environment and decide whether it worked.

Olive exits 0 even when a pass fails, so success means the output folder holds a complete
GenAI pipeline. The QNN context-generation crash seen in M0 (access violation after all
partitions compile, which an identical rerun does not reproduce) is retried; Olive resumes
from its cache, so a retry only repeats the compile pass.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from npu_model.core.errors import NpuModelError

Progress = Callable[[str], None]
Popen = Callable[..., "subprocess.Popen[str]"]

ENVSCRIPTS = Path(__file__).resolve().parent / "envscripts"
# Windows NTSTATUS crash codes as Python reports a child's exit status (unsigned and signed).
CRASH_STATUSES = {"3221225477", "-1073741819", "3221226505", "-1073740791"}  # 0xC0000005, 0xC0000409
RETRYABLE_PASSES = {"epcontextbinarygenerator"}
_PASS_RE = re.compile(r"Running pass (\S+)")
_STATUS_RE = re.compile(r"non-zero exit status (-?\d+)")
_ERROR_RE = re.compile(r"^(?:\w+\.)*\w*(?:Error|Exception):\s.+$")


@dataclass
class OliveResult:
    ok: bool
    attempts: int
    seconds: float = 0.0
    peak_rss_gb: float | None = None
    passes: list[dict[str, Any]] = field(default_factory=list)
    failed_pass: str | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def olive_env() -> dict[str, str]:
    """Environment for Olive: no telemetry, no hub access (inputs are local), UTF-8 I/O."""
    env = {**os.environ, "OLIVE_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop("HF_TOKEN", None)  # nothing in the run needs credentials
    return env


def pipeline_files(output_dir: Path) -> list[str] | None:
    """Model files named by genai_config.json, or None when the config is missing/unreadable."""
    try:
        config = json.loads((output_dir / "genai_config.json").read_text(encoding="utf-8"))
        decoder = config["model"]["decoder"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    files: list[str] = []
    for stage in decoder.get("pipeline", []):
        files.extend(str(spec["filename"]) for spec in stage.values() if isinstance(spec, dict) and "filename" in spec)
    if decoder.get("filename"):
        files.append(str(decoder["filename"]))
    return files


def output_complete(output_dir: Path) -> bool:
    files = pipeline_files(output_dir)
    return bool(files) and all((output_dir / name).is_file() for name in files or [])


def diagnose(log_text: str) -> tuple[str | None, str | None, bool]:
    """(pass that was running at the end, last error line, whether it is a retryable crash)."""
    passes = _PASS_RE.findall(log_text)
    failed = passes[-1] if passes else None
    errors = [line.strip() for line in log_text.splitlines() if _ERROR_RE.match(line.strip())]
    statuses = _STATUS_RE.findall(log_text)
    kind = failed.split(":", 1)[-1].lower() if failed else ""
    crash = bool(statuses) and statuses[-1] in CRASH_STATUSES and kind in RETRYABLE_PASSES
    # A pass running in another environment fails as CalledProcessError; its own error is
    # printed earlier (from the child's stderr) and is the useful one.
    root = [e for e in errors if not e.startswith("subprocess.CalledProcessError")]
    if root:
        error: str | None = root[-1]
    elif statuses:
        error = f"a pass subprocess exited with status {statuses[-1]}" + (
            " (Windows crash: access violation or stack overrun)" if statuses[-1] in CRASH_STATUSES else "")
    else:
        error = "No output model produced" if "No output model produced" in log_text else None
    return failed, error[:500] if error else None, crash


def run_olive(python: Path, config: dict[str, Any], work_dir: Path, *, attempts: int = 3,
              progress: Progress | None = None, popen: Popen = subprocess.Popen) -> OliveResult:
    """Run Olive for `config` (its output_dir must be set) with up to `attempts` tries."""
    say = progress or (lambda _msg: None)
    output_dir = Path(config["output_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "olive_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    result = OliveResult(ok=False, attempts=0)
    first_log = len(list(work_dir.glob("olive-*.log"))) + 1  # keep logs of interrupted earlier runs
    for attempt in range(1, attempts + 1):
        log = work_dir / f"olive-{first_log + attempt - 1}.log"
        result.attempts = attempt
        result.logs.append(str(log))
        cmd = [str(python), str(ENVSCRIPTS / "run_olive.py"), "--config", str(config_path), "--log", str(log),
               "--cwd", str(work_dir)]
        try:
            proc = popen(cmd, cwd=str(work_dir), env=olive_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
        except OSError as exc:
            raise NpuModelError("convert", "OLIVE_START_FAILED", f"Could not start Olive: {exc}") from exc
        summary: dict[str, Any] = {}
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.startswith("@@pass "):
                name = line.split()[1]
                say(f"pass {name}")
            elif line.startswith("@@summary "):
                summary = json.loads(line[len("@@summary "):])
        proc.wait()
        result.seconds += float(summary.get("seconds", 0.0))
        peak = summary.get("peak_rss_gb")
        if peak is not None:
            result.peak_rss_gb = max(result.peak_rss_gb or 0.0, float(peak))
        result.passes.extend(summary.get("passes", []))
        if output_complete(output_dir):
            result.ok, result.failed_pass, result.error = True, None, None
            return result
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        result.failed_pass, result.error, crash = diagnose(text)
        if not crash or attempt == attempts:
            break
        say(f"the NPU compiler crashed in {result.failed_pass} (known intermittent QNN issue); "
            f"retrying ({attempt + 1}/{attempts}) from Olive's cache")
    return result
