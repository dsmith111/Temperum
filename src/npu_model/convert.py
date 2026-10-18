"""The conversion pipeline: plan, then toolchains -> fetch -> calibrate -> Olive -> assemble -> verify.

`plan()` does only cheap work (host detection, Hugging Face metadata, recipe choice, estimates)
so the user can review what will happen. `run()` executes it as a resumable job under
<home>/jobs/<id>: the id is derived from every input, so rerunning the same conversion resumes
(Olive also resumes from its cache) and `job.json` records each stage.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from npu_model import __version__, calib, paths
from npu_model.assemble import MIN_CONTEXT, AssembleInputs, assemble
from npu_model.bundle.inspect import inspect_prebuilt
from npu_model.checkpoint import read_checkpoint
from npu_model.core.errors import NpuModelError
from npu_model.host import HostInfo, detect_host, resolve_target
from npu_model.olive_run import output_complete, run_olive
from npu_model.profiles import DEFAULT_RUNTIME, RuntimeProfile, TargetProfile, get_runtime
from npu_model.recipes import RecipeChoice, RenderOptions, choose, precision_label, render, resolve_quality
from npu_model.sources import ResolvedModel, fetch, parse_ref, resolve
from npu_model.templates import ChatTemplate, custom, from_jinja, load_jinja
from npu_model.toolchains import ENV_SIZE_GB, PYTHON_SIZE_GB, EnvStatus, Toolchains
from npu_model.verify import verify

Progress = Callable[[str], None]

PERMISSIVE_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0", "unlicense"}
VERIFICATION_FILE = "npu_model_verification.json"


@dataclass
class ConvertOptions:
    model: str
    target: str = "auto"
    quality: str = "auto"
    device: str = "auto"  # auto | cpu | cuda
    context: int | None = None
    out: Path | None = None
    verify: str = "standard"
    overwrite: bool = False
    template: Path | None = None
    stops: list[str] = field(default_factory=list)
    num_splits: int | None = None
    keep_cache: bool = False
    runtime: str = DEFAULT_RUNTIME
    python: Path | None = None  # base interpreter for the toolchains (default: uv-managed)


@dataclass
class ConvertPlan:
    options: ConvertOptions
    host: HostInfo
    model: ResolvedModel
    target: TargetProfile
    runtime: RuntimeProfile
    recipe: RecipeChoice
    quality: str
    device: str
    destination: Path
    job_id: str
    toolchains: list[EnvStatus]
    estimates: dict[str, float]
    warnings: list[str]
    verify_level: str
    resumed: list[str] = field(default_factory=list)  # stages an interrupted run of this job finished

    @property
    def job_dir(self) -> Path:
        return paths.jobs_dir() / self.job_id

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model.to_dict(), "target": self.target.id, "runtime": self.runtime.id,
                "recipe": {"repo": self.recipe.recipe.repo, "kind": self.recipe.kind, "note": self.recipe.note,
                           "upstream_path": self.recipe.recipe.upstream_path},
                "quality": self.quality, "device": self.device, "destination": str(self.destination),
                "job": str(self.job_dir), "verify": self.verify_level,
                "toolchains": [t.to_dict() for t in self.toolchains], "estimates": self.estimates,
                "warnings": self.warnings, "resumed": self.resumed, "host": self.host.to_dict()}


def _device(requested: str, host: HostInfo) -> str:
    if requested not in ("auto", "cpu", "cuda"):
        raise NpuModelError("plan", "UNKNOWN_DEVICE", f"Unknown device {requested}", hint="Use auto, cpu or cuda.")
    if requested == "cuda" and not host.cuda_gpus:
        raise NpuModelError("plan", "NO_CUDA", "--device cuda was requested but no NVIDIA GPU was found.")
    if requested == "auto":
        return "cuda" if host.cuda_gpus else "cpu"
    return requested


def _job_id(model: ResolvedModel, recipe: RecipeChoice, target: TargetProfile, runtime: RuntimeProfile,
            quality: str, device: str, options: ConvertOptions) -> str:
    identity = {"model": model.ref.repo or str(model.ref.path), "commit": model.commit,
                "config": hashlib.sha256(json.dumps(model.config, sort_keys=True).encode()).hexdigest(),
                "recipe": recipe.recipe.sha256, "family": recipe.kind, "target": target.id, "runtime": runtime.id,
                "quality": quality, "device": device, "splits": options.num_splits, "converter": __version__}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def disk_free_gb(path: Path) -> float | None:
    """Free space on the volume holding `path` (or its nearest existing parent)."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            return shutil.disk_usage(candidate).free / 2**30
    return None


def _params(model: ResolvedModel) -> int:
    """Parameter count from the weight size (bf16/fp16 checkpoints: 2 bytes per parameter)."""
    dtype = str(model.config.get("torch_dtype") or model.config.get("dtype") or "bfloat16")
    width = 4 if "32" in dtype else 2
    return int((model.weight_bytes or 0) / width)


def estimates(model: ResolvedModel) -> dict[str, float]:
    """Rough resources for a CPU conversion, fitted to measured runs on the X1E78100.

    Olive peak RSS was 4.5 GB (0.36B), 3.8 GB (0.49B) and 7.9 GB (1.54B); RAM is that fit plus
    10%. Olive took 7 min (0.36B) and 44 min (1.54B, one compile retry); verification adds a
    few minutes. Olive's cache for 1.54B was 18 GB.
    """
    params = _params(model)
    billions = params / 1e9
    return {"params_billions": round(billions, 2),
            "download_gb": round((model.weight_bytes or 0) / 2**30, 1),
            "disk_gb": round(((model.weight_bytes or 0) + params * 13.5) / 2**30, 1),
            "ram_gb": round((3.5 + params * 3 / 2**30) * 1.1, 1),
            "minutes": round(5 + 28 * billions)}


def plan(options: ConvertOptions, host: HostInfo | None = None, *, resolver: Callable[..., ResolvedModel] = resolve,
         chains_factory: Callable[..., Toolchains] = Toolchains) -> ConvertPlan:
    host = host or detect_host()
    target = resolve_target(host, options.target)
    if target.format != "qnn":
        raise NpuModelError("plan", "WRONG_TARGET_FORMAT", f"{target.id} is a {target.format} target",
                            hint="GGUF targets are converted by npu_model.gguf (the CLI routes them there).")
    runtime = get_runtime(options.runtime)
    model = resolver(parse_ref(options.model))
    recipe = choose(model.ref.repo, model.model_type, model.architecture)
    device = _device(options.device, host)
    quality = resolve_quality(options.quality, device)
    on_target = target.matches(host.processor_name)
    level = options.verify if on_target else "none"
    job_id = _job_id(model, recipe, target, runtime, quality, device, options)
    destination = options.out or paths.models_dir() / f"{model.name}-{target.id}"
    # Checks that would otherwise fail only after the long quantize/compile stage.
    if destination.exists() and not options.overwrite and not _resumable_bundle(paths.jobs_dir() / job_id,
                                                                                  destination):
        raise NpuModelError("plan", "DESTINATION_EXISTS", f"{destination} already exists.",
                            hint="Pass --overwrite to rebuild it, --out to write elsewhere, or "
                                 "`npu-model verify` to re-check the existing bundle.")
    maximum = model.config.get("max_position_embeddings")
    if options.context is not None and (options.context < MIN_CONTEXT
                                        or (isinstance(maximum, int) and options.context > maximum)):
        raise NpuModelError("plan", "BAD_CONTEXT",
                            f"--context {options.context} is outside {MIN_CONTEXT}..{maximum or 'the model maximum'}.")
    chains = chains_factory(host, runtime, device, python=options.python)
    roles = ["quantize", "compile"] + (["verify"] if level != "none" else [])
    statuses = [chains.status(role) for role in roles]
    warnings: list[str] = []
    if recipe.kind == "family":
        warnings.append(recipe.note)
    if not target.verified:
        warnings.append(f"target {target.id} has not been verified on hardware")
    if not on_target:
        warnings.append(f"this machine is not a {target.id} device: the bundle is compiled but cannot be verified "
                        "here, and no compatibility declaration is written")
    if device == "cuda":
        warnings.append("the CUDA quantization path has not been tested yet (decision D8)")
    est = estimates(model)
    resumed = _finished_stages(paths.jobs_dir() / job_id)
    if "olive" not in resumed:  # once quantize/compile is done, the rest needs little memory or disk
        warnings += _resource_checks(est, host)
    license_id = (model.license or "unknown").lower()
    if license_id not in PERMISSIVE_LICENSES:
        warnings.append(f"licence '{model.license or 'unknown'}': you are responsible for complying with its terms")
    missing = [s.role for s in statuses if s.state != "ready"]
    if missing:
        size = sum(ENV_SIZE_GB[r] for r in missing) + PYTHON_SIZE_GB
        warnings.append(f"first use: sets up {', '.join(missing)} toolchains (~{size:.1f} GB; torch comes from "
                        "download.pytorch.org, everything else through your pip configuration)")
    return ConvertPlan(options, host, model, target, runtime, recipe, quality, device, destination, job_id,
                       statuses, est, warnings, level, resumed)


def _resource_checks(est: dict[str, float], host: HostInfo) -> list[str]:
    """Fail now on memory or disk the conversion cannot get; warn when memory is tight."""
    if host.total_ram_gb and est["ram_gb"] > host.total_ram_gb * 0.9:
        raise NpuModelError("plan", "NOT_ENOUGH_MEMORY",
                            f"Converting needs about {est['ram_gb']} GB of RAM; this machine has {host.total_ram_gb} GB.",
                            hint="Use a smaller model, or convert on a machine with more memory (--target).")
    free_gb = disk_free_gb(paths.home())
    if free_gb is not None and est["disk_gb"] > free_gb:
        raise NpuModelError("plan", "NOT_ENOUGH_DISK",
                            f"Converting needs about {est['disk_gb']} GB of disk; {free_gb:.1f} GB is free at "
                            f"{paths.home()}.", hint="Free some space or set NPU_MODEL_HOME to another drive.")
    if host.available_commit_gb is not None and est["ram_gb"] > host.available_commit_gb:
        return [f"Windows can commit only {host.available_commit_gb} GB more memory right now and the conversion "
                f"needs about {est['ram_gb']} GB: close applications first, or the conversion may fail and other "
                "programs may run out of memory"]
    if host.available_ram_gb and est["ram_gb"] > host.available_ram_gb:
        return [f"about {est['ram_gb']} GB of RAM is needed and {host.available_ram_gb} GB is free; "
                "close other applications or expect heavy paging"]
    return []


def _finished_stages(job_dir: Path) -> list[str]:
    """Stages an earlier, interrupted run of this job completed (empty for a new or finished job)."""
    try:
        stages = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))["stages"]
        done = [name for name, entry in stages.items() if entry.get("status") == "done"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []
    return [] if "verify" in done else done


def _resumable_bundle(job_dir: Path, destination: Path) -> str | None:
    """Fingerprint of the bundle that an interrupted run of this job assembled at destination.

    None when the job never assembled there or already finished verifying it, so an existing
    folder is someone else's (or a finished) bundle that must not be replaced silently.
    """
    try:
        stages = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))["stages"]
        assembled, verified = stages.get("assemble") or {}, stages.get("verify") or {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if assembled.get("status") != "done" or verified.get("status") == "done":
        return None
    if Path(str(assembled.get("bundle") or "")) != destination:
        return None
    return str(assembled.get("fingerprint") or "") or None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":  # os.kill(pid, 0) would terminate the process on Windows
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return bool(getattr(ctypes, "get_last_error")() == 5)  # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Job:
    """Receipt for one conversion: stage status and timing in job.json (no credentials).

    A lock file holding the owner's process id keeps two runs of the same conversion apart;
    a lock left by a killed run is taken over.
    """

    def __init__(self, directory: Path, plan_data: dict[str, Any]) -> None:
        self.dir = directory
        self.path = directory / "job.json"
        self.lock = directory / "job.lock"
        directory.mkdir(parents=True, exist_ok=True)
        self._acquire()
        previous = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        self.data: dict[str, Any] = {"plan": plan_data, "stages": previous.get("stages", {}),
                                     "created": previous.get("created") or _now(), "result": None}
        self.save()

    def _acquire(self) -> None:
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = int(self.lock.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner != os.getpid() and _pid_alive(owner):
                raise NpuModelError("convert", "JOB_RUNNING",
                                    f"Another npu-model process (pid {owner}) is running this conversion.",
                                    hint="Wait for it to finish; rerunning afterwards resumes.") from None
            self.lock.unlink(missing_ok=True)  # left by a killed run
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))

    def release(self) -> None:
        self.lock.unlink(missing_ok=True)

    def done(self, stage: str) -> bool:
        return bool(self.data["stages"].get(stage, {}).get("status") == "done")

    def record(self, stage: str, status: str, **details: Any) -> None:
        entry = self.data["stages"].setdefault(stage, {})
        entry.update({"status": status, "at": _now(), **details})
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _chat_template(options: ConvertOptions, snapshot: Path) -> ChatTemplate:
    if options.template:
        return custom(options.template.read_text(encoding="utf-8"), options.stops)
    return from_jinja(load_jinja(snapshot))


@dataclass
class ConvertResult:
    bundle: Path
    job_dir: Path
    verification: dict[str, Any]
    olive: dict[str, Any]


def run(conversion: ConvertPlan, progress: Progress | None = None, *,
        chains_factory: Callable[..., Toolchains] = Toolchains) -> ConvertResult:
    job = Job(conversion.job_dir, conversion.to_dict())
    try:
        return _run(conversion, job, progress or (lambda _msg: None), chains_factory)
    finally:
        job.release()


def _run(conversion: ConvertPlan, job: Job, say: Progress, chains_factory: Callable[..., Toolchains]) -> ConvertResult:
    options = conversion.options
    chains = chains_factory(conversion.host, conversion.runtime, conversion.device, python=options.python,
                            progress=say)

    roles = ["quantize", "compile"] + (["verify"] if conversion.verify_level != "none" else [])
    for role in roles:
        status = chains.ensure(role)
        job.record(f"toolchain-{role}", "done", path=str(status.path))

    say(f"fetching {conversion.model.ref.display}")
    snapshot = fetch(conversion.model, paths.sources_dir())
    job.record("fetch", "done", commit=conversion.model.commit)
    corpus = calib.ensure_corpus(calib.CALIBRATION, paths.data_dir())
    job.record("calibration-data", "done", **{k: v for k, v in corpus.to_dict().items() if k != "path"})
    chat = _chat_template(options, snapshot)

    info = read_checkpoint(snapshot)
    olive_out = job.dir / "out"
    render_opts = RenderOptions(
        model_dir=snapshot, target=conversion.target, quality=conversion.quality, device=conversion.device,
        corpus=corpus.path, olive_data_script=Path(__file__).resolve().parent / "envscripts" / "olive_data.py",
        compile_python_dir=chains.scripts_dir("compile"), output_dir=olive_out, cache_dir=job.dir / "cache",
        family=conversion.recipe.kind == "family", modules=info.modules, block_params=info.block_params,
        num_splits=options.num_splits)
    config = render(conversion.recipe.recipe, render_opts)
    precision = precision_label(config)
    # Reuse Olive's output only if it came from this exact config: a newer converter may render
    # the same job differently (for example with more graph splits).
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    olive_stage = job.data["stages"].get("olive", {})
    if job.done("olive") and olive_stage.get("config") == config_hash and output_complete(olive_out):
        say("reusing the finished quantize/compile stage")
        olive = olive_stage.get("result", {})
    else:
        if olive_out.exists() and olive_stage.get("config") != config_hash:
            shutil.rmtree(olive_out)  # output of a different config; Olive's cache still reuses what matches
        say(f"quantizing and compiling with Olive ({conversion.recipe.recipe.repo} recipe, {precision}, "
            f"{conversion.device}); about {conversion.estimates['minutes']:.0f} minutes")
        job.record("olive", "running", config=config_hash)
        result = run_olive(chains.python_exe("quantize"), config, job.dir / "olive", progress=say)
        olive = result.to_dict()
        if not result.ok:
            job.record("olive", "failed", result=olive)
            raise NpuModelError("convert", "OLIVE_FAILED",
                                f"Olive failed in {result.failed_pass or 'an unknown pass'}: {result.error}",
                                hint=f"Logs: {', '.join(result.logs)}. Rerunning the same command resumes from "
                                     "the last completed pass.")
        job.record("olive", "done", result=olive, config=config_hash)

    source = {"repo": conversion.model.ref.repo, "commit": conversion.model.commit,
              "license": conversion.model.license, "architecture": conversion.model.architecture}
    provenance = {
        "recipe": {"repo": conversion.recipe.recipe.repo, "kind": conversion.recipe.kind,
                   "upstream_path": conversion.recipe.recipe.upstream_path, "sha256": conversion.recipe.recipe.sha256},
        "quality": conversion.quality, "device": conversion.device, "num_splits": _splits(config),
        "toolchains": {role: chains.lock(role).path.name for role in roles},
        "calibration": {k: v for k, v in corpus.to_dict().items() if k != "path"},
        "olive": {"seconds": olive.get("seconds"), "peak_rss_gb": olive.get("peak_rss_gb"),
                  "attempts": olive.get("attempts")},
    }
    processor = conversion.host.processor_name if conversion.target.matches(conversion.host.processor_name) else None
    assembled = _resumable_bundle(job.dir, conversion.destination)
    previous = inspect_prebuilt(conversion.destination, processor_name=processor) if assembled else None
    if previous is not None and previous.importable and previous.content_fingerprint == assembled:
        say("reusing the bundle assembled before the interruption")
        bundle = conversion.destination
    else:
        say("assembling the bundle")
        bundle, inspection = assemble(AssembleInputs(
            olive_output=olive_out, snapshot=snapshot, destination=conversion.destination, target=conversion.target,
            runtime=conversion.runtime, chat=chat, context=options.context, precision=precision,
            processor_name=processor, source=source, provenance=provenance,
            overwrite=options.overwrite or assembled is not None))
        job.record("assemble", "done", bundle=str(bundle), fingerprint=inspection.content_fingerprint)

    heldout = calib.ensure_corpus(calib.HELDOUT, paths.data_dir()).path if conversion.verify_level in (
        "standard", "full") else None
    report = verify(bundle, snapshot, chat, conversion.verify_level,
                    quantize_python=chains.python_exe("quantize"),
                    verify_python=chains.python_exe("verify") if conversion.verify_level != "none" else Path(),
                    work=job.dir / "verify", heldout=heldout, params=info.total_params,
                    available_gb=conversion.host.available_ram_gb,
                    total_gb=conversion.host.total_ram_gb, progress=say)
    report["created"] = _now()
    (bundle / VERIFICATION_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    job.record("verify", "done", verdict=report["verdict"], reasons=report.get("reasons", []))
    # A failed bundle is usually fixed by recompiling (for example with more splits): keep
    # Olive's cache so the rerun reuses the quantization instead of recalibrating.
    if not options.keep_cache and report["verdict"] != "failed":
        shutil.rmtree(job.dir / "cache", ignore_errors=True)
    job.data["result"] = {"bundle": str(bundle), "verdict": report["verdict"]}
    job.save()
    return ConvertResult(bundle, job.dir, report, olive)


def _splits(config: dict[str, Any]) -> int | None:
    return next((p.get("num_splits") for p in config["passes"].values() if p["type"] == "CaptureSplitInfo"), None)


def elapsed(start: float) -> str:
    minutes, seconds = divmod(int(time.time() - start), 60)
    return f"{minutes}m{seconds:02d}s"
