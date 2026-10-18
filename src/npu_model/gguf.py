"""GGUF output target (decision D7): a Hugging Face model as one GGUF file for the NPU.

llama.cpp's Hexagon backend runs GGUF models on the Snapdragon NPU. The same file works on
every HTP version and on the CPU, so nothing is compiled ahead of time:

  resolve -> plan -> fetch -> convert (llama.cpp's convert_hf_to_gguf.py, in the quantize
  environment) -> quantize (the runtime's llama-quantize; Q4_0 by default, the type the
  Hexagon backend supports best) -> assemble (model.gguf, Modelfile, licence, manifest) ->
  verify (the runtime's llama-server on the NPU) -> install

The runtime is the Ollama fork's llama.cpp payload (lib/ollama next to the fork executable).
Using its llama-quantize and llama-server keeps the converter, the file and the runtime on
one llama.cpp commit; the converter source is pinned to that same commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from npu_model import __version__, paths
from npu_model.core.errors import NpuModelError
from npu_model.host import HEXAGON_OPT_IN, HostInfo, hexagon_opted_in
from npu_model.profiles import (DEFAULT_LLAMA_RUNTIME, DEFAULT_RUNTIME, LLAMA_RUNTIMES, LlamaRuntimeProfile,
                                TargetProfile, get_runtime)
from npu_model.sources import ResolvedModel, fetch
from npu_model.toolchains import ENV_SIZE_GB, Toolchains
from npu_model.toolchains.llama_cpp import ConverterSource

Progress = Callable[[str], None]
Runner = Callable[..., "subprocess.CompletedProcess[str]"]

QUANT_TYPES = ("Q4_0", "Q8_0", "Q4_K_M")  # all run on the Hexagon backend; Q4_0 is its best-supported type
DEFAULT_QUANT = "Q4_0"
DEFAULT_CONTEXT = 4096
MANIFEST = "npu_model_manifest.json"
LICENSE_PREFIXES = ("license", "notice", "use_policy", "readme")
HTP_LIBRARIES = ("libggml-htp-v73.so", "libggml-htp-v75.so", "libggml-htp-v79.so", "libggml-htp-v81.so",
                 "libggml-htp.cat")
# How users get the signed HTP set; the fork shows Qualcomm's terms first.
SETUP_COMMAND = "ollama npu setup --provider hexagon --accept-qualcomm-terms"
# Approximate GGUF bytes per parameter by type (block scales included).
BYTES_PER_PARAM = {"Q4_0": 0.5625, "Q8_0": 1.0625, "Q4_K_M": 0.6, "BF16": 2.0, "F16": 2.0}
_REGISTER = re.compile(r"@ModelBase\.register\(([^)]*)\)")


def _exe(name: str) -> str:
    return name + (".exe" if sys.platform == "win32" else "")


@dataclass(frozen=True)
class LlamaPayload:
    """The llama.cpp runtime files of an Ollama fork installation (<fork dir>/lib/ollama)."""

    root: Path
    htp: Path | None = None  # the signed HTP set the fork's runners use, if any

    @classmethod
    def from_executable(cls, executable: Path, run: Runner = subprocess.run) -> LlamaPayload:
        root = executable.resolve().parent / "lib" / "ollama"
        return cls(root, signed_htp_set(executable, root / "hexagon", run))

    @property
    def quantize(self) -> Path:
        return self.root / _exe("llama-quantize")

    @property
    def server(self) -> Path:
        return self.root / _exe("llama-server")

    @property
    def hexagon(self) -> Path:
        return self.root / "hexagon"

    def problems(self) -> list[str]:
        missing = [p.name for p in (self.quantize, self.server) if not p.is_file()]
        if not any(self.hexagon.glob("*ggml-hexagon*")):
            missing.append("hexagon/ggml-hexagon")
        if self.htp is None:
            missing.append(f"the signed HTP libraries ({SETUP_COMMAND})")
        return missing

    def version(self, run: Runner = subprocess.run) -> str | None:
        """Commit of the payload's llama.cpp build, from `llama-server --version`."""
        try:
            proc = run([str(self.server), "--version"], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PATH": f"{self.root}{os.pathsep}{os.environ.get('PATH', '')}"})
        except (OSError, subprocess.TimeoutExpired):
            return None
        # "version: 0.6.0-dev (build 1, commit e467e5b)"; older builds print "version: 11521 (b42b7e6)"
        match = re.search(r"commit ([0-9a-f]{7,40})|version:\s*\d+\s*\(([0-9a-f]{7,40})\)",
                          (proc.stdout or "") + (proc.stderr or ""))
        return (match.group(1) or match.group(2)) if match else None


def signed_htp_set(executable: Path, backend: Path, run: Runner = subprocess.run) -> Path | None:
    """The signed HTP set the fork's runners would use, as the fork reports it.

    Packages leave the set out (Qualcomm's terms restrict its use); `ollama npu setup --provider hexagon`
    installs it, and the fork checks it against its pin. Builds from before that command can only use a set
    bundled beside the backend.
    """
    try:
        proc = run([str(executable), "npu", "setup", "--provider", "hexagon", "--validate-only", "--json"],
                   capture_output=True, text=True, timeout=60)
        report = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        report = {}
    checks = report.get("checks") if isinstance(report, dict) else None
    stages = {c.get("stage"): c.get("status") for c in checks or [] if isinstance(c, dict)}
    if "htp_integrity" in stages:  # older builds answer --validate-only for the QNN runtime instead
        return Path(report["dir"]) if stages["htp_integrity"] == "pass" and report.get("dir") else None
    return backend if all((backend / name).is_file() for name in HTP_LIBRARIES) else None


def supported_architectures(source_dir: Path) -> set[str]:
    """Hugging Face architecture names the pinned converter registers."""
    names: set[str] = set()
    for path in [source_dir / "convert_hf_to_gguf.py", *sorted((source_dir / "conversion").glob("*.py"))]:
        if path.is_file():
            for args in _REGISTER.findall(path.read_text(encoding="utf-8", errors="replace")):
                names.update(re.findall(r"[\"']([A-Za-z0-9_]+)[\"']", args))
    return names


def estimates(model: ResolvedModel, quant: str) -> dict[str, float]:
    width = 4 if "32" in str(model.config.get("torch_dtype") or model.config.get("dtype") or "") else 2
    params = (model.weight_bytes or 0) / width
    return {"params_billions": round(params / 1e9, 2),
            "download_gb": round((model.weight_bytes or 0) / 2**30, 1),
            "gguf_gb": round(params * BYTES_PER_PARAM[quant] / 2**30, 1),
            # the 16-bit intermediate, the quantized file and the bundle copy
            "disk_gb": round(params * (2 + 2 * BYTES_PER_PARAM[quant]) / 2**30 + 0.5, 1),
            "ram_gb": round(params * 2.2 / 2**30 + 1, 1),
            "minutes": round(2 + 3 * params / 1e9)}


def job_id(model: ResolvedModel, target: TargetProfile, runtime: LlamaRuntimeProfile, quant: str,
           context: int | None) -> str:
    identity = {"model": model.ref.repo or str(model.ref.path), "commit": model.commit, "target": target.id,
                "runtime": runtime.id, "llama_cpp": runtime.llama_cpp_commit, "quant": quant,
                "context": context, "converter": __version__}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class GgufPlan:
    model: ResolvedModel
    host: HostInfo
    target: TargetProfile
    runtime: LlamaRuntimeProfile
    payload: LlamaPayload
    quant: str
    context: int
    destination: Path
    job_id: str
    estimates: dict[str, float]
    verify_level: str
    warnings: list[str] = field(default_factory=list)
    python: Path | None = None  # base interpreter for the quantize toolchain
    overwrite: bool = False

    @property
    def job_dir(self) -> Path:
        return paths.jobs_dir() / self.job_id

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model.to_dict(), "target": self.target.id, "runtime": self.runtime.id,
                "llama_cpp": self.runtime.llama_cpp_commit, "payload": str(self.payload.root),
                "htp": str(self.payload.htp) if self.payload.htp else None,
                "quant": self.quant, "context": self.context, "destination": str(self.destination),
                "job": str(self.job_dir), "verify": self.verify_level, "estimates": self.estimates,
                "warnings": self.warnings, "host": self.host.to_dict()}


def plan(model: ResolvedModel, host: HostInfo, target: TargetProfile, *, ollama_exe: Path | None,
         quant: str = DEFAULT_QUANT, context: int | None = None, out: Path | None = None, verify: str = "standard",
         overwrite: bool = False, python: Path | None = None, runtime_id: str = DEFAULT_LLAMA_RUNTIME,
         run: Runner = subprocess.run, chains_factory: Callable[..., Toolchains] = Toolchains) -> GgufPlan:
    if target.format != "gguf":
        raise NpuModelError("plan", "WRONG_TARGET_FORMAT", f"{target.id} is not a GGUF target")
    if quant not in QUANT_TYPES:
        raise NpuModelError("plan", "UNKNOWN_QUANT", f"Unknown GGUF type {quant}", hint=f"Use {', '.join(QUANT_TYPES)}.")
    runtime = LLAMA_RUNTIMES[runtime_id]
    if ollama_exe is None:
        raise NpuModelError("plan", "NEEDS_RUNTIME",
                            "The GGUF target quantizes and verifies with the Ollama fork's own llama.cpp build.",
                            hint="Pass --ollama-exe with the fork executable (its lib\\ollama folder is used).")
    payload = LlamaPayload.from_executable(ollama_exe, run)
    missing = payload.problems()
    if missing:
        raise NpuModelError("plan", "RUNTIME_INCOMPLETE",
                            f"{payload.root} lacks the llama.cpp Hexagon runtime: {', '.join(missing)}",
                            hint=f"Use a fork build with the hexagon_arm64 preset; get Qualcomm's signed HTP libraries "
                                 f"with `{SETUP_COMMAND}`.")
    warnings: list[str] = []
    commit = payload.version(run)
    if commit is None:
        warnings.append("could not read the runtime's llama.cpp version")
    elif not runtime.llama_cpp_commit.startswith(commit):
        warnings.append(f"the runtime's llama.cpp is {commit}, but this converter pins {runtime.llama_cpp_commit[:9]} "
                        f"({runtime.llama_cpp_tag}); the HTP libraries and file format follow the runtime")
    maximum = model.config.get("max_position_embeddings")
    ctx = context or min(int(maximum) if isinstance(maximum, int) else DEFAULT_CONTEXT, DEFAULT_CONTEXT)
    if isinstance(maximum, int) and not 128 <= ctx <= maximum:
        raise NpuModelError("plan", "BAD_CONTEXT", f"--context {ctx} is outside 128..{maximum}.")
    destination = out or paths.models_dir() / f"{model.name}-{target.id}"
    if destination.exists() and not overwrite:
        raise NpuModelError("plan", "DESTINATION_EXISTS", f"{destination} already exists.",
                            hint="Pass --overwrite to rebuild it or --out to write elsewhere.")
    on_target = target.matches(host.processor_name)
    if not on_target:
        warnings.append(f"this machine is not a {target.id} device: the file is built but not verified here")
    if not target.verified:
        warnings.append(f"target {target.id} has not been verified on hardware yet")
    if not hexagon_opted_in():
        warnings.append(f"the fork uses the Hexagon NPU only when its server starts with {HEXAGON_OPT_IN}=1")
    est = estimates(model, quant)
    status = chains_factory(host, get_runtime(DEFAULT_RUNTIME), "cpu", python=python).status("quantize")
    if status.state != "ready":
        warnings.append(f"first use: sets up the quantize toolchain (~{ENV_SIZE_GB['quantize']:.0f} GB, about 8 "
                        "minutes; torch comes from download.pytorch.org, everything else through your pip "
                        "configuration)")
    free = shutil.disk_usage(paths.home() if paths.home().exists() else Path.home()).free / 2**30
    if est["disk_gb"] > free:
        raise NpuModelError("plan", "NOT_ENOUGH_DISK", f"Converting needs about {est['disk_gb']} GB of disk; "
                            f"{free:.1f} GB is free.", hint="Free some space or set NPU_MODEL_HOME to another drive.")
    identity = job_id(model, target, runtime, quant, context)
    return GgufPlan(model, host, target, runtime, payload, quant, ctx, destination, identity, est,
                    verify if on_target else "none", warnings, python, overwrite)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream(cmd: list[str], env: dict[str, str] | None, log: Path, say: Progress, keep: Callable[[str], bool],
            popen: Callable[..., Any] = subprocess.Popen) -> tuple[int, str]:
    """Run cmd, tee its output to log, report selected lines; returns (exit code, last lines)."""
    tail: list[str] = []
    with log.open("a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        proc = popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                     encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            tail = (tail + [line.rstrip()])[-40:]
            if keep(line):
                say(line.strip()[:160])
        code = proc.wait()
    return code, "\n".join(tail)


def convert(python: Path, source: ConverterSource, snapshot: Path, out: Path, log: Path, say: Progress,
            popen: Callable[..., Any] = subprocess.Popen) -> Path:
    """HF snapshot -> GGUF in the model's own 16-bit type (convert_hf_to_gguf --outtype auto)."""
    partial = out.with_name(out.name + ".partial")
    partial.unlink(missing_ok=True)
    code, tail = _stream([str(python), str(source.script), str(snapshot), "--outfile", str(partial),
                          "--outtype", "auto"], source.environment(), log, say,
                         lambda line: "Model successfully exported" in line, popen)
    if code != 0 or not partial.is_file():
        raise NpuModelError("convert", "GGUF_CONVERT_FAILED", f"llama.cpp's converter failed (exit {code}).",
                            hint=f"Log: {log}\n{tail[-800:]}")
    partial.replace(out)
    return out


def quantize(payload: LlamaPayload, src: Path, out: Path, quant: str, log: Path, say: Progress,
             popen: Callable[..., Any] = subprocess.Popen) -> Path:
    partial = out.with_name(out.name + ".partial")
    partial.unlink(missing_ok=True)
    env = {**os.environ, "PATH": f"{payload.root}{os.pathsep}{os.environ.get('PATH', '')}"}
    code, tail = _stream([str(payload.quantize), str(src), str(partial), quant], env, log, say,
                         lambda line: line.startswith(("llama_model_quantize_impl: model size",
                                                       "llama_model_quantize_impl: quant size")), popen)
    if code != 0 or not partial.is_file():
        raise NpuModelError("convert", "GGUF_QUANTIZE_FAILED", f"llama-quantize failed (exit {code}).",
                            hint=f"Log: {log}\n{tail[-800:]}")
    partial.replace(out)
    return out


def modelfile(context: int) -> str:
    # The GGUF carries the model's own chat template (tokenizer.chat_template); the fork renders
    # it through llama-server, so no Go TEMPLATE is written.
    return f"FROM ./model.gguf\nPARAMETER num_ctx {context}\n"


def assemble(gguf: Path, snapshot: Path, plan: GgufPlan, provenance: dict[str, Any]) -> Path:
    final = plan.destination
    staging = final.with_name(final.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        os.link(gguf, staging / "model.gguf")
    except OSError:
        shutil.copy2(gguf, staging / "model.gguf")
    (staging / "Modelfile").write_text(modelfile(plan.context), encoding="utf-8", newline="\n")
    licences = []
    for item in sorted(snapshot.iterdir()):
        if item.is_file() and item.name.lower().startswith(LICENSE_PREFIXES):
            shutil.copy2(item, staging / item.name)
            licences.append(item.name)
    files = {p.name: {"size": p.stat().st_size, "sha256": _sha256(p)} for p in sorted(staging.iterdir())}
    manifest = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "converter": {"name": "npu-model", "version": __version__},
        "format": "gguf",
        "source": {"repo": plan.model.ref.repo, "commit": plan.model.commit, "license": plan.model.license,
                   "architecture": plan.model.architecture},
        "target": {"id": plan.target.id, "htp_arch": plan.target.htp_arch},
        "runtime": {"id": plan.runtime.id, "llama_cpp": plan.runtime.llama_cpp_commit,
                    "htp_libraries": plan.runtime.htp_libraries},
        "precision": plan.quant,
        "context_length": plan.context,
        "licence_files": licences,
        "provenance": provenance,
        "files": files,
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    return final


@dataclass
class GgufResult:
    bundle: Path
    job_dir: Path
    verification: dict[str, Any]


def run(gguf_plan: GgufPlan, progress: Progress | None = None,
        chains_factory: Callable[..., Toolchains] = Toolchains,
        verifier: Callable[..., dict[str, Any]] | None = None) -> GgufResult:
    from npu_model.convert import Job  # the same receipt and single-writer lock as the QNN target

    say = progress or (lambda _msg: None)
    job = Job(gguf_plan.job_dir, gguf_plan.to_dict())
    try:
        chains = chains_factory(gguf_plan.host, get_runtime(DEFAULT_RUNTIME), "cpu", python=gguf_plan.python,
                                progress=say)
        chains.ensure("quantize")  # torch + transformers for the converter and the source-model reference
        job.record("toolchain-quantize", "done")
        say(f"fetching {gguf_plan.model.ref.display}")
        snapshot = fetch(gguf_plan.model, paths.sources_dir())
        job.record("fetch", "done", commit=gguf_plan.model.commit)
        source = ConverterSource.load(gguf_plan.runtime.llama_cpp_tag)
        source.ensure(progress=say)
        architecture = gguf_plan.model.architecture
        if architecture and architecture not in supported_architectures(source.directory):
            raise NpuModelError("convert", "UNSUPPORTED_ARCHITECTURE",
                                f"llama.cpp {source.tag} cannot convert {architecture}.",
                                hint="Try the QNN target, or a model with a supported architecture.")
        job.record("converter", "done", tag=source.tag, commit=source.commit)

        log = job.dir / "gguf.log"
        src = job.dir / "model-16bit.gguf"
        if not (job.done("convert") and src.is_file()):
            say("converting to GGUF with llama.cpp's converter")
            convert(chains.python_exe("quantize"), source, snapshot, src, log, say)
            job.record("convert", "done", sha256=_sha256(src))
        quantized = job.dir / f"model-{gguf_plan.quant}.gguf"
        if not (job.done("quantize") and quantized.is_file()):
            say(f"quantizing to {gguf_plan.quant} with the runtime's llama-quantize")
            quantize(gguf_plan.payload, src, quantized, gguf_plan.quant, log, say)
            job.record("quantize", "done", sha256=_sha256(quantized))

        say("assembling the bundle")
        bundle = assemble(quantized, snapshot, gguf_plan, {"converter_source": source.commit,
                                                           "quantize": str(gguf_plan.payload.quantize)})
        job.record("assemble", "done", bundle=str(bundle))
        report: dict[str, Any] = {"level": "none", "verdict": "not-verified",
                                  "reasons": ["verification skipped"]}
        if verifier is not None and gguf_plan.verify_level != "none":
            report = verifier(bundle, snapshot, gguf_plan, chains, job.dir / "verify", say, source_gguf=src)
        report["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (bundle / "npu_model_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                            encoding="utf-8")
        job.record("verify", "done", verdict=report["verdict"], reasons=report.get("reasons", []))
        if report["verdict"] != "failed":
            src.unlink(missing_ok=True)  # the 16-bit intermediate is only needed to requantize
        job.data["result"] = {"bundle": str(bundle), "verdict": report["verdict"]}
        job.save()
        return GgufResult(bundle, job.dir, report)
    finally:
        job.release()
