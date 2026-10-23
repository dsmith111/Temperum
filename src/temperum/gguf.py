"""GGUF output target: a Hugging Face model as one GGUF file for the NPU.

llama.cpp's Hexagon backend runs GGUF models on the Snapdragon NPU. The same file works on
every HTP version and on the CPU, so nothing is compiled ahead of time:

  resolve -> plan -> fetch -> convert (llama.cpp's convert_hf_to_gguf.py, in the quantize
  environment) -> calibrate (the runtime's llama-imatrix on the CPU: an importance matrix over
  calibration text, when the runtime ships the tool) -> quantize (the runtime's llama-quantize;
  Q4_0 by default, the type the Hexagon backend supports best) -> assemble (model.gguf,
  Modelfile, licence, manifest) -> verify (the runtime's llama-server on the NPU) -> install

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
from typing import Any, Callable, Sequence

from temperum import __version__, paths
from temperum.calib import Corpus
from temperum.core.errors import TemperumError
from temperum.host import HEXAGON_OPT_IN, HostInfo, hexagon_opted_in
from temperum.profiles import (DEFAULT_LLAMA_RUNTIME, DEFAULT_RUNTIME, LLAMA_RUNTIMES, NPU_WINDOW_GB,
                               LlamaRuntimeProfile,
                                TargetProfile, get_runtime)
from temperum.repairs import prepare_source
from temperum.sources import ResolvedModel, draft_block_size, fetch, pairable_draft
from temperum.templates import ChatTemplate, from_jinja, load_jinja
from temperum.toolchains import ENV_SIZE_GB, Toolchains
from temperum.toolchains.llama_cpp import ConverterSource

Progress = Callable[[str], None]
Runner = Callable[..., "subprocess.CompletedProcess[str]"]

QUANT_TYPES = ("Q4_0", "Q8_0", "Q4_K_M")  # all run on the Hexagon backend; Q4_0 is its best-supported type
DEFAULT_QUANT = "Q4_0"
DEFAULT_CONTEXT = 4096
CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "n_ctx", "seq_length", "max_sequence_length")
MANIFEST = "temperum_manifest.json"
RECIPE = "temperum_recipe.json"  # the recipe search's trials and choice, when --recipe searched
RECIPE_MD = "temperum_recipe.md"  # the same, as a table for people
IMATRIX = "imatrix.gguf"  # the importance matrix a calibrated quantization was built with (in the job folder)
# Calibration: llama-imatrix over the calibration corpus (wikitext-2 train rows; the held-out text is the test
# rows), 512-token chunks on the CPU. Measured 2026-10-22/23 on the X1E78100: plain Q4_0's perplexity ratio
# fell from 1.150 to 1.118 on Qwen2.5-1.5B and from 1.057 to 1.018 on Llama-3.2-3B, decode and size unchanged;
# 100 chunks scored the same as 200 (1.119 and 1.018) in half the time: 4.1 min at 1.5B, 8.8 at 3B with the
# fork package's own build of the tool, about 2.7 minutes per billion parameters.
CALIBRATION_CHUNKS = 100
CALIBRATION_CONTEXT = 512
CALIBRATION_MINUTES_PER_BILLION = 2.7
DRAFT_FILE = "draft.gguf"  # a paired speculative-decoding draft, beside model.gguf in the bundle
# Measured 2026-10-22 on Qwen3-4B with z-lab/Qwen3-4B-DFlash-b16, both on the NPU: the draft pays only when its
# guesses are accepted (code 1.76x faster, prose 1.10x, short chat answers 0.53x to 0.59x); a draft call costs
# about 1.5 plain decode steps, so a step pays from about three accepted tokens.
DRAFT_NOTE = ("the draft speeds up decoding only when its guesses are accepted: measured on Qwen3-4B, code 1.8x "
              "faster, prose 1.1x, short chat answers 0.55x; verification reports what this pair does here")
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

    @property
    def imatrix(self) -> Path:
        return self.root / _exe("llama-imatrix")

    @property
    def has_imatrix(self) -> bool:
        """Packages before the imatrix target lack the tool; conversions then quantize without calibration."""
        return self.imatrix.is_file()

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


def estimates(model: ResolvedModel, quant: str, *, cached: bool = False, verify: str = "standard",
              reference: str | None = None, calibrate: bool = False,
              draft: ResolvedModel | None = None) -> dict[str, float]:
    """Time, disk and memory for a GGUF conversion; `reference` is the dtype the source model runs in, if it does;
    `calibrate` adds the importance matrix's time on the CPU; `draft` a paired draft's download and conversion."""
    width = 4 if "32" in str(model.config.get("torch_dtype") or model.config.get("dtype") or "") else 2
    params = (model.weight_bytes or 0) / width
    download = 0 if cached else (model.weight_bytes or 0)
    draft_bytes = (draft.weight_bytes or 0) if draft is not None else 0
    draft_params = draft_bytes / 2  # drafts ship in 16-bit
    # The download, the Q8_0 conversion (also verification's perplexity reference, deleted afterwards)
    # and the quantized file (the bundle hard-links it); a paired draft adds its download and its Q8_0.
    peak = (download + params * BYTES_PER_PARAM["Q8_0"] + (params * BYTES_PER_PARAM[quant] if quant != "Q8_0" else 0)
            + draft_bytes + draft_params * BYTES_PER_PARAM["Q8_0"])
    download += draft_bytes
    # Memory, as measured: the converter holds its largest tensor (the embedding) in float32 twice while
    # quantizing it (Phi-4-mini: 5.9 GB); CPU perplexity loads the Q8_0 conversion (Phi-3.5-mini: 4.8 GB);
    # the source model runs only when verification found room for it.
    vocab, hidden = int(model.config.get("vocab_size") or 0), int(model.config.get("hidden_size") or 0)
    conversion = (vocab * hidden * 8 if vocab and hidden else params * 0.6) + 2**30
    perplexity = params * BYTES_PER_PARAM["Q8_0"] + 2**29 if verify in ("standard", "full") else 0
    source = params * {"float32": 4, "bfloat16": 2}[reference] + 2**30 if reference else 0
    return {"params_billions": round(params / 1e9, 2),
            "download_gb": round(download / 2**30, 1),
            "gguf_gb": round(params * BYTES_PER_PARAM[quant] / 2**30, 1),
            "disk_gb": round(peak / 2**30 + 0.5, 1),
            "ram_gb": round(max(conversion, perplexity, source) / 2**30, 1),
            "minutes": round(2 + 3 * params / 1e9 + (CALIBRATION_MINUTES_PER_BILLION * params / 1e9 if calibrate else 0)
                             + (1 + 3 * draft_params / 1e9 if draft is not None else 0))}


def trained_context(config: dict[str, Any]) -> int | None:
    """The longest context the model was trained for, under whichever key its config uses (GPT-2: n_positions).
    Positions past it are garbage, and out of range for learned position embeddings."""
    for key in CONTEXT_KEYS:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def job_id(model: ResolvedModel, target: TargetProfile, runtime: LlamaRuntimeProfile, quant: str,
           context: int | None, calibrate: bool = False, draft: ResolvedModel | None = None) -> str:
    identity: dict[str, Any] = {"model": model.ref.repo or str(model.ref.path), "commit": model.commit,
                                "target": target.id, "runtime": runtime.id, "llama_cpp": runtime.llama_cpp_commit,
                                "quant": quant, "context": context, "converter": __version__, "pipeline": "q8_0-first"}
    if calibrate:  # a calibrated file is a different file; uncalibrated jobs keep their earlier ids
        identity["calibration"] = CALIBRATION_CHUNKS
    if draft is not None:  # a pair is a different bundle
        identity["draft"] = {"model": draft.ref.repo or str(draft.ref.path), "commit": draft.commit}
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
    recipe: str | None = None  # "auto": search quantization mixes on the NPU (temperum.optimize)
    calibrate: bool = False  # quantize with an importance matrix computed first (needs llama-imatrix)
    draft: ResolvedModel | None = None  # a speculative-decoding draft paired with the model (DFlash)
    draft_block: int = 0  # the draft's trained block: tokens proposed per step, the Modelfile's draft_num_predict

    @property
    def job_dir(self) -> Path:
        return paths.jobs_dir() / self.job_id

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model.to_dict(), "target": self.target.id, "runtime": self.runtime.id,
                "llama_cpp": self.runtime.llama_cpp_commit, "payload": str(self.payload.root),
                "htp": str(self.payload.htp) if self.payload.htp else None,
                "quant": self.quant, "recipe": self.recipe, "calibrate": self.calibrate, "context": self.context,
                "draft": self.draft.to_dict() if self.draft is not None else None, "draft_block": self.draft_block,
                "destination": str(self.destination),
                "job": str(self.job_dir), "verify": self.verify_level, "estimates": self.estimates,
                "warnings": self.warnings, "host": self.host.to_dict()}


def plan(model: ResolvedModel, host: HostInfo, target: TargetProfile, *, ollama_exe: Path | None,
         quant: str = DEFAULT_QUANT, context: int | None = None, out: Path | None = None, verify: str = "standard",
         overwrite: bool = False, python: Path | None = None, runtime_id: str = DEFAULT_LLAMA_RUNTIME,
         run: Runner = subprocess.run, chains_factory: Callable[..., Toolchains] = Toolchains,
         recipe: str | None = None, calibrate: bool = True, draft: ResolvedModel | None = None) -> GgufPlan:
    """`calibrate`: quantize with an importance matrix when the runtime ships llama-imatrix (Q8_0 has nothing
    to calibrate); off, or without the tool, the quantization is plain. `draft`: a speculative-decoding draft
    to pair with the model (converted with the model's metadata, bundled, named in the Modelfile)."""
    from temperum import optimize

    if target.format != "gguf":
        raise TemperumError("plan", "WRONG_TARGET_FORMAT", f"{target.id} is not a GGUF target")
    if model.draft_reason:
        raise TemperumError("plan", "DRAFT_NEEDS_TARGET",
                            f"{model.ref.display} is a speculative-decoding draft ({model.draft_reason}): it proposes "
                            "tokens for a larger target model and only runs paired with it.",
                            hint=f"Convert the pair: temperum convert {model.draft_target or '<target model>'} "
                                 f"--draft {model.ref.display}"
                                 + ("" if model.draft_target else " (the target is the model the draft was trained "
                                                                  "for; its model card names it)"))
    block = 0
    if draft is not None:
        block = _pair_check(model, draft)
    if quant not in QUANT_TYPES:
        raise TemperumError("plan", "UNKNOWN_QUANT", f"Unknown GGUF type {quant}", hint=f"Use {', '.join(QUANT_TYPES)}.")
    if recipe is not None and recipe not in optimize.RECIPES:
        raise TemperumError("plan", "UNKNOWN_RECIPE", f"Unknown recipe {recipe}",
                            hint=f"Use {', '.join(optimize.RECIPES)}: a search of quantization mixes on the NPU.")
    if recipe is not None and quant != "Q4_0":
        raise TemperumError("plan", "RECIPE_BASE", "--recipe searches mixes over a Q4_0 base.",
                            hint="Leave --quant at its default with --recipe.")
    runtime = LLAMA_RUNTIMES[runtime_id]
    if ollama_exe is None:
        raise TemperumError("plan", "NEEDS_RUNTIME",
                            "The GGUF target quantizes and verifies with the Ollama fork's own llama.cpp build.",
                            hint="Set TEMPERUM_OLLAMA_EXE to the fork's ollama.exe once (or pass --ollama-exe); "
                                 "--target qnn-x-elite builds a QNN bundle instead.")
    payload = LlamaPayload.from_executable(ollama_exe, run)
    missing = payload.problems()
    if missing:
        raise TemperumError("plan", "RUNTIME_INCOMPLETE",
                            f"{payload.root} lacks the llama.cpp Hexagon runtime: {', '.join(missing)}",
                            hint=f"Use a fork package built with -HexagonSDKRoot; get Qualcomm's signed NPU libraries "
                                 f"with `{SETUP_COMMAND}`. --target qnn-x-elite builds a QNN bundle instead.")
    warnings: list[str] = []
    commit = payload.version(run)
    if commit is None:
        warnings.append("could not read the runtime's llama.cpp version")
    elif not runtime.llama_cpp_commit.startswith(commit):
        warnings.append(f"the runtime's llama.cpp is {commit}, but this converter pins {runtime.llama_cpp_commit[:9]} "
                        f"({runtime.llama_cpp_tag}); the HTP libraries and file format follow the runtime")
    maximum = trained_context(model.config)
    ctx = context or min(maximum or DEFAULT_CONTEXT, DEFAULT_CONTEXT)
    if maximum is not None and not 128 <= ctx <= maximum:
        raise TemperumError("plan", "BAD_CONTEXT",
                            f"--context {ctx} is outside 128..{maximum}, the model's trained length.")
    destination = out or paths.models_dir() / f"{model.name}-{target.id}"
    if destination.exists() and not overwrite:
        raise TemperumError("plan", "DESTINATION_EXISTS", f"{destination} already exists.",
                            hint="Pass --overwrite to rebuild it or --out to write elsewhere.")
    on_target = target.matches(host.processor_name)
    if not on_target:
        warnings.append(f"this machine is not a {target.id} device: the file is built but not verified here")
    if not target.verified:
        warnings.append(f"target {target.id} has not been verified on hardware yet")
    if not hexagon_opted_in():
        warnings.append(f"the fork uses the Hexagon NPU only when its server starts with {HEXAGON_OPT_IN}=1")
    calibrated = calibrate and quant != "Q8_0"
    if calibrated and not payload.has_imatrix:
        warnings.append("the runtime has no llama-imatrix, so the quantization is not calibrated with an importance "
                        "matrix (a fork package built with the imatrix target ships it)")
        calibrated = False
    cached = model.ref.kind == "local" or (
        model.ref.repo is not None and model.commit is not None
        and (paths.sources_dir() / model.ref.repo.replace("/", "--") / model.commit / "_source.json").is_file())
    from temperum.convert import _resource_checks  # the QNN target's memory and disk checks
    from temperum.verify import reference_dtype

    level = verify if on_target else "none"
    params = estimates(model, quant)["params_billions"] * 1e9
    reference = (reference_dtype(int(params), host.available_ram_gb, host.total_ram_gb)
                 if level in ("standard", "full") else None)
    est = estimates(model, quant, cached=cached, verify=level, reference=reference, calibrate=calibrated, draft=draft)
    if draft is not None:
        warnings.append(DRAFT_NOTE)
    status = chains_factory(host, get_runtime(DEFAULT_RUNTIME), "cpu", python=python).status("quantize")
    if status.state != "ready":
        warnings.append(f"first use: sets up the quantize toolchain (~{ENV_SIZE_GB['quantize']:.0f} GB, about 8 "
                        "minutes; torch comes from download.pytorch.org, everything else through your pip "
                        "configuration)")
    warnings += _resource_checks(est, host)
    identity = job_id(model, target, runtime, quant, context, calibrated, draft)
    if recipe is not None:
        est["minutes"] = est["minutes"] + optimize.estimate_minutes(float(est["params_billions"]))
    return GgufPlan(model, host, target, runtime, payload, quant, ctx, destination, identity, est,
                    verify if on_target else "none", warnings, python, overwrite, recipe, calibrated, draft, block)


def _pair_check(model: ResolvedModel, draft: ResolvedModel) -> int:
    """That `draft` is a DFlash draft trained for `model`; returns its block size."""
    if not draft.draft_reason:
        raise TemperumError("plan", "DRAFT_NOT_A_DRAFT",
                            f"{draft.ref.display} is not a speculative-decoding draft: it has none of the draft "
                            "config keys, architectures or tags.",
                            hint="--draft takes the draft; the model to convert is the target.")
    if not pairable_draft(draft.config):
        raise TemperumError("plan", "DRAFT_NOT_PAIRABLE",
                            f"{draft.ref.display} ({draft.architecture}) cannot be paired: Ollama's runner runs "
                            "DFlash drafts beside a target and would launch this one as an MTP draft.",
                            hint="DFlash drafts (architecture DFlashDraftModel, DFlash2DraftModel) pair.")
    block = draft_block_size(draft.config)
    if block is None:
        raise TemperumError("plan", "DRAFT_NO_BLOCK", f"{draft.ref.display}'s config has no block_size.",
                            hint="A DFlash draft declares how many tokens it proposes per step.")
    layers = draft.config.get("num_target_layers")
    target_layers = model.config.get("num_hidden_layers") or model.config.get("num_layers")
    hidden, target_hidden = draft.config.get("hidden_size"), model.config.get("hidden_size")
    for what, got, want in (("layers", target_layers, layers), ("hidden size", target_hidden, hidden)):
        if isinstance(got, int) and isinstance(want, int) and got != want:
            raise TemperumError("plan", "DRAFT_MISMATCH",
                                f"{draft.ref.display} was trained for a target with {want} {what}; "
                                f"{model.ref.display} has {got}.",
                                hint=f"Pair it with the model its card names: {draft.draft_target or 'see its card'}.")
    if draft.model_type and model.model_type and draft.model_type != model.model_type:
        raise TemperumError("plan", "DRAFT_MISMATCH",
                            f"{draft.ref.display} is a {draft.model_type} draft; {model.ref.display} is "
                            f"{model.model_type}.",
                            hint=f"Pair it with the model its card names: {draft.draft_target or 'see its card'}.")
    return block


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


def last_error_line(tail: str) -> str | None:
    """The last `SomethingError: ...` line of a Python traceback: the part a person needs first."""
    for line in reversed(tail.splitlines()):
        if re.match(r"^\s*[A-Za-z0-9_.]*(Error|Exception)\s*:", line):
            return line.strip()[:300]
    return None


def convert(python: Path, source: ConverterSource, snapshot: Path, out: Path, log: Path, say: Progress,
            popen: Callable[..., Any] = subprocess.Popen) -> Path:
    """HF snapshot -> Q8_0 GGUF (convert_hf_to_gguf --outtype q8_0).

    Q8_0 rather than 16-bit halves the intermediate, and it doubles as verification's perplexity
    reference. Requantizing it cost Qwen2.5-1.5B 0.1% perplexity at Q4_0 (9.599 vs 9.588 from 16-bit)
    and 0.4% at Q4_K_M (9.246 vs 9.212).
    """
    partial = out.with_name(out.name + ".partial")
    partial.unlink(missing_ok=True)
    code, tail = _stream([str(python), str(source.script), str(snapshot), "--outfile", str(partial),
                          "--outtype", "q8_0"], source.environment(), log, say,
                         lambda line: "Model successfully exported" in line, popen)
    if code != 0 or not partial.is_file():
        reason = last_error_line(tail)
        raise TemperumError("convert", "GGUF_CONVERT_FAILED",
                            f"llama.cpp's converter failed (exit {code})" + (f": {reason}" if reason else "."),
                            hint=f"Log: {log}\n{tail[-800:]}")
    partial.replace(out)
    return out


def convert_draft(python: Path, source: ConverterSource, snapshot: Path, target_snapshot: Path, out: Path, log: Path,
                  say: Progress, popen: Callable[..., Any] = subprocess.Popen) -> Path:
    """A DFlash draft's snapshot -> Q8_0 GGUF. The draft carries no tokenizer and shares the target's
    embeddings, so the converter reads those from the target's snapshot (`--target-model-dir`)."""
    partial = out.with_name(out.name + ".partial")
    partial.unlink(missing_ok=True)
    code, tail = _stream([str(python), str(source.script), str(snapshot), "--outfile", str(partial),
                          "--outtype", "q8_0", "--target-model-dir", str(target_snapshot)], source.environment(), log,
                         say, lambda line: "Model successfully exported" in line, popen)
    if code != 0 or not partial.is_file():
        reason = last_error_line(tail)
        raise TemperumError("convert", "GGUF_DRAFT_CONVERT_FAILED",
                            f"llama.cpp's converter failed on the draft (exit {code})" + (f": {reason}" if reason else "."),
                            hint=f"Log: {log}\n{tail[-800:]}")
    partial.replace(out)
    return out


def calibration_text(corpus: Path, out: Path) -> Path:
    """The calibration corpus (JSONL rows, as calib caches it) as the plain text llama-imatrix reads."""
    if out.is_file():
        return out
    rows = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        text = json.loads(line).get("text", "") if line.strip() else ""
        if text.strip():
            rows.append(text.rstrip("\n"))
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return out


def compute_imatrix(payload: LlamaPayload, src: Path, text: Path, out: Path, log: Path, say: Progress,
                    popen: Callable[..., Any] = subprocess.Popen, chunks: int = CALIBRATION_CHUNKS) -> Path:
    """The importance matrix of the Q8_0 GGUF at `src` over `text`, computed on the CPU by the runtime's
    llama-imatrix: which values in each block matter, so the quantizer keeps them. Calibration changes the
    values chosen inside each Q4_0 block, not the block format, so the Hexagon kernel and the decode speed are
    untouched. The token embedding gets no matrix (the tool's default), as the head is left to the quantizer."""
    partial = out.with_name(out.stem + ".partial.gguf")  # the tool writes its GGUF format only to a .gguf name
    partial.unlink(missing_ok=True)
    env = {**os.environ, "PATH": f"{payload.root}{os.pathsep}{os.environ.get('PATH', '')}"}
    env.pop("GGML_BACKEND_PATH", None)  # the CPU: the matrix is activation statistics, not a speed measurement
    code, tail = _stream([str(payload.imatrix), "-m", str(src), "-f", str(text), "-o", str(partial),
                          "-c", str(CALIBRATION_CONTEXT), "--chunks", str(chunks), "--no-ppl", "-ngl", "0"],
                         env, log, say,
                         lambda line: "seconds per pass" in line or line.startswith("save_imatrix"), popen)
    if code != 0 or not partial.is_file():
        raise TemperumError("convert", "GGUF_CALIBRATE_FAILED", f"llama-imatrix failed (exit {code}).",
                            hint=f"Log: {log}\n{tail[-800:]}")
    partial.replace(out)
    return out


def calibration_record(payload: LlamaPayload, corpus: Corpus, chunks: int = CALIBRATION_CHUNKS) -> dict[str, Any]:
    """What the manifest and the recipe document record about a calibrated quantization."""
    record = corpus.to_dict()
    record.pop("path", None)
    return {"tool": str(payload.imatrix), "chunks": chunks, "context": CALIBRATION_CONTEXT, "corpus": record}


def quantize(payload: LlamaPayload, src: Path, out: Path, quant: str, log: Path, say: Progress,
             popen: Callable[..., Any] = subprocess.Popen, overrides: Sequence[str] = (),
             imatrix: Path | None = None) -> Path:
    """`overrides`: llama-quantize's per-tensor flags (`--tensor-type attn_v=q8_0`), as a recipe lists them;
    `imatrix`: the importance matrix from compute_imatrix, when the quantization is calibrated."""
    partial = out.with_name(out.name + ".partial")
    partial.unlink(missing_ok=True)
    env = {**os.environ, "PATH": f"{payload.root}{os.pathsep}{os.environ.get('PATH', '')}"}
    calibration = ["--imatrix", str(imatrix)] if imatrix is not None else []
    code, tail = _stream([str(payload.quantize), "--allow-requantize", *calibration, *overrides, str(src), str(partial),
                          quant],
                         env, log, say,
                         lambda line: line.startswith(("llama_model_quantize_impl: model size",
                                                       "llama_model_quantize_impl: quant size")), popen)
    if code != 0 or not partial.is_file():
        raise TemperumError("convert", "GGUF_QUANTIZE_FAILED", f"llama-quantize failed (exit {code}).",
                            hint=f"Log: {log}\n{tail[-800:]}")
    partial.replace(out)
    return out


def derived_template(snapshot: Path) -> ChatTemplate | None:
    """The converter's Ollama template: the model's chat format when recognized; the plain prompt (family
    "raw") for a base model with no chat template, so the fork cannot chat with it in a built-in format it
    never saw; None when the format is not recognized (the fork decides)."""
    try:
        return from_jinja(load_jinja(snapshot))
    except TemperumError:
        return None


def modelfile(context: int, chat: ChatTemplate | None = None, params: dict[str, Any] | None = None,
              draft: dict[str, Any] | None = None) -> str:
    """`params`: further `PARAMETER` lines the recipe search measured for this model (`num_thread`, say);
    `draft`: the bundle's paired draft (manifest `draft`: `file`, `block_size`), which the fork runs as
    speculative decoding (`DRAFT`, `draft_num_predict`; 0 turns it off)."""
    # The GGUF carries the model's own chat template (tokenizer.chat_template), which the fork can
    # render through llama-server. Without a TEMPLATE, though, `ollama create` swaps in a built-in Go
    # template when one is close to the model's (SmolLM2 lost its default system prompt that way), and
    # the fork renders chats with it. A template derived from the model's own and checked against it
    # (verify) prevents that; the fork still prefers the GGUF's template when it supports more, such
    # as tool calls.
    lines = ["FROM ./model.gguf"]
    if draft:
        lines += [f"DRAFT ./{draft['file']}", f"PARAMETER draft_num_predict {int(draft['block_size'])}"]
    lines.append(f"PARAMETER num_ctx {context}")
    lines += [f"PARAMETER {key} {value}" for key, value in (params or {}).items()]
    if chat is not None:
        lines.append(f'TEMPLATE """{chat.template}"""')
        lines += [f'PARAMETER stop "{stop}"' for stop in chat.stops]
    return "\n".join(lines) + "\n"


def set_template(bundle: Path, chat: ChatTemplate, context: int) -> None:
    """Write a checked Ollama template into an assembled bundle's Modelfile and manifest, keeping the
    runtime parameters the recipe search put there."""
    path = bundle / "Modelfile"
    manifest_path = bundle / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("modelfile_params") or {}
    path.write_text(modelfile(context, chat, params, manifest.get("draft")), encoding="utf-8", newline="\n")
    manifest["files"]["Modelfile"] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    manifest["ollama_template"] = {"family": chat.family, "variant": chat.variant or None,
                                   "checked_against": ("nothing: the model has no chat template, so the prompt "
                                                       "is sent as it is" if chat.family == "raw" else
                                                       "the model's own chat template (transformers)")}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def assemble(gguf: Path, snapshot: Path, plan: GgufPlan, provenance: dict[str, Any],
             extra_files: dict[str, str] | None = None, precision: str | None = None,
             params: dict[str, Any] | None = None, draft: tuple[Path, dict[str, Any]] | None = None) -> Path:
    """`extra_files`: name -> text written beside the model (the recipe document); `precision`: the
    manifest's precision label when it is not plain `plan.quant` (a mix); `params`: runtime Modelfile
    parameters the recipe search measured (`num_thread`), kept in the manifest so later rewrites keep them;
    `draft`: the paired draft's GGUF and its record (repo, commit, block_size), linked in as draft.gguf."""
    final = plan.destination
    staging = final.with_name(final.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        os.link(gguf, staging / "model.gguf")
    except OSError:
        shutil.copy2(gguf, staging / "model.gguf")
    draft_record: dict[str, Any] | None = None
    if draft is not None:
        draft_record = {**draft[1], "file": DRAFT_FILE}
        try:
            os.link(draft[0], staging / DRAFT_FILE)
        except OSError:
            shutil.copy2(draft[0], staging / DRAFT_FILE)
    (staging / "Modelfile").write_text(modelfile(plan.context, params=params, draft=draft_record), encoding="utf-8",
                                       newline="\n")
    licences = []
    for item in sorted(snapshot.iterdir()):
        if item.is_file() and item.name.lower().startswith(LICENSE_PREFIXES):
            shutil.copy2(item, staging / item.name)
            licences.append(item.name)
    for name, text in (extra_files or {}).items():
        (staging / name).write_text(text, encoding="utf-8")
    files = {p.name: {"size": p.stat().st_size, "sha256": _sha256(p)} for p in sorted(staging.iterdir())}
    manifest = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "converter": {"name": "temperum", "version": __version__},
        "format": "gguf",
        "source": {"repo": plan.model.ref.repo, "commit": plan.model.commit, "license": plan.model.license,
                   "architecture": plan.model.architecture},
        "target": {"id": plan.target.id, "htp_arch": plan.target.htp_arch},
        "runtime": {"id": plan.runtime.id, "llama_cpp": plan.runtime.llama_cpp_commit,
                    "htp_libraries": plan.runtime.htp_libraries},
        "precision": precision or plan.quant,
        "modelfile_params": dict(params or {}),  # runtime settings the recipe search measured (num_thread)
        "draft": draft_record,  # the paired speculative-decoding draft, when the bundle has one
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
    recipe: dict[str, Any] | None = None  # the recipe search's document, when --recipe searched the mixes
    calibration: dict[str, Any] | None = None  # calibration_record(), when the quantization was calibrated
    draft: dict[str, Any] | None = None  # the manifest's draft record, when a draft was paired


def run(gguf_plan: GgufPlan, progress: Progress | None = None,
        chains_factory: Callable[..., Toolchains] = Toolchains,
        verifier: Callable[..., dict[str, Any]] | None = None) -> GgufResult:
    from temperum.convert import Job  # the same receipt and single-writer lock as the QNN target

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
        snapshot, repairs = prepare_source(snapshot, job.dir / "source", say)  # known checkpoint quirks
        source = ConverterSource.load(gguf_plan.runtime.llama_cpp_tag)
        source.ensure(progress=say)
        architecture = gguf_plan.model.architecture
        if architecture and architecture not in supported_architectures(source.directory):
            raise TemperumError("convert", "UNSUPPORTED_ARCHITECTURE",
                                f"llama.cpp {source.tag} cannot convert {architecture}.",
                                hint="Try the QNN target, or a model with a supported architecture.")
        job.record("converter", "done", tag=source.tag, commit=source.commit, patches=sorted(source.patches))

        log = job.dir / "gguf.log"
        converted = job.dir / "model-Q8_0.gguf"  # also the perplexity reference
        if not (job.done("convert") and converted.is_file()):
            say("converting to GGUF (Q8_0) with llama.cpp's converter")
            convert(chains.python_exe("quantize"), source, snapshot, converted, log, say)
            job.record("convert", "done", sha256=_sha256(converted))
        imatrix: Path | None = None
        calibration: dict[str, Any] | None = None
        if gguf_plan.calibrate:
            from temperum import calib

            imatrix = job.dir / IMATRIX
            corpus = calib.ensure_corpus(calib.CALIBRATION, paths.data_dir())
            if not (job.done("calibrate") and imatrix.is_file()):
                say(f"calibrating: an importance matrix from {CALIBRATION_CHUNKS} chunks of {corpus.spec.dataset} "
                    f"({corpus.spec.split} rows) on the CPU")
                text = calibration_text(corpus.path, job.dir / "calibration.txt")
                compute_imatrix(gguf_plan.payload, converted, text, imatrix, log, say)
                job.record("calibrate", "done", chunks=CALIBRATION_CHUNKS, sha256=_sha256(imatrix))
            calibration = calibration_record(gguf_plan.payload, corpus)
        quantized = job.dir / f"model-{gguf_plan.quant}.gguf"
        recipe_doc: dict[str, Any] | None = None
        precision = gguf_plan.quant
        params: dict[str, Any] = {}  # runtime Modelfile parameters the recipe search measured
        if gguf_plan.recipe is not None:
            from temperum import calib, optimize

            say("searching quantization mixes on the NPU (each one built, scored and timed)")
            heldout = calib.ensure_corpus(calib.HELDOUT, paths.data_dir()).path
            objective = optimize.Objective(size_budget_mib=NPU_WINDOW_GB * 1024)  # must sit whole on the NPU
            found = optimize.search(gguf_plan.payload, converted, job.dir / "optimize", heldout, say=say,
                                    objective=objective, context=gguf_plan.context, imatrix=imatrix)
            quantized, precision = found.chosen.path, found.chosen.candidate.label
            runtime = optimize.runtime_search(gguf_plan.payload, quantized, job.dir / "optimize", heldout, say=say,
                                              context=gguf_plan.context)
            params = runtime.params
            recipe_doc = optimize.document(found, found.reference_ppl,
                                           objective, chunks=optimize.PPL_CHUNKS,
                                           host=gguf_plan.host.to_dict(), llama_cpp=gguf_plan.runtime.llama_cpp_commit,
                                           source={"repo": gguf_plan.model.ref.repo, "commit": gguf_plan.model.commit,
                                                   "calibration": calibration},
                                           runtime=runtime)
            for line in optimize.lines(found, objective, runtime):
                say(line)
            job.record("optimize", "done", chosen=found.chosen.candidate.name, trials=len(found.trials),
                       runtime=params)
        elif gguf_plan.quant != "Q8_0" and not (job.done("quantize") and quantized.is_file()):
            say(f"quantizing to {gguf_plan.quant} with the runtime's llama-quantize"
                + (", calibrated" if imatrix is not None else ""))
            quantize(gguf_plan.payload, converted, quantized, gguf_plan.quant, log, say, imatrix=imatrix)
            job.record("quantize", "done", sha256=_sha256(quantized), calibrated=imatrix is not None)

        draft_pair: tuple[Path, dict[str, Any]] | None = None
        if gguf_plan.draft is not None:
            say(f"fetching the draft {gguf_plan.draft.ref.display}")
            draft_snapshot = fetch(gguf_plan.draft, paths.sources_dir())
            job.record("fetch-draft", "done", commit=gguf_plan.draft.commit)
            draft_file = job.dir / "draft-Q8_0.gguf"
            if not (job.done("convert-draft") and draft_file.is_file()):
                say("converting the draft to GGUF (Q8_0) with the target's tokenizer and metadata")
                convert_draft(chains.python_exe("quantize"), source, draft_snapshot, snapshot, draft_file, log, say)
                job.record("convert-draft", "done", sha256=_sha256(draft_file))
            draft_pair = (draft_file, {"repo": gguf_plan.draft.ref.repo, "commit": gguf_plan.draft.commit,
                                       "architecture": gguf_plan.draft.architecture,
                                       "block_size": gguf_plan.draft_block, "precision": "Q8_0"})
        say("assembling the bundle")
        extra: dict[str, str] = {}
        if recipe_doc is not None:
            extra[RECIPE] = json.dumps(recipe_doc, indent=2, default=str)
            extra[RECIPE_MD] = optimize.markdown(recipe_doc)
        bundle = assemble(quantized, snapshot, gguf_plan, {"converter_source": source.commit,
                                                           "converter_patches": sorted(source.patches),
                                                           "quantize": str(gguf_plan.payload.quantize),
                                                           "repairs": repairs,
                                                           "calibration": calibration,
                                                           "recipe": (recipe_doc or {}).get("chosen", {}).get("name")},
                          extra_files=extra, precision=precision, params=params, draft=draft_pair)
        job.record("assemble", "done", bundle=str(bundle))
        report: dict[str, Any] = {"level": "none", "verdict": "not-verified",
                                  "reasons": ["verification skipped"]}
        chat = derived_template(snapshot)
        if verifier is not None and gguf_plan.verify_level != "none":
            report = verifier(bundle, snapshot, gguf_plan, chains, job.dir / "verify", say, reference_gguf=converted,
                              chat=chat)
        checked = report.get("template", {}).get("ollama", {}).get("status") == "match"
        if chat is not None and (chat.family == "raw" or checked):
            # raw: a base model with no chat template; the plain template keeps the fork from chatting with
            # it in a built-in format it never saw (llama-server's default ChatML)
            set_template(bundle, chat, gguf_plan.context)
            report.setdefault("template", {})["modelfile"] = chat.family
        report["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (bundle / "temperum_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                            encoding="utf-8")
        job.record("verify", "done", verdict=report["verdict"], reasons=report.get("reasons", []))
        if report["verdict"] != "failed" and gguf_plan.quant != "Q8_0":
            converted.unlink(missing_ok=True)  # only needed to requantize and as the reference
        job.data["result"] = {"bundle": str(bundle), "verdict": report["verdict"]}
        job.save()
        return GgufResult(bundle, job.dir, report, recipe_doc, calibration,
                          {**draft_pair[1], "file": DRAFT_FILE} if draft_pair else None)
    finally:
        job.release()
