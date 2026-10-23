"""Measuring one candidate on this machine's NPU: build it, score it, time it.

Everything is the fork package's own llama.cpp build, as the verifier uses it: llama-quantize from the
kept Q8_0 conversion, llama-perplexity on the NPU against the same Q8_0 on the CPU, and llama-server the
way the fork runs a model that sits whole on the NPU. Trials are cached by candidate name in the work
folder, so a rerun or a second strategy pays nothing for what is already measured.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from temperum import gguf
from temperum.gguf import LlamaPayload
from temperum.gguf_verify import LlamaServer, _heldout_text, perplexity, placement
from temperum.optimize.space import Candidate

Progress = Callable[[str], None]
PPL_CONTEXT = 512
PROMPT_CHARS = 2200  # about 540 tokens of the held-out text
DECODE_TOKENS = 256  # measured repeatability over 8 candidates: decode within 1.5% on average, 3.6% at most
DECODE_REQUESTS = 3
LEADS = ("First.", "Second.", "Third.")  # llama-server caches identical prompts; the first token varies


@dataclass(frozen=True)
class Trial:
    """One measured candidate. `ratio` is its NPU perplexity over the Q8_0 reference on the CPU."""

    candidate: Candidate
    path: Path
    size_mib: float
    ppl: float | None
    ratio: float | None
    decode: float | None
    prefill: float | None
    layers: str | None
    whole: bool
    seconds: float
    error: str | None = None
    cores: float | None = None  # the server's busy CPU cores while it answered

    def to_dict(self) -> dict[str, Any]:
        return {**self.candidate.to_dict(), "path": str(self.path), "size_mib": self.size_mib, "ppl": self.ppl,
                "ratio": self.ratio, "decode_tps": self.decode, "prefill_tps": self.prefill, "layers": self.layers,
                "whole": self.whole, "seconds": self.seconds, "error": self.error, "cores": self.cores}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trial:
        cand = Candidate(d["base"], tuple(sorted(d["overrides"].items())) if d["overrides"] else ())
        return cls(cand, Path(d["path"]), d["size_mib"], d["ppl"], d["ratio"], d["decode_tps"], d["prefill_tps"],
                   d["layers"], d["whole"], d["seconds"], d.get("error"), d.get("cores"))


class ForkServer(LlamaServer):
    """llama-server as the fork starts it for a model that sits whole on the NPU: flash attention on,
    no polling between tokens (the fork's `--poll 0`)."""

    def command(self) -> list[str]:
        return super().command() + ["--flash-attn", "on", "--poll", "0"]


@dataclass(frozen=True)
class RuntimeTrial:
    """One runtime setting measured on a built model: the CPU thread count llama-server runs beside the NPU.
    `threads` None is llama-server's default (every physical core). Quality does not change with threads, so
    only speed and the server's busy cores are measured."""

    threads: int | None
    decode: float | None
    prefill: float | None
    cores: float | None
    seconds: float
    error: str | None = None

    @property
    def label(self) -> str:
        if self.threads is None:
            return "default threads"
        return f"{self.threads} thread" + ("" if self.threads == 1 else "s")

    def to_dict(self) -> dict[str, Any]:
        return {"threads": self.threads, "label": self.label, "decode_tps": self.decode, "prefill_tps": self.prefill,
                "cores": self.cores, "seconds": self.seconds, "error": self.error}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RuntimeTrial:
        return cls(d["threads"], d["decode_tps"], d["prefill_tps"], d.get("cores"), d["seconds"], d.get("error"))


def cpu_seconds(pid: int) -> float | None:
    """Processor time a process has used, from the OS (Windows: PowerShell; elsewhere None)."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"(Get-Process -Id {pid}).TotalProcessorTime.TotalSeconds"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Evaluator:
    """Builds and measures candidates from one Q8_0 source; caches by candidate name."""

    def __init__(self, payload: LlamaPayload, source: Path, work: Path, heldout: Path, *, context: int,
                 chunks: int, say: Progress, imatrix: Path | None = None) -> None:
        self.payload, self.source, self.work, self.say = payload, source, work, say
        self.context, self.chunks = context, chunks
        self.imatrix = imatrix  # every candidate is quantized with it when the conversion is calibrated
        self.work.mkdir(parents=True, exist_ok=True)
        self.text = _heldout_text(heldout, self.work / "heldout.txt")
        self.prompt = self.text.read_text(encoding="utf-8")[:PROMPT_CHARS]
        self._log = self.work / "trials.jsonl"
        self._trials: dict[str, Trial] = {}
        self._runtime: dict[str, RuntimeTrial] = {}
        self.reference_ppl: float | None = None
        if self._log.is_file():
            for line in self._log.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                if d.get("name") == "_reference":
                    self.reference_ppl = d["ppl"]
                else:
                    t = Trial.from_dict(d)
                    self._trials[t.candidate.name] = t
        runtime_log = self.work / "runtime.jsonl"
        if runtime_log.is_file():
            for line in runtime_log.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                self._runtime[d["key"]] = RuntimeTrial.from_dict(d)

    @property
    def trials(self) -> list[Trial]:
        return list(self._trials.values())

    def reference(self) -> float:
        """Perplexity of the Q8_0 source on the CPU: the denominator of every ratio."""
        if self.reference_ppl is None:
            self.say("reference: the Q8_0 conversion on the CPU")
            ppl = perplexity(self.payload, self.source, self.text, npu=False, context=PPL_CONTEXT,
                             chunks=self.chunks, log=self.work / "reference.ppl.log")
            if ppl is None:
                raise gguf.TemperumError("optimize", "REFERENCE_FAILED",
                                         "llama-perplexity gave no result for the Q8_0 reference on the CPU.",
                                         hint=f"Log: {self.work / 'reference.ppl.log'}")
            self.reference_ppl = ppl
            self._append({"name": "_reference", "ppl": ppl})
        return self.reference_ppl

    def measure(self, candidate: Candidate) -> Trial:
        cached = self._trials.get(candidate.name)
        if cached is not None and cached.error is None:
            return cached
        started = time.monotonic()
        try:
            path = self._build(candidate)
            size = round(path.stat().st_size / 2**20, 1)
            self.say(f"{candidate.label}: perplexity on the NPU")
            ppl = perplexity(self.payload, path, self.text, npu=True, context=PPL_CONTEXT, chunks=self.chunks,
                             log=self.work / f"{candidate.name}.ppl.log")
            ratio = round(ppl / self.reference(), 4) if ppl else None
            self.say(f"{candidate.label}: decode and prefill through llama-server")
            speed = self._speed(path, self.work / f"{candidate.name}.server.log")
            trial = Trial(candidate, path, size, ppl, ratio, speed["decode"], speed["prefill"], speed["layers"],
                          speed["whole"], round(time.monotonic() - started, 1), cores=speed["cores"])
        except Exception as exc:  # noqa: BLE001 - one failed candidate must not end the search
            trial = Trial(candidate, self.work / f"model-{candidate.name}.gguf", 0.0, None, None, None, None, None,
                          False, round(time.monotonic() - started, 1), f"{type(exc).__name__}: {exc}"[:400])
            self.say(f"{candidate.label}: failed ({trial.error})")
        self._trials[candidate.name] = trial
        self._append(trial.to_dict())
        return trial

    def _build(self, candidate: Candidate) -> Path:
        if candidate.base == "Q8_0" and not candidate.overrides:
            return self.source
        out = self.work / f"model-{candidate.name}.gguf"
        if out.is_file():
            return out
        self.say(f"{candidate.label}: quantizing")
        gguf.quantize(self.payload, self.source, out, candidate.base, self.work / f"{candidate.name}.quantize.log",
                      self.say, overrides=candidate.quantize_args(), imatrix=self.imatrix)
        return out

    def measure_runtime(self, model: Path, threads: int | None) -> RuntimeTrial:
        """Speed and busy cores of a built model with llama-server on `threads` CPU threads (None: its default)."""
        key = f"{model.name}:{threads}"
        cached = self._runtime.get(key)
        if cached is not None and cached.error is None:
            return cached
        started = time.monotonic()
        label = "default threads" if threads is None else f"{threads} thread" + ("" if threads == 1 else "s")
        try:
            self.say(f"{model.name}, {label}: decode, prefill and busy cores through llama-server")
            args = [] if threads is None else ["-t", str(threads)]
            speed = self._speed(model, self.work / f"runtime-{model.stem}-t{threads or 'default'}.server.log", args)
            trial = RuntimeTrial(threads, speed["decode"], speed["prefill"], speed["cores"],
                                 round(time.monotonic() - started, 1))
        except Exception as exc:  # noqa: BLE001
            trial = RuntimeTrial(threads, None, None, None, round(time.monotonic() - started, 1),
                                 f"{type(exc).__name__}: {exc}"[:400])
            self.say(f"{label}: failed ({trial.error})")
        self._runtime[key] = trial
        with (self.work / "runtime.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, **trial.to_dict()}) + "\n")
        return trial

    def _speed(self, model: Path, log: Path, extra_args: list[str] | None = None) -> dict[str, Any]:
        decode: list[float] = []
        prefill: list[float] = []
        cores: float | None = None
        server = ForkServer(self.payload, model, self.context, log)
        server.extra_args = list(extra_args or [])
        with server:
            server.complete(f"Warm. {self.prompt[:400]}", 16)  # the first request pays one-time costs
            pid = server.process.pid if server.process is not None else None
            cpu0, wall0 = (cpu_seconds(pid) if pid else None), time.monotonic()
            for lead in LEADS[:DECODE_REQUESTS]:
                res = server.complete(f"{lead} {self.prompt}", DECODE_TOKENS)
                timings = res.get("timings") or {}
                if timings.get("predicted_per_second"):
                    decode.append(float(timings["predicted_per_second"]))
                if timings.get("prompt_per_second"):
                    prefill.append(float(timings["prompt_per_second"]))
            if pid and cpu0 is not None:
                cpu1 = cpu_seconds(pid)
                if cpu1 is not None:
                    cores = round(max(0.0, cpu1 - cpu0) / max(1e-3, time.monotonic() - wall0), 2)
            placed = placement(server.text())
        layers = (f"{placed['layers_offloaded']}/{placed['layers_total']}" if placed.get("layers_total") else None)
        whole = bool(placed.get("layers_total")) and placed["layers_offloaded"] == placed["layers_total"]
        return {"decode": round(statistics.median(decode), 1) if decode else None,
                "prefill": round(statistics.median(prefill), 1) if prefill else None,
                "cores": cores, "layers": layers, "whole": whole}

    def _append(self, record: dict[str, Any]) -> None:
        with self._log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
