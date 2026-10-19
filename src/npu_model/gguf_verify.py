"""Verification ladder for GGUF bundles, run with the runtime's own llama-server on the NPU.

Claims are reported separately, as for the QNN target:

  load        llama-server loads the bundle with the Hexagon backend
  placement   how many layers the server put on an HTP (NPU) device, from its own log
  generation  greedy answers for the source model's prompt token IDs, compared token by token
  quality     needle retrieval in long prompts, and perplexity of the bundle on the NPU against
              the 16-bit GGUF on the CPU (llama-perplexity, when the runtime ships it)
  speed       prompt and generation rates reported by the server
  template    the chat template the server applies (the GGUF's own) against transformers'
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from npu_model import calib, paths
from npu_model.core.errors import NpuModelError
from npu_model.gguf import GgufPlan, LlamaPayload, _exe, quantize
from npu_model.verify import (CONVERSATIONS, NEEDLE, PPL_PASS, PROMPTS, SETTINGS, reference_dtype, run_script,
                              verdict)

Progress = Callable[[str], None]
LOAD_TIMEOUT = 300.0
REQUEST_TIMEOUT = 600.0


def runtime_env(payload: LlamaPayload) -> dict[str, str]:
    """Environment the fork gives llama-server for its Hexagon runner directory."""
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env["PATH"] = os.pathsep.join([str(payload.root), str(payload.hexagon), env.get("PATH", "")])
    backend = sorted(payload.hexagon.glob("*ggml-hexagon*"))
    if backend:
        env["GGML_BACKEND_PATH"] = str(backend[0])
    existing = env.get("ADSP_LIBRARY_PATH")
    env["ADSP_LIBRARY_PATH"] = str(payload.htp or payload.hexagon) + (f";{existing}" if existing else "")
    return env


def placement(log: str) -> dict[str, Any]:
    """Layers and weights the server placed on HTP devices, parsed from its log."""
    offloaded = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
    buffers: dict[str, float] = {}
    for device, size in re.findall(r"\b(HTP\d+(?:-REPACK)?) model buffer size\s*=\s*([\d.]+) MiB", log):
        buffers[device] = buffers.get(device, 0.0) + float(size)
    result: dict[str, Any] = {"htp_weight_mib": round(sum(buffers.values()), 1), "buffers": buffers}
    if offloaded:
        result["layers_offloaded"], result["layers_total"] = int(offloaded.group(1)), int(offloaded.group(2))
    on_npu = bool(buffers) and result.get("layers_offloaded", 0) > 0
    result["status"] = "weights-on-npu" if on_npu else "not-established"
    return result


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaServer:
    """llama-server with one model on the NPU, for the duration of a with-block."""

    def __init__(self, payload: LlamaPayload, model: Path, context: int, log: Path, device: str = "HTP0"):
        self.payload, self.model, self.context, self.log_path, self.device = payload, model, context, log, device
        self.port = _free_port()
        self.process: subprocess.Popen[bytes] | None = None

    def command(self) -> list[str]:
        return [str(self.payload.server), "--model", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.context), "-np", "1", "-ngl", "999", "--device", self.device, "--no-webui",
                "--offline", "--log-verbosity", "4"]

    def __enter__(self) -> LlamaServer:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("wb")
        self.process = subprocess.Popen(self.command(), env=runtime_env(self.payload), stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.close()
        deadline = time.monotonic() + LOAD_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}: {self.tail()}")
            try:
                if self._request("GET", "/health", None, 5).get("status") == "ok":
                    return self
            except (OSError, http.client.HTTPException, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError(f"llama-server did not become ready in {LOAD_TIMEOUT:.0f}s: {self.tail()}")

    def __exit__(self, *_exc: object) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)

    def text(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace") if self.log_path.is_file() else ""

    def tail(self, chars: int = 1500) -> str:
        return self.text()[-chars:]

    def _request(self, method: str, path: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            payload = json.dumps(body).encode() if body is not None else None
            connection.request(method, path, body=payload, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
            if response.status != 200:
                raise ValueError(f"{path} returned HTTP {response.status}: {data[:300]!r}")
            value = json.loads(data)
            return value if isinstance(value, dict) else {"value": value}
        finally:
            connection.close()

    def complete(self, prompt: list[int] | str, n_predict: int) -> dict[str, Any]:
        """Greedy completion of token ids, or of text the server tokenizes with the model's own tokenizer."""
        return self._request("POST", "/completion", {
            "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "cache_prompt": False,
            "return_tokens": True, "stream": False}, REQUEST_TIMEOUT)

    def apply_template(self, messages: list[dict[str, str]]) -> str | None:
        try:
            value = self._request("POST", "/apply-template", {"messages": messages}, 60)
        except (OSError, http.client.HTTPException, ValueError):
            return None
        prompt = value.get("prompt")
        return prompt if isinstance(prompt, str) else None


def _degenerate(tokens: list[int]) -> bool:
    return len(tokens) >= 16 and max(tokens.count(t) for t in set(tokens)) > len(tokens) // 2


def _median(values: list[float]) -> float | None:
    return sorted(values)[len(values) // 2] if values else None


def perplexity(payload: LlamaPayload, model: Path, text: Path, *, npu: bool, context: int, chunks: int,
               log: Path) -> float | None:
    exe = payload.root / _exe("llama-perplexity")
    if not exe.is_file():
        return None
    cmd = [str(exe), "-m", str(model), "-f", str(text), "-c", str(context), "--chunks", str(chunks)]
    cmd += ["-ngl", "999", "--device", "HTP0"] if npu else ["-ngl", "0", "--device", "none"]
    proc = subprocess.run(cmd, env=runtime_env(payload), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=3600)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n" + (proc.stdout or "") + (proc.stderr or ""))
    match = re.search(r"Final estimate: PPL = ([\d.]+)", (proc.stdout or "") + (proc.stderr or ""))
    return float(match.group(1)) if match else None


def _heldout_text(path: Path, out: Path, rows: int = 400) -> Path:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines()[:rows]:
        text = json.loads(line).get("text", "")
        if text.strip():
            lines.append(text.rstrip("\n"))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_gguf(bundle: Path, snapshot: Path, plan: GgufPlan, chains: Any, work: Path, say: Progress,
                source_gguf: Path | None = None) -> dict[str, Any]:
    level = plan.verify_level
    settings = SETTINGS[level]
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"level": level, "format": "gguf"}
    quality: dict[str, Any] = {}
    reference: dict[str, Any] | None = None
    heldout = calib.ensure_corpus(calib.HELDOUT, paths.data_dir()).path if level in ("standard", "full") else None
    if heldout is not None:
        params = int(plan.estimates["params_billions"] * 1e9)
        dtype = reference_dtype(params, plan.host.available_ram_gb, plan.host.total_ram_gb)
        spec = {"prompts": PROMPTS[: settings.prompts], "max_new_tokens": settings.max_new,
                "ppl_chunks": settings.ppl_chunks, "chunk_len": 256, "needle": NEEDLE,
                "needle_sizes": [s for s in settings.needle_sizes if s + 256 <= plan.context],
                "conversations": CONVERSATIONS}
        spec_path = work / "reference_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        out = work / "reference.json"
        args = ["--model", str(snapshot), "--spec", str(spec_path), "--heldout", str(heldout), "--out", str(out)]
        if dtype is None:
            # Too large to run here: the source tokenizer still gives identical inputs and the chat
            # renderings, so retrieval and the template are checked; answers are not compared.
            quality["skipped"] = "not enough free memory to run the source model"
            args.append("--tokenizer-only")
            say("reading the source model's tokenizer and chat template")
        else:
            args += ["--dtype", dtype]
            say("running the source model for reference answers")
        code, tail = run_script(chains.python_exe("quantize"), "reference.py", args, say, subprocess.Popen)
        if code == 0 and out.is_file():
            reference = json.loads(out.read_text(encoding="utf-8"))
        else:
            quality["skipped"] = f"reference run failed: {tail[-400:]}"

    say("loading the bundle on the NPU with llama-server")
    rows: list[dict[str, Any]] = []
    prompt_rates: list[float] = []
    decode_rates: list[float] = []
    try:
        with LlamaServer(plan.payload, bundle / "model.gguf", plan.context, work / "llama-server.log") as server:
            report["load"] = {"ok": True}
            report["npu_placement"] = placement(server.text())
            prompts: list[tuple[str, list[int] | str, dict[str, Any] | None]]
            if reference:
                prompts = [(p["prompt"], p["input_ids"], p) for p in reference["prompts"]]
            else:  # no source-model answers (quick, or the reference was skipped): the bundle's own chat format
                prompts = [(text, server.apply_template([{"role": "user", "content": text}]) or text, None)
                           for text in PROMPTS[: settings.prompts]]
            for index, (question, prompt, ref) in enumerate(prompts, 1):
                say(f"prompt {index}/{len(prompts)}")
                result = server.complete(prompt, settings.max_new)
                tokens = [int(t) for t in result.get("tokens") or []]
                timings = result.get("timings") or {}
                if timings.get("prompt_per_second"):
                    prompt_rates.append(float(timings["prompt_per_second"]))
                if timings.get("predicted_per_second") and len(tokens) > 1:
                    decode_rates.append(float(timings["predicted_per_second"]))
                row = {"prompt": question, "output": result.get("content", ""), "tokens": len(tokens),
                       "degenerate": _degenerate(tokens), "prompt_ms": timings.get("prompt_ms")}
                if ref is not None and "output_ids" in ref:  # tokenizer-only references have no answers
                    want = ref["output_ids"]
                    same = next((i for i in range(min(len(tokens), len(want))) if tokens[i] != want[i]),
                                min(len(tokens), len(want)))
                    row.update({"reference": ref["output"], "matching_prefix": same, "reference_tokens": len(want),
                                "exact": tokens == want})
                rows.append(row)
            needles = []
            for needle in (reference or {}).get("needle", []):
                say(f"long prompt {needle['input_len']} tokens")
                result = server.complete(needle["input_ids"], 16)
                content = str(result.get("content", ""))
                needles.append({"input_len": needle["input_len"], "output": content,
                                "correct": NEEDLE["answer"] in content, "reference_correct": needle.get("correct"),
                                "prompt_ms": (result.get("timings") or {}).get("prompt_ms")})
            quality["needle"] = needles
            rendered = (reference or {}).get("rendered", [])
            checks = [(c, r, server.apply_template(c)) for c, r in zip(CONVERSATIONS, rendered)]
            report["template"] = {
                "status": "match" if checks and all(r == g for _, r, g in checks) else
                          ("mismatch" if checks else "unchecked"),
                "conversations": [{"match": r == g, "expected": r, "generated": g} for _, r, g in checks]}
    except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
        report["load"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1500]}
        report.setdefault("npu_placement", {"status": "not-established"})
    report["prompts"] = rows
    report["speed"] = {"median_prompt_tokens_per_s": _median(prompt_rates),
                       "median_tokens_per_s": _median(decode_rates)}
    timed = [n for n in quality.get("needle", []) if n["prompt_ms"]]
    if timed:  # the chat prompts are a few dozen tokens; the longest prompt shows prefill speed
        longest = max(timed, key=lambda n: n["input_len"])
        report["speed"]["long_prompt_tokens"] = longest["input_len"]
        report["speed"]["long_prompt_tokens_per_s"] = round(longest["input_len"] / (longest["prompt_ms"] / 1000), 1)
    scored = [r for r in rows if "exact" in r]
    if scored:
        quality["greedy_exact"] = sum(1 for r in scored if r["exact"])
        quality["mean_matching_prefix"] = round(
            sum(r["matching_prefix"] / max(1, r["reference_tokens"]) for r in scored) / len(scored), 3)
    if report["load"].get("ok") and heldout is not None and source_gguf is not None and settings.ppl_chunks:
        # The reference is a Q8_0 copy on the CPU: within about 0.5% of 16-bit, but ggml's 16-bit
        # CPU kernels are slow here (BF16 took 211 s per 4 chunks for a 1.5B model).
        say("measuring perplexity (the bundle on the NPU vs a Q8_0 copy on the CPU)")
        text = _heldout_text(heldout, work / "heldout.txt")
        log = work / "perplexity.log"
        reference_gguf = work / "reference-Q8_0.gguf"
        chunks = settings.ppl_chunks * 4  # 16 at standard: 8 gave 1.14x where 16 gave 1.09x
        npu = cpu = None
        try:
            quantize(plan.payload, source_gguf, reference_gguf, "Q8_0", log, say)
            npu = perplexity(plan.payload, bundle / "model.gguf", text, npu=True, context=512, chunks=chunks, log=log)
            cpu = perplexity(plan.payload, reference_gguf, text, npu=False, context=512, chunks=chunks, log=log)
        except (NpuModelError, OSError, subprocess.SubprocessError) as exc:
            quality["perplexity_error"] = str(exc)[:400]
        finally:
            reference_gguf.unlink(missing_ok=True)
        if npu and cpu:
            quality["perplexity"] = {"npu": npu, "reference": cpu, "ratio": round(npu / cpu, 4), "chunks": chunks,
                                     "reference_kind": "Q8_0 GGUF on the CPU (llama-perplexity, 512-token chunks)"}
    report["quality"] = quality
    if report["load"].get("ok") and report["npu_placement"].get("status") != "weights-on-npu":
        report["load"]["note"] = "the server loaded, but its log shows no weights on an HTP device"
    report["verdict"], report["reasons"] = verdict(report, level)
    if report["verdict"] != "failed" and report["npu_placement"].get("status") != "weights-on-npu":
        report["verdict"] = "failed"
        report["reasons"] = ["the model did not run on the NPU (no layers on an HTP device)"] + report["reasons"]
    ratio = quality.get("perplexity", {}).get("ratio")
    if report["verdict"] == "warning" and plan.quant == "Q4_0" and ratio and ratio > PPL_PASS:
        report["reasons"].append("Q4_0 is the fastest type; --quant Q4_K_M or Q8_0 keep more quality")
    return report
