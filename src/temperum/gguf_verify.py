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

from temperum import calib, paths, schedlog
from temperum.gguf import DRAFT_FILE, GgufPlan, LlamaPayload, _exe, last_error_line
from temperum.templates import ChatTemplate
from temperum.verify import (CONVERSATIONS, NEEDLE, PPL_PASS, PROMPTS, SETTINGS, reference_dtype, run_script,
                              template_check, verdict)

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


def placement_line(placed: dict[str, Any]) -> str:
    """One progress line on where the server put the model: for the person watching, and for the
    interface's NPU row."""
    if placed.get("status") == "weights-on-npu":
        line = (f"on the NPU: {placed['layers_offloaded']}/{placed['layers_total']} layers, "
                f"{placed['htp_weight_mib']} MiB of weights")
        return line + (f"; {schedlog.describe(placed['ops'])}" if placed.get("ops") else "")
    return "on the NPU: no layers (the server put no weights on an HTP device)"


def op_placement(payload: LlamaPayload, model: Path, work: Path, say: Progress) -> dict[str, Any] | None:
    """What runs where: one short completion with ggml's scheduler printing its assignments, read back as the
    steady-state graph's summary (nodes per backend, splits, the operations that fell back to the CPU)."""
    say("reading which operations the NPU runs (one short completion with the scheduler's log)")
    server = LlamaServer(payload, model, 512, work / "ops.log")
    server.extra_env = {"GGML_SCHED_DEBUG": "2"}
    server.extra_args = ["--log-verbosity", "5"]
    with server:
        server.complete("The capital of France is", 2)
        graph = schedlog.decode_graph(schedlog.parse(server.text()))
    return schedlog.summarise(graph) if graph is not None and graph.nodes else None


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
        self.extra_env: dict[str, str] = {}   # on top of the fork's runtime environment
        self.extra_args: list[str] = []       # after the standard command

    def command(self) -> list[str]:
        return [str(self.payload.server), "--model", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.context), "-np", "1", "-ngl", "999", "--device", self.device, "--no-webui",
                "--offline", "--log-verbosity", "4", *self.extra_args]

    def __enter__(self) -> LlamaServer:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("wb")
        self.process = subprocess.Popen(self.command(), env={**runtime_env(self.payload), **self.extra_env}, stdout=log,
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

    def bos_text(self) -> str | None:
        """The BOS token text when the server adds one to every prompt (the GGUF's add_bos_token)."""
        try:
            value = self._request("POST", "/tokenize", {"content": "", "add_special": True, "with_pieces": True}, 60)
        except (OSError, http.client.HTTPException, ValueError):
            return None
        tokens = value.get("tokens") or []
        piece = tokens[0].get("piece") if tokens and isinstance(tokens[0], dict) else None
        return piece if isinstance(piece, str) else None

    def apply_template(self, messages: list[dict[str, str]]) -> str | None:
        try:
            value = self._request("POST", "/apply-template", {"messages": messages}, 60)
        except (OSError, http.client.HTTPException, ValueError):
            return None
        prompt = value.get("prompt")
        return prompt if isinstance(prompt, str) else None


def _degenerate(tokens: list[int]) -> bool:
    return len(tokens) >= 16 and max(tokens.count(t) for t in set(tokens)) > len(tokens) // 2


def paired_check(plan: GgufPlan, bundle: Path, draft: Path, prompts: list[tuple[str, list[int] | str, Any]],
                 plain_rows: list[dict[str, Any]], max_new: int, work: Path, say: Progress) -> dict[str, Any]:
    """The bundle's draft beside the target on the NPU, the way the fork runs the pair: the same prompts again
    with speculative decoding, each against its plain decode speed, with the draft's own count of proposed and
    accepted tokens. The draft pays only where its guesses are accepted, and the numbers say where."""
    block = int(getattr(plan, "draft_block", 0) or 16)
    result: dict[str, Any] = {"block_size": block, "file": draft.name}
    say("loading the pair: the draft beside the target on the NPU")
    try:
        server = LlamaServer(plan.payload, bundle / "model.gguf", plan.context, work / "llama-server-draft.log")
        server.extra_args = ["--spec-type", "draft-dflash", "--spec-draft-model", str(draft),
                             "--spec-draft-n-max", str(block)]
        with server:
            result["loaded"] = True
            result["placement"] = placement(server.text())
            paired: list[dict[str, Any]] = []
            for index, (question, prompt, _ref) in enumerate(prompts, 1):
                say(f"prompt {index}/{len(prompts)} with the draft")
                out = server.complete(prompt, max_new)
                timings = out.get("timings") or {}
                plain = plain_rows[index - 1] if index - 1 < len(plain_rows) else {}
                speed, base = timings.get("predicted_per_second"), plain.get("decode_tokens_per_s")
                drafted, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
                paired.append({"prompt": question, "tokens": len(out.get("tokens") or []),
                               "decode_tokens_per_s": speed, "plain_tokens_per_s": base,
                               "speedup": round(float(speed) / float(base), 2) if speed and base else None,
                               "drafted": drafted, "accepted": accepted,
                               "acceptance": (round(float(accepted) / float(drafted), 3)
                                              if drafted and accepted is not None else None),
                               "same_answer": out.get("content") == plain.get("output")})
            result["prompts"] = paired
            speedups = [float(p["speedup"]) for p in paired if p["speedup"]]
            rates = [float(p["acceptance"]) for p in paired if p["acceptance"] is not None]
            result["median_speedup"] = _median(speedups)
            result["speedup_range"] = [min(speedups), max(speedups)] if speedups else None
            result["acceptance_range"] = [min(rates), max(rates)] if rates else None
    except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
        result["loaded"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"[:1500]
    return result


def draft_line(paired: dict[str, Any]) -> str:
    """One line on what the paired draft did: for the verdict's reasons and the ready panel."""
    where = paired.get("placement") or {}
    on_npu = where.get("status") == "weights-on-npu"
    span = paired.get("speedup_range")
    rates = paired.get("acceptance_range")
    speed = (f"decode {span[0]:.2f}x to {span[1]:.2f}x of plain across the prompts (median "
             f"{paired.get('median_speedup') or 0:.2f}x)" if span else "decode speed not measured")
    accept = f"; {rates[0]:.0%} to {rates[1]:.0%} of its guesses accepted" if rates else ""
    return (f"draft {'on the NPU beside the target' if on_npu else 'loaded (placement not established)'}: "
            f"{speed}{accept}; it pays where its guesses are accepted, and PARAMETER draft_num_predict 0 turns it off")


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


def _without_bos(expected: str, generated: str | None, bos: str | None) -> str:
    """transformers renders the BOS text into the prompt; llama-server adds that token when tokenizing instead."""
    if bos and expected.startswith(bos) and not (generated or "").startswith(bos):
        return expected[len(bos):]
    return expected


def verify_gguf(bundle: Path, snapshot: Path, plan: GgufPlan, chains: Any, work: Path, say: Progress,
                reference_gguf: Path | None = None, chat: ChatTemplate | None = None) -> dict[str, Any]:
    level = plan.verify_level
    settings = SETTINGS[level]
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"level": level, "format": "gguf"}
    quality: dict[str, Any] = {}
    reference: dict[str, Any] | None = None
    heldout = calib.ensure_corpus(calib.HELDOUT, paths.data_dir()).path if level in ("standard", "full") else None
    if level != "none":
        params = int(plan.estimates["params_billions"] * 1e9)
        dtype = reference_dtype(params, plan.host.available_ram_gb, plan.host.total_ram_gb) if heldout else None
        spec = {"prompts": PROMPTS[: settings.prompts], "max_new_tokens": settings.max_new,
                "ppl_chunks": settings.ppl_chunks, "chunk_len": 256, "needle": NEEDLE,
                "needle_sizes": [s for s in settings.needle_sizes if s + 256 <= plan.context] if heldout else [],
                "conversations": CONVERSATIONS}
        spec_path = work / "reference_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        out = work / "reference.json"
        args = ["--model", str(snapshot), "--spec", str(spec_path), "--out", str(out)]
        if heldout is not None:
            args += ["--heldout", str(heldout)]
        if dtype is None:
            # Quick level, or too large to run here: the source tokenizer still gives identical inputs and
            # the chat renderings, so the template (and at standard, retrieval) is checked; answers are not.
            if heldout is not None:
                quality["skipped"] = "not enough free memory to run the source model"
            args.append("--tokenizer-only")
            say("reading the source model's tokenizer and chat template")
        else:
            args += ["--dtype", dtype]
            say("running the source model for reference answers")
        code, tail = run_script(chains.python_exe("quantize"), "reference.py", args, say, subprocess.Popen)
        if code == 0 and out.is_file():
            reference = json.loads(out.read_text(encoding="utf-8"))
            if reference.get("model_error"):  # newer than the transformers here: its tokenizer still served
                quality["skipped"] = (f"{reference['model_error']}, so the source model was not run; its tokenizer "
                                      "gave the inputs and the chat template")
        else:
            quality["skipped"] = f"reference run failed: {last_error_line(tail) or tail[-400:]}"

    say("loading the bundle on the NPU with llama-server")
    rows: list[dict[str, Any]] = []
    prompt_rates: list[float] = []
    decode_rates: list[float] = []
    prompts: list[tuple[str, list[int] | str, dict[str, Any] | None]] = []
    try:
        with LlamaServer(plan.payload, bundle / "model.gguf", plan.context, work / "llama-server.log") as server:
            report["load"] = {"ok": True}
            report["npu_placement"] = placement(server.text())
            say(placement_line(report["npu_placement"]))
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
                       "degenerate": _degenerate(tokens), "prompt_ms": timings.get("prompt_ms"),
                       "decode_tokens_per_s": timings.get("predicted_per_second"),
                       "stop": result.get("stop_type")}  # eos | word | limit: how the answer ended
                if ref is not None and "output_ids" in ref:  # tokenizer-only references have no answers
                    want = ref["output_ids"]
                    same = next((i for i in range(min(len(tokens), len(want))) if tokens[i] != want[i]),
                                min(len(tokens), len(want)))
                    row.update({"reference": ref["output"], "matching_prefix": same, "reference_tokens": len(want),
                                "exact": tokens == want, "reference_degenerate": _degenerate(want)})
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
            bos = server.bos_text()
            checks = [(c, _without_bos(r, g, bos), g) for c, r, g in
                      ((c, r, server.apply_template(c)) for c, r in zip(CONVERSATIONS, rendered)
                       if r is not None)]
            report["template"] = {
                "status": "match" if checks and all(r == g for _, r, g in checks) else
                          ("mismatch" if checks else "unchecked"),
                "conversations": [{"match": r == g, "expected": r, "generated": g} for _, r, g in checks]}
            refused = (reference or {}).get("rendered_errors") or []
            if refused:  # transformers would not render some test conversations with the model's template
                report["template"]["not_rendered"] = refused
            if chat is not None and any(r is not None for r in rendered):  # the bundle's own Ollama template
                report["template"]["ollama"] = template_check(chat, rendered, server.bos_text())
    except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
        report["load"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1500]}
        report.setdefault("npu_placement", {"status": "not-established"})
    if report["load"].get("ok") and report["npu_placement"].get("status") == "weights-on-npu":
        try:
            ops = op_placement(plan.payload, bundle / "model.gguf", work, say)
        except (OSError, RuntimeError, ValueError, http.client.HTTPException) as exc:
            ops = None
            report["npu_placement"]["ops_error"] = f"{type(exc).__name__}: {exc}"[:400]
        if ops:
            report["npu_placement"]["ops"] = ops
            say("ops: " + schedlog.describe(ops))
    draft_file = bundle / DRAFT_FILE
    if report["load"].get("ok") and getattr(plan, "draft", None) is not None and draft_file.is_file():
        report["draft"] = paired_check(plan, bundle, draft_file, prompts, rows, settings.max_new, work, say)
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
    if report["load"].get("ok") and heldout is not None and reference_gguf is not None and settings.ppl_chunks:
        # The reference is the Q8_0 conversion on the CPU: within about 0.5% of 16-bit, but ggml's
        # 16-bit CPU kernels are slow here (BF16 took 211 s per 4 chunks for a 1.5B model).
        say("measuring perplexity (the bundle on the NPU vs the Q8_0 conversion on the CPU)")
        text = _heldout_text(heldout, work / "heldout.txt")
        log = work / "perplexity.log"
        chunks = settings.ppl_chunks * 4  # 16 at standard: 8 gave 1.14x where 16 gave 1.09x
        npu = cpu = None
        try:
            npu = perplexity(plan.payload, bundle / "model.gguf", text, npu=True, context=512, chunks=chunks, log=log)
            cpu = perplexity(plan.payload, reference_gguf, text, npu=False, context=512, chunks=chunks, log=log)
        except (OSError, subprocess.SubprocessError) as exc:
            quality["perplexity_error"] = str(exc)[:400]
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
    paired = report.get("draft")
    if paired is not None:
        if not paired.get("loaded"):
            report["verdict"] = "failed"
            report["reasons"] = [f"the draft did not load beside the target: {paired.get('error', 'unknown error')}"
                                 ] + report["reasons"]
        else:
            report["reasons"].append(draft_line(paired))
    if report["verdict"] != "failed" and rows and all(r.get("stop") == "limit" for r in rows):
        if chat is not None and chat.family == "raw":  # expected of a base model: it continues text
            report["reasons"].append(f"a base model with no chat template: every answer ran to the "
                                     f"{settings.max_new}-token limit, as it does not end its turns by itself")
        else:
            # Every answer ran to the token limit: the model does not end its turns (a chat format it was not
            # trained on, or a base model the converter did not recognize as one). Through the fork its
            # answers run on until the context is full.
            report["verdict"] = "warning"
            report["reasons"].append(f"no answer ended by itself within {settings.max_new} tokens: the model "
                                     "keeps generating after its reply (a base model, or a chat format it was "
                                     "not trained on), so answers through the fork run on")
    ratio = quality.get("perplexity", {}).get("ratio")
    lost = report["verdict"] == "failed" and any(r["degenerate"] and not r.get("reference_degenerate")
                                                   for r in rows)
    if ((report["verdict"] == "warning" and ratio and ratio > PPL_PASS) or lost) and plan.quant == "Q4_0":
        small = plan.estimates["params_billions"] < 0.5
        report["reasons"].append("Q4_0 is the fastest type; --quant Q4_K_M or Q8_0 keep more quality"
                                 + (" (a model this small loses a lot at 4 bits, and its file is tiny anyway)"
                                    if small else ""))
    return report
