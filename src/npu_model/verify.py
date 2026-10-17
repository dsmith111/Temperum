"""Verification ladder for a converted bundle.

  quick     load on the NPU and generate with the bundle's own tokenizer and chat format
  standard  quick + quality against the source model: perplexity ratio on held-out text,
            greedy agreement, long-prompt retrieval, and chat-template agreement
  full      standard with more prompts, perplexity chunks and longer prompts

Load, NPU placement, quality and speed are separate claims in the report; a passing smoke
test is never reported as a quality pass.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from npu_model.core.errors import NpuModelError
from npu_model.templates import ChatTemplate

Progress = Callable[[str], None]
Popen = Callable[..., "subprocess.Popen[str]"]

ENVSCRIPTS = Path(__file__).resolve().parent / "envscripts"
LEVELS = ("none", "quick", "standard", "full")
PPL_PASS = 1.20  # M0: 1.04 (Qwen2.5), 1.06 (Qwen3), 1.15 (SmolLM2) at 8-bit; 2.42 at RTN 4-bit
PPL_FAIL = 1.50

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 17 + 25? Reply with just the number.",
    "Name the three primary colors.",
    "Translate to Spanish: 'Good morning, how are you?'",
    "Write a Python function that returns the square of a number.",
    "Explain in one sentence why the sky looks blue.",
    "List three planets in our solar system.",
    "What is the opposite of 'cold'?",
    "Summarize in one sentence: The meeting was moved from Tuesday to Thursday because the manager was travelling.",
    "Who wrote 'Romeo and Juliet'?",
    "Give me a synonym for 'happy'.",
    "If a train travels 60 km in one hour, how far does it travel in three hours?",
]
NEEDLE = {"fact": "The secret code for the vault is 4721.",
          "question": "What is the secret code for the vault? Reply with just the number.", "answer": "4721"}
CONVERSATIONS = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"},
     {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "What is 2+2?"}],
]


@dataclass(frozen=True)
class Settings:
    prompts: int
    max_new: int
    ppl_chunks: int
    needle_sizes: tuple[int, ...]


SETTINGS = {
    "quick": Settings(3, 32, 0, ()),
    "standard": Settings(6, 32, 4, (900, 1800)),
    "full": Settings(12, 48, 8, (400, 900, 1800, 3500)),
}


def reference_dtype(params: int | None, available_gb: float | None, total_gb: float | None = None) -> str | None:
    """fp32 when the weights fit the budget, else bf16, else no reference.

    The budget is half of the larger of free memory and half the machine's memory: desktop
    load makes "free" memory small but reclaimable (M0 ran the fp32 reference with ~2 GB free).
    """
    if not params or not (available_gb or total_gb):
        return "float32"
    budget = max(available_gb or 0.0, (total_gb or 0.0) / 2) * 2**30 * 0.5
    if params * 4 <= budget:
        return "float32"
    if params * 2 <= budget:
        return "bfloat16"
    return None


def script_env() -> dict[str, str]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1",
           "PYTHONIOENCODING": "utf-8"}
    env.pop("HF_TOKEN", None)
    return env


def run_script(python: Path, script: str, args: list[str], say: Progress, popen: Popen) -> tuple[int, str]:
    proc = popen([str(python), str(ENVSCRIPTS / script), *args], env=script_env(), stdout=subprocess.PIPE,
                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("@@progress "):
            say(line[len("@@progress "):].strip())
        else:
            tail = (tail + [line.rstrip()])[-20:]
    return proc.wait(), "\n".join(tail)


def template_check(chat: ChatTemplate, rendered: list[str], bos_text: str | None) -> dict[str, Any]:
    """Compare the generated Ollama template with the model's own (Jinja) rendering."""
    if not rendered:
        return {"status": "not-checked", "detail": "the model has no chat template"}
    results = []
    for conversation, expected in zip(CONVERSATIONS, rendered):
        ours = chat.render(conversation)
        if ours is None:
            return {"status": "not-checked", "detail": "custom template"}
        if bos_text and expected.startswith(bos_text) and not ours.startswith(bos_text):
            expected = expected[len(bos_text):]  # the runtime tokenizer adds BOS itself (probed)
        results.append({"match": ours == expected, "expected": expected, "generated": ours})
    status = "match" if all(r["match"] for r in results) else "mismatch"
    return {"status": status, "conversations": results}


def verdict(report: dict[str, Any], level: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    load = report.get("load", {})
    if not load.get("ok"):
        return "failed", [f"bundle did not load on the NPU: {load.get('error', 'unknown error')}"]
    prompts = report.get("prompts", [])
    if not prompts or any(p["tokens"] == 0 or p["degenerate"] for p in prompts):
        return "failed", ["generation was empty or degenerate (repeating one token)"]
    status = "passed"
    quality = report.get("quality", {})
    if level in ("standard", "full"):
        ratio = quality.get("perplexity", {}).get("ratio")
        if quality.get("skipped"):
            status, reasons = "warning", [f"quality not checked: {quality['skipped']}"]
        elif ratio is None:
            status, reasons = "warning", ["quality not measured"]
        elif ratio > PPL_FAIL:
            return "failed", [f"perplexity {ratio:.2f}x the source model (limit {PPL_FAIL})"]
        elif ratio > PPL_PASS:
            status = "warning"
            reasons.append(f"perplexity {ratio:.2f}x the source model (pass <= {PPL_PASS})")
        for needle in quality.get("needle", []):
            if needle.get("reference_correct") and not needle.get("correct"):
                status = "warning"
                reasons.append(f"missed a fact in a {needle['input_len']}-token prompt that the source model found")
        if report.get("template", {}).get("status") == "mismatch":
            status = "warning"
            reasons.append("generated Ollama template differs from the model's own chat template")
    return status, reasons


def verify(bundle: Path, snapshot: Path, chat: ChatTemplate, level: str, *, quantize_python: Path | None,
           verify_python: Path, work: Path, heldout: Path | None, params: int | None,
           available_gb: float | None, total_gb: float | None = None, progress: Progress | None = None,
           popen: Popen = subprocess.Popen) -> dict[str, Any]:
    if level not in LEVELS:
        raise NpuModelError("verify", "UNKNOWN_LEVEL", f"Unknown verification level {level}",
                            hint=f"Use one of: {', '.join(LEVELS)}")
    say = progress or (lambda _msg: None)
    if level == "none":
        return {"level": "none", "verdict": "not-verified", "reasons": ["verification skipped (--verify none)"]}
    settings = SETTINGS[level]
    work.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((bundle / "genai_config.json").read_text(encoding="utf-8"))
    context = int(cfg["model"]["context_length"])
    report: dict[str, Any] = {"level": level}
    reference_path: Path | None = None
    quality: dict[str, Any] = {}
    if level in ("standard", "full"):
        dtype = reference_dtype(params, available_gb, total_gb)
        if dtype is None or quantize_python is None or heldout is None:
            quality["skipped"] = ("not enough free memory to run the source model" if dtype is None
                                  else "no reference environment or held-out text")
        else:
            spec = {"prompts": PROMPTS[: settings.prompts], "max_new_tokens": settings.max_new,
                    "ppl_chunks": settings.ppl_chunks, "chunk_len": 256, "needle": NEEDLE,
                    "needle_sizes": [s for s in settings.needle_sizes if s + 256 <= context],
                    "conversations": CONVERSATIONS}
            spec_path = work / "reference_spec.json"
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            reference_path = work / "reference.json"
            say(f"running the source model ({dtype}) for reference answers")
            code, tail = run_script(quantize_python, "reference.py",
                                    ["--model", str(snapshot), "--spec", str(spec_path), "--heldout", str(heldout),
                                     "--dtype", dtype, "--out", str(reference_path)], say, popen)
            if code != 0 or not reference_path.is_file():
                quality["skipped"] = f"reference run failed: {tail[-400:]}"
                reference_path = None
            else:
                quality["reference_dtype"] = dtype
    npu_out = work / "npu_check.json"
    args = ["--bundle", str(bundle), "--out", str(npu_out), "--max-new", str(settings.max_new),
            "--needle-answer", NEEDLE["answer"]]
    if reference_path:
        args += ["--reference", str(reference_path)]
    else:
        formatted = [chat.render([{"role": "user", "content": p}]) or p for p in PROMPTS[: settings.prompts]]
        args += ["--prompts", json.dumps(formatted)]
    say("loading the bundle on the NPU")
    code, tail = run_script(verify_python, "npu_check.py", args, say, popen)
    npu: dict[str, Any] = json.loads(npu_out.read_text(encoding="utf-8")) if npu_out.is_file() else {}
    report["load"] = npu.get("load") or {"ok": False, "error": tail[-600:] or f"exit code {code}"}
    report["npu_placement"] = (
        {"status": "npu-stages-executed",
         "detail": "QNN EPContext stages generated tokens; only the QNN execution provider can run them"}
        if report["load"].get("ok") and npu.get("prompts") else {"status": "not-established"})
    report["prompts"] = npu.get("prompts", [])
    report["speed"] = npu.get("speed", {})
    if reference_path:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        rows = report["prompts"]
        quality["perplexity"] = npu.get("perplexity", {})
        quality["greedy_exact"] = sum(1 for r in rows if r.get("exact"))
        quality["mean_matching_prefix"] = round(
            sum(r.get("matching_prefix", 0) / max(1, r.get("reference_tokens", 1)) for r in rows) / max(1, len(rows)), 3)
        quality["needle"] = npu.get("needle", [])
        adds_bos = bool(npu.get("tokenizer", {}).get("adds_bos"))
        bos_text = _token_text(snapshot, cfg["model"].get("bos_token_id")) if adds_bos else None
        report["template"] = template_check(chat, reference.get("rendered", []), bos_text)
    report["quality"] = quality
    report["verdict"], report["reasons"] = verdict(report, level)
    return report


def _token_text(snapshot: Path, token_id: Any) -> str | None:
    try:
        config = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    spec = (config.get("added_tokens_decoder") or {}).get(str(token_id))
    content = spec.get("content") if isinstance(spec, dict) else None
    return content if isinstance(content, str) else None
