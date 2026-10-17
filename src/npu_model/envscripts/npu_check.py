"""Load a bundle with ORT GenAI on the NPU and measure it, run in the verify environment.

With --reference, feeds the reference's exact token IDs and reports greedy agreement,
sampled-position perplexity and needle retrieval. Without it, encodes --prompts (already
formatted with the bundle's chat template) with the bundle's own tokenizer.

The NPU stages are EPContext graphs, which only the QNN execution provider can execute, so
a successful generation means those stages ran on the NPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime_genai as og

WINDOW = 64  # static prefill window of the QNN pipeline
MIN_MAX_LENGTH = 1024  # GenAI 0.12.2 misbehaves when max_length is below the prefill window


def _say(msg: str) -> None:
    print(f"@@progress {msg}", flush=True)


def _generator(model: Any, needed: int) -> Any:
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, top_k=1, max_length=max(needed, MIN_MAX_LENGTH))
    return og.Generator(model, params)


def _greedy(model: Any, ids: list[int], new: int, eos: set[int]) -> tuple[list[int], float, float]:
    # Every append is a padded prefill window: the padded prompt plus new tokens must fit.
    gen = _generator(model, -(-len(ids) // WINDOW) * WINDOW + new + WINDOW)
    start = time.perf_counter()
    gen.append_tokens(ids)
    out: list[int] = []
    first: float | None = None
    while not gen.is_done() and len(out) < new:
        gen.generate_next_token()
        token = int(gen.get_next_tokens()[0])
        if first is None:
            first = time.perf_counter() - start
        out.append(token)
        if token in eos:
            break
    total = time.perf_counter() - start
    rate = (len(out) - 1) / (total - first) if first is not None and len(out) > 1 and total > first else 0.0
    return out, first or 0.0, rate


def _sampled_nll(model: Any, chunk: list[int], stride: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for pos in range(stride, len(chunk), stride):
        gen = _generator(model, -(-pos // WINDOW) * WINDOW + WINDOW)
        gen.append_tokens(chunk[:pos])
        logits = np.asarray(gen.get_logits(), dtype=np.float64).reshape(-1)
        top = logits.max()
        out[pos] = -(logits[chunk[pos]] - (top + math.log(np.exp(logits - top).sum())))
        del gen
    return out


def _degenerate(tokens: list[int]) -> bool:
    return len(tokens) >= 16 and Counter(tokens).most_common(1)[0][1] > len(tokens) // 2


def _median(values: list[float]) -> float | None:
    return sorted(values)[len(values) // 2] if values else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--prompts", help="JSON list of formatted prompts (used without --reference)")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--ppl-stride", type=int, default=16)
    ap.add_argument("--needle-answer", default="")
    args = ap.parse_args()

    report: dict[str, Any] = {"load": {"ok": False}}
    try:
        start = time.perf_counter()
        model = og.Model(args.bundle)
        tok = og.Tokenizer(model)
        report["load"] = {"ok": True, "seconds": round(time.perf_counter() - start, 2)}
    except Exception as exc:  # report, don't hide
        report["load"]["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        sys.exit(1)
    eos = {int(x) for x in np.asarray(tok.eos_token_ids).reshape(-1)}
    # Ollama sends the rendered prompt text to the same tokenizer: does encoding add BOS?
    bos = json.loads((Path(args.bundle) / "genai_config.json").read_text(encoding="utf-8"))["model"].get("bos_token_id")
    probe = [int(i) for i in tok.encode("Hi")]
    report["tokenizer"] = {"adds_bos": bool(probe) and bos is not None and probe[0] == bos}

    ttfts: list[float] = []
    rates: list[float] = []
    rows: list[dict[str, Any]] = []
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8")) if args.reference else None
    if reference:
        prompts = [(p["prompt"], p["input_ids"], p) for p in reference["prompts"]]
        max_new = int(reference["max_new_tokens"])
    else:
        prompts = [(text, [int(i) for i in tok.encode(text)], None) for text in json.loads(args.prompts or "[]")]
        max_new = args.max_new
    for index, (text, ids, ref) in enumerate(prompts, 1):
        _say(f"prompt {index}/{len(prompts)}")
        out, ttft, rate = _greedy(model, ids, max_new, eos)
        ttfts.append(ttft)
        if rate:
            rates.append(rate)
        row: dict[str, Any] = {"prompt": text, "output": tok.decode(out), "tokens": len(out),
                               "degenerate": _degenerate(out)}
        if ref is not None:
            want = ref["output_ids"]
            same = next((i for i in range(min(len(out), len(want))) if out[i] != want[i]), min(len(out), len(want)))
            row.update({"reference": ref["output"], "matching_prefix": same, "reference_tokens": len(want),
                        "exact": out == want})
        rows.append(row)
    report["prompts"] = rows
    report["speed"] = {"median_ttft_s": _median(ttfts), "median_tokens_per_s": _median(rates)}

    if reference and reference.get("perplexity", {}).get("chunks"):
        npu_vals: list[float] = []
        ref_vals: list[float] = []
        chunks = reference["perplexity"]["chunks"]
        for index, (chunk, ref_nll) in enumerate(zip(chunks, reference["perplexity"]["token_nll"]), 1):
            _say(f"perplexity chunk {index}/{len(chunks)}")
            for pos, value in _sampled_nll(model, chunk, args.ppl_stride).items():
                npu_vals.append(value)
                ref_vals.append(ref_nll[pos - 1])
        if npu_vals:
            ppl_npu = math.exp(sum(npu_vals) / len(npu_vals))
            ppl_ref = math.exp(sum(ref_vals) / len(ref_vals))
            report["perplexity"] = {"npu": round(ppl_npu, 4), "reference": round(ppl_ref, 4),
                                    "ratio": round(ppl_npu / ppl_ref, 4), "positions": len(npu_vals)}
    needles = []
    for needle in (reference or {}).get("needle", []):
        _say(f"long prompt {needle['input_len']} tokens")
        try:
            out, ttft, _ = _greedy(model, needle["input_ids"], 16, eos)
            text = tok.decode(out)
            needles.append({"input_len": needle["input_len"], "output": text, "ttft_s": round(ttft, 2),
                            "correct": bool(args.needle_answer) and args.needle_answer in text,
                            "reference_correct": needle["correct"]})
        except Exception as exc:  # report, don't hide
            needles.append({"input_len": needle["input_len"], "error": f"{type(exc).__name__}: {exc}"[:300]})
    report["needle"] = needles
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
