"""Load a converted bundle with ORT GenAI + QNN and compare it with the CPU reference.

Feeds the reference's exact token IDs, so differences come from the converted model,
not from tokenization or chat templates. Reports greedy agreement, held-out
perplexity ratio, needle retrieval, and rough speed (time to first token, tokens/s).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

import sys


def _progress(msg: str) -> None:
    print(f"[progress] {msg}", file=sys.stderr, flush=True)


MIN_MAX_LENGTH = 1024  # static QNN pipelines overflow the KV buffer if max_length < prefill window


def _generator(model, max_length: int):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, top_k=1, max_length=max(max_length, MIN_MAX_LENGTH))
    return og.Generator(model, params)


def _greedy(model, ids: list[int], new: int, eos: set[int]) -> tuple[list[int], float, float]:
    window = 64  # static prefill window; padded prompt + new tokens must fit in max_length
    gen = _generator(model, -(-len(ids) // window) * window + new + window)
    t0 = time.perf_counter()
    gen.append_tokens(ids)
    out: list[int] = []
    first = None
    while not gen.is_done() and len(out) < new:
        gen.generate_next_token()
        tok = int(gen.get_next_tokens()[0])
        if first is None:
            first = time.perf_counter() - t0
        out.append(tok)
        if tok in eos:
            break
    total = time.perf_counter() - t0
    rate = (len(out) - 1) / (total - first) if first is not None and len(out) > 1 and total > first else 0.0
    return out, first or 0.0, rate


def _sampled_nll(model, chunk: list[int], stride: int) -> dict[int, float]:
    """NLL of chunk[pos] given chunk[:pos], at sampled positions.

    Static QNN pipelines process every append_tokens() call as a padded prefill window, so
    token-by-token teacher forcing is not possible; each position gets its own prefill.
    """
    out: dict[int, float] = {}
    for pos in range(stride, len(chunk), stride):
        _progress(f"  nll pos {pos}")
        gen = _generator(model, pos + 1)
        gen.append_tokens(chunk[:pos])
        logits = np.asarray(gen.get_logits(), dtype=np.float64).reshape(-1)
        m = logits.max()
        out[pos] = -(logits[chunk[pos]] - (m + math.log(np.exp(logits - m).sum())))
        del gen
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ppl-chunks", type=int, default=4)
    ap.add_argument("--ppl-stride", type=int, default=16, help="score every Nth position of each chunk")
    ap.add_argument("--plugin-qnn", action="store_true", help="register the onnxruntime-qnn 2.x plugin EP")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.plugin_qnn:
        import onnxruntime_qnn

        og.register_execution_provider_library("QNNExecutionProvider", onnxruntime_qnn.get_library_path())

    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    cfg = json.loads(Path(args.bundle, "genai_config.json").read_text(encoding="utf-8"))
    context = cfg["model"].get("context_length")
    max_len = cfg["search"].get("max_length") or context
    t0 = time.perf_counter()
    model = og.Model(args.bundle)
    load_s = time.perf_counter() - t0
    tok = og.Tokenizer(model)
    eos = set(int(x) for x in np.asarray(tok.eos_token_ids).reshape(-1))

    rows, ttfts, rates, exact, agree = [], [], [], 0, []
    for p in ref["prompts"]:
        _progress(f"prompt {p['prompt'][:30]!r}")
        out, ttft, rate = _greedy(model, p["input_ids"], ref["max_new_tokens"], eos)
        want = p["output_ids"]
        n = min(len(out), len(want))
        same = next((i for i in range(n) if out[i] != want[i]), n)
        exact += int(out == want)
        agree.append(same / max(1, len(want)))
        ttfts.append(ttft)
        if rate:
            rates.append(rate)
        rows.append({"prompt": p["prompt"], "npu": tok.decode(out), "reference": p["output"],
                     "matching_prefix": same, "reference_len": len(want)})

    npu_vals, ref_vals = [], []
    for chunk, ref_tok in zip(ref["perplexity"]["chunks"][: args.ppl_chunks],
                              ref["perplexity"]["token_nll"][: args.ppl_chunks]):
        _progress(f"perplexity chunk len={len(chunk)}")
        for pos, value in _sampled_nll(model, chunk, args.ppl_stride).items():
            npu_vals.append(value)
            ref_vals.append(ref_tok[pos - 1])
    ppl_npu = math.exp(sum(npu_vals) / len(npu_vals))
    ppl_ref = math.exp(sum(ref_vals) / len(ref_vals))

    needles = []
    for n in ref["needle"]:
        _progress(f"needle {n['input_len']}")
        if max_len and -(-n["input_len"] // 64) * 64 + 16 + 64 > max_len:
            needles.append({"input_len": n["input_len"], "skipped": f"exceeds max_length {max_len}"})
            continue
        try:
            out, ttft, _ = _greedy(model, n["input_ids"], 16, eos)
            text = tok.decode(out)
            needles.append({"input_len": n["input_len"], "output": text, "ttft_s": round(ttft, 2),
                            "correct": "4721" in text, "reference_correct": n["correct"]})
        except Exception as exc:  # report, don't hide
            needles.append({"input_len": n["input_len"], "error": str(exc)[:300]})

    summary = {
        "bundle": args.bundle, "context_length": context, "max_length": max_len,
        "load_s": round(load_s, 2),
        "greedy_exact": f"{exact}/{len(ref['prompts'])}",
        "mean_matching_prefix": round(sum(agree) / len(agree), 3),
        "ppl_npu": round(ppl_npu, 3), "ppl_reference": round(ppl_ref, 3),
        "ppl_ratio": round(ppl_npu / ppl_ref, 3),
        "median_ttft_s": round(sorted(ttfts)[len(ttfts) // 2], 3),
        "median_tokens_per_s": round(sorted(rates)[len(rates) // 2], 1) if rates else None,
        "needle": needles,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "prompts": rows}, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    for r in rows[:6]:
        print(f"- {r['prompt'][:50]!r}\n    npu: {r['npu'][:90]!r}\n    ref: {r['reference'][:90]!r}")


if __name__ == "__main__":
    main()
