"""Source-model reference on CPU (fp32): greedy answers, held-out perplexity, needle prompts.

Everything is stored as token IDs so the NPU check (verify_genai.py) feeds identical inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _chat_ids(tok, text: str) -> list[int]:
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


def _heldout_chunks(tok, path: Path, count: int, length: int) -> list[list[int]]:
    buf: list[int] = []
    chunks: list[list[int]] = []
    sep = tok.encode("\n\n", add_special_tokens=False)
    for line in path.read_text(encoding="utf-8").splitlines():
        text = json.loads(line)["text"]
        if not text.strip():
            continue
        buf += tok.encode(text, add_special_tokens=False) + sep
        while len(buf) >= length and len(chunks) < count:
            chunks.append(buf[:length])
            buf = buf[length:]
        if len(chunks) >= count:
            break
    return chunks


def _needle_ids(tok, filler: list[int], needle: dict, target: int) -> list[int]:
    head = tok.encode(needle["fact"] + "\n\n", add_special_tokens=False)
    tail_text = "\n\n" + needle["question"]
    # chat overhead is small; leave ~32 tokens of slack for the template and question
    body = filler[: max(0, target - len(head) - 64)]
    content = tok.decode(head + body) + tail_text
    return _chat_ids(tok, content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--chunk-len", type=int, default=256)
    ap.add_argument("--needle-sizes", default="400,900,1800,3500")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    spec = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    max_new = spec["max_new_tokens"]
    eos = model.generation_config.eos_token_id

    def greedy(ids: list[int], new: int) -> list[int]:
        with torch.no_grad():
            out = model.generate(
                torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                max_new_tokens=new, do_sample=False, eos_token_id=eos,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        return out[0, len(ids):].tolist()

    result: dict = {"model": args.model, "max_new_tokens": max_new, "prompts": [], "perplexity": {}, "needle": []}
    t0 = time.time()
    for text in spec["prompts"]:
        ids = _chat_ids(tok, text)
        gen = greedy(ids, max_new)
        result["prompts"].append({"prompt": text, "input_ids": ids, "output_ids": gen,
                                  "output": tok.decode(gen, skip_special_tokens=True)})

    chunks = _heldout_chunks(tok, Path(args.heldout), args.chunks, args.chunk_len)
    nlls, token_nll = [], []
    with torch.no_grad():
        for chunk in chunks:
            x = torch.tensor([chunk])
            logits = model(x).logits[0, :-1].float()
            per_token = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none")
            token_nll.append([round(v, 5) for v in per_token.tolist()])  # index i predicts chunk[i+1]
            nlls.append(per_token.mean().item())
    result["perplexity"] = {"chunks": chunks, "chunk_nll": nlls, "token_nll": token_nll,
                            "ppl": math.exp(sum(nlls) / len(nlls))}

    filler: list[int] = []
    for chunk in _heldout_chunks(tok, Path(args.heldout), 64, 256):
        filler += chunk
    for size in [int(s) for s in args.needle_sizes.split(",")]:
        ids = _needle_ids(tok, filler, spec["needle"], size)
        gen = greedy(ids, 16)
        answer = tok.decode(gen, skip_special_tokens=True)
        result["needle"].append({"target": size, "input_len": len(ids), "input_ids": ids,
                                 "output": answer, "correct": spec["needle"]["answer"] in answer})
    result["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({
        "ppl": round(result["perplexity"]["ppl"], 3),
        "needle": [(n["input_len"], n["correct"]) for n in result["needle"]],
        "sample": [p["output"][:80] for p in result["prompts"][:4]],
        "seconds": result["seconds"],
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
