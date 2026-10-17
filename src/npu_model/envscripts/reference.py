"""Source-model reference for verification, run in the quantize environment.

Records token IDs so the NPU check feeds identical inputs (differences then come from the
converted model, not tokenization). Also renders the model's own chat template for a few
conversations so the converter can compare its generated Ollama template.

Spec (JSON): prompts, max_new_tokens, ppl_chunks, chunk_len, needle {fact, question,
answer}, needle_sizes, conversations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    return [int(i) for i in value]


def _chat_ids(tok: Any, text: str) -> list[int]:
    messages = [{"role": "user", "content": text}]
    if tok.chat_template:
        # Thinking models answer directly; the converter's quality check is not a reasoning test.
        return _ids(tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False))
    return _ids(tok.encode(text))


def _heldout_chunks(tok: Any, path: Path, count: int, length: int) -> list[list[int]]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval()
    eos = model.generation_config.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos)

    def greedy(ids: list[int], new: int) -> list[int]:
        with torch.no_grad():
            out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                 max_new_tokens=new, do_sample=False, eos_token_id=eos, pad_token_id=pad)
        return [int(t) for t in out[0, len(ids):].tolist()]

    t0 = time.time()
    result: dict[str, Any] = {"dtype": args.dtype, "max_new_tokens": spec["max_new_tokens"], "prompts": [],
                              "needle": [], "rendered": [], "has_chat_template": bool(tok.chat_template)}
    for text in spec["prompts"]:
        ids = _chat_ids(tok, text)
        gen = greedy(ids, spec["max_new_tokens"])
        result["prompts"].append({"prompt": text, "input_ids": ids, "output_ids": gen,
                                  "output": tok.decode(gen, skip_special_tokens=True)})

    chunks = _heldout_chunks(tok, Path(args.heldout), spec["ppl_chunks"], spec["chunk_len"])
    token_nll: list[list[float]] = []
    nlls: list[float] = []
    with torch.no_grad():
        for chunk in chunks:
            x = torch.tensor([chunk])
            logits = model(x).logits[0, :-1].float()
            per_token = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none")
            token_nll.append([round(v, 5) for v in per_token.tolist()])  # index i scores chunk[i + 1]
            nlls.append(per_token.mean().item())
    result["perplexity"] = {"chunks": chunks, "token_nll": token_nll,
                            "ppl": math.exp(sum(nlls) / len(nlls)) if nlls else None}

    needle = spec.get("needle")
    if needle and spec.get("needle_sizes"):
        filler: list[int] = []
        for chunk in _heldout_chunks(tok, Path(args.heldout), 64, 256):
            filler += chunk
        head = tok.encode(needle["fact"] + "\n\n", add_special_tokens=False)
        for size in spec["needle_sizes"]:
            body = filler[: max(0, size - len(head) - 64)]
            ids = _chat_ids(tok, tok.decode(head + body) + "\n\n" + needle["question"])
            answer = tok.decode(greedy(ids, 16), skip_special_tokens=True)
            result["needle"].append({"target": size, "input_len": len(ids), "input_ids": ids, "output": answer,
                                     "correct": needle["answer"] in answer})

    if tok.chat_template:
        for conversation in spec.get("conversations", []):
            result["rendered"].append(tok.apply_chat_template(conversation, add_generation_prompt=True,
                                                              tokenize=False))
    result["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({"ppl": result["perplexity"]["ppl"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
