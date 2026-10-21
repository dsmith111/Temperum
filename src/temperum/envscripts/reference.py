"""Source-model reference for verification, run in the quantize environment.

Records token IDs so the NPU check feeds identical inputs (differences then come from the
converted model, not tokenization). Also renders the model's own chat template for a few
conversations so the converter can compare its generated Ollama template. With
--tokenizer-only (a model too large to run here) it records only the inputs and renderings.

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


def _progress(message: str) -> None:
    print(f"@@progress reference: {message}", flush=True)  # the converter shows these lines


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


def _float32_compute(model: Any) -> None:
    """Keep the large weights in 16-bit storage but compute in float32.

    PyTorch's 16-bit CPU matmuls can be about 100x slower than float32 (Windows on Arm), so a
    16-bit reference would take hours. Upcasting each weight as it is used gives float32
    results at half the memory: identical to a float32 load for a 16-bit checkpoint.
    """
    import torch.nn.functional as F

    def linear(module: Any) -> Any:
        return lambda x: F.linear(x.float(), module.weight.float(),
                                  None if module.bias is None else module.bias.float())

    def embedding(module: Any) -> Any:
        return lambda ids: F.embedding(ids, module.weight, module.padding_idx).float()

    for module in model.modules():
        if type(module) is torch.nn.Linear:
            module.forward = linear(module)
        elif type(module) is torch.nn.Embedding:
            module.forward = embedding(module)
        elif any(p.ndim >= 2 for p in module.parameters(recurse=False)):
            module.float()  # an unfamiliar layer with its own weights: plain float32
    for param in model.parameters():
        if param.ndim < 2:
            param.data = param.data.float()  # norms and biases are small


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
    ap.add_argument("--heldout", help="held-out text for perplexity and long prompts (not needed for renderings)")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="record inputs and chat renderings without running the model (too large for this machine)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model: Any = None
    if not args.tokenizer_only:
        _progress(f"loading the source model ({args.dtype})")
        if args.dtype == "float32":
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        else:  # 16-bit storage (the checkpoint's own 16-bit type when it has one), float32 compute
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
            if model.dtype not in (torch.bfloat16, torch.float16):
                model.to(torch.bfloat16)
            _float32_compute(model)
        model.eval()
        eos = model.generation_config.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos)

    def greedy(ids: list[int], new: int) -> list[int]:
        with torch.no_grad():
            out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                 max_new_tokens=new, do_sample=False, eos_token_id=eos, pad_token_id=pad)
        return [int(t) for t in out[0, len(ids):].tolist()]

    t0 = time.time()
    result: dict[str, Any] = {"dtype": None if model is None else args.dtype, "tokenizer_only": model is None,
                              "max_new_tokens": spec["max_new_tokens"], "prompts": [], "needle": [], "rendered": [],
                              "has_chat_template": bool(tok.chat_template)}
    for index, text in enumerate(spec["prompts"], 1):
        ids = _chat_ids(tok, text)
        row: dict[str, Any] = {"prompt": text, "input_ids": ids}
        if model is not None:
            _progress(f"prompt {index}/{len(spec['prompts'])}")
            gen = greedy(ids, spec["max_new_tokens"])
            row.update({"output_ids": gen, "output": tok.decode(gen, skip_special_tokens=True)})
        result["prompts"].append(row)

    chunks = (_heldout_chunks(tok, Path(args.heldout), spec["ppl_chunks"], spec["chunk_len"])
              if model is not None and args.heldout else [])
    token_nll: list[list[float]] = []
    nlls: list[float] = []
    with torch.no_grad():
        for index, chunk in enumerate(chunks, 1):
            _progress(f"perplexity chunk {index}/{len(chunks)}")
            x = torch.tensor([chunk])
            logits = model(x).logits[0, :-1].float()
            per_token = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none")
            token_nll.append([round(v, 5) for v in per_token.tolist()])  # index i scores chunk[i + 1]
            nlls.append(per_token.mean().item())
    result["perplexity"] = {"chunks": chunks, "token_nll": token_nll,
                            "ppl": math.exp(sum(nlls) / len(nlls)) if nlls else None}

    needle = spec.get("needle")
    if needle and spec.get("needle_sizes") and args.heldout:
        filler: list[int] = []
        for chunk in _heldout_chunks(tok, Path(args.heldout), 64, 256):
            filler += chunk
        head = tok.encode(needle["fact"] + "\n\n", add_special_tokens=False)
        for size in spec["needle_sizes"]:
            body = filler[: max(0, size - len(head) - 64)]
            ids = _chat_ids(tok, tok.decode(head + body) + "\n\n" + needle["question"])
            row = {"target": size, "input_len": len(ids), "input_ids": ids}
            if model is not None:
                _progress(f"long prompt of about {size} tokens")
                answer = tok.decode(greedy(ids, 16), skip_special_tokens=True)
                row.update({"output": answer, "correct": needle["answer"] in answer})
            result["needle"].append(row)

    result["rendered_errors"] = []
    if tok.chat_template:
        for conversation in spec.get("conversations", []):
            try:
                result["rendered"].append(tok.apply_chat_template(conversation, add_generation_prompt=True,
                                                                  tokenize=False))
            except Exception as exc:  # noqa: BLE001 - a template that refuses a test conversation (K2-Horizon
                # wants a thinking field on assistant turns) must not lose the answers and perplexity above
                result["rendered"].append(None)
                result["rendered_errors"].append(f"{type(exc).__name__}: {exc}"[:300])
    result["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({"ppl": result["perplexity"]["ppl"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
