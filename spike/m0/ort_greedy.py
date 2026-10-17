"""Greedy decode with a raw ONNX Runtime session for single-graph GQA decoders.

Handles graphs whose attention mask was replaced by past_seq_len/total_seq_len (Olive's
AttentionMaskToSequenceLengths surgery), which GenAI 0.12.2 cannot feed for a non-pipeline
model. Used to bisect quality across Olive stages on the CPU EP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

_NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64,
       "tensor(int32)": np.int32, "tensor(uint16)": np.uint16, "tensor(uint8)": np.uint8}


def run(model_path: str, ids: list[int], new: int, eos: set[int]) -> list[int]:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}
    kv = [n for n in inputs if n.startswith("past_key_values.")]
    kv_shape = {n: inputs[n].shape for n in kv}
    present = {n.replace("past_key_values.", "present."): n for n in kv}
    out_names = [o.name for o in sess.get_outputs()]
    past = {n: np.zeros((1, kv_shape[n][1], 0, kv_shape[n][3]), dtype=_NP[inputs[n].type]) for n in kv}
    total = len(ids)
    feed_ids = ids
    out: list[int] = []
    for _ in range(new):
        feed = {"input_ids": np.array([feed_ids], dtype=_NP[inputs["input_ids"].type])}
        if "past_seq_len" in inputs:
            feed["past_seq_len"] = np.array([[total - 1]], dtype=np.int32)
            feed["total_seq_len"] = np.array(total, dtype=np.int32)
        if "attention_mask" in inputs:
            feed["attention_mask"] = np.ones((1, total), dtype=np.int64)
        feed.update(past)
        res = dict(zip(out_names, sess.run(out_names, feed)))
        token = int(res["logits"][0, -1].argmax())
        out.append(token)
        if token in eos:
            break
        past = {present_to_past: res[p] for p, present_to_past in present.items()}
        feed_ids = [token]
        total += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.onnx path")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    for p in ref["prompts"][: args.count]:
        out = run(args.model, p["input_ids"], args.tokens, eos)
        print(f"{p['prompt'][:40]!r:44} -> {tok.decode(out, skip_special_tokens=True)[:60]!r}"
              f"   (ref {p['output'][:40]!r})", flush=True)


if __name__ == "__main__":
    main()
