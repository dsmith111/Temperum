"""Drive a StaticLLM pipeline (embeddings → context/iterator → lm_head) with raw ORT on CPU.

Emulates a shared-buffer KV cache: past/present tensors have length MAX; prefill is one
64-token window (right-padded); GQA gets seqlens_k (past_seq_len) = last valid index and
total_seq_len = MAX. Used to tell broken static graphs apart from runtime (GenAI) behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

WINDOW = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="StaticLLM output directory")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=10)
    ap.add_argument("--max", type=int, default=256)
    ap.add_argument("--left-pad", action="store_true", help="pad the window on the left instead of the right")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    d = Path(args.dir)
    cfg = json.loads((d / "genai_config.json").read_text(encoding="utf-8"))
    files = {}
    for group in cfg["model"]["decoder"]["pipeline"]:
        for name, stage in group.items():
            files[name] = stage
    sess = {n: ort.InferenceSession(str(d / s["filename"]), providers=["CPUExecutionProvider"]) for n, s in files.items()}
    emb_name = next(n for n in files if n.startswith("embed"))
    ctx_name = next(n for n in files if n.startswith("context"))
    it_name = next(n for n in files if n.startswith("iterator"))
    head_name = next(n for n in files if n.startswith("lm_head"))
    ctx_inputs = {i.name: i for i in sess[ctx_name].get_inputs()}
    kv_names = [n for n in ctx_inputs if n.startswith("past_key_values.")]
    hidden_in = next(n for n in ctx_inputs if not n.startswith("past_") and n not in ("past_seq_len", "total_seq_len"))
    np_t = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint16)": np.uint16}
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))

    def run_tx(name, hidden, past, seqlens_k):
        feed = {hidden_in: hidden, "past_seq_len": np.array([[seqlens_k]], dtype=np.int32),
                "total_seq_len": np.array(args.max, dtype=np.int32), **past}
        outs = [o.name for o in sess[name].get_outputs()]
        res = dict(zip(outs, sess[name].run(outs, feed)))
        hid = next(v for k, v in res.items() if not k.startswith("present."))
        new_past = {k.replace("present.", "past_key_values."): v for k, v in res.items() if k.startswith("present.")}
        return hid, new_past

    def logits(hidden_row):
        head_in = sess[head_name].get_inputs()[0].name
        return sess[head_name].run(None, {head_in: hidden_row})[0]

    for p in ref["prompts"][: args.count]:
        ids = p["input_ids"]
        assert len(ids) <= WINDOW, "demo handles one prefill window"
        past = {n: np.zeros((1, ctx_inputs[n].shape[1], args.max, ctx_inputs[n].shape[3]),
                            dtype=np_t[ctx_inputs[n].type]) for n in kv_names}
        pad = [0] * (WINDOW - len(ids))
        window = (pad + ids) if args.left_pad else (ids + pad)
        emb_in = sess[emb_name].get_inputs()[0].name
        hidden = sess[emb_name].run(None, {emb_in: np.array([window], dtype=np.int64)})[0]
        hid, past = run_tx(ctx_name, hidden, past, WINDOW - 1)  # whole window; causal mask hides trailing pads
        last = WINDOW - 1 if args.left_pad else len(ids) - 1
        out = [int(logits(hid[:, last:last + 1]).reshape(-1).argmax())]
        pos = len(ids)
        while len(out) < args.tokens and out[-1] not in eos:
            h = sess[emb_name].run(None, {emb_in: np.array([[out[-1]]], dtype=np.int64)})[0]
            hid, past = run_tx(it_name, h, past, pos)
            out.append(int(logits(hid).reshape(-1).argmax()))
            pos += 1
        print(f"{p['prompt'][:40]!r:44} -> {tok.decode(out, skip_special_tokens=True)!r}   (ref {p['output'][:40]!r})",
              flush=True)


if __name__ == "__main__":
    main()
