"""Minimal, timestamped GenAI load + generate, to locate stalls (load vs prefill vs decode)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import onnxruntime_genai as og

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:8.2f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--reference", help="use the first prompt's token IDs from this reference JSON")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=0, help="KV/search max_length (0 = prompt+tokens)")
    ap.add_argument("--genai-log", action="store_true")
    ap.add_argument("--plugin-qnn", action="store_true",
                    help="register the onnxruntime-qnn 2.x plugin EP library (ORT >= 1.25)")
    args = ap.parse_args()

    if args.plugin_qnn:
        import onnxruntime_qnn

        og.register_execution_provider_library("QNNExecutionProvider", onnxruntime_qnn.get_library_path())
        log(f"registered plugin EP: {onnxruntime_qnn.get_library_path()}")
    if args.genai_log:
        og.set_log_options(enabled=True, model_input_values=False, model_output_shapes=True)
    log(f"genai {getattr(og, '__version__', '?')}, qnn available: {og.is_qnn_available()}")
    log("loading model ...")
    model = og.Model(args.bundle)
    log("model loaded")
    tok = og.Tokenizer(model)
    if args.reference:
        ids = json.loads(Path(args.reference).read_text(encoding="utf-8"))["prompts"][0]["input_ids"]
    else:
        ids = list(tok.encode("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"))
    log(f"prompt tokens: {len(ids)}")
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, top_k=1, max_length=args.max_length or len(ids) + args.tokens)
    gen = og.Generator(model, params)
    log("generator created; appending prompt (prefill) ...")
    gen.append_tokens(ids)
    log("prefill done")
    out = []
    for _ in range(args.tokens):
        if gen.is_done():
            break
        gen.generate_next_token()
        out.append(int(gen.get_next_tokens()[0]))
        log(f"token {len(out)}: {out[-1]} {tok.decode([out[-1]])!r}")
    log(f"text: {tok.decode(out)!r}")
    sys.exit(0)


if __name__ == "__main__":
    main()
