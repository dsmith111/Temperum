"""Greedy-decode the first few reference prompts with any GenAI-loadable intermediate model.

Used to bisect quality loss across Olive passes (e.g. ModelBuilder int4 on CPU vs final QNN bundle).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime_genai as og


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16)
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    model = og.Model(args.model)
    tok = og.Tokenizer(model)
    for p in ref["prompts"][: args.count]:
        ids = p["input_ids"]
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, top_k=1, max_length=len(ids) + args.tokens)
        gen = og.Generator(model, params)
        gen.append_tokens(ids)
        out = []
        while not gen.is_done() and len(out) < args.tokens:
            gen.generate_next_token()
            out.append(int(gen.get_next_tokens()[0]))
        print(f"{p['prompt'][:40]!r:44} -> {tok.decode(out)[:60]!r}   (ref {p['output'][:40]!r})", flush=True)
        del gen


if __name__ == "__main__":
    main()
