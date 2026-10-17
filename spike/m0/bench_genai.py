"""llama-bench-style timing for a GenAI bundle: pp N + tg M (warmup + repetitions).

Mirrors geniex-bench's defaults (512-token prefill from real held-out text, 128 generated
tokens, greedy) so the converted QNN bundle and the GGUF/Hexagon route are comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import onnxruntime_genai as og


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--reference", required=True, help="reference JSON (held-out chunks supply the prompt)")
    ap.add_argument("-p", type=int, default=512)
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("-r", type=int, default=3)
    ap.add_argument("--plugin-qnn", action="store_true")
    args = ap.parse_args()
    if args.plugin_qnn:
        import onnxruntime_qnn

        og.register_execution_provider_library("QNNExecutionProvider", onnxruntime_qnn.get_library_path())

    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    ids: list[int] = []
    for chunk in ref["perplexity"]["chunks"]:
        ids += chunk
    ids = ids[: args.p]
    model = og.Model(args.bundle)
    window = 64
    max_length = -(-args.p // window) * window + args.n + window

    ttft, prefill, decode = [], [], []
    for rep in range(args.r + 1):  # first iteration is warmup
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, top_k=1, max_length=max_length)
        gen = og.Generator(model, params)
        t0 = time.perf_counter()
        gen.append_tokens(ids)
        t_pre = time.perf_counter()
        gen.generate_next_token()
        t_first = time.perf_counter()
        count = 1
        while count < args.n and not gen.is_done():
            gen.generate_next_token()
            count += 1
        t_end = time.perf_counter()
        del gen
        if rep == 0:
            continue
        ttft.append((t_first - t0) * 1000)
        prefill.append(len(ids) / (t_pre - t0))
        decode.append((count - 1) / (t_end - t_first) if count > 1 else 0.0)

    def fmt(xs, unit):
        return f"{statistics.median(xs):.1f} {unit} (min {min(xs):.1f}, max {max(xs):.1f})"

    print(f"pp{len(ids)}+tg{args.n}: TTFT {fmt(ttft, 'ms')} | prefill {fmt(prefill, 'tok/s')} | "
          f"decode {fmt(decode, 'tok/s')}")


if __name__ == "__main__":
    main()
