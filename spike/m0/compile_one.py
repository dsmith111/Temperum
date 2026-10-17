"""Compile one ONNX graph to a QNN EPContext model (no weight sharing), mirroring Olive's options.

Isolates EPContextBinaryGenerator failures: per-graph compile vs shared-context serialization.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import onnxruntime as ort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output *_ctx.onnx path")
    ap.add_argument("--soc-model", default="60")
    ap.add_argument("--share", choices=["none", "first", "last"], default="none",
                    help="weight-sharing role: first/middle graphs use 'first', the final graph 'last'")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_embed_mode", "0")
    so.add_session_config_entry("ep.context_file_path", str(out))
    if args.share != "none":
        so.add_session_config_entry("ep.share_ep_contexts", "1")
        if args.share == "last":
            so.add_session_config_entry("ep.stop_share_ep_contexts", "1")
    provider = {
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": "3",
        "soc_model": args.soc_model,
    }
    if args.share != "none":
        provider["enable_htp_weight_sharing"] = "1"
    t0 = time.perf_counter()
    sess = ort.InferenceSession(args.model, so, providers=["QNNExecutionProvider"], provider_options=[provider])
    print(f"compiled in {time.perf_counter() - t0:.1f}s; providers={sess.get_providers()}", flush=True)
    del sess
    print("session released", flush=True)
    for f in sorted(out.parent.glob(out.stem + "*")):
        print(f"  {f.name} {f.stat().st_size}")


if __name__ == "__main__":
    main()
