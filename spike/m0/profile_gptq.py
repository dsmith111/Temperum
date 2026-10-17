"""Profile Olive's native Gptq pass on CPU with a tiny calibration budget.

Usage: python profile_gptq.py --model <hf dir> --corpus <jsonl> --samples 4 --seq-len 512
Writes gptq.prof (cProfile) and prints the top hotspots, per-layer timing and dtypes.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2, help="truncate model to N layers for speed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Truncated copy so the experiment takes minutes, not hours.
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.num_hidden_layers = args.layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[: args.layers]
    small = AutoModelForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.float32)
    small_dir = out / "small"
    small.save_pretrained(small_dir)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        src = Path(args.model, name)
        if src.exists():
            (small_dir / name).write_bytes(src.read_bytes())
    print("weights dtype:", next(small.parameters()).dtype, "threads:", torch.get_num_threads())

    config = {
        "input_model": {"type": "HfModel", "model_path": str(small_dir)},
        "data_configs": [{
            "name": "calib", "type": "DataContainer", "user_script": str(HERE / "npu_calib.py"),
            "load_dataset_config": {"type": "npu_jsonl_text", "params": {"data_files": args.corpus}},
            "pre_process_data_config": {"type": "npu_text_gen", "params": {
                "model_name": str(small_dir), "strategy": "join", "max_seq_len": args.seq_len,
                "max_samples": args.samples}},
            "dataloader_config": {"type": "default_dataloader", "params": {"batch_size": 1}},
        }],
        "passes": {"g": {"type": "Gptq", "bits": 4, "sym": True, "group_size": -1,
                         "lm_head": False, "data_config": "calib"}},
        "output_dir": str(out / "olive_out"), "cache_dir": tempfile.mkdtemp(prefix="gptqprof_"),
        "log_severity_level": 1, "no_artifacts": True,
    }
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    from olive.workflows import run as olive_run

    t0 = time.perf_counter()
    prof = cProfile.Profile()
    prof.enable()
    olive_run(str(cfg_path))
    prof.disable()
    total = time.perf_counter() - t0
    prof.dump_stats(str(out / "gptq.prof"))
    print(f"total {total:.1f}s for {args.layers} layers, {args.samples}x{args.seq_len} tokens")
    stats = pstats.Stats(prof, stream=sys.stdout).sort_stats("tottime")
    stats.print_stats(18)


if __name__ == "__main__":
    main()
