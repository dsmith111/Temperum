"""Render an Olive config from a maintained olive-recipes QNN recipe plus a CPU overlay.

The upstream recipes quantize with GPTQModel on CUDA and load calibration text through
`datasets`. On a CPU-only Windows ARM64 host this overlay:
  * replaces GptqModel with QuaRot + Olive's native, layer-wise Gptq (CPU capable),
  * swaps the HuggingfaceContainer data configs for npu_calib.py (no `datasets`),
  * calibrates activations on the CPU EP with a reduced sample budget,
  * renames ModelBuilder options to the Olive 0.13 names,
  * points the AOT compile system at the env whose QNN matches the target runtime.
`--quality fast` uses round-to-nearest (Rtn) instead of Gptq.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MB_RENAMES = {
    "block_size": "int4_block_size",
    "accuracy_level": "int4_accuracy_level",
    "op_types_to_quantize": "int4_op_types_to_quantize",
}


def _calib_config(name: str, model: str, corpus: str, strategy: str, samples: int, seq_len: int,
                  add_special_tokens: bool) -> dict:
    return {
        "name": name,
        "type": "DataContainer",
        "user_script": str(HERE / "npu_calib.py"),
        "load_dataset_config": {"type": "npu_jsonl_text", "params": {"data_files": corpus}},
        "pre_process_data_config": {
            "type": "npu_text_gen",
            "params": {
                "model_name": model, "strategy": strategy, "max_seq_len": seq_len,
                "max_samples": samples, "add_special_tokens": add_special_tokens,
            },
        },
        "dataloader_config": {"type": "default_dataloader", "params": {"batch_size": 1}},
    }


def render(args: argparse.Namespace) -> dict:
    cfg = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    out = copy.deepcopy(cfg)
    out["input_model"] = {"type": "HfModel", "model_path": args.model}
    out["systems"]["qnn_system"]["python_environment_path"] = args.compile_env
    out["data_configs"] = [
        _calib_config("calib_joined", args.model, args.corpus, "join",
                      args.gptq_samples, args.gptq_seq_len, False),
        _calib_config("calib_act", args.model, args.corpus, "line-by-line",
                      args.act_samples, args.act_seq_len, True),
    ]

    passes: dict = {}
    for key, p in cfg["passes"].items():
        p = copy.deepcopy(p)
        kind = p["type"]
        if kind in ("GptqModel", "Gptq"):
            overrides = {}
            for pattern, spec in (p.get("dynamic") or {}).items():
                if "lm_head" in pattern:
                    overrides["lm_head"] = {
                        "bits": spec.get("bits", 8), "symmetric": spec.get("sym", True),
                        "group_size": spec.get("group_size", -1),
                    }
            common = {"bits": args.bits or p.get("bits", 4), "sym": p.get("sym", True),
                      "group_size": p.get("group_size", -1), "lm_head": p.get("lm_head", False)}
            if overrides:
                common["overrides"] = overrides
            if p.get("rotation") or "q" in cfg["passes"]:
                passes.setdefault("q", {"type": "QuaRot", "rotate_mode": "hadamard"})
            if args.quality == "fast":
                passes[key] = {"type": "Rtn", **common}
            else:
                passes[key] = {"type": "Gptq", **common, "data_config": "calib_joined"}
            continue
        if kind == "QuaRot":
            passes["q"] = p
            continue
        if kind == "ModelBuilder":
            p = {MB_RENAMES.get(k, k): v for k, v in p.items()}
        if kind == "OnnxStaticQuantization":
            p["data_config"] = "calib_act"
            p["calibration_providers"] = ["CPUExecutionProvider"]
        if kind == "EPContextBinaryGenerator":
            p.setdefault("provider_options", {})["soc_model"] = args.soc_model
        passes[key] = p

    # QuaRot must run first (before CaptureSplitInfo), matching the Qwen3 recipes.
    if "q" in passes:
        passes = {"q": passes.pop("q"), **passes}
    out["passes"] = passes
    out["output_dir"] = args.output_dir
    out["cache_dir"] = args.cache_dir
    out["log_severity_level"] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--model", required=True, help="local HF snapshot directory")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--compile-env", required=True, help="directory containing the AOT env python.exe")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--quality", choices=["fast", "balanced"], default="balanced")
    ap.add_argument("--soc-model", default="60")
    ap.add_argument("--bits", type=int, default=0, help="override weight bits (e.g. 8)")
    ap.add_argument("--gptq-samples", type=int, default=32)
    ap.add_argument("--gptq-seq-len", type=int, default=1024)
    ap.add_argument("--act-samples", type=int, default=64)
    ap.add_argument("--act-seq-len", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = render(args)
    Path(args.out).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps({k: v["type"] for k, v in cfg["passes"].items()}, indent=1))


if __name__ == "__main__":
    main()
