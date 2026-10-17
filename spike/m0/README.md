# M0 feasibility spike: HF → QNN on a CPU-only Snapdragon laptop

Throwaway scripts that test the plan in `CONVERTER_PLAN.md` (parent workspace) before any
product code: run Microsoft's maintained olive-recipes QNN recipe for a tiny model with
CPU-only quantization, compile for the Ollama fork's runtime, and verify against the
source model. Results are in [RESULTS.md](RESULTS.md).

Host: Snapdragon X Elite X1E78100, 16 GB RAM, Windows 11 ARM64 (26100), no NVIDIA GPU.
Roots are kept short because Windows long paths are disabled on this machine.

## Environments (Python 3.12 **ARM64**, three roles)

| Role | Path | Key packages |
|---|---|---|
| quantize | `C:\Users\smithdavi\npu-m0\envs\q` | torch 2.14.1+cpu, olive-ai 0.13.0, onnxruntime 1.24.4, onnxruntime-genai 0.12.2, transformers 4.57.3, optimum 2.3.0, psutil |
| compile | `C:\Users\smithdavi\npu-m0\envs\c` | olive-ai 0.13.0, torch 2.14.1+cpu, onnxruntime-qnn 1.24.1 (QNN 2.42.0, same build as the fork's runtime) |
| verify | `C:\Users\smithdavi\npu-m0\envs\v` | onnxruntime-qnn 1.24.1, onnxruntime-genai 0.12.2 (`--no-deps`) |

Setup notes (all hit on this machine):

- The default `python` here is the **x64 build under emulation** (`sysconfig.get_platform()`
  is `win-amd64` even though `platform.machine()` says `ARM64`). Request ARM64 explicitly:
  `uv python install cpython-3.12-windows-aarch64-none`.
- Package downloads must go through the admin-configured pip feed
  (`C:\ProgramData\pip\pip.ini`); direct `files.pythonhosted.org` access fails the TLS
  handshake. `uv pip` fails on this network, so use `uv` only for the interpreter and
  `python -m ensurepip --upgrade` + pip for packages.
- torch for ARM64 lives only on the PyTorch index (Python 3.11–3.13):
  `pip install --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`,
  then install its dependencies from the feed.
- `datasets` cannot be installed (no pyarrow for Windows ARM64), so calibration text is
  fetched as JSON (`fetch_calib.py`) and fed to Olive by `npu_calib.py`.
- `onnxruntime-genai` depends on the CPU `onnxruntime` package; in the verify env install
  it with `--no-deps` so `onnxruntime-qnn` provides the `onnxruntime` module.

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `fetch_model.py` | quantize | Pinned HF snapshot (standard HF auth) + `_source.json` |
| `fetch_calib.py` | quantize | Pinned wikitext rows via the dataset-viewer API + hash |
| `npu_calib.py` | (Olive user_script) | `join` / `line-by-line` calibration without `datasets` |
| `make_config.py` | quantize | Upstream recipe + CPU overlay → Olive config |
| `run_olive.py` | quantize | `olive run` with wall time, peak memory, per-pass timing |
| `reference_hf.py` | quantize | CPU fp32 reference: greedy answers, perplexity, needle prompts |
| `verify_genai.py` | verify | Same token IDs through GenAI + QNN; agreement, perplexity ratio, speed |

`upstream/` holds the unmodified recipe from `microsoft/olive-recipes` (MIT) at the
commit in `upstream/COMMIT`.

## Reproduce (Qwen2.5-0.5B-Instruct)

```bash
R=C:/Users/smithdavi/npu-m0; Q=$R/envs/q/Scripts/python.exe
$Q fetch_model.py --repo Qwen/Qwen2.5-0.5B-Instruct --out $R/models/qwen2.5-0.5b-instruct
$Q fetch_calib.py --out $R/data/wikitext2-train-3000.jsonl
$Q fetch_calib.py --split test --rows 1500 --out $R/data/wikitext2-test-1500.jsonl
$Q make_config.py --recipe upstream/qwen2.5-0.5b-config.json --model $R/models/qwen2.5-0.5b-instruct \
   --corpus $R/data/wikitext2-train-3000.jsonl --compile-env $R/envs/c/Scripts \
   --output-dir $R/out/q05b --cache-dir $R/cache/q05 --out $R/runs/q05-balanced/config.json
OLIVE_DISABLE_TELEMETRY=1 $Q run_olive.py --config $R/runs/q05-balanced/config.json \
   --log $R/runs/q05-balanced/olive.log --cwd $R/runs/q05-balanced
```
