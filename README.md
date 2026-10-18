# npu-model

Convert a Hugging Face model into a bundle that runs on this machine's NPU through the
[Ollama NPU fork](https://github.com/dsmith111/ollama-npu). Microsoft's maintained
[Olive](https://github.com/microsoft/Olive) recipes do the heavy lifting; `npu-model`
picks the recipe, adapts it to your machine, runs it in managed toolchains, assembles a
bundle the fork accepts, and checks the result on the NPU against the source model.

```powershell
npu-model convert Qwen/Qwen2.5-0.5B-Instruct
```

That one command:

1. detects the NPU target (Snapdragon X Elite → QNN, HTP v73) and the quantization
   device (CUDA if an NVIDIA GPU is present, otherwise CPU);
2. resolves the model to an exact commit and shows a plan: recipe, precision, time, RAM,
   disk, licence and any warnings;
3. on first use, sets up three isolated toolchains (~2.4 GB): quantize (PyTorch, Olive,
   ONNX Runtime), compile (Olive + `onnxruntime-qnn` built against the fork's QNN SDK)
   and verify (the fork's ONNX Runtime GenAI);
4. downloads the weights, quantizes, exports and compiles the NPU context binaries;
5. writes the bundle with a compatibility declaration (`ollama_npu.json`), an Ollama
   Modelfile with a chat template derived from the model's own, its licence files and
   a provenance manifest;
6. verifies it: loads it on the NPU, generates, and compares perplexity, answers and
   long-prompt retrieval with the original model.

Install into a running fork with `--install` (or `npu-model install` later). The tool
never picks or starts a server itself, so name the fork executable and its endpoint:

```powershell
npu-model convert Qwen/Qwen2.5-0.5B-Instruct --install `
  --ollama-exe C:\path\to\fork\ollama.exe --host http://127.0.0.1:11434
ollama run qwen2.5-0.5b-instruct:npu
```

On a Snapdragon X Elite laptop (X1E78100, 16 GB) that conversion took 22 minutes on the
CPU, covering model download, quantization, compilation and verification. The one-time
toolchain setup adds about 18 minutes. The result measured 1.04× the source model's
perplexity, and the fork loaded it as `declared` and served it on the NPU.

## What converts today

`npu-model models` lists everything. In short:

- **Maintained recipes**: Qwen2.5 (0.5B–7B, Coder), Qwen3 (0.6B–8B), DeepSeek-R1-Distill-Qwen,
  Llama 3.1/3.2, Phi-3/3.5/4 (vendored from `microsoft/olive-recipes`).
- **Other models of the same families** (`model_type` qwen2, qwen3, llama, phi3) use their
  family's recipe; verification decides whether the result is usable.
- **Size**: on a CPU the conversion needs RAM of roughly 8 bytes per parameter (about 5 GB
  for a 0.5B model), so 16 GB laptops are limited to models of about 1.5B parameters.
  Larger models need an x64 machine with an NVIDIA GPU (`--target qnn-x-elite`), a path
  that is implemented but not yet tested.

Quality modes: `standard` (default on CPU) uses rotated 8-bit round-to-nearest weights,
which measured 1.04–1.15× the source model's perplexity; `recipe` (default on CUDA) uses
the maintained recipe's own GPTQ settings, usually 4-bit.

Gated models (for example Llama) work with your Hugging Face access: request access on the
model page, then `hf auth login` or set `HF_TOKEN`. Tokens are never written to outputs.

## Commands

| Command | Purpose |
|---|---|
| `npu-model convert MODEL` | Plan, convert, assemble and verify. `--plan` stops after the plan; `--yes` skips the prompt; `--install` installs into the fork; `--context N`, `--verify quick\|standard\|full\|none`, `--out DIR`, `--template FILE --stop S` |
| `npu-model setup` | Create or check (`--check`) the toolchains |
| `npu-model doctor` | Target, NPU, quantization device, memory, disk, toolchain health |
| `npu-model models` | Verified conversions, maintained recipes, supported families |
| `npu-model verify BUNDLE` | Re-run verification on this machine's NPU |
| `npu-model inspect --input BUNDLE` | Structural and declaration report (never runs the model) |
| `npu-model install --input BUNDLE --name TAG --ollama-exe FORK` | Plan, then `--execute --expect-fingerprint FP` a local `ollama create` |

Everything lives under `%LOCALAPPDATA%\npu-model` (toolchains, source snapshots, jobs,
bundles); set `NPU_MODEL_HOME` to a short path if Windows long paths are disabled and your
profile path is long. A conversion is a resumable job: rerunning the same command
continues where it stopped, and `jobs/<id>/job.json` records every stage.

## How it works

- `recipes/`: the vendored olive-recipes (pinned commit, hashes) and a renderer that adapts
  them (CPU overlay, target SoC, calibration loader without `datasets`, split sizing).
- `toolchains/`: lock files per runtime, role and platform; a uv-managed native Python
  (the default Python on Windows on Arm is often an emulated x64 build); packages through
  your pip configuration, torch from the PyTorch index.
- `olive_run.py`, `assemble.py`, `verify.py`, `convert.py`: the pipeline. Rules learned in
  the M0 spike are enforced there, such as a pad token that never appears in prompts and
  automatic retries for an intermittent QNN compiler crash.

Background and measurements: [spike/m0/RESULTS.md](spike/m0/RESULTS.md).

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests scripts
python -m mypy --no-incremental src
```

Maintainer scripts: `scripts/vendor_recipes.py --commit SHA` refreshes the recipes;
`scripts/freeze_lock.py` turns a working environment into a toolchain lock file.
