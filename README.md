# npu-model

Convert a Hugging Face model to run on this machine's NPU through the
[Ollama NPU fork](https://github.com/dsmith111/ollama-npu). One command converts it, checks
the result on the NPU against the source model, and can install it into the fork:

```powershell
npu-model convert Qwen/Qwen2.5-1.5B-Instruct --install --host http://127.0.0.1:11434
ollama run qwen2.5-1.5b-instruct:npu
```

**Why the NPU:** a model running there leaves the CPU to you. On the CPU alone a model can
saturate a weaker machine, slowing your own work and the model alike; that matters most for
an agent working alongside you. Generating with Qwen2.5-1.5B through the fork, the NPU run
kept the server at 0.1 of a CPU core, against 5.3 cores on the CPU. Power matters mostly for
chat use. The CPU still decodes about twice as fast; the NPU processes prompts faster.

That command:

1. picks the target: on a Snapdragon X, a GGUF for the Hexagon NPU (QNN bundles are the
   alternative, see below);
2. resolves the model to an exact commit and shows a plan: time, RAM, disk, licence and
   any warnings;
3. on first use, sets up its conversion toolchain (~1 GB, about 8 minutes) and fetches
   llama.cpp's converter, pinned to the fork's llama.cpp;
4. downloads the weights, converts them straight to Q8_0 with llama.cpp's converter (about
   300 architectures) and quantizes that with the fork's `llama-quantize` (`--quant Q4_0`,
   the default and the fastest; `Q4_K_M` or `Q8_0` keep more quality);
5. writes the bundle: `model.gguf`, a Modelfile (with an Ollama chat template checked against
   the model's own when the format is recognized, so the fork does not swap in a built-in
   one), the licence files and a provenance manifest;
6. verifies it on the NPU with the fork's `llama-server`: where the layers run, answers
   and long-prompt retrieval against the source model, perplexity against the Q8_0
   conversion, and the chat template;
7. with `--install`, runs `ollama create` on the fork, then loads the model once to check
   that the fork really runs it on the NPU.

Qwen2.5-1.5B converts in under a minute; standard verification, which runs the source model
for reference answers, adds about 8. [docs/models.md](docs/models.md) has nine models from
0.36B to 3.8B with times, quality and speed.

## Install

You need a Snapdragon X laptop with Windows 11, Python 3.10 or newer, and Git.

1. **The Ollama NPU fork, with its Hexagon backend.** It has no published release yet:
   build its portable package from branch `npu/hexagon-spike` with
   `scripts\build_npu_windows.ps1 ... -HexagonSDKRoot <Hexagon SDK>` (see the fork's
   `docs/development.md`).

2. **The NPU libraries and the switch.** The backend needs a set of Microsoft-signed NPU
   libraries from Qualcomm, whose terms restrict their use, so the fork downloads them at
   setup (showing the terms first) instead of shipping them. The backend stays off unless
   the fork's server starts with `OLLAMA_HEXAGON=1`:

   ```powershell
   C:\path\to\fork\ollama.exe npu setup --provider hexagon --accept-qualcomm-terms
   setx OLLAMA_HEXAGON 1          # for new terminals; open one, then start the server
   C:\path\to\fork\ollama.exe serve
   ```

3. **`npu-model`, pointed at the fork once:**

   ```powershell
   git clone https://github.com/dsmith111/ollama-npu-model-converter
   cd ollama-npu-model-converter
   python -m pip install .
   setx NPU_MODEL_OLLAMA_EXE C:\path\to\fork\ollama.exe
   npu-model setup      # in a new terminal: the conversion toolchain, ~1 GB
   npu-model doctor     # checks the fork, its NPU libraries and the switch
   ```

   The CLI itself stays small. The conversion tools go into separate environments that it
   creates, each with its own native ARM64 Python, so the x64 Python that Windows on Arm
   machines often have works too.

`--install` needs the fork's endpoint (`--host` or `OLLAMA_HOST`): `npu-model` never picks
or starts a server itself. If the fork's server runs without the switch, install says the
model runs on the CPU and how to fix it.

## What converts today

- **GGUF (the default):** whatever llama.cpp's converter handles, about 300 architectures;
  verification on the NPU decides whether the result is usable. Nine models are checked so
  far ([docs/models.md](docs/models.md), or `npu-model models`). Models up to about 4 GiB
  (3.8B parameters at Q4_0 with a 4K context) load whole on the NPU; larger ones are split
  with the CPU, which is slower.
- **Gated models** (for example Llama) work with your Hugging Face access: request access
  on the model page, then `hf auth login` or set `HF_TOKEN`. Tokens are never written to
  outputs.
- **Models whose source is too large to run here** (the 3.8B Phi models on 16 GB) are still
  checked against their tokenizer and chat template; the verdict says what was not compared.

## QNN bundles (`--target qnn-x-elite`)

The fork also runs ONNX models compiled for the NPU with Qualcomm's QNN, through ONNX
Runtime GenAI. `npu-model` builds them from Microsoft's maintained
[Olive](https://github.com/microsoft/Olive) recipes: it picks the recipe, adapts it to the
machine, quantizes, exports and compiles the NPU context binaries, and writes a bundle with
a compatibility declaration (`ollama_npu.json`). Use it where the GGUF route cannot run, for
example without accepting Qualcomm's terms for the Hexagon libraries:

```powershell
C:\path\to\fork\ollama.exe npu setup --provider qnn   # the fork's QNN runtime (docs/npu_qnn.md)
npu-model setup --target qnn-x-elite                  # adds the compile and verify toolchains
npu-model convert Qwen/Qwen2.5-0.5B-Instruct --target qnn-x-elite --install --host http://127.0.0.1:11434
ollama run qwen2.5-0.5b-instruct:npu-qnn
```

`NPU_MODEL_TARGET=qnn-x-elite` makes it the default. It is slower all round: Qwen2.5-0.5B
took 22 minutes on the CPU (plus about 18 for the three toolchains) and Qwen2.5-1.5B 55,
against under a minute for its GGUF; through the fork the 1.5B bundle decoded at 20.2 tok/s
and prefilled 772 tok/s, against 28.7 and 1,229 for its GGUF on the NPU. Its quality is high
(8-bit weights: 1.02–1.15x the source model's perplexity). Quantizing on the CPU needs RAM
of about 8x the parameter count in bytes, which limits 16 GB laptops to about 1.5B
parameters; larger models need an x64 machine with an NVIDIA GPU, a path that is
implemented but not yet tested.

- **Maintained recipes:** Qwen2.5 (0.5B–7B, Coder), Qwen3 (0.6B–8B), DeepSeek-R1-Distill-Qwen,
  Llama 3.1/3.2, Phi-3/3.5/4 (vendored from `microsoft/olive-recipes`); other models of
  those families (`model_type` qwen2, qwen3, llama, phi3) use their family's recipe.
- **Quality modes:** `standard` (default on the CPU) uses rotated 8-bit round-to-nearest
  weights; `recipe` (default on CUDA) uses the recipe's own GPTQ settings, usually 4-bit.

## Commands

| Command | Purpose |
|---|---|
| `npu-model convert MODEL` | Plan, convert and verify. `--install` installs into the fork; `--plan` stops after the plan; `--yes` skips the prompt; `--quant Q4_0\|Q4_K_M\|Q8_0`, `--context N`, `--verify quick\|standard\|full\|none`, `--out DIR`, `--target qnn-x-elite` (QNN only: `--quality`, `--device`, `--template FILE --stop S`) |
| `npu-model setup` | Create or check (`--check`) what conversions need; `--target qnn-x-elite` adds the QNN toolchains |
| `npu-model doctor` | Target, the fork's NPU runtime and switch, memory, disk, toolchain health |
| `npu-model models` | Conversions verified on hardware, the GGUF route, QNN recipes and families |
| `npu-model verify BUNDLE` | Re-run verification on this machine's NPU (with the fork's current runtime) |
| `npu-model inspect --input BUNDLE` | Structural and declaration report for a QNN bundle (never runs the model) |
| `npu-model install --input BUNDLE --name TAG` | Install a bundle into the fork: GGUF with `--execute`; QNN as a reviewed plan, then `--execute --expect-fingerprint FP` |

Settings: `NPU_MODEL_OLLAMA_EXE` (the fork), `NPU_MODEL_TARGET` (default target),
`OLLAMA_HOST` (the fork's endpoint for installs), `HF_TOKEN` (gated models) and
`NPU_MODEL_HOME`. Everything lives under `%LOCALAPPDATA%\npu-model` (toolchains, source
snapshots, jobs, bundles); set `NPU_MODEL_HOME` to a short path if Windows long paths are
disabled and your profile path is long. A conversion is a resumable job: rerunning the same
command continues where it stopped, and `jobs/<id>/job.json` records every stage.

## How it works

- `gguf.py`, `gguf_verify.py`, `toolchains/llama_cpp.py`: the GGUF target (llama.cpp's
  converter fetched at a pinned commit and checked file by file; the fork's `llama-quantize`,
  `llama-server` and `llama-perplexity`).
- `templates.py`: recognizes the model's chat format and writes an Ollama template that
  verification compares with the model's own rendering.
- `install/`: local `ollama create` through a running fork, then a placement check.
- QNN: `recipes/` (vendored olive-recipes at a pinned commit and a renderer that adapts them),
  `toolchains/` (lock files per runtime, role and platform; a uv-managed native Python;
  packages through your pip configuration, torch from the PyTorch index), and
  `olive_run.py`, `assemble.py`, `verify.py`, `convert.py`, which enforce rules learned in the
  M0 spike (a pad token that never appears in prompts, retries for an intermittent QNN
  compiler crash, graph splits that keep each compiled context near 1 GB).

Background and measurements: [docs/models.md](docs/models.md),
[spike/m0/RESULTS.md](spike/m0/RESULTS.md).

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
