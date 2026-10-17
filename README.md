# npu-model

Convert a Hugging Face model into a bundle that runs on this machine's NPU through the
[Ollama NPU fork](https://github.com/dsmith111/ollama-npu), using maintained
[Olive](https://github.com/microsoft/Olive) recipes for the heavy lifting.

> **Status: restructuring (M1).** The original conversion code never produced a working NPU
> model and has been removed. The M0 feasibility spike (`spike/m0/`) showed the new approach
> working end to end on a Snapdragon X Elite laptop; the product commands are being rebuilt
> around it. See `CONVERTER_PLAN.md` in the parent workspace and
> [spike/m0/RESULTS.md](spike/m0/RESULTS.md).

## Target experience

```powershell
npu-model convert Qwen/Qwen2.5-0.5B-Instruct --install
ollama run qwen2.5-0.5b-instruct:npu
```

## Commands available today

| Command | Purpose |
|---|---|
| `npu-model inspect --input <bundle>` | JSON report: file hashes plus structural, declaration and target checks against the fork's consumer contract. Never runs the model. |
| `npu-model install --input <bundle> --name <tag> --ollama-exe <fork> --host http://127.0.0.1:PORT --modelfile <file>` | Plans a local `ollama create` (default); `--execute --expect-fingerprint <plan fingerprint>` performs it. Never downloads or uploads. |

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests
python -m mypy --no-incremental src
```

The CLI itself is lightweight. Conversion toolchains (PyTorch, Olive, ONNX Runtime QNN)
run in separate, pinned role environments that the tool provisions; see `spike/m0/README.md`
for the current manual equivalent.
