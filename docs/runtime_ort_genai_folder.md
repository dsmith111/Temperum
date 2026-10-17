# Runtime Format: ORT GenAI Folder

The `ort-genai-folder` runtime format assembles an ORT GenAI directory.
Assembly alone does not prove that its graphs/configuration can run.

## Bundle layout

```
ort_genai/
  genai_config.json          # (if provided by adapter)
  backend_metadata.json      # backend info for debugging
  onnx/
    *.onnx                   # ONNX graph files
  tokenizer/
    tokenizer.json           # tokenizer assets
    tokenizer_config.json
    special_tokens_map.json
    ...
  backend_artifacts/
    *.bin                    # context binaries, compiled artifacts
    *.json                   # backend-specific configs
```

## Validation

`validate_layout()` checks:

1. `onnx/` directory exists
2. At least one `.onnx` file is present in `onnx/`

This is layout validation only; placeholder or unreadable ONNX files can pass it.
Use the explicit validation levels below for stronger checks.

## Assembly process

1. Copy ONNX graphs from `BackendPreparedBundle.graphs` into `onnx/`
2. Copy tokenizer assets from `GraphBundle.tokenizer_dir` into `tokenizer/`
3. Copy backend artifacts from `BackendPreparedBundle.artifacts_dir` into `backend_artifacts/`
4. If `genai_config.json` exists in artifacts, promote it to bundle root
5. Write `backend_metadata.json` at bundle root
6. Run `validate_layout()` to confirm the bundle is well-formed

---

## Ollama Packaging Contract v1

### Overview

`npu-model pack-ollama` produces a flat local input directory for a compatible
Ollama fork. `publish` is a compatibility name for conversion plus this assembly,
**not** an install or remote upload. Output names `<publish_dir>` and
`_publish_manifest.json` are retained for compatibility.

Local installation with `ollama create <name> -f Modelfile`, local generation,
native execution validation, and any remote publication are separate operations.
No native-QNN supplier support is inferred from filenames, mocks, or assembly.

### Hard requirements (fail if missing)

| Requirement | Rule |
|---|---|
| ONNX model files | At least 1 `*_ctx.onnx` context wrapper; no non-context ONNX |
| QNN context binaries | At least 1 `*.bin` file |
| GenAI config | `genai_config.json` present |
| Tokenizer | `tokenizer.model` **or** `tokenizer.json` present |
| Raw external weights | No `*.onnx.data` files |

These are filename/layout requirements, not verification of graph contents or
context-binary compatibility with a supplier/runtime.

### File pickup rules (allowlist)

Only these files are copied into the publish directory:

| Pattern | Reason |
|---|---|
| `*.onnx` | Primary model graphs |
| `*.bin` | Context binaries / weights |
| `*.json` | Configs (genai_config, tokenizer_config, etc.) |
| `tokenizer.model` | SentencePiece tokenizer (exact name match) |
| `*.jinja` | Chat templates |

### Deny list (never included)

These extensions cause Ollama to mis-detect the model format and are **excluded**:

- `*.gguf`
- `*.safetensors`
- `*.lib`
- `*.pdb`

### Modelfile

```
FROM .

# Safe defaults for ORT GenAI KV cache
PARAMETER num_ctx 512
PARAMETER num_predict 128
```

`num_ctx` and `num_predict` are configurable via `--num-ctx` and `--num-predict`.

### Output structure

```
<publish_dir>/
  Modelfile
  _publish_manifest.json     # file hashes, tool version, params
  embeddings.onnx            # (example)
  decoder.onnx               # (example)
  genai_config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  weights.bin                # (example)
  chat_template.jinja        # (if present)
```

All files are placed **flat at the root** — no subdirectories. This matches
Ollama's expectation when using `FROM .`.

### Atomic staging

The packer uses atomic staging to prevent empty output on failure:

1. Build into `<out>.staging.<pid>`
2. Validate the staged directory passes hard requirements
3. Remove existing destination (if any)
4. Rename staging → destination

### Validation

```powershell
npu-model validate --input <publish_dir> --as ollama-ortgenai
```

- **Errors** (exit code 2): missing context ONNX/binary/config/tokenizer,
  non-context ONNX, raw `.onnx.data`, or denylisted files.
- **Warnings** (exit code 0): missing `chat_template.jinja` or optional configs.

### Warnings (soft, non-blocking)

| Warning | Reason |
|---|---|
| No `chat_template.jinja` | Chat mode may not work |
| Missing `config.json` | Optional HF config |
| Missing `generation_config.json` | Optional generation config |
| Missing `special_tokens_map.json` | Optional tokenizer metadata |

## Validation levels

`validate --input <directory>` defaults to basic nested bundle layout checks.
`--bundle` remains a deprecated alias for `--input`; conflicting paths and unknown
`--as` modes are errors. Legacy `--backend`/`--target` options do not select a
validation device; runtime generation uses the bundle's GenAI configuration.

| Mode | Evidence on exit 0 | Does not establish |
|---|---|---|
| omitted | Basic bundle layout | Valid ONNX, generation, NPU execution |
| `ollama-ortgenai` | Packaging filename checks | Valid graph/config contents or execution |
| `genai-config` | Selected config structure/numeric sanity checks | Complete schema/reference validation, generation or provider use |
| `runtime-load` | GenAI model loads and generates new tokens | QNN/native supplier execution |
| `strict-npu` (`npu-ready` alias) | Required configuration/input-shape checks and QNN-retained session loads with CPU fallback disabled | Generation, partition/profiling evidence, native-QNN supplier support |

Runtime validation requires an importable, compatible `onnxruntime-genai` API and
fails with exit 2 when unavailable, incompatible, or no tokens are generated.
`--max-tokens` is a positive **new-token** budget, added to encoded prompt length.
Strict preflight requires importable ONNX and ORT plus `QNNExecutionProvider`.
Missing checks are reported as SKIP and leave the overall result incomplete
(exit 2), never certified. Malformed/unreadable graphs/configs fail checks.
Provider registration alone is insufficient: the created session must retain
QNN after ORT's possible provider fallback. Successful loads are still load-only
evidence, not a substitute for running a model on the intended hardware.
