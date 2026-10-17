# Historical Conversion Notes (Not Current Certification)

These are historical observations and example commands, not reproducible current
end-to-end certification. The later artifact reconciliation below identifies a
prebuilt source, not a reproduced supplier build. C0 core tests cover CLI/control
flow only; they do not
revalidate these models, uploads, hardware, or upstream version-specific failures.

## October 16 pinned prebuilt consumer qualification

The current bounded consumer path is now locally exercised, separately from the
historical conversion recipes below: `acquire-prebuilt` of the full pinned
llmware revision, explicit `prepare-prebuilt --profile phi3-mini-4k-qnn`, then
reviewed `import-ollama` into an initially empty isolated store.

All 23 upstream files (1,970,617,736 bytes) were downloaded without ambient
credentials, structurally inspected and independently reverified from the
installed wheel. Source fingerprint:
`6fa2fa387024705b616ac12f9468e6671502e24468420eaa3095d3c617cac653`.
Two independently prepared 30-file derivatives have fingerprint
`b274a6b3443ac58cfe70405c6e5b3c595a981a5b665b95ffb937834c53aabf6d`.
Source/card/header attribution, complete license texts, original config,
source hashes, explicit changes, template and compatibility declaration survive
import; all 30 named blob hashes and the Ollama license layer were verified.

On the retained `abc185a2` ARM64 fork package and verified GenAI QNN 0.12.2 /
ORT QNN 1.24.1 runtime: 69 subtests / 50 exact raw prompts, templated generation,
chat, unload/reload, CPU/NPU switching, ten disconnects / 30 queued exact answers /
ten queued deadlines passed. Seven live official native pins plus separate OEM
RPC and clean child unload were verified. No callback/teardown or ORT native
errors recurred; DSP keys 49/50 remain visible.

Final converter checks: **439 passed / 11 optional skips** without ONNX;
**450 passed / no skips** with ONNX; Ruff/mypy and pip check clean.
This qualifies a local prebuilt journey and deterministic packaging, **not**
rebuilding the compiled weights, exact base-weight lineage, useful long context,
new OS/account, hosted blob pull/run, tool/agent use or a public release.
See [consumer commands and limitations](../README.md#prepared-phi-3-mini-4k-engineering-profile).

## October 15 artifact provenance reconciliation

The original compiled `dsmith111/phi3:mini-qnn` traces to
[`llmware/phi-3-mini-4k-instruct-onnx-qnn`](https://huggingface.co/llmware/phi-3-mini-4k-instruct-onnx-qnn/tree/6fa72cea4103de0bc69792a6655be538a8c4d7e0),
revision `6fa72cea4103de0bc69792a6655be538a8c4d7e0`. All 19 upstream model assets
are accounted for: 11 raw-byte matches (all four context binaries/all four
graphs included), eight differing only by Windows CRLF line endings. The local
twentieth asset is `_publish_manifest.json`. The pinned publisher card declares
Apache-2.0/int4, QAIRT 2.36, ORT 1.22.2 and GenAI 0.9.

The working artifact is therefore a **Phi-3 Mini 4K prebuilt import**, not evidence
of converting Phi-3.5 locally. The older `llmware/phi-3.5-onnx-qnn` has different
context/graph hashes. Shared tokenizers alone do not establish weight origin.
The fork has separately recorded short-prompt QNN qualification on its newer
pinned runtime and a two-option file-mapping config repair; those tests do not
qualify this converter's old production recipe.

`mini-qnn-exported` retains a `npu-model 0.1.0` packing receipt, historical input
`hf:microsoft/Phi-3-mini-4k-instruct` and cached revision
`f39ac1d28e925b323eae81227eaba4464caced4e`. The actual graph has 293 FP16
initializers, external `model.onnx.data` (7,643,004,928 bytes), no compiled
EPContext nodes and no explicit QNN provider in its GenAI config. This establishes
a portable ONNX export, **not a verified end-to-end QNN conversion**. Experimental
`compiled_phi2`/`compiled_phi3` context wrappers are zero-byte files; adjacent
nonempty binaries alone do not prove usable compiled inference.

Both existing public hosted manifests preserve the local named descriptors.
No fresh complete blob pull/run was performed in this reconciliation. Neither
tag contains a license layer; the prebuilt card's Apache declaration and
Microsoft's MIT license are recorded evidence, not permission for unrelated
runtime DLL redistribution. Preserve notices and source/local-change identities
in any new derived package. Historical claims below remain historical.

`npu-model publish` only converts/assembles a local folder. Assembly, Ollama local
installation, generation/native execution, and remote publication are distinct.
The current packer requires compiled context wrappers plus binaries and rejects
raw float/external-data bundles, so historical FP16 packing commands below are
not current working recipes.

## Historical Limitations (Version-Specific, Not Current Runtime Claims)

### ONNX External Data Files (`.onnx.data`)
The historical FP16/FP32 exports below exceeded 2GB and used ONNX external data
files (`model.onnx.data`). The recorded runtime did not materialize those files,
leading to `file_size: The system cannot find the file specified`.
This is not a claim about current Ollama forks. This converter's current NPU
packaging contract rejects raw external-data bundles regardless.

**Historical workaround suggestions (not qualified by C0):**
- Use `--mode prebuilt-ort-genai` with a pre-built model that has already been properly structured
- Use `--precision int4` (requires onnx-ir fix for serialization — track upstream onnx-ir releases)
- Wait for Ollama to add `.onnx.data` support in ORT GenAI model materialization

### INT4 Export
INT4 export was historically reported to fail with ORT GenAI builder v0.12.2 +
onnx-ir v0.2.0 (`producer_version=None`). The upstream cause/fix is not verified
in the lightweight C0 environment.

---

## Historical Observation Matrix

| Model Family | HF Repo / Input | Precision | Ollama Tag | Status |
|---|---|---|---|---|
| Phi-3 (prebuilt) | local prebuilt ORT GenAI folder | prebuilt | `dsmith111/phi3:mini-qnn` | Historical success claim; current/native supplier execution unverified |
| Phi-3 (export FP16) | `hf:microsoft/Phi-3-mini-4k-instruct` | fp16 | `dsmith111/phi3:mini-qnn-exported` | Export works, Ollama run fails (.data) |
| Phi-3 (export INT4) | `hf:microsoft/Phi-3-mini-4k-instruct` | int4 | — | Blocked by onnx-ir bug |
| Llama | local prebuilt | prebuilt | `dsmith111/llama3.2:1b-qnn` | Pending |

---

## Historical attempt: Phi-3 Mini (FP16 export) — Not Deployable

### Command
```powershell
npu-model convert `
  --input hf:microsoft/Phi-3-mini-4k-instruct `
  --out .\out\phi3-fp16 `
  --mode export `
  --precision fp16 `
  --pack-ollama dsmith111/phi3:mini-qnn-exported
```

### Historical reported result (not reverified by C0)
- [x] Export succeeded (builder ran to completion)
- [x] `ollama create` succeeded
- [x] `ollama push` succeeded — https://ollama.com/dsmith111/phi3:mini-qnn-exported
- [ ] `ollama run` — **FAILS** (missing `model.onnx.data` — Ollama doesn't materialize external data files)

### Output files
- `model.onnx` (249 KB — graph only, weights are external)
- `model.onnx.data` (7.6 GB — weights)
- `genai_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `Modelfile`
- `_publish_manifest.json`

### Versions
- npu-model: 0.1.0
- onnxruntime-genai: 0.12.2
- onnxruntime: 1.24.4
- torch: 2.10.0+cpu
- transformers: 5.3.0
- onnx-ir: 0.2.0
- Python: 3.14.3 (ARM64)
- OS: Windows 11 ARM64

---

## Recipe: Phi-3 Mini (prebuilt import)

### Prerequisites
- Pre-exported ORT GenAI folder with `.onnx` + `genai_config.json` + tokenizer

### Command
```powershell
npu-model convert `
  --input C:\models\phi3-mini-onnx-qnn `
  --out .\out\phi3-mini-qnn `
  --backend qnn `
  --target auto `
  --mode prebuilt-ort-genai `
  --pack-ollama dsmith111/phi3:mini-qnn `
  --num-ctx 512 `
  --num-predict 128
```

### Separate local installation and optional remote publication
```powershell
cd .\out\phi3-mini-qnn\ollama_publish
ollama create dsmith111/phi3:mini-qnn -f Modelfile
ollama push dsmith111/phi3:mini-qnn
```

### Expected output files
- [ ] `*.onnx` (at least 1)
- [ ] `genai_config.json`
- [ ] `tokenizer.json` or `tokenizer.model`
- [ ] `Modelfile`
- [ ] `_publish_manifest.json`

### Versions observed
- npu-model: `_____`
- onnxruntime: `_____`
- onnxruntime-genai: `_____`
- Python: `_____`

### Result
- [ ] `ollama create` succeeded
- [ ] `ollama push` succeeded
- [ ] `ollama run` produces output
- TTFT: `_____` ms
- tok/s: `_____`

---

## Recipe: Llama 3.2 1B (prebuilt import)

### Command
```powershell
npu-model convert `
  --input C:\models\llama-3.2-1b-onnx-qnn `
  --out .\out\llama-3.2-1b-qnn `
  --backend qnn `
  --target auto `
  --mode prebuilt-ort-genai `
  --pack-ollama dsmith111/llama3.2:1b-qnn
```

### Versions observed
- npu-model: `_____`
- onnxruntime: `_____`

### Result
- [ ] `ollama run` produces output
- TTFT: `_____` ms
- tok/s: `_____`

---

## Recipe: Phi-3 Mini (full export)

### Prerequisites
- `pip install npu-model[export]`

### Command
```powershell
npu-model convert `
  --input hf:microsoft/Phi-3-mini-4k-instruct `
  --out .\out\phi3-mini-export `
  --backend qnn `
  --target auto `
  --mode export `
  --pack-ollama dsmith111/phi3:mini-qnn-exported
```

### Versions observed
- npu-model: `_____`
- onnxruntime-genai: `_____`
- torch: `_____`
- transformers: `_____`

### Result
- [ ] Export succeeded
- [ ] `ollama run` produces output

---

## Using the publish script

Unlike the assembly-only CLI `publish` command, this historical script invokes
Ollama local installation and, **by default, a real remote upload**. Do not use it
as a core-only smoke test or proof of native execution. `-NoPush` suppresses remote
upload, not local installation.

Example (local installation only; still requires separately validated artifacts):

```powershell
.\scripts\publish_model.ps1 `
  -Input "C:\models\phi3-mini-onnx-qnn" `
  -Name "dsmith111/phi3:mini-qnn" `
  -Mode "prebuilt-ort-genai" `
  -OutDir ".\out\phi3-mini-qnn" `
  -NoPush
```

Add `-NoPush` to skip the push step. Add `-KeepWork` to preserve intermediate files.

---

## How to add a new entry

1. Run `npu-model doctor` to verify your environment
2. Run the conversion command
3. Fill in the template above with:
   - Exact command used
   - Tool + dependency versions (`npu-model doctor` output)
   - Output file list
   - Whether `ollama run` succeeded
   - Performance numbers (TTFT, tok/s) if available
4. Submit as a PR
