# npu-model (converter tool)

This repo is a plugin-based model conversion pipeline.

## What it does
- Takes an input model (local directory or HuggingFace repo)
- Auto-selects a model adapter (Phi-3 / Llama to start)
- Can export through the optional ORT GenAI builder, or import prepared ONNX files
- Runs the selected quantizer/backend (QNN tooling is optional and platform-dependent)
- Assembles an ORT GenAI folder bundle
- Writes a manifest.json with hashes + provenance

## Install (dev)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Quick commands

```powershell
npu-model list-backends
npu-model list-adapters
npu-model list-runtime-formats

npu-model explain --input C:\models\phi3-mini-onnx-qnn --mode prebuilt-ort-genai --backend qnn --target auto --runtime ort-genai-folder

npu-model convert --input C:\models\phi3-mini-onnx-qnn --mode prebuilt-ort-genai --backend qnn --target auto --runtime ort-genai-folder --out .\out\phi3-mini-qnn
```

## Local prebuilt inspection and explicit install (C1/C2/C4)

For an **existing consumer-ready folder**, bypass conversion and its cache:

```powershell
python -m pip install -e ".[prebuilt]"
npu-model inspect-prebuilt --input C:\models\engineering-prebuilt
npu-model import-ollama --input C:\models\engineering-prebuilt --name local/engineering:test --modelfile C:\models\engineering-prebuilt\Modelfile --ollama-exe C:\fork\ollama.exe --host http://127.0.0.1:18765
# Review the JSON plan; copy its top-level fingerprint and repeat the same options:
npu-model import-ollama --input C:\models\engineering-prebuilt --name local/engineering:test --modelfile C:\models\engineering-prebuilt\Modelfile --ollama-exe C:\fork\ollama.exe --host http://127.0.0.1:18765 --execute --expect-fingerprint REVIEWED_PLAN_FINGERPRINT --work-dir . --timeout 300
```

Inspection always emits JSON, including errors (exit 2), every asset's size/SHA256,
a full-folder content fingerprint, and separate structural, declaration, target,
and native-evidence levels. It reads `genai_config.json` decoder/pipeline references,
CPU ONNX external tensors and QNN EPContext cache references without loading weights.
Structural ONNX checking requires the optional **`onnx`** package in the selected
Python environment. It is not installed automatically; missing support fails
explicitly with `MISSING_ONNX`, while hashes/declarations are still reported.
Use an existing ONNX-capable environment or install the small `.[prebuilt]` extra;
this route does not need Torch, Olive, ORT or a new compiler.

Optional `ollama_npu.json` is a schema-1 **publisher declaration**, preserved byte
for byte, not certification. It must agree with every config stage/provider and
context length. The bounded consumer comparison is Windows ARM64 QNN, GenAI
0.12.2 / ORT 1.24.1; `--processor-name "exact Windows processor name"` compares a
caller-supplied target, never probes hardware. CPU embedding/head + QNN context
pipelines are allowed. Missing declaration leaves target compatibility unverified.
No producer, precision, source, license or role tokens are guessed.

Import defaults to a **read-only plan** (no server contact). `--execute` requires
the reviewed plan fingerprint, an already-running compatible local fork, and a
caller-supplied known-correct **Ollama Go TEMPLATE** in a Modelfile whose single
local `FROM` equals the input folder. Bundle Jinja is not an Ollama template.
Template semantics remain the caller's responsibility. Only `FROM` is rebound
to a fully rehashed/revalidated local snapshot; parameters, license/system/template
text and all bundle assets/nested paths remain unchanged. Existing tags require
a new reviewed plan with `--overwrite`. CLI return code and endpoint tag presence
are checked; generation and native execution are **not** tested.

Use explicit `--ollama-exe` / `--host` or existing `NPU_MODEL_OLLAMA_EXE` /
`OLLAMA_HOST`; there are no executable/PATH/port defaults. Numeric HTTP loopback
only, no proxy/auth inheritance or redirects. Use a standalone retained fork,
not an app installation (adjacent app executables are rejected to block implicit
app startup). No download, server startup, remote upload or "published" claim occurs.
Ollama create may transfer local artifact blobs over the selected loopback endpoint.
Staging uses an existing `--work-dir` outside input, needs space for a full copy,
and is removed after success/failure. Tag writers must be serialized externally;
Ollama create has no atomic compare-and-create tag operation.

Bounds: 4096 folder entries, depth 32, 512 GiB total assets, 4 MiB config JSON,
64 KiB declaration, 64 MiB tokenizer JSON, 256 MiB per ONNX metadata graph,
100000 graph nodes/initializers, 64 pipeline stages, 1 MiB Modelfile. References
must be canonical relative portable paths; symlinks/junctions, mixed native/model
formats, missing/empty referenced assets, duplicate JSON/stages and contradictory
declarations fail. Local functions/training or non-decoder graphs are outside this
route. Archives, recipes and general resumable conversion jobs remain future gates.

## Public HF prebuilt acquisition (bounded, Windows)

`acquire-prebuilt` is separate from legacy `convert --input hf:...`. It acquires
an already-prepared folder, never exports, compiles, installs or runs a model:

```powershell
npu-model acquire-prebuilt --repo public-owner/prebuilt-repo --revision main --subfolder prepared --out C:\models\new-prebuilt --work-dir C:\acquisition-work
# Review JSON inventory, immutable commit, destination, target and policy; obtain rights:
npu-model acquire-prebuilt --repo public-owner/prebuilt-repo --revision main --subfolder prepared --out C:\models\new-prebuilt --work-dir C:\acquisition-work --execute --expect-fingerprint REVIEWED_PLAN_HASH --accept-license
# After an orderly interruption, repeat every original option plus the printed receipt:
npu-model acquire-prebuilt --repo public-owner/prebuilt-repo --revision main --subfolder prepared --out C:\models\new-prebuilt --work-dir C:\acquisition-work --execute --expect-fingerprint REVIEWED_PLAN_HASH --accept-license --resume C:\acquisition-work\hf-acquire-SESSION_ID\receipt.json
```

Both parents must already exist; the work parent must be explicitly owned,
on the destination volume, and excluded from concurrent writers. Planning reads
public metadata only and writes nothing. Branches/tags resolve to a full immutable
commit before any data transfer; execution rechecks that commit's inventory.
If a branch moves between plan and a fresh execution, review a new plan (or use
the reviewed commit explicitly in both commands). Resume keeps the original
revision option but resolves **only the receipt's pinned commit**.

Only public, explicitly ungated model repositories at `https://huggingface.co`
are supported. No token/login/netrc, ambient HF endpoint, proxy, SDK cache, Xet
client, telemetry, automatic retries or remote code is used. Auth failures and
repository redirects fail explicitly; data redirects are HTTPS-only on a fixed
HF/CDN host allowlist. Other storage hosts require a separately reviewed change,
not a fallback. The existing general HF conversion source is unchanged.

Selection is the **entire root or subfolder**, not guessed file patterns.
All selected files must be in the bounded prebuilt subset: ONNX/data/bin/QNN,
JSON/tokenizer/model, text/Markdown/rst/Jinja assets, Modelfile and .gitattributes.
Bounded UTF-8 Python source (`.py`, at most 4 MiB, no NUL) is preserved as
**inert bytes**, including copyright headers: it is never imported, compiled or
executed. Training weights, native executables and archives fail before transfer.
Ancestor README/license/notice/authors/copying/attribution/copyright files are
also preserved under `_hf_repository\ORIGINAL_PATH` for a subfolder selection;
that namespace is reserved. No bytes, declarations or templates are rewritten.
Files outside this selection are not acquired; review the explicit inventory
and repository terms. Metadata is **not legal approval**: `--accept-license`
records caller confirmation of reviewed licenses/attribution and download/use
rights, not a tool-issued grant or a provenance/authenticity certification.

Acquisition rejects unsafe/colliding Windows paths, reparse ancestry, oversized
or incomplete inventories, and unsupported files before downloading where
metadata permits. Bounds include the existing inspection limits, 4096 repository
entries, 64 GiB per data file, conservative 240-character materialized paths,
4 MiB remote metadata and 16 MiB schema-1 receipt. LFS content uses its SHA256;
ordinary Git files use **Git blob SHA1** (including the blob header), never a
mislabelled content SHA256. Every completed local file also retains content
SHA256. These identities prove transfer consistency, not supplier authenticity.

Transfers write regular independent files, not symlinks/shared-cache references.
Only a full inventory/checksum and `inspect-prebuilt` pass permits a same-volume
Windows atomic, no-replace directory rename. Existing destinations, even empty
ones, are never erased. ONNX support is required for structural validation.
The receipt retains structural/declaration/target/native levels; native evidence
is always **not-run**. Import and caller-supplied Ollama templates remain separate.

Receipts and staging remain under `hf-acquire-SESSION_ID` in the work parent,
with atomic state replacement and an integrity digest (not a signature).
`--resume` requires execution, consent, the reviewed fingerprint and identical
options/receipt path. It rehashes completed files against both receipt SHA256 and
immutable remote identities. A recorded partial is checked for tampering, then
discarded and downloaded anew: **file-level resume only**, no byte-range claim.
Changed bytes/options/metadata/destination fail closed. Completed receipts are
reverified, not treated as an unconditional cache hit. Locks serialize one session;
an unclean kill, stale lock, unrecorded file or interrupted state write can require
manual review/new acquisition, never automatic deletion of suspect files.
`--timeout` is a cooperative 1..3600-second budget, checked between reads/hashes
and validation stages, with at most 30 seconds per socket operation; ONNX parsing
is not forcibly preempted. Failed/cancelled transfers do not create a completion
receipt. Keep failed work for review or remove only that owned session yourself.

## Prepared Phi-3 Mini 4K engineering profile

For the traced prebuilt below, `prepare-prebuilt` supplies the reviewed template,
notices and runtime compatibility configuration. No SDK, Torch, Olive, runtime
DLL copying or manual JSON editing is needed. This is **prebuilt acquisition and
local packaging**, not conversion of arbitrary HF models.

Install `.[prebuilt]` as above. Start the compatible fork separately with its
verified QNN runtime and an explicit local endpoint. Use new directories and a
new model tag; the examples do not start or replace your server.

```powershell
$revision = "6fa72cea4103de0bc69792a6655be538a8c4d7e0"
$acquire = @("--repo", "llmware/phi-3-mini-4k-instruct-onnx-qnn", "--revision", $revision,
             "--out", "C:\models\phi3-source", "--work-dir", "C:\acquisition-work")
npu-model acquire-prebuilt @acquire
# Review the JSON inventory, rights and fingerprint before executing:
npu-model acquire-prebuilt @acquire --execute --expect-fingerprint ACQUISITION_PLAN_HASH --accept-license --timeout 1800

$prepare = @("--input", "C:\models\phi3-source", "--out", "C:\models\phi3-compatible",
             "--profile", "phi3-mini-4k-qnn")
npu-model prepare-prebuilt @prepare
# Review changes, license notices, target and the new fingerprint:
npu-model prepare-prebuilt @prepare --execute --expect-fingerprint PROFILE_PLAN_HASH --accept-license

$install = @("--input", "C:\models\phi3-compatible", "--name", "local/phi3:qnn-compatible",
             "--ollama-exe", "C:\fork\ollama.exe", "--host", "http://127.0.0.1:18765")
npu-model import-ollama @install
npu-model import-ollama @install --execute --expect-fingerprint IMPORT_PLAN_HASH --work-dir C:\acquisition-work --timeout 600
# Then use that same server explicitly:
$env:OLLAMA_HOST = "http://127.0.0.1:18765"
& C:\fork\ollama.exe run local/phi3:qnn-compatible
```

The source is 23 files / 1,970,617,736 bytes (about 1.84 GiB); allow space for
source, derived bundle, import snapshot, store and runtime. Preparation accepts
only complete source fingerprint
`6fa2fa387024705b616ac12f9468e6671502e24468420eaa3095d3c617cac653`.
Extra, edited or missing files are rejected. Originals remain unchanged.

The new 30-file bundle preserves the upstream card/source/copyright bytes and
original config, adds Apache-2.0 text, supplementary Microsoft base-family MIT
text with its precise reference, source hashes, changes notice, an LF Phi
Modelfile and an explicit schema-1 compatibility declaration. The MIT reference
does **not** identify the exact compiled model's base-weight revision. Both QNN
sessions set `disable_file_mapped_weights="1"`; the supported decoder `log_id`
carries a changed-file notice inside valid JSON. Pipeline member order is retained.
The upstream hash record remains historical, not an inventory of the derivative.
Notices/licenses are also attached as an Ollama license layer.

The engineering declaration is by this packager, not vendor certification:
Windows ARM64, exact X1E78100 processor name, GenAI QNN 0.12.2 / ORT QNN 1.24.1,
CPU embedding/head stages and QNN compiled decoder stages. The default is
**512-token context**, 128 generated tokens. Declared 4096 is compiled capacity,
**not useful-long-context qualification**. No tools/agent/structured-output claim.
Preparation never runs the model, contacts HF, installs DLLs or uploads anything.
It requires a reviewed fingerprint and rights confirmation; output is atomically
promoted without overwrite on Windows. Runtime licenses remain separate.

October 16 local qualification: all 23 files acquired from the immutable public
source; two independently assembled derivatives have content fingerprint
`b274a6b3443ac58cfe70405c6e5b3c595a981a5b665b95ffb937834c53aabf6d`.
An independently installed wheel imported all 30 named assets into an initially
empty isolated store using the retained qualified fork. All 69 hardware subtests /
50 exact raw prompts, normal templated generation/chat, unload/reload, CPU/NPU
switching and ten disconnect/30 queued-answer/ten deadline checks passed.
Seven actual official native module pins and clean unload were verified; the old
callback/teardown errors did not recur. DSP driver keys 49/50 remain visible.
This is not a clean-machine, broad quality, hosted blob-pull or producer-build
certification. Converter source remains local; no model/binary release is implied.

## Scope and truthful status

* Prebuilt import and passthrough **copy files**; they do not compile or validate native execution.
* `pack-ollama`, `convert --pack-ollama`, and the compatibility command `publish`
  **only assemble local files**. They do not run `ollama create`, test generation,
  or upload anything. The historical output names `ollama_publish` and
  `_publish_manifest.json` remain for compatibility, not as publication evidence.
* Explicit `import-ollama --execute` installs locally using `ollama create`
  and requires a compatible runtime fork. The older assembly commands remain unchanged.
  Local generation and native NPU execution must then be tested independently.
  Remote publication is a separate, explicit operation after those checks.
* Export defaults to `int4` with passthrough and `fp16` before a real quantizer.
  Explicit `int4` with QDQ/Olive quantization is rejected, never silently replaced.
  `bf16` is not supported by this builder path. QNN helper absence is an error,
  not a request to substitute generic quantization.
* See [validation levels and packaging contract](docs/runtime_ort_genai_folder.md).
  Missing ONNX/ORT/GenAI packages or QNN providers cannot produce a strict
  validation success. A strict QNN preflight pass covers session loads only;
  it is **not** generation or native-QNN supplier certification.

## Lightweight development checks (C0)

After installing `.[dev]`, run from this repository:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra --basetemp=sandbox_tmp\core_pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m mypy src --no-incremental
.\.venv\Scripts\npu-model.exe --help
.\.venv\Scripts\python.exe -m pip check
```

Pytest discovers `tests` only, preserving unrelated models/output directories.
Its designated `--basetemp` scratch directory is replaced on each run.
Missing optional packages still skip existing ONNX/calibration integration
tests; these skips are **not** support claims. Core dependency/error/policy
regressions use isolated mocks and run without those packages.
Mypy ignores missing imports only for named optional dependencies; their native
APIs are outside the lightweight type-check evidence.

C0 does not qualify new recipes, compile models on hardware, supply a native-QNN
producer, implement an upload service, or certify a usable model.
Historical model notes in `docs/known_good.md` are not current qualification.
