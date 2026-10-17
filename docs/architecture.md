# Architecture

## Pipeline stages

The `npu-model convert` command runs through these stages in order:

1. **Materialize source** — resolve `hf:<repo>` or local path to a local directory
2. **Inspect** — read `config.json` to produce a `ModelInfo`
3. **Select adapter** — auto-detect the right `ModelAdapter` based on `model_type` and `architectures`
4. **Adapter export** — adapter copies/exports ONNX graphs + tokenizer into a `GraphBundle`
5. **Quantization** — apply a `QuantizationStrategy` (MVP: passthrough no-op)
6. **Backend resolve target** — backend resolves `auto` or named target into a `TargetSpec`
7. **Backend prepare** — backend processes graphs into a `BackendPreparedBundle`
8. **Runtime assemble** — `RuntimeFormat` lays out the final bundle directory
9. **Manifest** — write `manifest.json` with file hashes + provenance

## Data flow

```
Input (HF or local)
  -> ModelInfo
  -> ModelAdapter.export() -> GraphBundle
  -> QuantizationStrategy.apply() -> GraphBundle (possibly modified)
  -> Backend.resolve_target() -> TargetSpec
  -> Backend.prepare() -> BackendPreparedBundle
  -> RuntimeFormat.assemble() -> bundle directory
  -> manifest.json
```

## Plugin system

Plugins are discovered via Python entry points (defined in `pyproject.toml`):

| Group                      | Base class       | Purpose                          |
|----------------------------|------------------|----------------------------------|
| `npu_model.adapters`       | `ModelAdapter`   | Model family export logic        |
| `npu_model.backends`       | `Backend`        | Hardware-specific preparation    |
| `npu_model.runtime_formats`| `RuntimeFormat`  | Output bundle layout             |
| `npu_model.quantizers`     | `QuantizationStrategy` | Quantization (optional)    |

## Error codes

Errors are structured as `NpuModelError(stage, reason_code, message, hint)`.

| Stage      | Code                  | Meaning                                      |
|------------|-----------------------|----------------------------------------------|
| source     | LOCAL_NOT_FOUND       | Local path doesn't exist                     |
| source     | LOCAL_MISSING_CONFIG  | No config.json in local directory             |
| source     | HF_BAD_SPEC          | Bad `hf:` spec format                        |
| source     | HF_DOWNLOAD_FAILED   | HF download error                            |
| inspect    | MISSING_CONFIG        | config.json not found in model dir            |
| inspect    | CONFIG_PARSE_FAILED   | config.json can't be parsed                  |
| adapter    | NO_ADAPTER            | No adapter matches the model                 |
| backend    | UNKNOWN_BACKEND       | Backend id not found                         |
| runtime    | UNKNOWN_RUNTIME_FORMAT| Runtime format id not found                  |
| runtime    | MISSING_ONNX_DIR     | Bundle missing onnx directory                |
| runtime    | NO_ONNX_FILES         | No .onnx files in bundle                    |
| quant      | UNKNOWN_QUANTIZER     | Quantizer id not found                       |
| registry   | PLUGIN_NO_ID          | Plugin class has no `id` attribute           |
| registry   | PLUGIN_LOAD_FAILED    | Plugin failed to load                        |
| pack       | BUNDLE_NOT_FOUND      | Bundle directory for packing not found       |

## Manifest format

`manifest.json` includes:

- `tool`: name + version
- `input`: source info (HF repo or local path)
- `plan`: adapter, backend, target, runtime, quantizer selections
- `backend_metadata`: backend-specific metadata
- `bundle`: format id + path
- `files`: list of `{path, size, sha256}` for all files in the bundle

## C0 truthfulness boundaries

`publish` is a compatibility command for conversion plus local folder assembly.
It neither installs in Ollama nor uploads. Configuration, layout, QNN session
preflight, generation, and native supplier execution are separate evidence levels;
see [runtime validation](runtime_ort_genai_folder.md).

Export precision is chosen only when omitted: INT4 for passthrough, FP16 before
quantization. Unsupported or incompatible explicit values fail before cache
restoration/export instead of silently substituting another precision.
Cache keys include export/calibration options and context-shape settings.
Stopped workflows do not restore completed cached conversions; cached context
bundles must still pass required QNN preflight when packing.
Cache keys are not full content fingerprints of local models or prompt files:
use `--no-cache` when those inputs change in place.

The lightweight mypy check excludes missing optional dependency imports only.
Mocks test policy/control flow, not native APIs, compiled artifacts, or supplier
coverage. Mandatory validation skips make strict preflight incomplete, never a
certification pass.

## Bounded local prebuilt route (C1/C2/C4)

`inspect-prebuilt` and `import-ollama` deliberately bypass adapter selection,
export/quantization/compilation and the legacy convert cache. They consume an
existing root `genai_config.json` plus tokenizer and graph assets. No source
resolution, HF traffic, archive extraction, new recipe or job engine is involved.

`inspect/prebuilt.py` inventories all regular contained portable assets, streams
SHA256 hashes, and computes identity from the sorted relative-path/size/hash
inventory. It bounds JSON/graph/folder sizes and rejects duplicate JSON keys,
duplicate stages, traversal and Windows reparse points. It reuses the existing
config sanity helper, then checks consumer model metadata, inherited pipeline
providers, graph references, external ONNX tensor data and EPContext caches.
The optional ONNX parser/checker is required, never replaced by filename/size
heuristics. Missing ONNX yields structural failure, not certification.

Evidence is intentionally separate:

* **Structural**: config/tokenizer JSON, graph/checker and contained referenced assets.
* **Declaration**: optional existing schema-1 `ollama_npu.json`, consistent with all
  graph/provider/context declarations. Source/precision are neither inferred nor verified.
* **Target**: declared versions compared to the pinned Windows ARM64 QNN consumer;
  optional caller-supplied exact processor name. No automatic host/native probe.
* **Native evidence**: always `not-run`, `certified=false` on this route.

`packagers/local_import.py` parses a bounded conservative Modelfile subset,
requiring one local literal FROM and an explicit caller-supplied Go TEMPLATE.
The default plan binds the complete inventory, Modelfile and executable hashes,
destination tag/loopback endpoint, processor comparison and overwrite policy.
Execution must supply that fingerprint. It checks the local endpoint before
staging, prevents unconsented overwrite, copies every asset preserving paths,
revalidates and compares source/snapshot inventories, and invokes bounded
`ollama create` against that snapshot. Only the FROM line is rewritten in a
separate staged Modelfile; input assets and metadata are never written.

HTTP uses direct numeric loopback connections without proxy/auth/redirects.
The CLI environment is allowlisted, with isolated home/app paths and controlled
PATH; adjacent desktop app executables are rejected. CLI success plus endpoint
tag presence means local installation only. A failed/timed-out create may leave
blobs or a tag on the server; there is no rollback or native-success claim.
Concurrent local tag writers are not synchronized by this tool.

Full content identity is specific to this prebuilt route; it does **not** solve
all existing convert-cache local input identity. See README for exact commands,
size limits, optional dependency and supported-subset boundaries. Lightweight
tests exercise API seams and a real Python fake-CLI subprocess, not valid native
graphs/contexts or actual Ollama/hardware integration.

## Immutable public HF acquisition

`sources/prebuilt_hf.py` and `acquire-prebuilt` add one bounded acquisition route,
not a recipe/conversion/publishing engine. `plan_acquisition(repo_id, destination,
work_dir=..., revision="main", subfolder=None, processor_name=None, resume=None)`
reads unauthenticated public metadata and returns an immutable `AcquisitionPlan`.
Its fingerprint binds schema/policy, repository, original revision, resolved
40-hex commit, explicit complete selected path/size/remote-identity inventory,
destination, owned work parent and target comparison. The selected subfolder is
materialized as root; ancestor attribution is retained under `_hf_repository`.

`execute_acquisition(plan, expected_fingerprint=..., accept_license=True,
resume=None, timeout=300)` revalidates metadata/paths, creates an exclusive
`hf-acquire-UUID` session and atomically replaces a bounded schema-1 receipt.
Receipt fields bind its absolute path, immutable plan and fingerprint, caller
rights consent, stage, verified completed content SHA256 inventory, interrupted
partial identity, error and final inspection. `state_sha256` detects accidental
edits; it is not authentication against someone able to rewrite and rehash state.
Completed files are always independently checked against immutable remote
identities, so an editable receipt is never sufficient content evidence.

Stages are downloading, interrupted/failed, validating, promoting and completed.
Each caught transfer interruption records partial progress without rehashing a
large failed transfer beyond its time budget. Resume first checks all persisted
bytes and original options, then reuses only verified **complete** files. It
discards only a matching recorded partial; unknown/mutated bytes are left intact
and rejected. Resume metadata uses the receipt commit even when a branch moves.
Unclean kills/stale locks and crash windows with unrecorded files fail closed.
A promotion-stage receipt can recover a completed rename after a state-write
failure by revalidating the destination. No existing destination is removed.

Direct standard-library HTTPS was chosen after reviewing `huggingface_hub`:
SDK defaults inherit tokens/endpoints/cache, and download implementations can
retry/resume or use Xet. This route instead uses no SDK transport/global session,
proxy/auth inheritance, telemetry or cache, restricts redirects to explicit HF
HTTPS CDN hosts, and performs no automatic network retries. Metadata is capped
before parsing; selected inventory is checked before data acquisition. LFS
SHA256 and Git blob SHA1 are deliberately distinct. SHA256 of actual local bytes
is retained for every completed file. None of these are supplier signatures.

Independent regular files pass the existing `inspect_prebuilt` contract and are
fully rehashed before Windows same-volume no-replace atomic directory promotion.
The optional ONNX checker reads metadata only; no runtime/model/code is executed.
Declarations, licenses and templates are preserved, not inferred or certified.
Legal review, correct Ollama Go templates, local installation, generation and
native supplier qualification remain separate gates. See README for CLI syntax,
bounds, timeout limitations and explicit unsupported private/gated inputs.

Python source is a narrowly bounded inert-text acquisition type, not an execution
feature: at most 4 MiB, valid UTF-8 with optional BOM, no NUL. Exact bytes and
copyright headers are retained. Neither structural inspection nor importing this
folder evaluates its Python. No general code/archive/native download fallback
was added. The immutable inventory also retains each LFS pointer's Git blob ID.

## Explicit derived engineering profile

`packagers/prebuilt_profile.py` and `prepare-prebuilt` implement one offline,
content-pinned derivative, independently of acquisition and installation.
`phi3-mini-4k-qnn` requires the complete 23-file source fingerprint recorded in
the module. The plan binds source/output paths, every source byte through that
fingerprint, generated file hashes, exact target and profile/schema version.
Execution requires its reviewed fingerprint and caller rights confirmation,
rechecks the source, creates an independent snapshot, applies the declared
changes, validates exact output inventory/config/graphs/declaration, and uses a
Windows same-volume no-replace rename. Only owned staging is cleaned on failure;
existing destination collisions are never removed.

The source/card/copyright files stay intact except the derived root config;
the original config is retained separately. Both explicit QNN pipeline sessions
disable file-mapped weights. A supported `log_id` carries a prominent changed-file
notice without adding unknown GenAI JSON fields. **Config object member order is
preserved**, because pipeline group members execute in order; sorted JSON is used
for receipts/identities, not this config. Packaged full license texts, source
receipt, limitations, changes notice and schema-1 declaration are hash-bound.
The included LF Go template/parameters/license layer make this profile usable
with the existing `import-ollama` command without manual model edits.

Local tag checks can scan large verified ORT inventories. Import uses the
remaining explicit execution budget rather than an unrelated 15-second cap.
After a zero create exit, a failed confirmation explicitly reports that the tag
may already exist; it does not retry create, overwrite, auto-start a server or
claim success. Publication and native qualification remain independent.
