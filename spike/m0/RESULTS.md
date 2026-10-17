# M0 results (in progress)

Host: Snapdragon X Elite X1E78100, 16 GB RAM, Windows 11 ARM64 26100, **no NVIDIA GPU**.
Model: `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0).
Recipe: `microsoft/olive-recipes` `Qwen-Qwen2.5-0.5B-Instruct/QNN/config.json` @ `04040dcf`, with
the CPU overlay from `make_config.py`. Environments and pins: [README.md](README.md).

## Headline

**A Hugging Face model was converted entirely on this laptop's CPU, compiled for its NPU,
imported into the Ollama fork and served through `/api/chat` on the NPU**
(`provider=qnn state=qnn-compiled-with-cpu-stages`), answering "Paris." at ~79 tokens/s.
No CUDA, QNN SDK, manual DLLs or JSON edits; one hand-written Modelfile (template).

Open problems: quality with the fast (RTN int4) mode is poor; CPU GPTQ timing is unresolved;
two runtime `max_length` rules must be enforced (below).

## Conversion timings (`fast` = QuaRot + RTN int4 per-channel, LM head int8 g32)

| Pass | Time | Notes |
|---|---|---|
| QuaRot (Hadamard rotation) | 54 s | CPU |
| CaptureSplitInfo | 1.3 s | |
| Rtn | 8 s | CPU |
| ModelBuilder (GenAI 0.12.2) | 6 s | int4 MatMulNBits, int4 embedding Gather |
| MatMulNBitsToQDQ | 11 s | |
| GraphSurgeries | 11 s | |
| OnnxStaticQuantization | **672 s** | uint16 activations, 64×512 samples, CPU EP |
| SplitModel | 3.5 s | |
| StaticLLM | 7 s | 64-token prefill graph + 1-token iterator |
| EPContextBinaryGenerator | 69 s | `onnxruntime-qnn==1.24.1` (QNN 2.42), on-device |
| ComposeOnnxModels | 1.5 s | |
| **Total** | **~14 min** | peak RSS 4.0 GB |

First attempt failed only because the compile env lacked `requests` (Olive 0.13 imports it
at startup without declaring it). Olive's cache resumed from the compile pass after the fix.

## Output bundle (612 MB)

`embeddings.onnx` (CPU) → `context_ctx.onnx` (QNN, prefill) / `iterator_ctx.onnx` (QNN,
decode), both wrapping one shared `context_ctx_qnn.bin` (370 MB, weight sharing) →
`lm_head.onnx` (CPU), plus `genai_config.json` (`decoder-pipeline`) and tokenizer files.

Architecture is **hybrid**: each graph has 25 QNN EPContext partitions (projections, MLP,
norms) with `GroupQueryAttention` running on the CPU EP between them. The KV cache is
dynamic, so the declared 32,768-token context is real capacity, not a compiled limit.
QNN sessions must therefore allow CPU nodes (attention).

## Runtime compatibility

| Runtime | Result |
|---|---|
| Fork pins: GenAI 0.12.2 / ORT 1.24.1 / QNN 2.42 (Python) | loads, generates correctly |
| Newest: GenAI 0.18 / ORT 1.31 / `onnxruntime-qnn` 2.6 plugin (Python) | loads (QNN 2.42 contexts load on newer QNN), generates correctly |
| Fork binary (`153b36bc`, dev build) + verified scratch runtime | `ollama create` + `/api/chat` work; 79 tok/s decode, 0.09 s prompt eval |

**Rules discovered** (must be enforced by converter config and fork):

1. `search.max_length` must be ≥ the 64-token prefill window. Below it, GenAI 0.12.2 silently
   overflows the KV buffer (garbage output, later a hang); GenAI 0.18 raises a GQA error.
2. `max_length` must be ≥ the prompt length **rounded up to the 64-token window** plus new
   tokens. A 1,763-token prompt pads to 1,792; with `max_length` 1,779 GenAI 0.12.2
   **segfaults** and 0.18 raises `seqlens_k out of range`. (Compare the fork's recorded
   3,525-token failure on the older prebuilt bundle.)
3. Every `append_tokens()` call on these static pipelines is a padded window prefill; feeding
   tokens one at a time consumes 64 KV positions each.

## Quality (fast/RTN int4) vs CPU fp32 reference

Measured with `verify_genai.py` on GenAI 0.18 (identical token IDs to the reference):

| Metric | Result |
|---|---|
| Greedy exact match | 0/12 |
| Mean matching prefix | 7.3% |
| Perplexity (sampled positions) | 43.6 vs 18.4 reference (**×2.36**) |
| Needle retrieval | 379 ✓, 875 ✓, 3,445 ✓ (1,763 hit rule 2) |
| Median TTFT / decode | 0.13 s / ~35 tok/s (Python harness); ~79 tok/s via fork |

Bisection on CPU (raw ORT, `ort_greedy.py`): the QDQ-weight stage and the uint16
static-quant stage both answer correctly ("Paris.", "17 + 25 = 42."), so the numeric graph is
sound through calibration; the remaining loss comes from RTN int4 itself plus real HTP
arithmetic (NPU answered "92" where the CPU QDQ simulation said "42").

## CPU GPTQ

- First full run (32×1024 calibration): ~920 s/layer (~6 h projected), CPU ~4 cores, working
  set only 0.6 GB for a 2 GB model. Stopped.
- Profile on a 2-layer slice (`profile_gptq.py`): 13.9 s/layer at 2K tokens, 30.5 s/layer at
  8K tokens: linear, ~7 s/layer fixed solver cost. Projects to ~100 s/layer (~40 min) at 32K
  tokens. The 10× gap in the full run is unexplained (suspect memory pressure); rerun pending.
- torch 2.14.1 win_arm64 uses Arm Performance Libraries BLAS, ~60–90 GFLOP/s fp32 matmul,
  `CPU capability: DEFAULT`, no oneDNN.

## Pending

- 8-bit RTN weights (in progress) — Qwen3 recipes use 8-bit linear weights.
- GPTQ balanced run with reduced calibration; quality vs RTN.
- Second architecture (Qwen3-0.6B or Llama-3.2-1B, gated).
- GenieX (signed llama.cpp Hexagon) comparison.
