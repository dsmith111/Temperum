# Model coverage

What `temperum` has converted on a Snapdragon X Elite laptop (X1E78100, 16 GB, Windows 11),
how long it took, how the result compares with the source model, and how fast it runs through
the [Ollama NPU fork](https://github.com/dsmith111/ollama-npu). Every row is one
`temperum convert MODEL` run with default settings, then installed into the fork and asked
two chat turns; nothing was tuned per model.

## GGUF on the Hexagon NPU (the default)

`temperum convert MODEL --install` (the runs used `--target gguf-hexagon`, now the
default): Q4_0, standard verification. The runtime is the fork's portable package
(branch `npu/hexagon-spike`, llama.cpp b11644) with Qualcomm's signed HTP libraries from
`ollama npu setup --provider hexagon`. Measured 2026-10-19; the Llama 3.2 rows on 2026-10-20, Llama-3.1-8B on 2026-10-21.
Since 2026-10-23 the conversion calibrates the quantization with an importance matrix by default when the
fork package ships `llama-imatrix`; the rows marked † were re-measured that day with calibration on, the
others are plain Q4_0 from before.

| Model | Params | Convert + verify | NPU layers | Perplexity | Greedy exact | Long prompts | Template the fork uses | Verdict |
|---|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct † | 0.36B | 1.7 + 1.9 min | 33/33 | 1.07x | 3/6 | 884 ✓, 1784 ✓ | checked (chatml) | passed |
| Qwen/Qwen2.5-0.5B-Instruct | 0.49B | - + 1.9 min | 25/25 | 1.16x | 2/6 | 875 ✓, 1763 ✓ | GGUF | passed |
| Qwen/Qwen3-0.6B | 0.75B | 0.5 + 3.0 min | 29/29 | 1.12x | 2/6 | 858 ✓, 1746 ✓ | GGUF | passed |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 0.5 + 3.7 min | 23/23 | 1.03x | 2/6 | 871 ✓ | built-in (Zephyr) | passed |
| meta-llama/Llama-3.2-1B-Instruct | 1.24B | 1.2 + 11.7 min | 17/17 | 1.13x | - | 869 ✓, 1749 ✓ | GGUF | passed |
| Qwen/Qwen2.5-1.5B-Instruct † | 1.54B | 4.7 + 7.2 min | 29/29 | 1.12x | 2/6 | 875 ✓, 1763 ✓ | GGUF | passed |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | 0.8 + 8.3 min | 29/29 | 1.11x | 2/6 | 851 ✗, 1739 ✗ (source ✗ too) | GGUF | passed |
| Qwen/Qwen3-1.7B | 2.03B | 1.0 + 8.4 min | 29/29 | 1.21x | 3/6 | 858 ✓, 1746 ✓ | GGUF | warning |
| meta-llama/Llama-3.2-3B-Instruct † | 3.21B | 10.1 + 2.6 min | 29/29 | 1.02x | - | 869 ✓, 1749 ✓ | GGUF | warning |
| microsoft/Phi-3.5-mini-instruct | 3.82B | 2.4 + 3.5 min | 33/33 | 1.10x | - | 858 ✓, 1758 ✓ | checked (phi3) | warning |
| microsoft/Phi-4-mini-instruct | 3.84B | 2.8 + 3.0 min | 33/33 | 1.16x | - | 848 ✓, 1737 ✓ | GGUF | warning |
| Qwen/Qwen2.5-7B-Instruct | 7.62B | 7.2 + 5.4 min | 29/29 | 1.08x | - | 875 ✓, 1763 ✓ | GGUF | warning |
| meta-llama/Llama-3.1-8B-Instruct | 8.03B | 6.6 + 7.3 min | 33/33 | 1.06x | - | 869 ✓, 1749 ✓ | GGUF | warning |

Speed through the fork (tokens per second; first token in seconds):

| Model | NPU prefill | NPU decode | First token | CPU prefill | CPU decode | CPU first token |
|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct | 1,920 | 71.1 | 0.30 | 1,372 | 162.1 | 0.41 |
| Qwen/Qwen2.5-0.5B-Instruct | 2,546 | 68.1 | 0.23 | 1,483 | 156.6 | 0.38 |
| Qwen/Qwen3-0.6B | 1,743 | 43.4 | 0.33 | 845 | 80.1 | 0.65 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1,510 | 53.4 | 0.42 | 623 | 99.6 | 0.98 |
| meta-llama/Llama-3.2-1B-Instruct | 1,940 | 46.7 | 0.29 | 766 | 79.9 | 0.70 |
| Qwen/Qwen2.5-1.5B-Instruct | 1,230 | 34.5 | 0.48 | 559 | 68.3 | 1.01 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1,179 | 34.8 | 0.48 | 524 | 69.4 | 1.03 |
| Qwen/Qwen3-1.7B | 1,054 | 26.3 | 0.55 | 443 | 50.4 | 1.24 |
| meta-llama/Llama-3.2-3B-Instruct | 893 | 19.0 | 0.65 | 270 | 33.7 | 1.99 |
| microsoft/Phi-3.5-mini-instruct | 491 | 9.7 | 1.32 | 198 | 27.9 | 3.05 |
| microsoft/Phi-4-mini-instruct | 713 | 12.8 | 0.78 | 231 | 30.5 | 2.20 |
| Qwen/Qwen2.5-7B-Instruct | 390 | 10.0 | 1.53 | 118 | 13.8 | 5.00 |
| meta-llama/Llama-3.1-8B-Instruct | 376 | 8.9 | 1.52 | 108 | 17.3 | 4.96 |

How to read it:

- **Convert + verify:** convert is the download, llama.cpp's converter and `llama-quantize`;
  verify runs the source model for reference answers, the bundle on the NPU and perplexity
  on both. The two smaller Qwen2.5 jobs reused a quantized file from an earlier run, so only
  their verification is timed; Qwen2.5-7B's 7.2 minutes include 3.9 of download. Peak memory
  was 5.8 GB for Phi-4-mini (llama.cpp's converter), 4.8 GB for Phi-3.5-mini and 8.2 GB for
  Qwen2.5-7B (the CPU perplexity reference). Llama-3.1-8B's 6.6 minutes include 2.9 of download;
  its converter peaked at 4.0 GB with 2 GB of RAM free at the time, so the perplexity stage paged
  (the 7.3 minutes of verification are an upper bound).
- **Calibration (†):** the rows re-measured on 2026-10-23 quantized with an importance matrix that
  `llama-imatrix` computed on the CPU from 100 chunks of 512 tokens of wikitext-2 train rows (the perplexity
  column uses the test rows). Their convert time includes it: 0.7 minutes for SmolLM2-360M, 3.6 for
  Qwen2.5-1.5B and 6.4 for Llama-3.2-3B, whose 10.1 also holds 1.2 of download. Against the same models'
  plain Q4_0 rows the perplexity ratio fell from 1.13x to 1.07x, 1.14x to 1.12x and 1.06x to 1.02x, with
  the file size within 0.3% and decode within measurement noise (the block format is unchanged, so the
  speed table keeps its earlier rows). Greedy exact moved by one prompt either way, inside what six
  prompts can tell apart.
- **Verdict "warning" for the Phi models, Qwen2.5-7B and Llama-3.1-8B:** their source models need more
  memory than this laptop has free, so verification compares the source model's tokenizer
  and chat template but not its answers ("Greedy exact" is blank); perplexity, long prompts
  and the template are still checked. Qwen3-1.7B's warning is its Q4_0 perplexity, just over
  1.20x.
- **NPU layers:** layers `llama-server` placed on the NPU (HTP0) during verification.
- **Perplexity:** the bundle on the NPU against a Q8_0 copy of the same conversion on the CPU
  (`llama-perplexity`, 16 × 512 tokens of held-out text); Q8_0 is within about 0.5% of the
  16-bit model. Verification passes at 1.20x or below; Q4_0 is the fastest type, and
  `--quant Q4_K_M` or `Q8_0` keep more quality.
- **Greedy exact:** of six prompts, how many gave the same first 32 tokens as the source
  model in transformers. With 4-bit weights, answers usually agree in substance and drift
  in wording. The column also measures the NPU's arithmetic, not only the quantization:
  measured 2026-10-23 with the same Q4_0 file served plain on the NPU and on the CPU (six
  prompts, five identical greedy requests each, 128 tokens), Qwen2.5-1.5B left the CPU's
  greedy path within the first few tokens on every prompt, Qwen3-4B and SmolLM2-360M part way
  through most, and Llama-3.2-3B stayed on it for five of six. Both devices repeat their own
  answer exactly, with one exception in 24 prompt-and-model pairs (Qwen3-4B's email prompt
  varied once in five on the NPU). Perplexity, which averages over many tokens, moves far less
  (Q8_0 on the NPU scores 1.035x its CPU perplexity on Qwen2.5-1.5B and 0.999x on
  Llama-3.2-3B).
- **Long prompts:** a fact placed at the start of a prompt of about 900 and 1,800 tokens
  (TinyLlama's 2,048-token context fits only the first). "Source ✗" means the source model
  missed it too: DeepSeek-R1 spends its 16 answer tokens thinking.
- **Template the fork uses**, from the fork's own log: "GGUF" is the model's own chat
  template, rendered by llama-server (the fork prefers it when it supports tool calls);
  "checked" is the Ollama template the converter derived and compared with the model's own,
  which the bundle carries; "built-in" is a template the fork substituted at
  `ollama create` because it resembles the model's. TinyLlama's Zephyr format is not one the
  converter recognizes, and `temperum install` reports the substitution as unchecked.
  SmolLM2 shows why the bundle carries a template: before it did, the fork substituted its
  built-in ChatML, which drops SmolLM2's default system prompt. Its row is from a rerun with
  the template (the first run's verification gave 1.12x).
- **Speed** comes from the fork's `/api/generate`: a prompt of about 540 tokens of held-out
  text, 128 greedy tokens, the median of three runs after a warm-up, with a new prompt each
  run. First token is the prompt pass plus one decode step. CPU is the same model with
  `num_gpu 0`. The Phi models load whole on the NPU with the fork's 5,120 MiB budget and
  Qwen2.5-7B (4.2 GiB on the NPU) with its 6,144 MiB one; under smaller budgets the fork split
  them with the CPU, which was slower (Phi-4-mini: 384 tok/s prefill, 9.0 decode; the 7B: 294
  and 6.5). The 7B's CPU figures varied with other work on the laptop: an editor was using
  about two cores around these runs, and a quieter run of `llama-server` alone decoded at 20.6
  tok/s and prefilled 124, while the NPU stayed at about 10 tok/s.

The NPU's main gain is leaving the CPU to you: generating with Qwen2.5-1.5B through the fork
(a prompt of 540 tokens and 256 new tokens), the server used 0.1 of a CPU core on the NPU
and 5.3 cores on the CPU. That is what lets an agent work alongside you without slowing your
own work; power matters mostly for chat use. On speed, the NPU processes prompts 1.4 to 3.3
times faster than the CPU and reaches the first token in 31 to 75% of the CPU's time, while
the CPU decodes two to three times as fast when nothing else needs it (Qwen2.5-7B: 1.4 times
with the editor busy). (These CPU figures are the package's CPU backend with dotprod and
i8mm kernels; the earlier generic ARMv8 build prefilled 3 to 7 times slower.)

## Paired drafts (DFlash)

`temperum convert Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16 --verify standard`, 2026-10-23, 16.1
minutes with the fork package that ships `llama-imatrix`: the target as calibrated Q4_0 (37/37 layers,
2,260 MiB, perplexity 1.04x), the draft converted with the target's tokenizer and metadata to Q8_0
(545 MiB, 6 layers), both on the NPU (2,825 MiB of weights), the same greedy answers with and without the
draft. The verdict's draft line for this pair: decode 0.23x to 2.18x of plain across the six prompts, median
0.71x; 0% to 63% of its guesses accepted.

| Prompt | Answer tokens | Plain tok/s | With the draft | Guesses accepted |
|---|---:|---:|---:|---:|
| Capital of France, one word | 3 | 12.8 | 3.0 (0.23x) | 0 of 30 |
| 17 + 25, just the number | 3 | 12.2 | 3.5 (0.29x) | 0 of 30 |
| Name the three primary colors | 32 | 13.1 | 9.3 (0.71x) | 19 of 131 |
| Translate a greeting to Spanish | 12 | 14.2 | 7.5 (0.53x) | 6 of 75 |
| A Python function (code) | 32 | 11.6 | 25.3 (2.18x) | 27 of 43 |
| Why the sky looks blue, one sentence | 32 | 11.0 | 9.8 (0.89x) | 20 of 122 |

A draft call costs about 1.5 plain decode steps on the NPU, so a step pays from about three accepted tokens:
code gets there, short factual answers do not, and a three-token answer pays for two whole blocks of guesses.
The draft stays on in the bundle because the user asked for the pair; `PARAMETER draft_num_predict 0` in the
Modelfile turns it off. Longer measurements of the same pair (256-token answers, four prompt kinds) are in
the handoff's section 4: 1.76x on code, 1.10x on prose, 0.53x to 0.59x on short chat answers, the same
acceptance against a Q8_0 target and on the CPU.

## QNN bundles (8-bit)

`temperum convert MODEL --target qnn-x-elite` (the default target until 2026-10-19; Olive
recipes, CPU quantization with rotated 8-bit round-to-nearest weights, QNN context binaries
for HTP v73). Measured 2026-10-17/18,
verification at the standard level:

| Model | Time | Perplexity | Greedy exact | Long prompts | Template | Verdict |
|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 22 min | 1.04x | 2/6 | 875 ✓, 1763 ✓ | match | passed |
| HuggingFaceTB/SmolLM2-360M-Instruct | 9 min (resumed) | 1.15x | - | ✓ | match | passed |
| Qwen/Qwen3-0.6B | M0 output | 1.06x | - | ✓ | match | passed |
| Qwen/Qwen2.5-1.5B-Instruct | 55 min | 1.02x | - | 875 ✓, 1763 ✓ | match | passed |

Perplexity here is against the source model in float32. The 1.5B conversion peaked at 7.9 GB
of RAM, which limits 16 GB laptops to about 1.5B parameters on this route. Through the fork,
Qwen2.5-1.5B decoded at 20.2 tok/s and prefilled 772 tok/s, against 28.7 and 1,229 for its
GGUF on the NPU (same harness, 556-token prompt).

## Not covered yet

- **Llama 3.1** (8B) needs its own licence acceptance on Hugging Face, separate from Llama
  3.2's; `temperum` stops at the plan with the access link until the account has it. The
  Llama 3.2 rows: the 1B's source model ran for reference answers (its greedy answers were
  not scored, as the row shows); the 3B's did not fit beside an editor in 16 GB, so its
  answers were not compared. Both keep the GGUF's own template (Llama 3.2's date preamble),
  which the converter's llama3 Go template lacks; the verifier compared that rendering with
  transformers' and, since 2026-10-20, ignores the BOS text llama-server adds at
  tokenization instead of in the prompt.
- **One unreproduced failure:** in one of about 19 two-turn chats, Phi-4-mini (then split
  between NPU and CPU) answered the second turn with unrelated words. Later runs, split and
  whole, did not repeat it.
- **The CUDA quantization path** (larger QNN models, recipe-exact GPTQ) needs an x64 machine
  with an NVIDIA GPU.
