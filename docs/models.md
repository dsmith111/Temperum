# Model coverage

What `npu-model` has converted on a Snapdragon X Elite laptop (X1E78100, 16 GB, Windows 11),
how long it took, how the result compares with the source model, and how fast it runs through
the [Ollama NPU fork](https://github.com/dsmith111/ollama-npu). Every row is one
`npu-model convert MODEL` run with default settings, then installed into the fork and asked
two chat turns; nothing was tuned per model.

## GGUF on the Hexagon NPU (the default)

`npu-model convert MODEL --install` (the runs used `--target gguf-hexagon`, now the
default): Q4_0, standard verification. The runtime is the fork's portable package
(branch `npu/hexagon-spike`, llama.cpp b11644) with Qualcomm's signed HTP libraries from
`ollama npu setup --provider hexagon`. Measured 2026-10-19.

| Model | Params | Convert + verify | NPU layers | Perplexity | Greedy exact | Long prompts | Template the fork uses | Verdict |
|---|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct | 0.36B | 0.9 + 1.8 min | 33/33 | 1.13x | 2/6 | 884 ✓, 1784 ✓ | checked (chatml) | passed |
| Qwen/Qwen2.5-0.5B-Instruct | 0.49B | - + 1.9 min | 25/25 | 1.16x | 2/6 | 875 ✓, 1763 ✓ | GGUF | passed |
| Qwen/Qwen3-0.6B | 0.75B | 0.5 + 3.0 min | 29/29 | 1.12x | 2/6 | 858 ✓, 1746 ✓ | GGUF | passed |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 0.5 + 3.7 min | 23/23 | 1.03x | 2/6 | 871 ✓ | built-in (Zephyr) | passed |
| Qwen/Qwen2.5-1.5B-Instruct | 1.54B | - + 7.6 min | 29/29 | 1.14x | 3/6 | 875 ✓, 1763 ✓ | GGUF | passed |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | 0.8 + 8.3 min | 29/29 | 1.11x | 2/6 | 851 ✗, 1739 ✗ (source ✗ too) | GGUF | passed |
| Qwen/Qwen3-1.7B | 2.03B | 1.0 + 8.4 min | 29/29 | 1.21x | 3/6 | 858 ✓, 1746 ✓ | GGUF | warning |
| microsoft/Phi-3.5-mini-instruct | 3.82B | 2.4 + 3.5 min | 33/33 | 1.10x | - | 858 ✓, 1758 ✓ | checked (phi3) | warning |
| microsoft/Phi-4-mini-instruct | 3.84B | 2.8 + 3.0 min | 33/33 | 1.16x | - | 848 ✓, 1737 ✓ | GGUF | warning |
| Qwen/Qwen2.5-7B-Instruct | 7.62B | 7.2 + 5.4 min | 29/29 | 1.08x | - | 875 ✓, 1763 ✓ | GGUF | warning |

Speed through the fork (tokens per second; first token in seconds):

| Model | NPU prefill | NPU decode | First token | CPU prefill | CPU decode | CPU first token |
|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-360M-Instruct | 1,920 | 71.1 | 0.30 | 1,372 | 162.1 | 0.41 |
| Qwen/Qwen2.5-0.5B-Instruct | 2,546 | 68.1 | 0.23 | 1,483 | 156.6 | 0.38 |
| Qwen/Qwen3-0.6B | 1,743 | 43.4 | 0.33 | 845 | 80.1 | 0.65 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1,510 | 53.4 | 0.42 | 623 | 99.6 | 0.98 |
| Qwen/Qwen2.5-1.5B-Instruct | 1,230 | 34.5 | 0.48 | 559 | 68.3 | 1.01 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1,179 | 34.8 | 0.48 | 524 | 69.4 | 1.03 |
| Qwen/Qwen3-1.7B | 1,054 | 26.3 | 0.55 | 443 | 50.4 | 1.24 |
| microsoft/Phi-3.5-mini-instruct | 491 | 9.7 | 1.32 | 198 | 27.9 | 3.05 |
| microsoft/Phi-4-mini-instruct | 713 | 12.8 | 0.78 | 231 | 30.5 | 2.20 |
| Qwen/Qwen2.5-7B-Instruct | 390 | 10.0 | 1.53 | 118 | 13.8 | 5.00 |

How to read it:

- **Convert + verify:** convert is the download, llama.cpp's converter and `llama-quantize`;
  verify runs the source model for reference answers, the bundle on the NPU and perplexity
  on both. The two smaller Qwen2.5 jobs reused a quantized file from an earlier run, so only
  their verification is timed; Qwen2.5-7B's 7.2 minutes include 3.9 of download. Peak memory
  was 5.8 GB for Phi-4-mini (llama.cpp's converter), 4.8 GB for Phi-3.5-mini and 8.2 GB for
  Qwen2.5-7B (the CPU perplexity reference).
- **Verdict "warning" for the Phi models and Qwen2.5-7B:** their source models need more
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
  in wording.
- **Long prompts:** a fact placed at the start of a prompt of about 900 and 1,800 tokens
  (TinyLlama's 2,048-token context fits only the first). "Source ✗" means the source model
  missed it too: DeepSeek-R1 spends its 16 answer tokens thinking.
- **Template the fork uses**, from the fork's own log: "GGUF" is the model's own chat
  template, rendered by llama-server (the fork prefers it when it supports tool calls);
  "checked" is the Ollama template the converter derived and compared with the model's own,
  which the bundle carries; "built-in" is a template the fork substituted at
  `ollama create` because it resembles the model's. TinyLlama's Zephyr format is not one the
  converter recognizes, and `npu-model install` reports the substitution as unchecked.
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

## QNN bundles (8-bit)

`npu-model convert MODEL --target qnn-x-elite` (the default target until 2026-10-19; Olive
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

- **Llama 3.2** needs access to Meta's gated repository; `npu-model` stops at the plan with
  the access link until the account has it. (SmolLM2 and TinyLlama use the Llama
  architecture.)
- **One unreproduced failure:** in one of about 19 two-turn chats, Phi-4-mini (then split
  between NPU and CPU) answered the second turn with unrelated words. Later runs, split and
  whole, did not repeat it.
- **The CUDA quantization path** (larger QNN models, recipe-exact GPTQ) needs an x64 machine
  with an NVIDIA GPU.
