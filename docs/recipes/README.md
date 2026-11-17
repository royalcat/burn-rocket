# Recipes

Ready-to-run command lines for deploying each model on a Rock 5B+. Every recipe uses the
unified `rocket-inference` CLI: the model family is detected from `--model-dir` (or
`-hf`), and each stage can be placed on the NPU or the CPU.

## Pick a profile

| profile | good for | placement |
|---|---|---|
| **balanced** | saving RAM and CPU together | everything the model supports on the NPU, bf16 weights, towers load when first needed |
| **full offload** | leaving the CPU free | everything on the NPU, highest-precision weights, towers loaded at startup |
| **speed** | fastest steady-state requests | everything on the NPU, highest-precision weights, towers load when first needed |

Models without a decode stage or towers have fewer distinct profiles; each recipe says so.
For a CPU-only run pass `--device cpu` (on the NPU build the default `--device auto`
prefers the NPU).

## Placement flags

```
--device auto|cpu|npu            default for every stage (default: auto)
--text-device ...                text-backbone projections
--text-attn-device ...           text-backbone attention (qwen3, embeddinggemma)
--vision-device ...              vision-tower projections (embeddinggemma)
--vision-attn-device ...         vision-tower attention (embeddinggemma)
--decode-device ...              single-token decode (chat models)
```

`auto` means "NPU when the build has one". `--device npu` is the one-flag full offload.
The program prints the result before loading:

```
placement: text=NPU(fp16) text-attn=NPU vision=NPU(lazy) vision-attn=NPU audio=CPU
```

The audio tower always runs on the CPU.

## Recipes

| model | recipe |
|---|---|
| EmbeddingGemma 2 (text, image, video, audio embeddings) | [`embeddinggemma2.md`](embeddinggemma2.md) |
| Qwen3-Embedding-0.6B | [`qwen3.md`](qwen3.md) |
| Qwen3.5-0.8B intent / query planner | [`intent.md`](intent.md) |
| Gemma 4 E2B-it chat | [`gemma4.md`](gemma4.md) |
| vllm-sr Decision 3.0 | [`decision.md`](decision.md) |
| Clef-Omni typed decisions | [`clef.md`](clef.md) |

Measurements behind these recipes, and the mapping from the older flag names, are in
[`../experiments/`](../experiments/README.md).
