# Recipes

How to run each model on a Rock 5B+, what the defaults do, and how to change them for your task.

## One default per model

Every model has a **preset**: the placement and memory settings that `--device auto` (the default)
picks for it. A preset puts as much of the model on the NPU as is actually faster, and keeps a
stage on the CPU when the NPU would be slower or when it would use much more memory. Presets
differ per model because the models differ.

Each run prints what was chosen:

```
placement: text=NPU(fp16) text-attn=NPU vision=NPU(lazy) vision-attn=NPU audio=CPU
preset: embeddinggemma defaults (no placement flags given)
```

If a stage was left on the CPU on purpose, a `hint:` line says why and which flag changes it.
If you passed flags, the `preset:` line lists the ones that overrode the preset.

## How to change a preset

| What you want | Use |
|---|---|
| Everything on the CPU | `--device cpu` |
| Everything on the NPU, including decode | `--device npu` |
| One stage elsewhere | `--text-device`, `--text-attn-device`, `--vision-device`, `--vision-attn-device`, `--decode-device` (each `auto`, `cpu` or `npu`) |
| Towers loaded at startup, not on first use | `--no-defer-towers` |
| Different precision | `--dtype`, `--npu-dtype`, `--quant` (see each recipe) |
| Lower peak memory on long inputs | `--attn-chunk`, `ROCKET_MATMUL_CHUNK_M` |

Without an NPU build (aarch64 with the `npu` feature), `auto` means the CPU everywhere.
`--device npu` on such a build is an error.

## Recipes

Each recipe has the same parts: the command, what the preset does and why, what it trades
away, and a table of changes for common goals with their measured costs.

| Model | Recipe |
|---|---|
| EmbeddingGemma 2 (text, image, video and audio embeddings) | [`embeddinggemma2.md`](embeddinggemma2.md) |
| Qwen3-Embedding-0.6B | [`qwen3.md`](qwen3.md) |
| Qwen3.5-0.8B intent / query planner | [`intent.md`](intent.md) |
| Gemma 4 E2B-it chat | [`gemma4.md`](gemma4.md) |
| vllm-sr Decision 3.0 | [`decision.md`](decision.md) |
| Clef-Omni typed decisions | [`clef.md`](clef.md) |

Measurements behind these recipes, and the mapping from the older flag names, are in
[`../experiments/`](../experiments/README.md).
