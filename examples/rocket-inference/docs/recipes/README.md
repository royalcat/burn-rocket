# Recipes

Command lines for the three deployment profiles, per model. The commands are
for the unified `rocket-inference` CLI: the model family is detected from
`--model-dir` (or `-hf`), and every stage is placed explicitly with the
per-stage device flags.

## Profiles

| profile | what it optimises | placement |
|---|---|---|
| **balanced** | RAM and CPU relief together | every stage the model supports on the NPU, bf16 body, towers loaded lazily |
| **full offload** | nothing on the CPU that can leave it | every stage on the NPU, highest-fidelity dtype, towers resident from startup |
| **speed** | fastest steady-state request | every stage on the NPU, highest-fidelity dtype, lazy towers (same compute as full offload, lower idle RAM, first media request pays the tower load) |

`balanced` and `speed` differ in dtype and laziness; `full offload` pays RAM
for zero lazy-load latency. Where a model has no separate decode stage or no
towers, the profiles collapse; each model file says so.

The profiles assume the aarch64 NPU build (`--no-default-features --features
npu`). On an x86-64/CPU build, `auto` resolves to the CPU; use `--device cpu`
explicitly and expect the CPU rows in the model files.

CPU-only fallback for every model: `--device cpu` (and, on the NPU build, the
default `--device auto` is NPU-first, so the fallback must be explicit).

## Placement flags

```
--device auto|cpu|npu            base for every stage (default auto)
--text-device ...                text-backbone projections (the only stage for d3)
--text-attn-device ...           text-backbone attention (qwen3, embeddinggemma)
--vision-device ...              vision-tower projections (embeddinggemma)
--vision-attn-device ...         vision-tower attention (embeddinggemma)
--decode-device ...              single-token decode projections (chat models)
```

`auto` = NPU when the build has it (aarch64 + the `npu` feature), CPU
otherwise; decode stays on the CPU under `auto` because NPU decode is slower.
`--device npu` is the one-flag full offload (it includes decode). Every run
prints the resolved split before loading:

```
placement: text=NPU(fp16) text-attn=NPU vision=NPU(lazy) vision-attn=NPU audio=CPU
```

The audio tower is CPU-only in every profile; it is not an NPU stage.

## Measured profiles

Board: Rock 5B+, 8 cores, 600 MHz patched `rocket` module, 2026-10-31,
`taskset -c 4-7`, EmbeddingGemma 2 safetensors (native bf16) at
`/root/models/embeddinggemma-2`, `home-llm-gateway-embed-1` stopped for the
run (other board services still active, load average ~7.5; the spread between
identical runs is ±10-20 %). Text is `data/one_long.txt` (2,587 tokens).
Media: `data/cat.jpeg` (2,394 patches) and `data/speech5s.wav` (129 tokens).
`encode` is input/tower work, `forward` the text backbone.

| model | balanced | full offload | speed |
|---|---|---|---|
| EmbeddingGemma 2 | 15.2 s text, 723 MiB | 13.6 s text, 2,605 MiB | 13.3 s text, 1,330 MiB |
| Qwen3-Embedding | ~54 tok/s (bf16) | ~58 tok/s (f32) | ~58 tok/s (f32) |
| Qwen3.5 intent | 11.7 s / 3.7 tok/s | 1.5 tok/s (CPU-free) | 9.3 s / 5.0 tok/s |
| Gemma 4 E2B-it | prefill NPU, decode CPU | prefill + decode NPU | prefill NPU, decode CPU |
| Decision 3.0 (d3-lite) | 147.3 s / 5 questions, 6,546 tokens (NPU) | same (collapses) | same (collapses) |

Only EmbeddingGemma 2 was re-measured for the 2026-10-31 profile round. The other
rows are derived from the dated experiment logs (`docs/experiment-log.md`,
`docs/experiment-log-gemma.md`) and are marked as such in their files; d3's row
is its own 2026-11-02 board window, below.

d3 has a single NPU stage (`text`), so its three profiles are literally the same
command (`--device npu --defer-towers`); the vision tower, the backbone
attention and the readout head stay on the CPU, and there is no decode. The row
is the 2026-11-02 memory window in [`decision.md`](decision.md) (`## Deployment
(current)`): 5 text questions / 6,546 prompt tokens at 147.3 s compute / 160.2 s
wall beside the live service (the 2026-11-01 idle-service window read 113 s NPU
vs 197 s CPU f32, and the contended CPU arm 265.6 s / 270.1 s); a text-only
process is 347 MiB anon resident plus 966.0 MiB of weights BO. Its server is
`POST /v1/systemone`, which takes images/videos as `data:` URLs only and has a
96 MiB body limit (one 32 MB video plus up to three 8 MB images, base64-encoded,
plus the JSON around them).

## Important: the experiment logs use the pre-refactor flags

The measurement logs in `docs/` predate the 2026-10-31 CLI refactor. Their
flag names (`--npu`, `--npu-attn`, `--vision-attn`, `--npu-int8`,
`--pure-npu`, `--npu-decode`, `--backend`, `--f16`) no longer exist. The
numbers remain valid for the corresponding placement; the mapping is:

| old | new |
|---|---|
| `--npu` | `--text-device npu` (+ `--text-attn-device npu`, `--vision-device npu`, `--vision-attn-device npu` as applicable) |
| `--npu-attn npu\|cpu` | `--text-attn-device npu\|cpu` |
| `--vision-attn npu` | `--vision-attn-device npu` |
| `--vision-attn fused\|blocked` | `--vision-attn-device cpu --vision-attn-kernel fused\|blocked` |
| `--npu-int8`, `--npu-int8-group N` | `--npu-dtype int8`, `--int8-group N` |
| `--pure-npu` | `--decode-device npu` |
| `--npu-decode` | `--decode-device npu --keep-cpu-weights yes` |
| `--no-npu` | `--device cpu` |
| `--backend` | `--cpu-backend` |
| `--attn fused\|blocked` | `--attn-kernel fused\|blocked` |
| `--f16` (gemma4) | `--dtype f16` |

## Files

| file | model |
|---|---|
| [`embeddinggemma2.md`](embeddinggemma2.md) | EmbeddingGemma 2 (text/image/video/audio embeddings), first-class |
| [`qwen3.md`](qwen3.md) | Qwen3-Embedding-0.6B |
| [`intent.md`](intent.md) | Qwen3.5-0.8B intent/query planner |
| [`gemma4.md`](gemma4.md) | Gemma 4 E2B-it chat |
| [`decision.md`](decision.md) | vllm-sr Decision 3.0 (d3-lite/d3-nano) typed decisions |
