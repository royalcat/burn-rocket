# Experiments

Measurements, design notes and verification records. These are developer and agent
material: they are dated, keep the flags that were current when each run was made, and
are not meant to be read top to bottom. Deployment commands live in
[`../recipes/`](../recipes/).

| file | contents |
|---|---|
| [`experiment-log.md`](experiment-log.md) | Qwen3-Embedding and the Qwen3.5 intent model: dev-host and board A/B, NPU offload, serving robustness, Vulkan probe, tiled attention, Q8_0-native int8, RoPE on the NPU |
| [`experiment-log-gemma.md`](experiment-log-gemma.md) | EmbeddingGemma 2 and Gemma 4: multimodal parity, Q8/NPU, QAT mobile, vision tower, memory rounds |
| [`embeddinggemma2-board.md`](embeddinggemma2-board.md) | EmbeddingGemma 2 board profile measurements (2026-10-31) and tuning notes |
| [`decision-d3.md`](decision-d3.md) | d3 (vllm-sr Decision 3.0): memory rounds, board results, int8 comparison, parity |
| [`clef-omni.md`](clef-omni.md) | Clef-Omni: request format, NPU placement, batching, media towers, gates and board results by phase |
| [`clef-pack.md`](clef-pack.md) | the Clef int8 pack format and packer |
| [`clef-oracle.md`](clef-oracle.md) | the Python layer-streaming oracle and the media-record notes |
| [`clef-phase0.md`](clef-phase0.md) | the expert-streaming format probe |

## Measured profiles (2026-10-31)

The profile names (balanced, full offload, speed) are historical: the recipes now describe
one preset per model plus a tuning table. These are the measurements behind them.

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
rows are derived from the dated experiment logs (`experiment-log.md`,
`experiment-log-gemma.md`) and are marked as such in their files; d3's row
is its own 2026-11-02 board window, below.

d3 has a single NPU stage (`text`), so its three profiles are literally the same
command (`--device npu --defer-towers`); the vision tower, the backbone
attention and the readout head stay on the CPU, and there is no decode. The row
is the 2026-11-02 memory window in [`decision-d3.md`](decision-d3.md) (`## Deployment
(current)`): 5 text questions / 6,546 prompt tokens at 147.3 s compute / 160.2 s
wall beside the live service (the 2026-11-01 idle-service window read 113 s NPU
vs 197 s CPU f32, and the contended CPU arm 265.6 s / 270.1 s); a text-only
process is 347 MiB anon resident plus 966.0 MiB of weights BO. Its server is
`POST /v1/systemone`, which takes images/videos as `data:` URLs only and has a
96 MiB body limit (one 32 MB video plus up to three 8 MB images, base64-encoded,
plus the JSON around them).


## The older logs use the pre-refactor flags

The measurement logs here predate the 2026-10-31 CLI refactor. Their
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

