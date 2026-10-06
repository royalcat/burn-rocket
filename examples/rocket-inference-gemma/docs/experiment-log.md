# EmbeddingGemma 2 — experiment log

Dev host: 32-thread x86_64, f32, `flex` backend, release build
(`cargo build --release -p rocket-inference-gemma --no-default-features`),
model `google/embeddinggemma-2` (1.49 GB safetensors, 1376 tensors).
Reference stack: `/mnt/hub/venvs/emb2` (transformers 5.19.0,
sentence-transformers 6.1.0, torch 2.14.0+cpu, torchvision 0.29.1+cpu).

Reference embeddings are produced by `tools/ref_embeddinggemma2.py`, which
reproduces the sentence-transformers pipeline directly on the transformers
model: `prompt prefix + text` → tokenizer → backbone (projected to 768-d) →
mean pooling → MRL truncation → L2 normalization (ST's order: pool, truncate,
normalize).

## 1. Text path

| case | tokens | dim | cosine |
|---|---|---|---|
| `what is the capital of france?` | 9 | 768 | 1.00000000 |
| same, `--prompt query` | 16 | 768 | 1.00000000 |
| `Paris is the capital and largest city of France.` (`document`) | 18 | 768 | 1.00000000 |
| 2594-token document (`data/one_long.txt`, `query`) | 2594 | 768 | 1.00000000 |
| same | 2594 | 256 | 1.00000000 |

Tokenizer ids match the HF tokenizer exactly (with and without prompt
prefixes, BOS/EOS included).

Architecture notes that the implementation encodes:

- 24 layers, hidden 512, 4 heads; sliding layers head_dim 256 / 2 KV heads,
  full layers (5, 11, 17, 23) head_dim 512 / 1 KV head.
- Bidirectional attention: sliding layers use the inclusive band
  `|q - kv| <= 512`; full layers are unrestricted. Attention scale is 1.0
  (QK-RMSNorm), value norm has no scale.
- RoPE: sliding θ=1e4, full θ=1e6 (per-layer `head_dim`).
- PLE: `per_layer_model_projection` (512→24×512) × 1/√512 → per-layer RMSNorm,
  gated into each layer's third residual block, then `layer_scalar`.
- Output: mean pooling over all tokens (prompt included), `embedding_projection`
  512→768, MRL truncation, L2 normalization.

Attention is computed in query chunks (`--attn-chunk`, default 1024) so the
score scratch stays bounded; sliding layers only materialize the band keys.

## 2. Vision path

Pipeline: aspect-ratio-preserving BICUBIC resize to the largest size that fits
the patch budget (280 soft tokens → 2520 patches, dims divisible by 48),
rescale to [0, 1], patchify channel-last, 2-D position ids (x, y), vision
encoder (16 layers, hidden 768, 12 heads, axial RoPE), k×k spatial mean
pooling ×√768, `embed_vision` (scale-free RMSNorm + 768→512 projection), then
scatter into the BOI/EOI placeholder slots.

The resize is a port of ATen's `upsample_bicubic2d_aa` uint8 kernel
(`_compute_indices_min_size_weights_aa`: scaled support, edge renormalization,
int16 fixed-point weights at the largest precision < 22), and was verified
bit-exact against `tvF.resize(..., BICUBIC, antialias=True)`.

| case | patches / soft tokens | cosine |
|---|---|---|
| PNG with the same decoded pixels as the reference | — | 1.00000000 |
| `data/cat.jpeg` (JPEG, mozjpeg decode) | 57×42 patches → 266 soft tokens | 1.00000000 |
| `data/cat.jpeg`, `task: search result \| query: <\|image\|>` | 266 + prompt | 1.00000000 |

JPEG decoding is the last piece that had to be fixed for exactness: zune-jpeg
(Rust `image`) vs libjpeg-turbo (PIL) differs by up to ±4 LSB on 5.1 % of
pixels (mean 0.07), which showed up as cosine 0.99987. `media::load_image` now
decodes JPEGs with **mozjpeg** (libjpeg-turbo) and falls back to `image` for
everything else; all image cases are then bit-exact (cosine 1.00000000). The
reference's own pixel-level resize difference (PIL BICUBIC vs torchvision on
uint8) is ±2 LSB and is covered by the ATen-exact resize port. Padding patches
are masked out of attention in the reference, so processing only the real
patches is exactly equivalent (verified by a garbage-padding experiment).

## 3. Audio path

Pipeline: 16 kHz mono → USM log-mel frontend (frame 320, hop 160, FFT 512,
semicausal left pad 160, periodic Hann, HTK mel 128 bins, `log(x + 1e-3)`,
frame mask) → subsample conv (2 × conv3x3/stride 2, LayerNorm+ReLU, 32×32
channels → 1024) → 12 conformer layers (chunked relative attention with
chunk 12 / left context 12, logit cap 50, clipped linears, light conv, two
feed-forwards) → output projection to 1536 → `embed_audio` (scale-free RMSNorm
+ 1536→512). Placeholder count is the twice-subsampled valid mask length; only
valid soft tokens are kept.

| case | mel frames / soft tokens | cosine |
|---|---|---|
| 5 s 16 kHz mono (`data/speech5s.wav`) | 499 / 125 | 1.00000000 |
| 30 s 16 kHz mono (`data/speech30s.wav`) | 2999 / 750 | 1.00000000 |
| same + `task: sentence similarity \| query: <\|audio\|>` | 125 | 1.00000000 |
| 44.1 kHz input (ffmpeg resampler vs librosa/soxr) | 125 | 0.99418 |

Bugs found on the way (both fixed): the relative-position `_rel_shift` pad
length is `context + 1`, and the local attention mask is strict
(`0 <= q - kv < 12`, not `<= 12`).

Clip bounds: the 120 `Gemma4ClippableLinear`s carry `input_min/max` and
`output_min/max` scalars in the checkpoint (480 tensors); they are read
separately into `ClipBounds` and applied as `clamp(input)` → linear →
`clamp(output)`.

## 4. Video path

Pipeline: 1 fps sampling (uniform overflow at 32 frames), frames extracted
with ffmpeg (rawvideo rgb24), each frame through the image pipeline with a
140-soft-token budget, BOI + `<|video|>` soft tokens + EOI per frame.

| case | sampled frames / soft tokens | cosine |
|---|---|---|
| `test.mp4` (4 s, 8 fps, 480×360) | 4 / 130 per frame, 520 total | 1.00000000 |

Frame indices follow the reference: `step = native_fps / fps`,
`num_sampled = max(1, int(duration * fps))`,
`index_i = min(total - 1, int(i * step))`, then a uniform linspace cap at
`max_frames`.

## 5. Performance and memory (dev host, f32, flex)

| case | ids | wall | tok/s | stages (attn/mlp/norms/ple) |
|---|---|---|---|---|
| 2587-token text | 2587 | 8.0 s | 322-326 | 3.9 / 2.8 / 0.2 / 0.9 s |
| image (cat.jpeg) | 270 | 0.70 s | 385 | 0.21 / 0.36 / 0.02 / 0.08 s |
| audio (5 s) | 129 | 0.34 s | 376 | 0.10 / 0.17 / 0.01 / 0.05 s |

- Model load: 4.0 s f32 (896 tensors applied; the 480 audio clip scalars are
  read separately).
- Resident after load: 2845 MiB anon (f32 weights ~2.9 GB); peak RSS during a
  run ~4.2 GiB including the 1.49 GB safetensors mmap.
- The text path is attention-heavy at long lengths (full layers have head_dim
  512 and no window).

## 6. Server

`serve` implements `/health`, `/v1/models`, `/v1/embeddings` (string or array
input, `dim`, `prompt`) and `/embed` (text/image/video/audio; paths or base64
data URIs). Smoke-tested on the dev host: text 9 tokens → 768-d, batch of 2 →
21 tokens usage, `dim 256`, image, audio and video all 200; unknown prompt →
400; a malformed `input` → 422; forwards are panic-contained (a panic cannot
wedge the model lock).

## 7. Reproduction

```sh
# text (compare with the HF reference)
B=$CARGO_TARGET_DIR/release/rocket-inference-gemma
M=/mnt/hub/models/embeddinggemma-2
$B embed --model-dir $M --prompt query --text-file data/one_long.txt --out /tmp/ours.json
/mnt/hub/venvs/emb2/bin/python tools/ref_embeddinggemma2.py embed \
    --prompt query --text-file data/one_long.txt --out /tmp/ref.json
python3 -c "import json,math;a=json.load(open('/tmp/ours.json'));b=json.load(open('/tmp/ref.json'));\
print(sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b))))"

# media (same pattern with --image/--video/--audio and the reference's
# image/video/audio subcommands)
```

## 8. Low-RAM mode (`--quant q8`)

Projection weights (text, vision and audio `Linear`s; the embedding table,
norms, scalars and position tables stay f32) are stored Q8_0-quantized
(symmetric int8, 32-value blocks, f16 scales) and dequantized per call, so only
the current layer's f32 weights are materialized.

| case | cosine vs the f32 reference |
|---|---|
| text 2594 tok | 0.999646 |
| image + query | 0.999610 |
| audio + query | 0.999861 |
| video | 0.999655 |

- Resident after load: **1216 MiB** anon (vs 2845 MiB f32); the load *peak*
  (~4.2 GiB) is unchanged because the checkpoint is materialized f32 before
  quantization and the 1.49 GB mmap is touched.
- Cost: 9.3 s vs 8.0 s for the 2594-token text forward on the dev host (the
  per-forward dequantization tax).
- f32-mode cosines are unaffected by the shared code path (all 1.00000000).

## 9. NPU offload (RK3588, 2026-10-07)

`burn-rocket` gained a bidirectional windowed attention op
(`burn_rocket::attention_window(q, k, v, h, kv, d, scale, softcap, window)`:
`|t - j| <= window`, `window < 0` = full bidirectional), because EmbeddingGemma
2's sliding layers need a band mask rather than the causal mask the existing op
builds.

The gemma example's `npu` feature (aarch64 only) offloads the **text backbone**:

- all 218 text projections (q/k/v/o, MLP, PLE block, PLE model projection,
  output projection) are packed into resident fp16 NPU buffers (0.25 GiB) and
  their f32 copies dropped; the loader registers each packed weight by
  `ParamId`, so `lin()` routes any registered weight to `burn_rocket::matmul`
  with no model-code changes;
- attention runs on the NPU (`--npu-attn npu`, default) through
  `attention_window`: sliding layers get the 512 band, full layers no mask;
  `--npu-attn cpu` keeps the chunked CPU attention;
- vision and audio towers, all norms/RoPE/gating and the embedding table stay
  on the CPU;
- `--npu` combines with `--quant q8` (quantize first, then pack the dequantized
  text weights), which is the lowest-memory configuration.

Board `rock-5b-plus.lan` (8 cores, 4 threads pinned to the A76s, `taskset -c
4-7`), 2587-token text, busy board (loadavg ~7, other workloads running, so
wall numbers are a lower bound on an idle board):

| config | forward wall | tok/s | user CPU |
|---|---|---|---|
| `--quant q8` (CPU) | 34.46 s | 75.1 | 94.0 s |
| `--quant q8 --npu --npu-attn cpu` | 24.46 s | 105.8 | 75.9 s |
| `--quant q8 --npu` | 21.39 s | 120.9 | 62.7 s |

The last row: 242 NPU calls, 1.67 s conversion, 11.67 s inside the NPU library,
8.05 s flex + overhead. Net effect vs the CPU baseline: **1.61x faster and 33 %
less user CPU**.

Numerics (board NPU+q8 vs the HF f32 reference):

| input | cosine |
|---|---|
| 9-token text | 0.999944 |
| 2587-token text | 0.999645 |
| image + query | 0.999606 |
| audio + query | 0.999861 |
| video (4 frames) | 0.999157 |

The cost is dominated by `--quant q8` (the dev-host q8 numbers are the same to
within 5e-5); the NPU adds almost nothing. Resident with `--quant q8 --npu`:
1068 MiB anon (vs 1216 q8-only, 2845 f32). The load peak is unchanged (~4.2 GiB:
the checkpoint is materialized f32 before quantizing/packing).

Notes: `--npu` alone (f32 CPU towers) is untested on the board; the q8+npu
combination is the validated deployment configuration. NPU attention keeps the
full `[n, n]` score matrix host-side, so long contexts (8k) need a memory check
before deployment. The board was busy during the A/B; re-run on an idle board
for headline numbers.

## 10. Negative result: f16

`--dtype f16` loads and runs (1508 MiB resident vs 2845 MiB f32) but is
numerically broken for this architecture: the reference computes RMSNorm,
softmax, PLE and the attention score matrices in f32, and in f16 the cosine
against the f32 reference drops to 0.984 (text, 2594 tok) and 0.700 (image,
JPEG), with an audio-path failure. The CLI now rejects `--dtype f16` with a
clear message. A real low-RAM mode needs Q8-resident weights dequantized per
forward (the `rocket-inference` `--quant q8` pattern), not f16.

## 11. Deferred

- Q8/low-RAM mode (the f32 resident model is 2.9 GB).
- NPU offload of the text backbone (RK3588; needs a board round).
- A libjpeg-turbo-parity decoder to close the JPEG gap.
