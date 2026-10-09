# EmbeddingGemma 2 — experiment log

Dev host: 32-thread x86_64, f32, `flex` backend, release build
(`cargo build --release -p rocket-inference --no-default-features`),
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
B=$CARGO_TARGET_DIR/release/rocket-inference
M=/mnt/hub/models/embeddinggemma-2
$B gemma embed --model-dir $M --prompt query --text-file data/one_long.txt --out /tmp/ours.json
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

## 11. Gemma 4 E2B-it text generation (2026-10-07)

Target: `google/gemma-4-E2B-it` (10.25 GB BF16, 5.12B params: ~4.3B text of
which 2.35B is the PLE token table, plus the round-1 vision/audio towers).
Reference: transformers 5.19.0 (bf16) with the checkpoint's chat template and
greedy decoding; the checkpoint was built with 5.5.0.dev0 and its KV-sharing
path was reworked since, so the pinned reference is the gate.

Architecture (all verified against the config and 5.19 source):

- 35 causal layers, hidden 1536, 8 q heads, 1 KV head, head_dim 256 (sliding,
  window 512) / 512 (full layers at 4, 9, ..., 34), attention scale 1.0;
- layers 15-34 **share K/V** with the last non-shared layer of their type
  (13 sliding, 14 full) and use double-wide (12288) MLPs; their k/v weights are
  present in the checkpoint but ignored;
- PLE: `embed_tokens_per_layer` [262144, 8960] x sqrt(256), blended with the
  context projection as `(norm(ctx) + token) / sqrt(2)`;
- full layers use proportional p-RoPE (partial factor 0.25: 64 of 256
  frequencies, tail zero); sliding layers use default RoPE (theta 1e4);
- tied LM head, `tanh(logits / 30) * 30` soft cap, eos [1, 106, 50];
- chat template `<bos><|turn>role\n...<turn|>\n...<|turn>model\n`, optional
  `<|think|>`.

Verification (dev host, f32, greedy):

| case | prompt tokens | generated | vs HF |
|---|---|---|---|
| `What is the capital of France?` | 16 | 9 | identical ids |
| system + user (`Name three colors.`) | 24 | 24 | identical ids |
| multi-turn (2+2) | 34 | 9 | identical ids |
| 631-token document | 631 | 16 | identical ids |

Template rendering is token-exact for system, multi-turn and `enable_thinking`
cases. The chat server returns the same text and usage counts as the reference.

Loading (the checkpoint does not fit as f32: 20.5 GB of weights plus the mmap):
a store adapter converts each tensor to its final dtype while it is applied —
the two token tables become f16 (exact: bf16 sources have fewer mantissa bits),
projections stay f32 by default, and `--f16`/`--quant q8` select f16 or Q8_0.
The tied LM head is materialized once as a transposed copy for the logits path.

| mode | resident | load | prefill (631 tok) | decode |
|---|---|---|---|---|
| f32 | 12494 MiB | 30 s | 7.5 s | 2.2-2.8 tok/s |
| `--f16` | 8923 MiB | 20 s | 11.3 s | 3.5 tok/s (short), 1.9 tok/s (631-token context) |
| `--quant q8` | 7290 MiB | 17 s + 24 s quantize | 13.8 s | 0.09 tok/s |

Notes: decode is memory-bandwidth-bound (f32 reads ~17 GB of weights per
token); f16 is token-identical on the short and multi-turn cases but flipped a
near-tie on the 631-token case, so f32 remains the parity mode; Q8 decode is
dominated by flex's scalar per-call `dequantize` (flex has no int8 GEMM), so q8
is a memory-only mode here.

### Multimodal prefill (2026-10-07)

`gemma gen` and the served chat endpoints accept `<|image|>` / `<|audio|>` placeholders in the rendered
conversation; `inputs::prepare` expands them (BOI/BOA + soft tokens + EOI/EOA)
and encodes the media with the checkpoint's own towers, then the text model runs
over the scattered embeddings. The PLE token-table lookup uses the media-pad ids
(pad 0 at the placeholder slots), exactly like the reference; the PLE context
term uses the scattered embeddings.

The E2B vision tower needed clip bounds (`use_clipped_linears: true`, unlike
EmbeddingGemma 2): the loader reads the 7 x 16 `Gemma4ClippableLinear` scalar
sets and the vision tower applies them; the audio tower is byte-for-byte the
EmbeddingGemma 2 tower (same keys, same clip bounds).

| case | prompt | result |
|---|---|---|
| image (`cat.jpeg`, 280 soft tokens) | "What is in this image? `<\|image\|>`" | 284-token prompt, 24/24 generated tokens identical to HF bf16 |
| audio (`speech5s.wav`, 125 soft tokens) | "What do you hear in this audio? `<\|audio\|>`" | 145-token prompt, 24/24 identical to HF **f32** |

The audio case diverges from the bf16 reference at token 20 ("vehicle" vs
"car"); the f32 reference matches our output exactly, so it is a bf16 near-tie,
not a numerical bug (the same caveat as the `--f16` text mode). Memory with the
towers loaded: f32 ~16.5 GiB resident (text 12.5 + towers ~4).

The served chat endpoints accept the same media as OpenAI content parts: `image_url` with a
data URI or path, `input_audio` with base64 `data` + `format` (at most one of
each per request). The placeholders are inserted by the server, so clients send
plain text. Smoke-tested on the dev host: the image request returns the CLI's
284-prompt/24-completion output and the audio request the 145/24 output; the
`/v1/models` id defaults to the generation model's directory name.

### NPU prefill (implemented 2026-10-07; board measurement pending)

`--npu` on the aarch64 build packs the generation model's *used* text
projections (q/k/v/o, MLPs, PLE; the KV-shared layers' unused k/v are skipped)
into resident fp16 NPU weights and runs the prefill pass on the NPU:

- matmuls through the same `ParamId` registry as the embedding round (`lin`
  routes registered weights to `burn_rocket::matmul`);
- attention through the new `burn_rocket::attention_causal_window` op: causal
  for full layers, causal + sliding window (`t - window < j <= t`) for sliding
  layers. The mask builders moved to `burn-rocket/src/masks.rs` and are unit
  tested on any host (`cargo test -p burn-rocket`: causal, band and
  causal-window semantics);
- **decode stays on the CPU**: `layers::set_prefill_mode` is set only around
  `GenRoot::prefill`, and the CPU f32 copies are kept (`keep_cpu`), because NPU
  matmuls pad M to 256 and a single decode row would waste that work.

Board attempt (2026-10-07): run under a cgroup guard
(`systemd-run --scope -p MemoryMax=8G -p MemorySwapMax=4G`) so only our process
could be killed. It was OOM-killed during load (global OOM, `anon-rss` 2.8 GB
at the time, `total-vm` 15.4 GB): `rock-5b-plus` was holding ~11 GB with other
workloads and has only ~4.5 GB available (8 GB zram, already 4.8 GB used), so
the ~9.5 GB f16 working set (8.6 GiB text + ~0.9 GiB towers) plus the 2.1 GiB
of packed fp16 NPU weights cannot fit alongside them. The guard contained the
kill to the scoped process; the board's other services were unaffected.

On an idle board the configuration fits (15.8 GB total RAM vs ~9.5 GB host +
2.1 GiB NPU + system), so the A/B only needs a free board. The model is
deployed at `/root/models/gemma-4-E2B-it/` and the aarch64 `--features npu`
binary at `/root/rocket-inference/` (the merged app; `gemma gen`); the same NPU machinery measured 1.61x
on prefill for the embedding model (log §9), and decode throughput is unchanged
by design.

### QAT mobile checkpoint (`google/gemma-4-E2B-it-qat-mobile-transformers`)

2.46 GB, 2780 tensors: packed INT2/INT4 weights (2/4 values per byte, low bits
first, unsigned shifted to signed), INT8 weights for the vision tower and the
PLE gate/projection, `weight_scale` `[N, 1]` per-output-channel scales,
`embedding_scale` `[rows, blocks]` block scales for the token tables (35 blocks
of 256 for the PLE table), plus SRQ (static range quantization) activation
scales. The checkpoint has `use_clipped_linears: false` for both towers and
omits the KV-shared layers' k/v weights (the reference ignores them too).

Implementation (`src/qat.rs` + a load adapter): the `quantization_config` regex
table selects the bit width per module (look-around patterns need
`fancy-regex`), the scale tensors are pre-read into per-module records, and the
adapter unpacks + dequantizes each packed tensor while the store applies it
(no conversion pass, no temp files). SRQ scales are registered per weight
`ParamId` from a module-path visitor and applied around the matmul in `lin()`;
rounding is ties-to-even like `torch.round`. `NO_SRQ=1` and
`DUMP_PARAM=<substr>` are debug hooks.

Verification (dev host, f32 unless noted):

| check | result |
|---|---|
| dequantized weights vs HF | **bit-exact** (4-bit `q_proj`, 2-bit MLP `gate_proj`, 4-bit PLE table: `max|d| = 0`) |
| short prompt, greedy | token-identical to HF f32 (9/9) |
| system+user prompt | token-identical (7/7) |
| 631-token prompt | 9/12 (f32), **12/12** (`--f16`) |
| image prompt (280 soft tokens) | 16/24 (f32) |
| control: bf16 checkpoint, our logits vs HF | `|d| <= 1e-4` (essentially exact) |

The QAT-specific residual (logits within 0.01-0.7 of ~27) comes from SRQ: it
rounds activations to an int8 grid, so the f32 accumulation-order difference
between flex and torch occasionally lands on the other side of a rounding
boundary and shifts an activation by a full quantum. The first long-prompt
divergence is a top-2 swap with a 0.23 logit margin, and the reference itself
flips at the same prompt between bf16 and f32. SRQ is essential, not optional:
with `NO_SRQ=1` the top-3 tokens diverge immediately (24.5 vs 21.3 logits).

The PLE table stays packed (`PackedTable` in `src/qat.rs`): its rows are
dequantized on lookup, which is bit-identical to dequantizing the whole table
and costs 1.13 GiB instead of 8.75 GiB (f32) / 4.4 GiB (f16). The parameter is
stubbed to `[1, 1]` before loading, so the dequantized table is never allocated.

| mode | resident | load | short decode | 631-token prompt |
|---|---|---|---|---|
| f32 (f32 projections) | 12.0 GiB | 56 s | 3.9 tok/s | prefill 0.8 s, 9/12 tokens |
| `--f16` | 6.6 GiB | 56 s | 4.9 tok/s | prefill 22.2 s, 3.5 tok/s, 12/12 tokens |
| `--quant q8` | 6.1 GiB | 61 s | 0.07 tok/s | memory-only |

Board attempt (2026-10-07): the QAT checkpoint (2.46 GB) and the aarch64 npu
binary are deployed (`/root/models/gemma-4-E2B-it-qat-mobile/`), but the f16
configuration was OOM-killed at 7.6 GB anon while `rock-5b-plus` was holding
7.4 GB with other workloads plus 2.6 GB of shared memory. All three attempts
were contained to their systemd scope; the board's services were unaffected.
With an idle board the (then) 6.6 GiB working set fits comfortably; the
2026-10-09 memory round (§13) brings the text-only f16 profile to 3.7 GiB anon
plus reclaimable table pages, so the busy-board constraint is materially
relaxed (board re-run pending).

## 12. Performance round (2026-10-08): canonical tiling, int8, chunked attention, fused glue

Status: implemented and numerically validated on the board; the timing A/B is
pending (the board was running the production emb2 server — `openviking-embed-1`,
NPU + cores 4-7 — throughout, so all numbers below are correctness/memory only).

### 12.1 Canonical tiling (small-M)

`burn_rocket::init` now creates the context with
`rocket_ctx_create_ex(threads, ROCKET_CTX_TILING_CANONICAL)`: the resident
weights are M-independent down to `M = 4`, so a request no longer pads every
matmul to 256 rows. `ROCKET_CTX_CANONICAL=0` restores the legacy pad-to-256
behaviour in the same binary (the A/B switch).

| check | result |
|---|---|
| 2587-token text, `--quant q8 --npu`, canonical off vs on | **cosine 1.000000000, max\|d\| = 0** (bit-identical) |
| 9-token text, same | **cosine 1.000000000, max\|d\| = 0** |
| long text vs HF f32 (off) | 0.999646657 (the logged q8 value) |

The tiling keeps the `M = 256` K/N tiling, so per-row results are unchanged
bit-for-bit; only the discarded padding rows disappear. Also benefits qwen3
short queries and the intent model's `--pure-npu` M = 1 decode.

### 12.2 Resident int8 text projections (`--npu-int8`)

`burn-rocket` gained the library's resident group-wise int8 path
(`rocket_i8_ctx_create` / `rocket_i8_weights_pack_gw` /
`rocket_matmul_int8_prepacked_gw`): symmetric int8, group configurable via
`--npu-int8-group` (default 32), one weight scale per output channel per K-group
(`b_scale[N, K/group]`, verified against the library's `rocket_prepacked_int8.c`),
one activation scale per row per K-group; the A rows are padded to `M % 4`
caller-side. The codes live on the NPU permanently; the host drops the f32
copies like the fp16 pack does. The `--npu-int8` flag (gemma emb2; requires
`--npu`) selects it; `--quant q8` is ignored with it (the int8 pack quantizes
straight from f32).

The torch probe (`tools/w8a8_probe.py`) ran before any Rust, simulating the
library's conventions on the 2594-token text: per-32-group weights + per-32-group
activations keep cosine **0.99986** vs f32, while whole-K (per-row) activation
scales lose much more (0.99964). Board results match the probe:

| config (2587-token text) | cosine vs HF f32 | resident text weights |
|---|---|---|
| `--npu` (fp16 weights) | 0.999999749 | 0.25 GiB |
| `--npu --npu-int8` | **0.999850521** | 0.11 GiB |
| `--quant q8 --npu` (the deployed config) | 0.999646657 | 0.25 GiB |
| 9-token text, int8 vs fp16 | 0.999916317 | — |

So int8 is *more* accurate than the deployed q8 path at the default group. Its
speed is another matter: the group-wise path is readback-bound, and the group
width trades accuracy for speed — see the sweep in §12.5 (g32 is 2.4x slower
than fp16; g512 matches fp16 speed at 0.9992).

### 12.3 Chunked banded attention for the sliding layers

20 of 24 text layers are sliding (`|q - kv| <= 512`). They now run in query
chunks (`--attn-chunk`, default 1024) against only the keys near the chunk
(`[q0 - w, q1 + w)`), through the new
`burn_rocket::attention_window_block` op and the
`masks::build_window_mask_block` band mask; the full layers are unchanged. The
library's FA already took separate `n_tokens`/`n_kv`; the wrapper previously
asserted them equal.

| check | result |
|---|---|
| 2587-token text, `--attn-chunk 4096` (full) vs `1024` (chunked) | cosine 0.999999742, max\|d\| 8.6e-5 |
| 2587-token, chunked vs HF f32 | 0.999999749 (same as the full path) |
| 7764-token text, chunk 1024 vs 2048 | cosine 0.999999867 |
| 7764-token, chunked vs HF f32 | 0.999667372 (q8 weight error, as at 2.6k) |

Memory is the bigger win: the chunked 7764-token run peaks at **1068 MiB anon**
(flat vs 2.6k), while the full-matrix path at the same length was OOM-killed
inside a 3 GB cgroup (anon-rss 2.1 GB, `Memory cgroup out of memory`). Chunked
attention is what makes 8k deployable on the board; it also cuts the sliding
layers' QK/PV, host softmax and score traffic.

### 12.4 Fused CPU glue (NPU builds)

`burn-rocket` gained four host kernels (`src/host.rs`, unit-tested on any host):
fused `gelu_approximate(gate) * up`, weighted/scale-free RMSNorm, and rotate-half
RoPE, each a single rayon pass instead of flex's several single-threaded scalar
passes (`gelu_approximate` alone is ~9 passes with a `tanh`/`powf` per element).
The emb2 model routes its call sites through `gemma::layers::{gelu_mul, rms_norm,
rms_norm_noscale, rope_apply}`; `--npu` enables them (`ROCKET_GLUE=0` restores
the composite ops, `ROCKET_GLUE=1` forces the kernels in CPU modes for the A/B).

| check | result |
|---|---|
| `--quant q8` CPU mode, glue on vs off | cosine 1.000000000, max\|d\| 1.4e-7 |
| `--npu`, glue on vs off | cosine 0.999999537 (f16 attention rounding) |
| `--npu` (glue on) vs HF f32 | 0.999999749 (unchanged) |

Pure-CPU modes keep the composite ops, so their bit-parity with the HF reference
is untouched.

### 12.5 Timing (board, 2026-10-08)

All runs on the shared board (load avg 19-24 from docker churn, `tstor-scan` and
the production `openviking-embed-1` service, which shares cores 4-7 and the NPU),
so absolute walls are lower bounds; each comparison was run back-to-back. The
spread of identical repeats is ±20 %: the q8+npu 2587-token run ranged
12.6-19.5 s across the round.

Canonical tiling (P1) — short input is where it pays; the long input does the
same padded work either way (bit-identical output, the 12.6 vs 17.7 s gap is
board noise):

| 16-token text, `--quant q8 --npu` | rep 0 | rep 1 | rep 2 |
|---|---|---|---|
| `ROCKET_CTX_CANONICAL=0` | 0.73 s | 0.58 s | 0.60 s |
| canonical (default) | 0.41 s | 0.31 s | 0.28 s |

Int8 vs fp16 (P2, 2587 tokens, no q8): fp16 17.15 s (attn 10.18, mlp 3.78,
ple 0.96; NPU 10.96 s) vs **int8 g32 41.85 s** (attn 18.66, mlp 17.46, ple 3.44;
NPU 35.54 s). The group-wise path pins the K-tile to a divisor of the group, so
g32 makes 448 K-tiles at K=14336 and reads every tile's full `M x N` int32
partial back to the host (~1.5 GB per MLP layer at 2587 tokens) — it is
readback-bound, not compute-bound, and 2.4x slower than fp16 at g32. Wider
groups cut the readback linearly (`--npu-int8-group`):

| group | wall (pass 1 / 2) | NPU time (pass 1 / 2) | cosine vs HF f32 |
|---|---|---|---|
| fp16 | 15.23 / 15.93 s | 9.33 / 9.25 s | 0.999999749 |
| 32 | 38.96 / 34.64 s | 32.63 / 29.21 s | 0.999850521 |
| 128 | 22.57 / 18.29 s | 15.68 / 12.69 s | 0.999682960 |
| 256 | 17.66 / 15.07 s | 11.18 / 10.25 s | 0.999553430 |
| 512 | 13.89 / 14.03 s | 8.86 / 8.80 s | 0.999204856 |

The NPU time tracks `K / group` almost exactly (32.6 -> 8.9 s from g32 to g512),
confirming the readback diagnosis. At g512 the int8 path edges past fp16
(13.9-14.0 vs 15.2-15.9 s) but at 0.9992 cosine, well below the deployed q8's
0.99965; g256 is fp16-speed within noise at 0.99955. The int8 compute is ~2x
fp16, so the readback still eats most of the theoretical win (an ~4.7 s NPU
floor plus ~2-3 s of readback).

Chunked banded attention (P3, 2587 tokens): full (chunk 4096) 13.90 s
(attn 7.61) vs chunked 15.84 s (attn 9.73) — chunking costs ~14 % at 2.6k. Its
value is memory: at 7.7k tokens the chunked path ran in 1068 MiB anon while the
full-matrix path OOM-killed a 3 GB cgroup (§12.3).

Fused glue (P4) — both modes ~12 % faster, the NPU build mostly in the CPU
glue:

| 2587-token text | glue off (`ROCKET_GLUE=0`) | glue on (default) |
|---|---|---|
| `--quant q8` (CPU) | 35.31 s (mlp 12.90, ple 2.94) | 31.16 s (mlp 8.74, ple 1.98) |
| `--npu` | 19.47 s (flex+overhead 8.29) | 17.21 s (flex+overhead 3.50) |

Short-input latency (P5, 16 tokens): q8+npu 0.25-0.34 s, fp16 `--npu`
0.41-0.52 s, int8 g32 0.75-1.01 s, g128 0.47-0.53 s, g256 0.31-0.47 s. 8k text
(P5b, int8 g32 + chunk 1024, 7757 tokens): 117.73 s / 65.9 tok/s (attn 57.47,
mlp 45.10; NPU 95.76 s) — the q8+npu configuration ran the same input in 55.04 s
(§12.3); the gap is the g32 readback again.

Deployment verdict: fp16 `--npu` stays the speed configuration (near-lossless
0.9999997 at the same wall as the deployed q8+npu); q8+npu remains the memory
choice (1068 MiB vs ~2.3 GiB resident after pack); int8 is the numerics/memory
middle — g32 beats q8's numerics at 2.4x the time, and wider groups buy speed
back at a monotone numerics cost (`--npu-int8-group`).

## 13. QAT mobile memory round (dev host, 2026-10-09)

Goal: the QAT mobile f16 profile (6.6 GiB anon) OOM-killed a busy
`rock-5b-plus` at 7.6 GB, so text generation had to fit alongside the other
board workloads. The round removes eager, anonymous copies from the load path:

0. **Stop forcing the stubbed tables' lazy initializers** (the biggest win): the
   pre-load stubs used `Param::map`, which materializes the parameter before
   mapping — writing an 8.75 GiB f32 PLE table (and the 1.5 GiB token table)
   and dropping it. That transient was the real load peak (10.3 GiB anon
   measured) and ~30 s of the load. `Param::from_tensor` replaces the
   parameter without consuming the old one: load peak 10.3 -> 4.7 GiB, load
   51 -> 22 s on the dev host. The same transient explains the original board
   OOM at 7.6 GB anon.
1. **Pack the token table too** (`embed_tokens`): the 2-bit
   `embedding_quantized` is gathered per forward (the same path as the PLE
   table) and the transposed LM head is built from it once — one 0.77 GiB f16
   copy instead of the table plus the head (1.5 GiB). Gathered rows are rounded
   to the dtype the eager table stored (f16/bf16; exact in f32 mode), which
   keeps the validated numerics: the first attempt gathered exact f32 and
   flipped a near-tie at token 10 of the 631-token prompt.
2. **File-backed packed tables**: both tables are views into a `memmap2` map of
   `model.safetensors` (`qat::PackedFile`), so their ~1.2 GiB are clean page
   cache instead of anonymous memory. Row gathers touch only the rows used;
   the f32 scales are copied out (the format gives no alignment guarantee).
3. **Deferred towers** (`--defer-towers`): the vision/audio towers load in a
   separate store pass on the first media request (~4 s) instead of at startup.
   They always load f32 (the validated multimodal mode): loading them in the
   text model's dtype made every media request panic on dtype-mixed ops
   (pre-existing — only the f32 mode had ever exercised media). `lin()` now
   checks the weight dtype before the f16/bf16 cast path (mixed f16 text +
   f32 towers), and the SRQ visitor registers the towers' 232 scales
   explicitly (they sit behind a `#[module(skip)]` field).

Measured (dev host, peak `RssAnon` over the whole run):

| config | peak anon | note |
|---|---|---|
| f16 baseline (pre-round) | ~10.3 GiB peak, 6630 MiB at loader end | forced PLE init |
| f16 `--defer-towers` | **4721 MiB** | loader-end 3736 MiB, load 22 s |
| f16, eager towers | 6536 MiB | towers +1.9 GiB |
| f32 `--defer-towers` | 9240 MiB | |
| q8 `--defer-towers` | 3758 MiB | 0.07 tok/s (memory-only) |

The towers add ~1.9 GiB when loaded; text-only serving (the busy-board case)
never pays it. Parity: f16 short output is token-identical to the pre-round
build; the 631-token prompt is **16/16 identical to the HF bf16 reference**
(the pre-round build matches it too), and the image prompt is **8/8 identical
to HF** with the lazily loaded towers (media works in f16 mode now; it used to
panic on dtype-mixed ops). Top-8 logits drift only at GEMM
accumulation-order scale (max |d| 0.15, no top-1 change; f32 mode 1e-5).

Open: the board re-run (guarded scope) and the NPU-decode pre-gate —
`ROCKET_NPU_DECODE=1` routes decode matmuls to the packed NPU weights with the
CPU copies kept, so the decode speed can be measured before switching to
`keep_cpu=false` (pack-and-drop, decode stays on the NPU).

## 14. Deferred

- QAT mobile board re-run (a guarded scope; the f16 text-only profile is now
  3.7 GiB anon plus ~1.2 GiB reclaimable table pages) and the
  `ROCKET_NPU_DECODE=1` decode-speed pre-gate (§13).
- 30k-token text with chunked attention (the old OOM point); needs the board.
- Production `openviking-embed-1` redeploy: **done** — `b09c082` on 2026-10-08
  (canonical tiling, chunking, fused glue) and `4a8657b` on 2026-10-09 (burn 0.22
  + native bf16). The deployed config stays `--dtype f32 --quant q8 --npu
  --npu-attn npu --attn-chunk 1024`: the native-bf16 default loses ~20–25 % to
  q8 on the board (log §18).
- Exposing `--npu-int8` / `--npu-int8-group` on `serve` (the server loads fp16
  text weights today).
- int8 for the vision/audio towers (small share of the work; text-only today).
