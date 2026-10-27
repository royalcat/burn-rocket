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

`wall` here starts after `build_inputs`, so the image and audio rows time only
the **text pass** of an already-encoded input: they exclude the vision tower,
the audio frontend and media decode. The tower dominates a media request — see
§18 (5.1 s f32 on this host for `cat.jpeg`, 25-27 s on the board, against
0.8-2.4 s of text forward).

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
| dequantized weights vs HF | **bit-exact** (4-bit `q_proj`, 2-bit MLP `gate_proj`, 4-bit PLE table: `max\|d\| = 0`) |
| short prompt, greedy | token-identical to HF f32 (9/9) |
| system+user prompt | token-identical (7/7) |
| 631-token prompt | 9/12 (f32), **12/12** (`--f16`) |
| image prompt (280 soft tokens) | 16/24 (f32) |
| control: bf16 checkpoint, our logits vs HF | `\|d\| <= 1e-4` (essentially exact) |

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

## 14. Idle-memory round: deferred towers + heap trims (2026-10-09)

Goal: the served emb2 backend (`openviking-embed-1`, text-only in production)
held ~1.7 GiB anon on the dev host and **2183 MiB on the board** while idle,
and kept growing with every long request. Three findings shaped the round:

1. **The tokenizer is a hidden 467 MiB peak.** `gemma tokenize` alone peaks at
   466.9 MiB on this checkpoint (262144 vocab / 514906 merges), and glibc kept
   ~270 MiB of that parse after the model load (A/B: setting
   `MALLOC_MMAP_THRESHOLD_=131072` alone dropped the served steady from 1694
   to 1423 MiB).
2. **Idle creeps per request.** Each request left 25-55 MiB resident (the
   allocator retains the request's high-water mark; the NPU full-attention
   layers read back a 4.2k-token score matrix of ~280 MB per layer). On the
   board the *same* config read anywhere from 0.75 to 2.3 GiB (§18).
3. **The towers are ~550 MiB of text-only dead weight.** The Q8 vision+audio
   towers load eagerly for `/embed`, which production never calls.

Changes (commit `b4634c9`):

- `--defer-towers` (gemma CLI + `serve`, emb2; mirrors the Gemma 4 flag): the
  towers live behind `Emb2Model.towers: Option<Towers>` (`#[module(skip)]`)
  and load in a towers-only store pass — eagerly by default, or on the first
  media request. A text-only load now applies **413 tensors (was 896) in ~1 s**
  instead of ~4 s.
- Heap release: `util::mem::{trim_heap, trim_after_request}` — `malloc_trim(0)`
  after serve startup (releases the tokenizer parse's retired heap) and after
  every request (`ROCKET_TRIM=0` disables).

Dev host (flex, q8+f32, `data/one_long.txt` = 2594 tokens; RssAnon):

| config | after load | steady served | 3x 6.6k-token requests |
|---|---|---|---|
| before (q8, eager towers) | 1216 | 1694 | +25-55 MiB per request, never released |
| new, eager towers | 1217 | 1306 | flat |
| new, `--defer-towers` | 663 | **754** | 771-831 (flat) |

Media works unchanged in deferred mode: the first `/embed` (image or audio)
loads the towers in 7.3-7.5 s (+550 MiB, one-time, Q8-resident). Eager and
deferred towers are numerically identical — text (q8+f32) matches the
previous build **bit-for-bit** (max|d| = 0), image/audio cosines eager-vs-deferred
are **1.000000000** (max|d| = 0), and the served `/embed` responses are
bit-identical between the two modes.

Board (`openviking-embed-1`, text-only; image `4a8657b` -> `b4634c9` with
`--defer-towers` added; the board was busy — `satellite-deposit` at ~7 GiB and
>100 % CPU — so latencies are lower bounds, memory is exact):

| RssAnon | before (`4a8657b`, 33 h uptime) | after (`b4634c9`) |
|---|---|---|
| idle | 2183 (steady over the 33 h, §18) | **605** (at boot) |
| 6-token request | 2183.1 (log) | 606 (0.50-0.55 s) |
| 980-token request | — | 708 (6.0 s, 163 tok/s) |
| 3767-token request | — | 1179-1186 (24-25 s, 150-159 tok/s) |
| next request (6 tok) | — | 781 |
| `docker stats` idle | 3.205 GiB | 1.552 GiB |
| NPU BO shmem | 1081 MiB | 1330 MiB (post-traffic) |

The +470 MiB long-request step is the glibc arena high-water of the 3.7k-token
forward (the PLE tensor `[1, n, 24, 512]` f32 alone is 185 MB); the *next*
request's trim releases it, so the steady state under traffic is ~0.8 GiB vs
2.2 GiB before. Forcing `MALLOC_MMAP_THRESHOLD_=131072` was tried on the board
as a fix and reverted: memory barely moved (1147 vs 1179 MiB at 3767 tokens)
while every request got ~60 % slower (6.0 -> 9.9 s at 980 tokens, 23.7 -> 40.7 s
at 3767), so the default allocator behavior stays. Text numerics are unchanged
(the dev-host bit-identical checks above; the towers and trims do not touch the
text path). The deployed service never calls `/embed`: if it ever does, the
towers load lazily (+~550 MiB, Q8) and stay resident.

## 15. Scratch-memory round: chunked full layers + bucketed/trimmed scratch (2026-10-21)

The pure-Rust driver's 8k footprint was dominated by two length-scoped caches:
the four **full-attention** layers (unchunked `s × s` scores/probs) and the
matmul per-`(M,K,N)` scratch cache (one exact-`M` slot set per request length).
The deployed service needed `--defer-towers` and still peaked at 6.68 GB cgroup.

Code changes (no model math change):
- **Full layers now use the chunked path** like the sliding layers: a query
  chunk attends to every key (`window = -1`, no mask) instead of materializing
  `[s, s]`. `src/ext.rs` gained `MaskMode::NoneBlock` so a no-mask block may have
  `n_q != n_k`; `embeddinggemma/model.rs` chunks every layer when `s > chunk`.
- **Per-geometry attention contexts** (`Engine.fa: Vec<FaState>`, max 4): emb2's
  sliding (`head_dim 256`, 2 KV heads) and full (`512`, 1 KV head) layers used to
  recreate the whole FA context on every geometry switch, reopening 4 fds and
  discarding the scratch. Each geometry now keeps one context and a two-entry
  scratch cache (drop-before-allocate, trimmed to the last request's shapes).
- **Matmul scratch capacity classes** (`M_CLASSES`, ≤25 % padding) with the live
  `M` planned per call, so padded rows are never computed; lazy `out_all`/`wt_all`
  (the KACC/resident paths never touch them); `ROCKET_SCRATCH_MB` byte budget
  (default 1024, 0 = unlimited) with free-before-allocate LRU; `trim_scratch()`
  called from the existing per-request `trim_after_request` keeps the just-used
  set and drops earlier requests' sets (`ROCKET_TRIM=0` disables both).
- `ROCKET_MEM_LOG=1` prints the live split (weights / matmul BO+host / attention
  BO+host, top slots and worker shapes) after load and per request.

Board A/B (second `serve` on :8399 against the live pre-round binary on :8383,
same GGUF Q8_0, `--defer-towers --npu --npu-attn npu --attn-chunk 1024`,
cores 4-7; the board was busy — deposit ~4.2 GiB and load >10 — so timings are
indicative, memory is exact):

| input | pre-round (live) | new | cosine |
|---:|---|---:|---:|
| 1,532 tok | 8.42 s / 181.9 tok/s | 6.89-7.02 s / 218-222 tok/s | 1.0000003 |
| 6,122 tok | 44.86 s / 136.5 tok/s | 39.23 s / 156.1 tok/s | 1.0000000 |
| 8,192 tok | not re-run (board tight) | 57.7-62.1 s / 132-142 tok/s | — (6,122 exact) |

Driver scratch at 8,192 tokens (new, `ROCKET_MEM_LOG`): weights 327 MiB, matmul
681 MiB BO + 300 MiB host, attention 289 MiB BO + 270 MiB host = **1.27 GiB BO +
0.56 GiB host** (the pre-fix Rust image held 2.8 GB shmem at 8k plus anon; the C
image 2.1 GB shmem). Peak `RssAnon` 1.49-1.56 GiB. A 3-token request immediately
after the 8k one trims the same process to **8 MiB matmul BO, 3.5 MiB attention
BO, 514 MiB anon**, so the ceiling is one request's working set, not a leak.
The hot set stays resident for repeated same-length requests; a different length
evicts it on the next trim.

## 16. OpenRouter multimodal `/v1/embeddings` (2026-10-22)

The embedding endpoint now accepts the OpenRouter multimodal input shape
(`input: [{"content": [{"type": ...}]}]`) on EmbeddingGemma 2: `text`,
`image_url` (object or bare string), `input_audio` and `input_video`; one
embedding per block, encoded jointly in part order. Images may repeat in a
block (the shared assembly expands the k-th `<|image|>` with image k's own
soft-token count), audio/video are limited to one each, and `input_file` is
rejected. Qwen3 accepts text-only blocks and rejects media parts. Also new:
`dimensions` as an alias of `dim`, an `id` in the response, and a 32 MiB JSON
body limit (axum's 2 MiB default rejected base64 images).

Media values may be local paths, `data:` URIs or `http(s)` URLs; remote media
are fetched on the blocking compute thread with a shared client (10 s timeout,
20 MiB cap, 3 redirects, a User-Agent because Wikimedia rejects anonymous
requests; `ROCKET_MEDIA_FETCH=0` disables). Base64 uploads and downloads go to
`$TMPDIR/rocket-inference/` and are deleted by a `ResolvedMedia` guard once the
forward has read them (the old `/embed` and chat paths leaked these files).

Drive-by fix: the audio tower's zero padding used f32 `Tensor::zeros` against
dtype-matched activations, which panicked (`cat: dtype mismatch`) whenever the
towers loaded as bf16 (the default) — q8/f32 was unaffected. Pads now follow
the activation dtype; audio with q8 text + bf16 towers scores **0.999355**
cosine vs the HF f32 reference.

Measured on the dev host (`--quant q8 --defer-towers`, native bf16 body, bf16
towers; native `/embed` is the baseline because it uses the same input
assembly):

| check | result |
|---|---|
| text block vs plain string | cosine 1.0 |
| image block vs `/embed` image | cosine 1.0 |
| text+image block vs `/embed` (both part orders) | cosine 1.0 |
| data-URI image vs local path | cosine 1.0 |
| audio block vs `/embed` audio | cosine 1.0 |
| video block vs `/embed` video | cosine 1.0 |
| 3-item batch (text, image, joint) vs single calls | cosine 1.0 each, indices 0-2 |
| 2-image block repeat | cosine 1.0 |
| 2-image joint vs HF f32 (`ref_embeddinggemma2.py image --image a b`) | **0.999268**, n=529 on both |
| `dimensions: 256` | 256-d vector |
| 5.8 MB data-URI body | accepted (old 2 MiB limit would 413) |
| `input_file` / unknown part type | 400 |

The `--image` argument of `tools/ref_embeddinggemma2.py image` now takes several
files (`nargs="+"`) so multi-image joint inputs can be checked against HF.

## 17. Deferred

- QAT mobile board re-run (a guarded scope; the f16 text-only profile is now
  3.7 GiB anon plus ~1.2 GiB reclaimable table pages) and the
  `ROCKET_NPU_DECODE=1` decode-speed pre-gate (§13).
- 30k-token text with chunked attention (the old OOM point); needs the board.
- Production `openviking-embed-1` redeploy: **done** — `b09c082` on 2026-10-08
  (canonical tiling, chunking, fused glue), `4a8657b` on 2026-10-09 (burn 0.22
  + native bf16) and `b4634c9` on 2026-10-09 (deferred towers + heap trims, §14;
  idle 2183 -> 605 MiB anon). The deployed config stays `--dtype f32 --quant q8
  --npu --npu-attn npu --attn-chunk 1024` plus `--defer-towers`: the native-bf16
  default loses ~20–25 % to q8 on the board (board text comparison, not yet
  written up).
- Exposing `--npu-int8` / `--npu-int8-group` on `serve` (the server loads fp16
  text weights today).
- int8 for the vision/audio towers: a media request is tower-bound (§18), and
  the q8-resident towers are slower than f32, so this is a memory lever only.

## 18. Vision tower cost (2026-10-24)

The towers had no timing: the CLI started its timer after `build_inputs`, so
the image row in §5 measured the text pass only and the tower was invisible.
This round adds instrumentation and establishes the real baseline.

### 18.1 P0: instrumentation and baseline

#### Instrumentation

`gemma/vision.rs` accumulates per-forward wall time in process-global atomic
microsecond counters (`vision_stats` / `vision_stats_reset`, mirroring the text
`stage_stats`): total, patch embed + position embeddings, attention (input norm,
q/k/v and o projections, axial RoPE, chunked attention, post-attention
norm/residual), MLP (both feed-forward norms, gate/up/down, residual) and
pooling, plus patch and soft-token counts. Timing is additive only: the same
image rerun through the instrumented binary is bit-identical (cosine 1.0,
max|d| 0), and no flag or dtype behavior changed.

- `ROCKET_VISION_LOG=1` prints one stderr line per tower forward (dev f32
  example): `vision: 2394 patches -> 266 soft tokens, tower 5.10s (patch 0.04, attn 2.22, mlp 2.84, pool 0.00)`.
- `gemma embed|bench` print the aggregated line whenever media was encoded
  (regardless of the env var) and add `encode=%.2fs` to their output line.
  `encode` covers all of `build_inputs`: tokenizer/prompt setup + media
  decode/resize/patchify + tower + `embed_vision`. The text-only `encode`
  (~0.6 s) is the tokenizer part, so `encode - text-encode - tower` is the
  media pipeline outside the tower (≤0.2 s in every run).
- `serve` reports the tower forwards as `encode_s` in embedding request logs
  only; the field is absent on text-only requests. `compute_s` still includes
  media decode, the tower and a deferred tower load.
- Text-only CLI output is unchanged apart from the new `encode=` field.

#### Dev host (32 threads, release flex, `--defer-towers`)

`data/cat.jpeg` = 57×42 = 2394 patches → 266 soft tokens; text baseline
`--text 'a photo of a cat'` (7 ids). Mean of 3; `wall` is the whole process and
includes the text load (~1.1 s) and the tower load (f32 2.3 s, q8 7.2 s, bf16
2.0 s); `text forward` is the model call (`forward=` in the CLI line).

| config | request | tower | patch / attn / mlp / pool | encode | text forward | wall | user CPU |
|---|---|---|---|---|---|---|---|
| f32 | image | 5.10 s | 0.04 / 2.22 / 2.84 / 0.00 | 5.74 s | 0.88 s | 10.53 s | 52.5 s |
| f32 | text | — | — | 0.64 s | 0.08 s | 1.87 s | 1.9 s |
| `--quant q8` | image | 7.86 s (7.44-8.19) | 0.07 / 3.45 / 4.34 / 0.00 | 8.49 s | 2.97 s | 21.81 s | 66.4 s |
| `--quant q8` | text | — | — | 0.56 s | 1.03 s | 4.31 s | 6.4 s |
| native bf16 | image | 8.14 s | 0.05 / 4.44 / 3.64 / 0.00 | 8.75 s | 1.11 s | 13.32 s | 50.1 s |
| native bf16 | text | — | — | 0.62 s | 0.10 s | 1.71 s | 1.8 s |

Pooling is 0.00 s everywhere (a spatial mean over the pooled grid). The tower is
**5.8× (f32), 6.9× (q8) and 7.3× (bf16) the text forward** of the same request.

Video (`data/test.mp4`, `--video-fps 1 --video-max-frames 4`, f32): 4 sampled
frames, 1170 patches → 130 soft tokens each (4680 patches, 520 soft tokens).

| case | tower per frame | tower total | encode | text forward | wall |
|---|---|---|---|---|---|
| 4-frame video, f32 | 2.53-2.62 s | 10.24 s | 11.18 s | 1.49 s | 16.53 s |

These measurements correct the numbers derived before instrumentation: the
tower is 5.1 s f32 (not ~5.4), 7.9 s q8 (not ~8, and the derived figure included
part of the 7.2 s q8 tower load), and 8.1 s bf16 (not ~10; `encode` is 8.75 s).

#### Board (`rock-5b-plus`, 4×A76, `taskset -c 4-7`)

Patched 600 MHz NPU module (`rocket_npu_clk_hz=600000000`), `schedutil`, model
`/root/models/embeddinggemma-2` (safetensors), release `--features npu` build of
this source. The production embed service (`home-llm-gateway-embed-1`, which
moved from the old `openviking-embed-1` entry) was stopped for the window and
restored afterwards; other containers kept running (1-min loadavg 3.6 after the
stop, 6.5-7.4 before), so wall is a busy-board number. 3 interleaved reps, same
image and text baseline as above:

- **a** deployed shape: `--dtype f32 --quant q8 --npu --npu-attn npu --attn-chunk 1024 --defer-towers`
- **b** f32 CPU: `--dtype f32 --defer-towers`

| config | request | tower | patch / attn / mlp / pool | encode | text forward | wall | user CPU | tower load |
|---|---|---|---|---|---|---|---|---|
| a q8+npu | image | 27.05 s (26.84-27.19) | 0.24 / 14.48 / 12.32 / 0.00 | 29.35 s | 1.00 s | 69.9 s | 104.1 s | 24.4 s |
| a q8+npu | text | — | — | 2.12 s | 0.22 s | 14.9 s | 13.2 s | — |
| b f32 | image | 25.08 s (24.13-25.77) | 0.13 / 14.06 / 10.89 / 0.00 | 27.23 s | 2.43 s | 46.1 s | 81.6 s | 11.1 s |
| b f32 | text | — | — | 1.92 s | 0.53 s | 6.9 s | 6.0 s | — |

`wall` includes the loads (text ~4 s; q8 also quantizes/packs, and the tower
loads on the first media call). Resident anon from the loader lines: a image
1069 MiB (text backbone 515 + q8 towers), a text 515; b image 2846, b text 1038.
Steady-state request latency is the serve row below. Attention is 54 % of the
tower and MLP 46 % on q8; 56/43 % on f32.

**First media request through `serve`** (config a, `--defer-towers`,
`ROCKET_VISION_LOG=1`, port 8399, image via `/embed`): RssAnon 619,880 kB after
text load, 1,232,900 kB after the first image request (tower weights + scratch)
and unchanged after the next three. First request `compute_s=57.15`
(`encode_s=31.21`; the extra ~25 s is the deferred q8 tower load); steady
requests `compute_s` 31.38 / 29.68 / 32.87 s, `encode_s` 29.86 / 28.37 /
31.38 s, curl wall 31.44 / 29.76 / 32.95 s. A text request immediately after
logs `compute_s=0.228` with no `encode_s` field.

#### Gate assessment

- The tower **is the dominant media cost**: 25-27 s per 2394-patch image on the
  board against 0.2-2.4 s of text forward, i.e. 10-27× the text pass and ~95 %
  of steady request compute; on the dev host 5.1 s against 0.88 s (5.8×).
- q8-resident towers are **slower** than f32 on both hosts (dev 7.9 vs 5.1 s;
  board 27.1 vs 25.1 s) and add a 24.4 s deferred load on the board; bf16
  towers are slower still than f32 on the dev host (8.1 s). The deployed
  `--quant q8` tower dtype is a memory choice, not a media-latency one — this
  also corrects §17's "int8 for the towers: small share of the work": a media
  request is tower-bound.
- Suggested P1/P2 target: **≥4× tower speedup**, i.e. ≤6-7 s per 2394-patch
  image on the board (≤1.3 s f32 on the dev host), with attention (~14.5 s
  board / 2.2 s dev) and MLP (~12.3 s / 2.8 s) as separate budgets; and remove
  the ~24 s deferred q8 tower-load spike from the first request (f32 towers or
  a cheaper load-time quantization). A steady image request would then be
  ~6-7× a text request instead of ~30×.

### 18.2 P1: fused bidirectional vision attention

`--vision-attn fused|blocked` (EmbeddingGemma 2 `gemma embed|bench` and `serve`;
Gemma 4 shares the tower but keeps `blocked`, its CLI and `serve` were not
measured) selects how the tower's bidirectional self-attention is computed:

- `blocked` (**default**) — the P0 path, unchanged: `chunked_attention`
  materializes a `[heads, chunk, patches]` f32 score block per query chunk
  (`--attn-chunk`, 1024) and softmaxes it.
- `fused` — `burn::tensor::module::attention(q, k, v, None, None,
  AttentionModuleOptions { scale: Some(1.0), softcap: None, is_causal: false })`
  on `[1, heads, patches, head_dim]`: flex's tiled flash path (online softmax,
  `TILE_KV` = 64, no score matrix), GQA handled inside the kernel, so k/v are
  passed **unrepeated** (kv_heads == heads in both families today). The scale is
  exactly 1.0 because q/k go through QK-RMSNorm. bf16/f16 towers cast q/k/v to
  f32 for the call and the result back — the blocked path also casts its scores
  to f32 before `softmax`; in an f32 model both casts are no-ops.

Both modes report the mode and a new `kernel` column (the bare attention op;
`attn` keeps the P0 meaning: q/k/v + RoPE/norms + the op + o_proj) in the vision
line — `vision: 2394 patches -> 266 soft tokens, vision-attn fused, tower 4.11s
(patch 0.06, attn 1.27, kernel 0.71, mlp 2.79, pool 0.00)` — and `serve` logs
`vision_attn="fused"` beside `encode_s` on media requests.

Also in this round: the `[2, 10240, 768]` f32 position-embedding table is read
once into the tower (`OnceLock<Vec<f32>>`) instead of being copied out of the
parameter per image. The image embedding is **bit-identical** to the P0 build
(`cat.jpeg`, f32, `blocked`: cosine 1.0, max|d| 0 vs `p0_img_f32.json`), and the
patch phase is unchanged within noise (0.05-0.06 s dev, 0.12-0.20 s board) —
the copy was a modest part of it, and in a bf16 tower it also converted the
whole table to f32; the cost is ~59 MiB of resident anon after the first image
(the cached 63 MB table).

#### Dev host (32 threads, release flex, `--defer-towers`)

`data/cat.jpeg` (57×42 = 2394 patches → 266 soft tokens), 3 interleaved reps,
mean (range); `encode` is all of `build_inputs` (tower + media pipeline),
`wall` the whole process including the loads (f32 2.3 s / q8 7.2 s / bf16 2.0 s
tower load), `user` the process CPU.

| config | mode | tower | patch / attn (kernel) / mlp | encode | wall | user CPU |
|---|---|---|---|---|---|---|
| f32 | blocked | 5.02 s (4.92-5.14) | 0.05 / 2.21 (1.65) / 2.75 | 5.65 s | 10.34 s | 52.9 s |
| f32 | fused | **4.11 s** (4.04-4.24) | 0.06 / 1.27 (0.71) / 2.79 | **4.78 s** | 9.46 s | 44.7 s |
| `--quant q8` | blocked | 7.45 s (7.40-7.49) | 0.06 / 3.13 (1.80) / 4.25 | 8.10 s | 21.26 s | 66.1 s |
| `--quant q8` | fused | **6.54 s** (6.43-6.66) | 0.08 / 2.15 (0.75) / 4.31 | **7.18 s** | 20.29 s | 57.9 s |
| native bf16 | blocked | 8.20 s (8.15-8.25) | 0.05 / 4.46 (3.36) / 3.69 | 8.80 s | 13.35 s | 50.9 s |
| native bf16 | fused | **5.37 s** (5.28-5.54) | 0.06 / 1.82 (0.75) / 3.49 | **5.98 s** | 10.66 s | 41.5 s |
| f32, 4-frame video | blocked | 9.95 s (9.88-9.99) | 0.04 / 3.44 (1.92) / 6.46 | 10.79 s | 16.08 s | 80.1 s |
| f32, 4-frame video | fused | **8.44 s** (8.40-8.48) | 0.05 / 2.55 (0.96) / 5.83 | **9.25 s** | 14.71 s | 76.9 s |

Fused/blocked: tower **1.22×** (f32), **1.14×** (q8), **1.53×** (bf16), 1.18×
(video); attention phase 1.74× / 1.46× / 2.45× / 1.35×; the attention kernel
alone **2.32× / 2.40× / 4.48× / 2.00×**. `sys` CPU drops from 7.5 to 3.1 s
(f32) and from 7.7 to 2.6 s (bf16) — the score blocks are gone.

Chunking the fused call over queries was tried and is slower (dev f32 kernel:
0.73 s whole sequence vs 0.83 s at `--attn-chunk 1024`, 0.89 s at 512, 1.01 s at
256), so the fused mode makes one kernel call per layer.

#### Board (`rock-5b-plus`, 4×A76, `taskset -c 4-7`)

Patched 600 MHz NPU module, model `/root/models/embeddinggemma-2`, release
`--features npu` build. The production embed service (`home-llm-gateway-embed-1`)
was stopped for the window and restored (health ok) afterwards; loadavg was
6.5-7.4 with the other containers, so wall includes a busy board. 3 interleaved
reps, mean (range):

- **a** deployed shape: `--dtype f32 --quant q8 --npu --npu-attn npu --attn-chunk 1024 --defer-towers`
- **b** f32 CPU: `--dtype f32 --defer-towers`

| config | mode | tower | patch / attn (kernel) / mlp | encode | wall | user CPU |
|---|---|---|---|---|---|---|
| a q8+npu | blocked | 27.28 s (24.86-28.74) | 0.20 / 14.38 (10.35) / 12.69 | 29.64 s | 70.2 s | 101.6 s |
| a q8+npu | fused | **23.79 s** (23.00-24.59) | 0.20 / 11.00 (7.18) / 12.57 | **26.03 s** | 64.4 s | 100.3 s |
| b f32 | blocked | 25.85 s (25.09-26.48) | 0.12 / 14.65 (11.35) / 11.08 | 28.01 s | 47.0 s | 78.6 s |
| b f32 | fused | **21.93 s** (21.51-22.22) | 0.13 / 10.73 (7.24) / 11.06 | **24.03 s** | 46.1 s | 79.9 s |
| b f32, 4-frame video | blocked | 43.50 s (42.53-44.30) | 0.24 / 19.64 (12.41) / 23.61 | 46.41 s | 70.8 s | 136.4 s |
| b f32, 4-frame video | fused | **36.72 s** (36.22-37.52) | 0.19 / 13.34 (6.28) / 23.17 | **39.28 s** | 63.9 s | 131.1 s |

Fused/blocked: tower **1.15×** (a), **1.18×** (b), 1.18× (video); attention phase
1.31× / 1.37× / 1.47×; kernel 1.44× / 1.57× / 1.98×. `sys` CPU halves
(a 8.9 → 5.1 s — the blocked samples were 11.5 / 5.0 / 10.2 — b 8.5 → 4.0 s),
`user` is flat.

**First media request through `serve`** (config a, `--vision-attn fused`,
`--defer-towers`, port 8399): RssAnon 619,888 kB after the text load, 1,292,868
kB after the first image (+~59 MiB of cached position table vs P0's 1,232,900)
and unchanged after the next two. `compute_s` 52.87 (first; `encode_s` 25.49,
the rest the deferred tower load), then 27.34 / 27.18 with `encode_s` 25.65 /
25.74 — against P0's 31.38 / 29.68 / 32.87 and `encode_s` 29.86 / 28.37 / 31.38,
i.e. **-13 % steady media latency**; the text request after it logs
`compute_s=0.207` with no `encode_s`/`vision_attn` fields.

#### Parity

Normalized 768-d embeddings; the reference is
`tools/ref_embeddinggemma2.py {image,video}` (HF f32, transformers 5.19).

| case | fused vs HF f32 | blocked vs HF f32 | fused vs blocked | max\|d\| |
|---|---|---|---|---|
| dev f32 image | 1.000000000 | 1.000000000 | 0.999999999997 | 3.2e-07 |
| dev f32 video | 1.000000000 | 1.000000000 | 0.999999999999 | 1.4e-07 |
| dev q8 image | 0.999670037 | 0.999670029 | 0.999999999962 | 1.1e-06 |
| dev q8 video | 0.999654505 | 0.999654508 | 0.999999999999 | 2.1e-07 |
| dev bf16 image | 0.999829537 | 0.999886852 | 0.999853746 | 2.1e-03 |
| dev bf16 video | 0.999941769 | 0.999915897 | 0.999912331 | 1.6e-03 |
| board f32 image | 1.000000000 | 1.000000000 | 0.999999999992 | 5.1e-07 |
| board f32 video | 0.999652156 | 0.999652157 | 0.999999999999 | 1.6e-07 |
| board a image (q8+npu) | 0.999671021 | 0.999669094 | 0.999997856 | 2.3e-04 |

The tower output itself differs by max|d| 8.5e-5 (dev) / 9.0e-5 (board) at
~1e-5 relative on activations of magnitude ~1200: per-layer f32 reassociation
grows from 2.9e-7 (layer 0) to 1.4e-5 (layer 15) and the mean pool plus
`embed_vision` bring the final vector to 1e-7-1e-6 in f32. In the deployed
q8+npu shape the same 1e-5 soft-token difference is amplified to 2.3e-4 by the
**NPU f16 text path** (reproducible: two blocked runs are bit-identical, two
fused runs too, and every cross pair gives 0.999997856); it is far below the
deployed config's own 0.9992-0.9999 vs HF. bf16 differs more between the modes
because the blocked path casts its softmax probabilities to bf16 *before* `P @ V`
while the fused path keeps them f32 — the fused output is not worse, both sit at
0.99983-0.99994 vs HF f32.

#### Gate assessment

- The gate was **≥2× faster attention phase and ≥1.4× tower on both hosts**:
  **not met**. Achieved: attention phase 1.35-2.45× (dev f32 1.74×, dev bf16
  2.45×, dev q8 1.46×, board 1.31-1.47×), tower 1.14-1.53× (dev f32 1.22×,
  dev bf16 1.53×, dev q8 1.14×, board 1.15-1.18×). Numerics passed everywhere
  except the strict `fused-vs-blocked ≥0.99999` item in bf16 and in the board's
  q8+npu shape (both explained above); f32 is ≥0.99999999999.
- Why: the attention kernel is only 32 % of the tower (dev f32 1.65 of 5.02 s)
  and 44 % (board b 11.35 of 25.85 s), so even a free kernel would cap the tower
  at 1.47× dev / 1.54× board. The `attn` *phase* also contains the q/k/v/o
  projections, RoPE and norms (dev 0.56 s, board 3.81 s), which the mode does not
  touch, and the fused kernel itself only reaches 1.44-1.57× on the board: flex
  parallelizes it over the tower's 12 heads (6.48 s single-threaded → 0.72 s on
  32 threads = 9×; the exp/online-softmax terms are serial per head), while the
  blocked path's `softmax` spreads over the whole score tensor, so its
  parallelism advantage shrinks from 32 to 4 cores.
- Default: **`blocked` stays** (the gate fell short), `fused` is opt-in per
  command (`--vision-attn fused` on `gemma embed|bench` and `serve`). It is
  nevertheless a strict, reproducible win in every measured shape — tower
  -8 %..-15 % on the board, -14 %..-35 % on the dev host, half the `sys` CPU,
  identical f32 numerics — so flipping the default is a one-line change (the
  `#[default]` on `VisionAttn` plus the `VisionAttn::Blocked` sites in the
  Gemma 4 paths, which would then inherit it) once a default swap is wanted
  (done in §18.3).
- Not done: NPU attention for the tower (the text model's attention already runs
  on the NPU at f16; the vision tower would need the same per-layer pack plus
  `window < 0`) — **measured in §18.4 and rejected: 1.05× at best, because the
  driver's host softmax dominates at head_dim 64**; and int8/per-row-quantized
  tower MLPs — after this round the **MLP was the largest tower phase** (12.6 s
  of 23.8 s deployed on the board; §18.5's fused glue then cut it to 4.4 s, so
  attention is the largest phase at 50 % now).

### 18.3 P2: NPU-packed vision projections

The tower's GEMMs now run on the NPU as resident fp16 weights. `--npu` packs the
16 layers' q/k/v/o + gate/up/down (`ClippableLinear::pack_npu`, 112 weights) and
`embed_vision.embedding_projection` (1) — 113 weights, **0.28 GiB** of resident
fp16 NPU memory — and shrinks each CPU weight to a `[1, 1]` stub via
`Param::map`, keeping the `ParamId` so `lin()` still routes through the registry
(the same pack-and-drop as the text backbone). Q8-resident towers are
dequantized once per weight while packing, so the packed weights are fp16 in
both arms. `burn_rocket::pack` gets HF `[N, K]` (Burn `[in, out]` transposed).
The patch embedder, every RMSNorm, RoPE and the attention kernel stay on the CPU;
the audio tower is untouched (not the media cost, and its linears carry clip
bounds). Clipped linears are skipped by `pack_npu`, which is also why Gemma 4's
tower is not wired: its `use_clipped_linears` would need the bounds applied
around a host-side GEMM.

Wiring: `load_towers`/`load_towers_mmproj` take an `npu_pack` flag (the CLI's
`--npu`, `serve`'s resolved NPU decision) and pack after the store pass and the
optional Q8 mapping. **`ROCKET_NPU_TOWERS=0`** is the A/B off-switch (the pack
is on by default under `--npu`). The eager path creates the NPU engine before
loading towers (`burn_rocket::init` is idempotent); the deferred path
(`--defer-towers`) packs on the first media request, so text-only loads allocate
nothing and stay bit-identical (serve: RssAnon 619,836 kB after the text load vs
P1's 619,888; a text request after media is `compute_s=0.198` with no
`encode_s`). The `towers: loaded` line reports the pack:

```text
towers: loaded (F32, q8-resident) in 29.92s (resident 1009 MiB anon), +113 projections packed (0.28 GiB NPU in 6.33s)
```

That line's duration includes the pack; the pack cost alone is the suffix. Also
in this round: `--vision-attn` now defaults to **`fused`** for EmbeddingGemma 2
(`VisionAttn`'s `#[default]`; `blocked` remains the parity fallback and Gemma 4
keeps it explicitly), and `ROCKET_VISION_LOG=1` appends the per-forward NPU
split to the vision line (`..., npu 112 calls convert 1.27s npu 3.96s`) measured
from `burn_rocket::stats()` around the layer loop.

#### Pack cost (board, `rocket-inference-p2`, 113 weights, 0.28 GiB)

| arm | tower load (store + q8 map) | pack | tower load total | resident anon after load |
|---|---|---|---|---|
| a q8+npu, pack off | 24.1 s | — | 24.1 s | 1069 MiB |
| a q8+npu, pack on | 24.2 s | **6.26 s** (6.18-6.31) | 30.4 s | **906 MiB** |
| b f32+npu, pack off | 12.6 s | — | 12.6 s | 2343-2350 MiB |
| b f32+npu, pack on | 12.8 s | **4.44 s** (4.08-4.76) | 17.2 s | **1745 MiB** |

The q8 arm's extra ~1.8 s is the per-weight Q8 dequant; both arms then convert
to fp16 (`pack`) and free the host copy. The CPU resident set drops by the
packed weights (a -163 MiB, b -598 MiB — the f32 tower is 1.2 GiB, Q8 0.14 GiB,
both become 0.28 GiB of NPU buffers).

#### Board timing (2394-patch image, 3 interleaved reps, mean (range))

`taskset -c 4-7`, patched 600 MHz NPU module, production embed service stopped
for the window and restored (health ok); other containers kept running, loadavg
1.9-3.4, so wall carries board noise. `fused` vision attention everywhere.

- **a** deployed shape: `--dtype f32 --quant q8 --npu --npu-attn npu --attn-chunk 1024 --defer-towers --vision-attn fused`
- **b** f32+npu: `--dtype f32 --npu --npu-attn npu --attn-chunk 1024 --defer-towers --vision-attn fused`

| config | pack | tower | patch / attn (kernel) / mlp | encode | wall | user CPU |
|---|---|---|---|---|---|---|
| a q8+npu | off | 24.84 (24.51-25.33) | 0.21 / 11.58 (7.39) / 13.04 | 27.08 | 67.6 s | 112.7 s |
| a q8+npu | **on** | **19.84** (19.30-20.46) | 0.19 / 11.26 (8.19) / **8.39** | **22.13** | 68.7 s | **92.9 s** |
| b f32+npu | off | 22.98 (22.63-23.66) | 0.18 / 11.29 (7.50) / 11.50 | 25.28 | 49.5 s | 92.6 s |
| b f32+npu | **on** | **19.55** (19.34-19.76) | 0.16 / 11.10 (7.85) / **8.28** | **21.81** | 50.4 s | **71.4 s** |

Tower **-20.1 %** (a) / **-14.9 %** (b); MLP phase **-35.7 %** / **-28.0 %**;
`encode` -18.3 % / -13.7 %; `user` CPU -19.8 s (-17.6 %) / -21.2 s (-22.9 %);
`sys` rises 5.0 → 7.2 s (a) as the driver's BO mapping/DMA replaces part of the
CPU GEMM. `wall` is flat because a one-shot CLI run pays the pack (+6.3/+4.4 s)
and saves the same amount in the tower; for serving only the steady request
matters. The attention phase barely moves (it is the kernel: 7.4-8.2 s) and the
kernel itself reads ~0.4-0.6 s slower packed in all six runs (7.39 → 8.19 a,
7.50 → 7.85 b), i.e. 5-8 % — reproducible but small, plausibly cache/bandwidth
pressure from the resident BOs; it is why the attention phase gain is only ~3 %.

Video (config a, `data/test.mp4`, 4 frames, 4680 patches): pack off **45.24 s**
(all MLP 29.18 s), pack on **31.09 s** (MLP 16.84 s) — **-31.3 %**; per frame
7.48 / 7.73 / 7.89 / 7.98 s. The smaller M = 1170 wins relatively more because
the CPU GEMM degrades faster than the NPU at that size.

#### Where the time goes (per 2394-patch forward, packed)

`ROCKET_VISION_LOG=1` (`burn_rocket::stats()` deltas around the 16 layers), both
arms, 2 reps:

| arm | NPU calls | input/output conversion | NPU kernel | GF/s in the driver |
|---|---|---|---|---|
| a q8+npu | 112 | 1.22-1.27 s | 3.90-3.96 s | **183** |
| b f32+npu | 112 | 1.26 s | 3.96 s | **183** |
| a video, per frame | 112 | 0.62-0.68 s | 1.73-1.90 s | 193 |

The tower's GEMM work is 361.5 GMAC = 723 GFLOP (150.99 M MACs/token × 2394
patches), so the driver executes it at ~183 GF/s — **not** the 300-380 GF/s the
round assumed. The rest of the 19.8 s is: attention kernel 8.0 s, conversions
1.25 s, and ~6.3 s of host marshalling + flex glue (`read_floats_f32` copies
each activation out of the flex tensor before every matmul, i.e. ~1.2 GB of
input copies per forward, plus RoPE/norms/residual/`gelu_mul`, patch and pool).
The unpacked CPU path spends 17.2 s in the same two phases (11.58 - 7.39 = 4.2 s
of q/k/v/o + 13.04 s of MLP, ~45 GF/s effective), so the packed GEMM+conversion
cost (5.2 s) is a ~3× win that only shows as -20 % of the tower.
The practical fp16 throughput is consistent with the driver's own microbenchmarks
(§Port status: int8 512×3072×1024 in 62-70 ms ≈ 46-52 GF/s, bf16 stream 24-27 ms
≈ 119-134 GF/s), so the ceiling is the driver/SoC, not this integration.

#### Serve first request (config a, deferred, port 8399)

| request | compute_s | encode_s | RssAnon |
|---|---|---|---|
| text load | — | — | 619,836 kB (P1 619,888) |
| first image (load 24 s + pack 6.33 s) | 51.27 | 18.42 | 1,152,144 kB (P1 1,292,868) |
| steady image 2-4 | 20.88 / 20.44 / 20.56 | 19.31 / 19.34 / 18.95 | 1,152,148-1,152,176 kB |
| text after | 0.198 | — | 1,075,7xx kB |

Steady media latency **-24.3 %** vs P1 (27.34 / 27.18 s); the first request is
**not slower** (51.27 vs P1's 52.87 s): the 6.3 s pack is more than repaid by
the first forward (18.4 vs 25.5 s), and the deferred load itself is unchanged
(24.1 s of store+q8 map, P1 24.4). Resident media memory drops 140 MB (P1's +59 MiB
position-table cache is included in both).

#### Parity

Normalized 768-d embeddings vs the HF f32 reference (`p1_ref_*.json`), and the
tower's own soft tokens (266 × 512, `--dump-vision`) vs the same binary's f32 CPU
tower; `packed vs unpacked` compares against `ROCKET_NPU_TOWERS=0` in the same arm.

| case | vs HF f32 | max\|d\| | tower vs f32 CPU | packed vs unpacked |
|---|---|---|---|---|
| a q8+npu image | 0.999687984 | 2.4e-03 | 0.996689271 | 0.999987637 |
| a q8+npu video | 0.999158561 | 6.4e-03 | — | 0.999999433 |
| b f32+npu image | 0.999997457 | 2.6e-04 | 0.999998817 | 0.999997309 |
| b f32+npu video | 0.999649904 | 3.0e-03 | — | — |

- The pack itself is **0.99999858-0.99999882** at the tower output (max|d|
  3.1e-2 on activations of magnitude ~1200, i.e. 2.6e-5 relative), and
  0.9999876-0.9999973 at the embedding: the fp16 weight rounding lands far
  below the arm's own gap to HF.
- The q8 arm's 0.99969 (image) is **better** than P1's unpacked 0.99967 (the
  packed fp16 weights round the Q8-dequantized values, which happens to help).
  The tower-only q8 cosine (0.99669) is the Q8 mapping's own error: comparing
  the same tower with the q8 CPU dump gives 0.99999858, i.e. the pack adds
  nothing to it.
- Video's 0.99916 is the q8 + f16 NPU text path, not the pack: pack off gives
  0.9991587 (pack A/B 0.9999994), and P1's board f32 CPU video was 0.9996522 —
  the video reference already carries a ~3.5e-4 gap from frame
  decode/sampling. Likewise f32+npu image is 0.9999975, i.e. the fp16 text
  GEMMs, not the tower.
- Determinism: repeated runs are **bit-identical** (cosine 1.0, max|d| 0) in
  both arms, image and video.

#### Gate assessment

- **Tower ≤12 s: not met.** Achieved 19.84 s (a, -20 %) and 19.55 s (b, -15 %)
  from 24.84 / 22.98 s. The bound is structural: the CPU attention kernel alone
  is 7.4-8.2 s, the NPU GEMMs need ≥4 s at the measured 183 GF/s, and the
  extension's per-call host round-trip adds ~1.3 s of conversions plus several
  seconds of marshalling. A ≤12 s tower would need the attention kernel on the
  NPU (P1's remaining lever) and/or a zero-copy `matmul` that keeps activations
  on the device between layers.
- **Numerics: passed** for the requested image cases — 0.99969 (q8, gate
  ≥0.9995) and 0.9999975 (f32, gate ≥0.9999) — and the video difference is
  attributable to the pre-existing reference/q8 gap, not the pack (A/B
  0.9999994).
- **Text-only unchanged**: no towers → no pack, no NPU memory, same loader
  numbers and RssAnon as P1.
- Decision: keep the pack **on by default** under `--npu` (strict wall/CPU/memory
  win on every media shape), `fused` as the vision-attention default, and the
  env switch as the A/B off. A redeploy of this source therefore gets both
  without new flags (the live image predates them).
- Not done / next levers: fused resident packs (`pack3` q|k|v and `pack2`
  gate|up, as the text loader does) would cut 3 of the 7 per-layer input copies
  and 1 input conversion each — the cheapest remaining win, **landed in
  §18.5.1** (packs 113 → 65, calls 112 → 64, but only -0.13 s of tower); NPU
  attention for
  the tower (bidirectional `window < 0`, host softmax and a 2394² score
  matrix are the risks) — **measured in §18.4 and rejected: 1.05× at best,
  because the driver's host softmax dominates at head_dim 64**; int8 tower
  weights (readback-bound, memory lever only).
  The bf16 tower was not re-measured: its fused-vs-blocked near-ties in §18.2
  (0.99985 / 0.99991) are below the strict 0.99999 bar because the **blocked**
  path rounds the softmax probabilities to bf16 before `P @ V` while fused keeps
  them f32 — both modes sit at 0.99983-0.99994 vs HF, so `fused` is not the
  less accurate one, but a bf16 default should re-run the §18.2 A/B first.


### 18.4 P3: NPU vision attention (negative, 2026-10-24)

The P2 tower still spends 7.9-8.2 s per 2394-patch image in the fused CPU
attention kernel (11.1-11.3 s for the whole attention phase). This round asked
whether the NPU can run that op faster: the text backbone already offloads
attention through `burn_rocket::attention_window` (f16 in, `window = -1` = no
mask), and the vision geometry is the same bidirectional batch-1 shape.

**Decision gate, set before measuring:** the NPU kernel must be **≥1.3× faster**
than the fused CPU kernel *and* cut tower wall by **≥10 %**. Otherwise stop,
keep `fused`, and leave the op unimplemented.

#### Microbenchmark

`examples/rocket-inference/src/bin/vision_attn_bench.rs` (bin gated on the `npu`
feature, aarch64 only) runs one tower layer's attention on the real geometry:
q/k/v `[1, 12, p, 64]` f32, scale 1.0, bidirectional, `taskset -c 4-7`, NPU at
the patched 600 MHz module. The production embed service
(`home-llm-gateway-embed-1`) was stopped for the window and restored each time
(health ok); the other containers kept the board at loadavg ~1.4-2.5, so wall
carries some noise. `fused` = `burn::tensor::module::attention`
(`is_causal: false`); `npu` = `burn_rocket::attention_window(..., 12, 12, 64,
1.0, None, -1)` on `[1, p, 768]`, whose `convert` (f32→f16 head-major layouts)
and `npu` (driver call, including its host softmax) are the
`burn_rocket::stats()` deltas; `npu-chunk-N` = the same through
`attention_window_block` per query chunk (`window = -1`, `kv_start = 0`).

p = 2394 (57×42 image), 5 reps after a warmup, seconds per layer (`RssAnon` is
the process's resident anonymous memory after the arm — the NPU scratch stays
cached, so it is that arm's high-water):

| arm | s/layer | convert | npu | calls | vs fused cos | max\|d\| | RssAnon |
|---|---|---|---|---|---|---|---|
| fused CPU kernel | **0.455** | — | — | — | 1 | 0 | 61 MiB |
| npu (full p) | 0.461 | 0.018 | 0.429 | 1 | 0.999999844 | 3.2e-5 | 195 MiB |
| npu-chunk-2048 | 0.452 | 0.030 | 0.382 | 2 | 0.999999844 | 3.2e-5 | 122 MiB |
| npu-chunk-1024 | **0.434** | 0.031 | 0.365 | 3 | 0.999999844 | 3.2e-5 | 148 MiB |
| npu-chunk-512 (3 reps) | 0.685 | 0.072 | 0.478 | 5 | 0.999999844 | 3.2e-5 | 149 MiB |

p = 1170 (one of the 4 frames of `data/test.mp4`), 3 reps:

| arm | s/layer |
|---|---|
| fused CPU kernel | **0.094** |
| npu (full p) | 0.129 |
| npu-chunk-1024 | 0.156 |
| npu-chunk-512 | 0.194 |

Best case at the image size is **1.05×** (chunk 1024: 0.434 vs 0.455); the
full-sequence call is 0.99×, chunk 2048 1.01×, chunk 512 0.66×, and at the
video-frame size the CPU kernel wins outright (1.37-2.06×). The op's numerics
are excellent — 0.999999844 against the f32 fused kernel, max|d| 3.2e-5 on
outputs of order 0.1, the same for every chunking and bit-identical across
runs — so accuracy is not what rules it out.

#### Why it cannot win here

Per layer the GEMMs are small and the softmax is not. The QK/AV matmuls are
only 2 × 2394² × 64 × 12 = 8.8 GMAC (17.6 GFLOP), ~0.1 s at the driver's
measured ~183 GF/s, while the softmax covers 12 × 2394² = **68.8M scores**. The
remaining ~0.33 s of the 0.43 s driver call is host-side score work: 138 MB read
back from the device, the serial scalar softmax over those scores (scale, mask
add, max, exp, sum, f16 round) and 138 MB uploaded again as the AV operand. That
is the same arithmetic flex's fused kernel does, except flex overlaps it with
the GEMMs in one tiled online-softmax pass and spreads it over 12 heads, while
the driver serializes 3 heads per worker across 4 workers. The per-call
f32→f16 layout conversion (0.02-0.03 s) is not the limit; the score traffic and
the scalar softmax are. `rocket-npu` has no on-NPU softmax ("Missing surface" in
the root AGENTS.md — this is the C library's default host-softmax path), so a
≥1.3× win would need that or a fused flash kernel in the driver (deferred in
§18.7).

Extrapolated to the tower: 16 layers × 0.434 s = 6.9 s against P2's 7.9-8.2 s
kernel, i.e. **-5..-6 % of the 19.8 s tower**, and ~0 on the video path — versus
the 10 % gate. (P2's in-situ kernel is 0.49-0.51 s/layer, 8-12 % above this
isolated fused arm, which would flatter the NPU ratio to 1.13-1.18×; still short
of 1.3×, and the tower delta stays ~6 %.) The f16 attention would also sit on
top of the q8 arm's existing 0.99969-vs-HF error, where the P3 numerics gate
(≥0.9999 vs fused) would have needed care. **`--vision-attn npu` was not
implemented**: the tower keeps
`fused` as its default, `blocked` as the parity fallback, and its attention on
the CPU. The bench bin stays in the tree (gated on `npu`) as the reproducer:

```sh
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu --bin vision_attn_bench
# on the board, with the embed service stopped for the window:
taskset -c 4-7 ./vision_attn_bench 2394 5 1024,2048
```


### 18.5 P4: residual overhead + tower load policy

P2 left 19.8 s of tower per 2394-patch image with the GEMMs already on the NPU:
8.2 s of CPU attention kernel, 1.25 s of f16 conversions and ~6 s that the log
called "host marshalling + flex glue". This round attacked that last bucket and
the tower load policy, and re-measured the deployed shapes on a quiet board.

Two windows on `rock-5b-plus` (2026-10-24 06:47-07:03 and 07:09-07:22 board
time), production `home-llm-gateway-embed-1` stopped for each and restored
(`/health` ok both times), patched 600 MHz NPU module, `taskset -c 4-7`,
1-min loadavg 1.5-1.8 (P2's window ran at 1.9-3.4, P3's at 1.4-2.5). Binaries:
`rocket-inference-p4` (this round) and `rocket-inference-p2` (the P3-era build)
so the baseline is measured **in the same window** rather than cross-window.

#### 18.5.1 Fused projection groups (landed, `ROCKET_NPU_FUSED`)

The tower now packs q|k|v as one `pack3` weight and gate|up as one `pack2`
weight (`VisionAttention::pack_npu` / `VisionMlp::pack_npu`, handles stored as
`#[module(skip)] Option<WeightId>`), and each layer runs one `matmul` per group
and splits the output along N (`q,k,v` = 768|128|128 columns, `gate,up` =
3072|3072). The per-projection path stays for non-NPU builds, for clipped
linears (Gemma 4's tower; `ClippableLinear::group_weight_nk` returns `None` when
clip bounds exist) and for `ROCKET_NPU_FUSED=0`, the A/B switch. Packed weights
**113 → 65**, matmuls per forward **112 → 64**; the resident bytes are unchanged
(0.28 GiB) apart from segmented-pack padding (`weights_mib` 704.7 → 712.7).

Same-window A/B, arm a = the deployed shape (`--dtype f32 --quant q8 --npu
--npu-attn npu --attn-chunk 1024 --defer-towers --vision-attn fused`), 3
interleaved reps, mean (range):

| build | tower | patch / attn (kernel) / mlp | calls | convert | npu | user CPU |
|---|---|---|---|---|---|---|
| p2 (P3-era) | 19.83 (19.17-20.31) | 0.20 / 11.35 (8.18) / 8.28 | 112 | 1.21 | 3.83 | 92.2 s |
| p4, `ROCKET_NPU_FUSED=0` | 15.64 (15.02-16.06) | 0.17 / 11.11 (7.94) / 4.36 | 112 | 1.16 | 3.62 | 82.5 s |
| p4, fused (default) | **15.51** (14.77-16.25) | 0.18 / 10.85 (7.81) / **4.47** | **64** | **1.06** | **3.12** | 82.9 s |

The fusing itself is the small part: **-0.13 s of tower** (inside the ±0.7 s rep
spread), **-0.50 s of NPU time**, -0.10 s of conversions and -43 % of the calls;
the two group calls cost ~49 ms each against ~32 ms for the seven separate ones.
The **4.2 s** in that row is the fused MLP glue, measured separately in window 1
(1 rep each, arm a): `ROCKET_GLUE=0` tower **19.77 s** (MLP **8.77 s**) vs
`ROCKET_GLUE=1` **15.56 s** (MLP **4.36 s**). The tower's MLP still computed
`gelu_approximate(gate) * up` as two flex ops — serial scalar loops over
2394 x 3072 x 16 layers = 235 M element evaluations — while `lin()` was already
routed through the NPU; `VisionMlp::forward` now uses `crate::gemma::layers::
gelu_mul` (the `burn_rocket::gelu_mul` single-pass rayon kernel when the glue is
on), which is where the 4.4 s of MLP disappeared. CPU modes (`ROCKET_GLUE` off,
no `--npu`) keep the composite ops bit-identically: the dev host f32 image
embedding is unchanged (cosine 1.0, max|d| 0 vs the P2 binary).

Parity of the group GEMM vs the per-projection path, at the final normalized
embedding (same weights, same fp16 pack): a **0.9999967** (max|d| 4.4e-4), b
**0.9999979** (3.6e-4) — *not* bit-identical, because the driver's plan (and
with it the fp16 accumulation order) depends on N; both arms are deterministic
(repeats cosine 1.0, max|d| 0) and both sit 3 orders of magnitude above the
0.9999 parity gate. Against HF f32 the fused arm is 2.7e-6 below the
per-projection one in the f32+npu arm (0.9999963 vs 0.9999990) and equal in the
q8 arm (0.9998836 vs 0.9998878), i.e. the split is correct (a wrong column order
would not be a 1e-3 perturbation) and the difference is fp16 GEMM noise.

#### 18.5.2 Tower load policy: f32 vision + Q8 audio (landed)

Under `--npu` the towers' Q8 pass now **excludes the packed part**:
`tower_quant_group(npu_towers)` is `projection_group()` minus
`^(vision_tower|embed_vision)\.`, so the vision tower (and `embed_vision`) keep
the load dtype (f32 in the deployed arm) and the pack starts from full-precision
weights, while the audio tower, which stays on the CPU, keeps its Q8 mapping
(~0.4 GiB saved) and `embed_audio` stays f32 (3 MiB). Non-NPU `--quant q8`
behavior is unchanged, and so is `ROCKET_NPU_TOWERS=0` (no pack → no exclusion,
the whole tower tree is Q8 again). The `towers: loaded` line reports the policy
(`F32, q8 audio, f32 vision`).

Board, `--defer-towers`, one `towers: loaded` line per run; the store+map
column is that line's duration minus the pack it reports (window 2 / window 1):

| arm | store + q8 map | pack | tower load total | resident anon after load |
|---|---|---|---|---|
| p2 a q8+npu (q8 vision) | 24.20 s | 6.18 s | 30.38 s | 1009 MiB |
| **p4 a q8+npu (f32 vision)** | **20.5 s** | **4.3 s** (3.9-4.7) | **24.8 s** (24.3-25.4) | **908 MiB** |
| p2 b f32+npu | ~12.8 s | 4.4 s | 17.16 s | 1745 MiB |
| p4 b f32+npu | 13.1 s | 4.5 s (4.3-4.7) | 17.6 s (16.6-17.7) | 1745 MiB |

The f32 policy removes ~3.7 s of Q8 mapping from the store pass (24.2 → 20.5 s)
and ~1.9 s from the pack (the per-weight Q8 dequant, 6.18 → 4.3 s): **-5.6 s of
deferred load**. The remaining 7.4 s above the f32 arm's 13.1 s store is the
**audio** tower's Q8 pass (20.5 - 13.1), i.e. the cost of keeping the audio
weights at 0.4x; the vision tower no longer pays it. Resident anon after load is
unchanged (**908** vs 906 MiB in P2) because the f32 vision weights are dropped
as they are packed, and the serve high-water is flat: RssAnon after the text
load 619,828 kB and 1,184,068 kB after two image requests (P2 1,152,176 kB; the
+32 MB is the fused group's wider matmul scratch slot, `matmul_host_mib` 37 →
67, not the load policy — the per-projection p4 build measures 1,154,024 kB).

The deployed image's route (`-hf unsloth/embeddinggemma-2-GGUF:Q8_0`, towers from
`mmproj-Q8_0.gguf`) takes the same policy: `towers: loaded from mmproj (F32, q8
audio, f32 vision) in 24.3 s (907 MiB anon), +65 weights packed (0.28 GiB NPU in
4.5 s)`, tower 15.75 s / 64 calls on the 2394-patch image where
`ROCKET_NPU_TOWERS=0` gives 21.22 s / 1068 MiB — i.e. -26 % of the tower for
-161 MiB of resident memory on the GGUF path too (smoke run, safetensors not
involved).

#### 18.5.3 Attention kernel dtype: f16/bf16 rejected (kept f32)

`vision_attn_bench` now runs the fused CPU kernel with f32, f16 and bf16
operands (plus the NPU arms of §18.4), `taskset -c 4-7`, 3 reps after a warmup:

| arm | p=2394 s/layer | p=1170 s/layer | vs f32 cosine | max\|d\| |
|---|---|---|---|---|
| fused f32 (**kept**) | **0.442** | **0.089** | 1 | 0 |
| fused f16 | 0.483 (+9 %) | 0.104 (+17 %) | 0.999999945 | 1.5e-5 |
| fused bf16 | 0.468 (+6 %) | 0.112 (+26 %) | 0.999996566 | 1.2e-4 |
| npu (for reference, §18.4) | 0.452 | 0.104 | 0.999999844 | 3.2e-5 |

The gate was "≥15 % faster **and** cosine ≥0.9995"; the 16-bit arms are *slower*
at both shapes, because flex dispatches attention by dtype and casts f16/bf16
operands to f32 for the computation (`dispatch_attention_dtype!` in
`burn-flex/src/ops/attention.rs`) — 16-bit operands only add two conversions of
`[1, 12, p, 64]` around the same f32 kernel. No f16 kernel path was
implemented; the tower keeps casting q/k/v to the kernel's f32 operands (a no-op
in the f32 arms).

#### 18.5.4 Arms, serve and parity

Window-1 timing, 3 interleaved reps, mean (range), `--defer-towers`:

- **a** deployed: `--dtype f32 --quant q8 --npu --npu-attn npu --attn-chunk 1024 --vision-attn fused`
- **b** f32+npu: `--dtype f32 --npu --npu-attn npu --attn-chunk 1024 --vision-attn fused`

| config | tower | patch / attn (kernel) / mlp | encode | wall | user CPU |
|---|---|---|---|---|---|
| a q8+npu | **15.17** (14.78-15.41) | 0.17 / 10.52 (7.56) / 4.47 | 17.4 | 58.6 s | 82.4 s |
| b f32+npu | **15.41** (14.36-16.14) | 0.20 / 10.56 (7.67) / 4.65 | 17.4 | 46.6 s | 69.2 s |

Against the same-window p2 build (19.83 s, 92.2 s user) the deployed arm is
**-21.8 % tower** and **-10 % CPU**; against P2's own window (19.84 s, 112.7 s)
it is -23.5 %. Video (`data/test.mp4`, 4 frames, 4680 patches, arm a): tower
**21.95 s** (frames 5.57 / 5.42 / 5.81 / 5.14 s, 64 calls and 0.5 s of
conversions per frame), was 31.09 s in P2 (-29 %).

`serve` steady media (arm a, deferred towers, image-only `/embed`, port 8399):

| build | first request compute_s | steady compute_s | steady encode_s | RssAnon |
|---|---|---|---|---|
| p2 | 52.81 | 21.83 | 20.20 | 1,152,176 kB |
| p4, `ROCKET_NPU_FUSED=0` | 42.71 | 17.46 | 15.83 | 1,154,028 kB |
| p4, fused | 44.12 | **16.42** | **15.36** | 1,184,068 kB |

Steady media **-24.8 %** vs the same-window p2 build (-20.0 % per-projection).
A text request right after media logs `compute_s=0.263` with no `encode_s`, and
a text-only serve load is 619,828 kB RssAnon — the text path is untouched.

Parity, normalized 768-d embeddings vs the HF f32 reference (dev-host
`p1_ref_cat.json` / `p1_ref_video.json`):

| case | vs HF f32 | P2 |
|---|---|---|
| a q8+npu image | **0.9998836** | 0.9996880 |
| b f32+npu image | **0.9999963** | 0.9999975 |
| a q8+npu video | **0.9994970** | 0.9991586 |
| b f32+npu video | 0.9996491 | 0.9996499 |

The q8 arm's image parity improved by 2.0e-4 (its tower now packs f32 weights
like the f32 arm, so the 1.1e-4 that separates a from b is entirely the **text**
backbone's Q8 path, which this round does not touch); the video difference is
the reference's own frame-sampling gap (§18.3). Repeated runs are bit-identical
(cosine 1.0, max|d| 0), image and video.

#### 18.5.5 Gate assessment and decisions

- **Tower ≤16 s: met.** 15.17 s (a) / 15.41 s (b) from 19.83 s same-window,
  i.e. -22 %; the gate's alternative "user CPU -20 %" was not reached (-10 %,
  because the CPU attention kernel is unchanged at 7.6-8.2 s and now dominates
  the tower at 50 %).
- **Image cosine ≥0.9999 with the new default: met by the tower** (b f32+npu:
  0.9999963, i.e. f32-pack numbers) and **not by the q8+npu arm** (0.9998836,
  up from 0.9996880): both arms now share the same f32-packed tower, so what is
  left there is the text backbone's Q8-resident weights. The deployed q8 arm is
  therefore 1.2e-4 short of the bar for a reason outside the tower; `--dtype
  f32 --npu` without `--quant q8` clears it (0.9999963) at 1745 MiB anon instead
  of 908.
- **Text-only path unchanged**: no towers → no pack, same loader numbers, text
  request `compute_s=0.263`, and the non-NPU f32/q8 image path is unchanged
  (dev host: f32 cosine 1.0 / max|d| 0 vs the pre-round output, q8 0.999670 vs
  HF, the P1 value).
- **Landed defaults**: fused q|k|v and gate|up groups (`ROCKET_NPU_FUSED=0`
  restores 113 packs), f32 vision tower + Q8 audio under `--npu`, `gelu_mul` in
  the tower MLP whenever the glue is on, f32 attention kernel. Costs: +8 MiB of
  NPU weights and ~32 MB of matmul scratch in `serve`; the fused group's tower
  gain is small (-0.13 s, within rep noise) so the switch is cheap to flip back.
- **Next levers** (not done): the attention phase's 3.0 s outside the kernel —
  the two QK-RMSNorms, `AxialRope::apply` (4 slices + 6 elementwise ops + a
  `cat` per q and k, all flex composite chains on 2394 x 12 x 64) and the `o`
  projection — is now the largest addressable residual; an axial-RoPE host
  kernel of the `burn_rocket::rope_apply` kind is the obvious candidate
  (~2 s by the `gelu_mul` precedent). Then the audio tower's Q8 map (7.4 s of
  the deferred load) and the q8 text backbone's 1.1e-4 image-parity floor.

### 18.6 P5: frame batching (negative, 2026-10-24)

Hypothesis: video frames sampled at one fps share a geometry, so a batch of
frames could run as one tower forward — `[F*P, hidden]` projections (one NPU
matmul per weight instead of `F`) with the fused attention kernel taking `F` as
its batch dimension — amortizing the per-call conversions/marshalling. Expected
gate: 4 frames **>=15 %** below `4 x` single-frame, 8 frames >=20 %, image
latency unchanged, `RssAnon` +<=200 MB.

A candidate build implemented exactly that: `VisionTower::forward_batch`
(`&[PreparedImage]` of identical geometry -> one `[F*P, hidden]` patch matmul
with the shared position table, `[F, P, H, D]` RoPE/norms, fused attention
`[F, 12, P, 64]`, per-frame pooling), `MediaModel::encode_images` (default
per-image loop, EmbeddingGemma 2 override that groups same-geometry frames and
projects a group's pooled rows through `embed_vision` as one matmul), and
`ROCKET_VISION_BATCH` (default 4, `0`/`1` off) windowing the video loop so long
videos stream in bounded batches. Microbenchmark and gate measurement are the
same thing here: the real tower on the board, batch 1 vs 4, because a synthetic
tower cannot capture the allocator/NPU behaviour that turned out to decide it.

Three windows on `rock-5b-plus` (2026-10-24 08:01-08:24, 08:25-08:27 and
08:37-08:40 board time), production `home-llm-gateway-embed-1` stopped for each
and restored (`/health` ok), patched 600 MHz NPU module, `taskset -c 4-7`, 1-min
loadavg 2.3-3.5 (busier than P4's 1.5-1.8). Arm a (deployed shape):
`--dtype f32 --quant q8 --npu --npu-attn npu --attn-chunk 1024 --vision-attn
fused --defer-towers`; one process per run, so the 21.8 s deferred tower load is
outside every tower number. Candidate binary and logs:
`/root/rocket-inference-p5/` (`rocket-inference-p5`, 3 reps interleaved per arm).

#### 18.6.1 Result: batching does not help

Video `data/test.mp4` (1170 patches/frame): 4 frames = 4680 patches,
8 frames (`--video-fps 2`) = 9360. Tower seconds per run:

| frames | rep | batch 1 | batch 4 | delta |
|---|---|---|---|---|
| 4 | 1 | 18.60 | 18.11 | **-2.6 %** |
| 4 | 2 | 19.14 | 19.71 | +3.0 % |
| 4 | 3 | 17.39 | 19.16 | +10.2 % |
| 4 | mean | **18.38** | **18.99** | **+3.3 %** |
| 8 | 1 | 36.59 | 43.95 | +20.1 % |
| 8 | 2 | 37.52 | 38.93 | +3.8 % |
| 8 | 3 | 35.74 | 35.32 | -1.2 % |
| 8 | mean | **36.62** | **39.40** | **+7.6 %** |

Phase means over the 3 reps (tower seconds; `npu`/`convert` from the per-forward
lines):

| phase | 4f batch 1 | 4f batch 4 | 8f batch 1 | 8f batch 4 |
|---|---|---|---|---|
| patch | 0.20 | 0.22 | 0.32 | 0.31 |
| attention (of which kernel) | 11.07 (6.99) | 10.71 (6.32) | 22.87 (14.55) | 21.84 (13.21) |
| mlp | 7.09 | 8.05 | 13.41 | 17.23 |
| calls | 256 | **64** | 512 | **128** |
| convert | 1.45 | 1.93 | 2.86 | 4.47 |
| npu | 4.75 | 5.08 | 9.24 | 9.57 |
| wall / user CPU | 58.5 s / 89.8 s | 58.9 s / 91.4 s | 79.7 s / 141.1 s | 82.7 s / 149.1 s |

The **only** consistent gain is the attention op itself: **-9.6 %** at 4 and
**-9.2 %** at 8 frames (109 ms per frame-layer at 1170 patches in the per-frame
path, 99 ms batched), and the call count drops 4x. That is offset by the MLP
phase (+13.5 %) and the conversions (+33 % at 4 frames, +56 % at 8), so the net
tower change is **0 to +3 %** — the gate (-15 % at 4 frames) is missed by an
order of magnitude, and the 8-frame expectation too. `RssAnon` peaks are
indistinguishable (2.473-2.495 GB, dominated by the f32 load peak). The 8-frame
batch-4 mean is pessimistic: rep 1 ran into board contention (6.20 s of
conversions for the 8 patches-pairs against 2.83 s in rep 3), and the clean rep
is -1.2 %.

#### 18.6.2 Why: the tower is proportional-to-rows work

`vision_attn_bench` grew two primitive modes (`matmul`, `attn-batch`) that
isolate what batching could actually change, on the tower's real shapes.

NPU matmuls (16 layers of q|k|v 2304x768, o 768x768, gate|up 6144x768 and
down 768x3072; 3 reps, one warmup), per row:

| M | q\|k\|v | o | gate\|up | down |
|---|---|---|---|---|
| 1170 | 35.2 us | 14.4 us | 72.4 us | 35.6 us |
| 2340 | 20.9 us | 7.5 us | 65.8 us | 39.2 us |
| 4680 | 26.7 us | 11.9 us | 66.1 us | 51.9 us |
| 9360 | 29.6 us | 9.3 us | 64.1 us | 51.7 us |

No consistent per-row gain with `M` (gate|up, the biggest shape, is -11 % from
1170 to 9360; q|k\|v and down bounce with the board's NPU contention), i.e. the
driver is compute-bound at ~205 GF/s on these shapes and only per-call setup
scales with the call count. The real runs say the same: the P4 image (2394
patches) spends 1.30 ms of NPU time per row and 0.44 ms of conversion per row,
the 1170-patch frame 1.32 and 0.43. The fused attention op at 1170 patches is
flat per frame in batch: 0.093 / 0.091 / 0.092 / 0.091 s at B = 1 / 2 / 4 / 8,
because 12 heads already fill the four threads (the kernel parallelizes over
`batch * heads`).

So the batchable overhead is the ~0.17 s/frame of per-call setup inside the
attention phase (layout copies, output allocation, rayon job setup), and the
larger transient buffers of a batched call (`gate|up`'s 4680x6144 f32 output is
115 MB against 29 MB per frame) cost at least as much on this board's memory
path. Frame batching is therefore not the lever — the kernel share (P4's 7.6 s
of an image's 15.2 s) and the CPU glue are.

#### 18.6.3 Parity, memory and the serve A/B

Batch vs per-frame (independent of the batching code, same weights): normalized
768-d video cosine **0.999999885** at 4 frames and **0.999999944** at 8,
max|d| 6.3e-5 / 3.9e-5, i.e. above the 0.99999 gate; repeats of every arm are
bit-identical (cosine 1.0, max|d| 0). The batch-of-1 refactor is **exact**:
the P4 binary and the candidate produce bit-identical image vectors
(cosine 1.0, max|d| 0), and the image/video parity against the HF f32
references is unchanged (image 0.999883625 = P4's 0.9998836; video 0.9994970
per-frame vs 0.9994962 batched, P4 0.9994970). The 8-frame arm has no HF
reference (the reference samples 4 frames at fps 1) and is only compared
batch-vs-per-frame.

`serve` (deferred towers, port 8399, one request per 4-frame video):

| build | first request compute | steady compute | steady encode | RssAnon | matmul BO / host |
|---|---|---|---|---|---|
| batch 1 (kept) | 53.4 s wall | 24.62 / 25.90 | 21.15 / 22.86 | **1.181 GB** | 237 / 48 MB |
| batch 4 | 51.6 s wall | 25.12 / 24.57 | 21.61 / 21.26 | 1.236 GB | 603 / 129 MB |

Same compute within noise, **+55 MB** of `RssAnon` and 2.5x the NPU matmul
scratch (the batch's 4680-row capacity classes) for no gain — the memory cap
(+<=200 MB) would have held, but it buys nothing.

#### 18.6.4 Decision

**Negative; the video path stays per-frame.** The candidate's batching code was
not kept: `vision.rs`, `inputs.rs` and `embeddinggemma/model.rs` are back at the
P4 state (verified: the reverted build reproduces the P4 binary's image and
video vectors bit-for-bit), and `MediaModel` has no `encode_images`. What
remains from the round is the reproducible probe — `vision_attn_bench matmul
[reps]` and `vision_attn_bench attn-batch [patches] [reps]` (the candidate
binary and all logs stay under `/root/rocket-inference-p5/`). Re-open only if
the NPU matmul becomes call-overhead-bound or the attention kernel stops being
the largest single share: with a proportional tower, batching can recover at
most the per-call setup (~5 % of a frame). The NPU attention path's own blocker
is the driver's host softmax, deferred in §18.7.

### 18.7 Deferred

- **On-NPU softmax in `rocket-npu`** — the one lever that would re-open NPU
  vision attention. The port has no device-side softmax (root AGENTS.md
  "Missing surface"; the C library provides it), so `attention_window` /
  `attention_window_block` bring the full score matrix host-side and softmax it
  serially: at the tower's head_dim 64 that is 12 × 2394² = **68.8M scores per
  layer** (138 MB down + 138 MB up, f16) against only ~0.1 s of QK/AV GEMMs per
  layer, which is why §18.4 landed at 1.05× at best and why `--vision-attn`
  stays `fused|blocked`. Re-open after the port: re-run
  `vision_attn_bench 2394 5 1024,2048` and require the §18.4 gate (≥1.3× the
  fused kernel *and* ≥10 % tower wall) before adding `--vision-attn npu`.
  Shared with the causal models' long-context NPU attention, which has the same
  host-softmax limit.
- The remaining tower levers are the ones §18.5.5 lists (axial-RoPE host kernel
  ~2 s, the audio tower's Q8 deferred-load map, the q8 text backbone's 1.1e-4
  image-parity floor); frame batching is closed by §18.6.4.

## 19. S1: fast host softmax (2026-10-24)

§18.4 closed NPU vision attention at 1.05× the fused CPU kernel and blamed the
host score pass: 12 × 2394² = 68.8M scores per layer, a serial scalar
`f32::exp` over them, 138 MB read back and 138 MB uploaded as the AV operand.
The C library's on-NPU softmax is not the fix (it is a composed, transfer-bound
op — host row-max, NPU EXP LUT, NPU reduce, host reciprocal, NPU row scale — and
its own FA comment says it is 5-10 % *slower* than the host pass, which is why
the C FA defaults to host softmax). S1 instead made the **host pass fast**,
dependency-free, and measured whether that alone re-opens the gate.

### 19.1 Implementation (`crates/rocket-npu/src/attn/softmax.rs`)

`Attention::flash` no longer carries a local `softmax()`; each worker owns a
`Softmax` (`attn::Softmax`, `SoftmaxMode`) with one reused `f32` row buffer.
Two kernels:

- **Fast (default)** — `exp(x)` as `2^(x·log2 e)`: a degree-6 Chebyshev fit of
  `2^f` on `[-0.5, 0.5]` (`C0` pinned to 1, ≤2.5e-9 in f64, ~1e-7 in f32) with
  the exponent assembled from bits, branch-free (clamped at `2^-126`). On
  aarch64 the rows run 8 lanes per step: `fcvtl` f16→f32, vector max, 4-lane
  exp2 (`frintm` + `fma` Horner + `shl` exponent), `fcvtn` f32→f16, hardware
  throughout (`.cargo/config.toml` sets `+fp16`). The mask-free, softcap-free
  rows — the whole vision workload — take a single fully branch-free kernel;
  the masked path vectorizes `score·scale + mask` the same way, and the
  soft-cap path stays scalar (`tanh` has no cheap vector form) with the fast
  `exp`. Three passes over an L1-resident row (scale+max, exp+sum,
  `·1/sum`+f16), no per-row allocation.
- **Exact** — the pre-spike scalar `f32::exp` implementation, unchanged and
  bit-identical (`SOFTMAX_EXACT=1` or `ROCKET_SOFTMAX_EXACT=1` selects it
  process-wide; `Softmax::with_mode` selects per value). Semantics are shared:
  `clamp(softcap·tanh(s/softcap) + mask, -30000)`, a fully clamped row is
  uniform over `kn`, padded columns are clamped like masked-out keys.

`vision_attn_bench softmax [kn] [reps] [tp]` times the entry point alone on a
synthetic f16 `[tp, kn]` matrix (padded like the driver, `tp % 4`, `kn % 32`),
exact vs fast, with and without a causal mask, and prints the fast path's
deviation. Unit tests compare both kernels against the pre-spike reference on
random rows (f16 code distance ≤1 ULP, cosine >0.9999999), plus exp accuracy
vs `f32::exp`, monotonicity, padded columns, fully masked rows and soft-cap;
the NEON kernels were also run under `qemu-aarch64` (`cargo test -p rocket-npu
--target aarch64-unknown-linux-gnu`).

### 19.2 Softmax alone (board, 4 A76 threads not needed — one core)

`taskset -c 4-7`, production embed service stopped for the window, patched
600 MHz NPU module, 5 reps after a warmup. `GB/s` counts the f16 traffic
(read the scores + write the probabilities); one `exp` per element.

| shape | mask | exact | fast | speedup | fast GB/s | fast Gexp/s |
|---|---|---|---|---|---|---|
| 2396×2400 (vision, 2394²) | none (the tower) | 0.0672 s | **0.0155 s** | **4.33×** | 1.48 | 0.370 |
| 2396×2400 | causal | 0.0684 s | **0.0190 s** | **3.60×** | 1.21 | 0.302 |
| 64×3648 (text, 3633 keys) | none | 0.0031 s | **0.0006 s** | **4.86×** | 1.48 | 0.370 |
| 64×3648 | causal | 0.0027 s | **0.0006 s** | **4.28×** | 1.49 | 0.372 |

Both shapes sit at ~0.085 Gexp/s for the exact path and ~0.37 Gexp/s (2.7
ns/element, ~6 cycles) for the fast one, i.e. the cost is per-element, not
per-shape. Deviations from exact: cosine 0.999999999-1.0, max|d| 1.5e-5 (vision)
/ 7.6e-6 (text), **≤1 f16 ULP** everywhere (the f16 probability grid is 1e-3
relative, so the exp error is far below the output's own quantization).

The isolated number also corrects §18.4's attribution: the host pass is
0.067 s per head, and the op runs 3 heads per worker, so it is **~0.20 s of the
0.43 s op (47 %)**, not 0.33 s. The rest (~0.15 s after the softmax) is the
QK/AV GEMMs plus the two 138 MB BO transfers.

### 19.3 Op A/B (p = 2394, image size)

Three interleaved rounds (exact, fast, exact, fast, exact, fast), 1 warmup + 1
measured run each, `vision_attn_bench 2394 1 1024,2048`; medians below. The
`npu` column is the `burn_rocket::stats()` driver call (which contains the
softmax); `total` adds the f32→f16 head-major conversion.

| arm | npu call | total s/layer | vs fused CPU |
|---|---|---|---|
| fused CPU kernel (reference) | — | 0.456 | 1.00× |
| NPU, exact softmax | 0.432 | 0.459 | 0.99× |
| **NPU, fast softmax** | **0.204** | **0.229** | **1.99×** |
| NPU chunk-2048, exact | 0.438 | 0.444 | 1.03× |
| NPU chunk-2048, fast | 0.227 | 0.271 | 1.68× |
| NPU chunk-1024, exact | 0.436 | 0.436 | 1.05× |
| NPU chunk-1024, fast | 0.267 | 0.302 | 1.51× |

The softmax share of the driver call drops from 47 % to **25 %** (0.20 → 0.05 s
per layer), and the whole op from parity with the CPU kernel to ~2×. Whole-bench
CPU time per round: user 17.9-18.6 s exact vs 13.8-14.9 s fast (−21 %). Full
sequence beats chunking now (the per-chunk conversion and BO setup were what
chunking bought back when the softmax dominated); `RssAnon` is 174 MiB for the
full call vs 90-122 MiB chunked.

At the video-frame size (p = 1170, 2 reps) the same change turns §18.4's
0.73× into a win: fused CPU 0.088, NPU exact 0.097 (npu 0.087), **NPU fast
0.082 (npu 0.068) = 1.07×**.

Op-level parity against the f32 fused kernel is unchanged by the fast path:
cosine 0.999999844 / max|d| 3.19e-5 at p = 2394 and 0.999999872 / 4.12e-5 at
p = 1170, identical to the exact arm at the printed precision, bit-identical
across runs.

### 19.4 Text paths (general mask/soft-cap semantics)

Both runs on cores 0-3 with the production service live (contended wall, the
numbers are for direction only); repeats of the same mode are bit-identical, so
the deltas below are the mode difference.

| model | tokens | exact | fast | delta |
|---|---|---|---|---|
| qwen3 embed (causal mask, `--npu`) | 2777 | 58.38 s (47.6 tok/s), user 121.5 s | 50.0-50.3 s (55.6 tok/s), user 89.6 s | wall −11 %, CPU −26 %, cosine 0.999998366 |
| emb2 text (window/full masks, q8+npu attention) | 2594 | forward 26.76 s (attn 14.19) | forward 24.04-24.23 s (attn 11.79) | attn −17 %, forward −10 %, cosine 0.999977266 |

Both deltas are far below the models' own quantization floor (q8 text sits at
0.9996-0.9999 vs HF f32) and come from ≤1 ULP f16 probability differences
amplified by the layer stack; the deployed text embeddings change by ≤2.3e-5
cosine, and `SOFTMAX_EXACT=1` reproduces the old vectors bit-for-bit.

### 19.5 Gate and decision

The S1 gate was "op ≤0.35 s/layer **or** at least parity with the CPU kernel".
Fast NPU attention is 0.229 s/layer at p = 2394 (**1.99×** the 0.456 s fused
kernel) and 0.082 s at p = 1170 (1.07×), so it **passes** with margin.

**S2 should wire NPU attention as the tower default**: the tower's kernel time
(7.9-8.2 s per image, §18.4) becomes ~3.5 s and the 15.2-15.5 s tower ~11 s
before any in-situ overhead, clearing the §18.4 ≥10 % wall gate. Do not port the
C library's on-NPU softmax for this: the host pass is now 0.05 s/layer, and the
remaining op time is GEMM + BO traffic. The causal models' long-context NPU
attention benefits from the same path (qwen3 2777 tokens: −11 % wall, −26 %
CPU) with no separate work.

Reproduce (board, in a service-free window):

```sh
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu --bins
# exact vs fast, both shapes, plus the op A/B (SOFTMAX_EXACT=1 = exact arm)
taskset -c 4-7 ./vision_attn_bench softmax 2394 5
taskset -c 4-7 ./vision_attn_bench softmax 3633 5 64
SOFTMAX_EXACT=1 taskset -c 4-7 ./vision_attn_bench 2394 5 1024,2048
taskset -c 4-7 ./vision_attn_bench 2394 5 1024,2048
```

## 20. S2: NPU vision attention wired (2026-10-24)

§19 left the decision made and the code unwired: the fast host softmax took
`attention_window` to 1.99× flex's fused kernel at the tower's image geometry,
so the tower's attention should move to the NPU. S2 adds the mode, makes it the
`--npu` default for EmbeddingGemma 2, and measures the tower, the serve path and
the parity on the board.

### 20.1 Implementation

- `VisionAttn` gains `Npu`; `--vision-attn` accepts `npu|fused|blocked` on the
  `gemma` CLI and on `serve` (EmbeddingGemma 2 only; Gemma 4 keeps its hardcoded
  `blocked` tower and rejects the flag). `npu` is scoped: it requires `--npu` on
  an aarch64 build with `--features npu` and is rejected with a clear message
  otherwise (`--vision-attn npu requires --npu (the vision tower's attention
  runs on the NPU)`, `... requires an aarch64 build with --features npu`). There
  is no off-device implementation to fall back to, so it is not silently
  downgraded.
- `VisionAttention::forward`'s NPU arm calls
  `burn_rocket::attention_window(q, k, v, heads, kv_heads, head_dim, 1.0, None,
  -1)` on the `[1, patches, H*D]` layout, i.e. the whole patch sequence, no mask,
  scale 1.0 — exactly the P3 bench's `npu` arm (§18.4). The projection outputs
  are `[patches, heads, head_dim]`, which reshape to `[1, patches, H*D]` for the
  op; its `[1, patches, H*D]` result reshapes back for the packed `o_proj`. No
  chunking: after S1 the full sequence is the fastest arm at both the image and
  the video shape (§19.3). The fused and blocked arms run exactly the previous
  ops (their `[h, p, d] -> [p, H*D]` reshape just moved inside the arm).
- Default resolution, both entry points: an explicit `--vision-attn` wins; else
  `npu` when the tower's NPU attention path is actually in use (`serve`'s
  resolved `npu`: default-on for an aarch64 `npu` build, so `--no-npu` or a
  `--quant q8` without an explicit `--npu` falls back to `fused`) and `fused`
  otherwise (CPU builds, `--no-npu`, Gemma 4). `serve` prints the resolved mode
  as `vision_attn="npu"` on every media request log line, and the tower line
  names it too.
- The tower's attention is independent of `--npu-attn` (that flag stays the text
  backbone's). `ROCKET_NPU_TOWERS=0` (tower projections on the CPU) combines
  with `--vision-attn npu`: only the attention op moves to the device then.

### 20.2 Board A/B (2026-10-24, 07:22-07:46 UTC)

Window: production `home-llm-gateway-embed-1` stopped and restored (`/health`
ok), patched 600 MHz NPU module, `taskset -c 4-7`, 1-min loadavg 1.9 at the
start rising to 3.6-4.3 with the board's other containers during the window.
Safetensors model `/root/models/embeddinggemma-2`, one process per run
(`gemma embed`), `--defer-towers` so the tower load (14.7-15.4 s in the f32
arms, 22.3-23.2 s with the q8 audio map) stays outside every tower number;
3 interleaved reps per arm (order per rep: a image fused,
b image fused, a image npu, b image npu, then the same for video). Arms:

- **a** deployed shape: `--dtype f32 --quant q8 --npu --npu-attn npu
  --attn-chunk 1024 --defer-towers --vision-attn <mode>`
- **b** f32+npu: the same without `--quant q8`

#### Image (2394 patches, `data/cat.jpeg`)

Mean of 3 reps (range in parentheses), tower seconds:

| arm | vision-attn | tower | patch / attn (kernel) / mlp | encode | wall | user CPU | peak RssAnon |
|---|---|---|---|---|---|---|---|
| a q8+npu | fused | 13.84 (13.51-14.03) | 0.16 / 9.98 (**7.73**) / 3.70 | 15.95 | 53.8 s | 76.4 s | 2419 MiB |
| a q8+npu | **npu** | **10.57** (9.47-11.47) | 0.17 / 5.99 (**3.44**) / 4.41 | **12.66** | 50.5 s | **58.6 s** | 2420 MiB |
| b f32+npu | fused | 14.07 (13.85-14.21) | 0.15 / 9.79 (**7.60**) / 4.13 | 16.17 | 41.5 s | 62.0 s | 2408 MiB |
| b f32+npu | **npu** | **10.58** (10.46-10.66) | 0.16 / 5.89 (**3.48**) / 4.52 | **12.71** | 37.7 s | **44.1 s** | 2510 MiB |

The tower drops **-23.6 %** (a) / **-24.8 %** (b) and the attention op **-55 %**
(7.7 → 3.4 s); everything else in the attention phase (norms, axial RoPE, the
`o` projection) is unchanged, so the phase goes 10.0 → 6.0 s while the MLP phase
does not depend on the mode (see below). Whole-process wall -6.1 % / -9.2 % and
**user CPU -23 % / -29 %** on the image, which is the point of the offload: the
process now needs 58.6 s of user CPU instead of 76.4 s. Peak `RssAnon` is
unchanged in the q8 arm (the ~2.4 GiB f32 load peak dominates) and +102 MiB in
the f32 arm (the attention context; the S1 bench measured its geometry at
72 MiB of device BO + 44 MiB of host scratch).

#### Video (`data/test.mp4`, 4 frames, 4680 patches)

| arm | vision-attn | tower | patch / attn (kernel) / mlp | encode | wall | user CPU | peak RssAnon |
|---|---|---|---|---|---|---|---|
| a q8+npu | fused | 18.55 (17.87-19.50) | 0.17 / 11.39 (**7.17**) / 6.98 | 21.16 | 60.1 s | 82.6 s | 2418 MiB |
| a q8+npu | **npu** | **17.21** (16.92-17.56) | 0.22 / 8.50 (**4.00**) / 8.48 | 19.84 | 58.5 s | **70.9 s** | 2424 MiB |
| b f32+npu | fused | 18.57 (18.29-18.83) | 0.20 / 11.57 (**7.31**) / 6.79 | 21.18 | 47.8 s | 69.0 s | 2391 MiB |
| b f32+npu | **npu** | **17.28** (15.58-18.13) | 0.19 / 8.21 (**3.80**) / 8.83 | 19.88 | 46.4 s | **58.0 s** | 2389 MiB |

At the frame shape the op win is still large (**-44 % / -48 %**, 7.2-7.3 →
3.8-4.0 s) but the tower only improves **-7 %**: the per-frame attention phase
falls 11.4 → 8.5 s, yet the MLP phase reads 1.5-2.0 s higher in that arm, which
eats most of it. That is window drift, not a cost of the NPU attention: a
second, busier window (07:50-07:57 UTC, loadavg 2.0 → 4.4) that alternated modes
(`npu, fused, fused, npu`, video, then image) reproduced the MLP phase at a
higher level in *both* modes — video MLP **8.73 / 8.79 s fused** vs **9.08 /
10.37 s npu** with the same tower 22.0-22.2 vs 18.4-21.1 s, and image MLP
**4.70 fused / 4.78 npu** (tower 15.48 → 11.75). The CPU-side phases of that
window ran 12-25 % slower than the A/B's (its fused video tower is 22.1 s
against 18.6 s), i.e. the A/B's MLP difference is the board's contention ramping
between the earlier and later runs of each group, not the attention mode. The
video arm's real gain is the CPU relief (-14 % / -16 % user) and a small wall win
(-2.6 % / -2.9 %).

### 20.3 Parity and determinism

Normalized 768-d embeddings vs the HF f32 reference (regenerated on the dev host
with `tools/ref_embeddinggemma2.py`, transformers 5.19 CPU f32; the P4 column is
§18.5.4's number for the same reference):

| case | fused | **npu** | P4 fused |
|---|---|---|---|
| a q8+npu image | 0.999883357 | **0.999884446** | 0.9998836 |
| b f32+npu image | 0.999997076 | **0.999989974** | 0.9999963 |
| a q8+npu video | 0.999496202 | **0.999505592** | 0.9994970 |
| b f32+npu video | 0.999649039 | **0.999653036** | 0.9996491 |

The NPU arm is within **1e-5** of the fused one in both directions — 7.1e-6
lower on the f32 image, 1.1-9.4e-6 higher at the other three points, i.e. the
S1 softmax deviation is the same order as the fp16 GEMM noise the tower already
carries. Fused-vs-npu at the
final embedding: **0.999986356** (image, max|d| 6.3e-4) and **0.999998268**
(video, 3.7e-4), i.e. above the 0.9999 op-parity gate, and both arms' repeats
are **bit-identical** (cosine 1.0, max|d| 0, three runs each, image and video).
The video reference's own ~3.5e-4 frame-sampling gap (§18.3) is unchanged.

The serve path is the same vector: an image request through the default
(`--npu`, no `--vision-attn`) is bit-identical to the CLI's `--vision-attn npu`
run, and a text request is bit-identical between the fused and npu serves
(cosine 1.0) — the text backbone does not see the tower.

### 20.4 Serve steady state (image, port 8399)

Both modes: `--dtype f32 --quant q8 --npu --npu-attn npu --attn-chunk 1024
--defer-towers`, deferred towers, image-only `/embed`, first request then 3
steady requests, then a text request (`compute_s` from the request log):

| vision-attn | first compute (encode) | steady compute | steady encode | RssAnon | text compute |
|---|---|---|---|---|---|
| fused | 40.26 s (12.66) | 13.38 / 13.73 / 13.51 (**13.54**) | 12.07-12.36 (**12.17**) | 1161 MiB | 0.263 s |
| **npu** | **38.58 s** (10.50) | 12.39 / 12.25 / 12.69 (**12.44**) | 10.82-11.37 (**11.06**) | 1262 MiB | 0.251 s |

Steady media **-8.1 %** compute and **-9.1 %** tower, +101 MiB anon (the
attention context). The tower's geometry (12 heads, 12 KV heads, head_dim 64) is
a third entry in the engine's per-geometry attention-context cache — the text
backbone's sliding (4, 2, 256) and full (4, 1, 512) layers are the other two —
so no context is created per layer or per request, and the slot count stays
inside the engine's four. The per-request tower lines show the same split as
the CLI
(kernel 6.2 → 3.3-3.8 s; 64 → 80 NPU calls, +0.45 s of conversions, +3.5 s of
device time for 16 layers ≈ the S1 op's 0.22 s/layer); the fused serve's kernel
is faster than its CLI number (6.2 vs 7.7 s), presumably because the steady
process's longer-lived buffers and warm caches suit flex's kernel better, so the
same-window serve delta is the smaller -9 % and the CLI's -24 % is the
cold-request number. Text is untouched (0.263 → 0.251 s, within noise).

### 20.5 Gate and decision

- Gate: image tower **-24 %** on the CLI harness — the §18.4 ≥10 % tower-wall ask
  is met (the warm serve harness lands at -9 %, because the fused kernel speeds
  up more with warm buffers); parity **≥0.9995** q8+npu (0.999884) and
  **≥0.9999** f32+npu (0.999990), fused-vs-npu **≥0.9999** (0.999986);
  determinism bit-identical; text unchanged.
- **Landed:** `VisionAttn::Npu`, `--vision-attn npu|fused|blocked`, and `npu` as
  the EmbeddingGemma 2 default whenever the NPU path is active (`--npu` on an
  aarch64 `npu` build, including `serve`'s default-on NPU offload);
  `--vision-attn fused` stays the explicit CPU fallback and `blocked` the parity
  reference. Cost: +101 MiB anon on media requests, +16 NPU calls and ~0.45 s of
  host conversions per forward; no loader change, no change to the text path.
- The next tower levers are now the attention phase's ~2.5 s outside the op
  (two QK-RMSNorms, axial RoPE, o projection — an axial-RoPE host kernel is the
  §18.5.5 candidate) and the CPU-side MLP glue; the kernel itself is 3.4 s of
  the 10.6 s tower. On the causal text models the same S1 softmax already
  applies, and the tower's `npu` default does not change their budget.

Reproduce (board, service-free window, `--bins` for the bench):

```sh
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu --bins
# A/B (one process per run; the tower load is printed by the `towers:` line)
for mode in fused npu; do
  taskset -c 4-7 ./rocket-inference gemma embed --model-dir /root/models/embeddinggemma-2 \
      --dtype f32 --quant q8 --npu --npu-attn npu --attn-chunk 1024 --defer-towers \
      --vision-attn $mode --image data/cat.jpeg --out /tmp/$mode.json
done
# serve steady state (logs vision_attn= and encode_s per request)
ROCKET_VISION_LOG=1 ./rocket-inference serve --model-dir /root/models/embeddinggemma-2 \
    --dtype f32 --quant q8 --npu --npu-attn npu --attn-chunk 1024 --defer-towers --port 8399
```

## 21. Video input: reference layout, frame extraction, container ffmpeg (2026-10-24)

The video path (§4) put frames in the shared context from the start, but the
deployment image never had a decoder: the runtime stage installed only
`curl`/`ca-certificates`, so `/embed` video, `/v1/embeddings` `input_video` and
non-WAV audio failed in the container with `run ffprobe (is it installed?)` /
`run ffmpeg`. This round pins the reference layout in code and tests and ships
ffmpeg in the image.

### 21.1 Layout (already the documented one)

`EmbeddingGemma2Processor.replace_video_token` (transformers 5.19) with
`add_timestamps=false` — the checkpoint's `video_processor` config — expands one
`<|video|>` placeholder into `num_frames` blocks of
`BOI + <|video|> * n + EOI`, concatenated without separators, where `n` is the
per-frame soft-token count (140 at the default `max_soft_tokens`) and `BOI`/`EOI`
are `<|image>`/`<image|>`. Frames are sampled at 1 fps
(`step = native_fps / fps`, `min(total-1, int(i * step))`) and capped to
`max_frames` with `np.linspace(0, n-1, max_frames)`
(`overflow_strategy="uniform"`). That is what the app already did; measured
against the HF f32 reference on the dev host (`--dtype f32`,
`tools/video_parity.py`):

| case | ids (ref = ours) | frames | cosine |
|---|---|---|---|
| video only (`data/test.mp4`, 4 s at 8 fps) | 530 | 4 | 1.000000000 |
| text + image + video (model-card interleaving) | 817 (520 video + 266 image soft tokens) | 4 | 1.000000000 |
| 40 s clip, uniform overflow | 4226 | 32 | 1.000000000 |

The 40 s row also pins the sampler: 40 frames at 1 fps are uniformly re-sampled
to 32, and both sides encode the same 32 native indices.

### 21.2 Frame extraction fix

`extract_frames` asked ffmpeg for the sampled indices with
`select='eq(n\,i)+...'`. `select` matches a frame at most once, so repeated
indices (the sampler clamps to `total-1`, so metadata that overestimates the
frame count repeats the last index) silently produced fewer frames than
requested, while the reference indexes the decoded array and keeps the repeats.
It now requests the distinct indices once and replicates them, so
`frames.len() == indices.len()`. The pathological clip that exposed this
(`-stream_loop` + `-c copy`: ffprobe reports 40 fps / 40.21 s while 322 frames
decode) is not the fixture — the parity script re-encodes its 40 s clip at 8 fps
so the container metadata is exact.

Unit tests pin both halves without a model or ffmpeg:
`media::tests::frame_indices_match_the_reference_processor` (six rate/overflow
cases, including NTSC 30000/1001 and the clamp) and
`inputs::tests::video_placeholder_expands_to_one_block_per_frame`.

### 21.3 Container

The runtime stage now installs Debian's `ffmpeg` 7.1.5 (`ffmpeg` + `ffprobe`,
`--no-install-recommends`, gated on `-version` in the build). Measured on
`debian:trixie-slim` (amd64): `/usr` grows 78.5 MB → 503 MB (**+426 MiB**, 207
packages; `libavfilter` and `libavdevice` pull Mesa/X11/SDL), the apt layer is
464 MB per `docker history`, and the runtime image is **809 MB** (562 MB rootfs;
the pre-change image was base + binary only). The build-time
`ffmpeg -version`/`ffprobe -version` calls fail the image build if a future
package split moves either binary.

Container smoke (amd64 image, `serve --backend flex --dtype f32`, model mounted
read-only, port 8395):

| request | result |
|---|---|
| `POST /embed {"video": "/data/test.mp4"}` | 200, 768-d, cosine **1.000000000000** vs the CLI f32 vector (530 tokens, `encode_s=9.401`) |
| `POST /v1/embeddings` with an `input_video` `data:video/mp4;base64,...` block | 200, 768-d, cosine **1.000000000000** vs the same vector |
| `POST /embed {"audio": "/media/speech5s.mp3"}` (non-WAV) | 200, 768-d, norm 1.0; mp3-vs-WAV cosine 0.985605 (lossy codec) |

These requests were decoded by the installed package — nothing else in the
container provides ffmpeg, and the pre-change image answered
`run ffprobe (is it installed?)`.

### 21.4 Reproduce

```sh
cargo build --release -p rocket-inference --no-default-features
/mnt/hub/venvs/emb2/bin/python examples/rocket-inference/tools/video_parity.py \
    --bin $CARGO_TARGET_DIR/release/rocket-inference
docker build -f examples/rocket-inference/Dockerfile -t rocket-inference:video .
docker run --rm --entrypoint ffprobe rocket-inference:video -version
```

## 22. Split deferred tower load (2026-10-27)

`--defer-towers` used to install the whole multimodal set on the first media
request: one `Option<Towers>` holding vision and audio, one store pass, one Q8
mapping, then the vision NPU pack. An image or video request therefore allocated
and kept the audio tower too — the larger half — and paid its f32 transient
before the Q8 mapping (or kept it resident in f32 on Gemma 4).

| half | checkpoint (bf16) | f32 | Q8 (`--quant q8`) | tensors |
|---|---|---|---|---|
| vision + `embed_vision` | 320 MiB | 638 MiB | ~170 MiB (0.28 GiB NPU-packed under `--npu`) | 211 |
| audio + `embed_audio` | 583 MiB | 1163 MiB | ~300 MiB | 272 |

### 22.1 Structure

The set is now two independent halves per family, each behind a
`#[module(skip)]` field:

- EmbeddingGemma 2: `Emb2Model.vision_towers: Option<VisionTowers>`
  (`vision_tower`, `embed_vision`) and `.audio_towers: Option<AudioTowers>`
  (`audio_tower`, `embed_audio`); `VisionTowers::pack_npu` keeps the NPU pack.
- Gemma 4: the same pair on `GenInner` (`model.vision_tower.*`,
  `model.audio_tower.*`), with `VisionTower::new_with_clip` and the per-half
  SRQ registration inside the towers pass.

`gemma::TowerSet { vision, audio }` carries the need. The server derives it from
the request (image/video -> vision, audio -> audio; a mixed `/v1/embeddings`
batch or `/embed` body sets both), the CLIs from their media flags, and eager
runs pass `TowerSet::ALL`. `load_towers(..., need)` first subtracts
`towers_loaded()`, so a later request for the other modality loads only the
missing half. Each half is loaded directly into its part struct (the fields keep
the checkpoint names, so the file keys match without remapping); the other
half's tensors report as unused, which the focused-pass checks already allow.
The `mmproj-*.gguf` path is split the same way (only the needed clip scalars are
read).

### 22.2 Measurements (dev host, 32 threads)

EmbeddingGemma 2, safetensors, `--quant q8 --defer-towers`, one media request
per fresh process (idle text-only process 510 MiB `RssAnon`):

| first request | before | after |
|---|---|---|
| image (`/embed`, 266 soft tokens) | **+640 MiB**, tower load 7.2 s | **+275 MiB**, 2.5 s (vision only) |
| audio (`speech5s.wav`, 125 soft tokens) | **+569 MiB**, 7.2 s | **+373 MiB**, 4.8 s (audio only) |

The deltas include the request's own activations; the tower halves are ~170 MiB
(vision) and ~300 MiB (audio) of Q8 weights. A second image request adds no
`towers: loaded` line, a following audio request loads only the audio half
(272 tensors), and the reverse order loads audio first and then vision.

Gemma 4 E2B QAT mobile, `--f16 --defer-towers` (`gemma gen`, 12 tokens):

| first media | before | after |
|---|---|---|
| image | 6543 MiB `RssAnon`, towers 4.1 s | **5286 MiB**, 1.6 s (vision, 211 tensors; 112 SRQ linears) |
| audio | 6543 MiB, 4.1 s | **5865 MiB**, 2.5 s (audio, 272 tensors; 120 SRQ linears) |

Eager runs (no `--defer-towers`) still load both halves at startup, one store
pass per half; the logs show two `towers: loaded` lines
(`..., 211 tensors` / `..., 272 tensors`).

### 22.3 Parity

Every arm is bit-identical to the pre-split binary (vectors/ids, not just
cosine): emb2 text (eager), image and audio (deferred safetensors), and image
and audio through the production-style GGUF path
(`-hf unsloth/embeddinggemma-2-GGUF:Q8_0`, `mmproj-Q8_0.gguf`); Gemma 4 QAT
mobile text, image and audio, deferred and eager. The GGUF server log shows
`towers: loaded vision from mmproj (...)` followed by
`towers: loaded audio from mmproj (...)` only when the audio request arrives.

### 22.4 Reproduce

```sh
cargo build --release -p rocket-inference --no-default-features
# image first, then audio (each on its own line; audio does not load with the image)
$CARGO_TARGET_DIR/release/rocket-inference serve --model-dir /mnt/hub/models/embeddinggemma-2 \
    --quant q8 --port 8399 --max-tokens 512 --defer-towers
curl -s localhost:8399/embed -H 'Content-Type: application/json' -d '{"image":"data/cat.jpeg"}'
curl -s localhost:8399/embed -H 'Content-Type: application/json' -d '{"audio":"data/speech5s.wav"}'
# the production checkpoint format (downloads Q8_0 + mmproj)
$CARGO_TARGET_DIR/release/rocket-inference serve -hf unsloth/embeddinggemma-2-GGUF:Q8_0 \
    --dtype f32 --quant q8 --port 8399 --max-tokens 512 --defer-towers
# Gemma 4 (placeholders are the CLI's, unlike the OpenAI content parts)
$CARGO_TARGET_DIR/release/rocket-inference gemma gen \
    --gen-model-dir /mnt/hub/models/gemma-4-E2B-it-qat-mobile --f16 --defer-towers \
    --image data/cat.jpeg --text "Describe this image briefly. <|image|>" --max-new-tokens 12
```
