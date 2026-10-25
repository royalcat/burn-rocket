# burn-rocket

RK3588 NPU offload for [Burn](https://burn.dev) models, through the mainline `rocket`
driver: a pure-Rust userspace driver (`rocket-npu`, in `crates/rocket-npu`) and a set
of Burn **backend-extension** ops that run a model's projection matmuls and masked
grouped-query attention on the RK3588 NPU while everything else stays on the CPU
backend.

The library is `aarch64`-only for anything that touches the NPU: the driver talks to
`/dev/accel/accel0` through the in-kernel `rocket` driver. There is no C toolchain,
CMake, kernel-header, or vendor-archive step: the register programs, tiling planner,
buffer objects, matmul and attention are all Rust in `crates/rocket-npu` (an
interface-compatible port of `librocketnpu`, byte-identical register programs and
planner output).

## AI Disclosure

This project was fully developed by AI (coding agents under human direction). Every
result was validated on real hardware: numerics against reference implementations
(cosine parity, token-identical output) and performance on the dev host and the
Rock 5B+ board — nothing is taken on trust.

## Inference server

The examples are complete HTTP servers with OpenAI- and Ollama-compatible APIs, so
standard clients work unchanged:

| model | API | command |
|---|---|---|
| Qwen3-Embedding-0.6B | OpenAI `/v1/embeddings`, `/v1/models`, `/health` | `rocket-inference serve --model-dir /models/qwen3-embedding-0.6b` |
| Qwen3.5-0.8B intent/query planner | OpenAI `/v1/chat/completions`; Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` | `rocket-inference serve --model-dir /models/ov-intent-analysis-sft` |
| EmbeddingGemma 2 (text/image/video/audio) | OpenAI/OpenRouter `/v1/embeddings` (text + `image_url`/`input_audio`/`input_video` blocks) + native multimodal `/embed` | `rocket-inference serve --model-dir /models/embeddinggemma-2` |
| Gemma 4 E2B-it chat (text/image/audio) | OpenAI `/v1/chat/completions`; Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` | `rocket-inference serve --model-dir /models/gemma-4-E2B-it` |

The container image ships `ffmpeg`/`ffprobe`, so EmbeddingGemma 2's
`/v1/embeddings` accepts `input_video` blocks and non-WAV audio without extra
host packages.

One `serve` command covers every model: it detects the checkpoint's family and
registers exactly the endpoints that model supports (embedding models get
`/v1/embeddings`/`/embed`, chat models get both the OpenAI chat and the Ollama
APIs).

OpenViking's embedding backend in production on a Rock 5B+ is EmbeddingGemma 2
(`rocket-inference serve --dtype f32 --quant q8 --npu --npu-attn npu
--defer-towers`): the text-only service idles at 605 MiB anon (2183 MiB before the
idle-memory round; the vision/audio towers load only if a media request ever
arrives). On a media request the towers' projections and the vision tower's
attention now run on the NPU (`--vision-attn` defaults to `npu`): a
2394-patch image's tower is 10.6 s against 15.2 s with the CPU attention kernel,
and a served image request takes 12.4 s instead of 13.5 s at +101 MiB anon. Its
3,633-token predecessor configuration (Qwen3-Embedding-0.6B on the
NPU, `--npu --npu-attn cpu --max-tokens 8192`) ran a 3,633-token input in ~87 s
(~42 tok/s on 4 A76 cores) at ~34% fewer CPU-seconds than the CPU-only flex path
(the default NPU attention is slightly faster: ~45 tok/s, -51% CPU). The intent
model answers the 166-token v7 planner prompt in 11.7 s with `--npu` (89.9 s
CPU-only). Flags, endpoints and deployment:
[examples/rocket-inference/README.md](examples/rocket-inference/README.md).

Container images for both architectures are built and published to GHCR on every
push to `main`: `ghcr.io/royalcat/rocket-inference:<sha>` / `:latest` is the aarch64
RK3588 image (NPU offload is `serve`'s default there; `--no-npu` forces the flex CPU
path) and `:latest-amd64` is an x86-64-v3 (AVX2) flex-only CPU image
(`.github/workflows/build-images.yml`).

## What it provides

- **Burn backend extension** (`#[backend_extension(Flex)]`, default `npu` feature): ordinary
  `Tensor`s in and out, no FFI in model code.

  ```rust,ignore
  burn_rocket::init(5)?;                       // once, before any op

  let qkv = burn_rocket::pack3(wq, wk, wv);    // [N, K] f32 weights, pack-and-drop
  let qkv_out = burn_rocket::matmul(x, &qkv);  // [.., M, K] @ [N, K]^T -> [.., M, N]

  let out = burn_rocket::attention(
      q, k, v, 16, 8, 128, scale, None, true,  // n_head, n_kv_heads, head_dim, scale, softcap, causal
  );

  // bidirectional band attention (encoders): |q - kv| <= window, window < 0 = no mask
  // (a no-mask block may have n_k != n_q, which is how full layers are chunked)
  let out = burn_rocket::attention_window(
      q, k, v, 4, 2, 256, 1.0, None, 512,
  );

  // causal attention over a query chunk against its key prefix (decoders): the
  // mask/score matrix is [n_q, n_kv], not the model's whole [n, n]
  let out = burn_rocket::attention_causal_block(
      qc, kc, vc, 16, 8, 128, scale, None, q_start, kv_start,
  );
  // ... and with a sliding window (t - window < j <= t)
  let out = burn_rocket::attention_causal_block_window(
      qc, kc, vc, 16, 8, 128, scale, None, 512, q_start, kv_start,
  );
  ```

  `pack`/`pack2`/`pack3` consume f32/f16 weights and keep fp16 resident copies in NPU
  memory for the process lifetime (`WeightId` handles), and bf16 weights pack the same
  way via an exact bf16->f16 conversion. Fused packs
  concatenate several weights that share one input along N, so one matmul produces all
  their outputs (the example packs q|k|v and gate|up that way). Matmuls are chunked
  above `ROCKET_MATMUL_CHUNK_M` rows (env, default 8192, `0` disables) to bound the
  per-call input buffer; rows are independent, so chunking is bit-identical.

  The per-shape scratch cache is bounded: `M` is keyed by capacity classes (≤25 %
  padding, live rows planned per call so padded rows are never computed), the
  streaming-weight and CPU-accumulation BOs are allocated lazily, and
  `ROCKET_SCRATCH_MB` (default 1024, `0` = unlimited) caps the cache with
  free-before-allocate LRU. A served request keeps its own shape set; the next
  request's trim drops earlier sets (`burn_rocket::trim_scratch`, run by the
  example after each request; `ROCKET_TRIM=0` disables it). `ROCKET_M_BUCKET=0`
  restores exact-`M` slots, and `ROCKET_MEM_LOG=1` prints the live split
  (weights / matmul BO+host / attention BO+host).

  Causal decoding attention has a chunked entry too: `attention_causal_block` (and
  `attention_causal_block_window` for a sliding layer) takes `q_start`/`kv_start` as the
  chunk's absolute positions, so one call covers a query chunk against just its key
  prefix and the additive mask and score matrix stay `[n_q, n_kv]`. That is what lets
  a causal model run 30k-token prompts without the dense `[n, n]` f16 mask
  (~1.8 GiB at 30k) — see the example's `--npu` path, which chunks at
  `ROCKET_CAUSAL_CHUNK` (default 1024, `0` = dense).

- **Low-level driver API** (`rocket_npu::Device`, `rocket_npu::matmul::Context`,
  `rocket_npu::matmul::Weight`, `rocket_npu::attn::Attention` and its stateless
  `flash_attn`/`flash_attn_mt`/`flash_attn_ref`/`mha_self`/`mha_self_ref` siblings,
  the `rocket_npu::ops` families — activation/elementwise, reduce/softmax, the
  transformer and vision norms, the gated FFN, the encoder block and the fp16
  convolution family (direct/depthwise/1D, the conv→activation fusion and the
  transposed conv) — each with its device-free `*_ref`,
  `f16` helpers, driver/counter helpers) for callers that do not want Burn tensor
  routing. The `RocketCtx`/`RocketFaCtx` compatibility wrappers expose the old
  surface. See `examples/probe.rs` for a runnable smoke test.

The ops are **inference-only**: the extension macro generates no autodiff and calling
them with an autodiff context panics inside Burn. The driver contexts are not
thread-safe; the extension routes every call through one global engine behind a mutex
(`burn_rocket::stats()` reports accumulated convert/NPU time). Resident weights use
the canonical tiling (`M >= 4`; `ROCKET_CTX_CANONICAL=0` restores the legacy
`M >= 256` padding).

## Features

| feature | effect |
|---|---|
| `default` | `npu` — NPU offload is the first-class target |
| `flex` | compiles against Burn's `Flex` backend (required by the extension catalog; the feature name must stay exactly `flex`) |
| `npu` | the `RocketOps` extension ops on the pure-Rust driver (implies `flex`) |

`--no-default-features` gives the no-Burn crate (helpers and masks only);
`--no-default-features --features flex` the Burn dependency without the extension
(feature-graph check).

## Building

There is no external driver archive to build. `rocket-npu` is a workspace member and
compiles on any host (device access is runtime-only):

```sh
# compile check (npu is the default; works on any target, no linking)
cargo check -p burn-rocket

# aarch64 build of the probe example
cargo build --release -p burn-rocket --example probe \
  --target aarch64-unknown-linux-gnu
```

The example app builds with `--no-default-features --features npu` (aarch64) or
`--no-default-features` (flex CPU on any target).

Port status: the fp16 matmul (one-shot, streaming, resident, segmented weights),
dense and group-wise int8 (one-shot, plus resident per-channel/group-wise weights),
dense and group-wise int4 (one-shot with the C's int16-saturation `kt_cap`, plus the
resident `I4Context` in both modes and its planner), `int16_exact`, the native bf16
stream, the tf32 one-shot (raw-f32 operands; the MAC truncates to a 10-bit mantissa,
fp32 out with f64 host K-accumulation), masked GQA attention, the activation +
elementwise family (`rocket_activation.c`: every `ROCKET_ACTIVATION_*` kind, the
parametric LeakyReLU/ELU/SELU/PReLU, the conv-main EW mul/add/sub/max/min, clip and
divide), the reduce + softmax family (`rocket_reduce.c`, `rocket_softmax.c` and
`rocket_scale_rows_fp16`: the PPU global average/max/min pools with the telescoping
decomposition, the feature-axis ones-matmul reduce, the triangular-ones cumsum, row
softmax/log-softmax/cross-entropy and the per-row broadcast scale), the transformer
norms (`rocket_norm.c`: RMSNorm — fp16 square under the exact power-of-two prescale,
fp32-accumulated mean-square, host rsqrt, one folded scale — and LayerNorm with the
C's stacked `[x ; x²]` single-reduce job and the `x·A + B` affine fold), the vision
norms (`rocket_normvision.c`: inference BatchNorm from running stats, GroupNorm,
InstanceNorm and L2-Normalize over the `[N,C,P]` channels-major layout), the gated FFN
family (`rocket_ffn.c`: the GeGLU/SwiGLU core, the three-matmul block, the pinned-tile
`ffn_fused` and `mlp_fused`, the host-handoff `mlp`) and the encoder block
(`rocket_encoder.c`: one pre-norm Whisper layer over the two norms, `mha_self`, the
residual adds and the fused MLP), the fp16 convolution family (`rocket_conv.c`: the
native CNA conv — direct, depthwise, 1D and the conv→activation fusion, with the OC /
spatial / channel tiling and the resident-BO context — and `rocket_conv_transpose.c`:
the hardware deconvolution route and the dilated-input lowering), the native int8
convolution (`rocket_conv.c`'s other half: the direct int8→int32 accumulator with its
multicore pool and the int8-out depthwise requant), the PPU pooling family
(`rocket_pool.c`: MaxPool/AveragePool plus the int8/uint8 entries) and the
integer-factor upsample (`rocket_resize.c`: nearest and bilinear as a depthwise
transposed conv) are ported and validated on the board. Planner and regcmd output are byte-equal to the C library for
every matmul dtype
(fp16/int8/int8-group-wise/int4/tf32/bf16), for the activation program (1070 words,
word-for-word) and for the PPU pool program (30 words, over an 18-case
shape/method/pad/stride matrix), the pool decomposition (20 `C H W` shapes: the plan
reason code and both factor lists), the fp16 average-reciprocal convention and the
convolution family (`conv`: 23 plan cases over the direct/depthwise/dilation/stride/
explicit-extent matrix, `emit-conv`: 18 programs including the CNA deconvolution
strides and the refusals, `emit-conv-act`: 6 fused-activation programs) and the int8
convolution (`emit-conv-int8`: 11 direct programs incl. the +1 CBUF bank slack and the
int32-output stride quirk, `emit-conv-dw-int8`: 9 depthwise programs over a
zero-point/scale matrix, including the int32-raw diagnostic form and both refusals);
the LUT tables and epilogue constants are byte-identical to the C builder. fp16/bf16 cosine
>= 0.99999, int8 matches its dequantized reference, int4 is int32-exact against
the int64 CPU reference and tf32 matches a tf32-rounded f64 reference.

The activation/elementwise device output is **bit-identical to the C library on the
board for all 19 activation kinds and the six EW ops** (same table, same program, same
BO layout); the EW add/mul/sub/max/min are exact against the host arithmetic, and a
`lut_apply` host model of the DPU LUT chain (BN mul/bias, LE/LO interpolation with the
unsigned table clamp, OUT_CVT decode) tracks the device to within a fraction of an
fp16 ULP. Qwen3 3,633-token NPU bench 59.4 s vs the C library's 61.5 s; int8 and bf16
microbenchmarks are at or above the C library's speed.

The reduce/softmax device output is **bit-identical to the C library on the board for
every family** (`tools/c-oracle/ab_compare.py`: average/max/min pools including the
host-fallback shape, the feature reduce in sum and mean modes, all four cumsum
variants, softmax/log-softmax/cross-entropy up to 257x1000 and the per-row scale) —
both sides run the same program and the same host glue, so even the fp16 reciprocal
quantization cancels. Against the device-free fp64 references: max/min pools are
exact, the feature reduce is within 2.3e-7 relative (fp32 accumulator), cumsum is
bit-exact after its fp16 read-back, softmax sits at cosine 0.99999992, log-softmax's
absolute error is <= 7.8e-3 and cross-entropy's <= 4.2e-5. Board wall for the families
at 257 rows: a 4096-wide softmax 97 ms, the 151936-wide LM-head softmax 3.4 s, the
same log-softmax 3.7 s and cross-entropy 1.8 s (busy board, load ~8.4).

The M7 norm / vision-norm / FFN family is also **bit-identical to the C library on the
board** (`ab_compare.py` again: RMSNorm with a real and a unit weight, LayerNorm with
and without beta, BatchNorm/GroupNorm/InstanceNorm/L2-Normalize over two `[N,C,P]`
shapes, GeGLU in both GELU and SiLU form, `ffn`, `ffn_fused` and `mlp_fused` with and
without the fc1 bias, at 4x256x1024 and 64x512x2048, plus two shapes that force the
fused entries' host-handoff fallback — `M = 1` and the `M = 260` KACC EW-stride
refusal — so both sides have to make the same fallback decision). The encoder block is
the only case that is not bit-identical (198/512 and 2040/8192 elements exact): its
attention sublayer is `attn::mha_self`, whose host fast softmax is the documented
deviation from the C's on-NPU softmax (set `ROCKET_ATTN_HOST_SOFTMAX=0` for the on-NPU
arm, which is bit-identical to the C — see the placement section below). Against the device-free fp64 references:
RMSNorm/LayerNorm cosine
0.99999992-0.99999994 with <= 2.9e-3 absolute and <= 2.1e-3 relative error on the
significant elements, the vision norms 0.9999999 with <= 2.1e-3 relative, GeGLU
0.99999993 (1.5e-3), the FFN/MLP 0.9999998-0.9999999 (<= 5e-3 relative at a peak of
1.2e3) and the encoder block 0.9999985-0.9999992 (1.9e-2 relative on the significant
elements, 4.7e-2 absolute at a peak of 20.6) — the fp16 pipeline's own K-accumulation
noise. Its two MLP routes are bit-identical to each other on the measured shapes
(`ROCKET_ENCODER_NOFUSE` vs the fused default, 512/512 and 8192/8192), because the
pinned `Kt = 256` equals the planner's own choice below `d_ff = 512`. Board wall (cores
0-3, busy board): a 257x1024 LayerNorm 69 ms, RMSNorm 44 ms,
a 64x512x2048 FFN 36 ms and the encoder blocks 32 ms / 48 ms.

The M8a convolution family is **bit-identical to the C library on the board across the
whole measured matrix** (`ab_compare.py` again): direct convs over the kernel/stride/
pad/dilation sweep, the RGB stem (`IC = 3` zero-padded to 32), an `OC = 24` head (padded
to 32), the tiling shapes that force OC-only and OC+spatial bands (64x56x56 k3,
128x28x28 k3 s2, 32x112x112 k3), four depthwise shapes, conv1d at both strides,
the fused SiLU/tanh/GELU epilogues, and seven transposed convs covering both routes
(hardware deconvolution at strides 2/4, the lowering at `IC = 24`, stride 3 and
stride 1) plus depthwise. Against the device-free fp32-accumulate references: the
direct/depthwise/1D convs hold cosine 1.0 with <= 1e-3 relative on the significant
elements (>= 99.7 % of elements bit-identical per shape, the rest one fp16 ULP), the
transposed convs the same against the route-independent scatter-add oracle, and the
fused activation sits at <= 2e-3 relative on the significant elements with the C's own
single-pass table artifacts reported separately (the `x≈0` mux spike and gelu's
saturated tail). Board wall (cores 0-3, busy board at load ~7, 600 MHz module): 32x16x16
k3 1.3 ms through the alloc-per-call entry and 0.5 ms on a resident context, a
64x56x56 k3 5.3 ms, a 128x28x28 k3 s2 5.8 ms, a 32x112x112 k3 7.4 ms, a 32x56x56
depthwise 4.7 ms, the Whisper front-end conv1d (80x3000 k3 s1, 384 outputs) 80 ms and
its stride-2 form 31 ms, and the transposed convs 1.6-3.9 ms.

The M8b families are **bit-identical to the C library on the board across the whole
measured matrix** as well (`ab_compare.py`): nine int8 direct convs (3x3/1x1/k7,
stride 1/2, the RGB stem at `IC = 3`, an `OC = 16` head padded to 32, and the tiling
shapes 64x56x56 k3 / 32x112x112 k3 / 128x28x28 k3 s2), six int8 depthwise convs
(`C = 64`/`128`, strides 1/2, asymmetric strides, 5x5, zero points at both ends of the
int8 range, with and without the bias cube), nine pools (fp16 max/avg with and without
padding plus int8 and uint8 max/avg) and eight upsamples (nearest and bilinear at
scales 2/3/4, non-square factors). Against the device-free references the int8 direct
conv is **int32-exact** on every shape (int8 × int8 → int32 is exact integer
arithmetic; a float gate would hide an accumulator bug), the int8 depthwise output is
**bit-exact against the DPU requant model** (nearest, ties to even — the rule measured
on the part, in `conv2d_dw_int8_model`), max pooling is exact, an average sits within
the `fp16(65536/k)` reciprocal's tolerance (<= 1e-3 absolute at the array peak), the
int8/uint8 averages are within one rounding step, nearest upsampling is bit-exact and
bilinear is within the fp16 triangle kernel's rounding. Board wall (cores 0-3, busy
board, 600 MHz module): an int8 3x3 32x16x16 0.4-1.0 ms (alloc-per-call entry 2.1 ms for
the 3x32x32 RGB tile), 64x56x56 k3 5.6 ms, 32x112x112 k3 6.6 ms, 128x28x28 k3 s2
4.9-23 ms depending on load; the int8 depthwise jobs 1.0-7.9 ms; fp16 pools 0.5-0.7 ms;
the upsamples 1.4-5.8 ms.

Two *encoder-level* envelopes the C library does not check showed up while measuring,
and this port mirrors them because it emits the same programs: the PPU's pooling
window is only reliable up to a kernel of 8 when more than one output window is
needed (a 9x9 or larger kernel, or a stride larger than the kernel, writes zeros or
garbage — a 14x14 kernel is fine only when the plane is exactly one window, which is
what the global-pool decomposition uses), and the int8 depthwise writer walks channels
in `G = 64` groups, so a `C` that is a multiple of 16 but not of 64 (32, 96) produces
an identically-wrong surface on both sides. The self-test matrix stays inside those
envelopes; the A/B keeps one case of each to pin the bit-identity.

Attention also has its long-context form, the opt-in tiled path
(`ROCKET_FA_TILE_KV`, off by default, `ROCKET_FA_TILE_MIN_KV` sets the crossover):
one key tile at a time with the online softmax, instead of the `[Tp, Kn]` score
matrix. It is bit-for-bit independent of `ROCKET_FA_CHAIN`, matches the f64 reference
at cosine 0.9999999 (30,016 keys, causal, GQA 8/2) and cuts the attention scratch
14x at that length (37 vs 529 MiB) and 6.4x at 16k (248 vs 1597 MiB), which is what
takes a 30k-key request from OOM to a 1 GiB working set. Wall is parity to ±20 %:
it wins once the key axis dominates the query block (0.88-0.95x measured) and pays
more, smaller submits at a square 8k with a 2k tile.

It also covers an MLA head geometry, where the value dim differs from the head dim
(`dv != dk`): the tiles carry `dk` and `dv` separately (`kh [wp,dk]`, `vh [dv,wp]`,
the AV matmul `Tp x wp x dv` and the fp32 `acc [Tp,dv]`), so a 576/512 geometry tiles
like any other. That is a Rust extension — the C `fa_heads_range_tiled` takes a single
`d` and has no MLA variant, so there is no C byte parity there; the gate is the device-free
`flash_attn_ref` plus the materialized arm. Board (T=64 kv=2048 GQA 8/2 dk 576 dv 512,
tile 512, causal): tiled cosine 0.99999994 vs the f64 reference (materialized
0.99999996), 0.99999992 between the arms, and 23.6+20.1 → 8.6+6.3 MiB of scratch; at
512x8,192 170.8+140.4 → 58.8+45.4 MiB (2.9x) at wall 1.09x, i.e. the same trade as the
`dv == dk` path (memory for submits, never faster below the crossover). A `dv == dk`
geometry is bit-for-bit unaffected by the change.

### Softmax placement: host vs on-NPU (`ROCKET_ATTN_HOST_SOFTMAX`)

The C library's attention entries can run their row softmax either host-side or through
`rocket_softmax_fp16`, the composition of host row-max plus the DPU `EXP` LUT, the
feature-axis row sum and the per-row scale. Both arms are ported: `attn::RowSoftmax`
owns the host pass (`attn::Softmax`, the default) and the on-NPU arm, which writes the
same f16 masked scores the C does (`fa_score` in f32, one f16 narrowing, pad columns at
the `-30000` clamp) and hands them to `ops::softmax::softmax` — the same composition the
M6b gate already found bit-identical to C. `ROCKET_ATTN_HOST_SOFTMAX=0` selects the
on-NPU arm, `1`/unset the host one (the port's default everywhere; the C's `mha_self`
defaults to on-NPU, so `=0` is what reproduces it). The **materialized** paths honor
the knob — `Attention::flash`'s per-head and chained group paths, and `mha_self` — while
the tiled `ROCKET_FA_TILE_KV` path stays host-only by construction (its online
running max/rescale is sequential across key tiles, and the C tiled range takes no
`host_sm` either). `ROCKET_FA_CHAIN=0/1` and the `mha_self` thread fan-out stay
bit-identical under either placement.

The knob is global — `burn_rocket`'s attention ops ride the same `Attention::flash`, so
setting it also switches the app's NPU attention — and the on-NPU arm is a
parity/comparison knob, not a default; the measurements say why
(board, cores 0-3, busy board at load ~10; `attn_selftest --sm-ab`): at T=257 kv=257 with
8 heads / 2 kv heads and dk 64 it costs **1.8-2.2x the host wall** (128.1 vs 59.7 ms in
one run, 100.5 vs 56.3 in another — the board scatters ±20 %) with a single fd and
**1.2-1.3x** through the 4-worker fan-out, and at kv=1024 **2.55x / 1.76x**: the gap grows
with the key count, as the C's own comment predicts, but it is far wider here because the
C compares against a scalar `expf` pass and this port has the S1 fast host softmax.
`mha_self` reads 1.1-1.25x (T=257) and 1.75x at the kv=1024 run's T=256. Numerics against
the device-free f64 references: cosine 0.99999990 (flash) and 0.99999887-0.99999989
(`mha_self`) — the `EXP` LUT's approximation, as documented for `rocket_softmax_fp16`.

The C-vs-Rust A/B closes the surface (`ab_board` grew `flash`/`mha` families with the
`attn_selftest --io` layout; `ab_compare.py --attn-only`): the **on-NPU arms are
bit-identical on every case** — flash causal, unmasked, soft-cap 8, MLA (`dv != dk`) and
a one-hot `n_kv > T` prefix, plus `mha_self` at T=8/5/33 with the `Tp`/`Tn` padding —
while the host arms differ by at most ~2.4 f16 ULP at the output peak (cosine
0.9999992-1.0), which is exactly the documented f32-vs-f16 score write. The C's
single-fd entry chains its heads by default (`ROCKET_FA_CHAIN`), and so does the Rust
one, so those cases pin the chained path's placement too.

### The model encoders (M9): SAM ViT-Det and SigLIP

`rocket-npu` also carries the two **model-level** encoders of the C library — complete
graphs assembled from the primitives above, not new kernels. Each has a flat fp16 weight
blob (the `tools/sam_extract.py` / `tools/siglip_extract.py` format), a **device-free
host reference** (the C `fd < 0` datapath), a one-shot device entry and a **resident**
context that packs the static GEMMs once and reuses them across images.

| | graph | blob |
|---|---|---|
| **SAM ViT-Det** (`facebook/sam-vit-base`) | patch conv (k = stride = 16, with bias) → learned 2D position table → 12 pre-norm blocks (fused qkv → **windowed** attention on all but the periodic global layers → the decomposed **relative-position bias** added to the *unscaled* `QK^T` before the softmax → o-projection, then an erf-GELU MLP) → neck (1×1 conv → channels-first LayerNorm → 3×3 conv → LayerNorm) → `[1,256,64,64]` | 183.8 MB |
| **SigLIP-B/16** (the SmolVLM-256M front-end) | patch conv (16, 16, with bias) → position table → 12 pre-norm blocks (q/k/v/o + GELU-tanh MLP) → post-LayerNorm → `[1024,768]` | 172.9 MB |

The host/NPU split is the C's: the big GEMMs (patch, qkv/o, fc1/fc2, the neck convs and
the per-head `QK^T` / `P·V`) run on the NPU, and the `O(tokens·d)` glue (im2col, the
position add, window partition/unpartition, the LayerNorms, the erf-GELU, the softmax and
the decomposed rel-pos matmul+add) on the host. The rel-pos add is faithful in fp16 (the
C measures cos ≥ 0.9999991/layer), so there is no host-f32 fallback for it. SAM's window
grid is zero-padded up to a multiple of the window size, and — as in the C — the pad
tokens *do* participate as keys/queries in the boundary windows; the real 64×64 grid with
14×14 windows (70×70 padded, 25 windows) is exactly that case.

`SiglipModel` also carries the C's plain-ViT variant knobs, honoured by the resident
path only (as in C): a CLIP `cls` prefix row, a pre-encoder LayerNorm, quick-GELU instead
of GELU-tanh, and DPT-style multi-tap exits. A v1 blob leaves them at the SigLIP
defaults.

**Validation.** The primary gate is the two host references against the C's: both are
**bit-identical** to `rocket_sam_encode(-1, …)` / `rocket_siglip_encode(-1, …)` — the
output *and* every intermediate token block — on the synthetic fixture (x86 and the
board) and on the real checkpoints (x86). That pins the whole datapath (im2col ordering,
SAM's f32-accumulate vs SigLIP's f64-accumulate patch matmul, the LayerNorm/softmax/GELU
rounding, the windowed rel-pos bias, the neck) without needing a device. On the board the
device arms are then scored against that reference: the synthetic fixture
(d=64, 2 layers, grid 8/window 4) gives SAM one-shot **0.99999951** / resident
**0.99999951** worst-layer, and SigLIP one-shot **0.99999724** / resident **0.99999724**;
SAM's one-shot and resident outputs are **bit-identical to each other**, and repeats are
bit-identical. On the real checkpoints — the board, cores 0-3, load average ~9-11
(the live embedding service keeps 4-7) — the same comparison gives SAM one-shot worst
layer **0.99999848** / neck **0.99999938** (150 s) and resident worst **0.99999842** /
neck 0.99999936 (113 s after a 1.8 s pack), SigLIP one-shot output **0.99998189**
(18.8 s, worst layer 0.99999144) and resident output **0.99993420**
(4.4 s after a 0.8 s pack, worst layer 0.99998237; the
per-head fallback `ROCKET_SIGLIP_FA=0` gives 0.99992939 in 13.3 s, so the flash-attention
fan-out is 3× the per-head path on this shape). For
scale on the same artifacts, the C library's own `sam_rocket` gate reports mean-layer
0.999999 / neck 0.999999 with a 153.9 s one-shot, so the Rust one-shot is at parity
with the C's and the SigLIP resident arm is 4.3× faster than its one-shot (it skips
the per-call weight re-pack and the per-layer `mha_self` fd churn).

`siglip_rocket`, the C library's own gate, passes on the board against the HF f32
reference with mean-layer cosine 0.999985 and post-LN 0.999884, which is what makes the
port's target meaningful. The C's erf-GELU (device one-shot) and GELU-tanh (resident)
differ by ~1e-3 as its header says, so the resident arm carries that delta on top.

### Missing surface

`rocket-npu` covers the RK3588 functional surface of `librocketnpu` (the 219
RK3588-relevant entries; the remainder are RK3576-only). The C ABI, a few aliases and
plane variants, performance mechanisms and every RK3576 entry are not reproduced.
Implemented: fp16 one-shot/streaming/resident/segmented/multi-fd/
fp32-out and batched matmul, dense and group-wise int8 (one-shot, resident per-channel
and resident group-wise), dense and group-wise int4 (one-shot, resident `I4Context` in
both modes, nibble-packed host pack/gather), `int16_exact`, the native bf16 stream,
the tf32 one-shot and its planner, the activation + elementwise family (all 19
`ROCKET_ACTIVATION_*` kinds, LeakyReLU/ELU/SELU/PReLU, EW mul/add/sub/max/min, clip,
divide, the LUT builder and the `*_ref` host references), the reduce + softmax family
(the PPU global average/max/min pools with `global_avgpool_plan`/`reduce_factor_axis`
and the exact host fallback, the feature-axis ones-matmul reduce, cumsum, row
softmax/log-softmax/cross-entropy and the per-row scale, each with its `*_ref`), the
transformer norms (`rmsnorm`/`layernorm` plus the shared `square_prescale_k` fp16-square
guard), the vision norms (`batchnorm`/`groupnorm`/`instancenorm`/`l2norm`), the FFN
family (`geglu`, `ffn`, the pinned-tile `ffn_fused`/`mlp_fused`, the host-handoff
`mlp`, and `ffn_ref`/`geglu_ref`/`mlp_ref`), the encoder block
(`encoder_block`/`encoder_block_ref`), the fp16 convolution family
(`conv2d`/`conv2d_act`/`conv1d`/`conv2d_ref` with `ConvContext`, `conv2d_plan`/
`conv2d_plan_code`, the `conv2d_deconv`/`conv2d_deconv_fits` deconvolution route and
the weight/feature cube index math; `conv_transpose2d`/`conv_transpose2d_ctx`/
`conv_transpose2d_ref` with `conv_transpose2d_plan`/`_plan_code` and the route enum),
the native int8 convolution (`conv2d_int8`/`conv2d_int8_ctx`/`conv2d_int8_mt` with
`ConvInt8Pool`, `conv2d_ref_int8`, `conv_int8_tile_plan` and the int8 cube index
helpers; `conv2d_dw_int8`/`conv2d_dw_int8_ctx` with the on-chip requant, its host
model `conv2d_dw_int8_model` and the `requant_*` arithmetic), the PPU pooling family
(`pool_fp16`/`pool_int8`/`pool_uint8` and their references, `pool_fp16_plan` with the
C's `-3`..`-10` classes, `PoolDesc`), the upsample family
(`upsample_nearest`/`upsample_bilinear` and the references, `build_kernel`,
`bilinear_ksize`, the plans), the two model encoders (`ops::sam` and `ops::siglip`:
the blob loaders, the device-free host references, the one-shot entries and the resident
contexts, with the CLIP/DPT variant knobs), and
masked GQA attention (the persistent
`attn::Attention` context, the stateless `rocket_flash_attn_fp16` / `_mt` entries and
the `rocket_mha_self_fp16` encoder sublayer) with chained per-head QK/AV batching, the
selectable host / on-NPU softmax placement (`ROCKET_ATTN_HOST_SOFTMAX`) and
the tiled long-context path (`ROCKET_FA_TILE_KV`: online softmax over key tiles, no
`[Tp, Kn]` score matrix, at any `dk`/`dv` — MLA included), the two f64 host oracles
(`flash_attn_ref`,
`mha_self_ref`), the device/BO/submit primitives (including multi-job submit and the
opt-in chained layout), and the dtype parity harness (`tools/c-oracle` + `parity.py`).

The activation family's own self-test (`crates/rocket-npu/src/bin/activation_selftest.rs`)
runs every kind and EW op on the board against that model, and its `--dump-lut` /
`--dump-gen` modes reproduce the C `rocket_lut_epilogue_build` dump and the generated
program byte for byte. `crates/rocket-npu/src/bin/reduce_selftest.rs` does the same for
the reduce/softmax families (`--pool`, `--feature`, `--cumsum`, `--softmax`, `--shape`,
`--reps`), `crates/rocket-npu/src/bin/ops_selftest.rs` for the norm/vision/FFN/encoder
families (`--norm`, `--vision`, `--ffn`, `--encoder`, the four per-family shape
overrides, `--reps`), `crates/rocket-npu/src/bin/conv_selftest.rs` for the convolution
family (`--conv`, `--dw`, `--act`, `--conv1d`, `--transpose`, `--int8`, `--int8-dw`,
`--pool`, `--resize`, the per-family shape overrides, `--reps`), and
`crates/rocket-npu/src/bin/encoders_selftest.rs` for the two model encoders (`--write-blob`,
`sam`/`siglip` with `--ref`/`--npu`/`--ctx N`, `--ref-in`/`--ref-out` to move a host
reference between machines, and a `--io` mode); their `--io` raw-file modes are the
exchange format of `tools/c-oracle/ab_compare.py` for the C-vs-Rust A/B on the board
(`--enc`, or `--enc-only` for just the M9 section). `attn_selftest` covers the attention
side: the chained A/B by default, `--oneshot` for the stateless entries against the f64
oracle, `--mha` for the encoder sublayer, `--host-sm 0|1` to pick the softmax placement,
`--sm-ab` to run both placements and report the wall/CPU and the deviation, `--tile KBP`
(`--tile-min`, `--tile-only`) for the tiled long-context A/B at any `dk`/`dv` — MLA
included — and an `--io flash|mha` raw-file mode (`ab_compare.py --attn-only`).

`ROCKET_FA_TILE_KV` is opt-in (default 0 = off): it takes the materialized path below
`ROCKET_FA_TILE_MIN_KV` padded keys (default 8192, where the small-GEMM dispatch floor
rather than the score matrix dominates), tiles at any `dk`/`dv` (MLA included), and
never chains. Set it to the tile width you want (`2048` is the tuned value) and the
attention scratch stops growing with the key count; it trades more, smaller NPU
submits for the flat footprint and the cache-resident score tile.

Not exposed yet: resident bf16 (the C library has no variant either, and it has none
for tf32 either — that surface is complete), the `*_ref_fp16` host references of the
remaining non-matmul families, the
`_planes`/transposed variants, RK3576 entries (including the RK3576 int8 LUT
activation), the public `rocket_stream_*`
aliases, the non-fp16 batch wrappers, the cube-resident *mechanism* of the fused
FFN/MLP (the pinned tiling, the matched-tiling gate and the KACC routing are
bit-identical to the C; only the host round-trip is kept, so the C's resident-cube
optimization is the remaining work), the RK3576-specific convolution and pooling
entries (the part's own conv encoder, the packed-image datapath, the int8 pooling
entry with its zero point and the prepacked/cube handles), the int8 convolution's
opt-in `ROCKET_CONV_BATCH` batched submit (bit-identical to the per-tile path this
port runs), and graph execution (the RK3576-only graph runner).
The activation + elementwise, the reduce + softmax, the transformer and vision norms,
the FFN/MLP, the encoder block, the fp16 and int8 convolution families (direct/
depthwise/1D, the conv→activation fusion, the transposed conv, the int8 requant), the
PPU pooling family, the upsample and the two model encoders (SAM ViT-Det and
SigLIP-B/16) are implemented. See the `rocket-npu` crate docs for the full list.

On the board the NPU boots at 200 MHz; the patched 600 MHz module
(`/root/npu-poc/rocket-patched-600/rocket-npu600.ko`) is ~3× faster under load and
reverts on reboot. Everything works at the stock clock.

## Example

[`examples/rocket-inference`](examples/rocket-inference) is a complete inference app
built on this library: one binary, four model families, selected by the first argument
(`serve`, `qwen3`, `intent`, `gemma`). One `serve` command loads any of the models and
exposes its compatible endpoints; the Qwen3-Embedding-0.6B embedding model (CLI
`qwen3 bench|embed|gemm|tokenize`), the Qwen3.5-0.8B intent/query-planner model
(`intent gen`), the multimodal EmbeddingGemma 2 (`gemma embed|bench|tokenize`) and
Gemma 4 E2B-it generation (`gemma gen`) are one-shot CLI paths, with measurements from the Rock 5B+ in
`docs/experiment-log.md` and `docs/experiment-log-gemma.md` there.

On the board the Qwen3 embedding NPU path runs the 3,633-token input at 45 tok/s while
using ~51% fewer CPU-seconds than the CPU-only flex path (the CPU path remains ~2.3×
slower than the production `ik_llama.cpp` Q8_0 server); the intent model is
token-identical to the HF reference and runs a 166-token v7 planner prompt in 11.7 s
with `--npu` (89.9 s CPU-only, 7.7× wall). EmbeddingGemma 2 matches the HF f32 reference
on every modality (cosine 1.0; `--quant q8` gives 0.9996-0.9999 at 1216 MiB resident,
605 MiB anon idle in the deployed text-only `--defer-towers` configuration)
and its text backbone runs on the NPU: 218 projections packed into 0.25 GiB of resident
fp16 weights plus chunked sliding **and full** attention, for 1.61x wall and -33 %
user CPU vs the CPU baseline (2587-token text). Its vision tower packs 65 resident
projections and runs its attention on the NPU too (`--vision-attn npu`, the
`--npu` default): a 2394-patch image's tower is 10.6 s (15.2 s with the CPU
attention kernel) at cosine 0.99999 vs the HF f32 reference in the f32 arm and
0.99988 with the deployed q8 text backbone. Gemma 4 E2B-it greedily reproduces
the HF reference token-for-token, and its QAT mobile checkpoints load bit-exactly.
Long causal prompts run in query chunks on the NPU (`ROCKET_CAUSAL_CHUNK`, default
1024): at 8k tokens the Qwen3 path is 246 → 181 s of compute (32.5 → 44.3 tok/s) and
bit-identical to the dense causal op, and one 1024-row chunk against a 30k-key prefix
keeps ~0.9 GiB of chunked attention scratch instead of the dense 1.8 GiB mask plus its
`[n, n]` score matrices.

## License

GPL-3.0-or-later, matching the upstream `librocketnpu` design this port follows.
