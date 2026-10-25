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

- **Low-level driver API** (`rocket_npu::Device`, `rocket_npu::matmul::Context`,
  `rocket_npu::matmul::Weight`, `rocket_npu::attn::Attention` and its stateless
  `flash_attn`/`flash_attn_mt`/`flash_attn_ref`/`mha_self`/`mha_self_ref` siblings,
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
divide) and the reduce + softmax family (`rocket_reduce.c`, `rocket_softmax.c` and
`rocket_scale_rows_fp16`: the PPU global average/max/min pools with the telescoping
decomposition, the feature-axis ones-matmul reduce, the triangular-ones cumsum, row
softmax/log-softmax/cross-entropy and the per-row broadcast scale) are ported and
validated on the board. Planner and regcmd output are byte-equal to the C library for
every matmul dtype (fp16/int8/int8-group-wise/int4/tf32/bf16), for the activation
program (1070 words, word-for-word) and for the PPU pool program (30 words, over an
18-case shape/method/pad/stride matrix), the pool decomposition (20 `C H W` shapes: the
plan reason code and both factor lists) and the fp16 average-reciprocal convention;
the LUT tables and epilogue constants are byte-identical to the C builder. fp16/bf16
cosine >= 0.99999, int8 matches its dequantized reference, int4 is int32-exact against
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

Attention also has its long-context form, the opt-in tiled path
(`ROCKET_FA_TILE_KV`, off by default, `ROCKET_FA_TILE_MIN_KV` sets the crossover):
one key tile at a time with the online softmax, instead of the `[Tp, Kn]` score
matrix. It is bit-for-bit independent of `ROCKET_FA_CHAIN`, matches the f64 reference
at cosine 0.9999999 (30,016 keys, causal, GQA 8/2) and cuts the attention scratch
14x at that length (37 vs 529 MiB) and 6.4x at 16k (248 vs 1597 MiB), which is what
takes a 30k-key request from OOM to a 1 GiB working set. Wall is parity to ±20 %:
it wins once the key axis dominates the query block (0.88-0.95x measured) and pays
more, smaller submits at a square 8k with a 2k tile.

### Missing surface

`rocket-npu` is a focused matmul/attention driver, not a full replacement of every
`librocketnpu` entry. Implemented: fp16 one-shot/streaming/resident/segmented/multi-fd/
fp32-out and batched matmul, dense and group-wise int8 (one-shot, resident per-channel
and resident group-wise), dense and group-wise int4 (one-shot, resident `I4Context` in
both modes, nibble-packed host pack/gather), `int16_exact`, the native bf16 stream,
the tf32 one-shot and its planner, the activation + elementwise family (all 19
`ROCKET_ACTIVATION_*` kinds, LeakyReLU/ELU/SELU/PReLU, EW mul/add/sub/max/min, clip,
divide, the LUT builder and the `*_ref` host references), the reduce + softmax family
(the PPU global average/max/min pools with `global_avgpool_plan`/`reduce_factor_axis`
and the exact host fallback, the feature-axis ones-matmul reduce, cumsum, row
softmax/log-softmax/cross-entropy and the per-row scale, each with its `*_ref`), and
masked GQA attention (the persistent
`attn::Attention` context, the stateless `rocket_flash_attn_fp16` / `_mt` entries and
the `rocket_mha_self_fp16` encoder sublayer) with chained per-head QK/AV batching and
the tiled long-context path (`ROCKET_FA_TILE_KV`: online softmax over key tiles, no
`[Tp, Kn]` score matrix), the two f64 host oracles (`flash_attn_ref`,
`mha_self_ref`), the device/BO/submit primitives (including multi-job submit and the
opt-in chained layout), and the dtype parity harness (`tools/c-oracle` + `parity.py`).

The activation family's own self-test (`crates/rocket-npu/src/bin/activation_selftest.rs`)
runs every kind and EW op on the board against that model, and its `--dump-lut` /
`--dump-gen` modes reproduce the C `rocket_lut_epilogue_build` dump and the generated
program byte for byte. `crates/rocket-npu/src/bin/reduce_selftest.rs` does the same for
this milestone's families (`--pool`, `--feature`, `--cumsum`, `--softmax`, `--shape`,
`--reps`), and its `--io` raw-file mode is the exchange format of
`tools/c-oracle/ab_compare.py` for the C-vs-Rust A/B on the board.

`ROCKET_FA_TILE_KV` is opt-in (default 0 = off): it takes the materialized path below
`ROCKET_FA_TILE_MIN_KV` padded keys (default 8192, where the small-GEMM dispatch floor
rather than the score matrix dominates), needs `dv == dk`, and never chains. Set it to
the tile width you want (`2048` is the tuned value) and the attention scratch stops
growing with the key count; it trades more, smaller NPU submits for the flat footprint
and the cache-resident score tile.

Not exposed yet: resident bf16 (the C library has no variant either, and it has none
for tf32 either — that surface is complete), the `*_ref_fp16` host references of the
remaining non-matmul families, the
`_planes`/transposed variants, RK3576 entries (including the RK3576 int8 LUT
activation), the public `rocket_stream_*`
aliases, the non-fp16 batch wrappers, tiled MLA (a `dv != dk` head geometry stays
materialized), on-NPU **attention** softmax, and the remaining op families: the
`rocket_pool.c` MaxPool/AveragePool API (only the PPU global-pool decomposition is
ported), conv (2d/1d/transpose/depthwise), the rest of `rocket_norm.h` (RMSNorm,
LayerNorm — M7), FFN/encoder, resize/upsample, SAM/SigLIP and graph execution.
The activation + elementwise and the reduce + softmax families are implemented. See
the `rocket-npu` crate docs for the full list.

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

## License

GPL-3.0-or-later, matching the upstream `librocketnpu` design this port follows.
