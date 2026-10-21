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
| EmbeddingGemma 2 (text/image/video/audio) | OpenAI `/v1/embeddings` + native multimodal `/embed` | `rocket-inference serve --model-dir /models/embeddinggemma-2` |
| Gemma 4 E2B-it chat (text/image/audio) | OpenAI `/v1/chat/completions`; Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` | `rocket-inference serve --model-dir /models/gemma-4-E2B-it` |

One `serve` command covers every model: it detects the checkpoint's family and
registers exactly the endpoints that model supports (embedding models get
`/v1/embeddings`/`/embed`, chat models get both the OpenAI chat and the Ollama
APIs).

OpenViking's embedding backend in production on a Rock 5B+ is EmbeddingGemma 2
(`rocket-inference serve --dtype f32 --quant q8 --npu --npu-attn npu
--defer-towers`): the text-only service idles at 605 MiB anon (2183 MiB before the
idle-memory round; the vision/audio towers load only if a media request ever
arrives). Its 3,633-token predecessor configuration (Qwen3-Embedding-0.6B on the
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
  `rocket_npu::matmul::Weight`, `rocket_npu::attn::Attention`, `f16` helpers,
  driver/counter helpers) for callers that do not want Burn tensor routing. The
  `RocketCtx`/`RocketFaCtx` compatibility wrappers expose the old surface. See
  `examples/probe.rs` for a runnable smoke test.

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
group-wise int8, the native bf16 stream and masked GQA attention are ported and
validated on the board. Planner and regcmd output are byte-equal to the C library;
fp16/bf16 cosine >= 0.99999, int8 matches its dequantized reference. Qwen3
3,633-token NPU bench 59.4 s vs the C library's 61.5 s; int8 and bf16 microbenchmarks
are at or above the C library's speed.

### Missing surface

`rocket-npu` is a focused matmul/attention driver, not a full replacement of every
`librocketnpu` entry. Implemented: fp16 one-shot/streaming/resident/segmented matmul,
resident group-wise int8, the native bf16 stream, masked GQA attention, and the
device/BO/submit primitives.

Not exposed yet: dense/per-channel int8, one-shot int8 group-wise, int4/int16/tf32,
one-shot bf16 and resident bf16, `mha_self`, the `*_ref` host references, the
`_planes`/transposed variants, RK3576 entries, batched/chained submit, multi-job
submit, the public `rocket_stream_*`/`rocket_mm_batch_*` aliases, the long-context
tiled attention path, on-NPU softmax, and every non-matmul op family (conv, pool,
activation, norm, FFN, softmax, reduce, resize, encoders, graph). See the
`rocket-npu` crate docs for the full list.

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
user CPU vs the CPU baseline (2587-token text). Gemma 4 E2B-it greedily reproduces
the HF reference token-for-token, and its QAT mobile checkpoints load bit-exactly.

## License

GPL-3.0-or-later, matching the upstream `librocketnpu` design this port follows.
