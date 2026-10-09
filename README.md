# burn-rocket

RK3588 NPU offload for [Burn](https://burn.dev) models, through the mainline `rocket`
driver: a safe Rust wrapper around [`librocketnpu`](https://github.com/gregordinary/rocket-userspace)
and a set of Burn **backend-extension** ops that run a model's projection matmuls and
masked grouped-query attention on the RK3588 NPU while everything else stays on the CPU
backend.

The library is `aarch64`-only for anything that touches the NPU: it links the static
`librocketnpu.a`, which talks to `/dev/accel/accel0` through the in-kernel `rocket`
driver.

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

The Qwen3 embedding server is the configuration running in production on a Rock 5B+
(OpenViking's embedding backend, `--npu --npu-attn cpu --max-tokens 8192`): a
3,633-token input runs in ~87 s (~42 tok/s on 4 A76 cores) at ~34% fewer CPU-seconds
than the CPU-only flex path (the default NPU attention is slightly faster: ~45 tok/s,
-51% CPU). The intent model answers the 166-token v7 planner prompt in 11.7 s with
`--npu` (89.9 s CPU-only). Flags, endpoints and deployment:
[examples/rocket-inference/README.md](examples/rocket-inference/README.md).

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
  let out = burn_rocket::attention_window(
      q, k, v, 4, 2, 256, 1.0, None, 512,
  );
  ```

  `pack`/`pack2`/`pack3` consume f32/f16 weights and keep fp16 resident copies in NPU
  memory for the process lifetime (`WeightId` handles), and bf16 weights pack the same
  way via an exact bf16->f16 conversion (the library's bf16 stream — no resident
  variant, re-packs per call — is opt-in behind `ROCKET_BF16_STREAM=1`). Fused packs
  concatenate several weights that share one input along N, so one matmul produces all
  their outputs (the example packs q|k|v and gate|up that way). Matmuls are chunked
  above `ROCKET_MATMUL_CHUNK_M` rows (env, default 8192, `0` disables) to bound the
  per-call input buffer; rows are independent, so chunking is bit-identical.

- **Low-level FFI API** (`RocketCtx`, `RocketWeight`, `RocketStream`, `RocketBf16Stream`,
  `RocketFaCtx`, `f32_to_f16`, `pad_rows`, driver/counter helpers) for callers that do
  not want Burn tensor routing. See `examples/probe.rs` for a runnable smoke test.

The ops are **inference-only**: the extension macro generates no autodiff and calling
them with an autodiff context panics inside Burn. `librocketnpu` contexts are not
thread-safe; the extension routes every call through one global engine behind a mutex
(`burn_rocket::stats()` reports accumulated convert/NPU time). Resident weights use
the canonical tiling (`M >= 4`; `ROCKET_CTX_CANONICAL=0` restores the legacy
`M >= 256` padding).

## Features

| feature | effect |
|---|---|
| `default` | `npu` — NPU offload is the first-class target |
| `flex` | compiles against Burn's `Flex` backend (required by the extension catalog; the feature name must stay exactly `flex`) |
| `npu` | the `RocketOps` extension ops; links `librocketnpu.a` (implies `flex`) |

`--no-default-features` gives the FFI-only crate (no Burn, no archive);
`--no-default-features --features flex` the Burn dependency without the extension or
archive (feature-graph check).

## Building

`librocketnpu.a` is supplied automatically: `build.rs` uses `ROCKETNPU_DIR` when set,
else a matching `vendor/rocketnpu/` archive, else builds one into `$OUT_DIR/rocketnpu`
with `scripts/build-rocketnpu.sh` (pinned
[`gregordinary/rocket-userspace`](https://github.com/gregordinary/rocket-userspace)
commit, aarch64 cross by default). The first build clones and compiles the C library
(clone/cmake cache under `$OUT_DIR/rocket-userspace`, removed by `cargo clean`); later
builds reuse it. `ROCKETNPU_AUTO=0` disables the auto-build (warn only) and
`ROCKETNPU_SRC=<checkout>` builds offline from an existing checkout.

Run the script directly to install the archive into `vendor/rocketnpu/` (gitignored) —
for the docker image, a fixed deployment artifact, or a different pin:

```sh
# aarch64 archive for the board (cross; the default)
scripts/build-rocketnpu.sh
# host archive, for link checks on this machine
scripts/build-rocketnpu.sh --target host
```

The script records the built commit and architecture in `vendor/rocketnpu/{COMMIT,ARCH}`
and skips the build when the directory already matches. `--commit <sha>` moves the pin;
`--cache <dir>` and `--profile <name>` (default `release`) control the cache location
(`<target-dir>/<profile>/build/burn-rocket` for manual runs).

```sh
# compile check (npu is the default; works on any target, no linking)
cargo check -p burn-rocket

# aarch64 build of the probe example (links vendor/rocketnpu/librocketnpu.a)
cargo build --release -p burn-rocket --example probe \
  --target aarch64-unknown-linux-gnu
```

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
on every modality (cosine 1.0; `--quant q8` gives 0.9996-0.9999 at 1216 MiB resident)
and its text backbone runs on the NPU: 218 projections packed into 0.25 GiB of resident
fp16 weights plus windowed attention, for 1.61x wall and -33 % user CPU vs the CPU
baseline (2587-token text). Gemma 4 E2B-it greedily reproduces the HF reference
token-for-token, and its QAT mobile checkpoints load bit-exactly.

## License

GPL-3.0-or-later, matching `librocketnpu`.
