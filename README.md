# burn-rocket

RK3588 NPU offload for [Burn](https://burn.dev) models, through the mainline `rocket`
driver: a safe Rust wrapper around [`librocketnpu`](https://github.com/gregordinary/rocket-userspace)
and a set of Burn **backend-extension** ops that run a model's projection matmuls and
masked grouped-query attention on the RK3588 NPU while everything else stays on the CPU
backend.

The library is `aarch64`-only for anything that touches the NPU: it links the static
`librocketnpu.a`, which talks to `/dev/accel/accel0` through the in-kernel `rocket`
driver.

## What it provides

- **Burn backend extension** (`#[backend_extension(Flex)]`, feature `npu`): ordinary
  `Tensor`s in and out, no FFI in model code.

  ```rust,ignore
  burn_rocket::init(5)?;                       // once, before any op

  let qkv = burn_rocket::pack3(wq, wk, wv);    // [N, K] f32 weights, pack-and-drop
  let qkv_out = burn_rocket::matmul(x, &qkv);  // [.., M, K] @ [N, K]^T -> [.., M, N]

  let out = burn_rocket::attention(
      q, k, v, 16, 8, 128, scale, None, true,  // n_head, n_kv_heads, head_dim, scale, softcap, causal
  );
  ```

  `pack`/`pack2`/`pack3` consume f32 CPU weights and keep fp16 resident copies in NPU
  memory for the process lifetime (`WeightId` handles). Fused packs concatenate several
  weights that share one input along N, so one matmul produces all their outputs (the
  example packs q|k|v and gate|up that way). Matmuls are chunked above
  `ROCKET_MATMUL_CHUNK_M` rows (env, default 8192, `0` disables) to bound the per-call
  input buffer; rows are independent, so chunking is bit-identical.

- **Low-level FFI API** (`RocketCtx`, `RocketWeight`, `RocketStream`, `RocketFaCtx`,
  `f32_to_f16`, `pad_rows`, driver/counter helpers) for callers that do not want Burn
  tensor routing. See `examples/probe.rs` for a runnable smoke test.

The ops are **inference-only**: the extension macro generates no autodiff and calling
them with an autodiff context panics inside Burn. `librocketnpu` contexts are not
thread-safe; the extension routes every call through one global engine behind a mutex
(`burn_rocket::stats()` reports accumulated convert/NPU time). Resident weights are
packed for the `M >= 256` tiling, smaller requests are padded.

## Features

| feature | effect |
|---|---|
| `default` | none |
| `flex` | compiles against Burn's `Flex` backend (required by the extension catalog; the feature name must stay exactly `flex`) |
| `npu` | the `RocketOps` extension ops; links `librocketnpu.a` on aarch64 (implies `flex`) |

## Building

`librocketnpu.a` is expected at `vendor/rocketnpu/` (gitignored; copy it from the board's
`/root/npu-poc/rocket-userspace/build` or build
[`gregordinary/rocket-userspace`](https://github.com/gregordinary/rocket-userspace)), or
point `ROCKETNPU_DIR` at the directory holding it.

```sh
# compile check, works on any target that has no NPU (no linking)
cargo check -p burn-rocket --features npu

# aarch64 build of the probe example (links librocketnpu.a)
ROCKETNPU_DIR=<dir with librocketnpu.a> \
  cargo build --release -p burn-rocket --example probe --features npu \
    --target aarch64-unknown-linux-gnu
```

On the board the NPU boots at 200 MHz; the patched 600 MHz module
(`/root/npu-poc/rocket-patched-600/rocket-npu600.ko`) is ~3× faster under load and
reverts on reboot. Everything works at the stock clock.

## Example

[`examples/qwen3-embeddings`](examples/qwen3-embeddings) is a complete inference app
built on this library. It serves the Qwen3-Embedding-0.6B embedding model (CLI `bench`,
`embed`, `gemm`, `serve`, `tokenize`; OpenAI-compatible `/v1/embeddings`) and the
Qwen3.5-0.8B intent/query-planner model (`gen`, `serve-ollama`; Ollama-compatible
`/api/chat`, `/api/generate`), with measurements from the Rock 5B+
(`docs/experiment-log.md` there). On the board the embedding NPU path runs the
3,633-token input at 45 tok/s while using ~51% fewer CPU-seconds than the CPU-only flex
path (the CPU path remains ~2.3× slower than the production `ik_llama.cpp` Q8_0 server);
the intent model is token-identical to the HF reference and runs a 166-token v7 planner
prompt in 11.7 s with `--npu` (89.9 s CPU-only, 7.7× wall).

## License

GPL-3.0-or-later, matching `librocketnpu`.
