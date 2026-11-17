# burn-rocket

Run the heavy parts of a [Burn](https://burn.dev) model on the **RK3588 NPU**
(Rock 5B+, Orange Pi 5, …) through the mainline `rocket` kernel driver. Burn keeps
everything else on the CPU backend, so a model needs only small changes to use the NPU.

`burn-rocket` adds Burn *backend-extension* ops for the two things that dominate a
transformer's runtime:

- **matmuls** — weights are packed once into resident fp16 buffers on the NPU, then
  reused for every request
- **attention** — grouped-query attention with causal, windowed and chunked variants

The NPU parts need `aarch64` and `/dev/accel/accel0`. On any other host the crate still
compiles, which keeps development and CI simple.

## Example

```rust,ignore
burn_rocket::init(5)?;                       // once, before any op (5 worker threads)

// pack weights once ([N, K] f32/f16/bf16), q|k|v fused into one resident weight
let qkv = burn_rocket::pack3(wq, wk, wv);

// ordinary Burn tensors in and out
let qkv_out = burn_rocket::matmul(x, &qkv);  // [.., M, K] @ [N, K]^T -> [.., M, N]

let out = burn_rocket::attention(
    q, k, v, 16, 8, 128, scale, None, true,  // n_head, n_kv_heads, head_dim, scale, softcap, causal
);
```

The same module has windowed attention for encoders (`attention_window`) and chunked
causal attention for long prompts (`attention_causal_block`), which keeps the score
matrix at `[chunk, prefix]` instead of `[n, n]`.

## Cargo features

| feature | effect |
|---|---|
| `npu` (default) | the NPU ops, on the pure-Rust `rocket-npu` driver |
| `flex` | Burn's `Flex` CPU backend (needed by the extension; implied by `npu`) |

`--no-default-features` builds only the small helper functions, without Burn.

## Good to know

- The ops are **inference-only**; autodiff is not supported.
- All calls go through one global engine, so NPU work from several threads is
  serialized.
- Device-free reference implementations exist for every op, which is how the numerics
  are tested.
- The NPU clock matters: a patched `rocket` module at 600 MHz is about 3× faster than
  the stock 200 MHz. Everything works at the stock clock.

See [`examples/probe.rs`](examples/probe.rs) for a runnable smoke test, and the
[`rocket-inference`](../../README.md) app for a complete user of this crate. The
[driver crate](../rocket-npu/README.md) holds the low-level pieces.

Developer notes, the full op list, environment knobs and measurements are in
[`AGENTS.md`](AGENTS.md).

## License

GPL-3.0-or-later.
