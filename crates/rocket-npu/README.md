# rocket-npu

A pure-Rust userspace driver for the **RK3588 NPU**, talking to the mainline Linux
`rocket` accelerator driver (`/dev/accel/accel0`). No C toolchain, vendor SDK or
kernel headers are needed; the register programs, tiling planner, buffer objects and
kernels are all Rust.

It is an interface-compatible port of the `librocketnpu` C library, and its register
programs are byte-identical to the C ones.

## What it can do

| Area | Operations |
|---|---|
| Matrix multiply | fp16, bf16, tf32, int8 and int4 (dense and group-wise), with weights that stay resident on the NPU |
| Attention | flash-style grouped-query attention with causal, windowed and tiled long-context modes |
| Transformer pieces | softmax, RMSNorm, LayerNorm, gated FFN, a full encoder block |
| Convolution | fp16 and int8 conv (direct, depthwise, 1D, transposed), pooling, upsampling |
| Models | SAM and SigLIP image encoders |

Every operation has a device-free `*_ref` reference implementation that the tests and
board self-tests compare against.

## Use it

Most users reach this crate through [`burn-rocket`](../burn-rocket/README.md), which
exposes it as Burn tensor ops. To use the driver directly:

```rust,ignore
let dev = rocket_npu::Device::open()?;
let ctx = rocket_npu::matmul::Context::new(/* worker threads */ 3)?;
```

The board self-tests in `src/bin/` (`selftest`, `attn_selftest`, `conv_selftest`, …)
double as usage examples.

## Requirements

- Runs on `aarch64` with the in-kernel `rocket` driver. It compiles on any host.
- A kernel module patched to a 600 MHz NPU clock is about 3× faster than stock.

Developer notes (parity tests against the C library, board protocol, knobs) are in
[`AGENTS.md`](AGENTS.md).

## License

GPL-3.0-or-later.
