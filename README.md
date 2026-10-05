# embeddings-fast

Qwen3-Embedding-0.6B inference in Rust on the [Burn](https://burn.dev) framework, targeting
CPU-only embedding serving that is faster than the current `ik_llama.cpp` Q8_0 production
setup on a Rock 5B+ (RK3588, 4×Cortex-A76).

Current status: **working model, CLI, and OpenAI-compatible server**; dev-host
measurements beat the target comfortably. Deployed and benchmarked on the Rock 5B+
(2026-10-05): numerically faithful and low-RAM, but 2.3× slower than the production
int8 `ik_llama.cpp` path — see "Board results" below.

## Results so far

Dev host: AMD Ryzen 9 5950X (Zen3, AVX2+FMA, no AVX-512), Burn 0.22.0-pre.4, `flex` backend.

Two attention paths are available (`--attn fused|blocked`):

- **`fused` (default)** — the backend's fused attention kernel
  (`burn::tensor::module::attention`; flex selects a tiled flash-attention path with
  online softmax for long sequences, and parallelizes over heads).
- **`blocked`** — a portable tensor-op implementation of the same blocked online
  softmax, tuned with `--chunk`/`--key-block` (defaults 256/256). Faster
  single-threaded at short/medium lengths; slower for long inputs and with many threads.

| Attention | Tokens | Threads | Time | Speed |
|---|---|---|---|---|
| blocked | 3633 | 1 | 51.7 s | **70.2 tok/s** |
| fused | 3633 | 1 | 79.9 s | 45.5 tok/s |
| blocked | 3633 | 4 | 25.6 s | 141.8 tok/s |
| fused | 3633 | 4 | 25.6 s | 142.1 tok/s |
| blocked | 6501 | 1 | 125.5 s | 51.8 tok/s |
| fused | 6501 | 1 | 180.2 s | 36.1 tok/s |
| blocked | 6501 | 4 | 68.5 s | 94.9 tok/s |
| fused | 6501 | 4 | 61.6 s | **105.6 tok/s** |
| blocked (chunk 512/key 1024) | 30000 | 32 | 973.9 s | 30.8 tok/s |
| fused | 30000 | 32 | 572.8 s | **52.4 tok/s** |
| low-RAM `--quant q8` (blocked) | 3633 | 1 | 53.1 s | 68.5 tok/s |
| low-RAM `--quant q8` (fused) | 3633 | 4 | 22.4 s | 162.3 tok/s |

Thread scaling (3633 tokens, blocked, chunk/key 256): 1→70.2, 2→105.9, 4→141.8,
8→155.0, 16→155.4, 32→144.2 tok/s (saturates around 8 threads; ~2.2× total — the
non-GEMM share of the work is largely memory-bound).

All rows use Q8 weights (Q8_0-quantized, production-parity numerics). Rows without the
"low-RAM" tag were measured with the older f32-resident Q8 mode; the low-RAM mode costs
~2 s per forward (see Memory). f16 weights are a regression on this CPU (flex's f16 path
is ~2× slower than f32 at model level, despite near-parity in the GEMM microbenchmark),
so f32 is the only configuration used.

Target was ≥50 tok/s single-core; the board's current production path (patched
`ik_llama.cpp` Q8_0, 4 threads) does the same 3633-token input in 33.7 s.

Numerical parity vs production (`ik_llama.cpp` Q8_0, last-token pooling, unnormalized),
cosine similarity on newline-free text (llama-embedding splits prompts on `\n`):

| Tokens | f32 weights | Q8 weights |
|---|---|---|
| 82 | 0.999375 | 0.999280 |
| 955 | 0.999411 | 0.999126 |
| 6501 | 0.999485 | 0.999209 |

Both attention paths produce identical embeddings (cosine 1.000000 between them).

## Board results (rock-5b-plus, 2026-10-05)

Deployed to the board (`/root/embeddings-fast/` + `/root/models/qwen3-embedding-0.6b/`)
and measured against the production `ik_llama.cpp` Q8_0 server, both on the A76 cores 4-7
with 4 threads, identical text:

| Input | Production ik_llama.cpp | embeddings-fast (flex f32, low-RAM) |
|---|---|---|
| 3,633 tokens | **46.7 s / 77.8 tok/s** | 106.2 s / 34.2 tok/s |
| 6,501 tokens | 124.2 s / 52.3 tok/s | 262.2 s / 24.8 tok/s |

- The 50 tok/s target is **not met** on the board (34.2 tok/s at 4 threads, 38.6 at 8;
  f16 36.1, blocked 29.1). Production is 2.28× faster on the same input.
- Scaling: 1 core 11.4, 2 cores 20.6, 4 cores 34.2, 8 cores 38.6 tok/s. Our single-core
  is 35% of the A76 f32 peak; production achieves ~30% of the int8 SDOT peak — it wins by
  using int8 kernels, which flex does not have.
- Numerics: board cosine vs the unquantized reference is 0.99928 at 82 tokens, identical
  to the dev host. Against the live server the number is lower only because production
  runs a Q8_0 KV cache (`-ctk q8_0 -ctv q8_0`): ours vs server-f16-KV 0.99906, server
  q8_0-KV vs f16-KV 0.99781.
- Memory: 771 MiB resident in low-RAM mode vs ~4 GB RSS for the production server at
  `-c 32768`.
- Verdict: production stays on `ik_llama.cpp`; see `docs/experiment-log.md` §9 for the
  full analysis. Closing the gap needs int8 GEMM, which flex lacks.

### Memory

`--quant q8` keeps the projection weights Q8_0-quantized in memory and dequantizes each
weight on the fly (only the current layer's f32 weights are materialized, ~62 MB). The
token-embedding table is kept in f16. Resident footprint:

| Mode | Anonymous RAM (after load) | Notes |
|---|---|---|
| `--quant none` (f32 weights) | ~2.4 GB | fastest for short inputs |
| `--quant q8` (low-RAM) | **~0.78 GB** | +~2 s per forward (flex dequantizes single-threaded) |

Peak RSS during load is ~2.3 GB even in low-RAM mode (the safetensors file is converted
to f32 before quantization); the resident figure is the steady state a server keeps. At
30k tokens activations dominate: peak 3,003 MiB anon in low-RAM mode vs 3,971 MiB in f32
mode. `libc::malloc_trim` is called after quantization because glibc otherwise retains
the freed f32 weight pages in its arenas.

## Usage

```sh
# single-core benchmark on a fixed text
taskset -c 2 cargo run --release -- bench --backend flex --dtype f32 --tokens 3633 --reps 2 --chunk 256

# embed one text (writes a JSON float array)
cargo run --release -- embed --backend flex --dtype f32 --quant q8 --text "Hello world" --out out.json

# OpenAI-compatible server
cargo run --release -- serve --backend flex --dtype f32 --quant q8 --port 8383 --max-tokens 30000
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":"Hello world","encoding_format":"float"}'
```

Flags: `--model-dir` (default `~/models/qwen3-embedding-0.6b`), `--backend cpu|flex`,
`--dtype f32|f16` (bf16 is broken in flex), `--quant none|q8` (q8 = Q8-resident low-RAM
mode), `--attn fused|blocked`, `--chunk`, `--key-block` (blocked attention only),
`--tokens`, `--text`, `--text-file`, `--out`, `--port`, `--max-tokens`, `--model-name`.
Subcommands: `bench`, `embed`, `gemm`, `serve`, `tokenize`.

Single-core runs are fastest with `--attn blocked --chunk 256 --key-block 256`; the
default `fused` path is better for long inputs and for multi-threaded serving.

### Cross-compiling for the board (aarch64)

```sh
cargo build --release --target aarch64-unknown-linux-gnu --no-default-features
```

`--no-default-features` drops the CubeCL/LLVM `cpu` backend, whose prebuilt LLVM bundle is
host-architecture and cannot link for aarch64. The `flex` backend (NEON on aarch64) is
always available and is the one this project uses. The linker is configured in
`.cargo/config.toml`. The resulting binary was smoke-tested under `qemu-aarch64`:
tokenizer ids and embeddings match the x86 build exactly (cosine 1.0).

## Design

- **Embedding-only model** (`src/model.rs`): 28 layers, GQA 16/8 heads, head_dim 128,
  QK-RMSNorm, RoPE θ=1e6, SwiGLU MLP, final RMSNorm + last-token pooling. `lm_head` is
  skipped, matching the patched production `ik_llama.cpp`.
- **Attention**: two interchangeable paths, both causal and numerically identical:
  - `fused` (default): `burn::tensor::module::attention` with `is_causal: true`; flex
    dispatches to a tiled flash-attention kernel (online softmax, `TILE_KV=64`) that
    parallelizes over heads and handles GQA natively (16 q heads vs 8 kv heads, no
    repeat needed). Wins on long inputs and multi-threaded runs.
  - `blocked`: a portable tensor-op implementation of the same blocked online softmax
    (`--chunk`/`--key-block`, defaults 256/256). Large GEMM shapes make it faster
    single-threaded at short/medium lengths.
- **Weights**: HF safetensors loaded through `SafetensorsStore` +
  `PyTorchToBurnAdapter` (Linear `[out,in]` → Burn `[in,out]`).
- **Q8 (low-RAM mode)**: `QuantScheme::default().with_value(Q8S).per_block([32], F16)`
  reproduces llama.cpp's Q8_0 quantization (scale = max_abs/127 per 32-value block).
  `--quant q8` keeps projection weights quantized and dequantizes each weight per
  forward (`Qwen3Embedding::quantized` / `linear_forward` in `src/model.rs`), which cuts
  resident RAM from ~2.4 GB to ~0.78 GB at a cost of ~2 s per forward — flex's
  `dequantize` is a single-threaded scalar loop and flex has no fused quantized GEMM.
  The embedding table is kept in f16 (flex's bf16 gather is broken; f16 is exact for
  bf16-sourced values in range). `libc::malloc_trim` releases the freed f32 pages.
- **Server** (`src/server.rs`): axum, `/v1/embeddings` (string, string list, token ids,
  base64), `/v1/models`, `/health`; requests are serialized through a mutex because the
  flex backend already uses rayon internally.

## Notes

- `Device::sync()` is required before reading wall-clock times; `to_data()` alone does not
  wait for execution.
- Tokenization uses the HF `tokenizer.json` with `add_special_tokens=true`, which appends
  the EOS token (151643) exactly like `llama-embedding` does.
- Backend choice: `flex` (AVX2/NEON CPU kernels, no JIT) is measurably faster here than
  `cpu` (CubeCL/LLVM) for small-batch inference; `cpu` pays a 3–8 s JIT cost per process
  and a ~3.3× penalty on transposed-B GEMM.
