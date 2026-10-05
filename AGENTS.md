# AGENTS.md — embeddings-fast

Qwen3-Embedding-0.6B inference in Rust/Burn. Read `README.md` for usage and
`docs/experiment-log.md` for measurements and findings.

## Environment facts

- `$CARGO_TARGET_DIR` is a shared cache (`/home/royalcat/.cache/rust/target`); never
  assume a project-local `target/`.
- Model: `~/models/qwen3-embedding-0.6b/` (HF bf16 safetensors + tokenizer.json).
- Production reference: `~/projects/ik_llama.cpp/build/bin/llama-embedding` +
  `~/models/qwen3-embedding-0.6b-q8_0.gguf` (patched skip-lm-head build).
  Correct reference invocation:
  `llama-embedding -m <gguf> -p "$(cat file)" --pooling last --embd-normalize -1
  --embd-output-format json -t 8 -c 8192 -b 8192 -ub 8192`
  (`-b` must be ≥ `-c`; `-f` and multi-line `-p` split the text on `\n` into separate
  sequences — use newline-free text for single-embedding comparisons).
- Board `rock-5b-plus.lan` is accessed as `root@rock-5b-plus.lan` (the `royalcat` user
  has no key there). Do not touch the board unless the task says so.

## Board deployment (2026-10-05)

- Deployed: `/root/embeddings-fast/embeddings-fast` (cross-built aarch64 binary + `data/`)
  and `/root/models/qwen3-embedding-0.6b/`. Run from `/root/embeddings-fast` (the bench
  uses the relative `data/bench_text.txt`); pin to the A76s with `taskset -c 4-7`.
- Production A/B (reproducible live): `/opt/llama-ik/bin/llama-server -m
  /var/lib/docker/volumes/llama-swap_models/_data/qwen3-quants/Qwen3-Embedding-0.6B-Q8_0.gguf
  --embedding --pooling last -c 32768 -b 32768 -ub 32768 -np 1 -ctk q8_0 -ctv q8_0 -t 4
  --host 127.0.0.1 --port 9293` with `LD_LIBRARY_PATH=/opt/llama-ik/lib`; `pkill -x
  llama-server` to stop (do NOT use `pkill -f` — the pattern matches the SSH wrapper's own
  command line and kills the session).
- Measured verdict (4 threads, 3,633 tokens): production 46.7 s (77.8 tok/s) vs ours
  106.2 s (34.2 tok/s) — **target ≥50 tok/s is not met on the board**; production is 2.28×
  faster. Our single-core is 35% of the A76 f32 peak and flex's unary/binary ops have no
  rayon, so scaling caps at ~3×; flex has no int8 GEMM. f16 does not help (36.1 tok/s).
- RAM: low-RAM mode is 771 MiB resident on the board vs ~4 GB for the production server;
  the Q8 dequant tax is ~7 s per forward on A76 (single-threaded scalar).
- Numerics: board cosines match the dev host (0.99928 at 82 tokens vs the unquantized
  reference). The production server's `-ctk q8_0 -ctv q8_0` lowers server-vs-ours cosine
  to ~0.998 by itself (server q8_0-KV vs f16-KV is 0.99781).
- The `data/true3633.txt` file on the board is a true 3,633-token input (built by byte
  truncation + token count check); `one_3633.txt` is 6,501 tokens despite the name.

## NPU offload (`--npu`, aarch64)

- `crates/burn-rocket` = FFI to `librocketnpu` (rocket-userspace). Build the app with
  `--no-default-features --features npu`; the crate links `vendor/rocketnpu/librocketnpu.a`
  (override with `ROCKETNPU_DIR`, e.g. `/root/npu-poc/rocket-userspace/build` on the board).
  aarch64 only: the dep is target-gated in Cargo.toml.
- `--npu` = pack-and-drop: 196 projections packed into resident fp16 NPU BOs (0.82 GiB),
  f16 embedding table, 298 MiB CPU-resident. `--npu-attn npu` (default) also offloads
  attention via `rocket_flash_attn_fp16_ctx`; `cpu` keeps flex attention (faster wall,
  less CPU relief). `--npu` requires `--dtype f32` and excludes `--quant q8`.
- Resident weights are packed for the M>=256 tiling; requests with fewer rows are padded
  to 256 rows (the extra rows are ignored on readback). One pack serves all lengths.
- Board: the 600 MHz patched module (`insmod /root/npu-poc/rocket-patched-600/rocket-npu600.ko
  rocket_npu_clk_hz=600000000` after `rmmod rocket`; contained, reboot reverts) is ~3x the
  stock 200 MHz boot clock. `librocketnpu` must be the built archive from the board's
  `/root/npu-poc/rocket-userspace` (it is GPL-3.0-or-later).
- Measured (3,633 tok, 4 threads, cores 4-7, 2 reps): CPU-only 99.9 s / 677 CPU-s;
  `--npu --npu-attn cpu` 80.1 s / 484 CPU-s; `--npu` (NPU attention, default) 81.3 s /
  351 CPU-s. The aarch64 build sets `target-feature=+fp16` (hardware FCVT; RK3588 is
  ARMv8.2) and the NPU glue (f32<->f16, head-major gather/scatter) is rayon-parallel.

## Stack notes

- `burn = "=0.22.0-pre.4"`, `burn-store = "=0.22.0-pre.4"`; `flex` backend is the primary
  path (`cpu` = CubeCL LLVM pays JIT cost and has a transposed-B penalty).
- `Device::sync()` is required before timing or reading results.
- `Linear` weights are `[in, out]`; use `PyTorchToBurnAdapter` when loading HF
  safetensors.
- flex's quantized path (`QFloat`) is a trap for inference: `q_matmul` has no fused
  kernel in flex (it dequantizes + f32 matmul), and `dequantize` is a single-threaded
  scalar loop (~125 M elem/s), while layout ops (`unsqueeze`, `slice`, …) on
  block-quantized tensors dequantize + requantize. Dequantize a weight at most once per
  forward.
- `--quant q8` = low-RAM mode: projection weights stay Q8_0-quantized and are
  dequantized once per layer per forward (`linear_forward` + `Qwen3Embedding::quantized`
  in `src/model.rs`); the embedding table is f16; `libc::malloc_trim` after quantization
  releases the freed f32 pages (glibc otherwise retains ~1.3 GB in arenas). Resident
  ~0.78 GB anon vs ~2.4 GB for f32, cost ~2 s per forward. Load peak is still ~2.3 GB
  (the file is materialized as f32 before quantization).
- `--dtype bf16` panics in flex 0.22.0-pre.4 (bf16 embedding gather, "storage: dtype
  mismatch (expected BF16, got F32)"); `load_model` bails with a clear message. f16 works
  but is ~1.9× slower at model level.
- Tokenizer must run with `add_special_tokens=true` (appends EOS 151643), matching
  `llama-embedding` token counts.
- f16 weights are ~1.9× slower than f32 at model level on this CPU despite similar GEMM
  microbenchmarks — use f32.
- Long inputs are attention-bound: at n=6501 tokens attention FLOPs ≈ all linear layers
  combined (28 layers × 4096 × n FLOP/token).
- Attention has two paths (`--attn fused|blocked`), both causal and numerically identical:
  - `fused` (default) = `burn::tensor::module::attention(..., AttentionModuleOptions {
    is_causal: true })`; flex uses a tiled flash-attention kernel (TILE_KV=64, rayon over
    heads, native GQA — do not `repeat_dim` for it). Best for long inputs / multi-thread.
  - `blocked` = tensor-op blocked online softmax (`--chunk`/`--key-block`, defaults
    256/256; the flags are ignored by `fused`). Best single-threaded at short/medium
    lengths (70.2 vs 45.5 tok/s at 3.6k on the dev host).
  - `perf` shows the forward is GEMM-bound at ~98% of the flex f32 microbenchmark peak;
    further speedups need a faster GEMM (int8), not more attention tuning.
- Cross-compile: `.cargo/config.toml` sets `aarch64-linux-gnu-gcc` for
  `aarch64-unknown-linux-gnu`; artifacts land in
  `$CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/`. Build with
  `--no-default-features` (the default `cpu` feature pulls in cubecl-llvm, whose prebuilt
  LLVM bundle is host-arch and fails to link for aarch64; `flex` needs none of it).

## Verification commands

```sh
B=$CARGO_TARGET_DIR/release/embeddings-fast
# single-core speed gate (blocked path is the single-core record)
taskset -c 2 $B bench --backend flex --dtype f32 --tokens 3633 --reps 2 --attn blocked --chunk 256 --key-block 256
# multi-threaded / long-input (fused default)
$B bench --backend flex --dtype f32 --text-file /tmp/opencode/long30k.txt --tokens 30000 --reps 0
# embedding + cosine against a reference JSON
$B embed --backend flex --dtype f32 --quant q8 --text-file /tmp/opencode/one_64.txt --out /tmp/opencode/our.json
# server smoke test
$B serve --backend flex --dtype f32 --quant q8 --port 8383 &
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"hi"}'
```
