# AGENTS.md — embeddings-fast

Qwen3-Embedding-0.6B inference in Rust/Burn. Read `README.md` for usage and
`docs/experiment-log.md` for measurements and findings. Git repo (`main`), created
2026-10-05.

## Status (2026-10-05)

- **CPU path** (`flex` backend, Q8 low-RAM): numerically faithful (cosine 0.9991-0.9995
  vs the production Q8_0 reference) and ~5× lower RAM (776 MiB vs ~4 GB), but 2.3× slower
  than the production `ik_llama.cpp` int8 server on the board (34.2 vs 77.8 tok/s at
  3,633 tokens). Production stays on `ik_llama.cpp`; flex has no int8 GEMM.
- **NPU path** (`--npu`, RK3588 `rocket` driver via `crates/burn-rocket`): projections and
  attention on the NPU. 45.7 tok/s and **-49% CPU-seconds** vs the CPU-only path at 3,633
  tokens, 298 MiB CPU-resident. Wall is ~20% better than CPU-only but still below
  production; its value is freeing CPU for other board services.
- 30k-token inputs work on the CPU path (fused flash attention, 52.4 tok/s at 32 dev
  threads); 30k with `--npu` is not yet measured. The OpenAI-compatible server is
  implemented and smoke-tested.
- Not done: int8 GEMM (the only lever that would close the speed gap; flex lacks it, and
  burn-cpu/CubeCL quantized matmul is unverified and cannot cross-compile) and board
  service deployment. The OpenViking entity
  (`viking://user/royalcat/memories/entities/software_project/embeddings_fast.md`) is
  written.

## Repo layout

| path | contents |
|---|---|
| `src/main.rs` | CLI (`bench`, `embed`, `gemm`, `serve`, `tokenize`), flags, model loading (f32 / `--quant q8` / `--npu` pack-and-drop), NPU packing |
| `src/model.rs` | model: layers, RoPE, RMSNorm, attention paths, `Proj::Cpu\|Npu`, stage timers |
| `src/npu.rs` | aarch64+npu only: `NpuModel` (resident weights), `NpuRef`, `NpuAttention`, `FusedRef`, NPU/convert timers |
| `src/server.rs` | axum OpenAI-compatible `/v1/embeddings` |
| `crates/burn-rocket/` | FFI to `librocketnpu`: RocketCtx/RocketWeight/RocketStream/RocketFaCtx, `pack_weight_seg`, `flash_attn`, `examples/probe.rs` |
| `vendor/rocketnpu/` | **gitignored**: `librocketnpu.a`, `librocketgraph.a`, headers — copy from the board's `/root/npu-poc/rocket-userspace/build` or build `gregordinary/rocket-userspace` |
| `docs/experiment-log.md` | all measurements: §1-8 dev-host/board CPU work, §9 board A/B, §10 NPU |

## Rebuild + deploy to the board

```sh
# NPU build (links vendor/rocketnpu/librocketnpu.a, or ROCKETNPU_DIR=<dir>)
cargo build --release --target aarch64-unknown-linux-gnu --no-default-features --features npu
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/embeddings-fast \
    root@rock-5b-plus.lan:/root/embeddings-fast/
# CPU-only build: drop --features npu
```

The binary is self-contained (`librocketnpu` is statically linked); the model lives at
`/root/models/qwen3-embedding-0.6b/` on the board. On the board run from
`/root/embeddings-fast` (the bench uses the relative `data/bench_text.txt`).

The `bench` summary prints `stages: attention/mlp/norms` and, with `--npu`, an
`npu breakdown: calls/convert/npu/flex+overhead` line — check these before profiling.
The NPU counters live in `src/npu.rs` (`npu_stats`), the stage counters in
`src/model.rs` (`stage_stats`).

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
- Measured verdict, CPU path (4 threads, 3,633 tokens): production 46.7 s (77.8 tok/s) vs
  ours 106.2 s (34.2 tok/s) — **target ≥50 tok/s is not met on the board**; production is
  2.28× faster. (The NPU path below reaches 45.7 tok/s at -49% CPU.) Our single-core is 35% of the A76 f32 peak and flex's unary/binary ops have no
  rayon, so scaling caps at ~3×; flex has no int8 GEMM. f16 does not help (36.1 tok/s).
- RAM: low-RAM mode is 771 MiB resident on the board vs ~4 GB for the production server;
  the Q8 dequant tax is ~7 s per forward on A76 (single-threaded scalar).
- Numerics: board cosines match the dev host (0.99928 at 82 tokens vs the unquantized
  reference). The production server's `-ctk q8_0 -ctv q8_0` lowers server-vs-ours cosine
  to ~0.998 by itself (server q8_0-KV vs f16-KV is 0.99781).
- The `data/true3633.txt` file on the board is a true 3,633-token input (built by byte
  truncation + token count check); `one_3633.txt` is 6,501 tokens despite the name.
- The deployed binary as of 2026-10-05 is the NPU-enabled build and the board is running
  the 600 MHz patched `rocket` module (revert: `rmmod rocket && modprobe rocket`, or
  reboot). `--npu` runs need no extra setup beyond that.

## NPU offload (`--npu`, aarch64)

- `crates/burn-rocket` = FFI to `librocketnpu` (rocket-userspace). Build the app with
  `--no-default-features --features npu`; the crate links `vendor/rocketnpu/librocketnpu.a`
  (override with `ROCKETNPU_DIR`, e.g. `/root/npu-poc/rocket-userspace/build` on the board).
  aarch64 only: the dep is target-gated in Cargo.toml.
- `--npu` = pack-and-drop: 196 projections packed into resident fp16 NPU BOs (0.82 GiB),
  f16 embedding table, 298 MiB CPU-resident. `--npu-attn npu` (default) also offloads
  attention via `rocket_flash_attn_fp16_ctx`; `cpu` keeps flex attention (within ~1 s on
  wall, ~40% more CPU). `--npu` requires `--dtype f32` and excludes `--quant q8`.
- Resident weights are packed for the M>=256 tiling; requests with fewer rows are padded
  to 256 rows (the extra rows are ignored on readback). One pack serves all lengths.
- Board: the 600 MHz patched module (`insmod /root/npu-poc/rocket-patched-600/rocket-npu600.ko
  rocket_npu_clk_hz=600000000` after `rmmod rocket`; contained, reboot reverts) is ~3x the
  stock 200 MHz boot clock. `librocketnpu` must be the built archive from the board's
  `/root/npu-poc/rocket-userspace` (it is GPL-3.0-or-later).
- Measured (3,633 tok, 4 threads, cores 4-7, 2 reps): CPU-only 99.9 s / 677 CPU-s;
  `--npu --npu-attn cpu` 78.7 s / 488 CPU-s; `--npu` (NPU attention, default) 79.5 s /
  344 CPU-s. The aarch64 build sets `target-feature=+fp16` (hardware FCVT; RK3588 is
  ARMv8.2) and the NPU glue (f32<->f16, head-major gather/scatter) is rayon-parallel.
- q|k|v and gate|up are packed as one segmented resident weight each
  (`pack_weight_seg`, concatenated along N): 196 tensors -> 112 resident weights, 4
  matmuls per layer, one input conversion per group. Loader packs everything first and
  assigns handles after (each handle clones the model Arc; `Arc::get_mut` needs the sole
  owner during packing).
- Attention trade-off: `rocket_flash_attn_fp16_ctx` brings the full score matrix
  host-side for the causal mask + softmax, so at 2-7k tokens it is about as fast as
  flex's fused flash kernel (within ~1 s) while using far less CPU. `ROCKET_FA_TILE_KV`
  (tiled path) is *worse* at 3.6k (96.9 s vs 79.5 s); it engages automatically >8k keys.
  The mask + head-major f16 scratch are cached per sequence length in `NpuAttention`.
- Safetensors keys in `model.safetensors` have **no `model.` prefix**
  (`layers.0.self_attn.q_proj.weight`, `embed_tokens.weight`); the loader reads the
  projection tensors by that key after `load_from` has taken the rest.
- Reverting the board NPU clock: `rmmod rocket && modprobe rocket` (or reboot) restores
  the stock 200 MHz in-tree module; nothing in `/lib/modules` was modified.

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

## Future work

- **int8 GEMM** is the only lever that would close the speed gap with production. flex has
  none; the candidates are the `burn-cpu` (CubeCL/LLVM) backend (CPU quantized matmul
  unverified, cannot cross-compile — would need a native build + JIT on the board) or a
  custom int8 microkernel (upstream contribution).
- Measure 30k tokens with `--npu` (the CPU path does 52.4 tok/s at 32 dev threads) and
  re-check `ROCKET_FA_TILE_KV` above 8k keys, where the tiled path engages by default.
- Board service: run `serve --npu --max-tokens 30000` under a supervisor, and decide
  whether the CPU-relief mode should be the default there (it is now).
- Optional: quantize tensor-by-tensor during load to remove the ~2.3 GB load-time peak in
  `--quant q8` mode; re-pack small-M weights lazily instead of padding to 256 rows.

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

On the board (after the cross-build + scp above; the NPU-enabled binary):

```sh
ssh root@rock-5b-plus.lan
cd /root/embeddings-fast
# speed / CPU-relief A/B (2 reps; watch the stages + npu breakdown lines and `time`)
taskset -c 4-7 ./embeddings-fast bench --backend flex --dtype f32 --npu --tokens 3633 --reps 1
taskset -c 4-7 ./embeddings-fast bench --backend flex --dtype f32 --npu --npu-attn cpu --tokens 3633 --reps 1
# numerics vs the production reference (copy the JSON back and compare cosines)
./embeddings-fast embed --backend flex --dtype f32 --npu --text "Hello world, this is a test." --out /tmp/npu_short.json
```

Reference JSONs used for the cosine checks live on the dev host in `/tmp/opencode/`
(`ref_causal.json`, `ref1_{64,512,3633}.json`, `one_{64,512,3633}.txt`) and partially on
the board in `/root/embeddings-fast/data/`. Regenerate them with the `llama-embedding`
invocation in "Environment facts" if missing.
