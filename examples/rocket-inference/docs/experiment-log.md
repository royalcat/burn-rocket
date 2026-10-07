# Experiment log: Qwen3-Embedding-0.6B on Burn

> Layout note (2026-10-06): the repository was inverted — `burn-rocket` is now the
> root crate and this app lives in `examples/rocket-inference/` (renamed from
> `examples/qwen3-embeddings/` on 2026-10-06, when it gained the Qwen3.5 intent-model
> server). Paths like `crates/burn-rocket` and board dirs like `/root/embeddings-fast/`
> below are the historical pre-inversion locations; the measurements are unchanged.

Goal: run Qwen3-Embedding-0.6B (1024-dim, last-token pooling) in Rust/Burn, at least
50 tok/s on one core, with Q8 weights and an OpenAI-compatible server. Board (RK3588)
work is deliberately deferred; all numbers below are from the development host.

Environment:

- Host: AMD Ryzen 9 5950X (Zen3, 16c/32t, AVX2+FMA, no AVX-512), Linux, rustc 1.98.1.
- `CARGO_TARGET_DIR=/home/royalcat/.cache/rust/target` (shared).
- Stack: `burn = "=0.22.0-pre.4"` with `flex` + `rayon` + `simd` features,
  `burn-store` safetensors, `tokenizers` 0.21, axum server.
- Reference/production: patched `~/projects/ik_llama.cpp` (`embeddings-skip-lm-head`),
  `llama-embedding` with `~/models/qwen3-embedding-0.6b-q8_0.gguf`.
- Model source: `~/models/qwen3-embedding-0.6b/` (HF bf16 safetensors + tokenizer.json).

## 1. Backend and GEMM characterization

GEMM shape `[3632,1024] × [1024,3072]` (the model's `o_proj`-sized matmul), best of runs,
after `device.sync()`:

| Backend | dtype | threads | GFLOPS | notes |
|---|---|---|---|---|
| burn-cpu (CubeCL LLVM) | f32 | 32 | 506 | 5–8 s JIT per process |
| burn-cpu | f32, transposed rhs | 32 | 155.7 | 3.3× penalty |
| burn-cpu | f16 / bf16 | 32 | 547 / 391 | |
| burn-flex | f32 | 1 | 124.3 | ~60 % of AVX2 peak |
| burn-flex | f32 | 32 | 493.8 | memory-bandwidth limited |
| burn-flex | f16 / bf16 | 1 | 104.6 / 116.7 | |
| burn-flex | f32, transposed rhs | 1 | 126.2 | no penalty (row-major rhs native) |

Findings:

- `burn-cpu` (CubeCL LLVM) wins on big multi-threaded GEMMs but pays a JIT cost and a
  transposed-B penalty; `flex` has no JIT, handles row-major weights natively, and is
  faster single-threaded. `flex` was chosen as the primary backend.
- Burn's autotune picked `cpu_gemm_a0.5` for the CubeCL CPU backend (46 ms vs 169 ms for
  the generic path) when the backend is used at all.
- Timing without `device.sync()` is meaningless (first attempt reported 491k GFLOPS).

## 2. Model implementation

`src/qwen3_embedding/model.rs`: embedding-only Qwen3 (no `lm_head`), 28 layers, hidden 1024, 16/8 heads,
head_dim 128, intermediate 3072, QK-RMSNorm, RoPE θ=1e6, SwiGLU, final RMSNorm,
last-token pooling. Chunked causal attention: query chunks of size `--chunk` attend to
all preceding keys, so the score tensor stays `[1,16,chunk,seq]`.

Weights load via `SafetensorsStore` + `PyTorchToBurnAdapter` (transposes Linear
`[out,in]` to Burn `[in,out]`) chained with `FloatCastAdapter` for dtype conversion.
All 310 tensors map; no missing/unused keys.

### Full-model results

Input: 3633 tokens from `data/bench_text.txt` (markdown concatenation), all cores vs one
pinned core (`taskset -c 2`):

| Threads | chunk | time | speed |
|---|---|---|---|
| 32 | 512 | 16.9 s | 215.3 tok/s |
| 1 | 256 | 55.1 s | **65.9 tok/s** |
| 1 | 512 | 56.1 s | 64.8 tok/s |
| 1 | 1024 | 59.0 s | 61.6 tok/s |
| 1 | 2048 | 64.3 s | 56.5 tok/s |

Single-core effective throughput: linear layers are 2 × 382 MFLOP/token (382 M matmul
params; the 155 M embedding-table params are a lookup, not FLOPs). Attention adds
28 layers × 4096 × n FLOP/token, i.e. 417 MFLOP/token at n=3633 and 745 MFLOP/token at
n=6501 — on par with all linear layers combined. Including attention, the single-core
runs sustain ≈78 GFLOPS (4.3 TFLOP in 55.1 s at 3633 tokens; 10.4 TFLOP in 136.0 s at
6501 tokens), i.e. ~63 % of the best measured f32 GEMM rate. 32-thread scaling is only
~3.3×, indicating memory-bandwidth limitation.

Additional single-core results:

| Config | Tokens | chunk | time | speed |
|---|---|---|---|---|
| f32/Q8 | 6501 | 512 | 136.0 s | 47.8 tok/s |
| f32/Q8, via server | 6501 | 512 | 136.4 s | 47.6 tok/s |
| f16 | 3633 | 256 | 102.7 s | 35.4 tok/s |

f16 weights are a dead end on this host: despite near-parity in the isolated GEMM
microbenchmark (104.6 vs 124.3 GFLOPS), the full model is ~1.9× slower in f16. The server
adds no measurable overhead over the CLI for the same input.

Chunk sweep at 6501 tokens (single core, f32/Q8): 128 → 137.9 s, 256 → 143.4 s,
512 → 136.0 s, 1024 → 170.4 s. Between 128 and 512 the timing is roughly flat (~5 %
spread); 1024 and above degrade clearly because the `[1,16,chunk,seq]` score workspace
and its elementwise passes grow with chunk. The default chunk of 512 is a good choice;
avoid ≥1024 for long inputs.

## 3. Q8 weights

Scheme: `QuantScheme::default().with_value(QuantValue::Q8S).per_block([32], ScaleDtype::F16)`
→ symmetric int8 with scale = max_abs/127 per 32-value block, i.e. exactly llama.cpp
Q8_0. Applied to `q/k/v/o/gate/up/down_proj.weight` only (regex `ParamGroup`).

Pathology found: keeping weights as `QFloat` and using `Linear::forward`'s quantized
path was catastrophic for short inputs — a 9-token forward took 16.3 s (vs 0.171 s f32).
Cause: for 3-D inputs `Linear::forward` can hit the `Keep` branch and call
`weight.unsqueeze::<3>()`; flex implements layout ops on block-quantized tensors as
dequantize → op → **re-quantize** (`block_safe_layout_op`), so every matmul paid a full
weight dequantize + dynamic requantize. Even without that, flex's default `q_matmul`
dequantizes the entire weight on every call (microbenchmark: `[9,1024]×QFloat[1024,3072]`
= 21 ms vs 1 ms f32; `[3632,1024]×QFloat[1024,3072]` = 76 ms vs 46 ms f32). There is no
fused int8 matmul on the flex CPU backend.

Chosen approach (`--quant q8`): quantize at load, then immediately dequantize
(`Q8Dequant` `ModuleMapper`), so weights are Q8-rounded but stored as f32. Speed is
identical to f32 (65.9 tok/s single-core), memory is the same (2.4 GB), and the values
match production Q8.

## 4. Numerical validation

Reference: `llama-embedding -m qwen3-embedding-0.6b-q8_0.gguf -p <text> --pooling last
--embd-normalize -1 --embd-output-format json -t 8 -c 8192 -b 8192 -ub 8192`.

Two gotchas discovered while validating:

1. `llama-embedding` splits the prompt on `\n` (default `--embd-sep`) and embeds each line
   as a separate sequence. References must use newline-free text to compare against a
   single embedding.
2. `llama-tokenize` does **not** append EOS, while `llama-embedding` and the HF tokenizer
   (`add_special_tokens=true`, `add_eos_token=true` in `tokenizer_config.json`) append
   token 151643. With EOS included, our token counts match `llama-embedding` exactly.

Cosine similarity (raw, unnormalized vectors), text slices of the bench file:

| Tokens | f32 weights | Q8 weights |
|---|---|---|
| 82 | 0.999375 | 0.999280 |
| 955 | 0.999411 | 0.999126 |
| 6501 | 0.999485 | 0.999209 |

Also verified: causal vs non-causal attention changes the embedding
(cos(default, non-causal) = 0.14), and the default (causal) matches production
(cos = 1.0 against the production default).

## 5. Server

`src/qwen3_embedding/server.rs` (axum): `POST /v1/embeddings` with OpenAI-compatible request/response,
accepting a string, a list of strings, token ids, or a list of token id lists;
`encoding_format: float|base64`; `GET /v1/models`, `GET /health`; inputs truncated at
`--max-tokens` (default 30000). A mutex serializes requests because flex already
parallelizes internally with rayon.

## 6. Board build and next steps

- aarch64 cross-build works:
  `cargo build --release --target aarch64-unknown-linux-gnu --no-default-features`
  (flex only). The default `cpu` feature pulls in cubecl-llvm, whose prebuilt LLVM
  bundle is host-arch and fails to link for aarch64; the `cpu` feature is now optional in
  `Cargo.toml`. `.cargo/config.toml` sets the `aarch64-linux-gnu-gcc` linker; flex uses
  `macerator` NEON kernels and the `gemm` crate's aarch64 support, no JIT involved.
- The aarch64 binary was validated under `qemu-aarch64`: tokenizer ids identical to x86,
  embedding cosine 1.000000000 (max abs diff 3e-5), model load and Q8
  quantize/dequantize all work.
- Still to do on the board: copy the binary + model, run pinned to cores 4–7
  (`taskset -c 4-7`), 4 threads, same 3633-token input; compare with the current
  production 33.7 s / ~108 tok/s.
- On the bandwidth-limited RK3588, re-check: thread count (4 vs 8), chunk size, and
  whether per-matmul QFloat dequantization (0.6 GB streamed instead of 2.4 GB f32) wins
  over resident f32 despite extra dequant compute.
- Long inputs: see section 7 — blocked online softmax and the backend's fused flash
  kernel, with 30k-token measurements.

## 7. Long-input attention: blocked online softmax and the fused kernel

### 7.1 Problem

At 6501 tokens, attention FLOPs (28 layers × 4096 × n per token ≈ 745 MFLOP/token) match
all linear layers (764 MFLOP/token). The original implementation materialized a
`[1,16,chunk,q1]` score tensor per query chunk and ran ~8 full-tensor passes over it
(mask build, `mask_fill`, softmax, scale, casts). At 30k tokens the summed score element
count is ~205 G elements (~820 GB per pass at f32), making attention memory-bound and
slow.

### 7.2 Blocked online softmax (tensor ops, `--attn blocked`)

K/V are pre-sliced into fixed blocks once per layer; each query block iterates key blocks
up to the diagonal with a running max/normalizer/accumulator (flash-attention structure)
expressed in plain Burn ops. Only diagonal blocks build a causal mask. Score tensors stay
at `[1,16,chunk,key_block]` (4.2 MB at 256/256) instead of growing with the sequence.

- Single core, 3633 tokens: 55.1 → 51.7 s; 6501 tokens: 136.0 → 125.5 s (≈7% faster).
- Numerics identical: the 9-token embedding is bit-identical to the previous
  implementation, all reference cosines unchanged.
- `perf` profile of the 3633-token forward (62k samples): 57.5% f32 GEMM microkernel,
  ~11% softmax/exp/sigmoid, ~3% GEMM packing, remainder scattered elementwise ops. The
  GEMM runs at ~98% of the 124 GFLOPS single-core microbenchmark peak — the flex f32
  path is saturated.
- Chunk/key_block sweep, 6501 tokens, single core: 256/256 125.5 s (best), 512/256
  130.2, 512/512 127.7, 512/1024 128.9, 1024/512 134.8. At 30k tokens (32 threads):
  256/256 1078.6 s, 512/512 1017.1 s, 512/1024 973.9 s — larger blocks win as the
  per-block op overhead dominates.

### 7.3 The backend's fused attention kernel (`--attn fused`, default)

flex exposes a fused attention op through the standard Burn API:
`burn::tensor::module::attention(q, k, v, mask, attn_bias, AttentionModuleOptions {
scale, softcap, is_causal })`. flex auto-selects a naive path (materialize scores, two
GEMMs; score matrix ≤256K elements) or a tiled flash path (online softmax, `TILE_KV=64`,
`gemm` microkernels, rayon over batch×heads). GQA is native (`q_heads % kv_heads == 0`),
so the `repeat_dim` expansion is skipped.

| Attention | Tokens | Threads | Time | Speed |
|---|---|---|---|---|
| blocked 256/256 | 3633 | 1 | 51.7 s | 70.2 tok/s |
| fused | 3633 | 1 | 79.9 s | 45.5 tok/s |
| blocked 256/256 | 3633 | 4 | 25.6 s | 141.8 tok/s |
| fused | 3633 | 4 | 25.6 s | 142.1 tok/s |
| blocked 256/256 | 6501 | 1 | 125.5 s | 51.8 tok/s |
| fused | 6501 | 1 | 180.2 s | 36.1 tok/s |
| blocked 256/256 | 6501 | 4 | 68.5 s | 94.9 tok/s |
| fused | 6501 | 4 | 61.6 s | 105.6 tok/s |
| blocked 512/1024 | 30000 | 32 | 973.9 s | 30.8 tok/s |
| fused | 30000 | 32 | 572.8 s | 52.4 tok/s |

- 1 thread: blocked wins at short/medium lengths. The flash kernel computes QK^T per
  64-wide KV tile, re-reading Q (and its per-row softmax state) once per tile; the blocked
  path uses one large-n GEMM per query block.
- 4 threads: equal at 3.6k, fused wins at 6.5k (105.6 vs 94.9 tok/s).
- 32 threads, 30k: fused 1.7× faster (572.8 vs 973.9 s); blocked score traffic and op
  count grow with n², while the flash kernel keeps scratch O(TILE_KV) per row and
  parallelizes over 16 heads.
- Both paths produce identical embeddings (cosine 1.000000 at 9 and 6501 tokens; cosines
  vs production unchanged: 0.999209 at 6501).

Decision: `fused` is the default (multi-threaded serving and long inputs are the
production workload); `--attn blocked --chunk 256 --key-block 256` is the single-core
tuned path (70.2 tok/s at 3.6k, above the 50 tok/s single-core target).

Thread scaling (blocked 256/256, 3633 tokens): 70.2 (1t), 105.9 (2t), 141.8 (4t), 155.0
(8t), 155.4 (16t), 144.2 (32t) tok/s — a ~2.2× cap shows the non-GEMM share is
memory-bound.

### 7.4 30k-token validation

The 30k-token path (server default `--max-tokens 30000`) completes with fused attention
at 32 threads in 572.8 s (52.4 tok/s), no OOM. Embeddings stay finite and path-consistent
(cosine 1.000000 fused vs blocked at 9 and 6501 tokens).

## 8. Low-RAM Q8-resident mode

Motivation: the f32 model is 2.37 GB of weights (592 M params × 4 B), and the target
board has 15.4 GiB shared with other services. Measured footprint of the f32 build:
2,400 MiB anonymous (hard), 1,176 MiB file-backed (the mmapped bf16 safetensors,
reclaimable), 3,462 MiB peak RSS; 3,971 MiB peak at 30k tokens (activations). Production
`ik_llama.cpp` peaks at ~3.6 GiB at 28k (Q8 weights + KV cache).

### 8.1 Keeping weights quantized

`--quant q8` now keeps projection weights Q8_0-quantized in memory and dequantizes each
weight once per layer per forward (`linear_forward` in `src/util/proj.rs`), so only the
current layer's f32 weights (~62 MB) are materialized. First attempt was numerically
exact (cosine 1.000000 vs the f32-resident Q8 path) but had two problems:

- **Speed**: a 9-token forward took 6.9 s instead of 0.16 s. flex has no fused quantized
  GEMM — `q_matmul` falls back to `dequantize` + `float_matmul` — and flex's
  `dequantize` is a single-threaded scalar loop with a per-element block lookup
  (`BlockLayout::block_of(index)`), ~63-125 M elem/s, i.e. ~2 s for the 437 M projection
  values per forward. Amortized over long inputs this is small; for short requests it is
  a fixed tax.
- **RSS didn't drop**: glibc retained the freed f32 pages in its arenas (~1.3 GB), so
  resident anon stayed at ~2.4 GB. Fix: `libc::malloc_trim(0)` after quantization.

The embedding table is kept in f16 (155 M params, 0.31 GB): a bf16 table panics in flex
0.22.0-pre.4 (`bf16 embedding gather: "storage: dtype mismatch (expected BF16, got
F32)"` — flex bug, `float_select` dispatches bf16 but a lower layer reads f32 storage),
while f16 works and is exact for bf16-sourced values in range. `load_model` now bails
with a clear message for `--dtype bf16`.

### 8.2 Results

| Metric | f32 (`--quant none`) | low-RAM (`--quant q8`) |
|---|---|---|
| Resident anon after load | 2,400 MiB | **776 MiB** |
| Peak RSS incl. load | 3,462 MiB | 3,450 MiB (load transient) |
| 3633 tok, 1 core, blocked | 51.0 s / 71.3 tok/s | 53.1 s / 68.5 tok/s |
| 3633 tok, 4 threads, fused | 20.6-25.6 s | 22.4 s / 162.3 tok/s |
| 6501 tok, 4 threads, fused | 61.6 s / 105.6 tok/s | 54.7 s / 118.9 tok/s |
| 30000 tok, 32 threads, fused | 550.1 s / 54.5 tok/s | 565.9 s / 53.0 tok/s |
| 9-token request | 0.16 s | 3.4 s |

Peak anonymous RSS at 30k tokens: 3,971 MiB (f32) vs 3,003 MiB (low-RAM) — at long
inputs activations dominate and the weight saving shrinks to ~1 GB.

The steady-state win is 3.1× (2,400 → 776 MiB). The load transient is unavoidable in the
current design: safetensors tensors are materialized as f32 before quantization, so peak
RSS during load is ~2.3-2.5 GB; quantizing tensor-by-tensor during load would fix it.
The per-forward dequant cost (~2 s, single-threaded) is invisible for long inputs and
dominates very short ones.

Decision: `--quant q8` is the low-RAM mode (keeps production Q8_0 numerics); `--quant
none` remains the f32 mode for maximum short-request speed.

## 9. Board deployment and A/B vs production (rock-5b-plus, 2026-10-05)

Deployed with user approval: the cross-built aarch64 binary plus the HF model were copied
to `/root/embeddings-fast/` and `/root/models/qwen3-embedding-0.6b/`. All runs pinned to
the A76 cores 4-7 with 4 threads (matching production), `schedutil` governor (2.4 GHz
under load). The production stack was not running, so it was reproduced live from
`/opt/llama-ik/bin/llama-server` with the documented args
(`--embedding --pooling last -c 32768 -b 32768 -ub 32768 -np 1 -ctk q8_0 -ctv q8_0 -t 4`)
on a spare port.

### 9.1 Speed

| Input (identical text) | Production ik_llama.cpp Q8_0 | embeddings-fast (flex f32) |
|---|---|---|
| 3,633 tokens | **46.69 s / 77.8 tok/s** | 106.2 s / 34.2 tok/s |
| 6,501 tokens | 124.24 s / 52.3 tok/s | 262.2 s / 24.8 tok/s |

Ours at 3,633 tokens, other configs (4 threads): f32 102.6 s (35.4 tok/s), f16 100.6 s
(36.1), low-RAM blocked 256/256 124.8 s (29.1); all 8 cores 94.2 s (38.6 tok/s).

**Target ≥50 tok/s at 3.6k tokens is not met** on the board: 34.2 tok/s at 4 threads,
38.6 at 8. Production is 2.28× faster on the same input.

- Thread scaling (low-RAM, fused): 1 core 11.4, 2 cores 20.6, 4 cores 34.2, 8 cores 38.6
  tok/s (3.0× from 1 to 4 threads).
- Ours single-core: 13.5 GFLOPS = 35% of the A76 f32 peak (38.4 GFLOPS/core); production
  effective ≈92 GFLOPS = 30% of the int8 SDOT peak (~307 GOPS), i.e. production wins by
  using int8 kernels, not by better efficiency.
- flex has no int8 GEMM; its f32 NEON kernels reach ~35% of peak per core, and its
  unary/binary ops have no rayon (single-threaded — verified in `ops/unary.rs`,
  `ops/binary.rs`), so scaling caps at ~3×. f16 does not help (36.1 tok/s) despite the
  A76's fp16 support.
- The documented 33.7 s production baseline did not reproduce live (46.7 s on the same
  text/config); use 46.7 s as the current number.
- The production server with f16 KV cache is faster than with `-ctk q8_0 -ctv q8_0` on
  short/medium inputs (955 tokens: 9.07 s vs 12.59 s) — KV quantization costs both
  precision and speed here.

### 9.2 Numerics

- Board parity vs CLI references: 82 tokens cosine 0.99928 (identical to the dev host).
- The 3,633-token comparison against the live production *server* gives 0.99795, but that
  is the server's `-ctk q8_0 -ctv q8_0` KV quantization, not our error:
  955 tokens: ours vs CLI ref (no KV quant) 0.999126; ours vs server f16 KV 0.999057;
  ours vs server q8_0 KV 0.997552; server f16 KV vs CLI ref 0.999774; server q8_0 KV vs
  f16 KV 0.997813.

### 9.3 Memory and operations

- Low-RAM mode resident: **771 MiB** on the board (dev host 776 MiB) vs ~4 GB RSS for the
  production server at `-c 32768`.
- Model load 9.9 s (f32 materialization) + 1.6 s Q8 conversion; per-forward dequant tax
  ~7 s on A76 (single-threaded scalar dequant), so short requests are dominated by it
  (2 tokens: 7.2 s).
- Deployment left in place for future experiments: `/root/embeddings-fast/` (binary +
  `data/`) and `/root/models/qwen3-embedding-0.6b/`. No stray processes left; the
  production stack was not started or modified.

### 9.4 Verdict

The Burn/flex implementation is numerically faithful, aarch64-portable (NEON, no JIT) and
uses ~5× less RAM, but it is 2.3× slower than the production int8 `ik_llama.cpp` path and
misses the 50 tok/s target on this board. Production stays on `ik_llama.cpp`. Closing the
gap would require int8 GEMM, which flex lacks; the only candidate is the `burn-cpu`
(CubeCL/LLVM) backend, whose CPU quantized matmul is unverified and which cannot be
cross-compiled (host-arch LLVM bundle), so it would need a native build and JIT compile on
the board.

## 10. NPU offload (`burn-rocket`, 2026-10-05)

Goal: move the model's projection matmuls to the RK3588 NPU through the mainline `rocket`
driver, freeing CPU time for other board services (the ≥50 tok/s wall target is relaxed
for this path). Stack: `librocketnpu` (gregordinary/rocket-userspace) wrapped by a new
`burn-rocket` crate; the Burn model calls it through a `Proj::Npu` variant
(`--npu`, pack-and-drop: no CPU-side projection weights).

### 10.1 M0 — baseline measurements (600 MHz, resident weights, M=3636)

The stock `rocket` module boots the NPU at 200 MHz; the patched
`/root/npu-poc/rocket-patched-600/rocket-npu600.ko` (`rmmod` + `insmod
rocket_npu_clk_hz=600000000`) raises it to 600 MHz under load (verified via
`scmi_clk_npu`; reboot reverts). Measured with `matmul_stream_vs_prepacked_rocket`
(20 iterations, T=5; ms/call includes host A-pack and C de-tile):

| shape | KACC=0 prepacked | KACC=1 prepacked | KACC=1 stream |
|---|---|---|---|
| q [3636,1024]x[3072,1024]^T | 81.1 ms (282 GF/s) | **69.1 ms (331 GF/s)** | 70.3 |
| k [3636,1024]x[1024,1024]^T | 44.8 (170) | **32.6 (234)** | 33.2 |
| o [3636,2048]x[1024,2048]^T | 76.7 (199) | **61.2 (249)** | 64.2 |
| down [3636,3072]x[1024,3072]^T | 100.0 (229) | **93.6 (244)** | 94.8 |
| q|k|v fused [3636,1024]x[5120,1024]^T | 125.1 (305) | **99.2 (384)** | 105.5 |
| gate|up fused [3636,1024]x[6144,1024]^T | 146.8 (312) | **119.8 (382)** | 133.6 |

- On-NPU K-accumulation (`ROCKET_KACC=1`) is always faster (up to 40% on k), and resident
  weights win by ~2-12% (most on the fused shapes). Fusing weights along N (q|k|v,
  gate|up) is the most efficient form (382-384 GF/s vs 234-331 for the small-N singles).
- All runs verify max_abs=0 vs a CPU reference. CPU cost is ~60-80 ms user per call
  (host pack/de-tile across the library's worker threads).
- Full-model projection estimate from these numbers: 4 calls per layer (qkv-fused, o,
  gateup-fused, down) ≈ 374 ms → **~10.5 s for all 28 layers** at 600 MHz, vs ~70 s of
  CPU GEMM at 4 threads.
- For reference, the one-shot `_mt` entry is slower (108-142 ms for q) and does not use
  resident weights.

### 10.2 M1 — integration

`burn-rocket` is an FFI wrapper over `librocketnpu` (RocketCtx/RocketWeight/
RocketStream/RocketFaCtx; `build.rs` links the static archive, `ROCKETNPU_DIR` overrides
the vendored copy). `src/npu.rs` holds `NpuModel` (one context + 28x7 resident weights)
and `NpuAttention` (persistent attention context + per-length scratch). The model's
projections became `Proj::Cpu(Linear) | Proj::Npu(NpuRef)` with `#[module(skip)]`, loaded
manually so the store never materializes CPU copies (pack-and-drop).

Loading (`--npu`): 114 tensors through `load_from` (norms + embedding table), then the 196
projection tensors are read from the same safetensors store, converted to f16 and packed
into resident NPU buffers (**0.82 GiB**, HF `[out,in]` = `[N,K]`, no transpose). The
embedding table is converted to f16 and `malloc_trim` releases the f32 pages: **298 MiB
CPU-resident**. The resident weights are packed for the `M >= 256` tiling (M-independent
there); requests with fewer rows are padded up to 256 so one pack serves every length.

Board numbers (3,633 tokens, cores 4-7, 4 threads, Q8 CPU baseline vs `--npu`):

| mode | wall | speed | CPU-seconds |
|---|---|---|---|
| CPU-only (`--quant q8`) | 99.9 s | 36.4 tok/s | 677 |
| `--npu --npu-attn cpu` | 78.7 s | 46.2 tok/s | 488 (-28%) |
| `--npu` (NPU attention, default) | 79.5 s | 45.7 tok/s | **344 (-49%)** |

Instrumented breakdown of the 81.4 s run: attention stage 64.8 s, MLP 16.0 s, norms 0.6 s;
NPU matmuls 11.1 s, f32<->f16 conversions 9.7 s — i.e. after the projections moved to the
NPU, the flex attention kernel was 80% of the forward.

### 10.3 Attention on the NPU (`--npu-attn npu`, default)

`rocket_flash_attn_fp16_ctx` computes masked GQA attention (native 16/8 heads, additive
causal mask, per-head QK/PV on the NPU; the score matrix is brought host-side for the mask
+ softmax, which is the library's design). `NpuAttention` keeps the fa context and the
`[n][n]` causal mask plus head-major f16 scratch cached per sequence length; the f32->f16
gather (Q `[n_head][n][d]`, K `[n_kv][n][d]`, V transposed `[n_kv][d][n]`) and the output
scatter back to `[1, s, h*d]` run on rayon.

- Numerics: cosine 0.999381 vs the production Q8 reference (0.999380 with CPU attention);
  0.999997 vs the CPU-attention NPU run.
- Glue optimization: `-C target-feature=+fp16` (RK3588 A76/A55 have ARMv8.2-FP16, so
  `half` uses FCVT instead of the software conversion), rayon-parallel f32<->f16
  conversion, `into_data` instead of `to_data` (one less copy) and rayon-parallel
  head-major gather/scatter. Conversion wall time halved (9.2 -> 4.5 s) and the two
  attention modes converged (90.1 -> 81.3 s for NPU attention).
- Forced tiled attention (`ROCKET_FA_TILE_KV=2048 ROCKET_FA_TILE_MIN_KV=1024`) is worse
  at 3.6k tokens (96.9 s) — the default materialized path stays.
- Fused weights: q|k|v and gate|up are packed as one resident weight each
  (`rocket_weights_pack_seg`, concatenated along N) so the layer runs 4 matmuls instead
  of 7 — one A-pack and one conversion of the shared input per group. 196 tensors become
  112 resident weights; the per-run call count drops 224 -> 140 and the wall ~2%
  (81.3 -> 79.5 s with NPU attention, 80.1 -> 78.7 s with CPU attention).

Trade-off: at 2-7k tokens the host score round-trip + softmax costs about as much as
flex's fused flash kernel; with the glue optimized the two modes are within ~1 s of each
other on wall, and NPU attention frees ~130 CPU-seconds more (-48% vs CPU-only overall).
Per the CPU-relief goal it is the default; `--npu-attn cpu` is the alternative.

Operational notes: the 600 MHz patched module stays loaded on the board (contained;
reboot restores the stock 200 MHz in-tree module). The NPU deployment needs
`--features npu` (aarch64) and `ROCKETNPU_DIR` pointing at `librocketnpu.a`.

## 11. NPU ops as a Burn backend extension + CPU-loading fix (board re-verified)

### 11.1 Refactor

The NPU runtime moved out of the app into a separate `burn-rocket` crate as a Burn *backend
extension* (`#[backend_extension(Flex)]`, the out-of-tree op hook of
0.22.0-pre.4): the `RocketOps` trait declares `rocket_pack/pack2/pack3` (weights ->
resident handles), `rocket_matmul` and `rocket_attention`; the safe wrappers `pack`,
`pack2`, `pack3`, `matmul`, `attention` take ordinary `Tensor`s and do the
`into_dispatch`/`from_dispatch` plumbing, so `src/qwen3_embedding/model.rs` calls
`burn_rocket::matmul` / `burn_rocket::attention` directly and `src/npu.rs` is gone.
Handles are plain `u64`-backed `WeightId`s (no `ExtensionType` plumbing needed; an op
without a tensor input cannot select a backend, so there is no `release`). One global
engine (NPU context + resident weights + attention scratch) sits behind a mutex;
`burn_rocket::init(threads)` must be called once. Flex's catalog cfg is read in the
*consuming* crate, hence the crate's `flex` feature (`npu = ["flex"]` links
`librocketnpu`); `build.rs` emits the link flags only for `npu` builds.

Fixed at the same time: since the NPU commit the projection fields were
`#[module(skip)]` in *all* builds, so `load_from` never loaded them and the q8 mapper
never quantized them — every CPU path had been running on uninitialized weights
(cosine 0.027) since commit 4229434. `Proj` now has a transparent `Module` impl and
the skip is `#[cfg_attr(npu, module(skip))]`; the NPU build (where the fields must
stay skipped to keep pack-and-drop) loads the projections explicitly for its CPU
modes (`load_cpu_projections`, including the PyTorch `[out,in]` -> Burn `[in,out]`
transpose). Dev-host cosines after the fix: f32 0.999375, q8 0.999280 (82 tokens), q8
resident 776 MiB.

### 11.2 Board re-validation (same commands, 600 MHz rocket module)

3,633 tokens, cores 4-7, 4 threads, 1 warmup + 1 measured run (`time`):

| mode | wall | CPU-seconds | pre-refactor log |
|---|---|---|---|
| CPU-only | 100.1 s | 622 | 99.9 s / 677 |
| `--npu --npu-attn cpu` | 86.9 s | 409 | 78.7 s / 488 |
| `--npu` (NPU attention, default) | 80.8 s | 302 | 79.5 s / 344 |

CPU relief is preserved (`--npu` is -51% CPU-seconds vs CPU-only); the walls are within
board noise of the pre-refactor numbers. The NPU-attention run: 280 calls, npu 128.8 s,
convert 10.7 s for 2 forwards. (The pre-refactor CPU-only row ran on uninitialized
projections; wall time is unaffected because GEMM cost does not depend on the values.)

Numerics (82 tokens, `data/one_64.txt` vs the production Q8 reference
`data/ref1_64.json`): `--npu` 0.999365, `--quant q8` 0.999280 (771 MiB resident on the
board), f32 0.999375 — the extension-ops path reproduces the old NPU numbers exactly.
Server smoke test (`serve --npu`, `/v1/embeddings`) passes.

## 12. Serving robustness round (deployment findings; 2026-10-05)

The OpenViking deployment agent hit four failure modes in `serve`; this round fixed the
robustness ones (attention memory work deferred):

1. **Panic -> poison -> wedge.** A forward panic (e.g. NPU `ROCKET_CREATE_BO` ENOMEM) unwound
   while the server's `Inner` mutex was held; every later request then panicked on
   `lock().unwrap()`. `burn-rocket`'s engine mutex already recovered from poisoning
   (`PoisonError::into_inner`); the server now recovers too, and the blocking compute is
   wrapped in `catch_unwind`, so a panic becomes a 500 (`server_error`) for that request
   only.
2. **Oversized M.** `matmul` is split into fixed row chunks (`ROCKET_MATMUL_CHUNK_M`, default
   8192, 0 disables) so the ~200 MB of input BOs a 30k-row activation needed is bounded per
   call. Rows are independent, so results are unchanged (bit-identical, below).
3. **tokio starvation.** Immutable settings (`model_name`, caps, attention options) moved out
   of the mutex; `/v1/models` and request routing never take the lock, and the only
   acquisition happens inside `spawn_blocking`, so a long forward cannot starve `/health`.
4. **Raw-token inputs** bypassed `--max-tokens`; they now 400 when over the cap. Text inputs
   keep the existing truncation.

NPU failures now panic with a structured `OpFailure { error, m, k, n }` payload (`lib.rs`)
that the server maps: `ROCKET_E_NOMEM` -> 503 ("retry"), `E_SHAPE`/`E_TILING` -> 500, and
device/unsupported errors log + `exit(1)` so the container restarts a clean engine.

### 12.1 Dev-host verification (`--backend flex --quant q8`)

- `cargo test`: poison recovery, panic-to-500 mapping, handler-error passthrough.
- `/health` + `/v1/models` answered in 0.2-0.4 ms while a 2,919-token embed ran (15.2 s).
- `{"input":[999999999]}` panics in the flex gather -> 500
  (`index 999999999 out of bounds...`); the next request returned 200 (poison recovery
  end-to-end).
- Over-cap raw tokens -> 400, empty input -> 400, bad `encoding_format` -> 400.

### 12.2 Board verification (robustness only; production container untouched)

Second instance on port 8393 (`--npu --npu-attn cpu --max-tokens 8192`, pinned to the A55s),
run while the OpenViking stack was live:

- invalid token id -> 500, next request 200, server alive.
- 9,000 raw tokens -> 400 (over the 8192 cap).
- 3,646-token text: 218.8 s wall, `/health` + `/v1/models` at 0.7-4 ms throughout.
- two concurrent embeds serialize on the model lock; both 200.
- chunked matmul, 1,300-token input: `ROCKET_MATMUL_CHUNK_M=0` vs `=1024` -> cosine
  1.00000000, max abs diff 0.0 (timing within noise).
- test binary left at `/root/embeddings-fast/embeddings-fast-robust`; the test server was
  stopped and the production container stayed healthy.

Not in this round: the `--npu-attn npu` long-context OOM (n² mask + host score matrices;
the library's tiled escape hatch is the candidate) and any performance measurement.

## 13. Vulkan (wgpu) GPU probe — negative result (2026-10-06)

Question: the board's Mali-G610 is idle, its stack is finally usable (Mesa 26.1.6 panvk
reports Vulkan 1.4.354, `bufferDeviceAddress`, `vulkanMemoryModel`, `shaderFloat16` +
16-bit storage, `shaderInt8` + integer dot product, 4 GB storage-buffer range, an 11.6 GiB
heap with 6.1 GiB budget) and CubeCL's `wgpu` device requirements are all met. Could a
Burn wgpu backend + NPU (eventually a hybrid `--attn wgpu`) beat flex+NPU, whose
attention stage is 54-65 s of the ~81 s wall at 3,633 tokens?

Instrument: `src/bin/wgpu_probe.rs` (device init, step-by-step smoke ops, GEMM at the
model's shapes, causal f16 attention, Flex-vs-GPU cosine check), behind the
`gpu`/`gpu-spirv`/`gpu-wgsl`/`gpu-autotune` features. Build note: the host cross toolchain
links against glibc 2.44 but the board has 2.41 (the wgpu tree pulls libm symbols
versioned 2.43/2.44), so probe builds use
`cargo zigbuild --target aarch64-unknown-linux-gnu.2.41`. Mesa's on-disk shader cache
makes repeat runs fast; the first compile of a kernel is tens of seconds.

| test (f16 unless noted) | result |
|---|---|
| from_data / cast / add (256²) | 5-25 ms each |
| GEMM 1024³, fixed strategy (no autotune) | 3.8 GF/s |
| GEMM 2048³, fixed strategy | 3.8 GF/s (4.56 s/call, survives) |
| GEMM 1024³, autotuned | 79 GF/s (after ~80 s tuning) |
| attention 256, autotuned | 4.3 GF/s |
| attention 1024, autotuned | 34 GF/s (0.253 s/call) |
| real qkv GEMM shape (3636x5120x1024), both modes | panthor job timeout -> device lost |

Failure modes:
- **SPIR-V path** (`gpu-spirv` = burn's `vulkan` feature): SIGSEGV inside
  `libvulkan_panfrost.so` at the first shader/pipeline compile (gdb backtrace: frames in
  libvulkan_panfrost; the null-dispatch class the Armbian LiteRT write-up hit). Only the
  WGSL path (`gpu-wgsl`) runs.
- **panthor job timeout**: one dispatch of the real qkv GEMM (38.6 GFLOP at 3.8 GF/s)
  trips the driver watchdog and loses the device. The limit here is >4.6 s (2048³
  survived) and <~10 s — the blog's "1 s" figure is not what this kernel enforces.
- **Autotuner OOM**: `wgpu error: Out of Memory` at seq 512/1024 despite a 6.1 GiB
  budget (reproducible, once per fresh process; the fallback candidate's score-matrix
  allocations are the likely trigger).
- Tuning costs 20-80 s per new shape, while the *fixed* (non-autotuned) kernels are 20x
  slower than the tuned ones — both extremes are unattractive.

Verdict: **no profit today**. The gate was >=200-300 GF/s on attention shapes to beat the
current 54-65 s attention; the best measured is 34 GF/s and the model's real shape kills
the device. Even an optimistic extrapolation leaves the wall at today's level, with panvk
instability, a watchdog that forbids the long dispatches 3.6k+ tokens need, and no path to
30k context. The NPU stays the only practical accelerator; this lever reopens if CubeCL's
panvk support, Mesa's compiler, or the panthor watchdog improve.

## 14. Intent model (`ov_intent_analysis_sft` = Qwen3.5-0.8B) + Ollama API (2026-10-06)

Goal: serve `guoxuter/ov_intent_analysis_sft:v7_q8` (OpenViking's recommended local
query planner) from the board, with the RK3588 NPU offloading the projections. The
Ollama tag is Qwen3.5-0.8B: the HF safetensors are a **hybrid decoder**, not a sibling
of Qwen3-Embedding.

### 14.1 Architecture and implementation

`text_config` of `guoxuter/ov_intent_analysis_sft` (f32 text weights, 3.21 GB; the
`model.visual.*` bf16 vision tower is skipped):

- 24 decoder layers, `layer_types` = 18 × `linear_attention` + 6 × `full_attention`
  (indices 3, 7, ...): **Gated DeltaNet** (16 heads × 128, conv1d k=4, `A_log`/`dt_bias`
  gates, gated RMSNorm, all delta math f32) alternating with **gated full attention**
  (8 q / 2 kv heads, head_dim 256, partial RoPE 64 dims, `q_proj` emits query+gate,
  `sigmoid(gate)` before `o_proj`), hidden 1024, intermediate 3584, vocab 248320,
  embeddings tied, zero-centered RMSNorm (`1 + w`).

Port: `src/qwen35_intent/model.rs` (chunked gated delta rule for prefill, recurrent form for
decode, KV/conv/recurrent caches, explicit single-query attention for decode),
`src/qwen35_intent/loader.rs` (safetensors loader + NPU pack-and-drop), `src/qwen35_intent/ollama.rs`
(Ollama-compatible `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`), wired into
the binary as `intent gen` and `intent serve-ollama`. The chat template matches the model's
Ollama template exactly (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`,
stops `<|im_end|>`/`<|endoftext|>`).

### 14.2 Numerics — exact match against HF transformers 5.19

Reference: `Qwen3_5ForConditionalGeneration` (text path), greedy decoding, on the dev
host. Two prompts: a 10-token chat prompt and a 166-token rendered v7 planner prompt.

| Check | Result |
|---|---|
| per-layer hidden states, last position (25 entries) | cosine 1.000000, max abs diff 0.0 |
| greedy ids, chat prompt (9 tokens, stops at EOS) | identical to HF |
| greedy ids, v7 prompt (32 tokens) | 32/32 identical (`{"queries": [{"query": "Guoxuter …` |
| first-step logits (top-8) | identical to within 1e-4 |

The port is bit-faithful in f32 on flex (the debug bisect needed two fixes: burn's
`triu_mask(offset)` has numpy `k = offset - 1` semantics, and the decode path must
write the updated recurrent state back into the cache).

### 14.3 Board measurements (rock-5b-plus, 166-token v7 prompt, 32 greedy tokens)

`taskset -c 4-7`; the 600 MHz patched `rocket` module was loaded. The board was NOT
idle (`tstor-scan`, OpenViking and the embeddings service running; `embeddings-fast`
uses the NPU), so treat the absolute numbers as an upper bound and re-run on an idle
board for a production decision.

| Mode | Prefill | Decode | Total | Decode speed | Anon RSS |
|---|---|---|---|---|---|
| CPU f32 (no `--npu`) | 7.91 s | 82.05 s | 89.96 s | 0.4 tok/s | ~2.4 GB |
| `--npu --npu-threads 3` (f16 CPU copies for decode) | 2.94 s | 8.71 s | **11.65 s** | 3.7 tok/s | 2.06 GB |
| `--npu --embed-f16` | 2.84 s | 6.45 s | 9.29 s | **5.0 tok/s** | 1.57 GB |
| `--npu --pure-npu` (no CPU copies; NPU decode) | 1.39 s | — | — | 1.5 tok/s | 1.29 GB |

- NPU vs CPU-only: **7.7× wall** (11.65 s vs 89.96 s); projections run on the NPU at
  prefill, decode uses f16 CPU copies because the resident-weight matmul pads every
  request to `M >= 256` (single-token NPU matmuls are 2.5× slower than the CPU f16 path).
- The f16 token-embedding table (`--embed-f16`) is faster and ~0.5 GB smaller but its
  f16 LM head flips near-ties: the greedy stream diverges from the f32 reference after
  ~10 tokens (both outputs are valid JSON; the f32 default keeps exact parity).
- Resident NPU memory: 96 packed weights (497 M params f16 ≈ 1 GB logical) show up as
  ~1.2 GB of shared/BO memory; a `serve-ollama --npu` process holds ~2.06 GB anon +
  ~1.2 GB shared while serving.
- Server smoke test on the board (port 11434): `/api/chat` returns the same
  `"Hello! How can I assist you today?"`, `/health` answers in ~1 ms while the model is
  loaded, `prompt_eval 1.49 s`, `eval 2.59 s` for 9 tokens.

### 14.4 OpenViking wiring

```json
{
  "query_planner": {
    "provider": "litellm",
    "model": "ollama/guoxuter/ov_intent_analysis_sft:v7_q8",
    "api_base": "http://<board-host>:11434",
    "temperature": 0.0,
    "timeout": 120,
    "extra_request_body": { "think": false }
  },
  "retrieval": { "recall_intent_timeout_s": 120 }
}
```

The model string must stay `ollama/guoxuter/ov_intent_analysis_sft:v7_q8` so OpenViking
picks its bundled v7 prompt. Two integration facts the deployment surfaced
(2026-10-06):

- litellm's Ollama client posts its JSON bodies as `application/octet-stream` and takes
  the `ollama/…` route through `/api/generate`; the server must accept any request
  `Content-Type` (like Ollama itself — the fix is in `ollama.rs`, commit `0bc0774`).
- Decode runs at ~3.3 tok/s, and the model emits ~185 tokens for the bundled v7
  prompt, so a planner call is ~55–60 s. Timeouts must cover that (the first wiring
  used 60 s and one slow decode away from tripping); measured end to end through
  OpenViking on the board: a session-listed search 64 s, context assembly 84 s.

Keep `shutdown`/supervision in mind: the service holds ~2.6 GB anon + ~1.7 GB NPU BOs
after a 1.7k-token prompt (the original ~3.3 GB steady figure was the bare 32-token
measurement), so give the container ≥6 GB.

Deployment status: this wiring ran live 2026-10-06 → reverted 2026-10-07 (the planner
is back on OpenCode Zen and the model was removed from the board); the recipe above
stands if it is ever re-enabled.

### 14.5 Open items

- Re-run the A/B on an idle board (NPU contention with `embeddings-fast`; other
  services were active during these runs).
- NPU decode without the `M >= 256` padding: pack a second, small-`M` resident weight
  per layer (the library documents a `-2` re-pack fallback for small `M`) — probe first.
- The server is one-at-a-time (a global model lock); quantify OpenViking's concurrent
  intent calls before a production switch.
