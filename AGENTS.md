# AGENTS.md — burn-rocket

`burn-rocket` is the main crate: RK3588 NPU offload for Burn models — an FFI wrapper
over `librocketnpu` plus the `RocketOps` Burn backend extension (projections and
attention on the mainline `rocket` driver). The Qwen3-Embedding-0.6B inference app is
an example of it, at `examples/rocket-inference/`. Read `README.md` (library) and
`examples/rocket-inference/README.md` (app) first; measurements live in
`examples/rocket-inference/docs/experiment-log.md`.

The repo was inverted on 2026-10-06 with no functional changes (commits `8c7eabb`
structure + `cb2f872` docs): the former root package `embeddings-fast` moved to
`examples/rocket-inference/` (renamed from `examples/qwen3-embeddings/` on 2026-10-06 when it
gained the Qwen3.5 intent-model server), the former `crates/burn-rocket` became the root
package.
The inversion is verified on the dev host: `cargo check -p burn-rocket` (plain / `flex` /
`npu`) and `-p rocket-inference` (default `cpu` and `--no-default-features`), aarch64
cross-builds of the example (`--features npu`) and of `--example probe`, plus a tokenize +
q8 embed smoke run. The git remote is still `embeddings-fast.git`. Board artifacts
deployed before that date live under `/root/embeddings-fast/`; new deploys go to
`/root/rocket-inference/`.

## Status (2026-10-06)

- **Library**: `burn-rocket` exposes the NPU as a Burn backend extension
  (`#[backend_extension(Flex)]`, `src/ext.rs`); ordinary `Tensor`s in/out via
  `burn_rocket::{init, pack, pack2, pack3, matmul, attention, stats}`. Failure payloads
  are typed (`OpFailure`), one global engine serializes the not-thread-safe FFI contexts.
  Compile-checks on x86 with `--features npu`; linking needs aarch64.
- **Example CPU path** (`flex` backend, Q8 low-RAM): numerically faithful (cosine
  0.9991-0.9995 vs the production Q8_0 reference) and ~5× lower RAM (776 MiB vs ~4 GB),
  but 2.3× slower than the production `ik_llama.cpp` int8 server on the board
  (34.2 vs 77.8 tok/s at 3,633 tokens). Production stays on `ik_llama.cpp`; flex has no
  int8 GEMM.
- **Example NPU path** (`--npu`): projections and attention on the NPU, 45 tok/s and
  **-51% CPU-seconds** vs the CPU-only path at 3,633 tokens, 298 MiB CPU-resident. The
  value is freeing CPU for other board services; wall is ~20% better than CPU-only but
  still below production.
- The extension-ops refactor surfaced a pre-existing bug: projection fields were
  `#[module(skip)]` in every build, so the store and q8 mapper had not touched them since
  the NPU commit. Fixed (log §11.1): dev-host cosine 0.9994 (f32) / 0.9993 (q8), board q8
  is 771 MiB resident again.
- 30k-token inputs work on the CPU path (fused flash attention, 52.4 tok/s at 32 dev
  threads); 30k with `--npu` is not yet measured. The OpenAI-compatible server is
  implemented and smoke-tested.
- **Serving robustness** (2026-10-05, log §12): a forward panic no longer poisons the
  server or wedges it (catch_unwind + poison recovery), `/health`/`/v1/models` never take
  the model lock, raw-token inputs are capped at `--max-tokens`, NPU failures panic with a
  typed `OpFailure` payload (NOMEM -> 503, shape/tiling -> 500, device -> log + `exit(1)`
  for a container restart), and matmuls chunk above `ROCKET_MATMUL_CHUNK_M` (default 8192,
  0 disables). Board-verified for robustness only; long-context attention and performance
  are unchanged/deferred.
- **Intent model server** (2026-10-06, log §14): `guoxuter/ov_intent_analysis_sft:v7_q8`
  is Qwen3.5-0.8B — a hybrid decoder (18 Gated DeltaNet + 6 gated full-attention
  layers), not a relative of Qwen3-Embedding. It is implemented in the same example
  (`src/intent_model.rs`, `src/intent_loader.rs`, `src/ollama.rs`; commands `gen` and
  `serve-ollama`, an Ollama-compatible API on port 11434). Greedy output is
  token-identical to HF transformers 5.19 on two prompts (32/32 on the v7 planner
  prompt; per-layer hidden cosine 1.0). Board (166-token prompt, 32 tokens, busy board):
  CPU f32 89.96 s vs `--npu` 11.65 s (prefill 2.94 s, decode 8.71 s) — decode stays on
  the CPU with f16 copies because NPU matmuls pad `M` to 256; `--embed-f16` is faster
  (9.29 s, 1.57 GB anon) but its f16 LM head changes near-ties. `--pure-npu` (no CPU
  copies) runs decode on the NPU too: 1.5 tok/s, 1.29 GB anon.
- **Vulkan GPU probe — negative** (2026-10-06, log §13): the Mali-G610 via Mesa panvk
  + Burn's wgpu backend was measured with `examples/rocket-inference/src/bin/wgpu_probe.rs`.
  Only the WGSL path works (CubeCL's SPIR-V shaders segfault panvk's compiler); the best
  attention throughput is **34 GF/s at seq 1024** (gate was >=200-300), the real qkv shape
  trips the panthor job watchdog (device lost), and the tuner OOMs at seq 512/1024.
  Fixed-strategy kernels are 3.8 GF/s. The flex+NPU configuration remains the fastest; the
  GPU lever is closed until CubeCL/panvk improve.
- Not done: int8 GEMM (the only lever that would close the speed gap; flex lacks it, and
  burn-cpu/CubeCL quantized matmul is unverified and cannot cross-compile) and board
  service deployment.

## Repo layout

| path | contents |
|---|---|
| `src/lib.rs` | FFI wrappers (`RocketCtx`/`RocketWeight`/`RocketStream`/`RocketFaCtx`, `pack_weight_seg`, `flash_attn`), driver/counter helpers, `Error`/`OpFailure` |
| `src/ffi.rs` | raw `extern "C"` declarations for `librocketnpu` |
| `src/ext.rs` | the `RocketOps` Burn backend extension, the global NPU engine (`init`, `WeightId`, `burn_rocket::stats`) and the `Tensor`-level helpers (`pack`/`matmul`/`attention`) |
| `build.rs` | links `librocketnpu.a` for `npu` builds (`ROCKETNPU_DIR`, default `vendor/rocketnpu`) |
| `examples/probe.rs` | low-level FFI probe (open device, pack, matmul, verify vs CPU) |
| `examples/rocket-inference/Cargo.toml` | workspace member `rocket-inference`: features `cpu` (default), `npu` (aarch64-gated path dep on the root crate), `gpu*`; `wgpu_probe` bin behind `gpu` |
| `examples/rocket-inference/src/main.rs` | app CLI (`bench`, `embed`, `gemm`, `serve`, `tokenize`, `gen`, `serve-ollama`), flags, model loading (f32 / `--quant q8` / `--npu` pack-and-drop), NPU packing (`load_npu_projections`), CPU projection loading for the NPU build (`load_cpu_projections`), intent model loading |
| `examples/rocket-inference/src/model.rs` | Qwen3-Embedding model: layers, RoPE, RMSNorm, attention paths, `Proj::Cpu\|Npu` (transparent `Module` wrapper, `Npu` only in the NPU build), stage timers |
| `examples/rocket-inference/src/intent_model.rs` | Qwen3.5-0.8B text model: chunked/recurrent gated delta rule, gated full attention, caches, greedy generation |
| `examples/rocket-inference/src/intent_loader.rs` | intent weight loading (`model.language_model.*`, vision skipped) + NPU pack-and-drop (fused groups, f16 CPU decode copies) |
| `examples/rocket-inference/src/ollama.rs` | Ollama-compatible intent API (`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`) |
| `examples/rocket-inference/src/server.rs` | axum OpenAI-compatible `/v1/embeddings`; immutable settings outside the model lock, panic containment, typed NPU-error mapping, `--max-tokens` enforcement |
| `examples/rocket-inference/src/bin/wgpu_probe.rs` | Vulkan/wgpu GPU probe (`gpu-*` features) |
| `examples/rocket-inference/data/` | bench/embedding fixtures (`bench_text.txt`) |
| `examples/rocket-inference/docs/experiment-log.md` | all measurements: §1-8 dev-host, §9 board A/B, §10 NPU, §11 extension-ops refactor + CPU-loading fix, §12 serving robustness, §13 Vulkan |
| `examples/rocket-inference/Dockerfile`, `docker/build.sh` | container image (aarch64, built on the board, pushed to the Forgejo registry) |
| `examples/rocket-inference-gemma/` | EmbeddingGemma 2 multimodal example (text/image/video/audio embeddings, OpenAI-compatible + multimodal server, `--quant q8`, NPU text backbone); layout, flags and deploy in its `README.md`, measurements in `docs/experiment-log.md` |
| `vendor/rocketnpu/` | **gitignored**: `librocketnpu.a`, `librocketgraph.a`, headers — copy from the board's `/root/npu-poc/rocket-userspace/build` or build `gregordinary/rocket-userspace` |
| `.cargo/config.toml` | aarch64 linker + `target-feature=+fp16` |

## EmbeddingGemma 2 example (`examples/rocket-inference-gemma`)

Full multimodal inference for `google/embeddinggemma-2` (Gemma 4 towers): text,
image, video and audio embeddings, plus an OpenAI-compatible server
(`/v1/embeddings` + native multimodal `/embed`). Verified against the HF f32
reference (transformers 5.19) — see
`examples/rocket-inference-gemma/docs/experiment-log.md`: all modalities
cosine 1.00000000 (JPEG decode via mozjpeg = libjpeg-turbo parity; the resize
is a bit-exact port of ATen's antialiased bicubic uint8 kernel). `--quant q8`
keeps Q8-resident projections (1216 MiB vs 2845 MiB resident) at 0.9996-0.9999
cosine; f16 is numerically broken (rejected). Dev-host throughput: 2587-token
text 8.0 s (~322 tok/s), image 0.70 s, 5 s audio 0.34 s.

Build: `cargo build --release -p rocket-inference-gemma --no-default-features`
(flex). Model `/mnt/hub/models/embeddinggemma-2`, reference venv
`/mnt/hub/venvs/emb2` (transformers 5.19 + sentence-transformers 6.1,
torch/torchvision CPU wheels). Board deploy at `/root/rocket-inference-gemma/`
(aarch64 `--features npu` build; model at `/root/models/embeddinggemma-2/`).

Generation round (2026-10-07, log §11): `gen` / `serve-chat` run
`google/gemma-4-E2B-it` (10.25 GB BF16, `/mnt/hub/models/gemma-4-E2B-it`) with a
causal Gemma 4 decoder: KV sharing (layers 15-34 reuse layer 13/14 K/V, double-
wide MLPs), PLE with the token table, proportional p-RoPE on full layers,
soft-capped tied LM head, `<|turn>` chat template. Greedy output is
**token-identical to HF 5.19** on a short question, a system+user prompt, a
multi-turn chat and a 631-token prompt; the template renders token-exactly.
Loading streams each tensor to its final dtype (f16 tables, f32 projections by
default; `--f16` = 8.9 GiB/3.5 tok/s decode, `--quant q8` = 7.3 GiB but 0.09
tok/s because flex has no int8 GEMM and `lin` dequantizes per call). f32 =
12.5 GiB/2.8 tok/s and is the parity mode; `serve-chat` is a non-streaming
OpenAI `/v1/chat/completions` (text + `image_url`/`input_audio` parts).

NPU prefill (log §11): `--npu` packs the used text projections into resident
fp16 NPU weights and runs prefill matmuls + `attention_causal_window` on the
NPU, while decode keeps the CPU f32 copies (`layers::set_prefill_mode` wraps
only `GenRoot::prefill`; NPU matmuls pad M to 256). Mask builders live in
`burn-rocket/src/masks.rs` with host unit tests (`cargo test -p burn-rocket`).
Board measurement pending an idle board (`--f16 --npu` needs ~11 GB resident;
rock-5b-plus has been busy). Not yet done: streaming, video input.

NPU round (2026-10-07, log §9): the `npu` feature packs all 218 text projections
into resident fp16 NPU weights (0.25 GiB; registered by `ParamId`, `lin()`
routes them to `burn_rocket::matmul`) and runs attention on the NPU through the
new `burn_rocket::attention_window` band-mask op. `--npu` combines with
`--quant q8` (validated deployment config). Board (4 A76 threads, 2587-token
text, busy board): q8 CPU 34.5 s / 94 s user CPU -> q8+npu 21.4 s / 63 s
(1.61x faster, 33 % less CPU); numerics vs HF f32 0.9996-0.9999 (text/image/
audio), 0.9992 (video). Not yet done: f32 `--npu` on the board, NPU offload for
the vision/audio towers, 8k-context NPU attention memory.

## Rebuild + deploy to the board

```sh
# NPU build of the example (links vendor/rocketnpu/librocketnpu.a, or ROCKETNPU_DIR=<dir>)
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu
ssh root@rock-5b-plus.lan 'mkdir -p /root/rocket-inference'
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/rocket-inference \
    root@rock-5b-plus.lan:/root/rocket-inference/
# the bench needs data/ relative to the deploy dir
scp -r examples/rocket-inference/data root@rock-5b-plus.lan:/root/rocket-inference/
# CPU-only build: drop --features npu (binary name is the same)
```

EmbeddingGemma 2 example (same pattern; the model is ~1.5 GB, copy it once):

```sh
cargo build --release -p rocket-inference-gemma --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu
ssh root@rock-5b-plus.lan 'mkdir -p /root/rocket-inference-gemma /root/models/embeddinggemma-2'
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/rocket-inference-gemma \
    root@rock-5b-plus.lan:/root/rocket-inference-gemma/
scp -r examples/rocket-inference-gemma/data root@rock-5b-plus.lan:/root/rocket-inference-gemma/
scp /mnt/hub/models/embeddinggemma-2/{model.safetensors,config.json,tokenizer.json,tokenizer.model,\
tokenizer_config.json,preprocessor_config.json,processor_config.json,config_sentence_transformers.json} \
    root@rock-5b-plus.lan:/root/models/embeddinggemma-2/
# board run (lowest-memory config; the board must be idle for meaningful A/B numbers)
taskset -c 4-7 /root/rocket-inference-gemma/rocket-inference-gemma bench \
    --model-dir /root/models/embeddinggemma-2 --quant q8 --npu --text-file data/one_long.txt --reps 1
```

The binary is self-contained (`librocketnpu` is statically linked); the model lives at
`/root/models/qwen3-embedding-0.6b/` on the board. On the board run from
`/root/rocket-inference` (the bench uses the relative `data/bench_text.txt`).

Library-only builds: `cargo check -p burn-rocket --features npu` compiles the extension
on any host (no linking); `cargo build --release -p burn-rocket --features npu --target
aarch64-unknown-linux-gnu` builds the library, and `--example probe` adds the FFI probe.

Container image: `examples/rocket-inference/docker/build.sh [git-ref]` stages the tree,
builds the aarch64 image on the board and pushes
`git.kmsign.org/royalcat/rocket-inference:<sha>`. It needs
`vendor/rocketnpu/librocketnpu.a` (or `VENDOR_SRC`) and registry credentials on the
control host; run it from anywhere in the repo.

The `bench` summary prints `stages: attention/mlp/norms` and, with `--npu`, an
`npu breakdown: calls/convert/npu/flex+overhead` line — check these before profiling.
The NPU counters live in `src/ext.rs` (`burn_rocket::stats`), the stage counters in the
example's `src/model.rs` (`stage_stats`).

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

- Deployed: `/root/embeddings-fast/` (pre-inversion dir; binary + `data/`) and
  `/root/models/qwen3-embedding-0.6b/`. New deploys go to `/root/rocket-inference/`;
  the old directory is left in place. Run from the deploy dir (the bench uses the
  relative `data/bench_text.txt`); pin to the A76s with `taskset -c 4-7`.
- Production A/B (reproducible live): `/opt/llama-ik/bin/llama-server -m
  /var/lib/docker/volumes/llama-swap_models/_data/qwen3-quants/Qwen3-Embedding-0.6B-Q8_0.gguf
  --embedding --pooling last -c 32768 -b 32768 -ub 32768 -np 1 -ctk q8_0 -ctv q8_0 -t 4
  --host 127.0.0.1 --port 9293` with `LD_LIBRARY_PATH=/opt/llama-ik/lib`; `pkill -x
  llama-server` to stop (do NOT use `pkill -f` — the pattern matches the SSH wrapper's own
  command line and kills the session).
- Measured verdict, CPU path (4 threads, 3,633 tokens): production 46.7 s (77.8 tok/s) vs
  ours 106.2 s (34.2 tok/s) — **target ≥50 tok/s is not met on the board**; production is
  2.28× faster. (The NPU path reaches 45.7 tok/s at -49% CPU.) Our single-core is 35% of
  the A76 f32 peak and flex's unary/binary ops have no rayon, so scaling caps at ~3×; flex
  has no int8 GEMM. f16 does not help (36.1 tok/s).
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
- `/root/embeddings-fast/embeddings-fast-robust` is the robustness-round test binary
  (log §12.2): run as a second instance on port 8393 (production container untouched,
  `taskset -c 0-3`); poison recovery, tokio starvation, input caps and chunked matmul are
  verified there. It is not the deployed service build.
- The board was later reported busy with another workload: re-run an A/B only on an idle
  board.

## NPU offload (`--npu`, aarch64)

- `burn-rocket` (repo root) = FFI to `librocketnpu` (rocket-userspace) plus the `RocketOps`
  Burn backend extension. Build the example with `--no-default-features --features npu`;
  the crate's `npu` feature implies `flex` (required because Burn's
  `#[backend_extension(Flex)]` reads `feature = "flex"` in the consuming crate) and links
  `vendor/rocketnpu/librocketnpu.a` (override with `ROCKETNPU_DIR`, e.g.
  `/root/npu-poc/rocket-userspace/build` on the board). aarch64 only: the dep is
  target-gated in `examples/rocket-inference/Cargo.toml`.
- Call `burn_rocket::init(threads)` once, then `pack`/`pack2`/`pack3` (weights ->
  `WeightId`s), `matmul` and `attention` — the model calls these directly. All ops share
  one global engine behind a mutex (the FFI contexts are not thread-safe), so NPU calls
  serialize. `attention_window(q, k, v, h, kv, d, scale, softcap, window)` (added
  2026-10-07 for EmbeddingGemma 2) applies a bidirectional band mask
  (`|q - kv| <= window`; `window < 0` = no mask); both attention ops cache their
  `[n, n]` f16 mask per sequence length in the engine.
- `--npu` = pack-and-drop: 196 projections packed into resident fp16 NPU BOs (0.82 GiB),
  f16 embedding table, 298 MiB CPU-resident. `--npu-attn npu` (default) also offloads
  attention via `rocket_flash_attn_fp16_ctx`; `cpu` keeps flex attention (within ~1 s on
  wall, ~40% more CPU). `--npu` requires `--dtype f32` and excludes `--quant q8`.
- Resident weights are packed for the M>=256 tiling; requests with fewer rows are padded
  to 256 rows (the extra rows are ignored on readback). One pack serves all lengths.
- Failures panic with a structured `OpFailure { error, m, k, n }` payload (detail logged
  by `op_failure` before the panic). `serve` maps `ROCKET_E_NOMEM` -> 503, shape/tiling ->
  500, device/unsupported -> log + `exit(1)` for a supervisor restart. Matmuls chunk above
  `ROCKET_MATMUL_CHUNK_M` rows (env, default 8192, 0 disables) so the per-call input-BO
  scratch stays bounded; rows are independent, so chunking is bit-identical.
- Failures/logs aside, a panic in an op can no longer poison the server's model lock
  (poison recovery + `catch_unwind` in the example's `src/server.rs`, log §12).
- Board: the 600 MHz patched module (`insmod /root/npu-poc/rocket-patched-600/rocket-npu600.ko
  rocket_npu_clk_hz=600000000` after `rmmod rocket`; contained, reboot reverts) is ~3x the
  stock 200 MHz boot clock. `librocketnpu` must be the built archive from the board's
  `/root/npu-poc/rocket-userspace` (it is GPL-3.0-or-later).
- Measured (3,633 tok, 4 threads, cores 4-7): CPU-only 100.1 s / 622 CPU-s;
  `--npu --npu-attn cpu` 86.9 s / 409 CPU-s; `--npu` (NPU attention, default) 80.8 s /
  302 CPU-s — re-verified after the extension-ops refactor (log §11; the pre-refactor
  numbers were 99.9/78.7/79.5 s and 677/488/344 CPU-s). The aarch64 build sets
  `target-feature=+fp16` (hardware FCVT; RK3588 is ARMv8.2) and the NPU glue (f32<->f16,
  head-major gather/scatter) is rayon-parallel.
- q|k|v and gate|up are packed as one segmented resident weight each
  (`pack_weight_seg`, concatenated along N): 196 tensors -> 112 resident weights, 4
  matmuls per layer, one input conversion per group. The loader packs each weight straight
  from the store and drops the host copy (`pack-and-drop`).
- Projection fields are `#[cfg_attr(npu, module(skip))]` so the store never materializes
  CPU copies in the NPU build. For that build's CPU modes (`--quant q8`, no `--npu`),
  `load_cpu_projections` loads the same tensors explicitly (with the PyTorch `[out,in]` ->
  Burn `[in,out]` transpose) and quantizes them in q8 mode; the non-NPU build loads them
  through the store as usual (`Proj` implements `Module` transparently).
- Attention trade-off: `rocket_flash_attn_fp16_ctx` brings the full score matrix
  host-side for the causal mask + softmax, so at 2-7k tokens it is about as fast as
  flex's fused flash kernel (within ~1 s) while using far less CPU. `ROCKET_FA_TILE_KV`
  (tiled path, opt-in — 0 is the default) engages above `ROCKET_FA_TILE_MIN_KV` (8192) and
  is *worse* at 3.6k (96.9 s vs 79.5 s); it bounds the score scratch but not the `[n][n]`
  mask, and NPU attention OOMs the board at ~30k (deployment finding). The mask +
  head-major f16 scratch are cached per sequence length in the crate's engine
  (`ext::FaState`).
- Safetensors keys in `model.safetensors` have **no `model.` prefix**
  (`layers.0.self_attn.q_proj.weight`, `embed_tokens.weight`); the loader reads the
  projection tensors by that key after `load_from` has taken the rest.
- Reverting the board NPU clock: `rmmod rocket && modprobe rocket` (or reboot) restores
  the stock 200 MHz in-tree module; nothing in `/lib/modules` was modified.

## Vulkan GPU probe (closed, negative; log §13)

- The board's GPU is reachable only through **Mesa panvk** (armbian 26.8.3 trixie, Mesa
  26.1.6 backports; Vendor `libmali` needs the vendor kernel, rusticl exposes no device).
  `vulkaninfo` shows Vulkan 1.4.354 on `Mali-G610 MC4` with all features CubeCL needs.
- Probe: `examples/rocket-inference/src/bin/wgpu_probe.rs`, built with the `gpu-wgsl`
  (+ optional `gpu-autotune`) cargo feature. `gpu-spirv` (burn's `vulkan` feature,
  CubeCL's SPIR-V compiler) segfaults inside `libvulkan_panfrost.so` at the first shader
  compile — use `gpu-wgsl`.
- Build with **`cargo zigbuild --target aarch64-unknown-linux-gnu.2.41`**: the plain GNU
  cross toolchain links against glibc 2.44 while the board has 2.41 (the wgpu tree pulls
  libm symbols at 2.43/2.44). Then scp the binary; run it from the deploy dir
  (`/root/rocket-inference`).
- Measured: fixed-strategy GEMM 3.8 GF/s, autotuned GEMM 79 GF/s (1024³), f16 causal
  attention 34 GF/s at seq 1024 (4.3 GF/s at 256). The real qkv GEMM shape trips the
  panthor job watchdog (>4.6 s dispatches die; device lost, `dmesg` "job timeout");
  the attention tuner OOMs at seq 512/1024 despite a 6.1 GiB heap budget. Mesa's shader
  cache makes repeat runs fast, but first compiles are tens of seconds.
- Verdict: no profit vs flex+NPU (attention needs >=200-300 GF/s to move the needle).
  Do not re-open without a faster CubeCL/panvk stack; the watchdog alone forbids the
  long-context dispatches this workload needs.

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
  in the example's `src/model.rs`); the embedding table is f16; `libc::malloc_trim` after
  quantization releases the freed f32 pages (glibc otherwise retains ~1.3 GB in arenas).
  Resident ~0.78 GB anon vs ~2.4 GB for f32, cost ~2 s per forward. Load peak is still
  ~2.3 GB (the file is materialized as f32 before quantization).
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
- **Burn backend extensions** (the sanctioned out-of-tree op hook; there is no way to
  register a custom `Backend` — `DispatchTensorKind`/`DispatchDevice` are closed enums,
  so a wrapper around `Flex` is not expressible on 0.22-pre.4):
  - `#[backend_extension(Flex)]` routes through `Dispatch`; `burn/extension` must be on,
    and the catalog cfg is read in the *consuming* crate, so `burn-rocket` must keep a
    cargo feature named exactly `flex` (`npu = ["flex"]`).
  - Extension args may be owned/borrowed tensor primitives, `#[extension_type]` values or
    plain `Clone+Send+Sync+'static` args — **no `Vec`/`Option` of tensors**, which is why
    the fused packing is `pack2`/`pack3`. Plain `u64` returns pass through untouched. An
    op with no tensor argument cannot select a backend, hence no `release` op.
  - The impl exists only for `Flex`: an op call with a `cpu`/CubeCL tensor panics on the
    dispatch side ("wrong backend"), and an autodiff context panics too (no backward is
    generated) — the ops are inference-only.
  - The one global engine serializes calls (the FFI contexts are not thread-safe), so NPU
    ops from several threads queue up rather than race.
- Cross-compile: `.cargo/config.toml` sets `aarch64-linux-gnu-gcc` for
  `aarch64-unknown-linux-gnu`; artifacts land in
  `$CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/`. Build with
  `--no-default-features` (the example's default `cpu` feature pulls in cubecl-llvm,
  whose prebuilt LLVM bundle is host-arch and fails to link for aarch64; `flex` needs
  none of it).

## Future work

- **int8 GEMM** is the only lever that would close the speed gap with production (example
  CPU path). flex has none; the candidates are the `burn-cpu` (CubeCL/LLVM) backend (CPU
  quantized matmul unverified, cannot cross-compile — would need a native build + JIT on
  the board) or a custom int8 microkernel (upstream contribution).
- **Vulkan/GPU path is closed** (log §13): measured 3.8-79 GF/s with CubeCL on panvk,
  panvk compiler crashes on CubeCL SPIR-V, and the panthor job watchdog kills the
  dispatches this workload needs. Re-open only with a materially faster CubeCL/panvk
  stack.
- Long-context NPU attention: the `[n][n]` mask + host score matrices OOM at ~30k. The
  library's `ROCKET_FA_TILE_KV` bounds the score scratch but not the mask and loses on
  speed; the candidate is app-side bounded-causal query blocking (per-block `[C, q1]`
  mask, prefix keys, ~half the causal MACs). Deferred from the robustness round; board
  work required. Measure 30k tokens with `--npu` (the CPU path does 52.4 tok/s at 32 dev
  threads).
- Board service: run `serve --npu --max-tokens 30000` under a supervisor, and decide
  whether the CPU-relief mode should be the default there (it is now).
- Optional: quantize tensor-by-tensor during load to remove the ~2.3 GB load-time peak in
  `--quant q8` mode; re-pack small-M weights lazily instead of padding to 256 rows.
- Intent model: re-run the CPU/NPU A/B on an idle board (the 2026-10-06 numbers were
  taken while `tstor-scan`/OpenViking ran, and `embeddings-fast` shares the NPU); probe a
  small-`M` resident pack (`rocket_weights_pack(m < 256)` documents a `-2` re-pack
  fallback) to move single-token decode onto the NPU; decide the production
  `query_planner` wiring (memory is ~2.06 GB anon + ~1.2 GB NPU BOs with f32 table).

## Verification commands

```sh
B=$CARGO_TARGET_DIR/release/rocket-inference
# library compile checks (npu needs aarch64 only for linking)
cargo check -p burn-rocket --features npu
# single-core speed gate (blocked path is the single-core record); run from
# examples/rocket-inference
taskset -c 2 cargo run --release -- bench --backend flex --dtype f32 --tokens 3633 --reps 2 --attn blocked --chunk 256 --key-block 256
# multi-threaded / long-input (fused default)
cargo run --release -- bench --backend flex --dtype f32 --text-file /tmp/opencode/long30k.txt --tokens 30000 --reps 0
# embedding + cosine against a reference JSON
cargo run --release -- embed --backend flex --dtype f32 --quant q8 --text-file /tmp/opencode/one_64.txt --out /tmp/opencode/our.json
# server smoke test
cargo run --release -- serve --backend flex --dtype f32 --quant q8 --port 8383 &
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"hi"}'
# robustness: a panicking forward (invalid token id) must 500 and leave the server alive
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":[999999999]}'
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"still alive"}'
```

On the board, chunked-matmul numerics (forced chunks vs disabled must match bit-for-bit):

```sh
cd /root/rocket-inference
ROCKET_MATMUL_CHUNK_M=0 ./rocket-inference embed --backend flex --dtype f32 --npu \
  --npu-attn cpu --text-file data/one_3633.txt --tokens 1300 --out /tmp/e0.json
ROCKET_MATMUL_CHUNK_M=1024 ./rocket-inference embed --backend flex --dtype f32 --npu \
  --npu-attn cpu --text-file data/one_3633.txt --tokens 1300 --out /tmp/e1.json
python3 -c "import json,math;a=json.load(open('/tmp/e0.json'));b=json.load(open('/tmp/e1.json'));d=sum(x*y for x,y in zip(a,b));na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(x*x for x in b));print('cosine',d/(na*nb))"
```

On the board (after the cross-build + scp above; the NPU-enabled binary):

```sh
ssh root@rock-5b-plus.lan
cd /root/rocket-inference
# speed / CPU-relief A/B (1 warmup + 1 measured run; watch the stages + npu breakdown
# lines and `time`). Only meaningful on an idle board.
taskset -c 4-7 ./rocket-inference bench --backend flex --dtype f32 --npu --tokens 3633 --reps 1
taskset -c 4-7 ./rocket-inference bench --backend flex --dtype f32 --npu --npu-attn cpu --tokens 3633 --reps 1
taskset -c 4-7 ./rocket-inference bench --backend flex --dtype f32 --tokens 3633 --reps 1
# numerics vs the production reference (copy the JSON back and compare cosines)
./rocket-inference embed --backend flex --dtype f32 --npu --text-file data/one_64.txt --out /tmp/npu_64.json
# CPU modes of the same NPU binary (projections are loaded explicitly there)
./rocket-inference embed --backend flex --dtype f32 --quant q8 --text-file data/one_64.txt --out /tmp/q8_64.json
./rocket-inference embed --backend flex --dtype f32 --text-file data/one_64.txt --out /tmp/f32_64.json
```

Vulkan/wgpu GPU probe (separate binary; built with
`cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.41 -p rocket-inference --no-default-features --features gpu-wgsl,gpu-autotune --bin wgpu_probe`;
see "Vulkan GPU probe" in this file for the verdict):

```sh
# attention throughput at the model's head layout (autotune runs once, ~20-80 s)
./wgpu_probe --skip-gemm --skip-check --attn-dtype f16 --seq 1024 --reps 2
# GEMM throughput (fixed shapes/args: --m --n --k --device igpu|cpu)
./wgpu_probe --skip-attn --skip-check --gemm-dtype f16 --m 1024 --n 1024 --k 1024
```

Reference JSONs used for the cosine checks live on the dev host in `/tmp/opencode/`
(`ref_causal.json`, `ref1_{64,512,3633}.json`, `one_{64,512,3633}.txt`) and partially on
the board in `/root/embeddings-fast/data/` (historical) and under
`examples/rocket-inference/data/`. Regenerate them with the `llama-embedding` invocation
in "Environment facts" if missing.
