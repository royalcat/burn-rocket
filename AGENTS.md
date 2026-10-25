# AGENTS.md — burn-rocket

`burn-rocket` is the main crate: RK3588 NPU offload for Burn models — the pure-Rust
`rocket-npu` driver (`crates/rocket-npu`) plus the `RocketOps` Burn backend extension
(projections and attention on the mainline `rocket` driver). No C toolchain, CMake,
kernel headers, or vendor archive are involved. The inference app is
`examples/rocket-inference/` (one binary, four model families: `qwen3`
Qwen3-Embedding-0.6B, `intent` Qwen3.5-0.8B, `gemma` EmbeddingGemma 2 and Gemma 4
E2B-it). Read `README.md` (library) and `examples/rocket-inference/README.md` (app)
first; measurements live in `examples/rocket-inference/docs/experiment-log.md`
(Qwen3 + intent) and `examples/rocket-inference/docs/experiment-log-gemma.md`
(Gemma). The root README opens with
`## AI Disclosure` and `## Inference server` (the four servers with their OpenAI/Ollama
APIs and commands, plus the production embedding deployment and its board speeds)
before "What it provides"; the speeds there mirror the experiment logs — keep both in
sync when measurements change.

The repo was inverted on 2026-10-06 with no functional changes (commits `8c7eabb`
structure + `cb2f872` docs): the former root package `embeddings-fast` moved to
`examples/rocket-inference/` (renamed from `examples/qwen3-embeddings/` on 2026-10-06 when it
gained the Qwen3.5 intent-model server), the former `crates/burn-rocket` became the root
package.
The inversion is verified on the dev host: `cargo check -p burn-rocket` (default `npu`,
plus `--no-default-features` and `flex`) and `-p rocket-inference` (default `cpu` and
`--no-default-features`), aarch64
cross-builds of the example (`--features npu`) and of `--example probe`, plus a tokenize +
q8 embed smoke run. The git remotes changed on 2026-10-07: `origin` is GitHub
`royalcat/burn-rocket.git` and is the push target; the old Gitea remote is kept as
`self-hosted` (`git.kmsign.org/royalcat/embeddings-fast.git`, stale). `main`
tracks `origin/main` now (it tracked the stale `self-hosted/main` until
2026-10-24, so older notes say to push explicitly). Board
artifacts deployed before that date live under `/root/embeddings-fast/`; new deploys
go to `/root/rocket-inference/`.

The two examples were merged on 2026-10-07 with no functional changes:
`examples/rocket-inference-gemma` moved into `examples/rocket-inference` as
`src/gemma/{embeddinggemma,gemma4}` (Gemma 4 building blocks shared in `src/gemma/`),
the Qwen families became separate top-level modules (`src/qwen3_embedding/`,
`src/qwen35_intent/` — Qwen3 and Qwen3.5 are different architectures; shared `Proj`/
`RopeCache` and infra live in `src/util/`), and the CLI is family-first:
`rocket-inference <qwen3|intent|gemma> <command>`. Dev-host checks after the merge:
cpu/flex/npu/gpu-wgsl builds, unit tests, aarch64 cross-builds (flex + npu), qwen3
cosine 0.99928 at 82 tokens, intent 10/10 tokens vs the HF oracle, EmbeddingGemma 2
text/image cosine ~1.0 vs the f32 reference, Gemma 4 gen token-identical to the
pre-merge binary.

The per-family servers were then unified (2026-10-07) into one top-level `serve`
command (`src/server/`): it detects the checkpoint family from `config.json`
(`--family` overrides), loads one model, and registers only the compatible routes
(`/v1/embeddings` + `/embed` for embedding models; `/v1/chat/completions` plus the
Ollama API for chat models; `/health` + `/v1/models` always). The old commands
`qwen3 serve`, `intent serve-ollama`, `gemma serve` and `gemma serve-chat` are gone;
the intent model gained an OpenAI chat surface and Gemma 4 gained the Ollama API.
`serve` flags: `--model-dir` (default `~/models/qwen3-embedding-0.6b`), `--family`,
`--backend`, `--port` (default 8383), `--model-name` (default: the model directory
name), `--max-tokens`, `--max-new-tokens`, `--temperature`, plus the detected
family's loading flags:

- qwen3: `--dtype`, `--quant`, `--npu`, `--npu-threads`, `--npu-attn`, `--chunk`, `--key-block`, `--attn`
- EmbeddingGemma 2: `--quant`, `--npu`, `--npu-threads`, `--npu-attn`, `--npu-int8`, `--npu-int8-group`, `--attn-chunk`, `--video-fps`, `--video-max-frames`, `--defer-towers`
- intent: `--npu`, `--npu-threads`, `--delta-chunk`, `--embed-f16`, `--pure-npu`, `--npu-decode`
- Gemma 4: `--f16`, `--quant`, `--npu`, `--npu-threads`, `--attn-chunk`

A flag that does not apply to the detected model is rejected.

`serve` defaults to NPU offload on builds that have it (aarch64 + `--features npu`);
`--no-npu` forces the flex CPU path, and `--quant q8` (the CPU low-RAM mode) keeps the
CPU path unless `--npu` is explicit. The standalone `qwen3`/`gemma`/`intent` commands
keep `--npu` opt-in.

Cleanup + formatting (2026-10-08): commit `2301d3b` removed the dead code across
the Gemma modules, cfg-gated the NPU-only helpers, annotated the checkpoint-config
structs, moved the safetensors load-report checks into `util/store.rs`, and made
`GenOptions.collect_top8` opt-in (serving no longer collects per-step top-8 logits;
`--dump-logits` enables it). Commit `9849926` applied `cargo fmt --all`, so the
workspace is `cargo fmt --all --check` clean. All builds below are warning-free and
`cargo clippy -p rocket-inference --no-default-features` is clean.

## C-library parity port (2026-10-25)

`main` contains the C-library parity port (fast-forward from the `npu-parity` branch):
the RK3588 surface of `librocketnpu` is implemented in `crates/rocket-npu` with the
same planners, register programs and numerics. The per-milestone write-up is in
"Port status" below (M0-M10); the app measurements are in
`examples/rocket-inference/docs/experiment-log.md` §19-§21. Start from
`crates/rocket-npu/src/lib.rs` ("Surface covered and deliberately missing") and
`README.md` ("Missing surface") rather than re-deriving the surface.

- **Reference + gate**: the C source is pinned at commit `b43a9ae`; the read-only copy
  used for the port was `/tmp/opencode/ref-rocket-userspace` (ephemeral). Re-sync it
  and build the host oracle with `tools/c-oracle/README.md` (the oracle needs a host
  build of `librocketnpu`; `tools/c-oracle/build.sh <src> <host-build> [out]`).
  `python3 tools/c-oracle/parity.py` must print `RESULT: PASS`: byte-identical plans
  and programs for the six matmul dtypes, the PPU pool, conv (direct/depthwise/act)
  and int8 conv. The board selftests are `selftest`, `attn_selftest`, `int8_selftest`,
  `int4_selftest`, `int16_selftest`, `bf16_selftest`, `tf32_selftest`,
  `activation_selftest`, `reduce_selftest`, `ops_selftest`, `conv_selftest`,
  `encoders_selftest`, `q8_bench`, `causal_attn_bench` and `rope_bench`; the C A/B
  driver is `tools/c-oracle/ab_compare.py` (its `ab_board` must be built natively on
  the board — a cross-built C binary cannot open the device).
- **App-visible changes**: defaults and flags are unchanged. The NPU build gained the
  chained per-head attention (`ROCKET_FA_CHAIN`, default on), the tiled long-context
  path (`ROCKET_FA_TILE_KV`, opt-in), the on-NPU softmax switch
  (`ROCKET_ATTN_HOST_SOFTMAX=0`, global — do not leak it into a deployment env; the
  host fast pass is the default) and the query-chunked causal path
  (`ROCKET_CAUSAL_CHUNK`, default 1024). qwen3 `--npu` now uses the fused RoPE in the
  `qkv_fused` branch (bit-identical, 64.8 -> 58.8 s at 2777 tokens). The Q8_0-native
  int8 projection route and NPU RoPE were measured and rejected (log §20/§21); do not
  re-open them without reading those numbers.
- **Redeploy**: the live image `94f99bd` predates S1/S2, ffmpeg and this port. A
  rebuild from `main` picks them up with no flag change; the emb2 service also gains
  the M1d FA chaining (its sliding layers were 1.08x) and the qwen3 fused-RoPE fix
  applies only to qwen3.
- **Board caution**: DRM BOs are not charged to a process memory cgroup, so
  `systemd-run -p MemoryMax=...` does not contain NPU work; a large dense attention
  allocation can push the box into global OOM (during the P2a probes `deposit` was
  killed twice and auto-restarted). Run long-context gates only with >=5 GB free; a
  30k-token f32 qwen3 forward peaks at ~4.08 GiB anon and did not complete on the busy
  board even though every op fits.
- **Still out** (deliberate; see `lib.rs`): RK3576 entries and the RK3576 graph
  runner, the C ABI, the `rocket_stream_*` aliases, `_planes`/transposed variants,
  non-fp16 batch wrappers, `ROCKET_CONV_BATCH`, and resident bf16 (the C library has
  none).

## Status (2026-10-06, extended 2026-10-07/08)

- **Library**: `burn-rocket` exposes the NPU as a Burn backend extension
  (`#[backend_extension(Flex)]`, `src/ext.rs`); ordinary `Tensor`s in/out via
  `burn_rocket::{init, pack, pack2, pack3, matmul, attention, stats}`. The NPU side is
  `crates/rocket-npu` (device/uAPI, regcmd, planner, cube layout, resident matmul,
  flash attention): planner and fp16 regcmd output are byte-identical to the C library
  on 132/132 planner shapes and every emitted program checked, and board numerics are
  cosine >= 0.99999. Failure payloads are typed (`OpFailure`), one global engine
  serializes the not-thread-safe contexts. Compile-checks on x86 with `--features npu`;
  device access needs aarch64.
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
- **Request speed logs** (2026-10-08): `serve` installs a `tracing` subscriber
  (`RUST_LOG`, default `info`) and emits one `info` line per successful inference
  request: `queue_s` (model-lock wait), `compute_s`, token counts + `tok_s`, `RssAnon`;
  chat lines additionally split `prefill_s`/`prefill_tok_s` and `decode_s`/`decode_tok_s`
  from the model's own timings. Errors and `/health`/`/v1/models` are not logged. The
  timing plumbing is `AppState::compute` (`src/server/mod.rs`), the line format
  `src/server/log.rs`; dev-verified for embeddings (qwen3 q8) and chat (Gemma 4 QAT).
- **EmbeddingGemma 2 performance round** (2026-10-08, log §12): four changes,
  validated on the board (timings in log §12.5: canonical ~2x on short input,
  fused glue -12 %, int8 g32 readback-bound and 2.4x slower than fp16).
  (1) *Canonical tiling*: `init` now uses `rocket_ctx_create_ex(threads,
  ROCKET_CTX_TILING_CANONICAL)` so resident weights serve any `M >= 4` and small
  requests no longer pad every matmul to 256 rows (`ROCKET_CTX_CANONICAL=0`
  restores legacy padding; off-vs-on is bit-identical at 9 and 2587 tokens).
  (2) *Resident int8* (`--npu-int8`, gemma emb2): the library's W8A8 group-wise
  path (group 32, per-row/per-channel per-group scales, A padded to `M % 4`);
  2587-token cosine vs HF f32 0.99985 (fp16 0.9999997, the deployed q8
  0.9996467), text weights 0.11 GiB vs 0.25. A torch probe
  (`tools/w8a8_probe.py`) gates the convention: per-32-group activations cost
  almost nothing (0.99986), whole-K row scales much more (0.99964). The group-wise
  path is readback-bound — at g32 it is 2.4x slower than fp16, so int8 is a
  numerics/memory option, not a speed one; the width is configurable
  (`--npu-int8-group`), and wider groups buy speed back at a monotone numerics
  cost (g512 = fp16 speed at 0.9992, log §12.5).
  (3) *Chunked banded attention* for the 20 sliding layers
  (`--attn-chunk`, default 1024, keys `[q0-w, q1+w)`) through the new
  `attention_window_block` op + `masks::build_window_mask_block`; 2.6k full-vs-
  chunked cosine 0.9999997, 7.7k chunk 1024-vs-2048 0.9999999, 7.7k vs HF
  0.99967; memory 1068 MiB anon at 7.7k where the full path OOM-kills in 3 GB.
  (4) *Fused CPU glue* (`src/host.rs`, `burn_rocket::{gelu_mul, rms_norm,
  rms_norm_noscale, rope_apply}`): single rayon passes instead of flex's serial
  scalar chains; q8 CPU glue on-vs-off cosine 1.0 (max|d| 1.4e-7), `--npu` glue
  on/off equivalent vs HF; `ROCKET_GLUE=0/1` is the A/B switch.
- **Idle-memory round** (2026-10-09, log-gemma §14): the emb2 server's towers load
  lazily and the process returns its heap. `--defer-towers` (serve + gemma CLI)
  moves the vision/audio towers into `Emb2Model.towers: Option<Towers>` behind
  `#[module(skip)]`; a towers-only store pass loads them on the first media request
  with the same dtype/Q8 policy (eager loads use the same pass; a text-only load
  applies 413 tensors in ~1 s, was 896 in ~4 s). `util::mem::{trim_heap,
  trim_after_request}` release the allocator's free pages at serve startup and after
  every request (`ROCKET_TRIM=0` disables the latter). Dev host q8+f32 steady
  1694 -> 754 MiB deferred (1306 eager) and flat under long requests. Board
  `openviking-embed-1` (`b4634c9` + `--defer-towers`): idle 2183 -> 605 MiB anon
  (`docker stats` 3.205 -> 1.552 GiB); a 3767-token request steps to ~1.18 GiB
  (glibc arena high-water) and the next request's trim returns it to ~0.78 GiB.
  Text and media outputs are bit-identical deferred vs eager;
  `MALLOC_MMAP_THRESHOLD_=131072` was tried and rejected (no memory gain, +60 %
  wall, log-gemma §14).
- **Scratch-memory round** (2026-10-21, log-gemma §15): the 8k footprint is now
  one request's working set. Emb2's **full layers use the chunked attention path**
  (query chunks against all keys, `MaskMode::NoneBlock`; no `[s, s]` materialization),
  the engine keeps **one FA context per attention geometry** (emb2 alternates
  sliding/full head_dim+KV and used to recreate the context every layer), and the
  matmul scratch cache keys **M capacity classes** (`M_CLASSES`, ≤25 % padding) with
  per-call live plans (no padded compute), lazy `out_all`/`wt_all`, a byte budget
  (`ROCKET_SCRATCH_MB`, default 1024, 0 = unlimited) and per-request trimming
  (`burn_rocket::trim_scratch`, called from `trim_after_request`; `ROCKET_TRIM=0`
  disables). Attention scratch keeps two shape classes per geometry with the same
  request-generation trim. `ROCKET_M_BUCKET=0` disables capacity classes.
  `ROCKET_MEM_LOG=1` prints live weights / matmul BO+host / attention BO+host plus
  the top slots and worker shapes after load and per request. Board: 1.5k tok
  8.42 -> 6.9 s (+18 %), 6.1k tok 44.9 -> 39.2 s (+14 %), both cosine 1.0 vs the
  pre-round binary; 8k driver BO 1.27 GiB + host 0.56 GiB (pre-fix Rust: 2.8 GiB
  shmem + anon; C: 2.1 GiB shmem); a 3-token request after it trims to 11.5 MiB
  scratch + 514 MiB anon.
- **Vision tower round** (2026-10-24, log-gemma §18): the tower now has phase
  timers (`vision_stats`, `encode=` in `gemma embed|bench`, `encode_s` +
  `vision_attn` in serve; `ROCKET_VISION_LOG=1` prints one line per forward and,
  on NPU builds, the NPU split of that forward). `--vision-attn npu|fused|blocked`
  (emb2 and serve; `npu` is the default under `--npu` since S2 §20, `fused` the
  CPU default since §18.3 and the explicit fallback, Gemma 4 keeps `blocked`)
  and `--npu` packs the tower's linears: `ClippableLinear::pack_npu` /
  `VisionTower::pack_npu` pack the 16 layers' q|k|v, o, gate|up, down +
  `embed_vision` as **65** resident weights (§18.5 fused groups; 113 per
  projection with `ROCKET_NPU_FUSED=0`), 0.28 GiB resident fp16, pack-and-drop;
  clipped linears are skipped, so Gemma 4's tower is not wired), wired through
  `load_towers`/`load_towers_mmproj` from `--npu`/serve and packed on the first
  media request under `--defer-towers`; `ROCKET_NPU_TOWERS=0` is the A/B
  off-switch. Under `--npu` the towers' Q8 pass excludes the packed part
  (`tower_quant_group`), so the vision tower packs **f32** weights while the
  audio tower keeps Q8 (§18.5.2). Board (2394-patch image, q8+npu): tower
  23.8-24.8 -> 19.8 (P2) -> **15.2-15.5 s** (§18.5, -22 % again; ~4.2 s of it is
  routing the MLP's gelu·up through the `ROCKET_GLUE` fused kernel, ~0.5 s the
  fused groups' NPU time), MLP phase 13.0 -> 4.4 s, `user` CPU -10 % vs the
  P3-era binary, steady serve media 27.2 -> 20.6 -> **16.4 s**, deferred tower
  load 30.4 -> 24.8 s (pack 6.2 -> 4.3 s) at 908 MiB anon after load. The driver
  runs the 723 GFLOP of tower GEMMs at ~183 GF/s, so at that point the CPU
  attention kernel (~7.6 s, then 50 % of the tower) plus ~3 s of flex norm/RoPE
  glue dominated; S2 (§20) moved the kernel to the NPU (3.4 s) for a 10.6 s
  tower, making the attention phase's ~2.5 s of CPU glue the largest residual.
  Parity: 0.9999963 (f32+npu) / 0.9998836 (q8+npu, the text Q8 path's floor;
  0.9996880 before the f32 load policy) image vs HF f32, bit-identical repeats;
  the fused groups are 0.9999967 vs per-projection (driver N-tiling changes the
  fp16 accumulation order). NPU vision attention was measured next and
  **rejected** (log-gemma §18.4): 0.434 vs 0.455 s/layer at best (1.05×, gate
  1.3×), because head_dim 64 makes the driver's host softmax (68.8M scores/layer)
  dominate; that rejection stood until S1's fast host softmax (§19) and the S2
  default (`--vision-attn npu`, §20). f16/bf16 attention operands
  are slower (§18.5.3: flex casts them to f32 internally), so the kernel stays
  f32. Frame batching was measured last and **rejected** too (log-gemma §18.6):
  grouping same-geometry video frames into one `[F*P, hidden]` tower forward
  cuts the attention op 9-10 % and the calls 4×, but the MLP phase (+13.5 %) and
  the conversions (+33-56 %) eat it — net **0 to +3 %** at 4-8 frames — so the
  candidate code was reverted (video stays per-frame) and only the
  `vision_attn_bench matmul|attn-batch` probes remain. The tower is
  proportional-to-rows work; NPU vision attention was re-opened by the S1 round
  below (fast host softmax) and wired as the tower default by S2 (§20).
- **Fast host softmax round (S1, 2026-10-24, log-gemma §19):** the §18.4
  rejection blamed the host score pass, so the new
  `crates/rocket-npu/src/attn/softmax.rs` owns it: a `Softmax` per worker with
  a reused row buffer, a branch-free
  `exp` (degree-6 Chebyshev `2^f` fit with the exponent assembled from bits;
  8-lane NEON with `fcvtl`/`fcvtn` on aarch64, scalar elsewhere), three
  L1-resident row passes and f16 probabilities written directly; the mask and
  soft-cap paths keep the old semantics, and `SOFTMAX_EXACT=1` (or
  `ROCKET_SOFTMAX_EXACT=1`) selects the original scalar `f32::exp` path, which
  stays bit-identical (`Softmax::with_mode` selects per value; the bench uses
  it). `vision_attn_bench softmax [kn] [reps] [tp]` times the entry point alone.
  Board (quiet window, 600 MHz module): 2396×2400 exact 0.0672 s vs fast
  **0.0155 s** (4.33×, 1.48 GB/s, 0.37 Gexp/s; causal 3.60×), text shape
  4.86×, deviations ≤1 f16 ULP (cosine 0.999999999-1.0). Isolated, the pass is
  ~0.20 s of the 0.43 s op (47 %, not §18.4's 0.33 s estimate — the rest is
  GEMM + the 2×138 MB BO transfers), so the share of the driver call drops to
  25 % and the op from parity to **1.99× the fused CPU kernel** at p = 2394
  (0.229 vs 0.456 s/layer, median of 3 interleaved rounds; full sequence now
  beats chunking; user CPU −21 %) and **1.07×** at p = 1170 (0.082 vs 0.088;
  §18.4 had 0.73×). Op parity vs the f32 kernel is unchanged
  (0.999999844/3.19e-5 at 2394) and bit-identical across runs. Text paths
  (general mask semantics, contended run): qwen3 2777 tok causal −11 % wall /
  −26 % CPU, emb2 2594 tok window masks attn −17 %, forward −10 %; exact-vs-fast
  cosine 0.999998366 / 0.999977266, both far below the q8 floor and
  bit-reproducible with `SOFTMAX_EXACT=1`. S1 gate (**≤0.35 s/layer or parity**)
  passed; **S2 wired it** (next bullet). Do not port the C library's on-NPU
  softmax for this: the host pass is now 0.05 s/layer and the remaining op time
  is GEMM + BO traffic.
- **NPU vision attention wired (S2, 2026-10-24, log-gemma §20):**
  `VisionAttn::Npu` and `--vision-attn npu|fused|blocked` (emb2 CLI + serve; the
  flag is rejected for Gemma 4, which keeps its hardcoded `blocked` tower).
  `npu` requires `--npu` on an aarch64 `npu` build and is rejected with a clear
  message otherwise (no silent fallback); the default for EmbeddingGemma 2 is
  **npu whenever the NPU path is active** (`--npu` on that build, including
  `serve`'s default-on offload) and `fused` otherwise — `--vision-attn fused`
  is the explicit CPU fallback, `blocked` the parity reference.
  `VisionAttention::forward`'s NPU arm is `burn_rocket::attention_window` on
  `[1, patches, H*D]` (scale 1.0, `window = -1`, whole sequence, no chunking —
  S1 measured full as fastest) -> `[1, patches, H*D]` for the packed `o_proj`;
  the fused/blocked arms are unchanged. Independent of `--npu-attn` (that stays
  the text backbone's). Board (2394-patch image, same-window 3 interleaved reps,
  production service stopped/restored, safetensors model, `--defer-towers`):
  tower **13.84 -> 10.57 s** q8+npu (**-23.6 %**, kernel 7.73 -> 3.44,
  **-55 %**) and 14.07 -> 10.58 s f32+npu, wall -6.1 % / -9.2 % whole-process,
  **user CPU -23 % / -29 %** (76.4 -> 58.6 s on the image), peak RssAnon +102
  MiB in the f32 arm (the attention context; its geometry is 72 MiB BO + 44 MiB
  host in the S1 bench; hidden by the load peak in q8). Video (4 frames, 4680
  patches): kernel 7.17 -> 4.00 s
  (-44 %), tower 18.55 -> 17.21 s (-7 %) — the MLP phase's apparent +1.5 s is
  window drift, a second window reproduced the higher MLP in *both* modes
  (8.7-8.8 s fused, 9.1-10.4 s npu) with the whole window's CPU phases 12-25 %
  slower; user CPU -14 % / -16 %. `serve` steady image compute **13.54 -> 12.44 s**
  (encode 12.17 -> 11.06, -9 %; the warm fused kernel runs ~6.2 s vs the cold
  CLI's 7.7 s, so serve's tower delta is the smaller one), first request 40.26
  -> 38.58 s, RssAnon 1161 -> 1262 MiB, text request unchanged (0.263 -> 0.251
  s, bit-identical vector). Parity vs HF f32: q8+npu image **0.999884446**
  (fused 0.999883357), f32+npu 0.999989974 (0.999997076), q8 video
  0.999505592 (0.999496202), f32 video 0.999653036 (0.999649039) — within 1e-5
  of the fused arm in both directions; fused-vs-npu
  0.999986356 (image) / 0.999998268 (video), every arm's repeats bit-identical.
  Default resolution is unit-tested (`resolve_vision_attn`), and a live
  default-mode serve was bit-identical to the explicit `--vision-attn npu`.
  P0→S2 end-to-end: 2394-patch tower 27.0 → 10.6 s, steady serve image
  27.2 → 12.4 s, 4-frame video 31.1 → 17.2 s.
- **Video + container ffmpeg round** (2026-10-24, log-gemma §21): the
  EmbeddingGemma 2 video context already matched the reference processor — one
  `<|video|>` placeholder expands to `BOI + <|video|>*n + EOI` per sampled frame
  (140 soft tokens/frame), 1 fps with a uniform cap to 32 frames, all frames in
  the single shared context; ids are identical and cosine 1.0 vs HF f32 for
  video-only (530 ids), interleaved text+image+video (817 ids) and a 40 s
  overflow clip (4226 ids). `extract_frames` no longer collapses repeated
  sampled indices (ffmpeg `select` matches once; the reference keeps the
  repeats), and the sampler plus the per-frame expansion are unit-tested.
  `tools/video_parity.py` reproduces the A/B. The runtime image now installs
  `ffmpeg`/`ffprobe` (video and non-WAV audio failed in the deployed container
  before; amd64 image 809 MB). Still one video per request, no
  `add_timestamps`, uniform overflow only.
- **OpenRouter multimodal /v1/embeddings** (2026-10-22, log-gemma §16): the
  embedding endpoint takes OpenRouter `input: [{content: [...]}]` blocks on
  EmbeddingGemma 2 (`text`, `image_url`, `input_audio`, `input_video`; images
  repeat, audio/video max one each, `input_file` 400) and text-only blocks on
  qwen3 (media rejected). One block = one joint embedding; `MediaInputs.images`
  is a vec and each `<|image|>` expands with its own soft-token count;
  `dimensions` aliases `dim`; bodies cap at 32 MiB. Media values may be paths,
  data URIs or `http(s)` URLs (blocking fetch in the compute closure: 10 s,
  20 MiB, 3 redirects, `ROCKET_MEDIA_FETCH=0`; temp uploads/downloads are
  deleted by `ResolvedMedia` after the forward). Fixed a pre-existing bf16
  audio-tower panic (f32 zeros in the conv/attention pads; `cat` dtype
  mismatch). Dev host q8 + bf16 towers: joint text+image 1.0 vs `/embed`, a
  2-image block 0.999268 vs HF f32 with n=529.
- **Intent model server** (2026-10-06, log §14): `guoxuter/ov_intent_analysis_sft:v7_q8`
  is Qwen3.5-0.8B — a hybrid decoder (18 Gated DeltaNet + 6 gated full-attention
  layers), not a relative of Qwen3-Embedding. It is implemented as the `intent` family
  (`src/qwen35_intent/{model,loader}.rs`; one-shot CLI `intent gen`; served via the
  unified `serve`, OpenAI chat + Ollama API). Greedy output is
  token-identical to HF transformers 5.19 on two prompts (32/32 on the v7 planner
  prompt; per-layer hidden cosine 1.0). Board (166-token prompt, 32 tokens, busy board):
  CPU f32 89.96 s vs `--npu` 11.65 s (prefill 2.94 s, decode 8.71 s) — decode stays on
  the CPU with f16 copies because NPU matmuls pad `M` to 256; `--embed-f16` is faster
  (9.29 s, 1.57 GB anon) but its f16 LM head changes near-ties. `--pure-npu` (no CPU
  copies) runs decode on the NPU too: 1.5 tok/s, 1.29 GB anon. The OpenViking planner
  wiring ran live 2026-10-06 → reverted 2026-10-07 (planner back on OpenCode Zen, model
  removed from the board; log §14.4) — the server stays in the example for a re-enable.
- **Q8_0-native int8 projections — measured and rejected** (P2b, 2026-10-25, log §20):
  a Q8_0 GGUF's 32-value blocks are layout-identical to the driver's resident W8A8
  group-wise weight at `group = 32`, so the app could feed `I8Context` with no
  requantization and a resident weight 1.88x smaller than fp16. The probe
  (`crates/rocket-npu/src/bin/q8_bench.rs`, both arms timed like `src/ext.rs` with
  the activation conversion included, one Q8_0-stored weight per shape, f64
  reference) says it is **1.6-2.2x** slower at 82x1024x1024 and **3.6-7.0x** slower
  at the 2777-token q|k|v|o / gate|up / down shapes, with 2-11x the CPU — while
  numerics are fine (0.9999929 vs the dequantized reference at every shape, i.e.
  block-wise int8 weights cost nothing; the loss is the activation's group
  quantize). The cause is structural: `matmul_gw` reads back one int32 partial per
  K-tile and applies the group scales on the host (no NPU-side int8 K-accumulation),
  so the traffic is `M*N*(K/32)*4` bytes — 128 B per output element at K=1024, 512 B
  at K=4096, ~5.1 GB per layer at 2777 tokens, readback ≈1.1 GB/s. No app path was
  added: `--npu` keeps packing Q8_0/bf16/f16 weights to resident fp16 at load and
  `--quant q8` keeps the CPU dequant path. Even requantizing to 16x wider groups
  (not Q8_0-native, worse weight fidelity) only ties on K=1024 and wins ~1.35x on
  the K=4096 down projection alone (log §20).
- **Vulkan GPU probe — negative** (2026-10-06, log §13): the Mali-G610 via Mesa panvk
  + Burn's wgpu backend was measured with `examples/rocket-inference/src/bin/wgpu_probe.rs`.
  Only the WGSL path works (CubeCL's SPIR-V shaders segfault panvk's compiler); the best
  attention throughput is **34 GF/s at seq 1024** (gate was >=200-300), the real qkv shape
  trips the panthor job watchdog (device lost), and the tuner OOMs at seq 512/1024.
  Fixed-strategy kernels are 3.8 GF/s. The flex+NPU configuration remains the fastest; the
  GPU lever is closed until CubeCL/panvk improve.
- **CI images** (2026-10-10, buildx rewrite 2026-10-17): `.github/workflows/build-images.yml`
  builds rocket-inference for every push to `main`/`v*` tags (PRs build without
  pushing — no smoke run, no local image, login skipped) on native runners with
  `docker/build-push-action` + `docker/metadata-action` and a per-architecture
  `type=gha` buildx cache (`scope=amd64`/`arm64`), and publishes to GHCR
  (`ghcr.io/royalcat/rocket-inference`): the aarch64 (NPU) image takes
  `<sha>`/`latest`, amd64 takes `<sha>-amd64`/`latest-amd64` (tag names unchanged:
  `type=sha,format=short,prefix=` plus the `-amd64` `flavor` suffix; a `v*` tag push
  adds `<v-tag>`/`<v-tag>-amd64`). `examples/rocket-inference/Dockerfile`
  branches on Docker's `TARGETARCH`: arm64 = `--features npu` + the staged archive,
  amd64 = `RUSTFLAGS=-C target-cpu=x86-64-v3`, flex only (the amd64 image needs an
  AVX2-capable CPU). `serve` now defaults to NPU offload on NPU builds (`--no-npu`
  opts out; `--quant q8` keeps the CPU path unless `--npu` is explicit), so the image
  CMD is arch-uniform; the standalone `qwen3`/`gemma`/`intent` commands keep `--npu`
  opt-in. The runtime stage also installs Debian's `ffmpeg` 7.1.5 (`ffmpeg` +
  `ffprobe`, gated on `-version` at build time) for EmbeddingGemma 2 video and
  non-WAV audio (+426 MiB in `/usr`, 207 packages; `libavfilter`/`libavdevice`
  pull Mesa/X11/SDL; the amd64 runtime image is 809 MB). Before 2026-10-24 the
  image had no decoder, so `input_video`/`/embed` video and non-WAV audio
  requests failed in the deployed container (log-gemma §21).
- **Port status**: fp16 one-shot/streaming/resident/segmented/multi-fd/fp32-out and
  batched matmul, dense *and* group-wise int8 (one-shot plus resident per-channel /
  group-wise weights), dense *and* group-wise int4 (one-shot with the C's
  int16-saturation `kt_cap`, plus the resident `I4Context` in both modes), its planner
  and nibble-cube pack/gather, `int16_exact`, the native bf16 stream, the tf32
  one-shot (raw-f32 operands, 4-byte cube, fp32 out and f64 host K-accumulation; the
  MAC *truncates* each mantissa to 10 bits toward zero, so the parity reference is the
  C's `tf32rt` (zero the low 13 bits), not a round-to-nearest tf32; planner + emitter
  byte-equal, `tf32_selftest` is the board check), masked GQA
  attention, the **reduce + softmax** family (the PPU global average/max/min pools
  with `reduce_factor_axis`/`global_avgpool_plan`, the feature-axis ones-matmul reduce,
  the triangular-ones cumsum, row softmax/log-softmax/cross-entropy and
  `rocket_scale_rows_fp16`; M6b below), the **transformer norms** (RMSNorm with the
  fp16 square under `square_prescale_k`, the fp32-accumulated mean-square, the host
  rsqrt and one folded fp16 scale; LayerNorm with the C's stacked `[x ; x²]`
  single-reduce job and the `x·A + B` affine fold), the **vision norms** (inference
  BatchNorm from running stats, GroupNorm, InstanceNorm, L2-Normalize over the
  `[N,C,P]` channels-major layout), the **FFN family** (`geglu`, the three-matmul
  `ffn` over the C's `NT = 3` `_mt` projections, the pinned-tile `ffn_fused` and
  `mlp_fused`, the host-handoff `mlp`) and the **encoder block** (one pre-norm Whisper
  layer over two LayerNorms, `mha_self`, the EW residual adds and the fused MLP over
  `Tp = (T+3) & ~3` rows, `ROCKET_ENCODER_NOFUSE` for the C's host-handoff A/B route;
  M7 below), and the **fp16 convolution family** (the native CNA conv — direct,
  depthwise, 1D and the conv→activation fusion — with its OC / spatial / channel tiling
  and resident-BO context, plus the transposed conv's hardware deconvolution route and
  dilated-input lowering; M8a below), the **native int8 convolution** (the direct
  int8→int32 accumulator with its multicore pool and the int8-out depthwise requant)
  and the **PPU pooling + upsample** families (M8b below) are ported
  (`int8_selftest --dense|--gw-one-shot|--prepacked-dense|--prepacked-gw`,
  `int4_selftest --dense|--gw-one-shot G|--prepacked|--prepacked-gw G`,
  `int16_selftest` and `tf32_selftest` are the board checks: int32/i64-exact vs the
  CPU reference on ragged shapes, one-shot gw bit-identical to the resident gw at
  group 32-512 for both int8 and int4, tf32 inside the fp32-accumulation envelope of a
  tf32-rounded f64 reference).
  Planner and regcmd output are byte-equal to the C library over the whole parity
  matrix — all six dtypes (`fp16/int8/int8-gw/int4/tf32/bf16`) PASS for plan and emit,
  and the `conv` section (`parity.py --kind conv`) PASSes for 23 plans + 18 programs +
  6 fused-activation programs;
  fp16/bf16 cosine >= 0.99999, int8 matches its dequantized reference, int4 is
  int32-exact against the int64 CPU reference and tf32 matches a tf32-rounded f64
  reference. Board microbenchmarks: int8 512x3072x1024
  62-70 ms vs C 59-66, bf16 stream 24-27 ms vs C 22-26. Attention uses a per-worker
  head split and batches each worker's per-head QK/AV matmuls into one job
  (`ROCKET_FA_CHAIN`, default on, `ROCKET_FA_CHAIN_ELEMS` = 32M cap): bit-identical,
  2394-patch vision 1.11× and emb2 sliding 1.08×, parity on the deployed KACC-on text
  shapes, 1.2-1.4× with `ROCKET_NO_KACC=1` (M1d, 2026-10-25). The S1 fast host softmax
  cut the score pass 4.3× (log-gemma §19); EmbeddingGemma 2's vision tower runs its
  attention on the NPU by default under `--npu` (S2, §20). End-to-end: qwen3
  3,633-token `bench --npu` 59.4 s vs the C binary's 61.5 s.
  **M5a one-shot / reference entries** (2026-10-25): the C library's stateless
  attention entries are wired too — `attn::flash_attn` (`rocket_flash_attn_fp16`),
  `attn::flash_attn_mt` (the `_mt` head fan-out, bit-identical to single-fd; it pays
  the whole context setup per call, so at 64-257 keys one fd is faster (T=257: 65-69 ms
  vs 74-115 ms) and it only wins at long context: 2394 keys 0.80 vs 1.01 s, 1.27×,
  CPU 1260 vs 690 ms — the workers wait on the shared NPU), the device-free f64 oracles
  `attn::flash_attn_ref` / `attn::mha_self_ref`, and the encoder sublayer
  `attn::mha_self` (`rocket_mha_self_fp16`, `Tp`/`Tn` padding + host softmax; its
  `nthreads` fans the q/k/v projections and the head ranges out without splitting any
  matmul, so it is bit-identical to `nthreads = 1`: T=257 d=1024 150 vs 557 ms, 3.7×;
  T=82 d=512 23 vs 54 ms, 2.4×). `fully_masked_rows` marks the one padding artefact
  inherited from C: a query row the mask empties is uniform over `Kn` on the device but
  over `n_kv` in the reference, so it reads `n_kv/Kn` (causal masks never empty a row;
  a band mask over a shorter key axis can).
  Board (`attn_selftest --oneshot`/`--mha`, cores 0-3, busy board): flash cosine
  0.99999989-0.99999996 vs the ref (vision 2394/64d, T=257 GQA 8/2 causal+causal-cross+
  band, softcap 0-50, T=64), zero differing bits single-fd vs mt everywhere; mha_self
  0.99999927 (d=512, T=82) / 0.99999812 (d=1024, T=257) vs its f64 ref. Deviations
  from C: host (fast) softmax by default in both places, and the host arm's scores
  stay f32 between matmul and softmax instead of round-tripping f16 (the on-NPU arm,
  `ROCKET_ATTN_HOST_SOFTMAX=0`, is the one that matches C bit for bit — see M10 below).
  **M5b tiled long-context attention** (`ROCKET_FA_TILE_KV`, 2026-10-25): the port of
  `fa_heads_range_tiled`. `attn::tile_kv_width(kn)` resolves the knob per call exactly
  as C (`0` = off by default, width rounded up to 32 with a 32 floor, engages only while
  `kn > ROCKET_FA_TILE_MIN_KV` (default 8192) and `kbp < kn`); `Worker::prepare` sizes a
  separate `TileScratch` (`kh [wp,dk]`, `vh [dv,wp]`, `sc`/`P` `[Tp,wp]`, `ctx`, plus
  `attn::Online`: the fp32 `m`/`l`/`acc` running state and its f32 row buffer) and drops
  the `[Tp,Kn]` host matrices and BOs a materialized call at that length would hold
  (host only when bigger than one tiled score matrix, so a mixed sliding/full model
  keeps its short buffers; per-head BOs any scratch whose `n`/`k` reaches `Kn`; the
  chained group scratch once its key axis reaches `Kn`); `Worker::run`
  dispatches it before the chained and per-head paths, at any `dk`/`dv` (the original
  port limited it to `dv == dk`; **M5c** below lifts that), and
  `ROCKET_FA_CHAIN` never applies. Board (`attn_selftest --tile`, cores 0-3, busy board
  at load 10-13, 600 MHz NPU): tiled vs the f64 `flash_attn_ref` cosine **0.99999994**
  at 30,016 keys (T=512, GQA 8/2, d=64, causal; materialized 0.99999996) and 0.99999993
  at 1024 keys with a 256-wide tile; tiled vs materialized 0.99999982-0.99999992 (online
  reassociation; max_abs one f16 ULP around 0.5), and the tiled output is bit-identical
  with `ROCKET_FA_CHAIN=0` vs `1` in every run. Scratch, bo+host from `mem_stats`:
  512x30,016 132+397 -> 27+10 MiB (**14x**), 2,048x16,384 174+267 -> 54+37, 8,192²
  314+528 -> 113+145, 16,384² 549+1,048 -> 95+153 (**6.4x**), 30,000² 1.07 GiB
  total where the materialized path would need 429 MiB *per head* of score matrix
  (~14 GiB over two workers) — the 30k request that used to OOM now runs in ~1 GiB; a
  3-token request after it trims to 0.7 MiB bo / 0.1 MiB host. Flat in `Kn`: 2,048 x
  30,016 keeps the same 18.5 MiB host per worker as 2,048 x 16,384 (the BO is 47.3 vs
  27.0 MiB because a key axis that is not a multiple of the tile adds a second
  last-tile class to the two-entry scratch cache). Wall is parity to slightly faster
  and never the point: 8,192² 0.95x (median of 3), 16,384² 0.97x, 2,048x16,384
  0.95x, 512x30,016 0.88x, vision 2,400² 0.99x; one 8k window read 1.40x at load
  12, i.e. board noise is +-20 %. Forced on below its crossover it loses as expected
  (1,024 keys, 256 tile: 1.39x), which is why the default `min_kv` is 8,192. The
  caller's `[n][n]` additive mask was still O(n²); **P2a below adds the app-side causal
  block mask** (the M5b tiled path is orthogonal: that one bounds the driver's score
  matrix along the *key* axis, P2a bounds the mask along the *query* axis). The round
  also fixed a **pre-existing MLA bug**: the
  chained group path indexed the V groups with the QK stride (`Kn*dk`) while
  `GroupBufs::ensure` sized them `Kn*dv`, so a `dv != dk` shape panicked when `dv < dk`
  and read the wrong columns when `dv > dk` (the M5a binary panics the same way). The C
  `fa_heads_range` skips the batched path for MLA and now so do we (per-head path;
  verified at 3,008 keys with dk/dv 64/32 and 32/64, cosine 1.0 vs the materialized
  arm, and `mha_self` unchanged).
  **M5c tiled MLA** (2026-10-25): M5b's `dv == dk` gate is gone, so a value dim that
  differs from the head dim tiles too. The tiled path was already parameterized by both
  dims (`TileScratch`'s `kh [wp,dk]` / `vh [dv,wp]` / `ctx [Tp,dv]`, `attn::Online`'s
  `acc [Tp,dv]`, `gather_v_tile`'s `dv*kv` stride and the per-tile AV `Tp x wp x dv`), so
  the whole round is two guards — `Worker::prepare`'s `if dk == dv { tile_kv_width(kn) }
  else { 0 }` becomes `tile_kv_width(kn)`, and `Worker::run`'s `tile_w > 0 && head_dim
  == dv` becomes `tile_w > 0`. The **chained** path keeps its MLA exclusion (`head_dim
  == dv` in the group-size condition): its `GroupBufs` still carry one stride per head
  for both operands, which only holds while the value dim is the head dim (as in C's
  `fa_heads_range`). This is a Rust extension — the C `fa_heads_range_tiled` takes one
  `d` and has no MLA variant, so there is no C byte parity; the gate is
  `flash_attn_ref` plus the materialized arm. Host test
  `attn::tests::an_mla_geometry_tiles_like_one_pass` runs the same fold at 576/512,
  64/32 and 32/64 against one-pass tile widths. Board (`attn_selftest --tile`, cores
  0-3, busy board at load ~11-12, 600 MHz NPU): T=64 kv=2048 GQA 8/2 dk=576 dv=512
  causal, tile 512 — tiled cosine **0.99999994** vs the f64 ref (materialized
  0.99999996), tiled vs materialized 0.99999992/max_abs 4.9e-4, chain-A/B
  bit-identical, scratch 23.6+20.1 -> 8.6+6.3 MiB; the same shape unmasked (all 2048
  keys live) 0.99999993 both arms, 0.99999989 between them; dk/dv 64/32 and 32/64 the
  same pattern (0.99999994/0.99999993 vs the ref). 512x8,192 dk 576 dv 512 tile 2048:
  170.8+140.4 -> 58.8+45.4 MiB (2.9x) at wall 1.09x / cpu 1.15x, tiled-vs-materialized
  0.99999992; 512x16,384 keeps the *same* 58.8+45.3 MiB (flat in `Kn`) at 617.7 ms vs
  322.4 ms at 8k, chain-A/B bit-identical; a 3-token request then trims it to 1.8 MiB bo
  / 0.4 MiB host. Below the crossover the tiles still lose on wall (T=64 kv=2048 tile
  512: 1.71x causal / 1.08x unmasked — 20-40 ms is dispatch-bound), which is why
  `ROCKET_FA_TILE_MIN_KV` stays 8,192. `dv == dk` is unchanged: base `181f94f` and this
  build print identical cosines, max_abs, footprints and chain verdicts on
  1024x1024 dk 64 tile 256 (0.99999995 ref / 0.99999993 tiled / 0.99999991 between
  arms, 64.4+55.0 -> 10.8+7.3 MiB); only wall moves (+-20 % board noise).
  **P2a causal block attention** (2026-10-25): `burn_rocket::attention_causal_block` and
  `attention_causal_block_window` (`MaskMode::CausalBlock`) plus
  `masks::build_causal_mask_block` / `build_causal_window_mask_block`. A query chunk at
  absolute `q_start` sees key `kv_start + j` iff `kv_start + j <= q_start + i` (and
  `q_start + i - window < kv_start + j` with a window, the same `t - window < j <= t`
  convention as the dense causal-window mask); `n_kv_len != n_q` is the normal case and
  the additive mask is `[n_q, n_kv_len]`, cached in the geometry's `FaState` on the
  translation-invariant key `(n_q, n_kv_len, window, q_start - kv_start)` exactly like
  the band block. `FaState`'s host q/k/v/out buffers now `resize` in place instead of
  being reallocated per shape (a causal chunk run walks the key axis upward, so the old
  reallocate-and-zero touched the whole prefix for every chunk of every layer); exact
  slices go to `fill_heads`/`flash_attn`, so nothing else changed. No driver change was
  needed (`crates/rocket-npu` untouched).
  App side: `examples/rocket-inference/src/util/causal.rs` — `causal_chunk_plan` (pure;
  host tests cover ragged tails, `q_offset`, a shorter key axis and the window band) and
  `causal_attention`, used by qwen3's `npu_attn` path and Gemma 4's NPU prefill. Chunk
  width `ROCKET_CAUSAL_CHUNK` (default 1024, `0` = the dense op); keys `[0, q1)` for
  causal, `[q0 - (w - 1), q1)` for a sliding layer; below the threshold the dense op runs
  unchanged (bit-identical to the pre-P2a binary). The probe
  `examples/rocket-inference/src/bin/causal_attn_bench.rs` (chunked vs dense vs
  `attn::flash_attn_ref`, `--window`, `--kv N` for a single long-context chunk) is the
  numeric gate.
  Numerics (board, cores 0-3): **chunked == dense bit for bit for pure causal at 512
  (chunk 128/256) and 8192** (8k: 21 signed zeros only, cos 1.0); sliding-window chunks
  differ from the dense mask in a fraction of the values by ≤ 6.1e-5 (n=512 w=256:
  1,146/1,048,576 values, max|d| 3.05e-5, cos 1.0; n=1000 w=200: 5 %, cos 0.999999997;
  n=2048 w=512: cos 0.999999981 chunk 256 / 0.999999990 chunk 1024) — the band's offset
  inside the K-tile grid changes the fp16 partial grouping, the pure-causal prefix does
  not. Against the f64 oracle `flash_attn_ref` every arm is 0.999999955 (max|d| 2.44e-4),
  causal and windowed. Within-process repeats are bit-identical in every arm.
  Footprint (`mem_stats` attention BO+host): dense whole-sequence 8k = **1706 MiB** vs
  chunk-1024 **1102 MiB** / chunk-256 **871 MiB** (both arms in one process, so the dense
  classes are still cached). Per-chunk shape (the real 30k-context working set), one
  1024-row chunk against an 8k / 16k / 30k key prefix: **944 / 1154 / 898 MiB** and
  1.32 / 2.42 / 3.78 s per chunk (two reps, probe peak anon 1.0 / 1.5 / 1.6 GiB including
  the whole q/k/v/o). The scratch is *linear* in the prefix, not `[n, n]`: the chained
  head group is capped at a 32M-element score matrix (`group_size`), so a longer prefix
  shrinks the group instead of growing the scratch, and the remaining prefix-linear part
  is the per-head `[Tp, Kn]` host/BO pair (~0.9 GiB at 30k with `chunk = 1024`;
  `ROCKET_CAUSAL_CHUNK=512` is the low-memory setting). A 30k prefill's attention is
  then ~27 min of chunk ops (Σ over 30 chunks × 28 layers), which is the n²/2 work, not
  a memory cliff.
  Model level (qwen3-embedding-0.6B f32 `--npu`, `data/long30k.txt`, busy board
  load 6-30): 8k dense 302 s wall / 246 s compute (32.5 tok/s) at 2153 MiB peak anon vs
  chunk-1024 203-245 s wall / 181-203 s compute (39.5-44.3 tok/s) at 2042-2126 MiB — the
  chunked path saves the masked half of the attention work. 8k chunked-vs-dense model
  cosine 0.9999914-1.0 (one pair bit-identical, i.e. the arms agree exactly and the
  spread is *between runs*: two identical 8k runs scatter by ~5e-6 cosine while
  within-process attention repeats and the `--npu-attn cpu` fallback at 1024 are
  bit-identical to the pre-P2a binary). 30k: **dense never completes** (killed at rc=137
  under a 5 GB cgroup in ~2 min — its `[n, n]` f16 mask alone is 1.8 GiB and the driver's
  score matrix another 1.8 GiB *per worker*); the chunked f32 run gets through the
  attention (~0.9 GiB of chunk scratch) but peaks at **4080 MiB anon** and was killed by the
  board (global OOM at 40 min — the box has only 2-4 GB headroom because `deposit` holds
  4-5 GB, so 30k f32 does not fit *this* board today; the residual is the model's 30k-row
  activations, not the mask). **Operational note:** DRM BOs are not charged to the
  process's memory cgroup, so `systemd-run -p MemoryMax=…` does **not** bound the NPU-side
  allocations — a large dense attention allocation still pushes the whole box into global
  OOM, and the kernel then kills the biggest anon task (that is how `deposit` was killed
  twice during these runs). Run long-context gates only when ≥5 GB is free.
  **M6a activation + elementwise** (2026-10-25): the whole of `rocket_activation.c` is
  ported as `crates/rocket-npu/src/ops/activation.rs` (+ the program in
  `regcmd/activation.rs`). API: `ops::activation::{activation, leaky_relu, elu, selu,
  prelu, ew_mul, ew_add, ew_sub, ew_max, ew_min, clip, div}` for the device,
  `{activation_ref, leaky_relu_ref, elu_ref, selu_ref, prelu_ref, ew_*_ref, clip_ref,
  div_ref}` for the fp16-rounded exact math (glibc `erf` for GELU, so it matches the C
  bit for bit), `lut_epilogue_build` (the 1026-word table + `LutEpilogue`, the C's
  `lut_epilogue_t`), `lut_apply` (the host model of the DPU LUT chain) and
  `table_domain`. All 19 `ROCKET_ACTIVATION_*` values are there; the gated kinds
  (HARDSWISH/SILU/GELU/MISH) keep the C's two-pass `x·gate(x)` route (host multiply by
  default, `ROCKET_ACT_NPU_MUL` present = the NPU EW multiply, `ROCKET_ACT_WIDE_LUT` =
  the single-pass table), and the knobs keep the C names and are read per call
  (`ROCKET_LUT_*`, `ROCKET_ACT_*`, `ROCKET_LEAKY_R`/`_NOREPAIR`, `ROCKET_ELU_R`/
  `_NOREPAIR`). Layout/numerics: `n` pads to the C2=8 cube atom, the LUT path tiles at
  `DPU_LUT_MAXN` = 32768 elements, the EW path reshapes to `[M,32]` with `M >= 12`,
  `M % 4`, 1020 rows per tile; one submit + one fence wait per tile. The identity-matmul
  EW wiring needed two new `MatmulParams` fields (`ew_mul`, `ew_op` 0/2/3 = add-mul/
  MAX/MIN) — the existing fp16 programs are byte-identical (parity.py still 6/6).
  Host byte-parity: the `--dump-lut` output (epilogue + 1026 words for kinds 0..=18)
  and the `--dump-gen` program (1070 words) are **byte-identical** to the C
  `rocket_lut_epilogue_build` / `gen_lut_activation_fp16`. Board (cores 0-3):
  `activation_selftest` PASS for every kind at n = 1/31/4096/262144 and every EW op at
  4096/262144, and a **C-vs-Rust A/B is bit-identical** for all 19 kinds and the six
  EW ops (the C side is built natively on the board; see the verification commands).
  Device-vs-`lut_apply` residual is <= 1 output ULP for the non-gated kinds and <= 2
  for the gated ones. Two HW facts the model had to learn: the interpolated table
  value **saturates to [0, 32767]** before OUT_CVT (that is what stops tanh's LE slope
  exactly at -1 and floors a gate's negative tail at 0), and the sign-based LE/LO mux
  still glitches in a razor-thin `x≈0` band for tables that straddle 0 (tanh's default
  affine path: an isolated +8.0 spike, exactly as in C — the C gates never sample
  exactly 0; the self-test counts those as `glitch=` and leaves them out of the gate).
  `cargo test -p rocket-npu` pins the LUT words/epilogue fields, the program hash, the
  refs' identities, the knob overrides and the table domains.
  **M6b reduce + softmax** (2026-10-25): `src/rocket_reduce.c` (538 lines),
  `src/rocket_softmax.c`, `gen_pool_fp16` from `src/npu_regcmd.c` and
  `rocket_scale_rows_fp16` from `src/rocket_norm.c` are ported as
  `ops/reduce.rs`, `ops/softmax.rs`, `regcmd/pool.rs` and `ops/norm.rs` (the last one
  says in its module docs that the rest of `rocket_norm.h` — RMSNorm/LayerNorm and the
  fp16-square prescale — is M7). API: `ops::reduce::{global_avgpool_plan,
  global_avgpool, global_avgpool_ref, global_maxpool, global_minpool,
  global_maxpool_ref, global_minpool_ref, reduce_factor_axis, reduce_feature,
  reduce_feature_ref, cumsum, cumsum_ref, feature_data, recip_pair}`,
  `ops::softmax::{softmax, softmax_ref, logsoftmax, logsoftmax_ref, cross_entropy,
  cross_entropy_ref}` (CE takes `target: &[i32]` and writes `&mut [f32]`),
  `ops::norm::{scale_rows, scale_rows_ref}` and
  `regcmd::pool::{gen_pool_fp16, PoolParams, PoolMethod, ppu_recip_kernel_fp16}`.
  There is **no "no device" variant**: the device entry points always take `&Device`,
  the `*_ref` functions are the host path, and a **non-decomposable** shape falls back
  to `*_ref` *inside* the device function (that is what the C does for `fd >= 0` too).
  Decomposition rules kept exactly: `reduce_factor_axis` is 16-smooth-only and
  ascending (largest divisor first, then sorted, so the running quotient never drops
  below 4 — a PPU-written sub-4 cube is not read back by the next chained pass), the
  plan needs `H`/`W` the same factor count (`Shape` for not-16-smooth / unequal,
  `NoMem` past `REDUCE_MAX_DIM` 8192, `Unsupported` past 8192 channels), the passes
  ping-pong NC1HWC2 cubes (`feature_data`, C2=8) with exact tiling, the average
  reciprocals are `fp16(65536/k)` per axis with the geometric sqrt split for an
  asymmetric kernel (`recip_pair`), and the submit carries
  `DRM_ROCKET_JOB_PPU_DONE` when `ppu_done_supported()`. The feature reduce and cumsum
  are `matmul::one_shot_f32out` calls (N padded to 16, column 0 read; triangular ones
  weight for cumsum), so they inherit the genuine fp32 K-accumulation; the softmax
  family is host row-max + f32 subtract → `activation(Exp)` → `reduce_feature` → host
  `1/s` (or `log`) → `scale_rows` (or `ew_sub`; CE is logsumexp + the host gather), with
  the C's `s <= 0` guards and the `INT_MAX` element-count ceiling. Program parity: the
  PPU program is **byte-identical** (30 words), and so are the decomposition
  (`pool C H W`: the plan reason code plus both factor lists, 20 shapes) and
  `ppu_recip_kernel_fp16` (19 kernel extents), over the new `pool` section of
  `parity.py`; the six matmul dtypes still PASS. Board (cores 0-3, busy board
  load ~9.5): `reduce_selftest` PASS for every family — pools 7x7/14x14/56x56/49x49/
  17x13-fallback/1x1 (avg cosine 0.9999999, max/min exact), feature 4x32..512x1024
  (max_rel 2.3e-7), cumsum 4x16..128x256 all four variants (bit-exact after the fp16
  read-back), softmax/logsoftmax/CE at 4x8/64x1000/257x4096 **and 257x151936**
  (softmax cosine 0.9999999, logsoftmax abs <= 7.8e-3, CE abs <= 4.2e-5; the LM-head
  shape is 3.7 s softmax / 1.8 s CE). A C-vs-Rust A/B (`tools/c-oracle/ab_compare.py`
  with `ab_board.c` built natively on the board) is **bit-identical for every family**,
  including the average pools and softmax — both sides run the same program and the same
  host glue, so even the reciprocal quantization cancels. `cargo test -p rocket-npu`
  pins the factor/plan decisions, the cube round trip, the four cumsum references, the
  softmax/logsoftmax/CE references, `scale_rows_ref`, the pool program word list
  (including the MAX `0xFC00` pad fill) and the reciprocal convention.
- **M7 norms, vision norms, FFN and encoder** (2026-10-25): the rest of
  `rocket_norm.h` (`src/rocket_norm.c`, 241 lines), `src/rocket_normvision.c`,
  `src/rocket_ffn.c` and `src/rocket_encoder.c` are ported as
  `ops/norm.rs` (extended), `ops/normvision.rs`, `ops/ffn.rs` and `ops/encoder.rs`.
  API: `ops::norm::{square_prescale_k, rmsnorm, rmsnorm_ref, layernorm, layernorm_ref}`
  (plus the existing `scale_rows`), `ops::normvision::{batchnorm, batchnorm_ref,
  groupnorm, groupnorm_ref, instancenorm, instancenorm_ref, l2norm, l2norm_ref}`,
  `ops::ffn::{geglu, geglu_ref, ffn, ffn_ref, ffn_fused, mlp, mlp_ref, mlp_fused}`,
  `ops::encoder::{EncoderParams, encoder_block, encoder_block_ref}` — all re-exported
  flat at `ops`. The device entries take `&Device`, `*_ref` are the host fp64 forms,
  and the `fd < 0` fallbacks are the `*_ref` functions (there is no "no device"
  variant, as in M6b). `INT_MAX` element guards on every flat-EW count (the C only
  guards some of the vision entries — the port guards all of them and reports
  `Error::Shape` where C would silently under-process), and typed `Shape` errors on
  short buffers/dims before any device access.
  Structural details kept from the C: the **exact power-of-two fp16-square prescale**
  (`square_prescale_k`: threshold 223, `ceil(log2)`, applied as `2^-k` and recovered
  as `4^k`, hence bit-exact); RMSNorm = EW square → `reduce_feature(mean)` → host
  `1/sqrt(ms·4^k + eps)` → one folded fp16 `s[m][h] = r[m]·weight[h]`; LayerNorm =
  **one stacked `[x ; x²]` reduce job** (`2M` rows) → host mean/var/rsqrt (var clamped
  at 0) → the `A[m][h] = r·gamma`, `B[m][h] = beta − mean·A` fold → EW mul + EW add;
  GroupNorm = the same stacked reduce over `[N*G, (C/G)*P]` → per-**channel** A/B
  broadcast; InstanceNorm = GroupNorm with `G = C`; L2 = square → `reduce_feature(sum)`
  → `scale_rows`. FFN: `ffn` is the C's `rocket_matmul_fp16_mt(..., NT = 3)` fan-out +
  `geglu` (activation LUT + EW mul) + the down projection; `ffn_fused`/`mlp_fused`
  reproduce the C's **pinned tiling** through the new `matmul::Plan::pinned` (the
  `mm_plan_init_pin` port: `Nt(projections) = Kt(down) = 256`, no `ROCKET_MM_*`
  override, the free dims shrink to fit the CBUF), the matched-tiling gate
  (`nNt == nKt && out_slot == in_slot && nM == nM && Mt == Mt && nMt*nNt <= 64`),
  **KACC-only** for the projections (`compute_kacc_cube` with
  `KaccPolicy::Only`: a `Tiling` refusal falls back to the whole host-handoff block,
  as the C's `goto fb`) and **KACC-then-CPU** for the down/fc2 (`compute_kacc_from`,
  `compute_cpu_from` on the sentinel, i.e. `KaccPolicy::Forced`'s routing — the C's
  `mm_compute_kacc` then `mm_compute`), the bias add in f32, and the 2-pass
  GELU. (The cube residency those entries originally left out is the M11 round
  below.) `encoder_block` is the C's order exactly: LN1 over `T` rows →
  `attn::mha_self` (single fd, as the C) → EW residual → LN2 → zero-pad to
  `Tp = (T+3) & ~3` → `mlp_fused` over `Tp` (`ROCKET_ENCODER_NOFUSE` = the C's
  `GELU_GATE` + EW-mul host-handoff route) → the bf2 add over `T` → EW residual;
  `EncoderParams` carries the 16 tensors (C argument order, `None` for the C `NULL`).
  A knobs/semantics note: the C's encoder MLP path ignores `ROCKET_NO_KACC` (it calls
  the KACC primitives directly), and so does the port.
  Host tests (`cargo test -p rocket-npu`, 183 pass): RMSNorm on hand-computed rows
  (unit RMS, per-column gamma, eps, the 223 threshold and the exact prescale),
  LayerNorm (zero-mean/unit-var rows, gamma/beta, negative-variance clamp, agreement
  with RMSNorm on a symmetric row), batchnorm/groupnorm/instancenorm/l2norm on
  hand-computed small arrays (including the per-channel affine layout trap: flat
  element `i` belongs to channel `i / P`), `geglu`/`ffn`/`mlp` references (the C's
  `act_exact` default-to-SiLU quirk, the f64 `ffn` intermediate vs the **fp16-narrowed**
  `mlp` one), the pinned-tiling contract (pin alignment, `nNt == nKt`,
  `out_slot == in_slot`), and the encoder reference (deterministic, finite, equal to the
  hand-assembled composition of the family primitives, plus the shape/buffer refusals).
  Board (`ops_selftest --all`, cores 0-3, busy board; `ops_selftest` is the new binary
  with `--norm`/`--vision`/`--ffn`/`--encoder`, per-family `--*-shape` and `--io`):
  norms 4x256..257x1024 PASS (rmsnorm cos 0.99999994, max_rel(sig) 9.8e-4; layernorm
  0.99999992, 1.9e-3), vision 2x8x49/2x16x196 PASS (0.9999999, <= 2.1e-3), ffn
  4x256x1024 and 64x512x2048 PASS (geglu 0.99999993; ffn/ffn_fused/mlp/mlp_fused
  0.9999998-0.9999999 with <= 5e-3 relative at a peak of 1.2e3), encoder
  T=8 d=64 h=2 ff=128 / T=64 d=128 h=4 ff=256 PASS (0.9999992/0.9999985, 6.9e-3/1.9e-2
  relative on the significant elements). Wall (busy board): LayerNorm 257x1024 64 ms,
  RMSNorm 41 ms, FFN 64x512x2048 54 ms, encoder blocks 46/53 ms. `d/n_head` must be a
  multiple of 32 — the brief's `h=4` for `d=64` gives `dh=16`, which the fp16 matmul's
  `K % 32` refuses **in the C too** (`rocket_mha_self_fp16` plans the same
  `[Tp, dh, Tn]` matmul), so the self-test's defaults are `h=2`/`h=4` (Whisper's real
  `dh` is 64). The metrics use a **relative error on the significant elements**
  (|ref| >= 10 % of the array peak): the normalized outputs are standardized rows, so a
  value near zero is a cancellation of two O(1) fp16 terms and its plain element-wise
  relative error is meaningless (0.2 on an output of 0.01 at 2e-3 absolute, i.e. one
  fp16 quarter-ULP of the pre-cancellation magnitude). The FFN/MLP relative gate is
  2e-2; the absolute norm gate is 4e-2 and the encoder's is 1e-1 (the
  brief's 1e-2 is *below* the fp16 pipeline's own noise at `d_ff = 256`).
  **C-vs-Rust A/B** (`tools/c-oracle/ab_board.c` extended with `rmsnorm`, `layernorm`,
  `batchnorm`, `groupnorm`, `instancenorm`, `l2norm`, `geglu`, `ffn`, `ffn_fused`,
  `mlp_fused`, `encoder` — same argument layout as `ops_selftest --io`, `-` for the C
  `NULL`, and the C side built natively on the board): **bit-identical for every norm,
  vision norm, geglu, ffn, ffn_fused and mlp_fused case** (RMSNorm/LayerNorm with and
  without beta and with a unit weight, BatchNorm/GroupNorm/InstanceNorm/L2 over both
  vision shapes, GELU and SiLU, with and without b1, at 4x256x1024 and 64x512x2048).
  That also proves `Plan::pinned` matches `mm_plan_init_pin` and that
  `compute_kacc`-under-a-pinned-plan matches the C's `mm_compute_kacc_cube` —
  a different `Kt` would have changed the fp16 accumulation. The encoder block is the
  only non-bit-identical case (198/512 and 2040/8192 exact, cos 0.9999997/0.9999994):
  its attention sublayer is `attn::mha_self`, whose host fast softmax is the documented
  M5a deviation from the C's on-NPU softmax.
  Two shapes also force the fused entries' **host-handoff fallback** and are
  bit-identical as well — `M = 1` (the `M % 4` pin refusal, then the one-shot height-1
  shim; `M = 2` is refused by *both* libraries, so it is not a state the APIs accept)
  and `M = 260` (the KACC EW-stride floor: `Mt = 256` leaves a 4-row last tile) — which
  is what proves the two sides make the same fallback decision, not just the same plan.
  The encoder's two MLP routes are also bit-identical to each other on the measured
  shapes (`ROCKET_ENCODER_NOFUSE` vs the fused default: 512/512 and 8192/8192 exact),
  because the pinned `Kt = 256` equals the planner's own choice below `d_ff = 512`.
  `ab_compare.py` grew a `--ops` argument (the `ops_selftest` binary) and a per-case
  `strict` gate: the EW/reduce/LUT families require bit-identity, the matmul-bearing
  ones a cosine/max-abs pair (the exact count is printed for both).
- **M8a fp16 convolution** (2026-10-25): `src/rocket_conv.c`'s RK3588 fp16 half
  (`conv2d_one_job` / `conv2d_run` / `dw_spatial` / the deconvolution route / the
  planner / the reference) and `src/rocket_conv_transpose.c` are ported as
  `ops/conv.rs` + `ops/conv_transpose.rs`, with the two generators
  (`gen_conv2d_task` / `gen_conv2d_fill` → `gen_conv2d_fp16` / `gen_conv2d_dw_fp16`)
  as `regcmd/conv.rs`.
  API: `ops::conv::{Conv2dDesc, conv2d_plan, conv2d_plan_code, conv2d, conv2d_ctx via
  ConvContext, conv2d_act, conv2d_act_code, conv1d, conv2d_ref, conv_out_dim,
  conv2d_oh, conv2d_ow, conv2d_trail_y, conv2d_trail_x, conv_dw_group, dw_chunk_fits,
  datain_h, balance_bands, feature_cube_index, weight_cube_index,
  weight_cube_dw_index, scatter_feature/gather_output/scatter_weights/
  scatter_weights_dw, conv2d_deconv_fits, conv2d_deconv, conv2d_job_extent}` and
  `ops::conv_transpose::{ConvTranspose2dDesc, ConvTransposeRoute,
  conv_transpose2d_route, conv_transpose2d_route_code, conv_transpose2d_plan,
  conv_transpose2d_plan_code, conv_transpose2d, conv_transpose2d_ctx,
  conv_transpose2d_ref, conv_transpose2d_oh/_ow, lower_desc,
  conv_transpose2d_deconv_eligible}`. `regcmd::conv::{ConvParams, ConvAct,
  gen_conv2d_fp16, gen_conv2d_dw_fp16, MAX_OPS, MAX_OPS_ACT}`.
  Faithful behaviour: the host scatters the feature into the NC1HWC2 cube (`C2 = 8`)
  and the weights into the `(OC/16, IC/32, KH, KW, 16, 32)` conv cube; **direct**
  pads `IC` up to 32 (the RGB stem, `IC = 3`, is accepted and zero-padded — no
  `IC % 32` requirement) and `OC` up to 16 (an `OC = 24` head is accepted), then tiles
  over OC (mult-16) and over **space** (OH rows, then OW columns); each spatial tile
  **materialises its edge padding** into the sub-input and runs with
  `pad_top = pad_left = 0` (`rocket_conv2d_trail_*` is the pure helper), the row
  budget never drops the materialised height below `CONV_MIN_DATAIN_H = 4` (the CNA
  DMA is 4-row blocked; a shorter sub-input is wrong at any IC), and
  `balance_bands` splits an axis into the fewest near-equal bands so no remainder stub
  falls under that floor. **Depthwise** tiles over channels (multiples of `G`, 32 for
  fp16, `ROCKET_CONV_DW_GROUP`) and, when even one `G`-chunk overflows the validated
  8-bank feature budget, over space as well (`dw_spatial`) — note the upstream header
  still says "not yet implemented" while its code does implement it; this port follows
  the code. The C's refusals are preserved as `conv2d_plan_code`'s `0/-1/-2/-3/-4`
  (malformed / depthwise mismatch / non-positive extent / CBUF) and mapped to typed
  `Error`s; `conv2d_act_code` keeps the fusion's own `-10` (depthwise) and `-11`
  (kind), reported through `Error::coded`/`Error::c_code`.
  `conv1d` is the **height-1** mapping (`IH = KH = OH = 1`, time on the height axis, so
  the row tiler can split a long sequence; the width-on-time form would leave `OH = 1`
  untilable for Whisper's `IC = 512`). The fusion is the same job's DPU LUT epilogue
  (BN index scale → EW LUT → affine OUT_CVT) built by
  `ops::activation::lut_epilogue_build`, direct-only and smooth-kind only.
  ConvTranspose: both routes, chosen by `conv_transpose2d_route` — `DECONV` (the CNA's
  hardware deconvolution mode: direct, dilation 1, power-of-two strides, `opad <
  stride`, `k-1-pad <= 11`, `IC % 32`, compact input fitting one CBUF pass beside a
  16-channel weight tile; kernel rotated + channel-transposed on the host) and
  `LOWERED` (interior-dilated + border-padded input, 180°-rotated kernel, stride-1
  forward conv) — with `ROCKET_CONV_TRANSPOSE_HW=0` forcing the lowering. `opad 0/1`
  and depthwise both work; `conv_transpose2d_ref` is the direct scatter-add oracle, so
  the gate validates the route rather than just the forward conv.
  Program parity: `oracle conv|emit-conv|emit-conv-act` + `regcmd_dump`'s same three
  modes, and `parity.py`'s new `conv` section compares **23 plans, 18 direct/depthwise
  /deconv programs and 6 fused-activation programs byte for byte** (the fused one is
  1152 words: 126 + the two 513-word table uploads). Board (`crates/rocket-npu/src/bin/
  conv_selftest`, cores 0-3, busy board at load ~7, 600 MHz module): the whole default
  matrix PASSes against `conv2d_ref`/`conv_transpose2d_ref` — direct convs cosine 1.0
  with <= 1e-3 relative on the significant elements (>= 99.7 % bit-identical, the rest
  one fp16 ULP), depthwise **bit-identical** (100352/100352, 12544/12544, 18816/18816),
  conv1d 80x3000 k3 s1 1.15 M/1.152 M exact and at 80 ms (s2 31 ms), transposes
  1.6-3.9 ms, wall 0.5 ms for a 32x16x16 k3 on a resident context vs 1.3 ms through the
  alloc-per-call entry, 5.3 ms at 64x56x56 k3, 5.8 ms at 128x28x28 k3 s2, 7.4 ms at
  32x112x112 k3 and 4.7 ms for a 32x56x56 depthwise. The fused path is the C's
  **single-pass** table, so its own `QUIRK 1` artifacts appear in both libraries
  (the `x≈0` mux spike and gelu's saturated tail, up to 128); the selftest gates the
  device against the `lut_apply` model of that table (<= 5e-3 relative on the
  significant elements) and prints the table-vs-exact-f deviation without gating it.
  C-vs-Rust A/B (`ab_board.c` + `ab_compare.py --conv`): **bit-identical for every
  M8a case** — 15 direct convs (incl. the RGB stem, the OC=24 head, the OC-only and
  OC+spatial tiling shapes), 2 conv1d, 3 fused activations, 7 transposed convs (both
  routes) — and the C library's own `conv2d_fp16_rocket` / `conv1d_rocket` /
  `conv_transpose_rocket` / `conv_act_rocket` gates PASS on the same board.
  `cargo test -p rocket-npu` pins the geometry helpers, the plan accept/refuse classes,
  the cube scatter/gather and weight-cube round trips, `conv1d`'s mapping, the
  transposed-conv dims/route table (including the `ROCKET_CONV_TRANSPOSE_HW=0`
  override) and hand-computed `conv2d_ref`/`conv_transpose2d_ref` cases (1x1, 3x3 pad
  1, stride 2, dilation 2, depthwise, a box spread).
  **M8b int8 conv + pool + resize** (2026-10-25): the rest of `rocket_conv.c` plus
  `rocket_pool.c` and `rocket_resize.c`. API: `ops::conv::{conv2d_int8,
  conv2d_int8_ctx, conv2d_int8_mt, ConvInt8Pool, conv2d_ref_int8, conv2d_dw_int8,
  conv2d_dw_int8_ctx, conv2d_dw_int8_model, conv2d_int8_plan_code,
  conv2d_dw_int8_plan_code, conv_int8_tile_plan, ConvInt8Tiles/ConvInt8Tile,
  dw_int8_chunk, requant_params/requant_round_shift/requant_apply/requant_apply_zp,
  weight_cube_index_i8, feature_cube_index_i8, scatter_feature_i8,
  scatter_weights_i8, scatter_weights_dw_i8, gather_output_i32, gather_output_i8,
  DW_INT8_G}` and the `ConvContext::conv2d_int8`/`conv2d_dw_int8` methods;
  `ops::pool::{PoolDesc, pool_fp16, pool_fp16_plan, pool_fp16_plan_code, pool_int8,
  pool_uint8, pool_ref_fp16, pool_ref_int8, pool_ref_uint8, pool_out_dim,
  POOL_ERR_*}`; `ops::resize::{upsample_nearest, upsample_bilinear,
  upsample_nearest_ref, upsample_bilinear_ref, upsample_nearest_plan,
  upsample_bilinear_plan, upsample_desc, build_kernel, bilinear_ksize}`. The int8
  generator is `regcmd::conv::{ConvInt8Params, gen_conv2d_int8, gen_conv2d_dw_int8}`
  (the emitter gained `bias_en`/`bias_base_addr` for the BS-ALU bias add, and the
  fp16 programs stay byte-identical), the pool reuses `regcmd::pool::gen_pool_fp16`
  (the C routes int8/uint8 through the fp16 program with a cube-boundary convert, so
  there is no second pool encoding), and the resize rides the depthwise transposed
  conv. Numerics: the int8 direct path is int32-**exact** vs the int64 reference on
  every shape; the int8 depthwise output is **bit-exact vs the DPU requant model**
  (`conv2d_dw_int8_model`: `sat8(round_half_even(acc*mul >> shift) + out_zp)`, the
  ties-to-even rule measured on the part; the C's own reference uses TFLite's
  `lrintf` and may differ by one at a boundary); max pooling exact, averages within
  the `fp16(65536/k)` reciprocal; nearest upsampling bit-exact, bilinear within the
  fp16 triangle kernel's rounding. Program parity: `parity.py`'s new `int8conv`
  section compares **11 direct + 9 depthwise programs byte for byte** over a
  shape/zero-point/scale matrix (including the int32-raw diagnostic depthwise form,
  the `+1` CBUF bank slack, the `size_e=7`/`surf_add*8` int32 writer, the OUT_CVT
  triple and both refusals), and four int8/uint8 pool geometries were added to the
  `pool` matrix; all six dtypes + pool + conv + int8conv PASS.
  Board (`conv_selftest`, cores 0-3, busy board at load ~5-13, 600 MHz module): the
  int8 direct matrix PASSes int32-exact — 3x32x32 k1 2.1 ms (alloc-per-call entry),
  32x16x16 k3 0.4-1.0 ms, 64x14x14 k3 1.2 ms, 64x56x56 k3 5.6 ms, 32x112x112 k3 6.6 ms,
  128x28x28 k3 s2 4.9-23 ms (load-dependent); the int8 depthwise matrix PASSes
  bit-exact vs the model — 64x16x16 k3 1.0-7.9 ms, 64x16x16 k3 s2 1.4 ms, 64x8x11 k5
  1.2 ms, 128x7x5 k3 1.0 ms, zero points at both ends and with/without the bias cube;
  the pools PASS (fp16 max exact, avg <= 1e-3 absolute at the peak, int8/uint8 max
  exact and avg within one rounding step) at 0.5-2.0 ms; the upsamples PASS (nearest
  bit-exact, bilinear <= 1e-3 absolute) at 1.4-5.8 ms. C-vs-Rust A/B
  (`ab_compare.py`): **bit-identical for every M8b case** — 9 int8 direct convs
  (incl. the RGB stem, the OC=16 pad and the three tiling shapes), 6 int8 depthwise
  convs (C=64/128, strides 1/2, 5x5, both zero-point ends, with/without bias), 9 pools
  (fp16/int8/uint8 max+avg) and 8 upsamples (nearest+bilinear, scales 2/3/4,
  non-square) — on top of the M6b/M7/M8a sections, which still pass. The multicore
  pool is checked in-process too (`conv_selftest --int8-mt`): a 3-device
  `ConvInt8Pool` output is **bit-identical to the serial entry and to the int64
  reference** on the OC-tiled shapes (64x56x56 oc64, 128x28x28 oc128 with
  `oc_tile = 96` so the second band is 32 channels, 32x112x112 oc32), and a single-OC
  shape falls back to serial as designed. The C library's own gates PASS on the same
  board for these families: `conv2d_int8_rocket`, `conv_dw_int8_runtime` (the Teflon
  capture: 11/4096 differ by one from TFLite, **0 vs the host model**, and `w_zp = 3`
  refused with -5), `resize_rocket`, `pool_fp16_rocket` and `pool_int8_rocket`.
  **Two encoder envelopes the C does not check** (found on the board, mirrored
  because the programs are identical): (1) the PPU's multi-window pooling is only
  reliable up to a **kernel of 8** — a 9x9+ kernel (or a stride larger than the
  kernel) writes zeros/garbage; a 14x14 kernel works only when the plane is exactly
  one window, which is what the global-pool decomposition uses (`reduce_selftest
  --pool` still PASSes, incl. its 14x14 tail pass); (2) the int8 depthwise writer
  walks channels in `G = 64` groups, so a `C` that is a multiple of 16 but not of 64
  (32, 96) yields an identically-wrong surface on both sides (the C's own gate only
  uses 64/128). The self-test matrix stays inside both envelopes; the A/B keeps one
  case of each to pin the bit-identity. `cargo test -p rocket-npu` pins the int8
  plan/ref hand cases (1x1, 3x3, depthwise, stride 2), the tile-plan coverage (every
  output element covered exactly once) and OB banks, the requant model vs the
  emitter's OUT_CVT words, the int8 cube round trips, the pool plan classes
  (incl. `avg_exclude_pad` -> -10 and the C's truncating `out_dim`), the pool
  references (count-include-pad, int8 rounding, uint8 recentring), the resize
  kernel/plan/geometry table (scales 1..4) and the half-pixel zero-boundary
  reference.
  **M9 SAM + SigLIP encoders** (2026-10-25, the parity capstone):
  `src/rocket_sam_encoder.c` (769 lines) and `src/rocket_siglip_encoder.c` (765) are
  ported as `ops/sam.rs` and `ops/siglip.rs`. `ops::sam` is the SAM ViT-Det image
  encoder: patch conv (k = stride = `patch`, with bias) + the learned 2D position
  table, then pre-norm blocks with a **fused qkv** (`[3d,d]`), **windowed** attention
  on all but the periodic global layers (the grid is zero-padded to a multiple of
  `win` and — faithfully — the pad tokens take part as keys/queries in the boundary
  windows), the scale folded into the softmax over the *unscaled* `QK^T`, the
  **decomposed relative-position bias** (`rel_h`/`rel_w` from the resolved
  `[side,side,dhead]` tables, added to the scores before the softmax), an erf-GELU
  MLP, and the neck (1x1 conv as a matmul, channels-first LayerNorm, 3x3 pad1 conv,
  channels-first LayerNorm). `ops::siglip` is the SigLIP-B/16 plain-ViT tower: patch
  conv + position table, pre-norm blocks built on `ops::encoder::encoder_block`, a
  post-LayerNorm, plus the C's plain-ViT knobs (`n_prefix`/`prefix` cls row,
  `pre_g`/`pre_b` pre-encoder LN, `act_kind` quick-GELU / exact-erf, DPT `taps`) in
  the resident path and `patch_dim_pad` (patch 14's 588 -> 608 K align).
  API: `sam_load`/`SamModel::{load, from_blob, from_parts, synthetic, to_blob}`,
  `sam_encode(&Device, ...)` (the C one-shot), `sam_encode_ref` (the exact `fd < 0`
  host reference, device-free), `SamContext::{create, encode}` + `sam_encode_ctx`
  (resident: the patch, per-layer qkv/o/fc1/fc2 and neck conv1 weights packed once),
  and the same six for SigLIP (`siglip_load`, `siglip_encode`, `siglip_encode_ref`,
  `SiglipContext`); `SamGeometry`/`SamLayerWeights`/`SamNeckWeights` and
  `SiglipGeometry`/`SiglipLayerWeights`/`SiglipVariant`/`SiglipAct` carry the model.
  Loader errors keep the C codes (`-1` open, `-2` size, `-3` magic/version, `-4`
  dims/blob size) and the C's exact-size check; the C mmaps, this port owns the
  decoded tensors.
  The host/NPU split is the C's: the big GEMMs on the NPU (`matmul::one_shot` for the
  one-shot arms, the resident `Context` for the ctx arms), the `O(tokens·d)` glue on
  the host — im2col (`patchify`, shared by both), the position add, window
  partition/unpartition, the LayerNorms, the fp16-input erf-GELU / GELU-tanh /
  quick-GELU tables (bit-exact to the C's f32 formulas through glibc `erff`/`tanhf`),
  the biased softmax row and the decomposed rel-pos matmul+add in fp16 (no host-f32
  fallback). SAM's reference accumulates in **f32** and SigLIP's in **f64** — the C's
  two different `host_matmul`s, kept apart. The ctx path runs the computed QK/PV
  matmuls through the resident `Context` (N fanned across its workers) where the C
  uses one stream + pthreads per (window, head); SigLIP's ctx also mirrors the C's
  `ROCKET_SIGLIP_FA` fan-out through `attn::Attention` (the per-head path when off).
  **Host-reference bit-identity** is the primary gate and it holds: Rust
  `*_encode_ref` == the C `*_encode(fd < 0)` for the output *and* every intermediate
  block, on the synthetic fixture (x86 and the board, via
  `encoders_selftest --io *_ref` against `ab_board sam_ref|siglip_ref`) and on the
  real checkpoints (x86; the same pair at the real SAM geometry was started on the
  board too but the naive reference takes ~1 h per side on the A55s, so it is left
  out — the board fixture already pins the same libm calls). Finding: `half`'s
  `f16::from_f64` is **not** correctly rounded on x86 with F16C (it detours through
  f32, so a value just past an fp16 tie rounds down one ULP), while the C's
  `(_Float16)(double)` is single-rounded and the
  aarch64 `half` path is the hardware `fcvt h, d`. A new `ops::f16_from_f64` does the
  exact RNE conversion in software (and is used by the SAM/SigLIP references,
  `norm::layernorm_ref`, `encoder::encoder_block_ref` and `attn::mha_self_ref` /
  `flash_attn_ref`); aarch64 results are unchanged, x86 now matches the C.
  Synthetic fixture (`SamGeometry::fixture()` / `SiglipGeometry::fixture()`: d=64,
  2 layers, 2 heads/dh=32, d_ff=64, grid 8 / win 4 / patch 8 / ic 3 / neck 16, and
  SigLIP L=64 / patch 8): the board (cores 0-3) gives SAM one-shot cos
  0.99999993 (layer 1) / 0.99999970 (global layer) / **0.99999951** (neck), resident
  ctx the same (its output is **bit-identical** to the one-shot, and repeats are
  bit-identical), and SigLIP one-shot 0.99999906 / 0.99999781 / **0.99999755**,
  resident 0.99999944 / 0.99999874 / **0.99999865** (the resident arm is the
  GELU-tanh host table, so it differs from the erf one-shot by design and lands
  *closer* to the erf host reference here). Wall (synthetic, busy board): SAM ref
  51.7 ms, one-shot 56.2 ms, ctx create 11.0 ms + 26.2 ms/image; SigLIP ref 41.3 ms,
  one-shot 72.4 ms, ctx create 10.7 ms + 9.6 ms/image (the one-shot pays
  `mha_self`'s per-call fds + a full weight re-pack).
  Real checkpoints (`facebook/sam-vit-base` 183.8 MB, the SmolVLM-256M SigLIP-B/16
  tower 172.9 MB, both extracted with the reference `tools/*_extract.py` in the emb2
  venv): the C's own gates PASS on the board (`siglip_rocket`: mean-layer 0.999985,
  post-LN 0.999884 vs the HF f32 reference; `sam_rocket`: mean-layer & neck 0.999999,
  worst 0.999998, one-shot 153.9 s), and the
  Rust arms are scored against it on the board (cores 0-3, load average 9-11):
  SAM one-shot worst-layer 0.99999848 / neck 0.99999938 in 149.9 s, resident worst
  0.99999842 / neck 0.99999936 in 113.5 s (+1.8 s create); SigLIP one-shot worst
  0.99998189 in 18.8 s, resident worst 0.99993420 in 4.4 s (+0.8 s create, the
  GELU-tanh vs erf delta included; `ROCKET_SIGLIP_FA=0`'s per-head path gives
  0.99992939 in 13.3 s, so the FA fan-out is 3x on this shape). The C's one-shot on
  the same artifacts takes 153.9 s, so the Rust one-shot is at parity; SigLIP's
  resident arm is 4.3x its one-shot. The real-geometry host reference is ~21 min of
  naive host matmul (Rust, ~20 min), and the C's is ~68 min because baseline x86-64
  has no F16C, so its per-element `(_Float16)` reads become libgcc `__extendhfsf2`
  calls; it is therefore computed once on the dev host and loaded with `--ref-in`
  (`encoders_selftest` grew `--ref-in`/`--ref-out` for exactly that) after the
  bit-identity check above (which holds for both real checkpoints on x86).
  `cargo test -p rocket-npu` pins the blob round trip and the `-1`..`-4` loader
  classes for both formats, the windowed partition/unpartition (including a padded
  grid's pad tokens), the rel-pos block indexing, the biased softmax definition, the
  erf-GELU / GELU-tanh / quick-GELU tables against their float formulas, the row
  softmax, the reference determinism and the variant validation (CLIP prefix / pre-LN
  / quick-GELU, ascending in-range taps, `patch_dim_pad` agreement), and
  `ops::f16_from_f64` against every fp16 round trip plus the double-rounding
  regression.
  Not covered: the CLIP/DPT variant knobs are implemented but only exercised
  structurally (no CLIP or DPT checkpoint was available; the C's own gate covers only
  the SigLIP shape too), and the resident path has no SAM worker-pool analogue of the
  C's per-(window, head) stream threads (same arithmetic, different task split).

  **M10 on-NPU attention softmax** (2026-10-25, the last attention surface): the C
  library's optional on-NPU softmax (`ROCKET_ATTN_HOST_SOFTMAX=0`) is ported as
  `attn::RowSoftmax` (`crates/rocket-npu/src/attn/softmax.rs`): it owns the host pass
  (`attn::Softmax`, unchanged and still the default) and, for the on-NPU arm, the f16
  masked-score write (the C `fa_mask_scores`, `fa_score` in f32 then one f16 narrowing,
  pad columns at the `-30000` clamp) that feeds `ops::softmax::softmax` — the M6b
  `rocket_softmax_fp16` composition already verified bit-identical to C. `host_softmax()`
  resolves the knob per call (`0` = on-NPU, `1`/unset = host), and the **materialized**
  paths honor it: `Attention::flash`'s per-head path, its chained group path (that mask +
  softmax is also a full-key-axis row softmax per head, and the C
  `fa_heads_range_batched` passes its `host_sm` through the same loop, so
  `ROCKET_FA_CHAIN=0/1` stays bit-identical under either placement — verified) and
  `mha_self` (whose C default is on-NPU). The tiled `ROCKET_FA_TILE_KV` path stays
  host-only by construction (the online running max/rescale is sequential across tiles;
  the C tiled range takes no `host_sm` either). Defaults are unchanged (host
  everywhere), i.e. the C's `mha_self` default is NOT reproduced unless asked for:
  `ROCKET_ATTN_HOST_SOFTMAX=0` is the parity switch, `1` the host arm.
  **C-vs-Rust A/B** (`ab_board` grew `flash`/`mha` families with `attn_selftest --io`'s
  layout, driven by `ab_compare.py --attn-only`, board cores 0-3): the on-NPU arms are
  **bit-identical for every case** — flash causal/no-mask/soft-cap 8/MLA `dv != dk`/
  n_kv > T one-hot at 64 keys, and `mha_self` at T=8 d=64 h=2, T=5 d=128 h=2 and
  T=33 d=64 h=2 (Tp/Tn padding; the C single-fd entry chains its heads by default and so
  does the Rust one, so the chained placement is pinned by the same cases) — while the
  host arms differ by <= 2.4 f16 ULP at the output peak (cos 0.9999992-1.0), which is the
  documented f32-vs-f16 score path.
  **Board wall/CPU** (busy board, load ~10, cores 0-3; `attn_selftest --sm-ab`): the
  on-NPU arm is 1.8-2.2x the single-fd host arm at T=257 kv=257 (two runs: 2.15x and
  1.79x, the board scatters +-20 %) and 2.55x at kv=1024, with 1.2-1.3x (kv=257) and
  1.76x (kv=1024) through the mt(4) fan-out; `mha_self` 1.1-1.25x at T=257 and 1.75x at
  the kv=1024 run's T=256. That is the expected direction and larger than the C's own
  ~5-10 %: the C compares against a scalar `expf` host pass, this port against the S1
  fast pass, so the on-NPU arm is a parity/comparison knob, not a default. Against the
  f64 references the on-NPU arm reads cos 0.99999990 (flash) /
  0.99999887-0.99999989 (`mha_self`) — the EXP LUT's approximation, as documented.
  `attn_selftest` grew `--host-sm 0|1`, `--sm-ab` and the `--io flash|mha` raw-file
  mode; the chain A/B and the `mha_self` fan-out stay
  bit-identical under `--host-sm 0` (0 differing of 131,584 / 5,248). The knob is
  **global**: it also reaches `burn_rocket`'s attention ops (they call
  `Attention::flash`), i.e. every app NPU attention path, so it is a comparison switch
  and must not leak into a deployment environment by accident (unset = the host pass,
  the shipped behaviour, which is unchanged).

  **M11 cross-op cube residency** (2026-10-25, the last matmul half): `ops::ffn`'s
  fused entries now run the C's `RESIDENT-FUSION NOTE` flow instead of de-tiling the
  `[M,I]` intermediates to row-major and re-packing them. The matmul side is
  `matmul::run`'s cube API (all `pub(crate)`, no emitter change): `alloc_cube`
  (`cube_elems` fp16 + one CBUF bank, **zeroed** — a ragged tile's pad lanes must be 0,
  not NaN: the consumer's zeroed weight lanes cannot mute `0*NaN`), `compute_kacc_cube`
  (`mm_compute_kacc_cube`: one batch, data-reuse mode 2 so `gi` *is* the canonical
  `(mi*nNt+ni)` slot, ping-pong across the `nKt` jobs with the cube holding the final
  parity, trailing `prep(0)` and **no de-tile**), `AOperand::of` (the C's
  `b.in_all = cube_g`), `compute_kacc_from`/`compute_cpu_from` (`mm_compute_kacc` then
  `mm_compute` on the tiny-M sentinel) and `layout::scatter_bias_cube`
  (`mm_scatter_bias_cube`). `policy_config(KaccPolicy::{Only,Forced})` forces the KACC
  scratch on regardless of `ROCKET_NO_KACC`, as the C's direct primitive calls do; the
  scaffolding the row-major version needed (`one_shot_plan`, `KaccPolicy::Auto`) is
  gone. Flow, in `ops/ffn.rs`: `projection_cube` = `pack_input` + `pack_weights` +
  `compute_kacc_cube` per projection; `activation` then `ew_mul` over the cube's fp16
  bytes (`bos::f16_words{,_ref}`), `store_cube` writes the product back (`prep(0)` +
  `fini`, the C's prep-for-device pair); `consume_cube` runs down/fc2 with the product
  cube as its A operand. `mlp_fused` inserts the C's cube bias: `scatter_bias_cube` +
  `ew_add` over `cube_elems` (skipped when `b1` is `None`) before the activation.
  Everything the C keys off the *matched-tiling gate* is unchanged, and so is the
  `goto fb` fallback (pin refusal, unmatched tiling, more tiles than one batch, KACC
  EW-stride refusal, any device error).
  Gates: `ops_selftest` before/after on identical raw fixtures is **bit-identical for
  every case** (`ffn`, `ffn_fused` and `mlp_fused` with/without the fc1 bias at
  4x256x1024 and 64x512x2048, and both encoder blocks — `encoder_block` inherits the
  win through `mlp_fused`), and `ab_compare.py --ops` still reports `ffn`,
  `ffn_fused`, `mlp_fused` (both bias arms), the two fallback shapes and the norms/
  geglu **bit-identical to C**, with the encoder block at the unchanged 198/512 and
  2040/8192 (the `mha_self` host-softmax deviation). New device-free host test
  `ops::ffn::tests::the_cube_bias_path_is_the_row_major_bias_add` (a ragged 384-wide
  `Plan::pinned` cube: `feat_scatter_tile` -> `scatter_bias_cube` -> `ew_add_ref` ->
  `detile_store` must equal `add_bias`), `cargo test -p rocket-npu` 252 pass.
  Wall (busy board, load ~9, cores 0-3, medians of 6 interleaved rounds, `--reps 3`;
  the identical-code arm scatters by ~10 %, so read the ratios): `ffn_fused`
  19.0 vs 19.9 ms at 4x256x1024 and 64.8 vs 63.5 ms at 64x512x2048 (neutral, and
  better *relative* to the unchanged `ffn` baseline: 0.81x vs 1.07x and 1.33x vs
  1.44x), `mlp_fused` 14.9 vs 12.2 ms and 48.5 vs 43.6 ms, encoder 36.7 vs 35.4 /
  60.5 vs 56.5 ms — the mlp pays 2-5 ms for the C's **on-NPU bias EW add** (5 EW tiles
  at 32640 elements each, 5 fresh BOs per call) where the row-major route added the
  bias on the host in f32; the removed work (de-tile 2 cubes + re-pack 1) is only
  ~0.2 ms at these shapes. User CPU is unchanged (2.58 vs 2.66 s for the 64x512x2048
  run). Verdict: keep the C mechanism (the win grows with the removed host traffic,
  and it is what the C ships), the row-major `ffn`/`mlp` stay the fallback.
  `ROCKET_FFN_DEBUG=1` prints the routing decision (`ffn_fused: ... cube-resident` /
  `... not cube-matchable -> host-handoff` / `kacc-cube ... refused`), the C's knob.
  New `log_info!` macro in `log.rs` for it.

  **Missing surface** (documented in `crates/rocket-npu/src/lib.rs`): resident bf16
  (the C library has no variant either — and neither has it for tf32, whose surface is
  therefore complete), the `*_ref_fp16` host references of the *remaining* non-matmul
  families, `_planes`/transposed variants, RK3576 entries (including
  `rocket_act_int8_rk3576`), the public `rocket_stream_*` aliases, the non-fp16 batch
  wrappers, the RK3576-specific convolution/pooling entries (the
  part's own conv encoder, the packed-image datapath, the int8 pooling entry with its
  zero point and the prepacked/cube handles; `rocket_conv2d_int8_rk3576`,
  `rocket_conv2d_dw_int8_rk3576`, `rocket_pool_int8_rk3576*`,
  `rocket_conv2d_int8_pack_rk3576`), the int8 conv's opt-in `ROCKET_CONV_BATCH`
  batched (gapped multi-task) submit — bit-identical to the per-tile path this port
  runs — and graph execution (the RK3576-only graph runner). (The tiled
  long-context attention path *is* implemented — M5b above; the activation family is
  M6a, the reduce + softmax family M6b, the norms/vision norms/FFN/encoder M7, the
  fp16 convolution family M8a, the int8 convolution + PPU pooling + upsample families
  M8b, and the SAM + SigLIP model encoders M9.)
- Not done: int8 GEMM on the *flex CPU* path (still the only lever that would close
  the qwen3 speed gap; flex lacks it, and burn-cpu/CubeCL quantized matmul is
  unverified and cannot cross-compile) — the *NPU* int8 path is now implemented for
  emb2 (above). The live OpenViking embedding backend
  (`home-llm-gateway-embed-1`, formerly `openviking-embed-1`) runs EmbeddingGemma 2
  from `unsloth/embeddinggemma-2-GGUF:Q8_0` on the CI `:latest` image (`94f99bd`,
  observed 2026-10-24), which predates the S1/S2 softmax/vision round and needs a
  rebuild + redeploy (see "Board deployment").

## Repo layout

| path | contents |
|---|---|
| `src/lib.rs` | compatibility wrappers (`RocketCtx`/`RocketWeight`/`RocketFaCtx`), driver/counter helpers, `Error`/`OpFailure` |
| `crates/rocket-npu/` | **the pure-Rust driver**: `device/` (uAPI, BO, submit, fence, sysfs, affinity), `regcmd/` (desc + the fp16/int8/int4/bf16/tf32 emitters, `activation` (the LUT flying-mode program), `pool` (the PPU pooling program + `ppu_recip_kernel_fp16`) and `conv` (`gen_conv2d_fp16`/`gen_conv2d_dw_fp16`: the direct/depthwise conv program with the fused-activation epilogue and the CNA deconvolution fields; `gen_conv2d_int8`/`gen_conv2d_dw_int8`: the int8 direct program, its `+1` CBUF bank slack and int32 writer, and the depthwise int8-out requant with the BS-ALU bias)), `matmul/` (plan, layout, bos, run, context + the int8/int4/tf32 one-shots and the resident int4 `I4Context`; `Plan::pinned` is the `mm_plan_init_pin` port the fused FFN/MLP need, and `run`'s `alloc_cube`/`compute_kacc_cube`/`compute_kacc_from`/`compute_cpu_from` + `layout::scatter_bias_cube` are the cross-op cube path those entries drive), `attn/` (flash attention + the fast host softmax), `ops/` (the activation/elementwise family: LUT builders, refs, the DPU LUT model, the EW conv-main ops; `reduce` (PPU global pools + feature reduce + cumsum), `softmax`, `norm` (the per-row scale, RMSNorm, LayerNorm, `square_prescale_k`), `normvision` (BatchNorm/GroupNorm/InstanceNorm/L2), `ffn` (geglu, the block, the cube-resident fused pair, the encoder MLP), `encoder` (the pre-norm block), `conv` (the fp16 CNA conv: planner, host cube scatter, OC/spatial/channel tiling, `ConvContext`, the deconvolution route, the reference; plus the int8 half: the int32-raw direct conv with its tiler and multicore `ConvInt8Pool`, the int8-out depthwise requant with its host model, the int8 cube index math) and `conv_transpose` (both routes + the scatter-add reference), `pool` (the PPU MaxPool/AveragePool family with the int8/uint8 route, the plan classes and the references), `resize` (nearest/bilinear upsample as a depthwise transposed conv, the kernel builder and the half-pixel references), `sam` and `siglip` (the two model encoders: the `*_extract.py` blob loaders, the device-free host references, the one-shot entries and the resident prepacked contexts, plus their geometry/weight structs and the `f16_from_f64` exact narrow shared with the reference paths)), `hw`, `knobs`, `error`; `src/bin/` has the parity dump, the dtype/attention/activation/reduce/norm-FFN board self-tests (`activation_selftest`: `--kind`, `--n`, `--shape M K N`, `--all`, `--ew`, `--dump-lut`, `--dump-gen`, `--write-in`/`--run-in`/`--model-in`/`--cross`/`--ew-in` for the C A/B; `reduce_selftest`: `--pool`, `--feature`, `--cumsum`, `--softmax`, `--shape C H W`, `--reps`, and the `--io` raw-file exchange for the C A/B; `ops_selftest`: `--norm`/`--vision`/`--ffn`/`--encoder`/`--all`, `--norm-shape M H`/`--vision-shape N C P`/`--ffn-shape M H I`/`--encoder-shape T D H DFF`, `--reps`, and its own `--io` exchange; `conv_selftest`: `--conv`/`--dw`/`--act`/`--conv1d`/`--transpose`/`--int8`/`--int8-mt`/`--int8-dw`/`--pool`/`--resize`, the per-family shape flags, `--reps` and its `--io`; `encoders_selftest`: `--write-blob DIR`, `sam`/`siglip` with `--ref`/`--npu`/`--ctx N`/`--reps R`/`--out`/`--hidden`/`--ref-in`/`--ref-out`, and the `--io` mode for the M9 A/B) |
| `src/ext.rs` | the `RocketOps` Burn backend extension, the global NPU engine (`init`, `WeightId`, `burn_rocket::stats`) and the `Tensor`-level helpers (`pack`/`matmul`/`attention`, `attention_window`/`_block`, `attention_causal_window`, `attention_causal_block`/`_window`) |
| `src/masks.rs` | host-built additive attention masks (causal, bidirectional band, causal sliding window, query-chunk band, query-chunk causal and causal-window) with unit tests |
| `src/host.rs` | fused host kernels for the NPU build's CPU glue (gelu·up, RMSNorm, RoPE) with unit tests |
| `tools/c-oracle/` | host-side C oracle (planner + regcmd dump for every matmul dtype, the PPU pool program and the average reciprocal, the conv plan/direct/dw/act programs and the int8 direct/depthwise programs), the `parity.py` dtype+pool+conv+int8conv parity harness, the board A/B benches used to validate the port (`ab_board.c` + `ab_compare.py` for the reduce/softmax, norm/vision/FFN/encoder, conv/transpose, int8-conv/pool/resize and – M9 – `sam|sam_ref|siglip|siglip_ref` raw-file A/B), and `build.sh` for the oracle itself |
| `crates/rocket-npu/src/bin/` | `regcmd_dump` (planner/emitter parity, including `recip`/`emit-pool`, `conv`/`emit-conv`/`emit-conv-act` and `emit-conv-int8`/`emit-conv-dw-int8`), `selftest` (fp16 matmul), `attn_selftest` (attention: the chained A/B by default, `--oneshot` for the stateless entries vs the f64 ref, `--mha` for the encoder pair, `--host-sm 0|1` for the softmax placement, `--sm-ab` for the host-vs-on-NPU arm report, and an `--io flash|mha` raw-file mode for the C A/B), `int8_selftest` (`--dense`/`--prepacked-dense`/`--gw-one-shot`/`--prepacked-gw`), `int4_selftest` (`--dense`/`--prepacked`/`--gw-one-shot G`/`--prepacked-gw G`, plus `--kt-cap`), `int16_selftest`, `bf16_selftest`, `tf32_selftest` (no args = the built-in shape list, vs a tf32-rounded f64 reference), `activation_selftest`, `reduce_selftest` (`--pool`/`--feature`/`--cumsum`/`--softmax`/`--shape C H W`/`--reps`, plus the `--io` raw-file mode), `ops_selftest` (`--norm`/`--vision`/`--ffn`/`--encoder`/`--all`, per-family `--*-shape`, `--reps`, plus its own `--io` raw-file mode), `conv_selftest` (`--conv`/`--dw`/`--act`/`--conv1d`/`--transpose`/`--int8`/`--int8-mt`/`--int8-dw`/`--pool`/`--resize`/`--all`, `--shape`/`--dw-shape`/`--conv1d-shape`/`--transpose-shape`, `--reps`, plus its own `--io` raw-file mode covering `conv_int8`, `conv_dw_int8`, the three pools and the two upscales), `encoders_selftest` (`--write-blob DIR`, `sam`/`siglip` with `--ref`/`--npu`/`--ctx N`/`--reps`/`--out`/`--hidden`/`--ref-in`/`--ref-out`, plus the `--io` mode covering `sam_ref`/`sam_npu`/`sam_ctx`/`siglip_ref`/`siglip_npu`/`siglip_ctx`), `q8_bench` (Q8_0-native projection probe: resident f16 `Context` vs group-wise `I8Context` at qwen3's shapes, `--reps`/`--threads`/`--groups 32,128,512`/`--rows`/`--only`, log §20) |
| `examples/probe.rs` | low-level driver probe (open device, pack, matmul, verify vs CPU) |
| `examples/rocket-inference/Cargo.toml` | workspace member `rocket-inference`: features `cpu` (default), `npu` (aarch64-gated path dep on the root crate), `gpu*`; `wgpu_probe` bin behind `gpu` |
| `examples/rocket-inference/src/main.rs`, `src/cli.rs` | dispatch (`serve` + `qwen3`/`intent`/`gemma`), shared flag parser and usage text |
| `examples/rocket-inference/src/server/` | one HTTP server: `cli` (flags + model loading), `engine` (family detection, capabilities, per-model compute), `error` (panic/NPU mapping), `log` (per-request speed lines), `openai` (`/v1/embeddings`, `/v1/chat/completions`, `/v1/models`), `multimodal` (`/embed`), `ollama` (`/`, `/api/*`) |
| `examples/rocket-inference/src/util/` | shared infra: backend/device, `rss_mib`, server lock/panic helpers (`http`), Q8 low-RAM mapper (`quant`), `Proj`/`ProjKind`/`FusedGroup` (`proj`), `RopeCache` (`rope`), load-report checks (`store`), the shared checkpoint trait (`checkpoint`), GGUF access (`gguf`), the GGUF tokenizer builder (`gguf_tokenizer`), query-chunked causal attention for the NPU path (`causal`), and `hf-hub` model resolution (`hub`) |
| `examples/rocket-inference/src/qwen3_embedding/` | Qwen3-Embedding family: model (layers, RoPE, attention paths, stage timers), loaders (f32 / `--quant q8` / `--npu` pack-and-drop + `load_cpu_projections`), CLI |
| `examples/rocket-inference/src/qwen35_intent/` | Qwen3.5-0.8B intent family: model (chunked/recurrent gated delta rule, gated full attention, caches, greedy generation), loader (`model.language_model.*`, NPU pack-and-drop, `load_for_serving`), CLI |
| `examples/rocket-inference/src/gemma/` | shared Gemma 4 building blocks: `config` schemas, `layers` (NPU/SRQ/`ClippableLinear`), `qat`, `media`, `audio` + `audio_frontend`, `vision`, `inputs` |
| `examples/rocket-inference/src/gemma/embeddinggemma/` | EmbeddingGemma 2 family: text backbone, multimodal assembly, loaders, CLI |
| `examples/rocket-inference/src/gemma/gemma4/` | Gemma 4 E2B-it family: causal decoder, loader (f32/f16/q8/QAT, `pack_text_for_prefill`), chat template + sampling (`GenStats.stopped`), CLI |
| `examples/rocket-inference/src/bin/wgpu_probe.rs` | Vulkan/wgpu GPU probe (`gpu-*` features) |
| `examples/rocket-inference/src/bin/vision_attn_bench.rs` | vision-tower attention/softmax microbenchmark (`fused-f16/bf16`, `npu`, `npu-chunk`, `matmul`, `attn-batch`, `softmax`; `npu` feature, log-gemma §18.4/§19/§20) |
| `examples/rocket-inference/src/bin/causal_attn_bench.rs` | causal block attention probe (P2a): the chunked NPU path (`util/causal.rs` compiled from the same source) vs the dense masked op and `attn::flash_attn_ref`, `--window`/`--ref`/`--reps` (`npu` feature) |
| `examples/rocket-inference/src/bin/rope_bench.rs` | RoPE probe (P2c): `burn_rocket::rope_apply` (fused host kernel) vs `RopeCache::apply` (composite) vs an `ops::activation::ew_mul`/`ew_add` composition on the NPU, at qwen3's q/k shapes, `--tokens`/`--dtypes`/`--layers`/`--reps`/`--skip-npu` (`npu` feature, log §21) |
| `examples/rocket-inference/data/` | bench/embedding/media fixtures (`bench_text.txt`, `cat.jpeg`, audio/video samples) |
| `examples/rocket-inference/docs/experiment-log.md` | Qwen3-Embedding + intent measurements: §1-8 dev-host, §9 board A/B, §10 NPU, §11 extension-ops refactor + CPU-loading fix, §12 serving robustness, §13 Vulkan, §14 intent model, §15-18 bf16/maintenance + production A/B, §19 tiled attention (M5b), §20 Q8_0-native int8 (P2b), §21 RoPE on the NPU (P2c) |
| `examples/rocket-inference/docs/experiment-log-gemma.md` | EmbeddingGemma 2 + Gemma 4 measurements (multimodal parity, Q8/NPU, QAT mobile, NPU prefill) |
| `examples/rocket-inference/tools/` | HF reference scripts (`ref_embeddinggemma2.py`, `ref_gemma4.py`, `debug_audio_hf.py`), the int8 group-wise probe (`w8a8_probe.py`), the GGUF tokenizer parity check (`tokenizer_parity.py`) and the video parity check (`video_parity.py`) |
| `examples/rocket-inference/Dockerfile`, `docker/build.sh` | one Dockerfile for both architectures: `docker build -f examples/rocket-inference/Dockerfile .` on aarch64 (NPU; needs the staged archive) or x86-64 (x86-64-v3 flex); `build.sh` builds/pushes the board image at the Forgejo registry |
| `.github/workflows/build-images.yml` | CI: `docker/build-push-action` (buildx, `docker/metadata-action` tags, per-arch `gha` cache) on native runners for every push to `main`/`v*` tag → GHCR `ghcr.io/royalcat/rocket-inference` (aarch64 `:sha`+`:latest` primary, amd64 `:sha-amd64`+`:latest-amd64`); PRs build without pushing |
| `.cargo/config.toml` | aarch64 linker + `target-feature=+fp16` |

## Gemma family (`examples/rocket-inference/src/gemma/`)

Full multimodal inference for `google/embeddinggemma-2` (Gemma 4 towers): text,
image, video and audio embeddings, plus an OpenAI-compatible server
(`/v1/embeddings` + native multimodal `/embed`). Verified against the HF f32
reference (transformers 5.19) — see
`examples/rocket-inference/docs/experiment-log-gemma.md`: all modalities
cosine 1.00000000 (JPEG decode via mozjpeg = libjpeg-turbo parity; the resize
is a bit-exact port of ATen's antialiased bicubic uint8 kernel). `--quant q8`
keeps Q8-resident projections (1216 MiB vs 2845 MiB resident) at 0.9996-0.9999
cosine; f16 is numerically broken (rejected). Dev-host throughput: 2587-token
text 8.0 s (~322 tok/s), image 0.70 s, 5 s audio 0.34 s.

Build: `cargo build --release -p rocket-inference --no-default-features`
(flex; the same binary as the Qwen families — `gemma embed|bench|tokenize|serve`).
Model `/mnt/hub/models/embeddinggemma-2`, reference venv
`/mnt/hub/venvs/emb2` (transformers 5.19 + sentence-transformers 6.1,
torch/torchvision CPU wheels). Board deploy at `/root/rocket-inference/`
(aarch64 `--features npu` build; model at `/root/models/embeddinggemma-2/`).

Generation round (2026-10-07, log §11): `gemma gen` and the served chat endpoints run
`google/gemma-4-E2B-it` (10.25 GB BF16, `/mnt/hub/models/gemma-4-E2B-it`) with a
causal Gemma 4 decoder: KV sharing (layers 15-34 reuse layer 13/14 K/V, double-
wide MLPs), PLE with the token table, proportional p-RoPE on full layers,
soft-capped tied LM head, `<|turn>` chat template. Greedy output is
**token-identical to HF 5.19** on a short question, a system+user prompt, a
multi-turn chat and a 631-token prompt; the template renders token-exactly.
Loading streams each tensor to its final dtype (f16 tables, f32 projections by
default; `--f16` = 8.9 GiB/3.5 tok/s decode, `--quant q8` = 7.3 GiB but 0.09
tok/s because flex has no int8 GEMM and `lin` dequantizes per call). f32 =
12.5 GiB/2.8 tok/s and is the parity mode; the served chat endpoints are
non-streaming OpenAI `/v1/chat/completions` and the Ollama API (text +
`image_url`/`input_audio` parts on the OpenAI side).

QAT mobile checkpoint (log §11, memory round log-gemma §13): `google/gemma-4-E2B-it-qat-mobile-transformers`
(2.46 GB) is supported natively — packed INT2/4/8 weights + per-channel scales +
SRQ activation rounding, unpacked by a load adapter (`src/gemma/qat.rs`), with the SRQ
scales registered per weight `ParamId` and applied in `lin()` (ties-to-even
rounding). Both token tables stay packed (rows dequantized on lookup, bit-identical
values) and are file-backed views of the mapped checkpoint: the PLE table (1.13 GiB
vs 8.75 GiB f32) and the 2-bit `embed_tokens` table, from which the transposed LM
head is built once (the only full-size table copy). The table stubs replace the
parameter directly (`Param::from_tensor`) — `Param::map` forced the lazy f32
initializer and wrote an 8.75 GiB table before dropping it, which was the real
load-time peak and ~30 s of the load. `--defer-towers` skips the
vision/audio towers at load and installs them (always f32, the validated
multimodal mode) on the first media request; `lin()` checks the weight dtype
before the f16/bf16 cast path, and the towers' SRQ scales are registered in the
towers pass (they sit behind a `#[module(skip)]` field). Weights are bit-exact
vs HF; short prompts are token-identical, the 631-token prompt is 16/16 with
`--f16` vs the HF bf16 reference. Text-only profile (dev host, peak `RssAnon`
over the whole run): `--f16 --defer-towers` 4.7 GiB / 5.2 tok/s (loader-end 3.7
GiB, load 22 s; was 6630 MiB at loader end), +1.9 GiB once towers load; f32
9.2 GiB; q8 3.7 GiB (memory-only). SRQ makes long-context near-ties
sensitive (a full-quantum activation jump from f32 accumulation order);
`NO_SRQ=1` / `DUMP_PARAM=<substr>` are debug hooks. `ROCKET_NPU_DECODE=1` routes
decode matmuls to the packed NPU weights (pre-gate for the pack-and-drop
config). Board attempt: deployed at
`/root/models/gemma-4-E2B-it-qat-mobile/`, OOM-killed at 7.6 GB (board busy);
a guarded re-run with `--defer-towers` is pending.

NPU prefill (log §11): `--npu` packs the used text projections into resident
fp16 NPU weights and runs prefill matmuls + `attention_causal_window` on the
NPU, while decode keeps the CPU f32 copies (`layers::set_prefill_mode` wraps
only `GenRoot::prefill`; NPU matmuls pad M to 256). Mask builders live in
`burn-rocket/src/masks.rs` with host unit tests (`cargo test -p burn-rocket`).
Board measurement pending an idle board: the model is deployed at
`/root/models/gemma-4-E2B-it/` and the merged aarch64 npu binary at
`/root/rocket-inference/` (the old `/root/rocket-inference-gemma/` artifact is
legacy), but rock-5b-plus holds ~11 GB with other
workloads and a cgroup-guarded attempt was OOM-killed during load (2.8 GB of
the ~9.5 GB f16 working set; only ~4.5 GB available). Not yet done: streaming,
video input.

NPU round (2026-10-07, log §9): the `npu` feature packs all 218 text projections
into resident fp16 NPU weights (0.25 GiB; registered by `ParamId`, `lin()`
routes them to `burn_rocket::matmul`) and runs attention on the NPU through the
new `burn_rocket::attention_window` band-mask op. `--npu` combines with
`--quant q8` (validated deployment config). Board (4 A76 threads, 2587-token
text, busy board): q8 CPU 34.5 s / 94 s user CPU -> q8+npu 21.4 s / 63 s
(1.61x faster, 33 % less CPU); numerics vs HF f32 0.9996-0.9999 (text/image/
audio), 0.9992 (video). Not yet done: f32 `--npu` on the board, NPU offload for
the vision/audio towers, 8k-context NPU attention memory.

## GGUF checkpoints + `-hf` (2026-10-12)

`rocket-inference` loads **GGUF** checkpoints as well as safetensors, through a shared
`Checkpoint` trait (`src/util/checkpoint.rs`): `SafetensorsCheckpoint`,
`GgufCheckpoint` and the erased `AnyCheckpoint`. Tensors are exposed in the
HF/PyTorch convention, so the same `PyTorchToBurnAdapter`-based chains apply to both
formats; `pack_tensors()` + `AnyCheckpoint::load_into_with(adapter)` let a family
supply its own adapter (gemma4's per-tensor `LoadDtypeAdapter`).

- **Reader** (`src/util/gguf.rs`): `gguf-rs` parses metadata/tensor descriptors and we
  mmap the file for zero-copy bytes; Q8_0/F16/BF16/F32/F64/I32 dequantize to f32.
  NB `gguf-rs`'s `Tensor::size` is wrong for unquantized types above kind 9 (it
  divides by a 256 block size) — `byte_len` computes the length instead. GGUF and HF
  share the element memory order, so only shape labels reverse (no permutation).
- **Config**: a sibling `config.json`, else the base repo's (fetched via hf-hub;
  dense Gemma configs carry per-layer overrides GGUF metadata lacks), else
  synthesized from GGUF metadata.
- **Tokenizer** (`src/util/gguf_tokenizer.rs`): built from GGUF metadata — byte-level
  BPE (Qwen), merge-based BPE for `gemma4` (`tokens` + `merges` over space-to-`▁` text
  with `byte_fallback`/`fuse_unk`), and score-based SPM/unigram for real SPM (`llama`).
  `gemma4` `scores` are placeholder constants, so feeding them to Unigram picks the
  fewest pieces and mis-segments compounds (`Dokumenten` became `Do|kum|enten`; fixed
  2026-10-17); byte-fallback `<0xXX>` tokens stay out of the added-special set (the
  reference `added_tokens` excludes them). A/B: `tools/tokenizer_parity.py` —
  token-identical to the reference `tokenizer.json` and `llama-tokenize`; a related
  checkpoint's file (same vocab/merges, different BOS/EOS policy — emb2 adds both, the
  gemma-4-*-it chat models neither) reports as `OK~` (segments match).
- **Key maps** in `checkpoint.rs`: per-family GGUF→HF tables (Qwen3/Qwen3.5,
  EmbeddingGemma 2, Gemma 4) plus the `clip` mmproj map for the towers.
- **Semantic conversions** (GGUF's convention differs from HF's):
  - qwen3.5 `ssm_a` = `-exp(A_log)` → `A_log = ln(-ssm_a)`;
  - qwen3.5 transformer RMSNorm weights carry a `+1` unit offset (the SSM norm does not);
  - audio `per_dim_scale` = `softplus(per_dim_scale)`;
  - the vision patch embed flattens channels-last (a real `(out,c,h,w)→(out,h,w,c)` permute);
  - depthwise conv `[k,C] → [C,1,k]`.
- **mmproj towers**: `mmproj-*.gguf` (a `clip` file) loads the vision/audio towers with
  their per-linear clip bounds; EmbeddingGemma 2 and Gemma 4. Gemma 4 text stays
  safetensors/QAT-capable.
- **`-hf`**: llama.cpp-style `-hf <owner>/<name>:<quant>` (quant required) resolves and
  downloads the main file plus a matching `mmproj-<quant>.gguf`/`mmproj.gguf`
  (`--mmproj`/`--revision` override) via `hf-hub` 1.0, then injects `--model-dir`
  (still supported for local dirs). The file guess strips a trailing `-GGUF` from the
  repo name (`X-GGUF` → `X-Q8_0.gguf`).

Verified (GGUF vs the safetensors path, dev host): qwen3 Q8 cosine 0.9993;
EmbeddingGemma 2 Q8 text 0.9990-0.9998 (short 0.9990 .. German paragraphs 0.9998,
re-measured 2026-10-16 after the tokenizer fix; against the F16 llama.cpp reference
every text now sits at 0.9995-0.9998, where the pre-fix GGUF path had drifted to
0.984-0.996 on German text), image 0.9996 / audio 0.9995 / video 0.9998 (mmproj);
Gemma 4 Q8 text token-identical, image/audio coherent (mmproj); Qwen3.5 Q8 coherent
chat. `-hf` tested end to end (text and mmproj image).

Known gap: Gemma 4 mmproj audio soft-token count differs by 1 (142 vs 141) although the
output is coherent — TODO.

## Rebuild + deploy to the board

```sh
# NPU build of the example (pure Rust; no archive/C toolchain)
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu
ssh root@rock-5b-plus.lan 'mkdir -p /root/rocket-inference'
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/rocket-inference \
    root@rock-5b-plus.lan:/root/rocket-inference/
# the bench needs data/ relative to the deploy dir
scp -r examples/rocket-inference/data root@rock-5b-plus.lan:/root/rocket-inference/
# CPU-only build: drop --features npu (binary name is the same)
```

Gemma family (same binary; the model is ~1.5 GB, copy it once):

```sh
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu
ssh root@rock-5b-plus.lan 'mkdir -p /root/models/embeddinggemma-2'
scp /mnt/hub/models/embeddinggemma-2/{model.safetensors,config.json,tokenizer.json,tokenizer.model,\
tokenizer_config.json,preprocessor_config.json,processor_config.json,config_sentence_transformers.json} \
    root@rock-5b-plus.lan:/root/models/embeddinggemma-2/
# board run (lowest-memory config; the board must be idle for meaningful A/B numbers)
taskset -c 4-7 /root/rocket-inference/rocket-inference gemma bench \
    --model-dir /root/models/embeddinggemma-2 --quant q8 --npu --text-file data/one_long.txt --reps 1
```

The binary is self-contained (the driver is statically linked Rust); the model lives at
`/root/models/qwen3-embedding-0.6b/` on the board. On the board run from
`/root/rocket-inference` (the bench uses the relative `data/bench_text.txt`).

Library-only builds: `cargo check -p burn-rocket` (the `npu` feature is the default)
compiles the extension on any host (no linking); `cargo build --release -p burn-rocket
--target aarch64-unknown-linux-gnu` builds the library, and `--example probe` adds the
driver probe. `--no-default-features` gives the no-Burn crate (helpers and masks only).

Container image: one Dockerfile serves both architectures, chosen by the build
platform — `docker build -f examples/rocket-inference/Dockerfile .` (context = repo
root) on an aarch64 host builds the NPU image, on x86-64 it builds the
x86-64-v3 (AVX2) flex CPU image. `examples/rocket-inference/docker/build.sh
[git-ref]` stages the tree, builds the aarch64 image on the board and pushes
`git.kmsign.org/royalcat/rocket-inference:<sha>` (needs the archive — `VENDOR_SRC`
overrides — and registry credentials on the control host; run it from anywhere in
the repo). `.github/workflows/build-images.yml` builds both on every push to `main`
and `v*` tags on native runners (buildx via `docker/build-push-action`, no
cross/QEMU) and pushes
`ghcr.io/royalcat/rocket-inference` with `docker/metadata-action` tags: aarch64 =
`<sha>` + `latest` (primary), amd64 = `<sha>-amd64` + `latest-amd64`; PRs build
without pushing. The image CMD is the unified server, identical on both architectures
(the NPU flags are inert on amd64): `serve --backend flex --dtype f32 --npu-attn cpu
--port 8383 --max-tokens 8192 --model-dir /models/qwen3-embedding-0.6b
--model-name qwen3-embedding`.

The `bench` summary prints `stages: attention/mlp/norms` and, with `--npu`, an
`npu breakdown: calls/convert/npu/flex+overhead` line — check these before profiling.
The NPU counters live in `src/ext.rs` (`burn_rocket::stats`), the stage counters in the
example's `src/qwen3_embedding/model.rs` (`stage_stats`).

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
- A real Ollama daemon listens on `localhost:11434` on the dev host — use another
  port (e.g. 11435) for `serve` smoke tests, or the responses come from Ollama.
- `serve` smoke footgun: the OpenAI chat content parts (`image_url`/`input_audio`)
  insert the `<|image|>`/`<|audio|>` placeholders themselves; putting the placeholder
  in the text part as well yields a 400 `placeholder mismatch`.

## Board deployment (2026-10-05)

- Deployed: `/root/embeddings-fast/` (pre-inversion dir; binary + `data/`) and
  `/root/models/qwen3-embedding-0.6b/`. New deploys go to `/root/rocket-inference/`;
  the old directory is left in place. Run from the deploy dir (the bench uses the
  relative `data/bench_text.txt`); pin to the A76s with `taskset -c 4-7`.
- **Live service**: OpenViking's embedding backend, container
  `home-llm-gateway-embed-1` in the `home-llm-gateway` Komodo stack
  (`/home/komodo/.komodo/stacks/home-llm-gateway/compose.yaml`, service `embed`;
  historically `openviking-embed-1` in the `openviking` stack). As observed on
  2026-10-24 it runs `ghcr.io/royalcat/rocket-inference:latest` (= `94f99bd`,
  2026-10-22, which includes the 2026-10-21 scratch-memory round, log-gemma §15)
  with `serve --backend flex --npu --npu-attn npu --attn-chunk 1024 --port 8383
  --max-tokens 8192 --defer-towers -hf unsloth/embeddinggemma-2-GGUF:Q8_0`,
  CPUs 4-7, NPU, `mem_limit` 8g. It predates the S1/S2 round (log-gemma §19/§20):
  a redeploy from `main` brings the fast host softmax to the text path
  (qwen3 2777-token −11 % wall / −26 % CPU; emb2 2594-token attention −17 %) and
  the NPU vision-attention default for media; the text-only config needs no flag
  change and the deployed service never calls `/embed`. It also predates the
  2026-10-24 ffmpeg install (log-gemma §21), so `input_video` blocks and non-WAV
  audio fail in that container until a redeploy. The previous C/GGUF image
  `a986d54` predates the 2026-10-17 `gemma4` tokenizer fix; the documents indexed
  in the GGUF-era window may need re-embedding (their vectors and queries were
  self-consistent but up to ~0.016 cosine off the reference on German text). The
  earlier idle-memory image was `b4634c9` (`--dtype f32 --quant q8 --npu
  --npu-attn npu --attn-chunk 1024 --defer-towers`, idle RssAnon 605 MiB at boot
  / ~780 MiB after long traffic). Any board run shares the NPU and cores 4-7 with
  it. Historically (2026-10-05) the backend was qwen3 on
  `git.kmsign.org/royalcat/embeddings-fast:0c99eed`, CPUs 4-7, `mem_limit` 8g,
  weights at `/root/models/qwen3-embedding-0.6b`; that image predates the repo
  inversion. `examples/rocket-inference/docker/build.sh` pushes
  `git.kmsign.org/royalcat/rocket-inference:<sha>`.
- **A/B against the live service without touching it**: port 8383 is
  container-internal (not published on the host). Run the candidate binary as a
  host process on `:8399` with the same flags and `taskset -c 4-7`, then send
  requests from inside `home-llm-gateway-embed-1` (formerly `openviking-embed-1`):
  `localhost:8383` for the live service and the compose gateway
  (`172.22.0.1:8399`, from
  `docker inspect ... .NetworkSettings.Networks[].Gateway`) for the candidate.
  Start it detached with `setsid env ... taskset -c 4-7 ./rocket-inference ... >
  serve.log 2>&1 < /dev/null &` (a plain `&` over ssh keeps the session open).
  `ROCKET_MEM_LOG=1` adds the `npu scratch` line; sample `/proc/<pid>/status`
  for peak `RssAnon` and stop it with a PID lookup (`ps -eo pid,comm,args`), not
  `pkill -f` (the pattern matches the ssh wrapper). The memory-round test binary,
  logs and request fixtures live in `/root/rocket-inference-mem/`; it is not a
  service. On this board, an 8k text request peaked at ~1.5 GiB `RssAnon` plus
  ~1.3 GiB of driver BOs and ~0.6 GiB of host scratch while deposit and swap
  were active.
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
- **2026-10-25 parity port**: for the redeploy, the app-visible knob changes and the
  DRM-BO/cgroup caution, see "C-library parity port" above. The deployed service's
  flags need no change.

## NPU offload (`--npu`, aarch64)

- `burn-rocket` (repo root) = the pure-Rust `rocket-npu` driver plus the `RocketOps`
  Burn backend extension. `npu` is the crate's **default** feature; `--no-default-features`
  gives the no-Burn crate and `--no-default-features --features flex` the Burn dependency
  without the extension (feature-graph check). `npu` implies `flex` (required because
  Burn's `#[backend_extension(Flex)]` reads `feature = "flex"` in the consuming crate) and
  `dep:rocket-npu`. Build the example with `--no-default-features --features npu`;
  aarch64 only: the dep is target-gated in `examples/rocket-inference/Cargo.toml`.
- Call `burn_rocket::init(threads)` once, then `pack`/`pack2`/`pack3` (weights ->
  `WeightId`s), `matmul` and `attention` — the model calls these directly. All ops share
  one global engine behind a mutex (the driver contexts are not thread-safe), so NPU calls
  serialize. `attention_window(q, k, v, h, kv, d, scale, softcap, window)` (added
  2026-10-07 for EmbeddingGemma 2) applies a bidirectional band mask
  (`|q - kv| <= window`; `window < 0` = no mask and allows `n_kv != n_q`,
  `MaskMode::NoneBlock` — the full-layer chunk path); `attention_window_block(..., window,
  q_start, kv_start)` (2026-10-08) runs a query chunk against a *subset* of keys
  (`n_kv != n_q`; key `kv_start + j` visible to query `q_start + i` iff
  `|(q_start + i) - (kv_start + j)| <= window`) — the sliding layers' chunked path,
  with the block mask cached on the translation-invariant offset;
  `attention_causal_block(..., q_start, kv_start)` and
  `attention_causal_block_window(..., window, q_start, kv_start)` (P2a, 2026-10-25) are the
  causal counterparts (key `kv_start + j` visible iff `kv_start + j <= q_start + i`,
  and `q_start + i - window < kv_start + j` with a window), the ops the example's
  query-chunked causal path calls (`util/causal.rs`, `ROCKET_CAUSAL_CHUNK`). All
  attention ops cache their f16 masks per geometry/shape in the engine.
- The host score pass is the **S1 fast softmax**
  (`crates/rocket-npu/src/attn/softmax.rs`): a branch-free NEON `exp2`
  polynomial (8-lane with native f16 convert on aarch64, scalar fallback),
  4.3× the old scalar pass at ≤1 f16 ULP; `SOFTMAX_EXACT=1` /
  `ROCKET_SOFTMAX_EXACT=1` restores the exact path bit-for-bit, and
  `vision_attn_bench softmax [kn] [reps] [tp]` measures it. It cut
  `attention_window` from parity to **1.99×** flex's fused CPU kernel at the
  vision tower geometry, so emb2's `--vision-attn` defaults to `npu` under
  `--npu` (S2, log-gemma §19/§20); `fused` is the CPU fallback and `blocked`
  the parity reference.
- Attention geometry + scratch (2026-10-21): `Engine.fa` is a vector of up to 4
  `FaState`s keyed by `(n_head, n_kv_heads, head_dim)`, so a model that alternates
  geometries (emb2 sliding 256/2 vs full 512/1) keeps one context per geometry
  instead of reopening fds and rebuilding scratch every layer. Each `Attention`
  worker keeps a two-entry `ScratchCache` (sliding chunk / full chunk), drops
  before allocating, and trims to the shapes touched by the last request via
  `trim_scratch`; host q/k/v/score/prob matrices shrink when the working shape
  drops more than 4x. Emb2's full layers use the same `--attn-chunk` (default
  1024): `chunk >= s` leaves a layer unchunked, so keep it below the 8k class.
  Library surface: `burn_rocket::{mem_stats, trim_scratch, clear_scratch}` plus
  the `RocketCtx`/`RocketFaCtx` methods.
- Matmul scratch policy (2026-10-21): `Context` caches per-shape scratch under
  `M` **capacity classes** (`M_CLASSES` in `matmul/context.rs`, ≤25 % padding) while
  the live `M` is planned per call, so resident weights stay valid and padded rows
  are never computed. `out_all` (CPU-accumulation) and `wt_all` (streaming) are
  lazy: resident/KACC slots never allocate them. `ROCKET_SCRATCH_MB` (default 1024,
  0 = unlimited) is a byte budget enforced with free-before-allocate LRU; each
  `trim_scratch` keeps the just-used slots and drops earlier requests' sets. A
  single request may exceed the budget (the hot set is never evicted mid-request).
  `ROCKET_M_BUCKET=0` disables classes (exact-`M` slots). Rows are independent, so
  capacity classes and trimming are numerically identical to exact-`M` slots.
- Canonical tiling (2026-10-08, default): `init` creates the context with
  `rocket_ctx_create_ex(threads, ROCKET_CTX_TILING_CANONICAL)`, so a resident weight
  serves any `M >= 4` and small requests no longer pad every matmul to 256 rows
  (`ROCKET_CTX_CANONICAL=0` restores the legacy pad for A/B; off-vs-on is
  bit-identical). The int8 ctx is always canonical.
- Resident int8 (W8A8, emb2 `--npu-int8`): `pack_i8(t, group)` quantizes host-side
  (symmetric int8, `group % 32`, weight scale per output channel per K-group, layout
  `[N, K/group]` — matches the library's `rocket_prepacked_int8.c`) and scatters the
  codes into NPU BOs once; `matmul` routes by `WeightId` to
  `rocket_matmul_int8_prepacked_gw`, quantizing A per row per group and padding rows
  to `M % 4` (an unaligned M miscomputes on HW — the library rejects it). The int8
  ctx owns its own fds (created lazily at the first int8 pack).
- Fused CPU glue (2026-10-08 emb2, extended to qwen3 2026-10-09):
  `burn_rocket::{gelu_mul, silu_mul, rms_norm, rms_norm_noscale, rope_apply}` are host
  rayon kernels (single pass where flex runs several serial scalar passes); the NPU
  loaders turn them on (`util/glue.rs`, shared by emb2/gemma4/qwen3), `ROCKET_GLUE=0`
  restores the composite ops, `ROCKET_GLUE=1` forces the kernels in CPU modes.
- `--npu` = pack-and-drop: projections offload in the model's dtype — f32/f16 pack
  resident fp16 BOs (0.82 GiB for qwen3) and bf16 packs the same way via an exact
  bf16->f16 conversion (log §17); the embedding table keeps the model dtype (f16 for
  an f32 body), 298 MiB CPU-resident for qwen3 f32. `--npu-attn npu` (default) also
  offloads attention via `rocket_flash_attn_fp16_ctx`; `cpu` keeps flex attention
  (within ~1 s on wall, ~40% more CPU). `--npu` accepts `--dtype f32|f16|bf16` and
  excludes `--quant q8`. `serve` defaults `--npu` on for this build (`--no-npu`
  opts out; `--quant q8` also keeps the CPU path unless `--npu` is explicit).
  EmbeddingGemma 2's **vision tower** packs its linears the same way — fused
  q|k|v and gate|up groups plus o/down, **65** resident weights (113 with
  `ROCKET_NPU_FUSED=0`), 0.28 GiB fp16 — from the emb2 loader, `ROCKET_NPU_TOWERS=0`
  off; it packs the loaded **f32** weights (the towers' Q8 pass excludes the
  packed part, `tower_quant_group`), so the audio tower and clipped linears
  stay on the CPU (log-gemma §18.3/§18.5).
- With canonical tiling (default) resident weights serve any `M >= 4`; only the
  `M % 4` alignment pad remains. `ROCKET_CTX_CANONICAL=0` restores the legacy
  `M >= 256` pad (A/B only). One pack serves all lengths.
- Failures panic with a structured `OpFailure { error, m, k, n }` payload (detail logged
  by `op_failure` before the panic). `serve` (via `src/server/error.rs`) maps
  `ROCKET_E_NOMEM` -> 503, shape/tiling -> 500, device/unsupported -> log + `exit(1)`
  for a supervisor restart. Matmuls chunk above
  `ROCKET_MATMUL_CHUNK_M` rows (env, default 8192, 0 disables) so the per-call input-BO
  scratch stays bounded; rows are independent, so chunking is bit-identical.
- Failures/logs aside, a panic in an op can no longer poison the server's model lock
  (poison recovery + `catch_unwind`; lock recovery lives in the example's
  `src/util/http.rs`, containment in each family's server, log §12).
- Board: the 600 MHz patched module (`insmod /root/npu-poc/rocket-patched-600/rocket-npu600.ko
  rocket_npu_clk_hz=600000000` after `rmmod rocket`; contained, reboot reverts) is ~3x the
  stock 200 MHz boot clock. The driver is the pure-Rust `crates/rocket-npu`
  (GPL-3.0-or-later); the board's `/root/npu-poc/rocket-userspace` tree is only
  historical reference.
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
- The C library's float matmul ladder (for reference): fp16 resident (`rocket_weights_pack` +
  `rocket_matmul_fp16_prepacked`) — what the extension uses for f32/f16 model weights
  **and for bf16** (packed via the exact bf16->f16 conversion); bf16 (full-speed
  2-byte geometry, fp32 accumulate and NO scaling, but streaming-only: no
  resident-weight variant, so `ROCKET_BF16_STREAM=1` keeps weights host-native and
  feeds `rocket_matmul_bf16_stream` an f32 conversion per call, `_mt` on a refusal);
  and tf32 (the only 4-byte-input path, explicitly half-rate; ported as
  `rocket_npu::matmul::one_shot_tf32` and unused by the extension). bf16 has no speed
  advantage over fp16 on the NPU (the device ties; the library docs call it
  token-identical), and its stream re-packs the weight per call — 26-30 % slower
  end-to-end than the resident path (log §16), which is why bf16 weights pack
  resident fp16 by default (log §17).
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
  mask, and the full-matrix NPU attention OOMs the board at ~30k (deployment finding).
  EmbeddingGemma 2's sliding layers now avoid that entirely with the chunked banded
  path (`attention_window_block`, keys `[q0-w, q1+w)`): at 7.7k tokens it runs at
  1068 MiB anon where the full-matrix path OOM-kills inside a 3 GB cgroup (log §12.3).
  The mask + head-major f16 scratch are cached per shape in the crate's engine
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

- `burn = "=0.22.0"`, `burn-store = "=0.22.0"`; `flex` backend is the primary
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
  dequantized once per layer per forward (`linear_forward` in `src/util/proj.rs`, the
  shared `LowRam` mapper in `src/util/quant.rs`, `Qwen3Embedding::quantized` in
  `src/qwen3_embedding/model.rs`); the dequant follows the model dtype, and the
  embedding table is f16 for an f32 body (native dtype for 16-bit bodies);
  `libc::malloc_trim` after quantization releases the freed pages (glibc otherwise
  retains ~1.3 GB in arenas).
  Resident ~0.78 GB anon vs ~2.4 GB for f32, cost ~2 s per forward. Load peak is still
  ~2.3 GB (the file is materialized as f32 before quantization).
- `--dtype` (and the family equivalents) defaults to the checkpoint's native float
  dtype — bf16 for every current checkpoint; `--dtype f32` selects the f32 parity
  path. The model body runs in the resolved dtype (qwen3/emb2); gemma4 keeps its f32
  body (f32 softmax/PLE/logits) with bf16 weights. Native bf16 costs CPU speed on
  flex (the bf16 GEMM converts operands to f32 per call): qwen3 82-token bf16
  142 tok/s vs 211 f32, 3.6k 222.7 vs 268.2 (2.9x at 82 tokens, 1.2x at 3.6k); f16 is
  the slowest 16-bit mode (60.8/131.9 tok/s). On the board, bf16 weights pack the
  resident fp16 path by default, so the NPU bf16 gap is now 6.2 % for qwen3 (54.4 vs
  57.8 tok/s) and emb2 bf16 is *faster* than its f32 arm (163.4 vs 156.7) at ~60 % of
  the memory; the opt-in `ROCKET_BF16_STREAM=1` path (the library's bf16 stream) is
  26-30 % slower (log §16/§17). Numerics verified: qwen3 bf16 cosine 0.99983 vs f32
  (82 tok) and 0.999845 on the board; emb2 text/image ~0.9999 vs HF f32 and 0.999939
  on the board; gemma4 bf16 9/9 tokens identical to HF (f32 body). `--dtype f16` is
  still rejected for emb2 (f16 numerics: 0.98 text / 0.70 image).
- Tokenizer must run with `add_special_tokens=true` (appends EOS 151643), matching
  `llama-embedding` token counts.
- 16-bit weights are slower than f32 at model level on this CPU (f16 2-4x, bf16 1.2-2.9x,
  length-dependent; see the bf16 bullet) despite similar GEMM microbenchmarks — use f32.
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
  - The one global engine serializes calls (the driver contexts are not thread-safe), so NPU
    ops from several threads queue up rather than race.
- Cross-compile: `.cargo/config.toml` sets `aarch64-linux-gnu-gcc` for
  `aarch64-unknown-linux-gnu`; artifacts land in
  `$CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/`. Build with
  `--no-default-features` (the example's default `cpu` feature pulls in cubecl-llvm,
  whose prebuilt LLVM bundle is host-arch and fails to link for aarch64; `flex` needs
  none of it).

## Future work

- **RK3576 entries**: the part's own conv encoder (`rocket_conv2d_int8_rk3576` and the
  depthwise/per-channel/act variants, the packed-image datapath, the prepacked weight
  handles and the cube/chain surface), its int8 pooling (a different arithmetic — the
  average rounds half to even and the pad value is the input zero point) and its LUT
  activation. A different geometry encoding from the RK3588 generators, so nothing in
  `regcmd::*` is shared.
- **`ROCKET_CONV_BATCH`**: the C's opt-in coalesced multi-task submit for the int8
  direct conv's tiles (one submit/fence per job slice instead of per tile). The port
  runs the per-tile path, which is bit-identical; wiring it needs
  `matmul::chain::pack` plus a gapped batched submit in the conv tiler.
- **Graph execution** (the RK3576-only graph runner) is the remaining op family of the
  C library.
- **int8 GEMM on the flex CPU path** is the only lever that would close the qwen3
  speed gap with production. flex has none; the candidates are the `burn-cpu`
  (CubeCL/LLVM) backend (CPU quantized matmul unverified, cannot cross-compile — would
  need a native build + JIT on the board) or a custom int8 microkernel (upstream
  contribution). The NPU int8 path (emb2) is done (log §12.2), and the
  Q8_0-native int8 projection route is **measured and rejected** (P2b, log §20:
  1.6-7x slower at qwen3's shapes because `matmul_gw` reads back one int32 partial
  per 32-value K-group, ~5 GB per layer at 2777 tokens). The only surviving niche is
  a *requantized* wide-group int8 route: at group 512 the probe hits 0.74x (1.35x
  faster) on the K=4096 down projection but only 1.03x on the K=1024 ones, at +25 %
  CPU, and wider groups cost weight fidelity on real (outlier-bearing) weights — it
  would need a real-checkpoint numerics gate before anyone wires it, and it is not
  Q8_0-native.
- **Vulkan/GPU path is closed** (log §13): measured 3.8-79 GF/s with CubeCL on panvk,
  panvk compiler crashes on CubeCL SPIR-V, and the panthor job watchdog kills the
  dispatches this workload needs. Re-open only with a materially faster CubeCL/panvk
  stack.
- **RoPE on the NPU — measured and rejected** (P2c, 2026-10-25, log §21): rotate-half
  RoPE is expressible with the conv-main EW ops (2 `ew_mul` + 1 `ew_add` + a host
  gather), but the op has **no broadcast** (cos/sin replicate to the full `[S, H, D]`
  layout) and tiles at 32,640 elements per submit+fence, so a 2777-token qwen3 forward
  is 22,092 submits × ~1.2 ms = **18.8-19.2x** the fused host kernel (`rope_bench`;
  numerics were fine at cos 0.999999935, and the arm's host half alone is 1.8x the
  fused arm). No app path, no flag — do not retry without a resident-EW context *and*
  a broadcast operand, and even then the host round-trip dominates. The probe stays as
  `examples/rocket-inference/src/bin/rope_bench.rs`. Side finding, fixed in the same
  round: `--npu`'s `qkv_fused` branch called the composite `RopeCache::apply`, not the
  fused `rope_apply`; it now uses the fused kernel (bit-identical to the previous
  binary, **64.8 -> 58.8 s at 2777 tokens, 42.9 -> 47.2 tok/s, cosine 1.0**).
- **On-NPU softmax in `rocket-npu`** is no longer on the critical path: the S1
  host pass (log-gemma §19) cut the host softmax to 0.05 s/layer, so
  `attention_window` now beats flex's fused kernel 1.99× at the tower geometry
  (and 1.07× at the video-frame size) and NPU vision attention is the tower
  default under `--npu` since S2 (log-gemma §20). The C librocket NPU's
  device-side softmax is a
  transfer-bound composition that its own FA comment calls 5-10 % slower than
  the host pass, so do not port it for this — the remaining op time is GEMM +
  BO traffic. Shared with the causal models' long-context NPU attention, which
  gets the same host-pass improvement (qwen3 2777 tok: −11 % wall, −26 % CPU).
- Long-context NPU attention: emb2's sliding **and full** layers now use chunked
  attention (log-gemma §15), fixing the 8k memory blow-up (full layers were the
  remaining `[s, s]` case). The driver-side `[Tp, Kn]` score matrix is no longer
  required either: the M5b tiled path (`ROCKET_FA_TILE_KV`, opt-in, default off)
  walks the key axis in tiles at a flat footprint (see "Port status"), and since **M5c**
  it also covers MLA (`dv != dk`) geometries, which no longer fall back to the
  materialized path. The
  **caller's** additive mask is now bounded too: P2a's causal block op + the
  example's `util/causal.rs` chunk causal prefill (qwen3 `--npu`, Gemma 4 prefill) at
  `ROCKET_CAUSAL_CHUNK` (default 1024) with per-chunk `[C, q1]` masks and prefix keys,
  measured bit-identical to the dense causal op at 8k, with the chunk scratch linear in
  the key prefix (~0.9 GiB at a 30k prefix instead of the dense 1.8 GiB mask plus its
  `[n, n]` score matrices). What is left for a *completed* 30k f32 run is board headroom, not the
  attention: the model's own 30k-row activations peak at ~4.1 GiB anon, so on the busy
  board (deposit 4-5 GB resident, 2-4 GB free) it is OOM-killed at ~40 min even though
  every op fits. Re-run the gate on an idle board (≥5 GB free) or in bf16; do **not**
  guard with a memory cgroup and expect containment (DRM BOs bypass it; see the P2a
  operational note). Gemma 4 prefill at 30k also needs its own headroom: the f16
  checkpoint is 8.9 GiB (12.5 GiB f32), which the board cannot host alongside the live
  services, so that path is validated structurally + by the shared qwen3 shapes and the
  geometry-agnostic probe.
- **RoPE on the NPU — measured and rejected** (P2c, 2026-10-25, log §21): the fused
  host `rope_apply` kernel stays the baseline. The conv-main EW ops can express
  rotate-half RoPE (2 `ew_mul` + 1 `ew_add` + a host gather), but they have no
  broadcast (cos/sin replicate to `[S, H, D]`, 34 MB f16 at 2777 tokens) and tile at
  32,640 elements per submit+fence, so qwen3's 2777-token forward is 22,092 submits
  (168 EW calls, 1.1-1.3 ms each): the probe measures **18.8-19.2x** the fused kernel
  (34.35 vs 1.79 s for 28 layers, f32, 8.9x the CPU; cos 0.999999935) and the arm's
  host half alone (3.29 s) exceeds the whole fused arm. No app path, no flag; the
  permanent probe is `examples/rocket-inference/src/bin/rope_bench.rs` (the P2c gate
  matrix: 82/2777 tokens x f32/bf16, `--skip-npu` for the host arms, `--layers/--reps`).
  Side finding, fixed: `--npu`'s `qkv_fused` branch used the composite `rope.apply`
  (1.88-1.91x the fused kernel); it now calls `rope_apply`, bit-identical and
  **64.8 -> 58.8 s at 2777 tokens** (42.9 -> 47.2 tok/s).
- Board service: redeploy `home-llm-gateway-embed-1` from `main` to pick up the
  S1 fast host softmax, the S2 NPU vision default and the ffmpeg runtime (the
  deployed `:latest` is `94f99bd`, pre-S1/S2 and pre-ffmpeg), then re-measure the
  text path on an idle board (the S1 A/B: qwen3 2777-token −11 % wall / −26 %
  CPU). The scratch-memory round (log-gemma §15) is already in that image; the hot
  8k set is ~1.3 GiB BO + 0.6 GiB host and a following short request trims it to
  ~12 MiB scratch, so the ceiling is now one request's working set.
- EmbeddingGemma 2 video: the context layout, sampling and per-frame expansion
  match the reference processor (one video; `tools/video_parity.py`, log-gemma
  §21). Not implemented: multiple videos per request, `add_timestamps` (mm:ss
  frame prefixes) and `overflow_strategy=truncate`.
- QAT mobile board re-run (log-gemma §13): the f16 text-only profile is now 3.7 GiB
  anon plus ~1.2 GiB reclaimable table pages (`--defer-towers`), so a guarded run
  should fit a busy board; then the `ROCKET_NPU_DECODE=1` decode-speed pre-gate
  and, if decode stays viable, the `keep_cpu=false` pack-and-drop config.
- Optional: quantize tensor-by-tensor during load to remove the ~2.3 GB load-time peak
  in `--quant q8` mode.
- Intent model: with canonical tiling the `M = 1` decode no longer pads to 256 rows,
  so re-run the CPU/NPU A/B (the 2026-10-06 numbers were taken on a busy board) and
  re-probe `--pure-npu` decode (was 1.5 tok/s when every matmul paid 256 rows); decide
  the production `query_planner` wiring (memory is ~2.06 GB anon + ~1.2 GB NPU BOs
  with f32 table).

## Verification commands

```sh
B=$CARGO_TARGET_DIR/release/rocket-inference
# library compile checks (npu is the default; no linking on the host)
cargo check -p burn-rocket
# no-Burn and flex-only variants
cargo check -p burn-rocket --no-default-features
cargo check -p burn-rocket --no-default-features --features flex
# driver unit tests (planner, capacity classes, emitters, layout)
cargo test -p rocket-npu
# dtype parity vs the C oracle: plan + every regcmd word, all six dtypes, plus the
# PPU pool section (20 decomposition plans + 19 reciprocals + 22 pool programs, the
# int8/uint8 geometries included), the conv section (23 plans + 18 direct/depthwise/
# deconv programs + 6 fused-act programs) and the int8conv section (11 direct + 9
# depthwise int8 programs over a shape/zero-point/scale matrix)
cargo build -p rocket-npu --bin regcmd_dump
python3 tools/c-oracle/parity.py
# ... or just one section:
python3 tools/c-oracle/parity.py --kind pool
python3 tools/c-oracle/parity.py --kind conv --verbose
python3 tools/c-oracle/parity.py --kind int8conv --verbose
# activation/elementwise: LUT + epilogue and program byte-parity vs the C builder
# (a throwaway C dumper in /tmp links build-host/librocketnpu.a and prints
# `rocket_lut_epilogue_build` for kinds 0..=18 plus `gen_lut_activation_fp16` for a
# fixed n=1024/TANH geometry; the same knobs must be unset on both sides)
cargo run -p rocket-npu --bin activation_selftest -- --dump-lut > /tmp/opencode/rust-lut.txt
cargo run -p rocket-npu --bin activation_selftest -- --dump-gen > /tmp/opencode/rust-gen.txt
diff /tmp/opencode/c-lut.txt /tmp/opencode/rust-lut.txt     # BYTE-IDENTICAL
diff /tmp/opencode/c-gen.txt /tmp/opencode/rust-gen.txt     # BYTE-IDENTICAL (1070 words)
# clippy must be clean on both targets (the driver has aarch64-only device paths)
cargo clippy -p rocket-npu --all-targets -- -D warnings
cargo clippy -p rocket-npu --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
# reduce + softmax on the board (cores 0-3): pools, feature reduce, cumsum and the
# softmax family vs the fp64 references; the wide LM-head shape runs only with headroom
cargo build --release -p rocket-npu --target aarch64-unknown-linux-gnu --bin reduce_selftest
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/reduce_selftest \
    root@rock-5b-plus.lan:/root/rocket-m6b/
taskset -c 0-3 ./reduce_selftest --all --reps 2
taskset -c 0-3 ./reduce_selftest --pool --shape 5 17 13      # the host-fallback shape
# C-vs-Rust A/B for the same families: ab_board.c links librocketnpu natively on the
# board (a cross-built C binary reads an empty driver name and cannot open the device),
# and both sides consume the same raw files via `reduce_selftest --io`.
#   scp ref-src.tgz (include src cmake graph tests CMakeLists.txt) + ab_board.c + ab_compare.py
ssh root@rock-5b-plus.lan 'cd /root/rocket-m6b && tar xzf ref-src.tgz -C ref &&
  cmake -S ref -B ref/build -DCMAKE_BUILD_TYPE=Release &&
  cmake --build ref/build --target rocketnpu -j4 &&
  gcc -O2 -I ref/include -I ref ab_board.c ref/build/librocketnpu.a -lm -lpthread -o ab_board'
taskset -c 0-3 python3 ab_compare.py --rust ./reduce_selftest --c ./ab_board
# the C library's own gates on the same board (they must PASS for the A/B to mean anything)
cmake --build ref/build --target reduce_mean_rocket softmax_rocket -j4
taskset -c 0-3 ref/build/reduce_mean_rocket && taskset -c 0-3 ref/build/softmax_rocket
# norms + vision norms + FFN + encoder on the board (cores 0-3): every case vs the
# device-free fp64 reference (--norm 4x256/64x512/257x1024, --vision 2x8x49/2x16x196,
# --ffn 4x256x1024/64x512x2048, --encoder T=8 d=64 h=2 ff=128 and T=64 d=128 h=4 ff=256)
cargo build --release -p rocket-npu --target aarch64-unknown-linux-gnu --bin ops_selftest
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/ops_selftest \
    root@rock-5b-plus.lan:/root/rocket-m7/
taskset -c 0-3 ./ops_selftest --all
taskset -c 0-3 ./ops_selftest --norm --norm-shape 4 256   # one family / one shape
taskset -c 0-3 ./ops_selftest --encoder --encoder-shape 8 64 2 128
# C-vs-Rust A/B for the same families: the M7 sections of ab_board.c use the same
# argument layout as `ops_selftest --io` (`-` is the C NULL), so the same script drives
# both binaries. --ops adds them; the norms/vision/geglu cases require BIT-IDENTITY,
# the matmul-bearing ones a cosine/max-abs pair (the exact count is printed either way).
# The encoder block is the one expected non-exact case: its attention sublayer is
# attn::mha_self, whose host fast softmax is the documented M5a deviation.
gcc -O2 -I ref/include -I ref ab_board.c ref/build/librocketnpu.a -lm -lpthread -o ab_board
taskset -c 0-3 python3 ab_compare.py --rust ./reduce_selftest --ops ./ops_selftest \
    --c ./ab_board --no-wide --work /tmp/ab_m7
# fp16 convolution family on the board (cores 0-3): direct, depthwise, conv1d, the
# fused activations and both transposed-conv routes vs the fp32-accumulate references
cargo build --release -p rocket-npu --target aarch64-unknown-linux-gnu --bin conv_selftest
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/conv_selftest \
    root@rock-5b-plus.lan:/root/rocket-m8a/
taskset -c 0-3 ./conv_selftest --all --reps 2
taskset -c 0-3 ./conv_selftest --shape 32 112 112 32 3 3 1 1 1 1 1 1   # one direct case
taskset -c 0-3 ./conv_selftest --transpose --reps 3                   # both routes
taskset -c 0-3 ./conv_selftest --int8 --int8-mt --int8-dw --pool --resize  # M8b families
# one case each through the raw-file A/B layout (ab_board.c shares it):
taskset -c 0-3 ./conv_selftest --io conv_int8 32 16 16 32 3 3 1 1 1 1 1 1 IN W OUT
taskset -c 0-3 ./conv_selftest --io pool_int8 1 32 8 8 2 2 2 2 0 0 0 0 IN OUT
taskset -c 0-3 ./conv_selftest --io resize_bilinear 32 8 8 2 2 IN OUT
# C-vs-Rust A/B for the same family: `conv_selftest --io` and ab_board.c share the
# raw-file layout (IN, W and OUT are fp16; W is [OC][IC][KH][KW] / [C][KH][KW] / [IC][OC][KH][KW])
taskset -c 0-3 python3 ab_compare.py --rust ./reduce_selftest --ops ./ops_selftest \
    --conv ./conv_selftest --c ./ab_board --work /tmp/ab_m8a --no-wide
# ... which also drives the M8b families (int8 conv/dw, the three pools, both
# upscales); its M8b section is `m8b()` in ab_compare.py, and the C side is the same
# natively-built ab_board (the new families need no extra C library target).
# the C library's own conv gates on the same board (they must PASS for the A/B to mean
# anything; conv2d_fp16_rocket covers the direct + depthwise sweeps, conv_transpose the
# 21 shapes / 13 deconv-route cases, conv_act the fusion's accuracy + QUIRK 1 counts)
cmake --build ref/build --target conv2d_fp16_rocket conv1d_rocket conv_transpose_rocket \
    conv_act_rocket -j4
for t in conv2d_fp16_rocket conv1d_rocket conv_transpose_rocket conv_act_rocket; do
    taskset -c 0-3 ./ref/build/$t || echo "C gate $t FAILED"
done
# M9 model encoders (SAM ViT-Det + SigLIP). The synthetic fixture drives the 3-way gate:
# the Rust host reference vs the C host reference (bit-identical, no NPU needed), then
# the one-shot and resident arms vs it (cosine >= 0.9999 per layer + the neck/final).
cargo build --release -p rocket-npu --target aarch64-unknown-linux-gnu --bin encoders_selftest
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/encoders_selftest \
    tools/c-oracle/ab_board.c tools/c-oracle/ab_compare.py root@rock-5b-plus.lan:/root/rocket-m9/
# the C side links librocketnpu built NATIVELY on the board (a cross-built one cannot
# open the device; `sam_ref`/`siglip_ref` alone work on x86 too)
ssh root@rock-5b-plus.lan 'cd /root/rocket-m9 && gcc -O2 -ffp-contract=off \
    -I /root/rocket-m6a/ref/include -I /root/rocket-m6a/ref ab_board.c \
    /root/rocket-m6a/ref/build/librocketnpu.a -lm -lpthread -o ab_board'
ssh root@rock-5b-plus.lan 'cd /root/rocket-m9 && taskset -c 0-3 ./encoders_selftest --write-blob .'
ssh root@rock-5b-plus.lan 'cd /root/rocket-m9 && taskset -c 0-3 python3 ab_compare.py --enc-only \
    --rust ./encoders_selftest --enc ./encoders_selftest --c ./ab_board --work /tmp/m9ab'
# the same fixture blobs drive the C `fd < 0` reference on the DEV HOST (the host
# librocketnpu needs no device for sam_ref/siglip_ref):
gcc -O2 -ffp-contract=off -I /tmp/opencode/ref-rocket-userspace/include \
    -I /tmp/opencode/ref-rocket-userspace tools/c-oracle/ab_board.c \
    /tmp/opencode/ref-rocket-userspace/build-host/librocketnpu.a -lm -lpthread -o /tmp/opencode/ab_board_host
# the per-arm cosine table (each arm is scored against the in-process host reference):
ssh root@rock-5b-plus.lan 'cd /root/rocket-m9 && taskset -c 0-3 ./encoders_selftest sam --ref --npu --ctx 3 --reps 2'
ssh root@rock-5b-plus.lan 'cd /root/rocket-m9 && taskset -c 0-3 ./encoders_selftest siglip --ref --npu --ctx 3 --reps 2'
# real checkpoints: extract once on the dev host in the emb2 venv (blobs are 184/173 MB),
# then prove the Rust and C host references agree bit for bit and load the reference on
# the board with --ref-in (the naive host matmul is ~20 min at SAM's 1024px):
/mnt/hub/venvs/emb2/bin/python ref/tools/sam_extract.py --model facebook/sam-vit-base --out /tmp/opencode/m9-art/sam
/mnt/hub/venvs/emb2/bin/python ref/tools/siglip_extract.py \
    --model HuggingFaceTB/SmolVLM-256M-Instruct --out /tmp/opencode/m9-art/siglip/siglip_weights.f16
/mnt/hub/venvs/emb2/bin/python ref/tools/siglip_reference.py --out /tmp/opencode/m9-art/siglip
$CARGO_TARGET_DIR/release/encoders_selftest --io sam_ref sam.f16 sam_pixels.raw rust_sam_ref.out
/tmp/opencode/ab_board_host sam_ref sam.f16 sam_pixels.raw c_sam_ref.out
cmp rust_sam_ref.out c_sam_ref.out && cmp rust_sam_ref.out.hidden c_sam_ref.out.hidden
scp rust_sam_ref.out rust_sam_ref.out.hidden sam.f16 sam_pixels.raw root@rock-5b-plus.lan:/root/rocket-m9/
ssh root@rock-5b-plus.lan 'cd /root/rocket-m9 && taskset -c 0-3 ./encoders_selftest sam \
    --blob sam.f16 --pixels sam_pixels.raw --ref-in rust_sam_ref.out --npu --ctx 3 --reps 1'
# the C library's own gates on the same artifacts (they must PASS for the above to mean
# anything; sam_rocket needs {sam_weights.f16,pixels,hidden,neck}.f32 in its dir):
cmake --build ref/build --target sam_rocket siglip_rocket -j2
taskset -c 0-3 ref/build/sam_rocket /root/rocket-m9/sam-art
taskset -c 0-3 ref/build/siglip_rocket /root/rocket-m9/siglip-art
# vision-tower attention/softmax probe (separate aarch64 npu bin; run on the board)
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu --bin vision_attn_bench
./vision_attn_bench 2394 5              # fused CPU vs NPU attention, chunk sweep
./vision_attn_bench softmax 2394 5      # exact vs fast host softmax (S1)
SOFTMAX_EXACT=1 ./vision_attn_bench 2394 5
# causal block attention probe (P2a): chunked vs dense vs the f64 oracle, plus the
# flat long-context footprint. Host side: the chunk-plan tests are device-free.
cargo test -p rocket-inference --no-default-features util::causal
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu --bin causal_attn_bench
./causal_attn_bench 512 256 2 --ref              # bit-identical + f64 ref (cos 0.999999955)
./causal_attn_bench 512 128,256 1 --window 256 --ref  # sliding layers: cos >= 0.99999999
./causal_attn_bench 8192 1024,256 1              # 8k: cos 1.0, scratch 1706 -> 1102/871 MiB
./causal_attn_bench 1024 1024 --kv 30000 1       # one 30k-context chunk: 898 MiB, 3.78 s
# ... and the model-level A/B (only when the board has >= 5 GB free; the dense arm's
# [n, n] mask does not fit at 30k, and a memory cgroup does not contain the NPU BOs):
ROCKET_CAUSAL_CHUNK=0 ./rocket-inference qwen3 embed --model-dir /root/models/qwen3-embedding-0.6b \
    --dtype f32 --npu --text-file data/long30k.txt --tokens 8000 --out /tmp/dense8k.json
./rocket-inference qwen3 embed --model-dir /root/models/qwen3-embedding-0.6b --dtype f32 --npu \
    --text-file data/long30k.txt --tokens 8000 --out /tmp/chunk8k.json
ROCKET_CAUSAL_CHUNK=512 ./rocket-inference qwen3 embed --model-dir /root/models/qwen3-embedding-0.6b \
    --dtype f32 --npu --text-file data/long30k.txt --tokens 30000 --out /tmp/chunk30k.json
# RoPE probe (P2c, log §21): fused host kernel vs composite vs the NPU EW composition;
# ~7 min for the full matrix on a busy board, the NPU arm is 18.8-19.2x the fused one
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu --bin rope_bench
./rope_bench                                      # 82 + 2777 tokens, f32 + bf16
./rope_bench --tokens 2777 --dtypes f32 --layers 2 --reps 1   # quick shape check
./rope_bench --skip-npu                           # the two host arms only
# example npu compile check (the dep is aarch64-gated, so use the target)
cargo check -p rocket-inference --target aarch64-unknown-linux-gnu --no-default-features --features npu
# formatting + lint gates (both clean)
cargo fmt --all -- --check
cargo clippy -p rocket-inference --no-default-features
# container image for this architecture (context = repo root)
docker build -f examples/rocket-inference/Dockerfile -t rocket-inference:local .
docker run --rm rocket-inference:local --help
# lint the CI workflow; actionlint catches expression bugs (e.g. the buildx
# round's `id.meta.outputs.*`, which must be `steps.meta.*`), but not runtime
# env like an undefined `env.REGISTRY`
docker run --rm -v "$PWD":/repo -w /repo rhysd/actionlint:latest -color
# single-core speed gate (blocked path is the single-core record); run from
# examples/rocket-inference
taskset -c 2 cargo run --release -- qwen3 bench --backend flex --dtype f32 --tokens 3633 --reps 2 --attn blocked --chunk 256 --key-block 256
# multi-threaded / long-input (fused default)
cargo run --release -- qwen3 bench --backend flex --dtype f32 --text-file /tmp/opencode/long30k.txt --tokens 30000 --reps 0
# embedding + cosine against a reference JSON
cargo run --release -- qwen3 embed --backend flex --dtype f32 --quant q8 --text-file /tmp/opencode/one_64.txt --out /tmp/opencode/our.json
# GGUF tokenizer parity: our `<family> tokenize` == the reference tokenizer.json ==
# llama-tokenize (built-in German regression words; --text/--text-file extend it);
# needs a python with `tokenizers` (the emb2 venv on this host)
/mnt/hub/venvs/emb2/bin/python examples/rocket-inference/tools/tokenizer_parity.py \
    --gguf /tmp/opencode/emb-compare/embeddinggemma-2-Q8_0.gguf \
    --ref /mnt/hub/models/embeddinggemma-2/tokenizer.json
# video parity (embeddinggemma): ids, sampled-frame count and cosine vs the HF f32
# reference for video-only, text+image+video and a 40 s overflow clip (log-gemma §21)
/mnt/hub/venvs/emb2/bin/python examples/rocket-inference/tools/video_parity.py \
    --bin $CARGO_TARGET_DIR/release/rocket-inference
# server smoke test (one command for every model; family auto-detected)
cargo run --release -- serve --model-dir ~/models/qwen3-embedding-0.6b --backend flex --dtype f32 --quant q8 --port 8383 &
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"hi"}'
# robustness: a panicking forward (invalid token id) must 500 and leave the server alive
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":[999999999]}'
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"still alive"}'
# OpenRouter multimodal /v1/embeddings (embeddinggemma): one embedding per
# block, N images per block, URLs/data URIs/paths, audio+video one each
curl -s localhost:8390/v1/embeddings -H 'Content-Type: application/json' -d '{
  "input": [
    {"content": [{"type": "image_url", "image_url": {"url": "/data/cat.jpeg"}}]},
    {"content": [{"type": "text", "text": "a cat"},
                 {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]}
  ],
  "dimensions": 256}'
# HF parity for a 2-image block (the reference takes several --image files;
# its text must use the same placeholder spacing as the server: one space)
/mnt/hub/venvs/emb2/bin/python examples/rocket-inference/tools/ref_embeddinggemma2.py \
    image --image data/cat.jpeg /tmp/cat_small.png --text '<|image|> <|image|>' --out /tmp/ref2.json
```

On the board, chunked-matmul numerics (forced chunks vs disabled must match bit-for-bit):

```sh
cd /root/rocket-inference
ROCKET_MATMUL_CHUNK_M=0 ./rocket-inference qwen3 embed --backend flex --dtype f32 --npu \
  --npu-attn cpu --text-file data/one_3633.txt --tokens 1300 --out /tmp/e0.json
ROCKET_MATMUL_CHUNK_M=1024 ./rocket-inference qwen3 embed --backend flex --dtype f32 --npu \
  --npu-attn cpu --text-file data/one_3633.txt --tokens 1300 --out /tmp/e1.json
python3 -c "import json,math;a=json.load(open('/tmp/e0.json'));b=json.load(open('/tmp/e1.json'));d=sum(x*y for x,y in zip(a,b));na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(x*x for x in b));print('cosine',d/(na*nb))"
```

On the board (after the cross-build + scp above; the NPU-enabled binary):

```sh
ssh root@rock-5b-plus.lan
cd /root/rocket-inference
# activation + elementwise self-test (cycles 0-3; the live service keeps 4-7)
cargo build --release -p rocket-npu --target aarch64-unknown-linux-gnu --bin activation_selftest
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/activation_selftest root@rock-5b-plus.lan:/root/rocket-m6a/
taskset -c 0-3 ./activation_selftest                  # all kinds x {1,31,4096,262144} + EW
taskset -c 0-3 ./activation_selftest --kind tanh --n 4096 --reps 1
taskset -c 0-3 ./activation_selftest --shape 64 64 64 --reps 1
taskset -c 0-3 ./activation_selftest --ew --n 262144
# C-vs-Rust A/B for one kind: write the input, run both, compare raw f16 (bit-identical)
taskset -c 0-3 ./activation_selftest --write-in silu 4096 in.raw
taskset -c 0-3 ./act_board act 3 in.raw c.raw          # C library binary, see below
taskset -c 0-3 ./activation_selftest --cross silu c.raw
# ... and for an EW op
taskset -c 0-3 ./activation_selftest --ew-in mul a.raw b.raw rust.raw
taskset -c 0-3 ./act_board ew mul a.raw b.raw c.raw     # then cmp c.raw rust.raw
# the C side: build librocketnpu natively on the board (the host cross build picks up
# the host's 32-bit __kernel_size_t and rocket_open() then reads an empty driver name,
# so a cross-built C binary cannot open the device)
scp ref-src.tgz root@rock-5b-plus.lan:/root/rocket-m6a/   # src include cmake graph tests CMakeLists.txt
ssh root@rock-5b-plus.lan 'cd /root/rocket-m6a && tar xzf ref-src.tgz -C ref &&
  cmake -S ref -B ref/build -DCMAKE_BUILD_TYPE=Release && cmake --build ref/build --target rocketnpu -j4 &&
  gcc -O2 -I ref/include -I ref act_board.c ref/build/librocketnpu.a -lm -lpthread -o act_board'
# driver dtype selftests (bare binaries, no model; any spare cores)
#   int8: --dense / --prepacked-dense are int32-exact, the gw pair matches the
#   dequantized reference (and each other bit-for-bit at a group-pinned Kt);
#   int4: --dense / --prepacked are int32-exact vs the int64 reference (kt_cap=0;
#   with --kt-cap 480 the reference saturates each Kt partial to int16 first), the
#   gw pair (--gw-one-shot G / --prepacked-gw G) matches the f64 dequantized
#   reference and each other bit-for-bit, and both resident modes re-compute a
#   weight packed at M=256 at a second M (the canonical-tiling contract);
#   int16_exact is i64-exact; tf32 matches an f64 dot product of the tf32-rounded
#   inputs (no args = the built-in shape list: tiny, medium, ragged M/N, K > Kt,
#   and the padded M == 1 shim).  Shape args are M K N (int8: M K N GROUP; the
#   int4 group is the flag argument).
#   attention (attn_selftest): the default runs the chained-vs-per-head A/B and
#   asserts bit-identity; --oneshot compares flash_attn / flash_attn_mt with
#   flash_attn_ref and asserts single-fd == mt; --mha does the same for
#   mha_self vs mha_self_ref at d = --dk * heads. --mask none|causal|window,
#   --softcap F, --mt N and --ref 0|1 pick the case (the f64 oracle auto-skips
#   above 2e9 MACs). --host-sm 0|1 sets ROCKET_ATTN_HOST_SOFTMAX for every mode
#   (0 = the on-NPU softmax, the C's mha_self default); --sm-ab instead runs BOTH
#   placements of flash_attn (single-fd and mt) and mha_self and prints each arm's
#   wall/CPU, its cosine/max-abs vs the f64 ref and the two arms against each other
#   (the on-NPU EXP LUT is approximate: expect cos ~0.9999999, max_abs ~1e-3, and
#   on the board ~2x the host wall, not bit-identity). --tile KBP switches to the
#   tiled long-context A/B
#   (ROCKET_FA_TILE_KV=KBP, --tile-min KV, default 8192): it runs the case
#   materialized and tiled, reports each arm's cosine/max-abs vs the f64 ref and
#   vs the other, its wall/CPU, the mem_stats footprint each leaves (Tp/Kn/tile,
#   bo/host), and asserts the tiled output ignores ROCKET_FA_CHAIN; --tile-only
#   skips the materialized arm (use it at 16k+ keys, where the [Tp,Kn] matrix the
#   path avoids is what would OOM); --mem adds the 3-token + trim_scratch check.
taskset -c 0-3 ./attn_selftest 257 257 8 2 --oneshot --mt 4 --mask causal --softcap 30
taskset -c 0-3 ./attn_selftest 2394 2394 12 12 --dk 64 --dv 64 --mask none --oneshot --ref 1
taskset -c 0-3 ./attn_selftest 82 82 8 8 --dk 64 --mha --mt 2
# the softmax placement: host (default) vs on-NPU, for the flash primitive and mha_self
taskset -c 0-3 ./attn_selftest 257 257 8 2 --dk 64 --sm-ab --mt 4 --mask causal --reps 3
taskset -c 0-3 ./attn_selftest 256 1024 8 2 --dk 64 --sm-ab --mt 4 --mask causal --reps 2
#   ... and both placements individually (the chain A/B and the mha fan-out must stay
#   bit-identical under either)
taskset -c 0-3 ./attn_selftest 257 257 8 2 --dk 64 --host-sm 0 --mask causal --reps 2
taskset -c 0-3 ./attn_selftest 82 82 2 2 --dk 32 --mha --host-sm 0 --reps 2
# C-vs-Rust A/B for the on-NPU softmax: ab_board grew `flash` / `mha` families with the
# same argument layout as `attn_selftest --io` (floats as f32 hex bit patterns, `-` =
# NULL). The script sets ROCKET_ATTN_HOST_SOFTMAX per side: the C `mha_self` treats the
# variable's PRESENCE as host, so its on-NPU arm is the unset one while the Rust side
# takes the value (0 = on-NPU). The on-NPU arms must be bit-identical; the host arms
# differ by ~1-2 f16 ULP (the Rust host pass keeps f32 scores, the C writes f16).
gcc -O2 -ffp-contract=off -I /root/rocket-m6a/ref/include -I /root/rocket-m6a/ref \
    ab_board.c /root/rocket-m6a/ref/build/librocketnpu.a -lm -lpthread -o ab_board
taskset -c 0-3 python3 ab_compare.py --attn-only --rust ./attn_selftest \
    --attn ./attn_selftest --c ./ab_board --work /tmp/a10
# tiled long-context A/B: small case with the f64 ref (tiling forced on), then the
# long cases (ref skipped, materialized arm skipped where it would not fit)
taskset -c 0-3 ./attn_selftest 1024 1024 8 2 --dk 64 --dv 64 --mask causal --ref 1 --tile 256 --tile-min 0
taskset -c 0-3 ./attn_selftest 4096 4096 8 2 --mask causal --tile 2048 --tile-min 0 --reps 3
taskset -c 0-3 ./attn_selftest 512 16384 8 2 --dk 64 --mask causal --tile 2048 --tile-only
# ... and the MLA (dv != dk) geometries: numerics vs the f64 ref at a small shape
# (the default 8,192 crossover keeps 2048 keys materialized, so --tile-min 0 forces
# the tiled arm; the literal `--tile 512` alone only reports that it stays
# materialized), then the footprint and its flatness in Kn
taskset -c 0-3 ./attn_selftest 64 2048 8 2 --dk 576 --dv 512 --mask causal --tile 512 --tile-min 0 --ref 1 --reps 1
taskset -c 0-3 ./attn_selftest 64 2048 8 2 --dk 576 --dv 512 --mask none --tile 512 --tile-min 0 --ref 1 --reps 1
taskset -c 0-3 ./attn_selftest 64 2048 8 2 --dk 64 --dv 32 --mask causal --tile 512 --tile-min 0 --ref 1 --reps 1
taskset -c 0-3 ./attn_selftest 64 2048 8 2 --dk 32 --dv 64 --mask causal --tile 512 --tile-min 0 --ref 1 --reps 1
taskset -c 0-3 ./attn_selftest 512 8192 8 2 --dk 576 --dv 512 --mask causal --tile 2048 --tile-min 0 --reps 2 --mem
taskset -c 0-3 ./attn_selftest 512 16384 8 2 --dk 576 --dv 512 --mask causal --tile 2048 --tile-only --reps 2
taskset -c 0-3 ./int8_selftest --dense 264 320 288
taskset -c 0-3 ./int8_selftest --gw-one-shot 512 1024 512 32
taskset -c 0-3 ./int4_selftest 64 256 128
taskset -c 0-3 ./int4_selftest --gw-one-shot 32 512 1024 512
taskset -c 0-3 ./int4_selftest --prepacked-gw 64 264 320 320
taskset -c 0-3 ./int16_selftest 64 256 128
taskset -c 0-3 ./tf32_selftest
taskset -c 0-3 ./tf32_selftest 512 1024 512
# Q8_0-native projection probe (P2b, log §20): resident f16 vs group-wise int8 at
# qwen3's shapes, activation quantize included; the int8/f16 ratio is the verdict
cargo build --release -p rocket-npu --target aarch64-unknown-linux-gnu --bin q8_bench
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/q8_bench root@rock-5b-plus.lan:/root/rocket-p2b/
taskset -c 0-3 ./q8_bench --reps 3                    # the five qwen3 shapes, group 32
taskset -c 0-3 ./q8_bench --reps 3 --rows 128 --groups 32,128,512,1024 2777 1024 3072 2777 4096 1024
# speed / CPU-relief A/B (1 warmup + 1 measured run; watch the stages + npu breakdown
# lines and `time`). Only meaningful on an idle board.
taskset -c 4-7 ./rocket-inference qwen3 bench --backend flex --dtype f32 --npu --tokens 3633 --reps 1
taskset -c 4-7 ./rocket-inference qwen3 bench --backend flex --dtype f32 --npu --npu-attn cpu --tokens 3633 --reps 1
taskset -c 4-7 ./rocket-inference qwen3 bench --backend flex --dtype f32 --tokens 3633 --reps 1
# numerics vs the production reference (copy the JSON back and compare cosines)
./rocket-inference qwen3 embed --backend flex --dtype f32 --npu --text-file data/one_64.txt --out /tmp/npu_64.json
# CPU modes of the same NPU binary (projections are loaded explicitly there)
./rocket-inference qwen3 embed --backend flex --dtype f32 --quant q8 --text-file data/one_64.txt --out /tmp/q8_64.json
./rocket-inference qwen3 embed --backend flex --dtype f32 --text-file data/one_64.txt --out /tmp/f32_64.json
# tiled long-context attention on the model path (env knob, no flag): compare the
# embedding and the wall against the default materialized run (log §19: parity at
# 3.6k, cosine 0.999994) and watch ROCKET_MEM_LOG's attention BO/host
ROCKET_FA_TILE_KV=2048 ROCKET_FA_TILE_MIN_KV=0 ./rocket-inference qwen3 embed --backend flex \
  --dtype f32 --npu --text-file data/long8k.txt --tokens 3600 --out /tmp/tile_3633.json
# intent model (model at /root/models/ov-intent-analysis-sft)
./rocket-inference intent gen --model-dir /root/models/ov-intent-analysis-sft --text "Hello!" --raw --max-new-tokens 32
# Gemma family (models at /root/models/embeddinggemma-2, /root/models/gemma-4-E2B-it)
./rocket-inference gemma bench --model-dir /root/models/embeddinggemma-2 --quant q8 --npu --text-file data/one_long.txt --reps 1
# vision attention: NPU default vs the CPU fallback (emb2, --npu)
./rocket-inference gemma embed --model-dir /root/models/embeddinggemma-2 --quant q8 --npu \
    --image data/cat.jpeg --out /tmp/vnpu.json
./rocket-inference gemma embed --model-dir /root/models/embeddinggemma-2 --quant q8 --npu \
    --vision-attn fused --image data/cat.jpeg --out /tmp/vfused.json
# performance-round A/Bs (log §12; validate.sh runs the whole set)
ROCKET_CTX_CANONICAL=0 ./rocket-inference gemma embed --model-dir /root/models/embeddinggemma-2 --quant q8 --npu --prompt query --text-file data/one_long.txt --out /tmp/off.json
./rocket-inference gemma embed --model-dir /root/models/embeddinggemma-2 --npu --npu-int8 --prompt query --text-file data/one_long.txt --out /tmp/i8.json
ROCKET_GLUE=0 ./rocket-inference gemma bench --model-dir /root/models/embeddinggemma-2 --quant q8 --text-file data/one_long.txt --reps 1
./rocket-inference gemma gen --gen-model-dir /root/models/gemma-4-E2B-it --text "What is the capital of France?" --max-new-tokens 16
# QAT mobile, busy-board low-memory config (add --defer-towers; towers load on media)
./rocket-inference gemma gen --gen-model-dir /root/models/gemma-4-E2B-it-qat-mobile --f16 --defer-towers --text "What is the capital of France?" --max-new-tokens 16
# servers (detected per model; ports are examples)
./rocket-inference serve --model-dir /root/models/qwen3-embedding-0.6b --quant q8 --npu --port 8383 --max-tokens 8192
./rocket-inference serve --model-dir /root/models/ov-intent-analysis-sft --npu --npu-threads 3 --port 11434 --max-new-tokens 256
./rocket-inference serve --model-dir /root/models/embeddinggemma-2 --quant q8 --npu --port 8390
./rocket-inference serve --model-dir /root/models/gemma-4-E2B-it --port 8391
# scratch attribution / policy: one `npu scratch` line per request; a 3-token
# request after an 8k one should drop the matmul/attention BOs to ~12 MiB
ROCKET_MEM_LOG=1 ROCKET_SCRATCH_MB=1024 ./rocket-inference serve --model-dir /root/models/embeddinggemma-2 --quant q8 --npu --port 8399
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
