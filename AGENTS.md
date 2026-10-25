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
  and nibble-cube pack/gather, `int16_exact`, the native bf16 stream
  and masked GQA attention are ported
  (`int8_selftest --dense|--gw-one-shot|--prepacked-dense|--prepacked-gw`,
  `int4_selftest --dense|--gw-one-shot G|--prepacked|--prepacked-gw G` and
  `int16_selftest` are the board checks: int32/i64-exact vs the CPU
  reference on ragged shapes, one-shot gw bit-identical to the resident gw at group
  32-512 for both int8 and int4).
  Planner and regcmd output are byte-equal to the C library (dtype parity
  harness: `tools/c-oracle` + `parity.py`); fp16/bf16 cosine >= 0.99999, int8
  matches its dequantized reference and int4 is int32-exact against the int64 CPU
  reference. Board microbenchmarks: int8 512x3072x1024
  62-70 ms vs C 59-66, bf16 stream 24-27 ms vs C 22-26. Attention uses a per-worker
  head split and batches each worker's per-head QK/AV matmuls into one job
  (`ROCKET_FA_CHAIN`, default on, `ROCKET_FA_CHAIN_ELEMS` = 32M cap): bit-identical,
  2394-patch vision 1.11× and emb2 sliding 1.08×, parity on the deployed KACC-on text
  shapes, 1.2-1.4× with `ROCKET_NO_KACC=1` (M1d, 2026-10-25). The S1 fast host softmax
  cut the score pass 4.3× (log-gemma §19); EmbeddingGemma 2's vision tower runs its
  attention on the NPU by default under `--npu` (S2, §20). End-to-end: qwen3
  3,633-token `bench --npu` 59.4 s vs the C binary's 61.5 s.
  **Missing surface** (documented in `crates/rocket-npu/src/lib.rs`): tf32
  (one-shot, resident, group-wise, plans), resident bf16 (the C library has no
  variant either), `mha_self`, `*_ref` host references, `_planes`/transposed variants,
  RK3576 entries, the public `rocket_stream_*` aliases, tiled long-context attention,
  on-NPU softmax, and all non-matmul op families.
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
| `crates/rocket-npu/` | **the pure-Rust driver**: `device/` (uAPI, BO, submit, fence, sysfs, affinity), `regcmd/` (desc + the fp16/int8/int4/bf16 emitters), `matmul/` (plan, layout, bos, run, context + the int8/int4 one-shots and the resident int4 `I4Context`), `attn/` (flash attention + the fast host softmax), `hw`, `knobs`, `error`; `src/bin/` has the parity dump and board self-tests |
| `src/ext.rs` | the `RocketOps` Burn backend extension, the global NPU engine (`init`, `WeightId`, `burn_rocket::stats`) and the `Tensor`-level helpers (`pack`/`matmul`/`attention`) |
| `src/masks.rs` | host-built additive attention masks (causal, bidirectional band, causal sliding window, query-chunk band) with unit tests |
| `src/host.rs` | fused host kernels for the NPU build's CPU glue (gelu·up, RMSNorm, RoPE) with unit tests |
| `tools/c-oracle/` | host-side C oracle (planner + regcmd dump for every matmul dtype), the `parity.py` dtype parity harness, and the board A/B benches used to validate the port; build with `tools/c-oracle/build.sh` |
| `crates/rocket-npu/src/bin/` | `regcmd_dump` (planner/emitter parity), `selftest` (fp16 matmul), `attn_selftest` (attention), `int8_selftest` (`--dense`/`--prepacked-dense`/`--gw-one-shot`/`--prepacked-gw`), `int4_selftest` (`--dense`/`--prepacked`/`--gw-one-shot G`/`--prepacked-gw G`, plus `--kt-cap`), `int16_selftest`, `bf16_selftest` |
| `examples/probe.rs` | low-level driver probe (open device, pack, matmul, verify vs CPU) |
| `examples/rocket-inference/Cargo.toml` | workspace member `rocket-inference`: features `cpu` (default), `npu` (aarch64-gated path dep on the root crate), `gpu*`; `wgpu_probe` bin behind `gpu` |
| `examples/rocket-inference/src/main.rs`, `src/cli.rs` | dispatch (`serve` + `qwen3`/`intent`/`gemma`), shared flag parser and usage text |
| `examples/rocket-inference/src/server/` | one HTTP server: `cli` (flags + model loading), `engine` (family detection, capabilities, per-model compute), `error` (panic/NPU mapping), `log` (per-request speed lines), `openai` (`/v1/embeddings`, `/v1/chat/completions`, `/v1/models`), `multimodal` (`/embed`), `ollama` (`/`, `/api/*`) |
| `examples/rocket-inference/src/util/` | shared infra: backend/device, `rss_mib`, server lock/panic helpers (`http`), Q8 low-RAM mapper (`quant`), `Proj`/`ProjKind`/`FusedGroup` (`proj`), `RopeCache` (`rope`), load-report checks (`store`), the shared checkpoint trait (`checkpoint`), GGUF access (`gguf`), the GGUF tokenizer builder (`gguf_tokenizer`), and `hf-hub` model resolution (`hub`) |
| `examples/rocket-inference/src/qwen3_embedding/` | Qwen3-Embedding family: model (layers, RoPE, attention paths, stage timers), loaders (f32 / `--quant q8` / `--npu` pack-and-drop + `load_cpu_projections`), CLI |
| `examples/rocket-inference/src/qwen35_intent/` | Qwen3.5-0.8B intent family: model (chunked/recurrent gated delta rule, gated full attention, caches, greedy generation), loader (`model.language_model.*`, NPU pack-and-drop, `load_for_serving`), CLI |
| `examples/rocket-inference/src/gemma/` | shared Gemma 4 building blocks: `config` schemas, `layers` (NPU/SRQ/`ClippableLinear`), `qat`, `media`, `audio` + `audio_frontend`, `vision`, `inputs` |
| `examples/rocket-inference/src/gemma/embeddinggemma/` | EmbeddingGemma 2 family: text backbone, multimodal assembly, loaders, CLI |
| `examples/rocket-inference/src/gemma/gemma4/` | Gemma 4 E2B-it family: causal decoder, loader (f32/f16/q8/QAT, `pack_text_for_prefill`), chat template + sampling (`GenStats.stopped`), CLI |
| `examples/rocket-inference/src/bin/wgpu_probe.rs` | Vulkan/wgpu GPU probe (`gpu-*` features) |
| `examples/rocket-inference/src/bin/vision_attn_bench.rs` | vision-tower attention/softmax microbenchmark (`fused-f16/bf16`, `npu`, `npu-chunk`, `matmul`, `attn-batch`, `softmax`; `npu` feature, log-gemma §18.4/§19/§20) |
| `examples/rocket-inference/data/` | bench/embedding/media fixtures (`bench_text.txt`, `cat.jpeg`, audio/video samples) |
| `examples/rocket-inference/docs/experiment-log.md` | Qwen3-Embedding + intent measurements: §1-8 dev-host, §9 board A/B, §10 NPU, §11 extension-ops refactor + CPU-loading fix, §12 serving robustness, §13 Vulkan, §14 intent model |
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
  with the block mask cached on the translation-invariant offset. Both attention ops
  cache their f16 masks per geometry/shape in the engine.
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
  and tf32 (the only 4-byte-input path, explicitly half-rate). bf16 has no speed
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

- **int8 GEMM on the flex CPU path** is the only lever that would close the qwen3
  speed gap with production. flex has none; the candidates are the `burn-cpu`
  (CubeCL/LLVM) backend (CPU quantized matmul unverified, cannot cross-compile — would
  need a native build + JIT on the board) or a custom int8 microkernel (upstream
  contribution). The NPU int8 path (emb2) is done (log §12.2).
- **Vulkan/GPU path is closed** (log §13): measured 3.8-79 GF/s with CubeCL on panvk,
  panvk compiler crashes on CubeCL SPIR-V, and the panthor job watchdog kills the
  dispatches this workload needs. Re-open only with a materially faster CubeCL/panvk
  stack.
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
  remaining `[s, s]` case). The *causal* models (qwen3, gemma4 prefill) still build
  `[n][n]` masks and OOM at ~30k; the same query-chunking idea applies with a causal
  block mask (per-chunk `[C, q1]` masks, prefix keys), plus a 30k measurement of the
  emb2 chunked path (the CPU path does 52.4 tok/s at 32 dev threads).
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
# vision-tower attention/softmax probe (separate aarch64 npu bin; run on the board)
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu --bin vision_attn_bench
./vision_attn_bench 2394 5              # fused CPU vs NPU attention, chunk sweep
./vision_attn_bench softmax 2394 5      # exact vs fast host softmax (S1)
SOFTMAX_EXACT=1 ./vision_attn_bench 2394 5
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
# driver dtype selftests (bare binaries, no model; any spare cores)
#   int8: --dense / --prepacked-dense are int32-exact, the gw pair matches the
#   dequantized reference (and each other bit-for-bit at a group-pinned Kt);
#   int4: --dense / --prepacked are int32-exact vs the int64 reference (kt_cap=0;
#   with --kt-cap 480 the reference saturates each Kt partial to int16 first), the
#   gw pair (--gw-one-shot G / --prepacked-gw G) matches the f64 dequantized
#   reference and each other bit-for-bit, and both resident modes re-compute a
#   weight packed at M=256 at a second M (the canonical-tiling contract);
#   int16_exact is i64-exact.  Shape args are M K N (int8: M K N GROUP; the int4
#   group is the flag argument).
taskset -c 0-3 ./int8_selftest --dense 264 320 288
taskset -c 0-3 ./int8_selftest --gw-one-shot 512 1024 512 32
taskset -c 0-3 ./int4_selftest 64 256 128
taskset -c 0-3 ./int4_selftest --gw-one-shot 32 512 1024 512
taskset -c 0-3 ./int4_selftest --prepacked-gw 64 264 320 320
taskset -c 0-3 ./int16_selftest 64 256 128
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
