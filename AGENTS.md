# AGENTS.md — burn-rocket

`burn-rocket` is the main crate: RK3588 NPU offload for Burn models — an FFI wrapper
over `librocketnpu` plus the `RocketOps` Burn backend extension (projections and
attention on the mainline `rocket` driver). The inference app is
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
The inversion is verified on the dev host: `cargo check -p burn-rocket` (plain / `flex` /
`npu`) and `-p rocket-inference` (default `cpu` and `--no-default-features`), aarch64
cross-builds of the example (`--features npu`) and of `--example probe`, plus a tokenize +
q8 embed smoke run. The git remotes changed on 2026-10-07: `origin` is GitHub
`royalcat/burn-rocket.git` and is the push target; the old Gitea remote is kept as
`self-hosted` (`git.kmsign.org/royalcat/embeddings-fast.git`, stale). `main` still
*tracks* `self-hosted/main`, so push explicitly with `git push origin main`. Board
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
- EmbeddingGemma 2: `--quant`, `--npu`, `--npu-threads`, `--npu-attn`, `--npu-int8`, `--attn-chunk`, `--video-fps`, `--video-max-frames`
- intent: `--npu`, `--npu-threads`, `--delta-chunk`, `--embed-f16`, `--pure-npu`, `--npu-decode`
- Gemma 4: `--f16`, `--quant`, `--npu`, `--npu-threads`, `--attn-chunk`

A flag that does not apply to the detected model is rejected.

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
- **Request speed logs** (2026-10-08): `serve` installs a `tracing` subscriber
  (`RUST_LOG`, default `info`) and emits one `info` line per successful inference
  request: `queue_s` (model-lock wait), `compute_s`, token counts + `tok_s`, `RssAnon`;
  chat lines additionally split `prefill_s`/`prefill_tok_s` and `decode_s`/`decode_tok_s`
  from the model's own timings. Errors and `/health`/`/v1/models` are not logged. The
  timing plumbing is `AppState::compute` (`src/server/mod.rs`), the line format
  `src/server/log.rs`; dev-verified for embeddings (qwen3 q8) and chat (Gemma 4 QAT).
- **EmbeddingGemma 2 performance round** (2026-10-08, log §12): four changes,
  numerically validated on the board (timing pending an idle board).
  (1) *Canonical tiling*: `init` now uses `rocket_ctx_create_ex(threads,
  ROCKET_CTX_TILING_CANONICAL)` so resident weights serve any `M >= 4` and small
  requests no longer pad every matmul to 256 rows (`ROCKET_CTX_CANONICAL=0`
  restores legacy padding; off-vs-on is bit-identical at 9 and 2587 tokens).
  (2) *Resident int8* (`--npu-int8`, gemma emb2): the library's W8A8 group-wise
  path (group 32, per-row/per-channel per-group scales, A padded to `M % 4`);
  2587-token cosine vs HF f32 0.99985 (fp16 0.9999997, the deployed q8
  0.9996467), text weights 0.11 GiB vs 0.25. A torch probe
  (`tools/w8a8_probe.py`) gates the convention: per-32-group activations cost
  almost nothing (0.99986), whole-K row scales much more (0.99964).
  (3) *Chunked banded attention* for the 20 sliding layers
  (`--attn-chunk`, default 1024, keys `[q0-w, q1+w)`) through the new
  `attention_window_block` op + `masks::build_window_mask_block`; 2.6k full-vs-
  chunked cosine 0.9999997, 7.7k chunk 1024-vs-2048 0.9999999, 7.7k vs HF
  0.99967; memory 1068 MiB anon at 7.7k where the full path OOM-kills in 3 GB.
  (4) *Fused CPU glue* (`src/host.rs`, `burn_rocket::{gelu_mul, rms_norm,
  rms_norm_noscale, rope_apply}`): single rayon passes instead of flex's serial
  scalar chains; q8 CPU glue on-vs-off cosine 1.0 (max|d| 1.4e-7), `--npu` glue
  on/off equivalent vs HF; `ROCKET_GLUE=0/1` is the A/B switch.
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
- Not done: int8 GEMM on the *flex CPU* path (still the only lever that would close
  the qwen3 speed gap; flex lacks it, and burn-cpu/CubeCL quantized matmul is
  unverified and cannot cross-compile) — the *NPU* int8 path is now implemented for
  emb2 (above). The live OpenViking embedding backend (`openviking-embed-1`) runs
  EmbeddingGemma 2 on the `c951944` image, which predates the performance round
  (canonical tiling, chunked attention, int8, glue).

## Repo layout

| path | contents |
|---|---|
| `src/lib.rs` | FFI wrappers (`RocketCtx`/`RocketWeight`/`RocketStream`/`RocketFaCtx`, `pack_weight_seg`, `flash_attn`), driver/counter helpers, `Error`/`OpFailure` |
| `src/ffi.rs` | raw `extern "C"` declarations for `librocketnpu` |
| `src/ext.rs` | the `RocketOps` Burn backend extension, the global NPU engine (`init`, `WeightId`, `burn_rocket::stats`) and the `Tensor`-level helpers (`pack`/`matmul`/`attention`) |
| `src/masks.rs` | host-built additive attention masks (causal, bidirectional band, causal sliding window, query-chunk band) with unit tests |
| `src/host.rs` | fused host kernels for the NPU build's CPU glue (gelu·up, RMSNorm, RoPE) with unit tests |
| `build.rs` | links `librocketnpu.a` for `npu` builds; resolves `ROCKETNPU_DIR` → matching `vendor/rocketnpu` → `$OUT_DIR/rocketnpu` → auto-build via the script (`ROCKETNPU_AUTO=0` disables, failures warn and are stamped) |
| `scripts/build-rocketnpu.sh` | builds `librocketnpu.a` from a pinned `rocket-userspace` commit on the host (aarch64 cross by default, `--target host` for link checks); cache: `$OUT_DIR/rocket-userspace` from build.rs, else `<target-dir>/<profile>/build/burn-rocket` |
| `examples/probe.rs` | low-level FFI probe (open device, pack, matmul, verify vs CPU) |
| `examples/rocket-inference/Cargo.toml` | workspace member `rocket-inference`: features `cpu` (default), `npu` (aarch64-gated path dep on the root crate), `gpu*`; `wgpu_probe` bin behind `gpu` |
| `examples/rocket-inference/src/main.rs`, `src/cli.rs` | dispatch (`serve` + `qwen3`/`intent`/`gemma`), shared flag parser and usage text |
| `examples/rocket-inference/src/server/` | one HTTP server: `cli` (flags + model loading), `engine` (family detection, capabilities, per-model compute), `error` (panic/NPU mapping), `log` (per-request speed lines), `openai` (`/v1/embeddings`, `/v1/chat/completions`, `/v1/models`), `multimodal` (`/embed`), `ollama` (`/`, `/api/*`) |
| `examples/rocket-inference/src/util/` | shared infra: backend/device, `rss_mib`, server lock/panic helpers (`http`), Q8 low-RAM mapper (`quant`), `Proj`/`ProjKind`/`FusedGroup` (`proj`), `RopeCache` (`rope`), safetensors load-report checks (`store`) |
| `examples/rocket-inference/src/qwen3_embedding/` | Qwen3-Embedding family: model (layers, RoPE, attention paths, stage timers), loaders (f32 / `--quant q8` / `--npu` pack-and-drop + `load_cpu_projections`), CLI |
| `examples/rocket-inference/src/qwen35_intent/` | Qwen3.5-0.8B intent family: model (chunked/recurrent gated delta rule, gated full attention, caches, greedy generation), loader (`model.language_model.*`, NPU pack-and-drop, `load_for_serving`), CLI |
| `examples/rocket-inference/src/gemma/` | shared Gemma 4 building blocks: `config` schemas, `layers` (NPU/SRQ/`ClippableLinear`), `qat`, `media`, `audio` + `audio_frontend`, `vision`, `inputs` |
| `examples/rocket-inference/src/gemma/embeddinggemma/` | EmbeddingGemma 2 family: text backbone, multimodal assembly, loaders, CLI |
| `examples/rocket-inference/src/gemma/gemma4/` | Gemma 4 E2B-it family: causal decoder, loader (f32/f16/q8/QAT, `pack_text_for_prefill`), chat template + sampling (`GenStats.stopped`), CLI |
| `examples/rocket-inference/src/bin/wgpu_probe.rs` | Vulkan/wgpu GPU probe (`gpu-*` features) |
| `examples/rocket-inference/data/` | bench/embedding/media fixtures (`bench_text.txt`, `cat.jpeg`, audio/video samples) |
| `examples/rocket-inference/docs/experiment-log.md` | Qwen3-Embedding + intent measurements: §1-8 dev-host, §9 board A/B, §10 NPU, §11 extension-ops refactor + CPU-loading fix, §12 serving robustness, §13 Vulkan, §14 intent model |
| `examples/rocket-inference/docs/experiment-log-gemma.md` | EmbeddingGemma 2 + Gemma 4 measurements (multimodal parity, Q8/NPU, QAT mobile, NPU prefill) |
| `examples/rocket-inference/tools/` | HF reference scripts (`ref_embeddinggemma2.py`, `ref_gemma4.py`, `debug_audio_hf.py`) |
| `examples/rocket-inference/Dockerfile`, `docker/build.sh` | container image (aarch64, built on the board, pushed to the Forgejo registry) |
| `vendor/rocketnpu/` | **gitignored**: `librocketnpu.a`, `librocketgraph.a`, headers, `COMMIT`/`ARCH` provenance — built on the host by `scripts/build-rocketnpu.sh` (pinned upstream commit; no board copy); `build.rs` auto-builds a per-target copy into `$OUT_DIR` when it is absent |
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

QAT mobile checkpoint (log §11): `google/gemma-4-E2B-it-qat-mobile-transformers`
(2.46 GB) is supported natively — packed INT2/4/8 weights + per-channel scales +
SRQ activation rounding, unpacked by a load adapter (`src/gemma/qat.rs`), with the SRQ
scales registered per weight `ParamId` and applied in `lin()` (ties-to-even
rounding). The PLE table stays packed (rows dequantized on lookup, bit-identical
values, 1.13 GiB vs 8.75 GiB f32). Weights are bit-exact vs HF; short prompts
are token-identical and the 631-token prompt is 12/12 with `--f16` (6.6 GiB
resident, 4.9 tok/s). SRQ makes long-context near-ties sensitive (a
full-quantum activation jump from f32 accumulation order); `NO_SRQ=1` /
`DUMP_PARAM=<substr>` are debug hooks. Board attempt: deployed at
`/root/models/gemma-4-E2B-it-qat-mobile/`, OOM-killed at 7.6 GB (board busy);
needs an idle board.

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

## Rebuild + deploy to the board

```sh
# optional: pre-build the aarch64 archive into vendor/rocketnpu (build.rs
# auto-builds a per-target copy into $OUT_DIR when it is absent)
scripts/build-rocketnpu.sh
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

The binary is self-contained (`librocketnpu` is statically linked); the model lives at
`/root/models/qwen3-embedding-0.6b/` on the board. On the board run from
`/root/rocket-inference` (the bench uses the relative `data/bench_text.txt`).

Library-only builds: `cargo check -p burn-rocket --features npu` compiles the extension
on any host (no linking); `cargo build --release -p burn-rocket --features npu --target
aarch64-unknown-linux-gnu` builds the library, and `--example probe` adds the FFI probe.

Container image: `examples/rocket-inference/docker/build.sh [git-ref]` stages the tree,
builds the aarch64 image on the board and pushes
`git.kmsign.org/royalcat/rocket-inference:<sha>`. It needs
`vendor/rocketnpu/librocketnpu.a` (run `scripts/build-rocketnpu.sh`; `VENDOR_SRC`
overrides) and registry credentials on the control host; run it from anywhere in the
repo. The image CMD is the unified server:
`serve --backend flex --dtype f32 --npu --npu-attn cpu --port 8383 --max-tokens 8192
--model-dir /models/qwen3-embedding-0.6b --model-name qwen3-embedding` (the image
needs a rebuild to pick up the post-merge CMD).

The `bench` summary prints `stages: attention/mlp/norms` and, with `--npu`, an
`npu breakdown: calls/convert/npu/flex+overhead` line — check these before profiling.
The NPU counters live in `src/ext.rs` (`burn_rocket::stats`), the stage counters in the
example's `src/qwen3_embedding/model.rs` (`stage_stats`).

## Environment facts

- `$CARGO_TARGET_DIR` is a shared cache (`/home/royalcat/.cache/rust/target`); never
  assume a project-local `target/`.
- `librocketnpu` source/build cache: `$OUT_DIR/rocket-userspace/` when the script
  runs from `build.rs`, else `<target-dir>/<profile>/build/burn-rocket/` (`src/`
  clone + `build-<target>-<sha>/` cmake trees); removed by `cargo clean`.
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
- **Live service**: OpenViking's embedding backend. Since 2026-10-08 it is
  EmbeddingGemma 2: container `openviking-embed-1` (image
  `git.kmsign.org/royalcat/rocket-inference:c951944`) runs `serve --backend flex
  --dtype f32 --quant q8 --npu --npu-attn npu --port 8383 --max-tokens 8192
  --model-dir /models/embeddinggemma-2`, CPUs 4-7, NPU. Any board run shares the
  NPU and cores 4-7 with it. Historically (2026-10-05) the backend was qwen3 on
  `git.kmsign.org/royalcat/embeddings-fast:0c99eed`, CPUs 4-7, `mem_limit`
  8g, weights at `/root/models/qwen3-embedding-0.6b`; that image predates the
  repo inversion. `examples/rocket-inference/docker/build.sh` pushes
  `git.kmsign.org/royalcat/rocket-inference:<sha>`.
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
  `vendor/rocketnpu/librocketnpu.a` (built by `scripts/build-rocketnpu.sh`; override with
  `ROCKETNPU_DIR`). aarch64 only: the dep is target-gated in
  `examples/rocket-inference/Cargo.toml`.
- `build.rs` supplies the archive without a manual step: `ROCKETNPU_DIR` (explicit;
  never auto-built) → matching `vendor/rocketnpu` (an absent `ARCH` counts as aarch64)
  → `$OUT_DIR/rocketnpu` → auto-build `scripts/build-rocketnpu.sh --out
  $OUT_DIR/rocketnpu` (`ROCKETNPU_AUTO=0` disables; `ROCKETNPU_SRC=<checkout>` is the
  offline path; cross vs native is picked from the target arch). A failed auto-build
  only warns, and a stamp in `$OUT_DIR` (keyed by the script's size+mtime) blocks
  retries until the script changes or `cargo clean -p burn-rocket`.
- The cross archive must be built against the aarch64 uapi headers: with the host's
  `/usr/include` first, `asm/posix_types.h` takes its non-x86 branch, `__kernel_size_t`
  becomes 32-bit and `struct drm_version` 56 bytes, so the wrong `DRM_IOCTL_VERSION`
  makes `rocket_open` fail with ENODEV on the board (found 2026-10-07 by the board A/B
  of the fresh archive). `scripts/build-rocketnpu.sh` shims libdrm for the cross build
  and keeps the host include dirs out; each build runs a `_Static_assert` ABI guard.
- Call `burn_rocket::init(threads)` once, then `pack`/`pack2`/`pack3` (weights ->
  `WeightId`s), `matmul` and `attention` — the model calls these directly. All ops share
  one global engine behind a mutex (the FFI contexts are not thread-safe), so NPU calls
  serialize. `attention_window(q, k, v, h, kv, d, scale, softcap, window)` (added
  2026-10-07 for EmbeddingGemma 2) applies a bidirectional band mask
  (`|q - kv| <= window`; `window < 0` = no mask); `attention_window_block(..., window,
  q_start, kv_start)` (2026-10-08) runs a query chunk against a *subset* of keys
  (`n_kv != n_q`; key `kv_start + j` visible to query `q_start + i` iff
  `|(q_start + i) - (kv_start + j)| <= window`) — the sliding layers' chunked path,
  with the block mask cached on the translation-invariant offset. Both attention ops
  cache their f16 masks per shape in the engine.
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
- Fused CPU glue (2026-10-08, emb2 `--npu`): `burn_rocket::{gelu_mul, rms_norm,
  rms_norm_noscale, rope_apply}` are host rayon kernels (single pass where flex runs
  several serial scalar passes); `ROCKET_GLUE=0` restores the composite ops,
  `ROCKET_GLUE=1` forces the kernels in CPU modes.
- `--npu` = pack-and-drop: 196 projections packed into resident fp16 NPU BOs (0.82 GiB),
  f16 embedding table, 298 MiB CPU-resident. `--npu-attn npu` (default) also offloads
  attention via `rocket_flash_attn_fp16_ctx`; `cpu` keeps flex attention (within ~1 s on
  wall, ~40% more CPU). `--npu` requires `--dtype f32` and excludes `--quant q8`.
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
  stock 200 MHz boot clock. `librocketnpu` is the host cross-built archive from
  `scripts/build-rocketnpu.sh` (GPL-3.0-or-later); the board's
  `/root/npu-poc/rocket-userspace` tree is an older revision and is no longer the source.
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
  dequantized once per layer per forward (`linear_forward` in `src/util/proj.rs`, the
  shared `LowRam` mapper in `src/util/quant.rs`, `Qwen3Embedding::quantized` in
  `src/qwen3_embedding/model.rs`); the embedding table is f16; `libc::malloc_trim` after
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

- **int8 GEMM on the flex CPU path** is the only lever that would close the qwen3
  speed gap with production. flex has none; the candidates are the `burn-cpu`
  (CubeCL/LLVM) backend (CPU quantized matmul unverified, cannot cross-compile — would
  need a native build + JIT on the board) or a custom int8 microkernel (upstream
  contribution). The NPU int8 path (emb2) is done (log §12.2).
- **Vulkan/GPU path is closed** (log §13): measured 3.8-79 GF/s with CubeCL on panvk,
  panvk compiler crashes on CubeCL SPIR-V, and the panthor job watchdog kills the
  dispatches this workload needs. Re-open only with a materially faster CubeCL/panvk
  stack.
- Long-context NPU attention: emb2's sliding layers now use chunked banded attention
  (log §12.3), which fixes the memory blow-up there. The *causal* models (qwen3,
  gemma4 prefill) still build `[n][n]` masks and OOM at ~30k; the same query-chunking
  idea applies (per-chunk `[C, q1]` causal masks, prefix keys), plus a 30k measurement
  of the emb2 chunked path (the CPU path does 52.4 tok/s at 32 dev threads).
- Board service: the emb2 backend runs the pre-round `c951944` image. Redeploy on
  this build (canonical tiling + chunked attention + glue; consider `--npu-int8` as
  the default — better numerics *and* faster) and re-measure the served latencies.
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
# library compile checks (npu needs aarch64 only for linking)
cargo check -p burn-rocket --features npu
# example npu compile check (the dep is aarch64-gated, so use the target)
cargo check -p rocket-inference --target aarch64-unknown-linux-gnu --no-default-features --features npu
# formatting + lint gates (both clean)
cargo fmt --all -- --check
cargo clippy -p rocket-inference --no-default-features
# single-core speed gate (blocked path is the single-core record); run from
# examples/rocket-inference
taskset -c 2 cargo run --release -- qwen3 bench --backend flex --dtype f32 --tokens 3633 --reps 2 --attn blocked --chunk 256 --key-block 256
# multi-threaded / long-input (fused default)
cargo run --release -- qwen3 bench --backend flex --dtype f32 --text-file /tmp/opencode/long30k.txt --tokens 30000 --reps 0
# embedding + cosine against a reference JSON
cargo run --release -- qwen3 embed --backend flex --dtype f32 --quant q8 --text-file /tmp/opencode/one_64.txt --out /tmp/opencode/our.json
# server smoke test (one command for every model; family auto-detected)
cargo run --release -- serve --model-dir ~/models/qwen3-embedding-0.6b --backend flex --dtype f32 --quant q8 --port 8383 &
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"hi"}'
# robustness: a panicking forward (invalid token id) must 500 and leave the server alive
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":[999999999]}'
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' -d '{"input":"still alive"}'
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
# performance-round A/Bs (log §12; validate.sh runs the whole set)
ROCKET_CTX_CANONICAL=0 ./rocket-inference gemma embed --model-dir /root/models/embeddinggemma-2 --quant q8 --npu --prompt query --text-file data/one_long.txt --out /tmp/off.json
./rocket-inference gemma embed --model-dir /root/models/embeddinggemma-2 --npu --npu-int8 --prompt query --text-file data/one_long.txt --out /tmp/i8.json
ROCKET_GLUE=0 ./rocket-inference gemma bench --model-dir /root/models/embeddinggemma-2 --quant q8 --text-file data/one_long.txt --reps 1
./rocket-inference gemma gen --gen-model-dir /root/models/gemma-4-E2B-it --text "What is the capital of France?" --max-new-tokens 16
# servers (detected per model; ports are examples)
./rocket-inference serve --model-dir /root/models/qwen3-embedding-0.6b --quant q8 --npu --port 8383 --max-tokens 8192
./rocket-inference serve --model-dir /root/models/ov-intent-analysis-sft --npu --npu-threads 3 --port 11434 --max-new-tokens 256
./rocket-inference serve --model-dir /root/models/embeddinggemma-2 --quant q8 --npu --port 8390
./rocket-inference serve --model-dir /root/models/gemma-4-E2B-it --port 8391
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
