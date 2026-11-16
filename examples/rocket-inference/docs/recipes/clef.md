# Clef-Omni (Cloudflare) — typed decisions, text-only, CPU

> **STATUS 2026-11-14 (phase 6): text-only. CPU path dev-host verified; the NPU expert
> streaming path is code complete and verified on the dev host through its host
> emulation (`ROCKET_CLEF_EMU=1`), an aarch64 cross build and unit tests. NOT yet run on the
> board.** **Images are served since phase 8c** (see "Vision tower and image requests"),
> **videos since phase 8d** (see "Video requests") and **audio plus audio-in-video since phase 8e**
> (see "Audio requests"); the towers are dev-host verified, not run on the board.
> Phase 5 adds `--pack-dir` (the int8 pack, `clef-pack.md`): same flags, packed weights.
> Phase 6 makes `--device npu` work with a pack (see "NPU placement" below).

`Cloudflare/clef-omni` is the Qwen3-Omni-30B-A3B thinker (48 MoE layers, 128 experts,
top-8) with a `JointSchemaHead`. A request is a `state` plus typed `questions`; one
forward over the whole record answers every question jointly (no generation,
`usage.output_tokens` is always 0). It is the same Jev/SystemOne `/v1/systemone` contract
as the d3 family, but a separate implementation (`src/clef/`): the answer format differs.

Checkpoint (66 GiB, 15 BF16 shards + `joint_head.safetensors`): `$CLEF_MODEL_DIR`.
Detection: `config.json` `model_type == "qwen3_omni_moe"` **and** `joint_head.safetensors`
present (the plain Qwen3-Omni base does not match); `--family clef` forces it.

## Run

```sh
B=$CARGO_TARGET_DIR/release/rocket-inference
cargo build --release -p rocket-inference --no-default-features

# batch: JSON arrays of cases like d3 (`name`, else `id`, is only the label; `model`
# defaults to the model dir name)
$B decide --model-dir $CLEF_MODEL_DIR \
    --requests examples/rocket-inference/tools/clef_fixtures.json --out /tmp/clef-out.json
# one case
$B decide --model-dir $CLEF_MODEL_DIR --request '{"model":"clef-omni",
  "state":"Our checkout started returning errors.","questions":{"outage":{"type":"noul"}}}'

# server
$B serve --model-dir $CLEF_MODEL_DIR --port 8399 --max-tokens 8192
curl -s localhost:8399/v1/systemone -H 'Content-Type: application/json' -d @request.json
```

`decide --requests` writes `{"schema": "clef-decision-v1", "model", "cases": [{"name",
"response" | "error"}]}`; `--request` writes the bare response. A refused case (invalid
request, media, a schema over the cap) is reported as `error` and the run exits non-zero
after writing the output.

## Request / response

Request (the release's `systemone()`; extra top-level keys are ignored):
`model` (string, echoed) and `state` (any JSON; a string verbatim, anything else as
compact sorted-key JSON) are required, `questions` is a non-empty object of
`{"type": "noul"|"choice"|"score", "instructions"?, "criteria"?}`; criteria must be
non-empty for choice (an object `{option: description}`) and score (a list; a dict walks
its keys); a noul question may override the `true`/`false` descriptions. Empty/missing
`instructions` fall back to the question id.

Response `{"model", "answers", "usage": {"input_tokens", "output_tokens": 0}}`, answers
in request order, probabilities rounded to 4 digits like the Python `round(x, 4)`:

| type | answer |
|---|---|
| `choice` | `choice`, `confidence`, `probabilities` (criteria in request order; ties go to the first) |
| `score` | `score` (expected index), `confidence` (max prob), `legend` (index -> criterion), `probabilities` |
| `noul` | `noul` (probability of `true`) |

Errors (`/v1/systemone`): `422 {"detail": "<reason>"}` for a body that is not JSON, a failed
validation (`model and state are required`, `at least one question is required`,
`<id>: type must be noul, choice, or score`, `<id>: criteria must not be empty`), a
schema larger than the cap (`schema requires N tokens before state; maximum is M`, media
tokens included), an unreadable/undecodable image, audio clip or video (`images[i]: ...`,
`audio[i]: ...`, `videos[i]: ...`), audio over `ROCKET_CLEF_AUDIO_MAX_SECONDS`, and
`media_kwargs`. Compute failures are 500.

## Flags and limits

- `--max-tokens` (serve) / `--max-state-tokens` (decide): per-record token cap, default
  **8192** (the release's Python default is 64000). The state is cut to what the schema
  leaves; a schema that alone exceeds the cap is a 422. The record's cost grows with its
  length, so the board wants the small cap.
- Placement: `--device auto|cpu|npu` (and `--text-device`) as for the other families.
  `auto` is the NPU on an aarch64 `npu` build **when the model dir is the int8 AutoRound
  checkpoint or a clef pack** (a BF16 checkpoint falls back to the CPU), else the CPU. An explicit `npu` on a CPU build is the
  shared "requires an aarch64 build with the `npu` feature" error, and on an NPU build with
  a BF16 checkpoint "needs the int8 AutoRound checkpoint". Flags of other families (`--dtype`, `--quant`,
  `--defer-towers`, `--text-attn-device`, `--decode-device`, `--vision-*`, ...) are rejected.
  The `placement:` line reads `text=CPU(f32, layer-streamed) head=CPU(f32)`, or
  `text=NPU(int8 g128 streamed experts) head=CPU(f32)`, or (with `ROCKET_CLEF_EMU=1`)
  `text=EMU(int8 g128 W8A8 host emulation, streamed experts) head=CPU(f32)`.
- Memory: the thinker streams one layer at a time from the mmapped shards (the pages are
  released after each layer): 480 MiB resident after load (head 436 MB f32), 590-770 MiB
  after a 256-295 token request. The page cache holds what it can of the 66 GiB.
- One model lock per process: requests are serialized; the request log line is
  `decision request endpoint=/v1/systemone questions=.. tokens=.. compute_s=..`; with
  `OTEL_SDK_ENABLED=true` the request's `inference` span has a `clef` child
  (`questions`, `tokens`).
- Threads: a private rayon pool of `min(cores, 8)` workers for the backbone and head.

## Measured (dev host, 8 threads, warm page cache, release build, 2026-11-14)

| record | tokens | time | max abs dp vs f32 oracle | answers vs oracle |
|---|---|---|---|---|
| checkout-errors | 295 | 20.6-21.2 s | 3.1e-7 | identical (rounded) |
| invoice | 256 | 20.0-20.4 s | 1.3e-8 | identical (rounded) |

Model-card example (`checkout-errors`): department technical 0.8327, urgency score 1.8917
(0.0425/0.0234/0.9341), outage 0.9552. The Python oracle needs ~60-73 s per record on the
same host. The first run after a cold cache is bound by reading the 66 GiB of shards.

## Parity gates

- **NPU path, host emulation** (`ROCKET_CLEF_EMU=1`, release, 8 threads, pack
  `clef-pack-g128`; `cargo test --release -p rocket-inference --no-default-features
  clef_emu_answers -- --ignored --nocapture`, `CLEF_LAYER_LOG=1` prints every layer):

  | record | tokens | emu vs oracle max / mean |dp| | emu vs CPU-dequant pack max / mean | time emu / CPU pack |
  |---|---|---|---|---|
  | checkout-errors | 295 | 0.0156 / 0.0066 | 0.0092 / 0.0054 | 24.8 / 21.5 s |
  | invoice | 256 | 0.0006 / 0.0003 | 0.0007 / 0.0004 | 20.2 / 21.0 s |
  | long-incident | 800 | 0.0262 / 0.0088 | 0.0150 / 0.0046 | 62.4 / 28.3 s |

  Top-1 equals the oracle on every question. Per-layer hidden cosine vs the oracle (all 48
  layers of checkout-errors): emu and CPU-dequant agree to <= 7e-4 at every layer (typically
  1e-5), both sit at 0.9978-0.9995 through layer 45 and fall to 0.9937/0.9865 (emu) at layers
  46/47 — the weight quantization of the checkpoint dominates, W8A8 activations add almost
  nothing. The emulation is *not* a speed path (it is a diagnostic).

- Encoder: `cargo test -p rocket-inference --no-default-features clef::encode` compares the
  token ids and question/option spans with the release's Python `encode_record` on the 8
  fixtures and 14 edge cases (unicode, empty instructions, JSON state with floats,
  truncation by `max_state_tokens` and `max_length`, a too-small cap, special tokens in
  the state, nested criteria, noul overrides, unicode option sorting): **identical**.
  Regenerate `src/clef/testdata/encode_cases.json` with
  `tools/clef_encode_expected.py` (see `clef-oracle.md` for the venv).
- End to end: `cargo test --release -p rocket-inference --no-default-features
  clef_answers_match_the_oracle -- --ignored --nocapture` (`CLEF_IDS=a,b` selects records;
  needs `$CLEF_MODEL_DIR` and the oracle dumps in `/tmp/opencode/clef/dump`,
  regenerate per `clef-oracle.md`): probabilities within 1e-4 and the rounded answers equal.

## NPU placement (phase 6)

Needs a clef pack (`clef-pack.md`) and an aarch64 build with `--features npu`. Everything
heavy of the thinker runs as **int8 g128 W8A8** through the driver's group-wise int8 path
(`I8Context::matmul_gw`): activations are quantized per row per 128-group on the host
(`scale = max|x|/127`, round half away from zero), the NPU computes exact int32 group
partials, the host applies `a_scale * w_scale` and sums in f32.

- **MoE** (`src/clef/engine.rs`, `engine_npu.rs`): per layer, after the router, tokens are
  bucketed per expert; a reader thread streams the hit experts (one aligned O_DIRECT read of
  4.57 MiB each, `ROCKET_CLEF_DEPTH` read-ahead buffers, default 8) while **three consumer
  threads, each owning an `I8Context::new_at(1, core)`** with resident gate|up (`K=2048
  N=1536`, gate and up fused) and down (`K=768 N=2048`) slots, take the next arrived expert, `rewrite_weight_gw`
  their slots, quantize and run gate|up, do SiLU*mul on the host, quantize and run down, scale
  the rows by the router weights and file them under the expert id. The layer ends with a
  barrier; per-expert outputs are then combined **per token in expert-id order**, so the
  result is bit-identical for any consumer timing. Rows are chunked at 512 and padded to
  one of 14 shape classes (`M_CLASSES`) because the driver caches one scratch per exact
  `(M,K,N,group)`.
- **Prefetch**: when `S * 8 >= 512` the whole layer is read speculatively as soon as the
  layer starts (before attention) and the next layer right after the MoE, so reads overlap
  attention/combine; consumers discard non-hit and stale items, so correctness does not
  depend on it. Small requests read exactly the hit experts. `posix_fadvise(DONTNEED)` is
  applied to the layer file after each layer on filesystems that refuse O_DIRECT; the
  resident mmap is `madvise(DONTNEED)`ed after every layer.
- **Attention projections** (`ROCKET_CLEF_ATTN_DEVICE=npu|cpu`, default npu): per layer a
  fused q|k|v weight (`K=2048 N=5120`) and `o` (`K=4096 N=2048`) are packed **once at load**
  as resident `I8Weight`s of one 3-worker context (~0.9 GB of BOs, the load logs the size and
  time). `cpu` keeps them on the CPU, dequantized to f32 per layer (the phase 5 path).
  Scores/softmax/AV stay on the CPU in this phase: a blocked GQA implementation
  (`attn.rs`, one `gemm` call per head and 128-row query block, working set
  `threads x 128 x keys x 4 B`, no `[S,S]` matrix). Moving them to the NPU
  (`rocket_npu::attn` / `burn_rocket::attention_causal_block`) is a follow-up. The router,
  norms, RoPE, embeddings, final norm and the joint head are CPU in every placement.
- **Failures**: a read error or a worker error aborts the request (HTTP 500) and the next
  request starts afresh; an NPU device error becomes the typed `OpFailure` panic payload the
  server already maps (NOMEM -> 503, dead device -> log + exit for a restart).

### Env knobs

| variable | meaning |
|---|---|
| `ROCKET_CLEF_EMU=1` | Test/diagnostic: run the *host emulation* of the NPU path (same bucketing, schedule, quantization and int32-exact group partials; any host, needs the int8 checkpoint or a pack). Overrides `--device`. |
| `ROCKET_CLEF_ATTN_DEVICE=cpu|npu` | attention projections on the CPU (f32) or int8 on the NPU/emulation (default). |
| `ROCKET_CLEF_DEPTH=N` | read-ahead buffers (default 8 x 4.57 MiB). |
| `ROCKET_CLEF_CONSUMERS=N` | NPU expert consumers (1-6, default 3 = one per core). |
| `ROCKET_PHASE_LOG=1|2` | one `clef phases [...]` line per request (2 adds a line per layer). |
| `ROCKET_KEEP_MB` | scratch budget kept after a request (default 64; the contexts' per-shape scratch is dropped above it). |
| `CLEF_PACK_NO_DIRECT=1` | read the pack buffered (+ fadvise) instead of O_DIRECT. |

### Board usage

```sh
# on the board (the live services share the NPU and cores 4-7: stop them first)
taskset -c 4-7 ./rocket-inference serve --pack-dir /root/models/clef-pack-g128 \
    --device npu --max-tokens 4096 --port 8385
# or one batch
ROCKET_PHASE_LOG=1 taskset -c 4-7 ./rocket-inference decide \
    --pack-dir /root/models/clef-pack-g128 --device npu \
    --requests tools/clef_fixtures.json --out /tmp/clef-npu.json
```

Expected log: `clef: thinker engine: int8 g128 on the NPU, streamed experts (3 contexts x 1
worker); attention projections int8 g128 W8A8 on the NPU (resident)`, the attention pack
line (48 layers, ~900 MiB), then per request
`clef phases [...] tokens=295 layers=48 experts=6144 ... | wall: moe=..s qkv=..s oproj=..s
attn=..s router=..s combine=..s | thread-s: read_wait=.. rewrite=.. quant=.. npu=.. glue=..
read_io=.. | peak_anon=.. MiB`. The phase-0 probe predicts ~1.4-1.9 s of expert matmul and
~0.5 s of disk per layer at M_e=64 (about 72 s of expert time per 1k-token pass).

## Batching (phase 7)

A pass costs a ~30 s floor on the board (all 128 experts of a layer are hit, so ~5.5k
expert loads/request, each a rewrite + two matmuls) plus ~0.1 s per extra token. The
expert reads and the per-expert fixed costs are shared by every row of a layer, so
several records can share ONE expert stream: **batching N records concatenates their
MoE rows per layer** (bucketed by expert over all rows, combined per token in expert-id
order); attention (q|k|v and o projections are row-independent and run over all rows
at once; RoPE with positions restarting per record, the causal scores and the router run
per record) never mixes records. The workload is background, so latency is traded for
throughput.

Flags (Clef only; any other family rejects them):

| flag | default | meaning |
|---|---|---|
| `serve --batch-window-ms <n>` | 0 | how long the first queued record waits for company, counted from its arrival (a record that queued behind a running batch is not delayed again). 0 = no waiting beyond what is already queued: an isolated request has today's latency |
| `serve --batch-max <n>` | 8 | most records per shared pass (a batch is dispatched as soon as it holds this many) |
| `serve --batch-max-tokens <n>` | 4096 | most rows (tokens) per shared pass; a record that would overshoot it opens the next batch, a single record over it runs alone |
| `decide --batch-max <n>` / `--batch-max-tokens <n>` | 8 / 4096 | the same limits for `--requests`: the cases are grouped in order, one `batch i/n: ... expert read(s) in ...s` line per batch, output order and per-case `error` entries unchanged |

Server semantics: a request is parsed, validated and **encoded in the handler** (media,
schema and over-cap records are `422` immediately and never queue), then queued; one
worker thread forms batches and runs each under the model lock (`/health` and
`/v1/models` never take it). Each request logs its own line with `queue_s`, `compute_s`
(the shared batch compute, the same for every member), `tokens`, `batch=N` and
`batch_tokens`, plus one `clef batch` line (`batch`, `batch_tokens`, `passes`,
`experts_read`). Traces: the `inference` span carries `batch`/`batch_tokens`, its `clef`
child `batch`, `tokens`, `passes`, `experts_read` and the phase totals. A client that
disconnects while queued is skipped (its reply channel is closed) or, once running,
simply not answered; SIGTERM stops accepting, runs everything already queued without
waiting out the window, then exits. A panic or NPU failure in a shared pass fails every
member of that pass with the usual mapping (503 NOMEM, 500, device failure exits).

**Memory bound.** Per-expert outputs are `rows * top_k * hidden * 4` B (rows * 8 * 2048 *
4 = 64 KiB per row, 268 MB at 4096 rows) plus the gathered inputs, so `--batch-max-tokens`
is the knob: the default 4096 bounds it at about that (dev host, emulation: a 881-row batch peaked at
718 MiB `RssAnon` vs ~480-615 MiB idle/after singles). The
per-expert NPU shapes are bucketed in M classes (<= 512 rows per call), so a larger
batch does not multiply the number of driver shapes.

**Bit-identity.** Quantization is per row per group, the int32 group partials are exact,
the router is per token and the combine order is per token in expert-id order, so on the
host emulation (`ROCKET_CLEF_EMU=1`, `Int8Emu`) each record's result is **bit-identical**
to running it alone, whatever its batch mates (tests: lengths 1/7/23/40, a duplicate,
chunking under a row cap; and the real-pack gate below). The plain f32 CPU path (no
engine) is **not** bit-identical: the expert GEMMs run on the bucket's row count and the
`gemm` crate's summation order depends on it (measured max|d|/peak 6e-7 on the mini
model: rounding noise). On the NPU engine bit-identity depends on the driver's int8
group-wise matmul being row-independent for any M class/chunk; that is the property the
board test checks (it was not measured on the board).

**When to use.** Throughput on a loaded queue: the floor is paid once per batch. On the
board (floor 30 s, 0.1 s/token) four 250-token records cost ~4 x 55 = 220 s alone and
should cost ~30 + 0.1 x 1000 = ~130 s as a batch (an estimate from the measured floor, not a
measurement). Recommended board deployment: `--batch-window-ms 5000 --batch-max 8`
(`--batch-max-tokens` left at 4096); with an isolated request the only cost is the 5 s
window. On the dev host the CPU emulation has no streaming floor, so the gain is small.

Dev-host measurements (release, `ROCKET_CLEF_EMU=1`, 8 threads, pack `clef-pack-g128`):

| run | wall | expert reads |
|---|---|---|
| 4 records alone (295 + 256 + 800 + 156 tokens) | 122.8 s | 5219 + 5179 + 5579 + 5071 = 21,048 |
| the same 4 as one batch (1507 rows) | 113.8 s | 5,700 |
| `serve`, 4 concurrent curls (881 rows), window 3 s | 70.2 s vs 76.7 s for the 4 singles | 5,533 vs 20,426 |

All answers byte-identical to the single runs (probabilities compared as exact f64 in the
gate, response bodies `cmp`-identical in the smoke). The reads per request collapse 3.7x;
the dev-host wall gain is only 7-8 % because the emulated expert math (not I/O) dominates
there. The gate: `cargo test --release -p rocket-inference --no-default-features
clef_batch_equals_single_records_on_the_emulated_engine -- --ignored --nocapture`.

Board check: see "Board test, phase 7" below.

## Media plumbing in the thinker (phase 8b, 2026-11-14)

`src/clef/media.rs` is everything between "the towers produced features" and "the decoder
ran"; the image tower exists since 8c (below), videos since 8d (see "Video requests") and the
audio tower, `audio` and soundtracks since 8e (see "Audio requests").

* **Placeholder block**: `MediaLayout { image_grids, video_grids, audio_frames (mel frames, explicit
  clips first then one soundtrack per video iff use_audio_in_video), explicit_audio,
  use_audio_in_video, second_per_grid }` -> `build_media_ids(&layout, newline_id)` (images, explicit
  audio, videos; audio-in-video interleave by `interleave_video_audio`) and
  `encode::encode_record_with_media(tok, req, max_length, max_state, &media_ids)` (the media block
  follows the 35-token `encode::prefix_tokens`, counts toward the fixed length; `encode_record` = empty
  block). Also `audio_out_len(frames)`, `grid_tokens`, `smart_resize` (factor 32, pixel budgets; for 8c).
  Test: ids of all 9 media fixtures equal the oracle's `input_ids`, question/option spans equal.
* **Positions**: `rope_index(ids, &layout) -> Positions {t,h,w}` is a literal port of HF
  `get_rope_index` (3-D branch only when an image or video grid exists; text/audio-only = plain `0..S`;
  `position_id_per_seconds` 13; audio-in-video merge by time; an error when pad counts disagree with the
  lengths, which HF would silently mis-position). Equal to the oracle's `position_ids` and `rope_delta`
  on all 9 fixtures. `Rope::from_positions` builds the interleaved `[24,20,20]` tables (slot `j` reads
  h if `j%3==1 && j<60`, w if `j%3==2 && j<60`, else t); with text positions it is bit-identical to
  `Rope::new`.
* **Splice + DeepStack**: `MediaFeatures { positions, splice_idx/rows, deepstack_idx, deepstack[3] }`
  (`MediaFeatures::assemble(ids, positions, &TowerOutputs, hidden)` locates the pads and checks counts;
  `splice`, `inject_deepstack(layer)` add after decoder layers 0-2 at visual rows only, before the tap).
* **Backbone API**: `forward_media[_tapped](ids, Option<&MediaFeatures>, opts)`,
  `forward_batch_media(&[RecordRef{ids, media}], opts, cap)` (per-record position streams, splice and
  DeepStack in a shared pass; one shared `Rope` is kept for all-text batches), `embed_input`. The text
  entry points delegate with `None`; layer functions take `&dyn RopeSource` (`Rope` = shared,
  `Vec<Rope>` = per record). CPU, Emu and NPU engines are unchanged for text-only (all existing tests,
  incl. bit-identity batching, pass).
* **Gates** (oracle dumps in `/tmp/opencode/clef/dump`, tower outputs fed from the dump):
  `cargo test --release -p rocket-inference --no-default-features media_backbone_matches_the_oracle_on_the_bf16_cpu_path -- --ignored --nocapture`
  and `..._on_the_emulated_int8_path` (AutoRound shards, = `ROCKET_CLEF_EMU=1`); `CLEF_IDS=a,b` selects.
  BF16 f32 CPU (dev host, 8 threads, ~26 s/record): layers 0-2 (DeepStack included) cosine 1.0, max|d|/peak
  1e-7..8e-7; image-small final_normed cosine 1.000000, max|dp| 2.5e-7; audio-5s 1.0, 3.7e-7; video-audio-small
  final cosine 0.99998 (25 near-tied router flips over 48 layers x 544 tokens), max|dp| 1.3e-5; all top-1 equal.
  Emulated int8 W8A8 (AutoRound shards): top-1 equal everywhere, max|dp| image 1.2e-2, audio 8.4e-3,
  video-audio 1.7e-2 (layer 0-2 cosine >= 0.99993; late layers drift as for text, cosine ~0.98, router sets differ).
  Non-ignored tests also cover the unit cases, the oracle layout/positions/ids, and a mini-model batch
  test (media batched == single, bit-identical on Emu; within 1e-4 on the CPU path).
* **Not done here**: the audio/video towers (8d, 8e), video/audio decoding (8f), board runs of media
  records. Images (8c) are wired: see "Vision tower and image requests".

## Vision tower and image requests (phase 8c, 2026-11-14)

`images` requests work end to end: `POST /v1/systemone` and `decide` accept
`"images": ["<path>" | "data:image/png;base64,..." | "https://..."]` (same fetch/decode helpers as the
EmbeddingGemma 2 server: 20 MiB, 10 s, 3 redirects, `ROCKET_MEDIA_FETCH=0` disables URLs; relative
`decide` paths resolve from the working directory). `audio` is still a 422 ("not implemented
yet"), `media_kwargs` is a 422, a source decoded above 16 MP is a 422 (this port's own memory guard).
A client error in an image is a 422 naming `images[i]`.

Files: `src/clef/vision.rs` (config, tensor index, forward, the lazy tower), `vision/preprocess.rs`,
`vision/compute.rs` (CPU / fp16 emulation / shared attention + padding helpers), `vision/npu.rs`
(aarch64 `npu` only), `vision/tests.rs`. Plumbing: `encode::encode_request` (the one entry the server,
`decide` and `ClefModel::encode` use: prepares the images, builds the placeholder block, counts it
against the cap, attaches `RecordMedia`), `ClefModel::media_features` (tower -> `rope_index` ->
`MediaFeatures::assemble` -> `forward_batch_media`), `ClefFront.images` (a `clef-pack` has no tower:
HTTP 422 for images).

* **Preprocessing** (`vision/preprocess.rs`): decode (PIL `convert("RGB")` semantics, mozjpeg for JPEG),
  `smart_resize` (factor 32, **65,536..1,048,576 pixels**), the ATen uint8 antialiased bicubic port of
  `gemma::media::resize_bicubic_aa`, `(v - 127.5) / 127.5` (bit-exact to the processor's fused
  rescale+normalize; `(v/255 - 0.5)/0.5` is not), patchify into merge-block order with `[channel,
  temporal, py, px]` rows, a still duplicated into both temporal slots (`patchify_pairs` also takes real
  frame pairs for video). **Gate**: decoded RGB bytes equal PIL's and the pixel values equal the oracle's
  `media_pixel_values` with **0 differing values** (bit for bit) for `media-image-small` (430,080 values),
  `media-image-cat` (3,870,720) and `media-images-two` (4,300,800), grids `14x20`, `42x60`, `14x20+42x60`.
* **Tower** (`vision.rs`): the `manual_vision` spec, host f32. Weights are **BF16 in both checkpoints**
  (all 351 `thinker.visual.*` tensors: shard 1 of the BF16 repo, shards 6/7 of the AutoRound repo: the
  AutoRound recipe quantizes only `thinker.model.layers`), mmapped from the shards named by
  `model.safetensors.index.json` and widened per use (no host copy, ~1 GiB in the page cache). The
  tower opens on the **first image request** (`vision tower opened in ...` log line); a request without
  images never touches it. Position table: the d3 `position_taps` (bilinear, `align_corners=True`) in
  block order; rotary and per-frame attention exactly as in the spec (a `[2,h,w]` grid equals two
  `[1,h,w]` runs). Output = `TowerOutputs` (`image`, `image_deepstack[0..3]`), rows concatenated in
  record order, consumed by `MediaFeatures::assemble`. The CPU attention is the blocked gemm kernel of
  `attn.rs` made non-causal.
* **Placements** (`TowerDevice`, `ROCKET_CLEF_TOWER_DEVICE=cpu|emu16|npu`; default = `npu` when the
  thinker runs on the NPU, else `cpu`):
  * `cpu`: f32 gemm. 2,520 patches in **5.7 s**, 280 patches in 1.0 s (dev host, 8 threads, includes
    widening the weights).
  * `emu16`: the host emulation of the NPU numerics (fp16 weights, fp16 activations in and out of every
    linear and the attention, f32 accumulation); an fp16 overflow is a `Fp16Range` error.
  * `npu` (aarch64 + `npu` builds): all **117 linears** (patch embed, 27 x qkv/proj/fc1/fc2, 4 mergers x 2) are
    packed once as **resident fp16** weights through `RocketCtx::pack_weight` (canonical tiling, any
    M >= 4; ~1.07 GiB of BOs that live as long as the tower), the path the EmbeddingGemma 2 and d3
    towers use: the BF16 -> fp16 conversion is exact for these weights (a non-finite result refuses the
    pack), and a native bf16 matmul (`Bf16Context`) has no speed advantage on this part and would need
    its own plumbing, so it was not used. `fc2` has `K = 4304` (not a multiple of 32): weight columns
    and activation are zero-padded to 4320. Bias, norms, rotary, GELUs and softmax stay on the host in
    f32. Attention goes per frame through `RocketFaCtx::flash_attn` (fp16 flash attention, no mask, the
    host fast softmax) with the **72-wide head zero-padded to 96 (q/k) and 80 (v/out)** (the driver wants
    `head_dim % 32`, `dv % 16`; padding with zeros changes neither the scores nor the softmax). The
    padding/layout contract is host-tested against a naive f64 model of the driver layout.
    `ROCKET_CLEF_TOWER_THREADS` (default 3) sizes the contexts; after every request the tower trims its
    scratch to `ROCKET_KEEP_MB` and releases the shard pages.
  * An activation that rounds to an fp16 infinity (or a driver `ROCKET_E_RANGE` score) is a `Fp16Range`:
    **that image reruns on the f32 CPU path** (`fallbacks()` counts them, one stderr line each). A device
    failure is the usual typed `NpuFailure` panic (503 / exit as for the thinker).
  * `ROCKET_CLEF_TOWER_CHECK=1` (non-CPU placements): after the tower ran, every image is rerun on the
    f32 CPU and a `tower check images[i] vs f32 CPU: merged cos ..., deepstack0..2 ...` line is logged: the
    on-board numerics check, no extra binary.
* **Tower gates** (dev host, oracle dumps `/tmp/opencode/clef/dump`; the AutoRound checkpoint's tower;
  `cargo test --release -p rocket-inference --no-default-features clef::vision`, the cat ones are
  `--ignored`):

  | image | arm | block 0 / 8 / 16 / 24 / 26 cosine | merged (cos, rel-to-peak) | DeepStack 0 / 1 / 2 rel-to-peak |
  |---|---|---|---|---|
  | small (280 patches) | f32 CPU | 1.000000000 everywhere; rel 7e-7 / 1.3e-5 / 2.9e-6 / 5.5e-6 / 1.9e-5 | 1.000000000, 1.05e-5 | 3.0e-6 / 9.6e-6 / 1.15e-5 |
  | cat (2,520 patches) | f32 CPU | 1.000000000 everywhere; rel 1.1e-6 / 1.3e-5 / 3.0e-6 / 1.3e-5 / 1.6e-4 | 1.000000000, 2.8e-5 | 4.3e-6 / 1.0e-5 / 2.8e-5 |
  | small | fp16 emulation | 0.9999999 / 0.9999973 / 0.9999925 / 0.9999901 / 0.9999596 | 0.99998985, 6.9e-3 | 1.5e-3 / 8.0e-3 / 1.2e-2 (cos 0.9999976 / 0.9999907 / 0.9999803) |
  | cat | fp16 emulation | 0.9999999 / 0.9999981 / 0.9999949 / 0.9999769 / 0.9999531 | 0.99998962, 1.2e-2 | 7.9e-4 / 9.7e-3 / 2.6e-2 (cos 0.9999991 / 0.9999906 / 0.9999605) |

  (rel = max |d| / peak of the oracle tensor; block 26's peak is ~1.8e3 on a few outlier channels, so its
  rel value is dominated by them: f32 max |d| 0.28, fp16-emulated 141 = fp16 spacing at 1.8e3.) The f32
  port is as close to the oracle as the oracle's own HF-vs-plain-torch selfcheck (1e-5..1e-4); the fp16
  emulation never overflowed (no CPU fallback) and is what the NPU numerics should reach: cosine
  >= 0.99995 on every tap, 0.99998 on the merged tokens. The same numbers come out of the BF16 repo's
  tower and the AutoRound repo's (`the_tower_opens_on_the_autoround_shards_too`: cosine 1.0, rel 1e-5).
* **Answers vs the oracle** (`image_answers_*` tests, `ClefModel::answer` = decode + tower + thinker +
  head on the AutoRound shards; the oracle is the f32 BF16 stream; **the BF16 checkpoint was not on the
  host for this round, so the tight BF16 gate was not run**: the test picks a 5e-3 bound when
  `CLEF_MODEL_DIR` is a BF16 directory):

  | path | `media-image-small` (319 tokens) | `media-image-cat` (980 tokens) |
  |---|---|---|
  | thinker f32 CPU on the int8 weights (weight-only quantization) | max \|dp\| 3.0e-3, top-1 equal, 25.8 s | 5.5e-3, top-1 equal, 41.1 s |
  | emulated int8 W8A8 thinker (`ROCKET_CLEF_EMU=1` semantics) | 2.5e-3, top-1 equal, 27.7 s | 1.8e-2, top-1 equal, 81.8 s |

  (text-only fixtures sit at 0.0136 / 0.026 max \|dp\| on the same paths, so the image records are inside
  the existing quantization band; with the oracle's own tower outputs the 8b gate gave 1.2e-2 on the
  small image.) `serve` smoke (emulated engine, tower on the CPU): a `data:` PNG 28.0 s (`tokens=319`,
  `colorful 0.9059, shape circle 0.9154`), two images by path 34.2 s (`tokens=392`); a missing file,
  an undecodable `data:` URL and an `audio` entry are 422s with the reason, nothing else is affected.
* **Memory/time on the board (expected, not measured)**: the tower adds ~1.07 GiB of BOs to the thinker's
  ~0.93 GiB (resident attention) and the expert streams; pack time ~10-30 s at the first image request (the
  d3 tower packed 0.28 GiB in 4.3 s); the NPU tower pass for the cat should be a few seconds (the emb2
  tower at 2,394 patches: 10.6 s, but that has 12 blocks of 1152 with 64-wide heads; this one has 27).
  The thinker pass is unchanged apart from the extra rows (`+0.1 s/token`, the cat adds 630).

### Board commands for 8c

```sh
# aarch64 build, copy as /root/rocket-inference/rocket-inference-p8c (+ data/ and tools/clef_fixtures_media.json)
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu --no-default-features --features npu
# stop home-llm-gateway-embed-1 / -decision-1 as for phase 6; image cases only:
python3 -c "import json;a=json.load(open('tools/clef_fixtures_media.json'));json.dump([c for c in a if c['id'] in ('media-image-small','media-image-cat','media-images-two')],open('/tmp/img_cases.json','w'))"
cd /root/rocket-inference     # relative `data/...` paths resolve from here
ROCKET_CLEF_TOWER_CHECK=1 ROCKET_PHASE_LOG=1 taskset -c 4-7 ./rocket-inference-p8c decide \
    --model-dir /mnt/nvme/models/clef-omni-int8-autoround --device npu \
    --requests /tmp/img_cases.json --out /tmp/clef-img.json
```

Expect: `clef: vision tower opened in N s (fp16 resident on the NPU (117 weights, ~1100 MiB of BOs))`, one
`vision tower: ... patches -> ... tokens in X s` line per record, and per image a `tower check` line
with merged cosine >= 0.9999 and DeepStack cosines >= 0.9999 (the fp16 emulation above: 0.99996-0.999999; a
cosine below 0.999 means a real device/layout problem: try `ROCKET_CLEF_TOWER_DEVICE=cpu` for the answers
and report the check line). The answers should match the host emulation of the shard run (small:
colorful ~0.90, shape circle ~0.92; cat: animal cat ~0.95) within the usual |dp| <= 0.03. Check the
thinker's expert-stream wall is unchanged and the peak anon (the tower's mapped pages are dropped after
every request). To measure the tower alone, compare the `vision tower` seconds against
`ROCKET_CLEF_TOWER_DEVICE=cpu`.

## Video requests (phase 8d, 2026-11-15)

`videos` requests work end to end (soundtracks since phase 8e, see "Audio requests"; the
paragraph "Soundtrack policy" below is the 8d state): `POST /v1/systemone` and `decide` accept
`"videos": ["<path>" | "data:video/mp4;base64,..." | "https://..."]` (same fetch helpers and limits as
images: 20 MiB, 10 s, 3 redirects, `ROCKET_MEDIA_FETCH=0`). Files go through ffprobe/ffmpeg (the same
binaries EmbeddingGemma 2 and d3 use; the image installs them). Several videos and images can share a
record; the block order is images, then videos whatever the JSON order.

**Soundtrack policy.** The release uses the tracks of the videos only when EVERY video of a record has
one (`use_audio_in_video = all(track is not None)`); one silent video makes it ignore all tracks. In 8d
the audio tower did not exist, so a record where the release WOULD use the tracks was a 422. **Since 8e
that 422 is gone**: those tracks are decoded and run through the audio tower, interleaved with the video
tokens (below); a record with at least one silent video is served with every track ignored, exactly as
the oracle treats it. The probe asks ffprobe for an audio stream and then decodes the first one (16 kHz
mono f32), the release's `_decode_audio is None` rule.

Files: `src/clef/vision/video.rs` (probe, time rule, ffmpeg decode, processor), `ProbedVideo` /
`PreparedVideo`, `VisionTower::encode_media` (images + videos, one tower run per item), `encode::
encode_request` (probe all -> soundtrack refusal -> decode/prepare each with the remaining token budget),
`ClefModel::media_features` (the tower, `rope_index`, `MediaFeatures::assemble`: video pad rows + DeepStack
for the video pad positions, per `media.rs` of 8b, which already had the video layout/positions),
`RecordMedia.videos`, `ClefRequest.videos`/`has_audio`.

* **Sampling** (`sample_frames`): the release's time rule, `frame.time is None or frame.time + 1e-6 <
  len(kept)/2.0` -> skip; `frame.time = pts * time_base` from ffprobe (`frame=pts` of the first video stream
  in decode/presentation order + the stream `time_base`, computed as `pts * num / den` like PyAV); an odd count
  repeats the last frame. **Frame cap** (this port's, the release has none): more than
  `ROCKET_CLEF_VIDEO_MAX_FRAMES` (default 64, i.e. 32 temporal groups) sampled frames is a 422 before any
  decode. A source above 16 MP, a file without a video stream and an ffmpeg decode failure are 422s naming
  `videos[i]`.
* **Per-frame cap** (`capped_size`): `scale = min(1, sqrt(262144 / (w*h)))`, `width = max(2,
  round(w*scale/2)*2)` (ties to even, applied even when no scaling happens: 481 -> 480). One ffmpeg run:
  `-noautorotate -map 0:v:0 -vf "select='eq(n\,i)+...',scale=W:H:flags=bilinear:in_h_chr_pos=0:in_v_chr_pos=128,
  format=rgb24" -fps_mode passthrough -f rawvideo`. **The chroma-position options matter**: PyAV's
  `to_ndarray(format="rgb24", width, height)` is swscale bilinear with swscale's default siting (4:2:0 =
  horizontal 0, vertical 128); the ffmpeg scale filter alone derives a different siting from the stream's
  chroma-location metadata and differs by up to 66/255 on a scaled clip (found by sweeping PyAV's
  `interpolation` and the filter flags; `-noautorotate` because PyAV ignores rotation side data).
* **Processor**: `smart_resize` with the VIDEO table (131,072..786,432 px, factor 32; small clips are
  upscaled), per-frame ATen antialiased bicubic uint8 resize (the same port as images), `(v - 127.5)/127.5`,
  frames paired into temporal groups, rows `[channel, temporal 2, py, px]` in merge-block order per group
  (`patchify_pairs`). The grid is `[frames/2, h/16, w/16]`; tokens = `t*h*w/4`, each video adds
  `tokens + 3` (`<|vision_start|>`, `<|vision_end|>`, `\n`) to the record's count and is refused up front
  (before any resize) when it does not fit `--max-tokens`.
* **Tower and thinker**: the 8c tower is reused unchanged (per temporal group attention segments, position
  table repeated per group, DeepStack taps of the video run feeding the video pad positions; video position
  ids `13 * group` on the time stream, `second_per_grid = 1.0`). The lazy tower and its NPU contexts are
  shared with images. NPU: attention already ran per segment (a frame group is at most 3,072 patches); the
  resident matmul now chunks above 4,096 rows (`MAX_ROWS`, bit-identical: rows are independent) so a video's
  `[P, 4320]` fp16 input BO stays bounded. `ROCKET_CLEF_TOWER_CHECK=1` also checks `videos[i]` against the f32
  CPU. aarch64 `npu` release build and clippy are clean; **not run on hardware**.

### Gates (dev host, 8 threads; oracle dumps `/tmp/opencode/clef/dump`, BF16 checkpoint for the towers)

| gate | result |
|---|---|
| decoded frames vs the oracle's `media_video_frames_0` (PyAV) | `media-video-silent` 8 frames 192x144: **0 of 663,552 bytes differ**; `media-video-test` 8 frames 480x360: **0 of 4,147,200** |
| pixel values vs `media_pixel_values_videos`, grids `[4,20,28]` / `[4,22,30]` | **0 of 3,440,640 / 0 of 4,055,040 values differ** (bit-identical) |
| scaled frames vs PyAV 19.0.2 `load_video` (clips above the frame cap: 1280x720 with AAC -> 682x384, 640x480 -> 592x444, NTSC 30000/1001 fps) | **0 differing bytes** in all three (6 frames each; frame count and sizes equal) |
| ids vs oracle `input_ids` (`media-video-silent`, 796 tokens, 560 video pads) and `rope_index` vs `position_ids` | equal |
| tower f32 CPU (`media-video-silent` 2,240 patches in 4.6 s; `media-video-test` 2,640 in 5.4 s) | cosine 1.000000000 on every tap; rel-to-peak: blocks 0/8/16/24 <= 8.1e-6, block 26 5.9e-5 / 1.3e-4, merged **1.9e-5 / 2.2e-5**, DeepStack 3.6e-6 / 2.3e-5 / 2.3e-5 and 2.1e-6 / 1.4e-5 / 2.7e-5 |
| tower fp16 emulation (what the NPU should reach) | silent: cosine 0.99998739 (block 26), merged 0.99999233, DeepStack 0.99999874 / 0.99999388 / 0.99999146; test: 0.99998322, 0.99999066, 0.99999939 / 0.99999724 / 0.99999285 |
| answers, BF16 f32 CPU thinker (`media-video-silent`) | max \|dp\| **1.2e-3**, top-1 equal, 796 tokens, 35 s (image-small on the same path: 1.9e-3) |
| answers, emulated int8 thinker on the AutoRound shards | max \|dp\| 8.0e-2, top-1 equal, 66.7 s. The 8b gate with the ORACLE's own tower outputs gives 9.7e-2 on the same `motion` question (p = 0.38, final-norm cosine 0.945, 17.5k router set mismatches), so this is the int8 thinker's band on a long record, not the video path |
| `serve` smoke (emulated engine, `--max-tokens 4096`) | silent video by path and as a `data:` URL: 200, 711 tokens, `motion` 0.298 (59 s each); `test.mp4` (has AAC), a `data:` clip with AAC, `audio` field, a non-video file, 8 copies over the token cap: all 422 with the reason; `/health` stays ok |

Unit tests (`cargo test -p rocket-inference --no-default-features`: 306 pass, 17 ignored): the time rule
(8/25/30 fps, the 1e-6 tolerance, missing pts), the frame-cap rounding, the VIDEO smart-resize table, the
temporal pairing layout, the budget refusal, the request parsing, soundtrack refusal vs "one silent video
makes it ignore every track", data-URL probing, the frame cap on a generated 40 s clip, the 20-frame
upscaled clip, `encode_media` slots + "a `[3,h,w]` run equals three `[1,h,w]` runs". The ignored gates run
with `cargo test --release -p rocket-inference --no-default-features -- --ignored --nocapture
clef::vision::tests::the_f32_tower clef::vision::tests::the_fp16 clef::vision::tests::video_answers`
(`HF_HOME=/mnt/BtrTrashCan/huggingface`). The PyAV scaled-frame dump is `tools/clef_pyav_video_dump.py`
(needs `av` + `numpy` in a scratch venv; the oracle venv no longer has `av`).

**Caveats.** (1) Decoder-version sensitivity: the bit-identity above is ffmpeg n9.0.2 CLI vs PyAV 19.0.2 (its
own bundled libswscale). An ffmpeg of another major (the board's Debian 7.1.5) can convert YUV->RGB 1-2/255
differently (d3 saw 2/255 on the board vs the dev host, AGENTS.md); not measured for Clef. (2) `ffprobe -show_frames`
decodes the whole video to list pts (cost ~ one decode; fine for the frame cap's clip lengths, slow for long 1080p
files, which the 20 MiB fetch limit and the 64-frame cap bound for URLs/uploads; local paths are unbounded).
(3) Frame times use ffprobe's `pts` of the first video stream; a stream with unset pts keeps no frames (PyAV:
`time is None`). (4) A still image file given as a video (one frame) is accepted, as PyAV would. (5) Mixing a video with
sound and a silent one works, but the release then ignores the track of the audible one too.

### Board commands for 8d

```sh
# aarch64 build as for 8c; copy as rocket-inference-p8d, plus data/ (the three mp4s) and tools/clef_fixtures_media.json
python3 -c "import json;a=json.load(open('tools/clef_fixtures_media.json'));json.dump([c for c in a if c['id']=='media-video-silent'],open('/tmp/vid_cases.json','w'))"
cd /root/rocket-inference
ROCKET_CLEF_TOWER_CHECK=1 ROCKET_PHASE_LOG=1 taskset -c 4-7 ./rocket-inference-p8d decide \
    --model-dir /mnt/nvme/models/clef-omni-int8-autoround --device npu \
    --requests /tmp/vid_cases.json --out /tmp/clef-vid.json
```

Expect `clef: videos[0]: 192x144 source, 8 sampled frames at 448x320 -> grid [4, 20, 28], 560 tokens`, the tower
line (`1 video(s), 2240 patches -> 560 tokens`; the CPU does it in ~4 s here, the NPU tower should be a few seconds), and
`tower check videos[0] vs f32 CPU: merged cos >= 0.9999, deepstack0..2 >= 0.9999` (fp16 emulation: 0.99998-0.999999;
below 0.999 is a device/layout problem: try `ROCKET_CLEF_TOWER_DEVICE=cpu`). The answer should be near the dev host's
`motion 0.298` (emulated int8) within the usual |dp| ~0.1 on this record. Check the ffmpeg frames on the board against
the dev host (`ffmpeg` 7.1.5 vs n9.0.2): the `video.frames` count and, if answers drift, dump both sides' frames. (Since
8e a request with `data/test.mp4` is served as audio-in-video: see "Board commands for 8e".)

## Audio requests (phase 8e, 2026-11-15)

`audio` requests and audio-in-video work end to end: `POST /v1/systemone` and `decide` accept
`"audio": ["<path>" | "data:audio/wav;base64,..." | "https://..."]` (same fetch helpers and limits as images and
videos: 20 MiB, 10 s, 3 redirects, `ROCKET_MEDIA_FETCH=0`). Any container ffmpeg opens works: the FIRST audio
stream is decoded (an mp4 with a soundtrack is a valid `audio` entry). Block order in the record: images,
explicit audio clips, videos, whatever the JSON order. A video whose soundtrack the release would use (EVERY video
of the record has an audio stream with at least one sample) is decoded alongside and interleaved as audio-in-video.

Files: `src/clef/audio.rs` (config, tensor index, the tower, lazy), `audio/mel.rs` (Whisper log-mel),
`audio/decode.rs` (ffmpeg decode, caps), `audio/tests.rs`; plumbing in `encode::encode_request`
(`RecordMedia.audio: Option<MelBatch>`), `ClefModel::{audio_tower, media_features}`, `ProbedVideo::soundtrack`,
`ClefRequest.audio`. The vision tower's `Exec` placements are shared: `compute::make_exec` now takes the linears
(`&[&Lin]`), so the NPU packing, fp16 emulation and fallback code is common.

* **Decode** (`audio/decode.rs`): `ffmpeg -map 0:a:0 -ac 1 -ar 16000 -t cap+1 -f f32le` = the release's PyAV
  `_decode_audio` (first stream, mono, 16 kHz, f32). **Bit-identical to the oracle's decoded samples** for every
  fixture (speech5s.wav 80,000 samples, the AAC 22.05 kHz / 44.1 kHz tracks of `clef-video-audio-small.mp4`
  32,000 / `test.mp4` 64,000: 0 differing samples). The same decoder gives a video's soundtrack
  (`None` = no audio stream or no samples = "silent", the release's rule). A clip under 400 samples (one STFT
  window) is a 422 (`audio[i]: the audio is too short`); a file without an audio stream, a missing file or an
  undecodable `data:` blob is a 422 naming `audio[i]`/`videos[i]`.
* **Length cap** (this port's, the release has none): `ROCKET_CLEF_AUDIO_MAX_SECONDS` (default **300**, 3,900
  tokens) per clip, a 422 over it; decoding stops one second past the cap so a huge file costs bounded memory.
  The audio block also counts against `--max-tokens` like every media block (`audio[i]` is named in the refusal).
* **Log-mel** (`audio/mel.rs`): `WhisperFeatureExtractor(feature_size 128, n_fft 400, hop 160, padding=longest)`
  = periodic Hann(400), `torch.stft(center=True)` (reflect padding 200), power over the first `N/160` frames (the
  last STFT frame is dropped), the 201x128 Slaney-scale Slaney-norm filter bank (f32-rounded like the extractor),
  `log10(clamp(1e-10))`, per-clip `max(x, max - 8)`, `(x + 4)/4`; valid frames = `min(ceil(n/160), N/160)` (the
  mask is `[::160]` trimmed by one when `N % 160 != 0`). **All clips of a record are ONE batch: each is zero-padded
  to the longest BEFORE the STFT** (a short clip's last frame sees zeros instead of reflect padding, clef-oracle.md
  §7.4); the explicit clips and the soundtracks share the batch. The FFT is f64 (rustfft) and the filter matmul f64
  over the f32 filters. Gate vs the oracle's `media_input_features`: `media-audio-5s` **max |d| 4.6e-6**
  (0 of 64,000 values over 1e-5); the AAC tracks 5.0e-5 / 6.3e-5 on 329 / 608 of 25,600 / 51,200 values over
  1e-5 **and that is torch's own f32 rounding, not this port**: the same samples through torch f64 differ from
  the oracle's f32 dump by 4.985e-5 and from this port by 1e-7 (low-energy mel bins 8 decades under the peak
  where f32 FFT noise shows). `media-mixed-av` (a 5 s explicit clip + a 2 s track, padded together) agrees to 5.0e-5
  as well, valid lengths `[500, 200]` equal.
* **Tower** (`audio.rs`): the `manual_audio` spec, host f32, in the order of the oracle doc: 100-frame chunks
  (the tail zero-padded), 3 x `Conv2d(k3 s2 p1)` + GELU(erf) (480 channels; `conv2d1` has `K = 9` and runs on the
  host in f32, `conv2d2/3` as im2col matmuls with `K = 4320`, batches of 8 chunks to bound the matrices),
  `(c, f)` flatten -> `conv_out` (7680 -> 1280, no bias) + the sinusoid table (positions 0..12, restarting per
  chunk), only the valid steps of each chunk (`audio_out_len`), attention windows of `max_valid * 8` = 104 tokens
  per clip (full attention inside, never across), 32 pre-LN layers (eps 1e-5, 20 x 64 heads, no rotary, GELU
  erf FFN 5120), `ln_post`, `proj1` + GELU, `proj2` (-> 2048). Weights are BF16 in both checkpoints (BF16 repo shard
  1, AutoRound repo shard 6), mmapped and widened per use (nothing copied on the host); the tower opens on the
  **first request with audio** (`clef: audio tower opened in ...`), independent of the vision tower (an audio-only
  record never touches it; a clef-pack has neither: 422).
* **Placements** (`ROCKET_CLEF_TOWER_DEVICE=cpu|emu16|npu`, the same switch and default as the vision tower):
  `cpu` f32; `emu16` the fp16 emulation (weights, activations in and out of every linear in fp16, f32 accumulation;
  an overflow reruns the batch on the CPU); `npu` (aarch64 + `npu` builds, **compiles and clippy-clean, never ran on
  hardware**): every linear resident fp16 (`conv2d2`, `conv2d3`, `conv_out`, 32 x q/k/v/out/fc1/fc2, `proj1/2` = 197
  weights, **~1.2 GiB of BOs**, packed when the tower opens: expect 15-40 s of pack time and budget the memory next to
  the vision tower's 1.07 GiB and the thinker; the tower stays resident once opened). **Window attention stays on
  the host in f32 in every placement** (`ROCKET_CLEF_AUDIO_ATTN=device` asks the placement to run it via
  `flash_attn` per window): a 104-token window is dispatch-bound on the NPU (about 1,200 dispatches for 300 s of
  audio, against ~2 GFLOP per layer for the CPU, ~2 s in all). `ROCKET_CLEF_TOWER_CHECK=1` also reruns the audio batch on
  the f32 CPU and logs `tower check audio vs f32 CPU: features cos ... rel-to-peak ...`.
* **Audio-in-video** (`encode_request`): when every video has a soundtrack with samples, the tracks are appended to
  the explicit clips (explicit first, then one per video in video order), the batch's mel is computed once, and
  `MediaLayout { audio_frames, explicit_audio, use_audio_in_video: true }` drives `build_media_ids` (the
  `interleave_video_audio` rule: video token k at time `group * 13`, audio token j at time j, video first on ties) and
  `rope_index` (the audio rule with the 3-D branch). Each video's `<|vision_start|><|audio_start|> ...
  <|audio_end|><|vision_end|>` block counts toward the token cap with its audio pads.
* **Positions**: an audio-only record gets plain `0..S` positions in all three streams (HF `get_rope_index`'s else
  branch: no image/video grid); audio next to an image/video takes the 3-D branch. Both were already in 8b; the gates
  below check them against `position_ids`.

### Gates (dev host, 8 threads; oracle dumps `/tmp/opencode/clef/dump`, BF16 checkpoint for the f32 arms)

| gate | result |
|---|---|
| decoded samples vs the dump (`media_audio_samples_*`), 5 clips over the 4 audio records | **0 differing samples** in all |
| log-mel vs `media_input_features` | 4.6e-6 (audio-5s), 5.0e-5 / 6.3e-5 / 5.0e-5 (the AAC tracks and the mixed batch): torch's f32 rounding, see above; valid lengths equal |
| ids / positions / audio-in-video flag vs the oracle (`audio_records_get_the_oracle_ids_positions_and_counts`) | equal for `media-audio-5s` (65 pads, text positions), `media-video-audio-small` (V\*140 A\*13 interleave, 280 + 26), `media-video-test` (660 + 52), `media-mixed-av` (explicit 65 + track 26, image, video) |
| f32 tower vs the dumped taps (`the_f32_tower_matches_the_oracle`, release, 0.6-1.8 s per record) | cosine **1.000000000 on every tap** of all four records; rel-to-peak: `conv_out` 3.9e-7, layer 0 1.0e-6, layer 15 1.4e-6, layer 31 4.0e-6..2.3e-5, `ln_post` 1.4e-6..2.7e-5, **features 3.3e-6 / 6.6e-6 / 2.6e-5 / 8.9e-6** (max abs 8.4e-7..4.9e-6) |
| same tower on the AutoRound repo's shard 6 (`media-audio-5s`) | identical numbers (the tower tensors are the same BF16) |
| fp16 emulation (`the_fp16_emulated_tower_follows_the_oracle`) | features cosine **0.9999997 / 0.9999996 / 0.9999993 / 0.9999963** (5s / small / test / mixed-av), rel-to-peak 8e-4 .. 1.0e-2 (the mixed-av's 91 tokens peak at layer 31 with rel 5.1e-3, cos 0.9999988) |
| answers, BF16 f32 CPU thinker (`audio_answers_match_the_oracle_on_the_bf16_cpu_path`) | max \|dp\| **3.6e-7** (`media-audio-5s`, 302 tokens, 23.8 s), **2.3e-5** (`media-video-audio-small`, 544 tokens), **2.1e-6** (`media-video-test`, 952), **2.1e-5** (`media-mixed-av`, 678); top-1 equal everywhere |
| answers, emulated int8 thinker on the AutoRound shards | max \|dp\| 9.3e-3 / 4.9e-3 / **5.3e-2** (`media-video-test`, the 952-token record, the int8 thinker's band as on the other long records) / 1.4e-3; top-1 equal everywhere |
| `serve` smoke (emulated engine, `--max-tokens 4096`, tower on the CPU) | `audio` speech5s.wav: **200**, 283 tokens, 25 s, `kind speech 0.8903`; `audio` = an mp4 with a soundtrack: 200, 244 tokens; audio-in-video `clef-video-audio-small.mp4`: 200, 526 tokens, 44 s; audio-in-video `test.mp4`: 200, 932 tokens, 77 s; image + explicit audio + audio-in-video record: 200, 667 tokens, 56 s (the audio tower logs `2 clip(s), 700 mel frames -> 91 tokens`). **422**: a text file (`the file has no audio stream`), a missing path, an undecodable `data:` blob (ffmpeg's reason), a PNG; with `ROCKET_CLEF_AUDIO_MAX_SECONDS=3`: speech5s (`the audio is longer than 3 s`) and a 4 s soundtrack (`videos[0] soundtrack: ...`). `/health` stays ok |

Unit tests (`cargo test -p rocket-inference --no-default-features`: 324 pass, 22 ignored): Slaney scale and filter
bank, the padding-before-STFT artefact, the trimmed-mask valid length, the log-mel floor, chunking and window
bounds, the sinusoid table, im2col + matmul == naive stride-2 conv, `conv2d1` == naive conv + GELU, a synthetic
2-layer tower checkpoint (open, run, deterministic, tokens of another clip and another window do not move a token,
a clip alone == inside a batch, the fp16 emulation follows f32), the request parsing, encode errors (undecodable,
missing, no stream, too short, over the cap, over the token cap, data: URL == path). The ignored gates:
`cargo test --release -p rocket-inference --no-default-features clef::audio -- --ignored --nocapture`
(`HF_HOME=/mnt/BtrTrashCan/huggingface`; `CLEF_IDS=a,b` selects the answer records). The 8d unit test that expected a
422 for a soundtrack now expects the audio-in-video layout (`soundtracks_are_used_only_when_every_video_has_one`).

**Caveats.** (1) The f32 tower is as close to the oracle as the oracle's own HF-vs-plain-torch selfcheck (1e-6..3e-5);
the log-mel differs from the oracle's f32 dump by torch's own rounding at the 1e-5..6e-5 level, which does not move the
tower output visibly (the tower gates above start from the dumped features, the answer gates from this port's own mel).
(2) A mel batch whose longest clip is shorter than 25 ms or any clip under 400 samples is a 422. (3) Mel frames beyond a
clip's valid length are computed but never read (padding, as in HF). (4) Not run on the board: the ~1.2 GiB resident
fp16 pack, ffmpeg 7.1.5's AAC decode (the CLI of the dev host's n9.0.2 is bit-identical to PyAV; the board's Debian
7.1.5 was not compared), the NPU numerics. (5) `ROCKET_CLEF_AUDIO_MAX_SECONDS` is read at request time, a clip is
fully decoded to RAM (300 s = 19 MB of f32 plus 15 MB of mel features).

### Board commands for 8e

```sh
# aarch64 build as for 8c; copy as rocket-inference-p8e, plus data/ and tools/clef_fixtures_media.json
python3 -c "import json;a=json.load(open('tools/clef_fixtures_media.json'));json.dump([c for c in a if c['id'] in ('media-audio-5s','media-video-audio-small','media-video-test','media-mixed-av')],open('/tmp/aud_cases.json','w'))"
cd /root/rocket-inference
ROCKET_CLEF_TOWER_CHECK=1 ROCKET_PHASE_LOG=1 taskset -c 4-7 ./rocket-inference-p8e decide \
    --model-dir /mnt/nvme/models/clef-omni-int8-autoround --device npu \
    --requests /tmp/aud_cases.json --out /tmp/clef-aud.json
# the tower on the CPU for comparison (answers + wall of the tower alone)
ROCKET_CLEF_TOWER_DEVICE=cpu taskset -c 4-7 ./rocket-inference-p8e decide ... (same arguments)
```

Expect `clef: packed a tower on the NPU: 197 weights, ~1240 MiB of BOs in N s` once, `clef: audio tower opened in N
s (fp16 resident on the NPU (197 weights, ~1240 MiB of BOs), window attention on the host (f32))`, per record
`clef: audio tower: 1 clip(s), 500 mel frames -> 65 tokens in X s` and `tower check audio vs f32 CPU: features cos
>= 0.9999 rel-to-peak < 0.02` (the host fp16 emulation above gives 0.9999963..0.9999997; below 0.999 is a
device/layout problem: rerun with `ROCKET_CLEF_TOWER_DEVICE=cpu` and report the check line). The answers should match
the dev host's emulated-int8 numbers (`media-audio-5s`: `speech` 0.8596 oracle / ~0.89 int8 emulated) within |dp| ~0.1.
`ROCKET_CLEF_AUDIO_ATTN=device` is an experiment: compare the tower seconds with the default. Check memory: `RssAnon`
and the BO total (`ROCKET_MEM_LOG=1`) with both towers opened (1.07 + 1.2 GiB) next to the thinker.

## Not done

`media_kwargs`, the audio/vision tower unload when idle (both stay resident once opened), attention scores/AV on the NPU of
the thinker, a CPU `sdot` expert path, a board run of the vision and audio towers, an NPU `Conv2d` for `conv2d1`.

## Board results, phase 6 (2026-11-14, rock-5b-plus)

`decide --pack-dir /mnt/nvme/models/clef-pack-g128 --device npu`, `taskset -c 4-7`, 600 MHz
module, `home-llm-gateway-embed-1`/`-decision-1` stopped for the runs and restored, board
load ~10 from the other services (+-20 %). Pack on the btrfs NVMe (30 GB, O_DIRECT works).
Model load 2-4 s, 496 MiB anon, 927 MiB of BOs for the resident attention projections.

| record | tokens | total | moe wall | qkv+oproj | attn (CPU) | peak anon |
|---|---|---|---|---|---|---|
| tiny (1 noul question) | 142 | 38.7 s | 29.7 s | 4.8 s | 0.5 s | 618 MiB |
| checkout-errors | 295 | 51.6 s | 38.2 s | 8.7 s | 1.4 s | 690 MiB |
| long-incident | 800 | 103.2 s | 66.7 s | 21.8 s | 7.2 s | 887 MiB |

- Floor ~30 s per pass at any short prompt (all 128 experts hit and streamed: 5-6k expert
  loads per request); roughly +0.1 s per extra token after that. Expert reads are fully hidden
  (`read_wait` 1-10 thread-s); `npu` thread-s dominate, `rewrite` is 35-38 thread-s.
- Answers vs the f32 BF16 oracle: checkout-errors technical 0.8171 (oracle 0.8327), urgency
  0.9272 (0.9341), outage 0.9547 (0.9552); long-incident owner payments 0.8444 (0.8202),
  severity 2.2406 (2.2751), double_charge 0.969 (0.9674), owners_assigned 0.0163 (0.0169).
  Top-1 equal everywhere; |dp| <= 0.025, in line with the host emulation.
- Not yet measured on the board: >= 2k tokens, `ROCKET_CLEF_ATTN_DEVICE=cpu`, `serve`, batching.

## Board test, phase 7 (not run yet)

```sh
# binary: aarch64 npu release build, scp to /root/rocket-inference/rocket-inference-p7
cd /root/rocket-inference
# stop home-llm-gateway-embed-1 / -decision-1 as for phase 6; run on cores 4-7
ROCKET_PHASE_LOG=1 taskset -c 4-7 ./rocket-inference-p7 serve --pack-dir /root/models/clef-pack-g128 \
  --device npu --port 8399 --max-tokens 4096 --batch-window-ms 5000 --batch-max 8 &
# four concurrent requests (tools/clef_fixtures.json cases with "model": "clef-omni"):
for f in checkout-errors invoice noul-only choice-routing; do
  curl -s localhost:8399/v1/systemone -H 'Content-Type: application/json' -d @$f.json > b_$f.out & done; wait
```

Expect: one `clef batch batch=4 ... passes=1 experts_read~5.5k` line and four
`decision request ... batch=4` lines; `ROCKET_PHASE_LOG` one `clef phases` line for the
batch; total wall about 30 s + 0.1 s x rows. Then send the same four with
`--batch-window-ms 0` one at a time and compare the `*.out` files with `cmp`: they must be
byte-identical (if not, the NPU int8 matmul is not row-independent across M classes: report
the differing probabilities; the emulation says they should match). Watch `peak_anon`
(<= ~1 GiB) and the `experts_read` counter against 4 x ~5.2k for the singles.

## Board results, phase 7 batching (2026-11-14, rock-5b-plus)

`serve --pack-dir /mnt/nvme/models/clef-pack-g128 --device npu --max-tokens 4096`, `taskset -c 4-7`,
same board conditions as phase 6 (services stopped for the run, load ~10). Four fixture
requests (295 + 256 + 174 + 192 tokens):

| mode | wall | expert loads | peak anon |
|---|---|---|---|
| 4 concurrent, `--batch-window-ms 5000 --batch-max 8` (one batch, 917 rows) | **107 s** (5 s of it is the window; compute 101.6 s) | 5,590 | 935 MiB |
| 4 sequential, `--batch-window-ms 0` | 179 s (51.4 + 44.5 + 41.8 + 40.8 s) | 20,497 | 630-705 MiB |

- **The 4 answers are byte-identical** between the batched and the single runs on the NPU, so
  the int8 `matmul_gw` is row-independent across M classes and chunks (what was only proven on
  the emulation before).
- Batching saves the 30 s streaming floor per extra record: 1.67x here; the gain grows with
  the batch (batch of 8 short records ~ 30 s + 0.1 s/row).
- The batch compute (101.6 s for 917 rows) is above the 30 s + 0.1 s/row estimate: at ~900 rows
  the per-row work (expert matmuls at M_e~56, qkv/o 23 s, attention) dominates. Budget
  ~35-40 s + ~0.08 s/row.

## Deployment (text-only, current)

Prerequisites on the board: the pack at `/mnt/nvme/models/clef-pack-g128` (30 GB, copy with
`rsync -a` from the dev host or rebuild with `rocket-inference clef-pack --src <autoround dir>
--out <dir>`), a `rocket` kernel module at 600 MHz (as for the other families), and an image
or binary built with `--features npu` (CI `ghcr.io/royalcat/rocket-inference:latest` after
merging this branch).

```sh
rocket-inference serve --pack-dir /models/clef-pack-g128 --device npu \
    --max-tokens 4096 --batch-window-ms 5000 --batch-max 8 --batch-max-tokens 4096 \
    --port 8385 --model-name clef-omni
```

- `--max-tokens` is the per-record cap (CPU attention is O(S^2) and per-expert outputs are
  `rows*8*2048*4` bytes; 4096 keeps a batch under ~1.2 GiB anon + ~1 GiB of BOs).
- Request timeouts on the client side must exceed the batch window plus the batch compute
  (minutes). `/health` and `/v1/models` answer during a batch.
- Container: mount the pack read-only (O_DIRECT works on btrfs/ext4 bind mounts; set
  `CLEF_PACK_NO_DIRECT=1` if the filesystem refuses it), pin CPUs 4-7, expose `/dev/accel`,
  `mem_limit` 3g is enough (NPU buffers are not charged to the cgroup, ~1 GiB more on top).
- Image inputs work since phase 8c (needs the BF16 checkpoint or the AutoRound shards, a `clef-pack` has no
  tower: HTTP 422 for images; +~1.07 GiB of NPU BOs on top of the thinker once the first image request has
  packed the tower). `audio` and soundtrack-bearing videos are still a 422; silent videos work (phase 8d, "Video requests").
- The measured per-request times and the knobs are in the phase 6/7 sections above.

## Step 0: serving straight from the AutoRound shards (no repack)

> **STATUS 2026-11-14: code complete, dev-host verified, board-verified (see "Board results,
> step 0" below).** The `clef-pack` tool, `--pack-dir` and
> `PackDir` stay as an explicit opt-in; they are no longer on the default path.

```sh
rocket-inference serve -hf moeshawky/clef-omni-int8-autoround --device npu ...   # downloads
rocket-inference serve --model-dir /models/clef-omni-int8-autoround --device npu ...
rocket-inference decide --model-dir <that dir> --requests ... [--device cpu]
```

`-hf <owner>/<name>` **without** `:quant` is a safetensors snapshot download
(`util/hub.rs::resolve_snapshot`): `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`joint_head.safetensors`, `joint_head_config.json`, `model.safetensors.index.json` and every
shard that holds a `thinker.*` tensor (shards 1-7 = ~37 GiB for this repo: shard 7 holds
`embed_tokens`/`lm_head` and the visual/audio towers; only shard 8, talker/code2wav, is skipped); a shard already on disk whose size matches its own header is not
fetched again. The repo (checked against the HF API 2026-11-14) ships `joint_head.safetensors`
and `joint_head_config.json`, so nothing else is needed. The dir is `$HF_HOME/models/
moeshawky--clef-omni-int8-autoround` (or `$HF_HUB_CACHE`, else `/tmp/rocket-hf`).

Detection: the config/joint-head rule above gives `Family::Clef`; the pipeline then picks the
weights by what is in the directory: `manifest.json` = pack, else a shard index that has
`...experts.0.gate_proj.qweight` = AutoRound shards (CPU f32 or emulated/NPU int8 engine),
else the BF16 checkpoint (CPU only). `--device npu`/`auto`(NPU build) accept the AutoRound dir.

### Tensor layout in the shards (measured, `model.safetensors.index.json` + shard headers)

8 shards of ~5.37 GB (index `total_size` 41,323,349,090; 65,258 tensors: I32 37,248 qweight/
qzeros, F16 18,624 scales, BF16 9,386). Layers 0-7 are in shard 1, 8-15 in 2, ... 42-47 in 6;
shards 7-8 are talker/code2wav only. Layer 0 expert 0 of shard 1 (offsets are absolute file
offsets, data starts at 1,264,488):

| tensor (`thinker.model.layers.0.mlp.experts.0.`) | dtype, shape | file range |
|---|---|---|
| `down_proj.qweight` / `.qzeros` | I32 `[192,2048]` / `[6,512]` | 1,264,488-2,837,352 / -2,849,640 |
| `gate_proj.qweight` / `.qzeros` | I32 `[512,768]` / `[16,192]` | 2,849,640-4,422,504 / -4,434,792 |
| `up_proj.qweight` / `.qzeros` | I32 `[512,768]` / `[16,192]` | 4,434,792-6,007,656 / -6,019,944 |
| expert 1 `down_proj.qweight` | | starts at **6,019,944** (the next expert follows directly) |
| `down_proj.scales`, `gate_proj.scales`, `up_proj.scales` | F16 `[6,2048]`, `[16,768]`, `[16,768]` | 5,287,779,688-5,287,853,416 (3 x 24,576 B) |
| expert 1 scales | | start at 5,287,853,416 |

So tensors are written in alphabetical order inside two regions: all `qweight`/`qzeros` of
the shard's experts (expert-major, down|gate|up, 4,755,456 B per expert) and then, at the end
of the shard, all `scales` (73,728 B per expert, same order). **An expert is two contiguous
runs: 2 aligned reads** (4.76 MB + 74 KB, each rounded out to 4 KiB, ~4.84 MB total vs
4.79 MB in a pack record). The extents are computed from the headers (`plan_extents`, runs
closer than 64 KiB merge), so another layout would only cost more reads. Attention
`qweight`/`qzeros`/`scales` sit in the same two regions; BF16 norms/router/`embed_tokens`/
`lm_head` elsewhere. All `qzeros` words are `0x7f7f7f7f` (checked at every unpack).

### Implementation

`src/clef/autoround_store.rs`: `ShardStore` (`ExpertReader` + `ClefStore`) opens the thinker
shards (mmap for the headers/resident tensors, one `DiskFile` per shard for aligned `O_DIRECT`
reads; `CLEF_PACK_NO_DIRECT=1` falls back to buffered + `fadvise(DONTNEED)` as for packs).
`read_expert` reads the extents into thread-local aligned scratch, then converts in RAM on the
reader thread with the packer's own `autoround::unpack` into the *pack record layout* (codes
`q ^ 0x80` as `[N,K]` i8, scales `[N,K/128]` f16, gate|up fused), so the engine, batching and
workers are untouched and a record is **bit-identical** to the pack's (test
`real_shards_equal_the_real_pack_bit_for_bit`, 10 experts across layers/shards vs
`/home/royalcat/models/clef-pack-g128`, skipped when absent; the synthetic-checkpoint tests
compare every expert and the whole backbone). `autoround::unpack` now writes 16-byte runs
(4 k-blocks per row): 5.5 -> 3.5 ms of single-core conversion per expert on the dev host
(2.1 ms wall with all cores). `engine::ClefStore` is the new seam: `PackDir` and `ShardStore`
implement it; the Emu/CPU/NPU dense paths take `Arc<dyn ClefStore>`. Resident weights are
used as stored: embeddings/norms/router/`lm_head` BF16 mapped and widened per use; the int8
attention is unpacked from `qweight` (`resident_linear`: per call on the emulated path, once
at load into the NPU weights). `AutoRoundWeights` is the backbone's `WeightSource` for the
f32 CPU path (experts widened from the same records).

Failure policy: `open` validates the index, every shard's presence and exact length (against
its own header), and the dtype/shape of all 55k+ tensors the text path reads, naming the
problem (exit code 1, e.g. `shard ... missing` / `inconsistent data offsets`); a read failure
at run time (`ExpertReader::read`, packs included) prints `clef: FATAL: reading layer L expert
E failed: ...` and `exit(1)` for the supervisor, no 503 mapping.

### Dev-host results (2026-11-14, page cache warm)

`ROCKET_CLEF_EMU=1 decide` on fixture `checkout-errors` (295 tokens, 5219 expert reads, 8
threads): shard dir 28.4 s (user 4m41s before the unpack tiling), pack dir 25.4 s (user
2m32s); the `cases` of both outputs are **identical** (department technical 0.8171, urgency
1.8781, outage 0.9547). Only the top-level `model` label differs (dir name). Conversion is the
cost of not repacking: ~3.5 ms of CPU per expert (vs 0 for a pack), ~18 CPU-s per 295-token
pass on this host, more on the A76s; measure it on the board (reader `read_io`/`read_wait` in
`ROCKET_PHASE_LOG=1`).

### Board command (for the caller)

```sh
# dir: the dev checkout, shards copied as is (shards 1-6 + joint head + tokenizer + configs + index)
ROCKET_PHASE_LOG=1 taskset -c 4-7 ./rocket-inference decide \
    --model-dir /mnt/nvme/models/clef-omni-int8-autoround --device npu \
    --requests tools/clef_fixtures.json --out /tmp/clef-shards.json
# compare with the pack run: same `cases`; check `read_wait`/`read_io` and the 5219-expert pass wall
```

## Board results, step 0 (2026-11-14, rock-5b-plus)

`decide --device npu`, `taskset -c 4-7`, 600 MHz module, `home-llm-gateway-embed-1`/`-decision-1`
stopped for the runs and restored healthy. Shards (shards 1-7, 37 GiB, rsynced to
`/mnt/nvme/models/clef-omni-int8-autoround`) vs the pack (`--pack-dir
/mnt/nvme/models/clef-pack-g128`), same window; board disk speed drifts (the pack's own
`read_io` went 39 -> 58 -> 72 thread-s on checkout-errors during one hour), so only
interleaved pairs are comparable. **`cases` are byte-identical to the pack run for both
records, in every shard variant.** Shard 7 must be present (a rsync of shards 1-6 fails at
open with ENOENT; the hub downloader already fetches 1-7).

| record (tokens) | pack moe wall | shards, reader converts (v0) | converter stage (final) | peak anon pack / shards |
|---|---|---|---|---|
| checkout-errors (295) | 41.9 / 61.3 / 72.4 s (drift) | 73.8, 77.4 s | **50.0, 51.6 s** (pack 43.4 s same window) | 686-691 / 819 MiB |
| long-incident (800) | 73.6, 73.5, 72.8 s | 91.4, 93.0 s | **77.8, 88.5 s** | 896-899 / 1020 MiB |

Phase line (long-incident): pack `read_io=39.5` thread-s (7.1 ms per 4.8 MB expert, 675 MB/s
O_DIRECT), `read_wait` 1-2; shards v0 `read_io=96-98` (reader thread = read 58 + unpack ~40 serial
on the A76; `read_wait` 46-56, the reader is the bottleneck: 1.25x the pack wall). Even raw I/O
of the shard layout is slower than the pack (58 vs 39 thread-s: two reads per expert, 465 vs
675 MB/s). Tried: (a) 3-way parallel unpack inside the reader: 91 s, no gain (the cores are
shared with the consumers); (b) unpack on the consumer threads (`glue` 55 thread-s): 96.8 s,
worse (consumers are the critical path); (c) **kept**: reader thread = I/O only, then
`ROCKET_CLEF_CONVERTERS` (default 2, max 4) converter threads unpack in parallel and deliver in stream
order (`ExpertReader::{needs_finish, finish}`, `ShardStore::{read_raw, convert}`, expert buffers
are `stride + raw extents`), `read_wait` 1-6 thread-s. Result +7-19 % wall vs the pack,
+130 MiB anon (buffers). The remaining cost is CPU: the unpack is ~10-14 ms per expert on
the A76s (~55-80 CPU-s per pass) on cores shared with the NPU consumers. Verdict: shard
streaming is usable and gives the same answers; **the pack is still the faster production
path (~10 %)**, and the lever left is a NEON `autoround::unpack` (the GPTQ word -> `[N,K]` i8
transpose) or a one-time repack. Dev-host: fmt/clippy clean, `cargo test -p rocket-inference
--no-default-features` 260+ pass (incl. `real_shards_equal_the_real_pack_bit_for_bit`).
Board artifacts (scratch, can be deleted): `/root/rocket-clef-s0/` (binaries v0-v3, run logs,
outputs); the shard dir at `/mnt/nvme/models/clef-omni-int8-autoround` is the working copy.

**Model storage (2026-11-15).** Both checkpoints live only in the shared HF cache
(`HF_HOME=/mnt/BtrTrashCan/huggingface`): `Cloudflare/clef-omni` (shards 1-13 + config/tokenizer/
joint head; shards 14-15 are talker/code2wav only and skipped) and
`moeshawky/clef-omni-int8-autoround` (shards 1-7; shard 8 + `vllm_patch.py` skipped). Tests resolve
them via `clef::testutil::{model_dir, autoround_dir}` (`CLEF_MODEL_DIR` / `CLEF_AUTOROUND_DIR`, else the
newest cache snapshot). Oracle venv rebuilt at `/mnt/BtrTrashCan/venvs/clef` (`$CLEF_VENV`). 8c gates re-run
on the real BF16 checkpoint: f32 tower cosine 1.000000000 (rel-to-peak <= 1.6e-4), image answers max|dp|
1.9e-3 (small) / 1.6e-4 (cat), top-1 equal. Known: the 8b tight backbone gate
(`media_backbone_matches_the_oracle_on_the_bf16_cpu_path`) fails on `media-image-cat` (layer cosine 0.99981,
280 router set mismatches) and `media-video-audio-small` (0.999985, 25), answers still within 1.2e-4.
