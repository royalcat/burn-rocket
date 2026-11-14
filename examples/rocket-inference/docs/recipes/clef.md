# Clef-Omni (Cloudflare) — typed decisions, text-only, CPU

> **STATUS 2026-11-14 (phase 6): text-only. CPU path dev-host verified; the NPU expert
> streaming path is code complete and verified on the dev host through its host
> emulation (`ROCKET_CLEF_EMU=1`), an aarch64 cross build and unit tests. NOT yet run on the
> board.** Images, audio and video are refused (422) until the multimodal towers are ported.
> Phase 5 adds `--pack-dir` (the int8 pack, `clef-pack.md`): same flags, packed weights.
> Phase 6 makes `--device npu` work with a pack (see "NPU placement" below).

`Cloudflare/clef-omni` is the Qwen3-Omni-30B-A3B thinker (48 MoE layers, 128 experts,
top-8) with a `JointSchemaHead`. A request is a `state` plus typed `questions`; one
forward over the whole record answers every question jointly (no generation,
`usage.output_tokens` is always 0). It is the same Jev/SystemOne `/v1/systemone` contract
as the d3 family, but a separate implementation (`src/clef/`): the answer format differs.

Checkpoint (66 GiB, 15 BF16 shards + `joint_head.safetensors`): `/home/royalcat/models/clef-omni`.
Detection: `config.json` `model_type == "qwen3_omni_moe"` **and** `joint_head.safetensors`
present (the plain Qwen3-Omni base does not match); `--family clef` forces it.

## Run

```sh
B=$CARGO_TARGET_DIR/release/rocket-inference
cargo build --release -p rocket-inference --no-default-features

# batch: JSON arrays of cases like d3 (`name`, else `id`, is only the label; `model`
# defaults to the model dir name)
$B decide --model-dir /home/royalcat/models/clef-omni \
    --requests examples/rocket-inference/tools/clef_fixtures.json --out /tmp/clef-out.json
# one case
$B decide --model-dir /home/royalcat/models/clef-omni --request '{"model":"clef-omni",
  "state":"Our checkout started returning errors.","questions":{"outage":{"type":"noul"}}}'

# server
$B serve --model-dir /home/royalcat/models/clef-omni --port 8399 --max-tokens 8192
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
schema larger than the cap (`schema requires N tokens before state; maximum is M`) and any
request carrying `images`/`audio`/`videos`
(`Clef-Omni image/audio/video input is not implemented yet ...`). Compute failures are 500.

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
  needs `/home/royalcat/models/clef-omni` and the oracle dumps in `/tmp/opencode/clef/dump`,
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

## Media plumbing in the thinker (phase 8b, 2026-11-14; no towers yet)

`src/clef/media.rs` is everything between "the towers produced features" and "the decoder
ran"; requests with `images`/`audio`/`videos` still get the 422 until the towers (8c-8f) exist.

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
* **Not done**: media towers (8c vision, 8d audio, 8e video, 8f decoding/preprocessing), wiring
  `ClefModel::answer*` and `request.rs` to media (still refused), NPU/board runs of media records.

## Not done

Image/audio/video inputs (vision and audio towers, the media placeholders), `media_kwargs`, attention scores/AV on the NPU, a CPU `sdot` expert path.

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
- Not supported yet: image/audio/video inputs (HTTP 422), see the plan's phase 8.
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
