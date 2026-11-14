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
  `auto` is the NPU on an aarch64 `npu` build **when the model dir is a clef pack** (a BF16
  checkpoint falls back to the CPU), else the CPU. An explicit `npu` on a CPU build is the
  shared "requires an aarch64 build with the `npu` feature" error, and on an NPU build with
  a BF16 checkpoint "needs a clef-pack". Flags of other families (`--dtype`, `--quant`,
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
| `ROCKET_CLEF_EMU=1` | Test/diagnostic: run the *host emulation* of the NPU path (same bucketing, schedule, quantization and int32-exact group partials; any host, needs a pack). Overrides `--device`. |
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

## Not done

Image/audio/video inputs (vision and audio towers, the media placeholders), batching several
records, `media_kwargs`, attention scores/AV on the NPU, a CPU `sdot` expert path.
