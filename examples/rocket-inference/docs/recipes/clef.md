# Clef-Omni (Cloudflare) — typed decisions, text-only, CPU

> **STATUS 2026-11-14 (phase 4): text-only, CPU-only, dev host verified. No board run.**
> Images, audio and video are refused (422) until the multimodal towers are ported;
> `--device npu` is refused until NPU expert streaming lands (phase 6).

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
- Placement: CPU only. `--device auto` resolves to CPU even on an NPU build; `--device npu`
  / `--text-device npu` fail with "NPU expert streaming arrives in a later phase". Flags of
  other families (`--dtype`, `--quant`, `--defer-towers`, `--text-attn-device`, ...) are
  rejected. The `placement:` line reads `text=CPU(f32, layer-streamed) head=CPU(f32)`.
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

## Not done

Image/audio/video inputs (vision and audio towers, the media placeholders), NPU expert
streaming (phase 6), quantized CPU experts, batching several records, `media_kwargs`.
