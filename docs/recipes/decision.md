# Decision 3.0 (d3) recipes

[vllm-sr Decision 3.0](https://huggingface.co/vllm-sr/d3-lite) answers typed questions
about a piece of text, an image or a video. A request is a `state` plus questions of type
`choice`, `noul` or `score`; the model reads the whole thing once per question and returns
probabilities. Nothing is generated, so it is fast and cheap to run. `d3-lite` is the
tested size; `d3-nano` works too but is only checked on a dev host.

## Serve (recommended board command)

```sh
rocket-inference serve --model-dir /root/models/d3-lite \
    --max-tokens 8192 --port 8384
```

- `--device auto` (the default) puts the text layers on the NPU.
- The vision tower loads only when the first image or video arrives, so a text-only service
  stays small (about 350 MiB of RAM plus 966 MiB of NPU buffers).
- `--max-tokens` caps the prompt length **per question**.

Ask it something:

```sh
curl -s localhost:8384/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Customer asks for a refund of a duplicate card charge.",
  "questions": {"route": {"type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {"returns": "Refunds and replacements",
                 "billing": "Payments and invoices",
                 "technical": "Firmware and connectivity faults"}}}
}'
```

Images and videos go in as `data:` URLs. `/health` and `/v1/models` are always available.

## One-shot (no server)

```sh
rocket-inference decide --model-dir /root/models/d3-lite --device npu \
    --requests cases.json --out answers.json
```

`cases.json` is a list of `{"name", "state", "questions"}` objects; media are local file
paths here. Use `--request '<json>'` for a single case.

## What the preset does

| Stage | Where | Why |
|---|---|---|
| text projections | NPU, fp16 | 5 questions over 6,546 tokens: 113 s instead of 197 s on the CPU, with about a third of the CPU time |
| attention, vision tower, readout | CPU | no NPU kernel for them in this model |
| token table | block-wise int8 | 347 MiB of RAM for a text-only service, with answers within 1.7e-3 of the f32 reference |
| text body | stored bf16, widened per matmul | half the resident bytes, output bit-identical |
| vision tower | loaded on the first image or video | +196 MiB, only for services that get media |

## What it trades away

- The first image or video request pays the tower load.
- The exact (f16) token table costs +211 MiB (`--embed-f16`). The int8 default stays within the
  1e-2 probability bar, with identical top choices on every checked case.
- Video answers depend on the `ffmpeg` version that decodes the frames (differences of about 1.5e-2
  between hosts); judge them by the top choice.

## Tune it

| You want | Flags | Cost |
|---|---|---|
| Exact token table | `--embed-f16` | +211 MiB (lite), +424 MiB (nano) |
| Tower loaded at startup | `--no-defer-towers` | +196 MiB RAM, no first-media delay |
| Smaller NPU buffers | `--npu-dtype int8 --int8-group 512` | 474 MiB instead of 966 MiB, but +6.5 % wall, +18 % CPU and text error 3.4e-2 (over the bar); not recommended |
| CPU only | `--device cpu` | f32 compute, about 1.3 GiB, 265 s instead of 147 s for the 5-question case |
| Longer prompts | `--max-tokens 32768` | the default on the CLI; the board recipe uses 8192 to bound the RoPE table |

## Limits

- Images: 8 MB decoded and 16 megapixels. Videos: 32 MB, 300 s, 16,384 tokens in total.
- A question longer than `--max-tokens` is reported as an error in its own answer; a
  malformed question rejects the whole request with a 422.
- The body limit on `/v1/systemone` is 96 MiB.

## More

Memory rounds, board timings, the int8 comparison and the parity tables are in
[`../experiments/decision-d3.md`](../experiments/decision-d3.md). The request contract is
in [`../../tools/decision.md`](../../tools/decision.md).
