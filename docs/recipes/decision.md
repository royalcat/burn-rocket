# Decision 3.0 (d3) recipes

[vllm-sr Decision 3.0](https://huggingface.co/vllm-sr/d3-lite) answers typed questions
about a piece of text, an image or a video. A request is a `state` plus questions of type
`choice`, `noul` or `score`; the model reads the whole thing once per question and returns
probabilities. Nothing is generated, so it is fast and cheap to run. `d3-lite` is the
tested size; `d3-nano` works too but is only checked on a dev host.

## Serve (recommended board command)

```sh
rocket-inference serve --model-dir /root/models/d3-lite \
    --device npu --defer-towers --max-tokens 8192 --port 8384
```

- `--device npu` puts the text layers on the NPU.
- `--defer-towers` loads the vision tower only when the first image or video arrives, so a
  text-only service stays small (about 350 MiB of RAM plus 966 MiB of NPU buffers).
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

## Profiles

d3 has one NPU stage, so balanced, full offload and speed are all the same command above.
For a CPU-only run use `--device cpu`. `--npu-dtype int8` is available but not
recommended: it is slower and misses the accuracy target.

## Limits

- Images: 8 MB decoded and 16 megapixels. Videos: 32 MB, 300 s, 16,384 tokens in total.
- A question longer than `--max-tokens` is reported as an error in its own answer; a
  malformed question rejects the whole request with a 422.
- The body limit on `/v1/systemone` is 96 MiB.

## More

Memory rounds, board timings, the int8 comparison and the parity tables are in
[`../experiments/decision-d3.md`](../experiments/decision-d3.md). The request contract is
in [`../../tools/decision.md`](../../tools/decision.md).
