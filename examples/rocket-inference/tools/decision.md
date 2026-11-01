# d3 reference harness

Phase 0 of the `decision` family (vllm-sr Decision 3.0, `d3-code-readout-v1`): the text
fixtures, the official-runtime reference, and the parity driver the Rust implementation
is validated against.  Nothing here is part of the Rust build.

## What d3 is

State + questions -> one forward pass -> readout from the **final-normed hidden state at
the last prompt token** through `readout.safetensors` (`weight` `[255, hidden]`, fp32).
`logits = hidden.float() @ readout.T`, invalid codes masked, `/ temperature`, softmax.
The prompt is ChatML (`enable_thinking=False`) with the system message "You are a
decision engine. ..." and a user turn `State: ... / Question: ... / Options: A: ... /
Reply with only the code of the best option.`  Full-attention layers are bidirectional
(`attention_mode: noncausal_full_attention`); the Gated DeltaNet layers stay causal.
Codes are `A..Z, AA..JT` (255, the ones that are single tokens).

## Environment

```sh
uv venv --python 3.12 /tmp/opencode/d3/venv
uv pip install --python /tmp/opencode/d3/venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python /tmp/opencode/d3/venv/bin/python \
    "transformers==5.17.0" safetensors pillow numpy huggingface_hub accelerate
```

`accelerate` is required by the runtime's `device_map` call; `flash-linear-attention`
and `causal-conv1d` are **not** installed: they are GPU-only and the runtime refuses
them on CPU (the transformers reference implementation is used instead).

```sh
/tmp/opencode/d3/venv/bin/hf download vllm-sr/d3-lite --local-dir /tmp/opencode/d3/models/d3-lite
/tmp/opencode/d3/venv/bin/hf download vllm-sr/d3-nano --local-dir /tmp/opencode/d3/models/d3-nano
```

## Reference

```sh
/tmp/opencode/d3/venv/bin/python tools/decision_reference.py \
    --model /tmp/opencode/d3/models/d3-lite \
    --requests tools/decision_fixtures.json \
    --out /tmp/opencode/d3/ref_text_lite.json \
    --dump /tmp/opencode/d3/dump-lite --limit 3 --check-dump
```

`--dump` writes per question the exact prompt, the input ids, the final-normed last
hidden (float32) and the raw 255 logits.  `--backbone-dtype f32` keeps the official
runtime but casts the loaded backbone to float32 (the checkpoint stores bf16, so this is
the bf16 weights with f32 activations -- the closest reference for an f32 Rust body, and
about 2x faster on CPU).  The default `bf16` is the byte-faithful official path: the
runtime itself fixes bf16 and its `D3Model` rejects a `dtype` argument.

Note the readout multiplies the hidden the model produces, so a reference must be
produced with the same body dtype the Rust side runs: compare f32 Rust against
`--backbone-dtype f32`, not against bf16.  Over the whole fixture set the two disagree
by a median 1.4e-3 and up to 1.6e-2 in probability (one of 67 top-1 answers flips), so a
mismatched reference would fail the 2e-3 gate.  `--backbone-dtype f32` is also ~3x
faster on CPU.

Measured on the 32-core host (no GPU, torch 2.15 CPU, `batch_size 1`): d3-lite bf16
503 s for 47 cases / 67 questions / 76,829 prompt tokens (8.4 min, 1.2-56 s per case);
d3-lite f32 173 s; d3-nano bf16 101 s for four cases (a 6,307-token case alone takes
92 s).  Repeated runs are bit-identical (4 cases run twice, choice/noul/score and a
41 s long case).

## Parity

The future CLI contract is `<bin> decide --model-dir DIR --requests ONE_CASE.json --out OUT.json`
writing `{"schema": "d3-decision-v1", "cases": [{"name", "questions": {qid:
{"type", "options", "codes", "probabilities"}}}]}` (`options`/`codes` optional, filled
from the reference; probabilities in option order or as a label map).

```sh
/tmp/opencode/d3/venv/bin/python tools/decision_parity.py \
    --bin target/release/rocket-inference \
    --model-dir /tmp/opencode/d3/models/d3-lite \
    --requests tools/decision_fixtures.json \
    --ref /tmp/opencode/d3/ref_text_lite.json \
    --out /tmp/opencode/d3/parity_report.json
```

It runs the CLI once per case, prints the top-1 match rate, the largest probability
delta and per-type stats, and exits 1 when top-1 < `--min-top1` (default 100%) or
max |dp| > `--max-dp` (default 2e-3).

## Fixtures

`decision_fixtures.json`: 47 text-only cases / 67 questions (25 choice-led, 10 noul,
12 score, several multi-question), 7 to 1936 words per state, six JSON (object) states,
domains: support-ticket routing, receipt totals, sentiment, policy compliance,
scheduling, structured account state, intent, language and audit findings.  Every
string is newline-free; receipt totals are computed from their line items.
