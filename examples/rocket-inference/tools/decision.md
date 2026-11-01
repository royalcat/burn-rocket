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
41 s long case) **at a fixed thread count**: `--threads N` changes ATen's GEMM
partition and moves probabilities by up to 1e-6, so regenerate a reference with the
same thread setting as the run you compare against (the numbers below use torch's
default count).

## Parity

The CLI contract is `<bin> decide --model-dir DIR (--requests CASES.json | --request OBJECT)
[--out OUT.json] [--max-state-tokens N]` (`rocket-inference help decide`). `--requests`
writes `{"schema": "d3-decision-v1", "model", "cases": [{"name", "response"}]}` where each
`response` is the official `{"model", "answers", "usage"}` object; `--request` writes the
bare response. The parity script reads the answers either from that official shape or from
`cases[].questions` (`options`/`codes`/`probabilities` in prompt order).

Implementation: `examples/rocket-inference/src/decision/` (the `d3_format.py` contract, the
readout head, the prompt/answer path) on the intent family's Qwen3.5 backbone
(`src/qwen35_intent/`), with `IntentModel::full_attn_causal = false` for the checkpoint's
`noncausal_full_attention` mode. The text path is complete; the vision tower and the
image processor landed in phase 2a (`src/decision/vision/`, gated by
`decision_vision_probe` + `tools/vision_parity.py` below), but the multimodal RoPE
splice and the request's `images` field are phase 2b, so requests still refuse images
with a 400.

```sh
/tmp/opencode/d3/venv/bin/python tools/decision_parity.py \
    --bin target/release/rocket-inference \
    --model-dir /tmp/opencode/d3/models/d3-lite \
    --requests tools/decision_fixtures.json \
    --ref /tmp/opencode/d3/ref_text_lite_f32.json \
    --out /tmp/opencode/d3/parity_report.json
```

It runs the CLI once per case, prints the top-1 match rate, the largest probability
delta and per-type stats, and exits 1 when top-1 < `--min-top1` (default 100%) or
max |dp| > `--max-dp` (default 2e-3). `--limit N` runs a prefix; `--arg FLAG` forwards an
extra CLI flag.

Measured (dev host, 32 cores, f32, 2026-11-01): d3-lite **67/67 top-1**, max |dp|
**2.36e-06** (choice 2.36e-06, noul 1.91e-06, score 1.32e-06) over 47 cases / 76,829
prompt tokens; 511 s as 47 processes. The same 47 cases in one `--requests` process take
428 s (model load included).

### CLI and server error semantics

`decide` is the runtime's own path: a question that cannot be answered keeps its
`{"type", "error", "message"}` record in `answers` and the other questions still run
(`invalid_question`, `max_length_exceeded`).

`POST /v1/systemone` is the reference server's contract (`d3_server.py`):

- the body must be a JSON object carrying only `state`, `questions`, `images`, `videos`,
  `model`; anything else (or a non-object, or a body that is not JSON) is
  `422 {"detail": "<reason>"}`;
- `questions` must be a nonempty object of nonempty ids (422 string otherwise);
- `images`/`videos` are phase 2: a non-empty list is `422 {"detail": "images/videos are
  not supported yet"}`, a non-list is `422` naming the field (data URLs only, never local
  paths or URLs);
- a question that fails validation refuses the whole request with
  `422 {"detail": {"invalid_questions": {qid: "<message>"}}}` and **nothing runs**;
- a question over the prompt cap is *not* a refusal: it keeps its `max_length_exceeded`
  record in a `200` answer, and `usage.input_tokens` sums the answered questions only.

`/v1/models` advertises the `decisions` capability; `/health` and `/v1/models` never touch
the model. The served name comes from `MODEL_MANIFEST.json` `model_name` unless
`--model-name` overrides it.

### Deferred

Left for phase 5 (documented here so the gap is explicit): **D8** (stricter
`decision_config.json` cross-checks such as `codes`/`token_ids` against
`MODEL_MANIFEST.json` and the manifest hash verification the reference optionally does)
and **D11** (README/recipes/docstring polish for the new family). The text path is
otherwise complete; images/videos are phase 2.

The checkpoint must be a local safetensors package: `decide` rejects `-hf` (a GGUF Hub
repo cannot carry `readout.safetensors`), and a directory with several sharded
`model-*.safetensors` files is refused (merge them into `model.safetensors` first).

## Fixtures

`decision_fixtures.json`: 47 text-only cases / 67 questions (25 choice-led, 10 noul,
12 score, several multi-question), 7 to 1936 words per state, six JSON (object) states,
domains: support-ticket routing, receipt totals, sentiment, policy compliance,
scheduling, structured account state, intent, language and audit findings.  Every
string is newline-free; receipt totals are computed from their line items.

## Images (phase 2a)

`decision_fixtures_images.json`: 7 image cases / 8 questions, each with an `images` list
of local paths the request shares across its questions.  Assets: the checkpoint's own
`assets/{example-receipt,index-pareto,banner,serving-throughput}.png` (absolute paths
under the model directory) plus `data/cat.jpeg`, relative to this directory — run the
reference from `examples/rocket-inference`.  The cases cover a receipt total, a chart
comparison, a photo, a banner, two images in one request, an image with a ticket state,
and one image with two questions.  Rendered prompts put every image (as
`<|vision_start|><|image_pad|><|vision_end|>`) immediately after `<|im_start|>user\n`
with no separator, then the `State:` block.

The vision dump needs `--batch-size 1` (the default), and the script refuses
`--dump-vision` with anything else: the image path feeds the processor one batch per
question chunk with the case's images **repeated once per question**, so a dump taken with
`N > 1` records a batch-sized `pixel_values`/`grid`/`pooled` that no single-image probe
run can be compared against.

```sh
/tmp/opencode/d3/venv/bin/python tools/decision_reference.py \
    --model /tmp/opencode/d3/models/d3-lite \
    --requests tools/decision_fixtures_images.json \
    --out /tmp/opencode/d3/ref_images_lite_f32.json \
    --dump-vision /tmp/opencode/d3/dump-images-lite-f32 \
    --backbone-dtype f32
```

`--dump-vision` writes per question the processor's `pixel_values`/`image_grid_thw` and
the expanded prompt, the processor's `input_ids`/`mm_token_type_ids`/`attention_mask`
rows, the tower's pooled output (`backbone.get_image_features(...).pooler_output`, the
concatenation over images, f32), the multimodal RoPE positions
(`compute_3d_position_ids`, `[3, 1, L]`), the final-normed hidden and the raw code
logits.  The processor tensors are the recorded `D3.image_logits` batch (the run's own
call, unchanged); `pooled`/`mrope` are the backbone's own methods on those tensors.

Measured (dev host, 32 cores, f32, 2026-11-01, torch default thread count): d3-lite
**53.7 s** for 7 cases / 8 questions / 10,680 planned tokens (per case 2.9-14.0 s), every
answer intended (`100_to_150`, `decision_3`, `cat`, `lite`, `first`, `matches=true`,
`total=100_to_150`, `payment=visa`).  d3-nano f32 (3 cases, 42.1 s) answers the same
receipt as `over_150` -- the reference convention, not a parity problem.

The Rust side is `src/decision/vision/` (tower + processor) and the standalone probe:

```sh
decision_vision_probe --model-dir /tmp/opencode/d3/models/d3-lite \
    --image <path> [--image <path> ...] --out /tmp/opencode/d3/probe/receipt
/tmp/opencode/d3/venv/bin/python tools/vision_parity.py \
    --ref /tmp/opencode/d3/dump-images-lite-f32 --probe /tmp/opencode/d3/probe/receipt \
    --stem image-receipt-total__total
```

The probe loads only `config.json` and the `visual.*` tensors (no text backbone), writes
`pixel_values.f32` (`[patches, 1536]`), `pooled.f32` (`[tokens, out_hidden]`) and a
`meta.json` (per image source/resized size, grid, patches, tokens, row offsets), and the
parity script gates pixel `max|Δ| <= 1e-6`, pooled cosine `>= 0.999999` and an exact
grid/token match.

Measured (dev host, 32 cores, f32 d3-lite, 2026-11-01, 8 dumped entries, after the review
fixes): pixel `max|Δ|` **0** on every image (the resize/normalize/patchify are
bit-identical to torchvision's processor), pooled cosine **0.999999999-1.000000000**,
`max|Δ|` 9.1e-5 to 9.0e-4 at a pooled peak of 3.7-5.5 (2.4e-5 to 1.7e-4 relative),
grid/token exact; tower wall 3.4-13.1 s per probe run (2520-9744 patches, load 0.2 s;
d3-nano 8.2-21.4 s).  The residual is the attention kernel, not the tower: swapping
torch's own SDPA backend moves the reference pooled output by 4.4e-5 to 1.9e-4 absolute,
which is why `vision_parity.py --pooled-max-abs` is an advisory opt-in rather than the
gate.  d3-nano (depth 24, hidden 1024, out 2048) gives the same 0 pixel delta and cosine
0.999999999-1.000000000 on receipt/pareto/cat, which is what pins the dimension-generic
loader.

**Review fix (phase 2a control): the merger's GELU.** `Qwen3_5VisionPatchMerger` builds
`nn.GELU()` — the **exact erf** form — while the blocks use the checkpoint's
`hidden_act` (`gelu_pytorch_tanh`).  The first cut used the tanh form in both places; with
`pixel` still bit-exact it produced pooled cosine 0.999999961 at `max|Δ|` 1.34e-3 (the
whole residual), against 5.95e-05 for a torch reference with the erf merger.  The merger
now calls `burn::tensor::activation::gelu` and a unit test pins both activations
(`the_merger_uses_the_exact_erf_gelu`).

**Review fix: image decoding.** `gemma::media::load_image` (shared with EmbeddingGemma)
now converts 16-bit PNGs with PIL's own rules instead of `to_rgb8`'s rounding — a 16-bit
gray PNG (`I;16`) is **clipped to [0, 255]** by `convert("RGB")` (so it reads as
near-white), while 16-bit colour and Pillow's 16-bit gray+alpha keep the **high byte**
(`v >> 8`, alpha dropped) — and a CMYK/YCCK Adobe JPEG no longer aborts the process:
libjpeg cannot convert it and mozjpeg raises that as a `resume_unwind`, so the loader
refuses it up front and falls back to the `image` crate (1 LSB from PIL, pooled cosine
0.99992 on a 640x480 CMYK photo).  Eight-bit and 8-bit-JPEG decodes are unchanged
(`to_rgb8` and mozjpeg as before).

Text behaviour is unchanged: the phase-1 parity still reports 67/67 top-1 and max
|Δp| 2.36e-06 over the 47 text cases, and `system_one`/`decide`/`/v1/systemone` still
refuse images (the multimodal RoPE splice and the request field are phase 2b).
