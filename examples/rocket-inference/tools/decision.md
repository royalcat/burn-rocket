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
`noncausal_full_attention` mode. Text and image requests are complete: the vision tower
and the image processor live in `src/decision/vision/` (phase 2a, gated by
`decision_vision_probe` + `tools/vision_parity.py`), and phase 2b added the placeholder
splice, the multimodal RoPE (`src/decision/mrope.rs`) and the request's `images` field.
Videos are phase 3.

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
- `images` is a list of `data:image/{png,jpeg,jpg,webp};base64,...` URLs shared by every
  question: the reference's strict loader (decoded bytes <= 8,000,000, source pixels
  <= 16,000,000, PNG/JPEG/WebP, a payload that decodes). A non-list is `422` naming the
  field; a local path, a remote URL, another subtype or a broken payload is a `422` whose
  detail is the loader's message, `images[i]: <reason>` (**never** a local path or a
  remote fetch — the server does not read files for a client);
- `videos` is phase 3: a non-empty list is `422 {"detail": "videos are not supported yet"}`;
- a question that fails validation refuses the whole request with
  `422 {"detail": {"invalid_questions": {qid: "<message>"}}}` and **nothing runs**;
- a question over the prompt cap is *not* a refusal: it keeps its `max_length_exceeded`
  record in a `200` answer, and `usage.input_tokens` sums the answered questions only.

`/v1/models` advertises the `decisions` capability; `/health` and `/v1/models` never touch
the model. The served name comes from `MODEL_MANIFEST.json` `model_name` unless
`--model-name` overrides it.

The per-question prompt cap defaults to **32768** tokens for this family
(`decide --max-state-tokens`, `serve --max-tokens`): the shipped checkpoints declare
`max_length: null` (the model's own 262144), an image spends one token per merged image
token (868 for a receipt page), and the RoPE table the cap sizes is only ~16 MiB. It stays
a flag: a memory-tight deployment passes 8192. The reference
server's `/health`/`/v1/models` still report its own `max_input_tokens` (null), which the
Rust server does not model; the cap it enforces is the flag.

### Deferred

Left for phase 5 (documented here so the gap is explicit): **D8** (stricter
`decision_config.json` cross-checks such as `codes`/`token_ids` against
`MODEL_MANIFEST.json` and the manifest hash verification the reference optionally does)
and **D11** (README/recipes/docstring polish for the new family). Text and image
requests are complete; **videos** are phase 3 (a non-empty `videos` list is a 422 and a
`<|video_pad|>` inside an image prompt is an `invalid_question`).

Known differences from the reference, both harmless but named:

- **Media work runs inside the model lock.** The reference decodes the request's images
  before taking its lock (`Decider.arguments` -> `load_images`, then `prepare`/`run` under
  the lock); this server decodes, preprocesses and tower-encodes inside the single
  compute closure, i.e. one image request blocks the next request's decode as well. Two
  concurrent image requests are therefore serialized end to end rather than at decode
  only. Deferred: decoding is ~10-50 ms against seconds of forward, so the lock is not
  what bounds throughput today; it would matter if a large upload arrived while another
  request was waiting.
- **`images` entries are validated in two steps.** The list shape is the envelope's
  (`images takes a list of base64 data URLs`), the entries are the loader's
  (`images[i]: <reason>`), exactly like `d3_server.py`; the difference is only that a
  non-string or empty entry cannot be checked before the transport is known, so its
  message is the loader's rather than the envelope's (same 422).

The checkpoint must be a local safetensors package: `decide` rejects `-hf` (a GGUF Hub
repo cannot carry `readout.safetensors`), and a directory with several sharded
`model-*.safetensors` files is refused (merge them into `model.safetensors` first).

## Fixtures

`decision_fixtures.json`: 47 text-only cases / 67 questions (25 choice-led, 10 noul,
12 score, several multi-question), 7 to 1936 words per state, six JSON (object) states,
domains: support-ticket routing, receipt totals, sentiment, policy compliance,
scheduling, structured account state, intent, language and audit findings.  Every
string is newline-free; receipt totals are computed from their line items.

`decision_fixtures_placeholders.json`: the two text cases whose state quotes a literal
`<|image_pad|>` / `<|video_pad|>` token (the reference's text path has no placeholder
check, so both must answer; see the phase-2b review-fixes section).  Reference:
`decision_reference.py --backbone-dtype f32` on the same file.

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
0.99992 on a 640x480 CMYK photo).  A CMYK case therefore cannot pass the probe's 1e-6
pixel gate: one source LSB becomes up to two LSB after the antialiased resize (measured
pixel `max|Δ|` 1.569e-2 at pooled cosine 0.999976862 for the review's CMYK case, and
7.8e-3 / 0.99992 for a 640x480 photo), so judge a CMYK fixture by the pooled cosine and
the RGB-domain decode, not by the pixel gate.  The same containment covers a malformed
JPEG header (an unsupported SOF precision, a bogus segment length): libjpeg fires
`error_exit` inside `jpeg_read_header`, which is now inside the guard, so the loader
reports a clean error instead of aborting the process.  Eight-bit and 8-bit-JPEG decodes
are unchanged (`to_rgb8` and mozjpeg as before).

Text behaviour is unchanged: the phase-1 parity still reports 67/67 top-1 and max
|Δp| 2.36e-06 over the 47 text cases after phase 2b (the text path passes `None` for the
multimodal rope tables and keeps the shared position ramp).

## Images end to end (phase 2b)

Every question of an image request renders `chatml_images(user, n_images)`: one
`<|vision_start|><|image_pad|><|vision_end|>` group per image immediately after
`<|im_start|>user\n`, then the normal `State:` body (byte-exact against the reference
dump's `expanded_text`). The prompt is tokenized with `add_special_tokens=false`; each
`<|image_pad|>` id is then expanded to its image's merged token count
(`H_patches * W_patches / merge^2`), `mm_token_type_ids` marks those rows `1`, and
`usage.input_tokens` counts the expanded ids.

The forward embeds the expanded ids, replaces each image's expanded rows with the tower's
`[tokens, out_hidden]` features, and rotates the full-attention layers with the multimodal
positions (`src/decision/mrope.rs`, a port of `get_rope_index` +
`recomposition_frequencies`): text runs advance `current_pos` on all three rows, an image
`[1, H, W]` contributes `T = current_pos`, `H = current_pos + 0..H/merge` row-major,
`W = current_pos + 0..W/merge`, then `current_pos += max(H, W) / merge`. The d3
`mrope_section = [11, 11, 10]` puts the height row on the frequency indices
`1, 4, .. 31` and the width row on `2, 5, .. 29`; `cos`/`sin` are `cat([freqs, freqs])`,
64 wide, and rotate the first 64 of the 256 head dims. A text-only request never builds a
table (it takes the unchanged `forward_last_normed` path).

```sh
# one case per process, like the text gate; run from examples/rocket-inference so the
# fixtures' relative `data/cat.jpeg` resolves
python3 tools/decision_parity.py --bin $CARGO_TARGET_DIR/release/rocket-inference \
    --model-dir /tmp/opencode/d3/models/d3-lite \
    --requests tools/decision_fixtures_images.json \
    --ref /tmp/opencode/d3/ref_images_lite_f32.json \
    --out /tmp/opencode/d3/parity_images.json
```

Measured (dev host, 32 cores, f32 d3-lite, 2026-11-01): **8/8 questions top-1 (100%)**,
max |dp| **7.70e-06** (`image-receipt-total/total`; noul 2.26e-07), 121 s as 7 processes;
the expanded token counts are exact on every case (1029 / 1657 / 740 / 1706 / 2550 / 982 /
1029 + 987 = 10,680 planned tokens, the reference's own total). One process for all seven
cases: 109 s. The noul case (`image-receipt-with-text`) lands at 0.820605 vs the
reference's 0.820605.

The tower matches the vision dump through the same code the app runs: phase 2a's
`decision_vision_probe` + `tools/vision_parity.py` gate (8/8 pixel `max|Δ|` 0, pooled
cosine >= 0.999999999) is unchanged, because phase 2b only feeds its `pooled.f32` into the
prompt.

Server smoke (d3-lite, port 8399, `serve --model-dir ...`): a `data:image/jpeg;base64,...`
cat request answers `cat` at 0.996876 in 11 s (`questions=1 tokens=740` in the request
log line); a local path, a remote URL, `data:image/gif`, invalid base64, an empty entry
and a non-string entry are each a `422 {"detail": "images[0]: <reason>"}`; an unknown
question type stays `422 {"detail": {"invalid_questions": ...}}`; a text-only request is
unchanged.

### Review fixes (phase 2b control)

**Rejected images are 422, not 500.** `decision::model::PrepareError` splits
`InvalidImage(String)` (the Qwen2-VL processor's own refusals — aspect ratio, budgets, an
image that produced no merged tokens, a checkpoint without a tower) from `Internal(anyhow::Error)`
(tokenizer/tower/device). The server maps the first to the reference's
`422 {"detail": "<message>"}` and the second to 500. A 4000x10 PNG (aspect 400) now
answers `image rejected by the processor: absolute aspect ratio must be smaller than 200,
got 400.0` (Python's float repr, so the `.0` stays), where it used to be a 500.

**JPEG integrity before any decoder.** `gemma::media::check_jpeg` walks the container: an
SOI, marker segments with in-range lengths, an 8-bit SOF and an EOI after the first scan
(stuffed `FF 00` and `FF D0-D7` restart markers are skipped). `load_image` (paths) and
`decode_data_url` (uploads) both run it on JPEG bytes, so a cut-off upload is
`the image could not be decoded (truncated JPEG (the scan has no EOI marker))` instead of
being accepted by the `image` crate, and a 12-bit frame or a nonsense segment length is a
clean error instead of libjpeg's `error_exit` (the phase-2a containment tests now pin the
structural message). libjpeg's CMYK/YCCK fallback to the `image` crate stays, because
that one is a decoder limitation, not a broken file.

**The declared subtype does not select the decoder.** `decode_image` sniffs the payload
(`FF D8 FF` JPEG, the PNG signature, `RIFF....WEBP`) and dispatches, like
`PIL.Image.open`: a valid JPEG sent as `data:image/png` decodes as a JPEG (200, same
pixels), a PNG sent as `data:image/jpeg` likewise. The subtype is only the reference's
whitelist. Nothing is written to disk any more (the old temp file made the payload's name
decide the decoder and could leak its path into the message).

**A literal placeholder is only refused in image prompts.** The reference's
`_prepare_text` has no placeholder check, so a text state quoting `<|image_pad|>` or
`<|video_pad|>` is answered like any other text (`tools/decision_fixtures_placeholders.json`
is the gate: `text-literal-image-token` 109 tokens, noul 0.9643; `text-literal-video-token`
117 tokens, `marker` at 0.9946; both exact to <=1.2e-07 against
`ref_placeholders_lite_f32.json`). `placeholder_problem(prompt, n_images)` and
`expand_placeholders` gate the count and video checks on `n_images > 0`.

**A text-only package loads.** `config.json` `image_token_id`/`video_token_id` and the
tokenizer's `<|vision_start|>` are required only when the package declares a
`vision_config`; without one the ids are `u32::MAX`, the tower is absent and an image
request is `422 {"detail": "image inputs are not available: this checkpoint has no vision
tower"}` (verified with d3-lite's `config.json` minus `vision_config`/`image_token_id`:
text answers unchanged at 107 tokens, 0.9293 for the routing case).

The tower is 100.5M f32 parameters (~0.40 GB for d3-lite) and loads eagerly with the
backbone: a d3-lite process is 2926 MiB anon with it, the same package without the tower
2488 MiB.

Re-verified after the fixes (dev host, f32 d3-lite): image parity **8/8**, max |dp|
**7.70e-06**; text parity **67/67**, max |dp| **2.36e-06**, one-process batch **76,829
tokens**; the literal-placeholder pair **2/2**, max |dp| 1.16e-07
(`ref_placeholders_lite_f32.json`); a 2-image x 2-question case (`which` choice +
one noul, ref `ref_2x2_lite_f32.json`) **5086/5086 tokens**, both answers, max |dp|
7.23e-08.
