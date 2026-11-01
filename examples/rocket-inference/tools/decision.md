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
`noncausal_full_attention` mode. Text, image and video requests are complete: the vision
tower, the image processor and the video processor/decoder live in
`src/decision/vision/` (phase 2a, gated by `decision_vision_probe` +
`tools/vision_parity.py`; phase 3 added `video.rs` and `decision_video_probe`), phase 2b
added the placeholder splice, the multimodal RoPE (`src/decision/mrope.rs`) and the
request's `images` field, and phase 3 added the `videos` field (per-pair prompt groups,
per-frame tower attention, the video sampling and the transport limits).

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
- `videos` is a list of `data:video/{mp4,webm,quicktime,x-matroska};base64,...` URLs shared
  by every question (a non-list is `422` naming the field, as for `images`): the
  reference's strict loader (decoded bytes <= 32,000,000, source frames <= 8,294,400
  pixels, a clip <= 300 seconds, a payload `ffmpeg` opens, 2 sampled frames per second
  with 4-32 frames), and every refusal is `videos[i]: <reason>`. All videos of one
  request together may not exceed 16,384 input tokens, else
  `422 {"detail": "the videos take N input tokens, over the 16,384 tokens a request may
  spend on videos; send fewer or shorter videos"}`. `/v1/systemone` takes bodies up to
  **96 MiB** (the per-item ceiling is one 32 MB video plus three 8 MB images, ≈75 MB
  once base64-encoded); every other route keeps the server's usual **32 MiB** media
  limit, so an embedding upload of 33 MiB or more is still a `413`;
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
and **D11** (README/recipes/docstring polish for the new family). Text, image and video
requests are complete.

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

`decision_fixtures_videos.json`: 5 video cases / 9 questions (see "Videos" below);
`data/test.mp4` (4 s, 8 fps, 32 frames, a colour-bar test card with a counter) and the
checkpoint's own `assets/example-video.mp4` (4 s, 8 fps, a square crossing the frame).

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
request is `422 {"detail": "image inputs are not available: the checkpoint has no vision
tower (visual.* weights)"}` (verified with d3-lite's `config.json` minus
`vision_config`/`image_token_id`:
text answers unchanged at 107 tokens, 0.9293 for the routing case).

The tower is 100.5M f32 parameters (~0.40 GB for d3-lite) and loads eagerly with the
backbone: a d3-lite process is 2926 MiB anon with it, the same package without the tower
2488 MiB.

`--defer-towers` is accepted for the decision family: the tower stays out of the process
until the first media request (`decision: vision tower deferred to the first media
request`, then one `decision: loaded vision tower ...` line), and a text-only service
never allocates it. The deferred load is retried on every media request, so a package
that declares `vision_config` but lacks the `visual.*` tensors starts healthy and fails
each media request with a 500, where an eager load fails at startup.

Re-verified after the fixes (dev host, f32 d3-lite): image parity **8/8**, max |dp|
**7.70e-06**; text parity **67/67**, max |dp| **2.36e-06**, one-process batch **76,829
tokens**; the literal-placeholder pair **2/2**, max |dp| 1.16e-07
(`ref_placeholders_lite_f32.json`); a 2-image x 2-question case (`which` choice +
one noul, ref `ref_2x2_lite_f32.json`) **5086/5086 tokens**, both answers, max |dp|
7.23e-08.

## Videos (phase 3)

A video request reads every clip at **2 frames per second**, at least 4 and at most 32
frames spread evenly over the whole clip, groups the frames in **temporal pairs**
(`temporal_patch_size = 2`) and gives every pair its own prompt group with a timestamp.

### Sampling, decoding and the prompt

`src/decision/vision/video.rs` is the sampling half, a port of
`d3_runtime.video_sample_indices` / `_decode_video` / `_video`:

- `count = int(total / fps * 2)`, clamped to `[4, 32]` and `<= total`, then
  `np.linspace(0, total - 1, count).round()` — **half to even** on an exact `.5`
  (Python's `round`, not Rust's half-away-from-zero) with numpy's endpoint correction.
  An **odd** count duplicates the last frame *and its source index*, so every frame is
  part of a temporal pair.
- The source rate is the container's `r_frame_rate`; an unknown or implausible one
  (`< 0.1`, `> 1000`, not finite) is **24**. The total is the container's `nb_frames`;
  a container that does not say has its frames counted (`ffprobe -count_frames`).
- Frames come from `ffmpeg` (`select` on the sampled indices, `rawvideo`/`rgb24`) through
  `gemma::media::extract_frames`. **Measured: `ffmpeg`'s RGB is bit-identical to
  OpenCV's `BGR -> RGB`** on both fixtures (8 frames each, `max|d| 0`), so the decoder
  is not a parity variable; the probe's frame gate is exact.
- The prompt group of one pair is `<{t:.1} seconds><|vision_start|><|video_pad|>*
  frame_seqlen<|vision_end|>` with `t = (idx[2k] + idx[2k+1]) / 2 / fps` and
  `frame_seqlen = H_patches * W_patches / merge^2`; the groups are concatenated with no
  separator. The chat template's own `<|vision_start|>`/`<|vision_end|>` stay **around
  the whole replacement** (the processor replaces only the single `<|video_pad|>` the
  template emitted), so the expanded prompt starts with a `vision_start` and ends with a
  `vision_end` around the groups — the reference's `expanded_text`, token for token.

The processor (`processor::prepare_video`) is `Qwen3VLVideoProcessor` with
`cap_pixels_per_frame=True`, `do_sample_frames=False` and the runtime's budget
(`shortest_edge` 4,096, `longest_edge` 200,704 x 32): the per-frame ceiling is
`max_video_tokens * 32^2 = 200,704` pixels, the budget's even share is
`longest_edge // frames`, a frame never drops below `int(4096 * 1.05) = 4300` pixels, and
`smart_resize` then uses `t_bar * h_bar * w_bar` against `pixels_per_frame * frames`.
Patches are packed `(pair, merge block, in-block, channel, temporal patch, patch row,
patch col)` and the grid is `[T = frames / 2, H/16, W/16]`.

### The tower on a multi-frame grid

`VisionTower::encode_video` runs the same forward as an image on the `[T, H, W]` grid: the
learned position table resamples each frame's `H x W` patches (`within % (H*W)`), the
axial rope is the single frame's table repeated `T` times (the tower has no temporal rope),
and the merger groups 2x2 patches **within a frame**, giving `T * H * W / 4` tokens.

**Attention is per frame.** The phase-3 brief said one sequence per video; the reference
(`get_vision_attention_seqlens` -> `get_vision_cu_seqlens(merge_temporal=False)` in
transformers 5.17) makes **`T` segments of `H*W`** per video, and the measurement is
unambiguous on the dumped `video-example__direction` clip: the per-frame arm reproduces
the reference pooled output **bit for bit** (`max|d| 0`) while the whole-clip arm lands at
cosine **0.680** (`max|d| 2.89`). So `encode_video` loops the frames, one attention call
each — which is also what keeps the image path (T = 1) bit-identical.

`mrope` gained `TYPE_VIDEO = 2`: each type-2 run is one temporal pair, positioned as a
`[1, H, W]` grid (`T = current_pos`, `H/W` row-major, then `current_pos += max(H, W) /
merge`), so the timestamps and vision markers between two pairs consume positions like any
other text. The prompt assembly (`model::prepare`) records a video's feature rows as **one
run per pair** (`MediaSpan { kind: Video, runs: [(start, end), ...] }`) and splices them in
order, leaving the timestamps between them untouched.

### Transport and limits

The CLI takes local paths (`videos: ["clip.mp4"]`), the server data URLs only, like
`d3_server.py`: `data:video/{mp4,webm,quicktime,x-matroska};base64,...` at most
**32,000,000** decoded bytes (the encoded text is length-checked before decoding), frames
of at most **8,294,400** pixels and a clip of at most **300** seconds. Uploads are written
to a temporary file (removed when the decoder returns) so they decode exactly like the same
clip by path. All videos of one request together may not exceed **16,384** input tokens
(`sum(T*H*W/4)`), else the request is refused before anything runs. The per-question cap
message carries both media counts:
`({visual} for {n} image(s) and {m} video(s))` where `visual` is the image tokens plus the
video prompt groups' token counts (the reference's `sum(visual) + expanded`).

Media are validated **images first**, like the reference (`prepare` loads `images` before
`videos`): every image is preprocessed before the video token budget is planned, so a
request with an unusable image *and* an over-budget video reports the image, and a
text-only package answers an image+video request with the image reason
(`image inputs are not available: the checkpoint has no vision tower (visual.*
weights)`).

### Reference, probe and gate

```sh
# reference answers + the video dumps (pixel_values_videos, video_grid_thw, the expanded
# prompt, ids/types/mask, pooled, mrope, hidden, logits, every sampled frame, metadata)
/tmp/opencode/d3/venv/bin/python tools/decision_reference.py \
    --model /tmp/opencode/d3/models/d3-lite \
    --requests tools/decision_fixtures_videos.json \
    --out /tmp/opencode/d3/ref_videos_lite_f32.json \
    --dump-vision /tmp/opencode/d3/dump-videos-lite-f32 --backbone-dtype f32

# the Rust processor + tower for the same clips
cargo build --release -p rocket-inference --no-default-features --bin decision_video_probe
$CARGO_TARGET_DIR/release/decision_video_probe --model-dir /tmp/opencode/d3/models/d3-lite \
    --video /tmp/opencode/d3/models/d3-lite/assets/example-video.mp4 \
    --out /tmp/opencode/d3/probe-video-example
/tmp/opencode/d3/venv/bin/python tools/vision_parity.py \
    --ref /tmp/opencode/d3/dump-videos-lite-f32 \
    --probe /tmp/opencode/d3/probe-video-example --stem video-example__direction

# end to end (one case per process), then the expanded prompts/token counts
python3 tools/decision_parity.py --bin $CARGO_TARGET_DIR/release/rocket-inference \
    --model-dir /tmp/opencode/d3/models/d3-lite \
    --requests tools/decision_fixtures_videos.json \
    --ref /tmp/opencode/d3/ref_videos_lite_f32.json \
    --out /tmp/opencode/d3/parity_videos.json
$CARGO_TARGET_DIR/release/rocket-inference decide --model-dir /tmp/opencode/d3/models/d3-lite \
    --requests tools/decision_fixtures_videos.json --dump-prompts DIR --out DIR/answers.json
```

The video dump also writes `frames.{i}.npy` (uint8 `[frames, height, width, 3]`, the
reference's own OpenCV decode) and `video_metadata.json` (fps, total frames, sampled
indices) per case, so a probe can be scored on three levels: the frames (exact), the
patches (`pixel_values_videos`, `max|Δ| <= 1e-6`) and the tower output
(`pooled_videos`, cosine `>= 0.999999`).

The fixtures name their media relative to `examples/rocket-inference` (`data/test.mp4`)
or absolutely (the checkpoint's `assets/`). Both scripts resolve a relative path from
`examples/rocket-inference` by default (`--media-base` overrides) and print the base, so
the gate is the same whether it runs from the example directory or from the repository
root — measured 9/9 both ways.

Measured (dev host, 32 cores, f32 d3-lite, 2026-11-01; watchdog on the 8-core box while
another job ran, wall +-20 %):

| case | clip(s) | frames | grid | tokens | top-1 | max abs Δp |
|---|---|---|---|---|---|---|
| video-example/direction | example-video | 8 | [4, 20, 36] | 868 (720 video) | right | 2.67e-06 |
| video-example/color_change | example-video | 8 | [4, 20, 36] | 852 (720) | true | 1.23e-06 |
| video-testcard/content | test.mp4 | 8 | [4, 22, 30] | 813 (660) | test_card | 6.52e-08 |
| video-testcard/counter | test.mp4 | 8 | [4, 22, 30] | 798 (660) | true | 2.62e-06 |
| video-longer/rate | example-video | 8 | [4, 20, 36] | 860 (720) | level 2 | 6.8e-07 |
| video-with-image/total | receipt + example-video | 8 | [4, 20, 36] | 1787 (720) | 100_to_150 | 1.09e-06 |
| video-with-image/direction | receipt + example-video | 8 | [4, 20, 36] | 1745 (720) | right | 1.42e-06 |
| video-two-clips/which | both | 8 + 8 | [4, 20, 36], [4, 22, 30] | 1569 (1380) | second | 3.97e-06 |
| video-two-clips/both_tests | both | 8 + 8 | [4, 20, 36], [4, 22, 30] | 1551 (1380) | true | 7.61e-07 |

End to end: **9/9 questions top-1 (100 %)**, max |Δp| **3.97e-06**
(`video-two-clips/which`; choice 3.97e-06, noul 2.62e-06, score 6.82e-07), 94.4 s as 5
processes (87.98 s for all five in one process, model load included), **10,843** planned
tokens. The expanded prompt of every question is **byte-identical** to the reference
dump's `expanded_text` (9/9, 9,243-21,433 characters) and `usage.input_tokens` matches
(1720 / 1611 / 860 / 3532 / 3120).

Tower (`decision_video_probe` + `vision_parity.py`, 3 runs over the two clips and both
together): sampled frames **max|Δ| 0** on every frame (ffmpeg == OpenCV, 5,529,600 and
4,147,200 values), `pixel_values_videos` **max|Δ| 0** (bit-identical smart_resize +
normalize + pair patchify), pooled cosine **0.999999999-1.000000000** with
`max|Δ| 1.7e-04..1.7e-03` at a pooled peak of 2.4-3.1 (the same attention-kernel residual
as the image path, not a tower change), grids `[4, 20, 36]` / `[4, 22, 30]` and token
counts exact, sampled indices/fps/total exact. Probe wall 3.3 s for the 720-token clip
(0.08 s decode, 3.3 s encode, 2880 patches).

Two ephemeral edge cases (not committed as fixtures) match as well: a **2.5 s** clip
samples 5 indices, duplicates the last (`[0, 5, 10, 14, 19, 19]`, 6 frames, grid
`[3, 22, 30]`, 495 video tokens, timestamps 0.3/1.5/2.4, prompt byte-identical, max |dp|
**1.38e-08**) and a **40-second** clip clamps to 32 frames (grid `[16, 22, 30]`, 2,640
video tokens, 16 timestamps, prompt byte-identical, max |dp| **3.71e-07**); the probe
scores both at frames `max|Δ| 0`, patches `max|Δ| 0`, pooled cosine 1.0.

Text and image requests are unchanged (see the numbers below): the text path never builds
a media prompt, and an image request takes `encode` on `[1, H, W]`, which is
`encode_video`'s T = 1 case with the same per-frame loop.

### Deviations, named

- **CLI videos are local paths.** `MediaMode::Paths` opens a file; an `http(s)` URL is
  `videos[i]: no video file at <url>`, where the runtime's in-process loader would
  download it. The server never downloads either (data URLs only), like `d3_server.py`.
- **`--dump-prompts` and images in a video request.** The reference's video path feeds
  `replace_image_token`'s *expanded* image placeholder to its tokenizer while its image
  path leaves one placeholder per image; the model builds one placeholder per image in
  both cases (the ids are identical) and `--dump-prompts` writes the reference's own
  rendering (`model::dump_prompt`), so the dumped text matches in both paths.
- **A container without `nb_frames`.** OpenCV counts the frames (with the 300-second
  guard); this side asks `ffprobe -count_frames`, which decodes the stream once. Both use
  the count the same way; no fixture exercises it.
- **A partially decodable clip keeps the frames that decode** (review fix F3): like
  `_decode_video`, which appends a frame only when OpenCV retrieves it, a truncated clip
  decodes fewer frames than were sampled and the prompt is built from those **with their
  actual source indices** — the pair count, grid and timestamps come from the actual
  frames, and an odd count is padded by repeating the last frame *and index*. Measured on
  a faststart copy of `data/test.mp4` cut at 55 %: OpenCV keeps `[0, 4, 9, 13]` of the
  eight samples and this side decodes exactly those four (grid `[2, 22, 30]`, 330 video
  tokens, timestamps 0.2/1.4). Only a clip none of whose frames decode is
  `no frame of the video could be decoded`.
- **Decoder details are logged, not returned** (review fix F2): both refusals carry the
  reference's own text (`the video could not be opened`, `no frame of the video could be
  decoded`) instead of `ffprobe`/`ffmpeg` stderr, which names the temporary file a
  `data:video/...` upload was written to; the stderr goes to `tracing::warn!`.

Re-verified after phase 3 (dev host, f32 d3-lite, 2026-11-01): text parity **67/67**,
max |Δp| **2.36e-06** (47 cases, 76,829 planned tokens, 583 s); image parity **8/8**,
max |Δp| **7.70e-06** (147.9 s); the image prompt dump is still byte-identical
(`image-cat__animal`, 740 tokens). Server smoke on `/v1/systemone` (d3-lite, port 8399):
a data-URL video answers in 9.2 s (802 tokens, `test_card` at 0.967), two videos 16.7 s,
an image + video 27.6 s (3014 tokens); a local path, an `https` URL, an empty entry, a
`gif` subtype and a non-base64 data URL are each `422 videos[0]: <reason>`; a 32,000,001
byte payload is `422 the decoded data has 32,000,001 bytes, over the 32,000,000 byte
limit`; a 305-second clip is `422 each video must be at most 300 seconds long`; seven
16-second clips (18,480 video tokens) are `422 the videos take 18,480 input tokens, over
the 16,384 tokens a request may spend on videos; send fewer or shorter videos`; an
unknown question type is the usual `422 {"invalid_questions": ...}`.

### Review fixes (phase 3 control)

**F1: `/v1/systemone` has its own body limit.** `1a57e22` raised the router-wide limit
to 48 MiB, which let a 33 MiB `/v1/embeddings` body through where it had been a 413. The
global limit is back to **32 MiB** (`MEDIA_BODY_LIMIT`) and only `/v1/systemone` is
layered at **96 MiB** (`DECISION_BODY_LIMIT`; the per-item ceiling is a 32 MB video plus
three 8 MB images, ≈75 MB of base64). Measured on the running servers: a 33 MiB and a
49 MiB `/v1/embeddings` body → **413** (`Failed to buffer the request body: length limit
exceeded`), 31 MiB → **400** (the transport accepted it), an **85.3 MB** two-video
`/v1/systemone` body → **200** (`test_card` at 0.9990), a 74.7 MB media-sum body →
**422** (`images[0]: the image could not be identified`) and a 98 MiB body → **413**
(the route limit is real).

**F2: decoder details stay out of the answer.** `ffprobe`/`ffmpeg` stderr names the
temporary file a `data:video/...` upload is decoded from; both refusals now carry
exactly the reference's text (`the video could not be opened`, `no frame of the video
could be decoded`) while the stderr goes to `tracing::warn!`. Measured: an upload of
garbage bytes answers `422 {"detail": "videos[0]: the video could not be opened"}` with
no `/tmp/`, `upload-` or `@ 0x` in it (a unit test pins that, and the temporary upload
file is still removed).

**F3: a partially decodable clip answers.** `_decode_video` keeps only the frames
OpenCV retrieves and hands their **actual indices** to `_video`; the decoder now does the
same (one `ffmpeg` pass for the whole list, then a per-index probe when it comes back
short, which is the only way to learn which planned frame numbers survived). A faststart
copy of `data/test.mp4` cut at 55 % decodes `[0, 4, 9, 13]` — OpenCV's own subset — so
the grid is `[2, 22, 30]` with 330 video tokens, timestamps 0.2/1.4 and **467** prompt
tokens; the answer is `test_card` 0.9967110944 against the reference's 0.9967111287
(max |dp| **4.05e-08**) and the prompt byte-identical. Only a clip none of whose frames
decode is `no frame of the video could be decoded` (the same file's header without its
media data: 422).

**F4: images are validated first.** `prepare_media` now preprocesses every image
*before* it plans the video token budget (the reference's `_prepare_videos` renders the
images before it counts the videos), and the server checks and decodes `images` before
`videos`. Measured: a 4000x10 PNG (aspect 400) together with seven 16-second clips
answers `422 {"detail": "image rejected by the processor: absolute aspect ratio must be
smaller than 200, got 400.0"}` instead of the video-budget error, while a valid cat
image with the same clips answers the budget error (`18,480` tokens over `16,384`).

**F5: the reference's tower wording.** `NO_VISION_TOWER` and `NO_VIDEO_TOWER` are the
reference's own strings, `image inputs are not available: the checkpoint has no vision
tower (visual.* weights)` and `video inputs are not available: …`, and a package without
a tower reports the image reason when a request carries both. Measured on a d3-lite copy
without `vision_config`: an image+video request gets the image text, a video-only request
the video text.

**F6: the gate runs from any directory.** The fixtures name their media relative to
`examples/rocket-inference` (`data/test.mp4`) or absolutely (the checkpoint's `assets/`);
`decision_parity.py` and `decision_reference.py` resolve a relative path against
`examples/rocket-inference` (`--media-base` overrides) and print the base. The
9-question gate was re-run from `examples/rocket-inference` and from the repository root:
**9/9** both ways, max |dp| **3.96977e-06**, 95.0 s, 10,843 tokens.

Re-verified after the review fixes (dev host, f32 d3-lite, 2026-11-01): video **9/9**,
max |Δp| **3.96977e-06** (95.0 s, 10,843 tokens, from `examples/rocket-inference` and
from the repository root); text **67/67**, max |Δp| **2.35687e-06** (47 cases, 597.3 s);
image **8/8**, max |Δp| **7.69724e-06**; the literal-placeholder pair **2/2**, max |Δp|
**1.15617e-07**; `cargo test -p rocket-inference --no-default-features` **162 passed**;
`cargo fmt --all -- --check`, clippy (cpu and aarch64 `--features npu`) and the aarch64
npu check clean.

## d3-nano dev-host (phase 4)

`vllm-sr/d3-nano` is the family's second geometry. `config.json` differs from d3-lite in
every text dimension (`text_config` hidden **2048** vs 1024, intermediate **6144** vs
3584, same 24 layers / 8 heads / head_dim 256 / `mrope_section` `[11, 11, 10]`) and in
the tower (`vision_config` depth **24** vs 12, hidden **1024** vs 768, `num_heads` 16 vs
12, intermediate 4096 vs 3072, `out_hidden_size` **2048** vs 1024). The family code is
dimension-generic, so the round's question is only whether the loader, the readout, the
multimodal RoPE and the tower follow those fields end to end. Nothing had to change.

Subset: 7 of the shipped cases / 10 questions (choice, noul, score, one long state, a
single image, two images, one clip), run as **one** `--requests` file — 12,155 prompt
tokens (the file is a plain filter of the three fixture arrays by name, so nothing new is
committed as a fixture):

| cases | questions | tokens | covers |
|---|---|---|---|
| choice-routing-short, noul-refund-window, score-urgency-billing | 3 | 549 | the three answer types at short state |
| choice-policy-long | 3 | 6,307 | a 9,047-character state (3 questions, 2,100-2,107 tokens each) |
| image-receipt-total, image-two-images | 2 | 3,579 | one and two `<\|image_pad\|>` spans |
| video-example | 2 | 1,720 | a 4-pair clip (868 + 852) |

The reference is **f32**, not the official bf16 (the f32 body is what the Rust side
runs); `--dump-vision` is batch-size 1 and writes the nano pooled tensors (`[tokens,
2048]`) for a later probe:

```sh
python3 tools/decision_reference.py --model /tmp/opencode/d3/models/d3-nano \
    --backbone-dtype f32 --requests subset.json --out ref_nano_f32.json \
    --batch-size 1 --dump-vision dump-nano-f32
# 7 cases / 10 questions / 12,155 tokens, 83.7 s of forwards (88.9 s process),
# 13,286 MiB peak RSS / 9,767 MiB peak anon

$CARGO_TARGET_DIR/release/rocket-inference decide \
    --model-dir /tmp/opencode/d3/models/d3-nano --requests subset.json \
    --out our_nano_f32.json --max-state-tokens 32768
# model ready in 3.00 s (24 layers), resident 7,716 MiB anon;
# 10 questions / 12,155 tokens, 152.2 s of forwards (156.4 s process)
```

`tools/decision_parity.py --model-dir /tmp/opencode/d3/models/d3-nano` on that pair:
**10/10 top-1 (100 %)**, max |Δp| **1.741e-06** (`score-urgency-billing/score`; choice
6.86e-07, noul 1.72e-06, score 1.74e-06), **token counts exact on all 7 cases**
(128 / 140 / 281 / 2,107 + 2,100 + 2,100 / 1,029 / 2,550 / 868 + 852 = 12,155),
`RESULT: PASS`. The two-image case answers `first` at 0.99779206 vs the reference's
0.99779201, i.e. the multi-span splice and the 2048-wide tower features are right.

| case | question | type | tokens | top | p | max abs Δp |
|---|---|---|---|---|---|---|
| choice-routing-short | route | choice | 128 | billing | 0.925584971 | 6.71e-08 |
| noul-refund-window | in_window | noul | 140 | true | 0.732728004 | 7.75e-07 |
| score-urgency-billing | score | score | 281 | level 1 | 0.417736483 | 1.74e-06 |
| choice-policy-long | verdict | choice | 2,107 | violation | 0.939493302 | 1.46e-07 |
| choice-policy-long | approval | noul | 2,100 | true | 0.998911441 | 6.06e-08 |
| choice-policy-long | risk | score | 2,100 | level 0 | 0.653321626 | 9.70e-08 |
| image-receipt-total | total | choice | 1,029 | over_150 | 0.896590022 | 5.33e-07 |
| image-two-images | which | choice | 2,550 | first | 0.997792122 | 1.17e-07 |
| video-example | direction | choice | 868 | right | 0.986120553 | 6.86e-07 |
| video-example | color_change | noul | 852 | true | 0.972875702 | 1.72e-06 |

The two checkpoints disagree on 4 of these 10 answers (nano says `billing`,
`violation`, risk level 0 and `over_150`; lite says `returns`, `needs review`, risk level
1 and `100_to_150`), which is why the gate is per-checkpoint: parity is against each
model's own reference, and lite answers the same subset exactly as its own references
do (`ref_text_lite_f32.json`, `ref_images_lite_f32.json`, `ref_videos_lite_f32.json`
— 10/10).

Wall and memory, d3-nano against d3-lite on the same 12,155-token subset (dev host, 32
cores, f32, 2026-11-01). The per-case columns are one process per case, so they include
the model load (3.0 s nano, 1.25 s lite); the totals row is the single `--requests`
process, whose "compute" is the CLI's own number (load excluded):

| case | tokens | nano wall | nano anon | lite wall | lite anon |
|---|---|---|---|---|---|
| choice-routing-short | 128 | 6.9 s | 7,774 MiB | 3.3 s | 2,978 MiB |
| noul-refund-window | 140 | 7.1 s | 7,844 MiB | 3.5 s | 2,983 MiB |
| score-urgency-billing | 281 | 8.2 s | 7,864 MiB | 4.0 s | 3,054 MiB |
| choice-policy-long | 6,307 | 51.2 s | 8,856 MiB | 37.3 s | 4,025 MiB |
| image-receipt-total | 1,029 | 25.3 s | 8,339 MiB | 12.0 s | 3,484 MiB |
| image-two-images | 2,550 | 56.0 s | 9,222 MiB | 28.6 s | 4,373 MiB |
| video-example | 1,720 | 28.2 s | 8,200 MiB | 14.1 s | 3,363 MiB |

| model | questions | load | compute | wall | peak RSS | peak anon | user CPU |
|---|---|---|---|---|---|---|---|
| d3-nano f32 (Rust) | 10 | 3.00 s | 152.2 s | 156.4 s | 12,061 MiB | 9,220 MiB | 1,722 s |
| d3-lite f32 (Rust) | 10 | 1.25 s | 89.4 s | 91.3 s | 4,631 MiB | 4,391 MiB | 788 s |
| d3-nano f32 (torch reference) | 10 | — | 83.7 s | 88.9 s | 13,286 MiB | 9,767 MiB | 1,181 s |

So nano costs **1.71x** the wall and **2.10x** the anon (2.60x the RSS, which also holds
the checkpoint's file-backed pages) for the same answers; the resident f32 body alone is
7,716 MiB anon against lite's 2,926 MiB. Two things to read out of the per-case column:
a short prompt is *weight-bandwidth*-bound, not attention-bound (a 128-token forward
streams the whole resident body — ~7.7 GiB anon on nano, ~2.9 GiB on lite — in ~3.9 s
against ~2.1 s), so the flat ~7 s for the three short cases is the load plus one
streaming pass; and a media request scales with **patches**, not with the text length
(the receipt is 3,472 patches for its 1,029 tokens, the two-image request 9,744 for
2,550, and the tower dominates that wall: 25.3 s vs 6.9 s for the short text case on
nano, 12.0 s vs 3.3 s on lite).

What the run pins, field by field (all read from the checkpoint, no constants):

- **text**: `IntentTextConfig` is `serde`-deserialized from `config.json` `text_config`;
  the loader checks `layer_types.len() == num_hidden_layers`, loads `0..num_hidden_layers`
  keys and prints the config's own count (`decision model ready in 3.00s (24 layers,
  embed F16)`). hidden 2048 / intermediate 6144 / 24 layers come from the file.
- **readout**: `load_readout(dir, backbone.cfg.hidden_size)` refuses any weight whose
  shape is not exactly `[255, hidden]`; nano's `readout.safetensors` is `[255, 2048]`,
  lite's `[255, 1024]`, and both load.
- **tower**: `VisionConfig::from_value` reads the `vision_config` object (depth 24,
  hidden 1024, `num_heads` 16 → head_dim 64, `out_hidden_size` 2048), the loader loops
  `0..depth` and bails unless the applied count equals `3 + 12 * depth + 6` = **297**
  tensors for nano (153 for lite). The checkpoint holds exactly 297 `visual.*` tensors
  over 24 blocks, so a wrong depth would be a load error, not a silent partial tower.
- **splice**: the tower's merged rows are `out_hidden_size` wide and the prompt write
  errors unless they match the text embedding width (`the vision features are {width}
  wide, the text embeddings {hidden}`) — 2048 here, 1024 for lite.
- **mrope**: the tables are built per prompt from the text config
  (`mrope::rope_tables(..., &self.backbone.cfg, merge, ...)`), so the rotary width is
  `head_dim * partial_rotary_factor` (64, the same in both checkpoints) and the section
  split is `rope_parameters.mrope_section`.

One measurement note: the reference must be f32 here too. nano bf16 (the existing
`ref_nano_4cases.json`) and this f32 reference agree on all six text answers but differ
by up to **6.6e-3** in probability (`noul-refund-window`, 3.1e-3 at
`choice-policy-long/risk`), i.e. above the 2e-3 gate — the same body-dtype rule the
phase-1 text section gives for lite.

Not measured here: **the board NPU run is still pending**. This round is the dev host
only (f32 CPU body, no device); the NPU offload needs the aarch64 build plus the
RK3588's 600 MHz module, and the numbers above say nothing about it. Also noted for a
future re-run: `image-two-images` names the *d3-lite* `serving-throughput.png` (the
fixtures' absolute asset paths), and d3-nano ships a **different** file under that name
(`4603c2c3…` vs `c6536167…`); both sides consumed the same file here, so the parity
holds, but a run that mixed the packages' assets would not be comparable.

## Token table (phase M2)

The tied token table (248,320 × hidden) is CPU resident in every deployment, and a
prompt only reads its own rows, so the decision family stores it as per-row symmetric
int8 by default (`--embed-int8`; `--embed-f16` keeps the dense table, and the two flags
are mutually exclusive). What that costs is measured on the full fixture sets against
the f16 arm (dev host, f32 body, 32k cap; the f16 table is bit-exact for these bf16
checkpoints, so the whole delta is int8):

| set | questions | top-1 | max \|Δp\| | median \|Δp\| |
|---|---|---|---|
| text | 67 | 67/67 | 6.96e-3 | 7.3e-4 |
| images | 8 | 8/8 | 8.36e-3 | 1.4e-4 |
| videos | 9 | 9/9 | 3.81e-3 | 8.6e-4 |

For the halved table: resident anon d3-lite 2,462 → **2,222** MiB and d3-nano
6,287 → **5,804** MiB (peak 2,583 → 2,397 and 6,457 → 5,969).

The phase's 2e-3 **probe** gate is exceeded by design: per-row symmetric int8 carries an
inherent `max|row|/254` error (1.46e-4 per element at this table's median row max
0.0747, 0.86 % of the row RMS), and even block-32 scales would only buy 1.56x. The
approved deployment bar is instead **max |Δp| ≤ 1e-2 with 100 % top-1**, which this
passes: no answer changes on the 84 fixture questions and the largest probability move
is 8.36e-3. `--embed-f16` restores the exact table (f16-vs-f32 max |Δp| 2.4e-6) for
+242 MiB on d3-lite or +485 MiB on d3-nano — use it where the probabilities themselves
have to match the f32 reference.

Reproduce (from `examples/rocket-inference`; `--embed-f16` is the baseline arm):

```sh
B=$CARGO_TARGET_DIR/release/rocket-inference
M=/tmp/opencode/d3/models/d3-lite
for set in decision_fixtures decision_fixtures_images decision_fixtures_videos; do
  for arm in f16 i8; do
    flag=""; [ $arm = f16 ] && flag=--embed-f16
    $B decide --model-dir $M --requests tools/$set.json --max-state-tokens 32768 \
        $flag --out /tmp/$set-$arm.json
  done
done
# top-1 and the largest per-option probability move, both arms
python3 - <<'PY'
import json
for s in ("decision_fixtures", "decision_fixtures_images", "decision_fixtures_videos"):
    a = {c["name"]: c for c in json.load(open(f"/tmp/{s}-f16.json"))["cases"]}
    b = {c["name"]: c for c in json.load(open(f"/tmp/{s}-i8.json"))["cases"]}
    worst = 0.0; bad = 0; n = 0
    for name, case in a.items():
        for qid, va in case["response"]["answers"].items():
            vb = b[name]["response"]["answers"][qid]; n += 1
            pa = va.get("probabilities") or {"_": va["noul"]}
            pb = vb.get("probabilities") or {"_": vb["noul"]}
            worst = max(worst, max(abs(pa[k] - pb[k]) for k in pa))
            if va.get("choice") != vb.get("choice"):
                bad += 1
    print(f"{s}: {n} questions, top-1 mismatches {bad}, max |dp| {worst:.3e}")
PY
```

Memory, text-only `--defer-towers`, one 128-token case (`RssAnon` sampled every 50 ms;
`--embed-f16` in parentheses): d3-lite resident **2,222** (2,462) MiB, peak **2,397**
(2,583); d3-nano resident **5,804** (6,287), peak **5,969** (6,457). The loader consumes
the checkpoint's f32 table in place, so the int8 load also peaks lower than the f16 cast.
