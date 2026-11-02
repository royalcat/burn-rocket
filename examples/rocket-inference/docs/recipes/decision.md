# vllm-sr d3 decision recipes

[vllm-sr Decision 3.0](https://huggingface.co/vllm-sr/d3-lite) (`d3-code-readout-v1`)
is the typed decision model: a Qwen3.5-0.8B backbone plus a 255-code readout head.
A request is a `state` and a set of typed `questions` (`choice`, `noul`, `score`);
the answer is one non-causal forward pass per question and a softmax over the
codes. Nothing is generated (`usage.output_tokens` is always 0), so both the CLI
and the server prefill only.

Checkpoint: `/root/models/d3-lite` (1.7 GB safetensors; the backbone is stored
bf16 and widened to f32 per matmul, plus `readout.safetensors`). d3-nano runs
through the same code but is validated on the dev host only; the numbers here
are d3-lite, except where nano is named.

## Decide

```sh
M=/root/models/d3-lite
B=./rocket-inference
CASES=cases.json          # JSON array of {"name", "state", "questions"[, "images"[,
                          # "videos"]]}; local paths for media

# CPU compute (f32; the body is stored bf16 and widened per matmul)
$B decide --model-dir $M --requests $CASES --out out-cpu.json \
    --max-state-tokens 8192 --device cpu

# text backbone projections packed as fp16 resident NPU weights
$B decide --model-dir $M --requests $CASES --out out-npu.json \
    --max-state-tokens 8192 --device npu
```

The placement line reports the split:

```
placement: text=CPU(f32) readout=CPU(f32)
placement: text=NPU(fp16) readout=CPU(f32)
```

`text` is the **only** NPU stage of this family. The vision tower used by image
and video questions, the attention inside the backbone and the readout head all
stay on the CPU in both configurations; there is no `--text-attn-device`,
`--vision-device` or `--decode-device` flag for d3. The NPU config packs the
text projections and runs the prefill GEMMs on them; it does not move the tower
or the attention.

`--max-state-tokens` is the per-question prompt cap (default 32768). The long
policy case runs three ~2,100-token prompts over the same state, so the cap
applies per question, not to a whole case; 8192 is a safe board setting.

The output is the official shape
`{"model", "answers", "usage"}`; `--requests` wraps the cases as
`{"schema": "d3-decision-v1", "model", "cases": [{"name", "response"}]}`.

### Token table

The tied token table (vocab 248,320 × hidden) is CPU resident in both
configurations. `decide` and `serve` store it as **block-wise symmetric int8** by
default (`--embed-int8`): one f32 scale per 32 hidden values (`blocks = hidden / 32`,
`scales[row * blocks + j] = max|block j| / 127`, codes `round(v / scale)` clamped to
±127; a hidden state that is not a multiple of 32 is refused at load with a pointer to
`--embed-f16`). The codes halve the table and only the current prompt's rows are
dequantized per forward; the scales add `vocab * blocks * 4 B` — **30.3 MiB** on
d3-lite (32 blocks) and 60.6 MiB on d3-nano (64) — so d3-lite is **1,302 MiB**
resident against 1,513 for the f16 table and 1,273 for the per-row scheme it replaced.
`--embed-f16` keeps the exact table — bit-exact for these bf16 checkpoints — and the
two flags are mutually exclusive.

The table is quantized **from the checkpoint's stored dtype in 1024-row blocks**, so
the load never holds a vocab-sized f32 copy (lite and nano startup peaked at
2,473 / 4,796 MiB before that; M5b measured 1,330 / 3,349 MiB with the body and table
stored in 16 bits, and the block scales add their 30 / 61 MiB on top).

Block scales cost probability, not answers: on the bounded dev-host subset (the
6-case text subset, `image-receipt-total` and `video-example`, f16 table as the
baseline, which is exact for these checkpoints) top-1 is unchanged (10/10, 1/1, 2/2)
while the largest option probability moves by **1.465e-3** (text), **1.480e-3**
(image) and **9.788e-4** (video) — 2.6x, 5.6x and 3.4x better than the per-row scheme
on the same cases (3.769e-3 / 8.362e-3 / 3.334e-3). The per-row full-set numbers were
6.96e-3 / 8.36e-3 / 3.81e-3 with top-1 unchanged over 67 / 8 / 9 questions; the block
arm was not swept over the full sets. Per-row int8's error is `max|row|/254` and
cannot meet the phase's 2e-3 **probe** gate; block scales do not either, but they halve
the distance, and the approved deployment bar is **max |Δp| ≤ 1e-2 with 100 % top-1**.

Use `--embed-f16` where the probabilities themselves must match the f32 reference: it
restores the exact table (f16-vs-f32 2.4e-6) for +211 MiB (lite) / +424 MiB (nano).
On the board the bar was tight once the NPU pack was added: with per-row scales the
text arm read 4.15e-3 but the image arm landed at 1.03e-2 (table 8.36e-3 + NPU
1.95e-3), marginally over. Block scales put the same image case at **1.664e-3** (the
f16 table reads 5.24e-4 there), inside the bar — see the Board section.

### Prefill memory

A request is one prefill per question and no decode, so nothing prompt-sized is
retained between questions. Two retentions used to make a long prompt expensive
and are gone (exact: the `decide` JSON is byte-identical before/after at 2.1k and
7,893 tokens):

- **Conv history**: `DeltaNet::conv` stored a Burn `slice` of the left-padded
  conv input, and a `slice` is a *view* of the parent buffer — every one of the
  18 linear layers pinned its `[1, 6144, s + 3]` f32 block (0.19 GiB per layer,
  3.5 GiB at 8k). The state is a 75 KiB copy now.
- **KV store**: the full-attention layers used to keep the prompt's K/V for a
  decode step that never comes (24.5 KiB/token: 194 MiB at 8k, ~800 MiB at the
  32k cap). `IntentModel.keep_kv = false` in the decision loader skips it.

Dev host (M4 round: `d3-lite`, text-only `--defer-towers`, resident 2,222 MiB at
the time — the body and table storage below bring it to 1,273 MiB per-row /
1,302 MiB with the block-32 scales — `RssAnon` sampled at 20 ms):

| prompt tokens | peak anon before | peak anon after | transient after | wall |
|---|---|---|---|---|
| 2,102 × 3 questions (6,307) | 3,380 MiB | **2,477 MiB** | 255 MiB | 37.0 s |
| 7,893 | 6,385 MiB | **3,048 MiB** | 826 MiB | 44.1 s |

The remaining transient is the delta-rule chunk working set plus the per-layer
MLP/attention activations, so a long prompt now fits a board that could not host
the old 8k peak. Blocking the chunk dimension was not needed (it does not touch
the conv term) and is not implemented.

### Text body storage

The text backbone's projections are stored in the checkpoint's own **bf16** and
widened per matmul (exact: bf16 is a truncated f32), which halves the body:
resident `RssAnon` d3-lite **2,222 -> 1,273 MiB**, d3-nano **5,804 -> 3,186**
(text-only `--defer-towers`, one 2,113-token case). The block-32 scale table
(M2b) adds 30.3 / 60.6 MiB on top; lite's measured current figure is **1,302
MiB**, nano's block-32 resident was not re-measured (3,186 is the M5 number with
the per-row table). Output is byte-identical; the
`placement:` line keeps saying `text=CPU(f32)` because the *compute* dtype is
still f32 (the ready line adds `, text body bf16`). It costs one pass over the
body per prefill: the 6-case text gate 90.9 -> 96.3 s (**+6.0 %**), image
9.97 -> 10.2-10.5 s, video 12.35 -> 13.4-13.6 s; nano's 2.1k prefill is
17.13 -> 16.79 s (its f32 body is cache-hostile enough that the widen does not
show). A rayon widen of the same conversion measured slower (97.9 s median), so
the plain `cast` stays.

The **startup peak** follows the resident figure now that the int8 table is
quantized from the stored dtype in row blocks (no f32 zeros table, no f32 read):
lite 2,473 -> **1,330** MiB against a 1,273 MiB resident and nano 4,796 ->
**3,349** MiB against 3,186 (text-only `--defer-towers`, one 2,113-token case,
`RssAnon` sampled at 20 ms; the M2b block scales add ~30 / 61 MiB of f32 scale
table on top, so d3-lite is **1,302 resident / 1,382 startup**). `--embed-f16` lite
loads at 1,607 MiB for its
1,513 MiB resident. A board deployment should budget nano's *startup* against
~3.4 GiB, and roughly the resident figure afterwards.


## Serve

```sh
# the recommended board line (see `## Deployment (current)`); this is the only
# command a d3 deployment needs
$B serve --model-dir $M --device npu --defer-towers --max-tokens 8192 --port 8384

# the same, with the NPU scratch line per request (dev host / debugging)
ROCKET_MEM_LOG=1 $B serve --model-dir $M --device npu --defer-towers \
    --max-tokens 8192 --port 8399
```

`/v1/systemone` takes the same request object (media as `data:` URLs), `/health`
and `/v1/models` are always available. `--max-tokens` is the per-question prompt
cap. The server logs one line per request with `questions`, `tokens` and
`compute_s`, and (with `ROCKET_MEM_LOG=1`) the `npu scratch` line after load and
after every request. With OpenTelemetry enabled (`OTEL_SDK_ENABLED=true`) the
`inference` span gains a `decision` child with `questions`, `images`, `videos`,
`answers` and `tokens` (`answers` is 0 on a refused or failed request).
The CLI commands install no tracing subscriber, so `ROCKET_MEM_LOG` prints
nothing in `decide` runs; use `serve` when the BO split matters.

## Profiles

d3 has a single NPU stage (`text`), so the three deployment profiles collapse
into two configurations:

| profile | d3 command |
|---|---|
| balanced | `--device npu --defer-towers` (text projections packed as resident fp16; the block-32 int8 table is automatic) |
| full offload | `--device npu --defer-towers` (same: there is nothing else to move) |
| speed | `--device npu --defer-towers` (same: the compute is the same command) |
| CPU fallback | `--device cpu` (f32 compute, bf16-stored body widened per projection) |

The vision tower, the backbone attention and the readout head run on the CPU in
both configurations: d3 has no `--text-attn-device`, `--vision-device`,
`--vision-attn-device` or `--decode-device` stage, and no decode at all.
`--npu-dtype int8` (resident int8 projections, **474.0 MiB** of weights BO
against 966.0) is available but **not recommended** for this family: it costs
+6.5 % wall and +18 % user CPU at `--int8-group 512` and misses the 1e-2
probability bar by 3-4.5x (text 3.4e-2, image 4.4e-2) — see the int8 table in
the Board section.

## Caps

- `decide --max-state-tokens <n>` and `serve --max-tokens <n>` are the same cap:
  the **per-question** prompt length (default **32768**, each question starts
  from the shared state). A case with three long questions therefore spends the
  cap three times.
- A question over the cap is not a request error: it keeps a
  `{"type": ..., "error": "max_length_exceeded", "message": ...}` record in
  `answers` of a 200 response and is excluded from `usage.input_tokens`; only a
  malformed *question* refuses the whole request (422) before anything runs.
- Media caps come from the reference runtime and are fixed: images 8,000,000
  decoded bytes / 16,000,000 pixels, videos 32,000,000 bytes / 8,294,400 pixels
  per frame / 300 s, and **all videos of one request together ≤ 16,384 input
  tokens** (the server takes `data:` URLs only).
- The board recipe passes 8192 (`--max-state-tokens 8192` /
  `--max-tokens 8192`): the long policy case is three ~2,100-token prompts over
  one 6,307-token state, so 8192 leaves headroom without sizing the RoPE table
  for 32k on a busy board.

## Dev host (d3-lite / d3-nano)

Both checkpoints, f32 compute with an f32-resident body and token table (the
2026-11-01 baseline), on the 32-core dev host. The subset
is seven cases: four text (`choice-routing-short`, `noul-refund-window`,
`score-urgency-billing` and the three-question `choice-policy-long`), one
receipt image, one two-image request and one video clip — 10 questions and
12,155 prompt tokens, identical for both models because the prompts come from
the checkpoint-agnostic fixtures. `compute` is the CLI's own decided time (load
excluded), `wall` adds the load, `user` is process CPU-seconds, peak RSS/anon is
`/proc/<pid>/status` sampled over the run.

| model | questions / tokens | top-1 (max \|Δp\|) | load | compute | wall | peak RSS | peak anon | user CPU |
|---|---|---|---|---|---|---|---|---|
| d3-nano f32 | 10 / 12,155 | 10/10 (1.741e-06) | 3.00 s | 152.2 s | 156.4 s | 12,061 MiB | 9,220 MiB | 1,722 s |
| d3-lite f32 | 10 / 12,155 | 10/10 | 1.25 s | 89.4 s | 91.3 s | 4,631 MiB | 4,391 MiB | 788 s |
| d3-nano f32 (torch reference) | 10 / 12,155 | — | — | 83.7 s | 88.9 s | 13,286 MiB | 9,767 MiB | 1,181 s |

The parity reference for d3-nano is its **f32** reference
(`tools/decision_reference.py --backbone-dtype f32`), the same body dtype the
Rust side runs. The checkpoint's bf16 path is not interchangeable: nano bf16 and
the f32 reference agree on all six text answers but differ by up to **6.6e-3**
in probability (`noul-refund-window`; 3.1e-3 at `choice-policy-long/risk`),
above the 2e-3 gate, so a bf16 reference would report false mismatches.

nano is the larger geometry (24 layers, hidden 2048, intermediate 6144, a
2048-wide tower output): 1.71x the wall and 2.10x the anon for the same answers, with the
resident f32 body alone at 7,716 MiB against lite's 2,926 MiB. The two
checkpoints answer 4 of these 10 questions differently, so parity is per
checkpoint — each against its own reference. Full detail (per-case wall/memory,
the field-by-field config read, the two-image splice) is in
[`tools/decision.md`](../../tools/decision.md).

This table is the f32-resident baseline, not the current memory profile: the
body is now stored bf16 (M5) and the table block-32 int8 (M2b), so a text-only
d3-lite process is **1,302 MiB** resident (2,926 here; 1,273 with the per-row
table) and d3-nano **3,186** (7,716 here). The body change is byte-identical on
the fixtures; the table's own deltas (1.465e-3 / 1.480e-3 / 9.788e-4) are in
`### Token table`. Compute is still f32, so the wall/compute columns describe
the same arithmetic.

## Deployment (current)

The recommended board configuration is this one line — every memory default
below is already in it, and nothing else needs a flag:

```sh
/root/rocket-inference-decision/rocket-inference serve \
    --model-dir /root/models/d3-lite --device npu --defer-towers \
    --max-tokens 8192 --port 8384
```

| part | current setting | measured cost |
|---|---|---|
| text projections | fp16 resident NPU pack, 96 weights | **966.0 MiB** of weights BO (948.0 MiB in the ready line) |
| token table | block-32 symmetric int8 (`--embed-int8`, the default) | **347 MiB** anon resident for the text-only service, 30.3 MiB of it the scale table |
| text body | the checkpoint's bf16, widened to f32 per matmul | half the CPU-resident body; output is bit-identical |
| vision tower | deferred (`--defer-towers`) | +195.6 MiB, only after the first media request |

Text-only service, measured with that line on the board (2026-11-02, beside the
live embedding service, load 15-21, `RssAnon` sampled every 50 ms):

- **347 MiB** anon resident and **966.0 MiB** of weights BO, i.e. about
  **1.28 GiB** of NPU-side total (347 + 966.0 = 1,313 MiB) against the pre-round
  **1.90 GiB** (943.6 anon + 1,006.7 BO). The per-row table arm is 317 MiB /
  **1.25 GiB**; the f16 table (`--embed-f16`) is 559 MiB.
- **Image** (1,029 tokens): **1.664e-3** max |Δp| against the f32 reference with
  top-1 unchanged; peak anon 1,027.4 MiB. The f16 table reads 5.238e-4 there,
  the per-row table 1.03e-2 (over the bar).
- **Video**: 1.55e-2, the board's `ffmpeg` 7.1.5 decode (the CPU arm reads
  1.51e-2 on the same clip); top-1 is correct. Gate cross-host video on top-1 or
  decode the reference with the same `ffmpeg`.
- **Wall**, all contended: text 5 questions / 6,546 tokens **147.3 s** compute /
  **160.2 s** wall (CPU f32 265.6 / 270.1 s); image 1,029 tokens **62.9 s**
  compute; video 1,720 tokens / 2 questions **61.7 s** wall, 54.0 s compute.
  The text and video walls are the M6 window's **per-row-table** arms; block-32
  was re-measured on the image case only (no wall change outside the board's
  noise: a table is dequantized per prompt row, and the scale table is 30.3 MiB
  of resident bytes). The 2026-11-01 idle-service window measured the same text
  case at 113 s NPU vs 197 s CPU f32; every 2026-11-02 wall is 25-35 % higher
  because the service stayed up, so compare only within a window.

The approved deployment bar is **max |Δp| ≤ 1e-2 with 100 % top-1**; the block-32
table is inside it on every shipped arm. `--embed-f16` restores the exact table
(f16-vs-f32 2.4e-6) for +211 MiB (lite) / +424 MiB (nano) where the
probabilities themselves must match the f32 reference.

`--npu-dtype int8` exists for this family but is **not recommended**: it halves
the weights BO (**474.0 MiB** vs 966.0) at **+6.5 %** wall and **+18 %** user CPU
at `--int8-group 512`, does not move the resident or peak anon, and moves text by
**3.4e-2** and image by **4.4e-2** — 3-4.5x over the bar. `--int8-group 128` is
both slower and less accurate (216.3 s wall, 4.5e-2). The per-arm tables are the
memory-round subsection at the end of the Board section.

The CPU fallback is the same binary with `--device cpu`: f32 compute on the
bf16-stored body — 1,265 MiB resident for a text-only service in the M6 window
(that arm ran the per-row table; the block-32 scales add their 30.3 MiB on top,
and the CPU arm was not re-measured with them).

## Board (d3-lite)

Board: Rock 5B+, 8 cores (A76 CPUs 4-7 via `taskset -c 4-7`), kernel
7.1.8-edge-rockchip64, patched `rocket` module at 600 MHz
(`rocket_npu_clk_hz=600000000`; dmesg reports `core N NPU clk -> 600000000 Hz
(reads back 600000000)`). Load average 7.8-10.7 from the other board services;
the live embedding service `home-llm-gateway-embed-1` was stopped for the runs
(see below). Runs are one process per config, two samples each (taken before the
service was restored); outputs are bit-identical between repeats. The deployed
copy is `/root/rocket-inference-decision/` (binary, fixtures and run outputs).
The video CPU `user` column has one valid sample (see the note below the table).
This table is the **2026-11-01 window and a pre-memory-round baseline** (f32-loaded
CPU-side body, per-row int8 table, eager vision tower); read the current
configuration from `## Deployment (current)` above. The memory/int8 round of
**2026-11-02** ran beside the live service on a busier board and is the
subsection at the end of this section.

Fixtures (board paths):

- text: `choice-routing-short`, `noul-receipt-short`, `choice-policy-long` from
  `tools/decision_fixtures.json` (5 questions, 6,546 prompt tokens; the long case
  has three questions).
- image: `image-receipt-total` from `tools/decision_fixtures_images.json`, with
  `images: ["/root/models/d3-lite/assets/example-receipt.png"]` (1,029 tokens).
- video: `video-example` from `tools/decision_fixtures_videos.json`, with
  `videos: ["/root/models/d3-lite/assets/example-video.mp4"]` (1,720 tokens over
  2 questions).

`compute` is the CLI's `decided ... in` time for the whole case set (it includes
image/video decode and the vision tower); `wall` adds the model load (4-9 s) and
teardown. `user` is process CPU-seconds; peak anon is `RssAnon` sampled every
50 ms, with peak RSS in parentheses (the RSS includes the 1.7 GB checkpoint's
file pages).

| input (questions) | placement | tokens | compute | wall | user CPU | peak anon (RSS) |
|---|---|---|---|---|---|---|
| text (5) | CPU f32 | 6,546 | 197 s (194-200) | 205 s (204-206) | 453 s (384-522) | 3,999 (4,586) MiB |
| text (5) | NPU fp16 | 6,546 | 113 s (111-116) | 122 s (119-125) | 144 s (141-146) | 2,505 (3,001) MiB |
| image (1) | CPU f32 | 1,029 | 56.7 s (54.5-58.9) | 63.0 s (60.9-65.1) | 130 s (127-132) | 3,468 (4,598) MiB |
| image (1) | NPU fp16 | 1,029 | 46.3 s (45.2-47.4) | 56.2 s (53.7-58.7) | 97.3 s (97.2-97.4) | 2,497 (2,993) MiB |
| video (2) | CPU f32 | 1,720 | 62.1 s (60.2-64.0) | 68.4 s (66.6-70.3) | 129 s | 3,354 (4,584) MiB |
| video (2) | NPU fp16 | 1,720 | 40.8 s (38.5-43.1) | 49.1 s (47.0-51.3) | 67.3 s (66.6-68.0) | 2,513 (3,009) MiB |

The video CPU `user` value is one sample: a second run's rusage reported 424 s of
user time for 66.6 s of wall on a 4-CPU affinity (6.4 core-seconds per wall
second, which is impossible); its compute/wall/anon are inside the ranges above
and are kept, the CPU reading is discarded. The other configs have two readings
each; the text CPU user range (384-522 s) is the widest.

NPU memory in the 2026-11-01 window (the same binary's `serve`, after load and
after a 128-token `/v1/systemone` request, `ROCKET_MEM_LOG=1`): 96 resident NPU
weights, **1,006.7 MiB** of weights BO; matmul BO **93.7 MiB** after load, trimmed
to **26.4 MiB** BO + 6.3 MiB host after the request (`ROCKET_KEEP_MB=64`);
attention BO/host **0** (there is no NPU attention stage). `RssAnon` is 943.6 MiB
after load and 1,034.3 MiB after the request — that is the f32-loaded body and
the per-row table; the current 347 MiB and 966.0 MiB BO are in
`## Deployment (current)`. The BOs are not charged to the process, so the anon
peaks above exclude the ~1 GiB of NPU weights. The 128-token server request
computed in 1.98 s (64.7 tok/s) with the model already resident.

Parity against the f32 reference (`/tmp/opencode/d3/ref_*_lite_f32.json`, the
same case names; gate 2e-3). Prompt-token counts match exactly in every arm.

| input | placement | top-1 | max \|Δp\| |
|---|---|---|---|
| text | CPU f32 | 5/5 | 1.4e-06 |
| text | NPU fp16 | 5/5 | 3.4e-04 |
| image | CPU f32 | 1/1 | 7.9e-06 |
| image | NPU fp16 | 1/1 | **1.63e-03** |
| video | CPU f32 | 2/2 | 1.40e-02 |
| video | NPU fp16 | 2/2 | 1.45e-02 |

The video arms miss the 2e-3 probability gate although both top-1 answers are
correct. The reference was produced on the dev host; the board's `ffmpeg`
7.1.5 decodes the sampled frames with a per-channel difference of up to 2/255
against the dev-host `ffmpeg` n9.0.2 (verified bit-identical to the OpenCV
decode the reference used): 67 % of the RGB bytes differ, mean 0.68/255, and the
vision tower turns that into a ~1e-2 probability shift. The same-board
CPU-vs-NPU video delta is only 4.6e-4, so the miss is the decoder version, not
the Rust body or the NPU pack. The image fixture's decode matches across the two
environments, which is why its deltas stay near the gate.

Reading the table:

- **Text is the NPU win**: 1.7x the compute (197 -> 113 s) and ~3x less user CPU
  (453 -> 144 s) at 37 % lower peak anon; its prefill is projection-heavy.
- **Image/video gain less** because the vision tower and the media decode are
  CPU work in both configs: image 56.7 -> 46.3 s (1.2x) and video
  62.1 -> 40.8 s (1.5x), while user CPU still drops 1.3-1.9x.
- **Memory**: the CPU body is f32 and ~2.9 GiB resident; the NPU config keeps
  ~1.0 GiB anon (embeddings, vision tower, readout) plus the ~1.0 GiB weights
  BO. Peak anon drops from 3.3-3.9 GiB to ~2.5 GiB on every input class. These
  board numbers predate the bf16 text-body storage (§ Text body storage): the CPU
  arm's resident figure is now about 1.0 GiB lower (dev-host −949 MiB for lite),
  and its wall carries the per-prefill widen (+2-4 % on a 1k-token question whose
  prefill is ~40 s on the board). The NPU arm's anon side drops the same way
  (**943.6 -> 317 MiB** on the per-row table, **347 MiB** with the default
  block-32 scales; the bf16 body plus the streamed int8 table) while its weights
  BO stays ~1 GiB — the 2026-11-02 round below has the current numbers.
- **Board noise**: the other services keep the load at ~8-10, so wall/CPU spread
  by ±20-30 % between samples. The runs in the table were taken with the
  embedding service stopped; the two samples of each config are the pre-restore
  pair.

Service stop/restore and free memory:

- Before: `home-llm-gateway-embed-1` `running healthy` (container 3.48 GiB),
  5,517 MiB host available. Stopped with
  `docker stop home-llm-gateway-embed-1` (it stayed `exited` for the whole run;
  Komodo did not restart it, so the compose stop was not needed). Port 8383 was
  no longer listening; available memory rose to **9,068 MiB** (used
  10.0 -> 6.6 GiB).
- During the runs `/usr/bin/time` is not installed on the board, so peak
  anon/RSS was sampled from `/proc/<pid>/status`; the largest run peaked at
  3.9 GiB anon, well inside the 8.7-9.1 GiB available.
- Restored with `docker start home-llm-gateway-embed-1`; `healthy` after ~65 s
  (start period), then one `/v1/embeddings` request from inside the container
  returned 200 in 0.32 s (4 tokens, `compute_s=0.294` in the service log). The
  service then resumed its normal production traffic (49-50-frame video
  requests), so later requests queue behind it. Nothing else was stopped or
  restarted.

Notes:

- The `decide` CLI keeps one process per run: the text samples include the
  6,307-token long case, so their ~3.4 min is dominated by it. For interactive
  latency, serve once: the 128-token short case is 1.98 s of compute with the
  model resident (the one-shot CLI pays 4-9 s of load on every invocation).
- Attention is CPU in both configs (the d3 family has no NPU attention stage).
  The NPU BO line's `attn_bo_mib=0` is expected, not a disabled feature.
- Video parity is decoder-version sensitive; compare board outputs against a
  reference produced with the same `ffmpeg`, or accept the top-1-only gate for
  cross-host video checks. Text and image are stable at the 2e-3 gate.

### Memory, int8 and tower round (2026-11-02)

The binary now carries the bf16 text-body storage, the streamed int8 token table
and the deferred tower halves (see `### Prefill memory`, `### Token table` and
`### Text body storage`), so this round re-measured the same named fixtures. It
ran **beside the live service** — `home-llm-gateway-embed-1` stayed `running
healthy` the whole time (available memory 5.5-7.2 GiB, `free` 1.3-2.5 GiB, swap
flat at 2.3-2.4 GiB used, no OOM), so no stop/restore was needed. Board load was
15.6-21.0 during the runs (13.2 at the end) against 7.8-10.7 for the 2026-11-01
table: **every wall/CPU number below is contended and only comparable within
this window** (the 2026-11-01 CPU/NPU ratio is the useful historical reference).
Same fixtures, same `taskset -c 4-7`, one process per run, `RssAnon` sampled
every 50 ms; the CLI prints `decision model resident … anon` after load and
`decided … (N anon)` after the requests. Command shape:

```sh
taskset -c 4-7 ./rocket-inference decide --model-dir /root/models/d3-lite \
    --requests board-text.json --out runs6/<tag>-out.json --max-state-tokens 8192 \
    --device npu --defer-towers [--npu-dtype int8 --int8-group 512|128] [--embed-f16]
```

Text, 5 questions / 6,546 prompt tokens (`--defer-towers`, so text-only):

| NPU weights / token table | compute | wall | user CPU | peak anon | anon after | resident after load | weights BO |
|---|---|---|---|---|---|---|---|
| fp16 / per-row int8 table (M2's default) | 147.3 s (144.8, 149.8) | 160.2 s (159.9, 160.5) | 153.8 s | 1,014-1,025 MiB | 564 / 618 MiB | **317 MiB** | **966.0 MiB** |
| int8 g512 / per-row int8 table | 157.2 s (156.1, 158.3) | 169.9 s (167.1, 172.7) | 182.2 s | 1,012-1,046 MiB | 760 MiB | 321 MiB | ~474 MiB |
| int8 g128 / per-row int8 table | 201.9 s | 216.3 s | 266.7 s | 1,054.5 MiB | 446 MiB | 332 MiB | ~474 MiB |
| fp16 / f16 table (`--embed-f16`) | 147.4 s | 164.8 s | 160.2 s | 1,523.6 MiB | 783 MiB | 559 MiB | 966.0 MiB |
| CPU f32 / per-row int8 table (`--device cpu`) | 265.6 s | 270.1 s | 426.0 s | 1,548.6 MiB | 1,300 MiB | 1,265 MiB | — |

The `int8 table` rows are M2's **per-row** scheme, the default when this window
ran; block-32 scales (M2b) are the current default and their row is in the M2b
subsection below.

The fp16 and int8-g512 rows are medians with both samples in parentheses; the
g128, `--embed-f16` and CPU rows are one sample each. `weights BO` is the
`npu scratch weights_mib` value from `serve`; the prepare/ready line's own
figure is 948.0 MiB for fp16 (the pack's payload, 1.9 % below the BO) and the
int8 context's BOs are **not** counted by `mem_stats`, so its 474.0 MiB is the
ready-line number only (fp16's own ratio suggests ~483 MiB of actual BO).

Memory headline: text-only resident `RssAnon` **943.6 -> 317 MiB** (−66 %) while
the weights BO only moved 1,006.7 -> 966.0 MiB (f16 table -> int8 table and
bf16 body do not touch the f16 pack). The NPU-side total is therefore
**1.25 GiB against 1.90 GiB** on the per-row table, and **1.28 GiB** with the
block-32 scales that M2b made the default (the M2b row: 347 MiB anon, same
966.0 MiB BO, 1,313 MiB together); the peak (prefill transient, not weights)
drops 2,505 -> 1,014-1,025 MiB. Image (1,029 tokens) and video (1,720 tokens,
2 questions) at the same config:

| input | arm | compute | wall | user CPU | peak anon | anon after |
|---|---|---|---|---|---|---|
| image | NPU fp16 | 58.2 s | 69.7 s | 100.2 s | 1,057.8 MiB | 749 MiB |
| image | NPU int8 g512 | 62.5 s | 77.4 s | 105.4 s | 1,061.5 MiB | 810 MiB |
| image | NPU fp16, f16 table | 63.3 s | 77.1 s | 99.8 s | 1,539.1 MiB | 992 MiB |
| image | CPU f32 | 79.9 s | 86.7 s | 147.0 s | 1,739.1 MiB | 1,703 MiB |
| video | NPU fp16 | 54.0 s | 61.7 s | 74.2 s | 1,029.0 MiB | 737 MiB |
| video | NPU int8 g512 | 59.4 s | 73.8 s | 80.8 s | 1,059.8 MiB | 551 MiB |
| video | CPU f32 | 91.7 s | 96.8 s | 146.9 s | 1,741.8 MiB | 1,660 MiB |

Same-window NPU gain over CPU: text 1.7x wall and 2.8x less user CPU, image
1.24x / 1.5x, video 1.57x / 2.0x (the 2026-11-01 window had 1.7x / 3.1x,
1.2x / 1.3x, 1.5x / 1.9x, so the direction and rough size reproduce; the walls
here are 25-35 % higher for every arm, contended). Peak anon is 1.0-1.1 GiB on
every NPU input class against 1.55-1.74 GiB CPU.

**Tower deferral** (serve, `--defer-towers`): after load `RssAnon` 317.2 MiB,
`decision: vision tower deferred to the first media request`; the first media
request prints `decision: loaded vision tower in 1.5s (12 blocks, bf16 weights)`
and takes the process to 573.4 MiB. The clean attribution is the eager arm:
`serve` without `--defer-towers` loads at **512.8 MiB**, i.e. the tower costs
**+195.6 MiB** resident (not +384; the bf16 storage halves it), and the tower
load is 1.0-1.7 s. `embed I8, text body bf16` is in the ready line of every
run, e.g.
`decision model ready in 11.7s (24 layers, embed I8, text body bf16, 96 resident NPU weights (fp16, 948.0 MiB))`.

Serve scratch after a request (`ROCKET_MEM_LOG=1`): weights BO 966.0, matmul
BO 107.6 MiB + 0 host right after load, trimmed to **25.7 BO + 7.6 host** after
the 128-token request (`ROCKET_KEEP_MB=64`), `RssAnon` 317.2 -> 354.1 MiB; the
1,029-token image request leaves 50.3 BO + 10.9 host and 573.4 MiB. Attention
BO/host stays **0** (d3 has no NPU attention stage). The 128-token request
computed in 2.54-2.77 s (46-50 tok/s) with the model resident; int8 g512
measured 2.69 s on the same request but its scratch line reports
`weights_mib=0.0 matmul_bo_mib=0.0` (the int8 context's BOs are outside
`mem_stats`), with `RssAnon` 321.0 -> 358.4 -> 643.8 MiB.

**int8 NPU weights are rejected for d3.** Halving the weights BO (474 vs 948
MiB) costs +6.5 % wall and +18 % user CPU at g512, does not move the resident
or peak anon, and leaves ~+70 MiB more anon after a media request. The parity
is the blocker:

| input | arm | top-1 | max \|Δp\| vs f32 ref | same-board CPU-vs-NPU |
|---|---|---|---|---|
| text | CPU f32 | 5/5 | 3.77e-03 | — |
| text | NPU fp16 (default) | 5/5 | 4.15e-03 | 3.85e-04 |
| text | NPU fp16, `--embed-f16` | 5/5 | **3.54e-04** | 4.06e-03 (table) |
| text | NPU int8 g512 | 5/5 | 3.39e-02 | 3.77e-02 |
| text | NPU int8 g128 | 5/5 | 4.52e-02 | 4.63e-02 |
| image | CPU f32 | 1/1 | 8.36e-03 | — |
| image | NPU fp16, per-row table (M6) | 1/1 | 1.03e-02 | 1.95e-03 |
| image | NPU fp16, block-32 table (M2b) | 1/1 | **1.664e-03** | — |
| image | NPU fp16, `--embed-f16` | 1/1 | **5.24e-04** | — |
| image | NPU int8 g512 | 1/1 | 4.41e-02 | 3.57e-02 |
| video | CPU f32 | 2/2 | 1.51e-02 (decoder) | — |
| video | NPU fp16 (default) | 2/2 | 1.55e-02 | 4.16e-04 |
| video | NPU int8 g512 | 2/2 | 4.95e-02 | 3.44e-02 |

Top-1 is 100 % everywhere; the gate that fails is the probability one. At
g512 the weights push text to 3.4e-2 and image to 4.4e-2, both far over the
M2 deployment bar (**max |Δp| ≤ 1e-2**), so the default stays fp16. g128 was
measured because g512 failed: it is **both slower and less accurate** here
(4.5e-2, 216 s wall). That ordering is the opposite of the emb2 finding
(where finer groups are monotone better) and was not chased — the config is
rejected either way and the arm is one sample. The two models differ in weight
provenance (bf16-stored projections here against emb2's Q8_0 GGUF), which may
interact with the group size; nothing was changed on that basis.

**The int8 table, not the NPU pack, is the dominant error term.** With the
same int8 table on the CPU these five cases read 3.77e-03 (text; 6.96e-3 over
the full 67-case text set in `### Token table`) and 8.36e-03 (image, the same
case as there), i.e. the NPU's own contribution is 3.9e-04 / 1.95e-03 — the
same as the 2026-11-01 window (3.4e-04 / 1.63e-03) and independent of the
table. So the per-row default *image* arm landed at 1.03e-02, marginally over
the 1e-2 bar, while `--embed-f16` returned 3.54e-04 / 5.24e-04 for **+211 MiB
resident** (559 vs 347 with the block-32 table) and +~0.5 GiB of load peak
(1,524 vs 1,024). **Block-32 scales (phase M2b) fixed the default**: the same
board image case now reads **1.664e-3** with top-1 unchanged, inside the bar,
at +30 MiB of resident scales. Text at the default was already fine (4.15e-03
per-row; the dev-host block-32 arm reads 1.465e-3 on its 6-case subset). Video
misses only through the board's `ffmpeg` decode (1.51e-2 on the CPU arm too);
the same-board NPU delta is 4.2e-4.

Reference note: the 2026-11-01 `ref_*_lite_f32.json` files and the modelling
venv were gone from `/tmp`, so the reference was regenerated for exactly these
five fixtures with `tools/decision_reference.py --backbone-dtype f32` in a
reproduced venv (`torch 2.15.0+cpu`, `transformers 5.17.0`, `torchvision
0.29.2`, `opencv-python-headless` 5.0.0; 473 tensors, 25 s for the five cases).
It is the same reference: the board CPU f32 arm reproduces the historical
1.409e-06 (text), 7.949e-06 (image) and 1.399e-02 (video, the decoder drift)
exactly. The current `main` build (`--features npu`) replaced
`/root/rocket-inference-decision/rocket-inference` (the 2026-11-01 binary is kept
as `rocket-inference.2026-11-01`); run outputs and metrics are in
`/root/rocket-inference-decision/runs6/` (`*.json` = wall/RSS samples,
`*-out.json` = answers, `serve-*.log` = the scratch lines).

#### Token table block scales (phase M2b, the current default)

The block-32 table was measured with the same protocol (one image case beside the
live service, `taskset -c 4-7`, `--device npu --defer-towers`, load ~15). Board:
`image-receipt-total`, 1,029 tokens, one sample (output
`runs6/m2b-image-npu-i8block-out.json`, metrics `runs6/m2b-image-npu-i8block.json`):

| arm | top-1 | max \|Δp\| vs f32 ref |
|---|---|---|
| NPU fp16, per-row table (M6) | 1/1 | 1.030e-2 |
| NPU fp16, block-32 table (M2b) | 1/1 | **1.664e-3** |
| NPU fp16, f16 table (M6) | 1/1 | 5.238e-4 |

The answer stays `100_to_150` (0.5708 ref, 0.5725 ours) and the arm is now 6x
inside the bar instead of 3 % over it. Cost: resident anon after load **317 ->
347 MiB** (the 30.3 MiB scale table), peak anon 1,027.4 MiB against 1,057.8 for
the per-row arm in the M6 window, and no wall change outside the board's noise
(compute 62.9 s / wall 80.2 s / user 103.7 s, contended; M6's per-row arm read
58.2 s compute under a lighter load). The `--embed-f16` escape hatch is now
+211 MiB rather than +242. Dev-host text-only resident 1,273 -> 1,302 MiB; the
load peak stays near the M5b number (1,330 -> 1,382 MiB sampled at 20 ms, which
includes the scales).

