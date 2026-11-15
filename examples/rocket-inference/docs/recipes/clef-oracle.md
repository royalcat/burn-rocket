# Clef-Omni layer-streaming oracle (phase 1)

> **STATUS: VERIFIED 2026-11-14** (transformers 5.23.0, torch 2.15.0+cpu, 8 threads, dev host
> 62 GB RAM). Checkpoint = the newest `Cloudflare/clef-omni` snapshot in the shared HF cache
> (`$HF_HOME`=/mnt/BtrTrashCan/huggingface; `$CLEF_MODEL_DIR`, 66 GiB / 70.74 GB, 15 shards, of which
> thinker needs 1-13), venv at `$CLEF_VENV` (1.2 GB).

`tools/clef_reference.py` computes Cloudflare/clef-omni (Qwen3-Omni-30B-A3B thinker
text decoder + `JointSchemaHead`) for text-only SystemOne records on a CPU host by
streaming ONE decoder layer at a time in float32. Peak weights in memory: one layer
(~2.4 GB f32 tensors + ~1.2 GB for the fused `gate_up_proj` copy) plus the activations.

## Setup

```sh
# venv with CPU torch + transformers >= 5 (Qwen3OmniMoe), safetensors, huggingface_hub, numpy
export CLEF_VENV=/mnt/BtrTrashCan/venvs/clef      # or any venv path
python3 -m venv $CLEF_VENV
$CLEF_VENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
$CLEF_VENV/bin/pip install 'transformers>=5.10' safetensors huggingface_hub numpy
# checkpoint: the shared HF cache (never copy it elsewhere)
export HF_HOME=/mnt/BtrTrashCan/huggingface
hf download Cloudflare/clef-omni --exclude 'model-00014-of-00015.safetensors' \
    --exclude 'model-00015-of-00015.safetensors'   # shards 14-15 are talker/code2wav only
export CLEF_MODEL_DIR=$(ls -d $HF_HOME/hub/models--Cloudflare--clef-omni/snapshots/*/ | tail -1)
```

`tools/clef_joint_schema_model.py` is a verbatim copy of the release's
`joint_schema_model.py` (fetched 2026-11-14, 719 lines, read before use: pure model
code; its only I/O is optional media decoding/URL fetch, which this oracle never reaches).
The download dir is untrusted: the script is run with `python -I` and imports the local
copy by explicit file path; do not use the model dir as CWD.

## Run

```sh
cd /home/royalcat/projects/burn-rocket
$CLEF_VENV/bin/python -I examples/rocket-inference/tools/clef_reference.py \
    --model-dir $CLEF_MODEL_DIR \
    --requests examples/rocket-inference/tools/clef_fixtures.json \
    --out-dir /tmp/clef-dump --only checkout-errors,invoice --selfcheck
```

Flags: `--dump-layers all|0,1,23,47` (final layer always dumped; default `all`),
`--max-tokens 512` (guard; the `long-incident` fixture is 800 tokens and needs `--max-tokens 1024`),
`--threads 8` (capped at 8), `--selfcheck`, `--only ids`.

Measured cost (warm page cache after the first run; the download itself took ~28 min at
~40 MB/s with 2 workers): every record streams the whole 70 GB checkpoint once, **~1.1-1.7 s
per layer (load ~1.2 s, compute 0.2-0.5 s at 256-800 tokens)**, i.e. **~60-73 s per record**
(checkout 295 tokens: 65.6-73.1 s; invoice 256: 58.0 s; long-incident 800: 59.8 s).
Peak anonymous RAM (`RssAnon`, sampled after each layer) **1.35 GiB (295 tokens), 2.5 GiB
(800 tokens)** with the per-layer mmap handles dropped (`Shards.close`). Note: `ru_maxrss`
of an earlier version reached 43 GB because touched mmap'd file pages count toward RSS; that
is reclaimable page cache, not allocation. Dumps: 119 MB (295 tokens, all 48 layers),
15 MB (256, layers 0,1,23,47), 47 MB (800, layers 0,1,23,47). Records are processed
sequentially, each re-streaming the weights.

## What it does

1. `encode_record` (release code) -> token ids + question/option spans.
2. Embedding rows read via `safe_open().get_slice` (only the needed token rows).
3. For each of 48 layers: tensors read via `model.safetensors.index.json`, experts fused as
   HF stores them (`gate_up_proj[e] = cat(gate_proj, up_proj)` on the output dim,
   `down_proj[e]`), a real `Qwen3OmniMoeThinkerTextDecoderLayer` is built on the meta
   device and filled with `load_state_dict(assign=True)`, run in f32 with an additive
   causal mask and HF's own rotary module (text-only: all three mRoPE streams = position).
   A forward hook on the router recomputes softmax/top-8/renorm independently and asserts
   it matches the layer's routing.
4. Final RMSNorm (`thinker.model.norm.weight`), then `JointSchemaHead` unchanged in f32
   with `output_embedding_weight` = `LazyRows` over `thinker.lm_head.weight` (reads only
   the token ids used by option spans).
5. `--selfcheck`: layers 0 and 47 are also computed by `manual_layer` (plain torch, no HF
   code, documented math in its docstring: RMSNorm, per-head QK-norm, rotate-half RoPE for
   equal positions, GQA 32/4, causal softmax attention, router softmax/top-8/renorm,
   SwiGLU experts). Reported in the summary JSON under `selfcheck`.
   Measured (checkout-errors, 295 tokens, f32 HF layer vs plain-torch layer, same input):

   | layer | max abs diff | diff / max abs output | router ids equal | router weight max abs diff |
   |---|---|---|---|---|
   | 0 | 2.4e-7 | 1.8e-7 | yes | 2.4e-7 |
   | 47 | 6.1e-5 | 1.7e-7 | yes | 2.4e-7 |

   (layer 47 outputs reach ~350, so 6e-5 is f32 rounding, ~1.7e-7 relative). No divergence.

## Dump format (`<out-dir>/<record id>.npz` + `<id>.json`)

All float arrays are float32, little-endian, `np.load`-able (`np.savez`, uncompressed).
`T` = token count, `H` = 2048, `L` = layer index 0..47.

| key | shape / dtype | meaning |
|---|---|---|
| `input_ids` | `[T]` int64 | token ids from `encode_record` |
| `embeddings` | `[T,H]` | `thinker.model.embed_tokens` rows (before layer 0) |
| `hidden_L{L}` | `[T,H]` | output of decoder layer L (for L in `dump_layers`; layer 47 always) |
| `router_ids_L{L}` | `[T,8]` int32 | top-8 expert ids per token, descending router prob (ties as `torch.topk`) |
| `router_w_L{L}` | `[T,8]` | softmax-over-128 then top-8 renormalised weights (sum to 1) |
| `final_hidden` | `[T,H]` | = layer 47 output (pre final norm) |
| `final_normed` | `[T,H]` | after final `RMSNorm(thinker.model.norm.weight, eps 1e-6)`; the head input |
| `logits_{qid}` | `[n_opt]` | head logits for question `qid`, in `option_ids` order |
| `probs_{qid}` | `[n_opt]` | softmax of the logits |

Router dumps exist for all 48 layers regardless of `--dump-layers`.

`<id>.json`: `answers` (SystemOne answer format via `systemone_answer`), `usage`,
`questions` (`id`, `type` 0 noul/1 choice/2 score, `question_span`, `option_spans`
as token index half-open ranges, `option_ids`), `dump_layers`, `selfcheck`, `wall_s`,
`maxrss_mib`.

## Model math notes for the Rust port

See `manual_layer` in the script. Checkpoint layout facts: experts are stored
per-expert (`thinker.model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight`), the
router is `mlp.gate.weight [128,2048]`, no shared expert, no biases; `thinker.lm_head`
is NOT used for logits (ClefModel replaces it with Identity): its rows serve only as the
lexical option embeddings (mean over each option span's token ids).
Head: `joint_head_config.json` = hidden 2048, width 1024, routing_layers 2, layers 4,
heads 16, feedforward 4096.

## Fixtures

`tools/clef_fixtures.json`: 8 text-only records (`checkout-errors` and `invoice` are the
model-card examples; choice, score, noul, mixed, JSON state, and a ~400-token state).
Model card values for `checkout-errors` (int8 build): technical ~0.83, urgency score
~1.88, outage ~0.95. **Oracle (f32 streaming of the BF16 weights)**: technical **0.8327**
(billing 0.1673), urgency score **1.8917** (P0/1/2 = 0.0425/0.0234/0.9341), outage noul
**0.9552**: matches the card within its rounding. Two further checks:
`invoice` -> status overdue 0.9773 (paid 0.0141, draft 0.0085), large 0.9860;
`long-incident` (800 tokens) -> owner payments 0.8202, severity score 2.2751, double_charge
0.9674, owners_assigned 0.0169 (all plausible for the text).
Determinism: two independent runs of `checkout-errors` give byte-identical arrays for all
110 keys of the shared dump (the second run used the handle-dropping code and a subset of
`--dump-layers`; compared on the keys both contain).

# Media records (phase 8a, 2026-11-14)

> **STATUS: VERIFIED 2026-11-14** on the same host/venv (transformers 5.23.0, torch 2.15.0+cpu), now also with
> `pillow 13.0.0`, `av 19.0.2`, `torchvision 0.29.2+cpu` (`pip install pillow av torchvision`; the release's
> `encode_record` needs PIL for images and PyAV for audio/video, the HF video processor needs torchvision).
> `tools/clef_reference.py` runs records with `images`, `audio` and `videos` end to end in f32; the media code
> is `tools/clef_media.py`; the nine fixtures are `tools/clef_fixtures_media.json` (assets in `data/`).

```sh
cd /home/royalcat/projects/burn-rocket
$CLEF_VENV/bin/python -I examples/rocket-inference/tools/clef_reference.py \
    --model-dir $CLEF_MODEL_DIR \
    --requests examples/rocket-inference/tools/clef_fixtures_media.json \
    --out-dir /tmp/opencode/clef/dump --only media-image-small --selfcheck --max-tokens 4096
```

New flags: `--media-base` (base dir for relative media paths, default `examples/rocket-inference`, so the
fixtures say `data/cat.jpeg` like the d3 fixtures), `--vision-dump-blocks 0,8,16,24,26` (vision block outputs
to dump, pre-merge), `--audio-dump-layers 0,15,31`, `--selfcheck` now also checks the towers. Use
`--dump-layers 0,1,2,3,23,47` for media records: with all 48 layers an npz is 0.35-0.55 GB (`all` is the default).

## 1. How the release puts media in a record (`joint_schema_model.py`, read in full)

Record fields (all optional lists, any combination): `images`, `audio`, `videos`, plus `media_kwargs`
(a dict that is `options.update(...)`d into the processor call kwargs, so it can replace `size` or the whole
`images_kwargs`/`videos_kwargs`; the fixtures do not use it). Every item is **a file path, an `http(s)` URL, a
base64 data URL (`^data:<type>/<sub>;base64,...`, regex `_DATA_URL`) or raw bytes** (`_media_bytes`: data URL ->
base64 decode; `http(s)` -> `urllib.request.urlopen`; anything else -> `Path(text).read_bytes()`).
Python-API-only extras (not reachable through JSON): PIL images, 16 kHz float sample arrays, RGB frame arrays.

* **Image** (`load_image`): PIL `Image.open` of the bytes, `.convert("RGB")` (alpha/palette/CMYK -> RGB).
* **Audio** (`_decode_audio`): PyAV opens the bytes, takes the FIRST audio stream, `AudioResampler(format="flt",
  layout="mono", rate=16000)`, frames concatenated -> f32 `[n]` (no stream or 0 samples -> `ValueError`).
  A file without an audio stream is an error for `audio`.
* **Video** (`load_video`): PyAV, first video stream, `thread_type="AUTO"`. Frame sampling is **a time rule, not
  an index rule**: iterate decoded frames; **skip a frame when `frame.time is None or frame.time + 1e-6 <
  len(kept)/2.0`**, i.e. keep the first frame at/after each 0.5 s multiple (VIDEO_FPS = 2.0). Each kept frame is
  converted with `frame.to_ndarray(format="rgb24", width, height)` where, with `scale = min(1, sqrt(262144 /
  (w*h)))` (`MAX_VIDEO_FRAME_PIXELS = 256*32*32`), `width = max(2, round(w*scale/2)*2)` and likewise height
  (Python `round`, ties to even; swscale does the YUV->RGB and any scaling). No frames -> `ValueError`; an
  odd number of frames gets the last frame duplicated (`frames.append(frames[-1])`). There is **no cap on the
  number of frames**. The soundtrack is `_decode_audio` of the same bytes (None when the file has no audio
  stream or no samples).
* **Audio of videos** (`_encode_media`): `use_audio_in_video = bool(videos) and all(track is not None for
  track in tracks)` - all-or-nothing over the record's videos. When true the tracks are appended to the audio
  list (explicit `audio` clips first, then one track per video, in video order) and passed with
  `videos_kwargs.use_audio_in_video=True`. When any video is silent NO track is used (a video with sound next to
  a silent one is heard as silent).
* **Text handed to the processor** (`add_special_tokens=False`):
  `IMAGE_PLACEHOLDER * n_images + AUDIO_PLACEHOLDER * n_audio + VIDEO_PLACEHOLDER * n_videos` where
  `n_audio` counts the explicit clips only and the placeholders are
  `"<|vision_start|><|image_pad|><|vision_end|>\n"`, `"<|audio_start|><|audio_pad|><|audio_end|>\n"` and
  `"<|vision_start|><|video_pad|><|vision_end|>\n"`. **The order in the sequence is always images, audio, videos,
  whatever the field order in the JSON.** The processor expands each `*_pad`:
  * image: `<|image_pad|>` x (`t*h*w/4` = grid product / merge^2); video: `<|video_pad|>` x (`t*h*w/4`);
    audio: `<|audio_pad|>` x `audio_out_len(feature_frames)` (below).
  * video WITH its track (audio-in-video): the single `<|video_pad|>` becomes `<|audio_start|>` +
    interleaved `<|video_pad|>`/`<|audio_pad|>` + `<|audio_end|>`, giving
    `<|vision_start|><|audio_start|> ... <|audio_end|><|vision_end|>\n`. Rule (processor
    `replace_video_with_audio_token`): video token k (raster over `t` groups, `h/2 x w/2` merged tokens each) has
    time `group_index * second_per_grid * 13`; audio token j has time `j`; walk both lists, emit a video token
    when `video_time <= audio_time` else an audio token, then the remainder of whichever list is left.
    `second_per_grid = temporal_patch_size / fps = 2 / 2.0 = 1.0` (a constant, not the clip duration), so group g
    sits at time `13*g`; each second of audio is 13 tokens (one 100-frame chunk), so each video group is followed
    by 13 audio tokens (measured: `V*140 A*13 V*140 A*13` for a 2 s, 2-group clip).
* **Record layout**: `prefix_ids` = tokenize(`<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n`)
  (35 tokens) **+ the media block**; then the state ids, the schema ids, the suffix. The media block is
  `media_ids` from the processor (one tokenizer pass over the expanded text, so the `\n` after each placeholder
  is a separate token). Limits: `DEFAULT_MAX_LENGTH = 64000` tokens counting media; the media block is part of the
  *fixed* length (`prefix + schema + suffix`), only the state is truncated (to `max_length - fixed_length`, after
  `max_state_tokens`), and `fixed_length > max_length` raises `ValueError`.
* **Preprocessing sizes** (effective on transformers 5.23.0; the release was tested on 5.10.2 and its processor
  defaults were not re-checked there):
  * images: `size = {"shortest_edge": 65536, "longest_edge": 1048576}` = **min/max PIXELS** (`IMAGE_SIZE`, passed as
    `size=` for image/audio-only records, as `images_kwargs.size` when videos are present); smart_resize with
    factor 32 (= patch 16 x merge 2): `h_bar = round(h/32)*32`, `w_bar` likewise; if `h_bar*w_bar > max`:
    `beta = sqrt(h*w/max)`, `h_bar = max(32, floor(h/beta/32)*32)`; elif `< min`: `beta = sqrt(min/(h*w))`,
    `h_bar = ceil(h*beta/32)*32`; aspect ratio > 200 -> `ValueError`. 96x64 -> 224x320 (grid 14x20, 70 tokens);
    960x686 -> 672x960 (grid 42x60, 630 tokens).
  * videos: **no `size` is passed**, so the Qwen3OmniMoeProcessor class default applies (NOT the
    `video_processor.size` in `processor_config.json`, which says 3136/12845056 and is ignored):
    `{"shortest_edge": 128*32*32 = 131072, "longest_edge": 768*32*32 = 786432}` pixels, same smart_resize, factor 32.
    So small clips are UPSCALED: 192x144 -> 320x448 (grid 20x28), 240x180 -> 320x448, 480x360 -> 352x480
    (grid 22x30). Both processors warn that `cap_pixels_per_frame` is unset (no total-video budget; irrelevant here
    because the release already caps frames at 262144 px).
  * pixels: resize = `torchvision.transforms.v2.functional.resize(uint8 [T,3,H,W], [h_bar,w_bar],
    interpolation=BICUBIC, antialias=True)` (**reproduced to 6e-8 max abs** on 4 media: image, cat, two videos, by a
    standalone script: resize on the uint8 tensor, `/255`, `(x-0.5)/0.5` in f32); then patchify below.
  * audio: `WhisperFeatureExtractor` (`feature_size 128`, `n_fft 400`, `hop 160`, 16 kHz, `padding=True`
    (pad to the LONGEST clip of the record with zeros), `truncation=False`, `return_attention_mask=True`; clips
    longer than 30 s are NOT truncated). Computation (reproduced to 1.2e-7): periodic Hann(400), torch `stft`
    (`center=True`, reflect pad), `power = |X|^2` over frames `[:-1]` (frames = samples/160), Slaney mel filter bank
    (128 filters, 0-8000 Hz, Slaney norm, `fe.mel_filters`), `log10(clamp(mel, 1e-10))`, `max(x, max(x) - 8)`
    (max per clip), `(x + 4)/4`. **Pitfall: the features of a clip depend on the other clips of the record**: a short
    clip is zero-padded to the longest clip's length BEFORE the STFT, so its last valid frame sees zeros where an
    unpadded clip would see reflect padding (measured: only frame 199 of a 200-frame clip differs, by up to 0.18).

## 2. Confirmed dimensions (`config.json` `thinker_config`, HF modules)

Text decoder: unchanged (48 layers, hidden 2048, 32/4 heads, head_dim 128, 128 experts top-8,
moe_intermediate 768, theta 1e6, QK-norm). `rope_scaling` = `{interleaved: true, mrope_interleaved: true,
mrope_section: [24,20,20], rope_type: default}`.

| | vision tower (`thinker.visual.*`) | audio tower (`thinker.audio_tower.*`) |
|---|---|---|
| depth / width | 27 blocks, hidden 1152, 16 heads (head_dim 72), MLP 4304 (`gelu_pytorch_tanh`) | 32 layers, d_model 1280, 20 heads (head_dim 64), FFN 5120 (`gelu` erf) |
| input | patches of `3 x 2 x 16 x 16 = 1536` values (`patch_size 16`, `temporal_patch_size 2`) | 128 mel bins x frames (10 ms hop) |
| stem | `patch_embed.proj` Conv3d(3,1152,(2,16,16),stride same, bias) + learned 48x48 table (`num_position_embeddings 2304`, `pos_embed.weight [2304,1152]`) resampled bilinear `align_corners=True` to each grid | 3x Conv2d(k3,s2,p1), 480 channels (`downsample_hidden_size 480`), GELU after each; `conv_out` Linear(480*16=7680 -> 1280, no bias); sinusoid table (not a weight) |
| position | 2-D axial RoPE, theta 10000, over head_dim 72 | none in the layers; sinusoid added once per 13-step chunk |
| spatial merge | `spatial_merge_size 2`: 4 consecutive patches -> 1 token (4608 -> 4608 -> 2048) | none |
| output | `out_hidden_size 2048` | `output_dim 2048` (`ln_post`, `proj1`+GELU, `proj2`) |
| extras | `deepstack_visual_indexes [8,16,24]`: after those blocks (0-based) a *postshuffle* merger (`merger_list.j`: view(-1,4608) -> LN(4608) -> Linear -> GELU -> Linear -> 2048) | `n_window 50` (chunk = 100 mel frames), `n_window_infer 800`, `conv_chunksize 500`, `max_source_positions 1500` |
| norms | LayerNorm eps 1e-6 | LayerNorm eps 1e-5 (torch default) |
| weights | **all 876 tensors in shard 1** (`model-00001-of-00015.safetensors`), BF16; towers have no lm tensors elsewhere | same shard |

Special token ids (`thinker_config`): `<|vision_start|>` 151652, `<|vision_end|>` 151653, `<|image_pad|>` 151655,
`<|video_pad|>` 151656, `<|audio_start|>` 151669, `<|audio_end|>` 151670, `<|audio_pad|>` 151675;
`position_id_per_seconds 13`, `seconds_per_chunk 2` (unused by the thinker positions).

**Patch layout** (`pixel_values`/`pixel_values_videos` rows, 1536 values): for each temporal group `g`, for each
2x2 merge block (raster over `h/2 x w/2`), for each in-block position `(dy, dx)` (row-major): the patch's values in
order `(channel 3, temporal 2, py 16, px 16)`; an image is one frame duplicated to make a 2-frame group, a video
with an odd frame count gets its last frame duplicated (the release already evens it). The tower treats 4
consecutive rows as one merged token; `grid_thw = [t, h, w]` counts patches (not merged tokens): tokens = `t*h*w/4`.

**Audio token count** `audio_out_len(n)` for `n` mel frames: `chunks = n // 100`, tail `l = n % 100`:
`13*chunks + ((((l-1)//2+1)-1)//2+1-1)//2+1` (the tail term is 0 for `l = 0`). 500 frames -> 65, 400 -> 52, 200 -> 26.

## 3. Positions (3-D mRoPE) and DeepStack, as the HF code does them

* `position_ids` = the real `Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index` (called unbound,
  `clef_media.position_ids`), `[3, S]`, float in HF but **integral in every fixture** (asserted into the json).
  `rope_delta` = `max(pos) + 1 - S`.
* **HF takes the 3-D branch only when `image_grid_thw` or `video_grid_thw` exists.** An **audio-only** record (and
  text-only) therefore gets plain `cumsum - 1` positions for all three streams: the audio placeholders are just
  consecutive text positions (`media-audio-5s`: `<as>` 35, pads 36..100, `<ae>` 101, text from 102, `rope_delta` 0).
  Audio next to an image/video goes through the 3-D branch and its audio gets the audio rule below.
* 3-D rule (per segment in sequence order; text = `arange + st`, all three streams): a segment starts at
  `st = max(previous positions) + 1`. `<vs>`/`<as>` markers take one position, the closing markers one position at
  `max(segment) + 1`.
  * image: `t = (g*13) + st` (g = 0 -> constant `st`), `h = st + row`, `w = st + col` over the MERGED grid
    (`h/2 x w/2`, raster); next position = max over all three streams + 1 (e.g. 14x20 grid -> `h` 36..42,
    `w` 36..45, `<ve>` at 46, text resumes at 47).
  * video: same with `t = st + g * second_per_grid * 13` (`13*g` for the fixed 2 fps).
  * audio (standalone): `arange(audio_len) + st` in all streams.
  * audio-in-video (`<vs><as>` adjacent: 2 opening markers, 2 closing): video tokens as above, audio tokens at
    `st + j`, merged into one list by the same `video_t <= audio_t` rule as the processor (so the order in the
    sequence is the order of the positions); closing markers at `max + 1` each (measured: `<ae>` 63, `<ve>` 64).
* Rotary tables: HF `Qwen3OmniMoeThinkerTextRotaryEmbedding` on `[3, 1, S]` positions; the interleaved layout is
  re-derived from first principles in `clef_media.manual_mrope_cos_sin` (inv_freq slot `j` uses stream `h` when `j%3==1 and j<60`,
  `w` when `j%3==2 and j<60`, else `t`; cos/sin = `cat(cos, cos)` over the head_dim=128 rotate-half layout). `manual_layer`
  now takes the `[3,T]` positions; its layer-0/47 selfcheck covers the mRoPE path (rel. diff 5e-8..2.6e-7).
* **Splice**: `inputs_embeds[audio_pad positions] = audio_features` (in record order = explicit clips, then tracks),
  `[image_pad positions] = merged image tokens` (images concatenated in record order), `[video_pad positions] = merged
  video tokens` (videos in order). The counts must match the placeholder counts (asserted).
* **DeepStack**: `visual_pos_idx` = sorted union of the image and video pad positions (audio pads are NOT visual).
  `deepstack_Lj` (j = 0,1,2; `[n_vis, 2048]`, rows aligned with `visual_pos_idx`) are the three merger taps of the
  tower run (image rows from the image run, video rows from the video run). After decoder layer `j` (0-based, j <
  3), `hidden[visual_pos_idx] += deepstack_Lj` - after the layer output, before layer `j+1` (so the dumped
  `hidden_L0..2` INCLUDE the injection; a plain layer output is `hidden_Lj - deepstack_Lj` at the visual rows). No
  scaling. Image and video towers run separately (the same weights; HF runs `get_image_features` and
  `get_video_features` apart).

## 4. Dump keys of media records (`<id>.npz`, additionally to the text keys above)

`T` tokens, `H` 2048, `P` patches (rows of pixel values), `A` audio tokens, `V` visual (image+video) tokens.
`embeddings` is the SPLICED input of layer 0 for media records (`embeddings_text` is the raw token rows).

| key | shape / dtype | meaning |
|---|---|---|
| `position_ids` | `[3,T]` f32 | (t,h,w) mRoPE positions (all records, text too = `arange`) |
| `rope_delta` | `[1]` f32 | `max(pos)+1-T` |
| `embeddings_text` | `[T,H]` | token rows before the splice |
| `placeholder_idx_{image,video,audio}` | `[n]` int64 | sequence positions of each `*_pad` token |
| `visual_pos_idx` | `[V]` int64 | positions that get DeepStack |
| `deepstack_L{0,1,2}` | `[V,H]` | features added after decoder layers 0,1,2 |
| `media_use_audio_in_video` | `[1]` int8 | release `use_audio_in_video` flag |
| `media_image_rgb_{i}` | `[H,W,3]` uint8 | `load_image` output (before resize) |
| `media_video_frames_{i}` | `[F,H,W,3]` uint8 | `load_video` frames (2 fps sampled, even F, <= 262144 px) |
| `media_audio_samples_{i}` | `[n]` f32 | 16 kHz mono clips in the processor's order (explicit, then tracks) |
| `media_pixel_values`, `media_pixel_values_videos` | `[P,1536]` f32 | processor output (patch layout above) |
| `media_image_grid_thw`, `media_video_grid_thw` | `[n,3]` int64 | patch grids `[t,h,w]` |
| `media_video_second_per_grid` | `[n_vid]` f32 | 1.0 |
| `media_input_features` | `[n_audio,128,Fmax]` f32 | log-mel (see padding pitfall) |
| `media_feature_attention_mask` | `[n_audio,Fmax]` int32 | valid frames (sum = feature length) |
| `{image,video}_patch_count` | `[1]` int64 | rows of the pixel values |
| `{image,video}_block_{k}` | `[P,1152]` | vision block k output (pre-merge), k in `--vision-dump-blocks` |
| `{image,video}_tower_last` | `[P,1152]` | after block 26, before the merger |
| `{image,video}_merged` | `[P/4,2048]` | final merger output = the embeddings spliced at the `*_pad` positions |
| `{image,video}_deepstack_{0,1,2}` | `[P/4,2048]` | the three taps (blocks 8/16/24 through `merger_list`), per tower run |
| `audio_conv_out` | `[A,1280]` | after `conv_out` + sinusoid, valid steps only, all clips concatenated |
| `audio_layer_{k}` | `[A,1280]` | audio layer k output (`--audio-dump-layers`) |
| `audio_ln_post` | `[A,1280]` | after `ln_post` |
| `audio_features` | `[A,2048]` | tower output = spliced at the `audio_pad` positions |

The json gains `media`: `use_audio_in_video`, token/patch counts, grids, audio feature lengths, decoded sizes,
`placeholders`, `special_tokens_in_input` (ids 151652/151653/151669/151670 counts), `rope_delta`,
`position_ids_integral`, `position_branch`, `tower_s`, and `tower_selfcheck` (below).

## 5. Selfcheck of the towers

`--selfcheck` runs `manual_vision` / `manual_audio` (plain torch, no HF modelling code, the docstrings are the
port spec) on the same inputs and compares with the HF modules (f32, same weights). Measured
over all nine fixtures (relative = max abs diff / peak of the HF output):

| tower | output | max rel to peak | max abs diff |
|---|---|---|---|
| vision (image 280 / 2520 patches, video 2240 / 2640 patches) | last (pre-merge, peak ~1.8e3) | 1.7e-5 .. 8.6e-5 | 0.02 .. 0.15 |
| vision | merged | 1.0e-5 .. 2.2e-5 | <= 2.1e-4 |
| vision | deepstack 0/1/2 | 2e-6 .. 3.3e-5 | <= 6.9e-5 |
| audio (500/400/200 frames, 1 and 2 clips) | conv_out | 2.0e-7 .. 3.0e-7 | <= 2.3e-5 |
| audio | features | 2.9e-6 .. 1.8e-5 | <= 3.2e-6 |

(Differences are f32 rounding; the vision tower's last hidden state peaks at ~1.2e3-1.8e3.)

## 6. Fixtures (`tools/clef_fixtures_media.json`) and results

Assets (all in `examples/rocket-inference/data/`): `clef-tiny.png` (96x64 synthetic, made with PIL: gradient,
white rectangle outline, red ellipse, diagonal line), `cat.jpeg` (960x686), `speech5s.wav` (5 s, 16 kHz mono),
`test.mp4` (480x360, 8 fps, 4 s, AAC 44.1 kHz mono), `clef-video-silent.mp4` (192x144, 4 s, no audio, ffmpeg
from test.mp4), `clef-video-audio-small.mp4` (240x180, 2 s, AAC 22.05 kHz, ffmpeg from test.mp4).

Runs on the dev host (8 threads, `--selfcheck`, warm page cache; tower runs take 0.4-4.7 s (plus the weight read/build), the thinker 48
layers dominate at ~1.2 s per layer; the wall includes the processor, decode, towers and the selfchecks;
`RssAnon` peak = sampled after each layer/tower, **the first nine runs were one process so later peaks include
earlier towers' retained pages**, see the single-process note below):

| record | media | tokens (media / total) | grids | wall | answers (rounded) |
|---|---|---|---|---|---|
| `media-image-small` | tiny.png | 70 / 319 | img 14x20 | 65.3 s | colorful 0.9034, shape circle 0.9171 |
| `media-image-cat` | cat.jpeg | 630 / 980 | img 42x60 | 82.4 s | animal cat 0.9521, indoors 0.0173, cuteness score 2.266 |
| `media-images-two` | tiny + cat | 700 / 946 | 14x20, 42x60 | 79.5 s | photo second 0.9541, same 0.0086 |
| `media-audio-5s` | speech5s.wav | 65 / 302 | audio 500 fr | 64.3 s | speech 0.8596, kind speech 0.8991 |
| `media-video-silent` | silent 4 s | 560 / 796 | vid 4x20x28 | 75.4 s | motion 0.3818, scene object 0.9849 |
| `media-video-audio-small` | 2 s + track | 280 + 26 / 544 | vid 2x20x28, audio 200 | 71.0 s | sound 0.1777, kind other 0.9463 |
| `media-video-test` | test.mp4 + track | 660 + 52 / 952 | vid 4x22x30, audio 400 | 79.6 s | sound 0.3403, urgency 0.2653 |
| `media-mixed` | image + audio + silent video | 70 + 65 + 560 / 947 | | 79.5 s | consistent 0.0213, priority score 0.1154 |
| `media-mixed-av` | image + audio + video with track | 70 + 91 (65 + 26) + 280 / 678 | | 72.9 s | consistent 0.0243, priority media 0.9417 |

(The questions are generic, the answers only sanity checks: the cat photo is a cat, the audio is speech, the tiny
picture has a red circle. No reference answer exists for these records except the oracle itself.)
Single-record runs (one process each, `RssAnon` after each layer): image-small peaked at 3.0 GiB (tower build
~0.5 GiB transient + 2.5 GiB thinker working set at 319 tokens); the 9-record batch grew to 6.3 GiB because glibc
kept the freed tower pages (a tower build holds ~3-5 GiB transiently: random init + f32 state dict) - since then
`_release()` calls `malloc_trim` after each tower.
**Determinism**: two independent runs of `media-video-audio-small` (the hardest record: audio + video + audio-in-video
interleave) give byte-identical arrays for all 187 keys (`tobytes()` equal; the second run without `--selfcheck`).

## 7. Pitfalls the Rust port must know

1. **Transformers version drift**: the release code targets 5.10.2; 5.23 rejects `use_audio_in_video` in `videos_kwargs`
   for a video whose track is passed as audio ("0 `<|audio_pad|>` tokens and 1 audio inputs"). `CompatProcessor`
   translates to `load_audio_from_video=True` + `audio_from_video_indices = [None]*n_explicit + [0..n_videos)`; the
   token layout it yields is the algorithm described in section 1 (same interleave code as the old release path,
   but the 5.10.2 processor itself was not run).
2. **Audio-only records use plain text positions** (HF's `get_rope_index` else-branch); only image/video records get 3-D
   positions. A record with audio AND an image/video takes the 3-D branch and gives the audio the 3-D audio rule.
3. **Videos are upscaled to >= 131072 pixels** by the video processor (min_pixels, factor 32) although the release
   caps them at 262144; images use min 65536 / max 1048576 pixels. Different tables, easy to mix up.
4. **Log-mel depends on batch padding** (zeros up to the longest clip before the STFT, last valid frame of short clips differs).
5. **Per-frame vision attention**: attention never crosses frames of a video (segments of `h*w` patches, `t` of them);
   images are one segment. Rotary row/col come from the frame position, not the time. The learned position table is
   resampled with `align_corners=True` (corner pixels map to table corners, `src = i*(47)/max(size-1,1)`) and reordered
   into the merge-block order of the patches.
6. **Two merger types**: the final merger LayerNorms each 1152-wide patch, THEN groups 4; the DeepStack mergers group 4 first and
   LayerNorm the 4608-wide row (`use_postshuffle_norm`). DeepStack taps are the block OUTPUTS of 8, 16, 24 (0-based).
7. **Audio windows**: the encoder attends inside windows of `max_chunk_out * (800/100)` = 104 tokens (the full-chunk
   length of the batch times 8; a batch of only short clips uses its longest chunk), per clip; chunks (100 frames) are
   convolved independently with a zero-padded tail, and the sinusoid position restarts at 0 for each chunk.
8. **Decode parity with ffmpeg**: on `test.mp4` the ffmpeg CLI (`select=not(mod(n\,4))`, `-pix_fmt rgb24`, unscaled)
   gives frames BIT-IDENTICAL to PyAV's, and `-vn -ac 1 -ar 16000 -f f32le` gives BIT-IDENTICAL samples to the release's
   `_decode_audio` (AAC 44.1 kHz -> 16 kHz). Not measured: swscale-scaled frames (clips above 262144 px), other codecs.
   The oracle dumps decoded frames and samples (`media_video_frames_*`, `media_audio_samples_*`) so decode and
   preprocessing can be gated separately.
9. The oracle towers are f32 from BF16 weights (exact widening); the release runs them in bf16, so answers differ at the
   bf16 level, as for the text path.
10. Dump sizes: use `--dump-layers 0,1,2,3,23,47` (all 48 layers is 0.35-0.55 GB per media record).
