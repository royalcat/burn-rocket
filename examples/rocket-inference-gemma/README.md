# rocket-inference-gemma

EmbeddingGemma 2 inference in Burn (`flex` backend), inference only.

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) is a
multimodal embedding model built from the Gemma 4 family: a 270M-parameter
bidirectional text decoder, a 170M vision tower and a 300M audio tower, all
projected into one 768-d embedding space (MRL: 128/256/512/768), with
per-layer embeddings (PLE) and an 8192-token context.

This example implements the full checkpoint — text, image, video and audio —
and reproduces the HF reference numerics (see `docs/experiment-log.md`).

## Layout

| path | contents |
|---|---|
| `src/config.rs` | `config.json` schemas (text / vision / audio sub-configs, per-layer overrides) |
| `src/model.rs` | text backbone (bidirectional Gemma 4 decoder: PLE, QK-RMSNorm, sliding/full layer types, chunked attention), `Emb2Model` assembly |
| `src/vision.rs` | Gemma 4 vision tower (patch embedder, axial 2-D RoPE, encoder, pooling, `embed_vision`) |
| `src/audio.rs` | Gemma 4 audio tower (subsample conv, chunked relative-attention conformer, clipped linears, `embed_audio`) |
| `src/audio_frontend.rs` | USM log-mel frontend (rustfft + HTK mel), waveform loading via hound / ffmpeg |
| `src/media.rs` | image decode + torchvision-exact bicubic resize + patchify; video sampling/extraction (ffprobe/ffmpeg) |
| `src/inputs.rs` | shared placeholder expansion (`<\|image\|>` / `<\|video\|>` / `<\|audio\|>`), media encoding, token assembly |
| `src/layers.rs` | `ClippableLinear` (audio/vision linears with checkpoint clip bounds), low-RAM `lin()` helper |
| `src/server.rs` | OpenAI-compatible `/v1/embeddings` + native multimodal `/embed` |
| `src/main.rs` | CLI (`embed`, `bench`, `tokenize`, `serve`) and model loading |
| `tools/ref_embeddinggemma2.py` | HF reference: token ids, text/image/video/audio embeddings, debug dumps |
| `tools/debug_audio_hf.py` | HF audio-tower intermediate dumps (debugging) |
| `docs/experiment-log.md` | measurements and verification results |

## Build

```sh
# dev host (flex backend; `--no-default-features` skips the CubeCL LLVM backend)
cargo build --release -p rocket-inference-gemma --no-default-features
# default build (adds the CubeCL/LLVM `cpu` backend)
cargo build --release -p rocket-inference-gemma
# aarch64 (board), NPU feature links librocketnpu
cargo build --release -p rocket-inference-gemma --target aarch64-unknown-linux-gnu \
    --no-default-features
```

## Model files

```sh
mkdir -p ~/models/embeddinggemma-2 && cd ~/models/embeddinggemma-2
base=https://huggingface.co/google/embeddinggemma-2/resolve/main
for f in config.json model.safetensors tokenizer.json tokenizer.model tokenizer_config.json \
         preprocessor_config.json processor_config.json chat_template.jinja \
         config_sentence_transformers.json sentence_bert_config.json modules.json; do
  curl -sLO "$base/$f"
done
mkdir -p 1_Pooling 2_Normalize
curl -sL -o 1_Pooling/config.json "$base/1_Pooling/config.json"
curl -sL -o 2_Normalize/config.json "$base/2_Normalize/config.json"
```

## CLI

```sh
B=target/release/rocket-inference-gemma   # or $CARGO_TARGET_DIR/release/...
M=~/models/embeddinggemma-2

# text embedding (768-d, L2-normalized; --dim for MRL truncation, --prompt for task prefixes)
$B embed --model-dir $M --text "what is the capital of france?" --out out.json
$B embed --model-dir $M --prompt query --dim 256 --text-file data/one_long.txt --out out.json

# low-RAM mode: Q8-resident projections, dequantized per call (1216 MiB vs 2845 MiB)
$B embed --model-dir $M --quant q8 --text-file data/one_long.txt --out out.json

# image (the placeholder <|image|> is expanded to BOI + soft tokens + EOI)
$B embed --model-dir $M --image data/cat.jpeg --out img.json
$B embed --model-dir $M --image data/cat.jpeg --text "task: search result | query: <|image|>" --out img_q.json

# video (1 fps, at most 32 frames, 140 soft tokens per frame)
$B embed --model-dir $M --video data/test.mp4 --out vid.json

# audio (16 kHz mono WAV via hound; anything else through ffmpeg)
$B embed --model-dir $M --audio data/speech5s.wav --out audio.json
$B embed --model-dir $M --audio data/speech5s.wav --text "task: sentence similarity | query: <|audio|>" --out a.json

# timings and stage breakdown
$B bench --model-dir $M --text-file data/one_long.txt --reps 2

# token ids (compare with the HF tokenizer)
$B tokenize --model-dir $M --text "hello"

# server
$B serve --model-dir $M --port 8390
```

Task prompts (`query`, `document`, `STS`, `classification`, `clustering`,
`code`, ...) are read from `config_sentence_transformers.json`; omit `--prompt`
for no prefix. `--prompt none` is also accepted.

### Server

```sh
curl -s localhost:8390/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":"what is the capital of france?"}'
curl -s localhost:8390/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["a","b"],"dim":256,"prompt":"query"}'
# native multimodal endpoint: paths, or data: URIs with base64 payloads
curl -s localhost:8390/embed -H 'Content-Type: application/json' \
  -d '{"image":"/path/to/cat.jpeg"}'
curl -s localhost:8390/embed -H 'Content-Type: application/json' \
  -d '{"audio":"/path/to/speech.wav","text":"task: sentence similarity | query: <|audio|>"}'
curl -s localhost:8390/health; curl -s localhost:8390/v1/models
```

`/v1/embeddings` accepts a string or an array of strings; `/embed` accepts
`text`/`image`/`video`/`audio` (media values are filesystem paths or
`data:<mime>;base64,<payload>` blobs). Every embedding is L2-normalized; `dim`
applies MRL truncation before normalization.

## Verification

`docs/experiment-log.md` has the full numbers. Summary against the HF
reference (transformers 5.19, f32):

| input | cosine |
|---|---|
| text (9 tok, 2594 tok; dims 768/256) | 1.00000000 |
| image, PNG or JPEG | 1.00000000 |
| video (4 frames) | 1.00000000 |
| audio, 16 kHz mono (5 s / 30 s) | 1.00000000 |
| text + image / text + audio | 1.00000000 / 1.00000000 |
| any modality, `--quant q8` | 0.9996-0.9999 |
| any modality, `--quant q8 --npu` (board) | 0.9992-0.9999 |

Text generation (`gemma-4-E2B-it`, greedy) is **token-identical** to the HF 5.19
reference on: a 16-token question, a system+user prompt, a multi-turn
conversation and a 631-token prompt (16 generated tokens each). The chat
template renders token-exactly for the same cases (system, multi-turn,
`enable_thinking`). Image-conditioned generation (280 soft tokens from
`data/cat.jpeg`) is token-identical to the bf16 reference, and audio-conditioned
generation to the f32 reference (the bf16 reference flips one near-tie:
"vehicle" vs "car"). `serve-chat` accepts the same media as OpenAI content parts
(`image_url` data URIs / `input_audio` base64) and produces the same outputs as
the CLI (image: 284 prompt + 24 completion tokens; audio: 145 + 24).

## Text generation (Gemma 4 E2B-it)

The same crate generates text with `google/gemma-4-E2B-it` (10.25 GB BF16
checkpoint): the causal Gemma 4 decoder with KV sharing (layers 15-34 reuse the
K/V of layer 13/14 and use double-wide MLPs), per-layer embeddings with the
token table, proportional p-RoPE on the full layers and a soft-capped tied LM
head. Output is token-identical to the HF 5.19 reference (see the log).

```sh
# model files (10.25 GB)
mkdir -p ~/models/gemma-4-E2B-it && cd ~/models/gemma-4-E2B-it
base=https://huggingface.co/google/gemma-4-E2B-it/resolve/main
for f in config.json generation_config.json model.safetensors processor_config.json \
         tokenizer.json tokenizer_config.json chat_template.jinja; do curl -sLO "$base/$f"; done

# greedy generation (f32 parity mode)
$B gen --gen-model-dir ~/models/gemma-4-E2B-it --text "What is the capital of France?" \
    --max-new-tokens 64
# chat-style messages and sampling
$B gen --gen-model-dir ~/models/gemma-4-E2B-it \
    --messages '[{"role":"system","content":"You are terse."},{"role":"user","content":"Hi"}]' \
    --sample --temperature 0.7 --top-k 64 --top-p 0.95
# image / audio input (the placeholder expands to the tower's soft tokens)
$B gen --gen-model-dir ~/models/gemma-4-E2B-it --text "What is in this image? <|image|>" \
    --image data/cat.jpeg --max-new-tokens 64
$B gen --gen-model-dir ~/models/gemma-4-E2B-it --text "What do you hear? <|audio|>" \
    --audio data/speech5s.wav --max-new-tokens 64

# NPU prefill (aarch64 `--features npu` build): text projections are packed into
# resident fp16 NPU weights and prefill runs on the NPU (projections + causal /
# causal+window attention); decode keeps the CPU f32 copies, because NPU
# matmuls pad M to 256 and a single decode row would waste that work.
$B gen --gen-model-dir ~/models/gemma-4-E2B-it --text "What is the capital of France?" --npu

# non-streaming OpenAI-compatible server
$B serve-chat --gen-model-dir ~/models/gemma-4-E2B-it --port 8391
curl -s localhost:8391/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64}'
# media parts: `image_url` (data URI or path) and `input_audio` (base64 + format),
# one of each per request; the server inserts the placeholder tokens itself
curl -s localhost:8391/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":[{"type":"text","text":"What is this?"},
         {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]}],"max_tokens":64}'
```

Precision modes (`gen`/`serve-chat`), measured on the dev host (32 threads,
16-token prompt / 9-token answer):

| mode | flags | resident | decode | notes |
|---|---|---|---|---|
| f32 (default) | - | 12494 MiB | 2.8 tok/s | token-identical to HF on every tested prompt |
| f16 | `--f16` | 8923 MiB | 3.5 tok/s | token-identical on short/multi-turn; **can diverge on long prompts** (631-token case flipped a near-tie) |
| Q8_0 | `--quant q8` | 7290 MiB | 0.09 tok/s | memory-only: `lin` dequantizes each weight per call, which dominates decode (flex has no int8 GEMM) |

Prefill is compute-bound (631 tokens: 7.5 s f32, 11.3 s f16); decode is
memory-bandwidth-bound (f32 reads 17 GB of weights per token). The chat server
is non-streaming and takes OpenAI-style `messages` (string or text-part content).

### QAT mobile checkpoint (`*-qat-mobile-transformers`)

The pre-quantized mobile checkpoints
([`google/gemma-4-E2B-it-qat-mobile-transformers`](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers),
2.46 GB) are supported natively — no conversion step. Weights are packed
INT2/INT4 (two/four values per byte) or stored INT8, dequantized with
per-output-channel scales (per-row block scales for the token tables), and the
checkpoint's SRQ (static range quantization) activation rounding is applied
around the affected linears. `--f16` is the useful configuration:

```sh
$B gen --gen-model-dir ~/models/gemma-4-E2B-it-qat-mobile --f16 \
    --text "What is the capital of France?" --max-new-tokens 16
```

The PLE table (2.35B parameters) stays **packed** and its rows are dequantized
on lookup — bit-identical values, 1.13 GiB instead of 8.75 GiB (f32) or 4.4 GiB
(f16):

| mode | resident | short decode | notes |
|---|---|---|---|
| f32 | 12.0 GiB | 3.9 tok/s | parity mode (f32 projections) |
| `--f16` | 6.6 GiB | 4.9 tok/s | 12/12 tokens identical to the f32 reference on the 631-token prompt |
| `--quant q8` | 6.1 GiB | 0.07 tok/s | memory-only |

Verified against HF 5.19: the loaded weights are **bit-exact** (checked for a
4-bit projection, a 2-bit MLP and the 4-bit PLE table), greedy output is
token-identical on the short and system-prompt cases, and the long prompt is
9/12 (f32) or 12/12 (f16) — the first divergence is a top-2 swap with a 0.23
logit margin. SRQ rounds activations to an int8 grid, so a long sequence can
turn the f32 accumulation-order difference between flex and torch into a
one-quantum activation jump; the reference itself flips tokens between bf16 and
f32 on the same prompt.

## NPU offload (aarch64)

The `npu` feature (RK3588, `librocketnpu` from the root `burn-rocket` crate)
offloads the text backbone: all 218 text projections are packed into resident
fp16 NPU weights (0.25 GiB, f32 copies dropped) and attention runs on the NPU
through the windowed attention op (`burn_rocket::attention_window`). The
vision/audio towers, norms, RoPE and the embedding table stay on the CPU.

```sh
# cross-build (links vendor/rocketnpu/librocketnpu.a; run scripts/build-rocketnpu.sh once)
cargo build --release -p rocket-inference-gemma --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu

# on the board: lowest-memory mode = Q8 CPU towers + NPU text backbone
./rocket-inference-gemma bench --model-dir /root/models/embeddinggemma-2 \
    --quant q8 --npu --text-file data/one_long.txt --reps 1
# --npu-attn cpu keeps the CPU attention; --npu-threads N sets the NPU threads
```

Measured on `rock-5b-plus.lan` (4 A76 threads, 2587-token text): `--quant q8`
34.5 s (75 tok/s, 94 s user CPU) -> `--quant q8 --npu` 21.4 s (121 tok/s, 63 s
user CPU): 1.61x faster, 33 % less CPU. Numerics vs the HF f32 reference:
0.9996-0.9999 for text/image/audio, 0.9992 for video. See
`docs/experiment-log.md` §9.

## Known limits

- **Audio resampling**: 16 kHz mono WAV is exact. Other sample rates go through
  ffmpeg's resampler, which differs from librosa/soxr used by the HF reference
  (~0.994 cosine on a 44.1 kHz clip). Resample to 16 kHz mono first for exact
  numbers.
- **Memory**: f32 weights are ~2.9 GB resident (2845 MiB anon), `--quant q8`
  brings that down to 1216 MiB and `--quant q8 --npu` to 1068 MiB; all share a
  ~4.2 GiB *load peak* because the checkpoint is materialized f32 before
  quantization/packing and the 1.49 GB mmap is touched.
- Batch inputs are processed one at a time.
- **NPU prefill is not yet measured on the board**: the implementation is in
  place and cross-builds, and the model is deployed at
  `/root/models/gemma-4-E2B-it/`, but `rock-5b-plus` was holding ~11 GB with
  other workloads (only ~4.5 GB available) and a guarded attempt was OOM-killed
  during load. It needs an idle board (~9.5 GB host + 2.1 GiB NPU).
- **f32 only**: `--dtype f16` loads (1.5 GiB resident) but is numerically broken
  in this architecture — RMSNorm/softmax/PLE need f32 precision (text cosine
  0.984, image 0.70 vs the f32 reference), so the CLI rejects it. Use
  `--quant q8` for the low-RAM mode instead.
