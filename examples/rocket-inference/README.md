# rocket-inference

Burn inference for the RK3588 (Rock 5B+) built on the [`burn-rocket`](../..)
library: one binary, four model families, `flex` CPU backend by default with
optional RK3588 NPU offload (`--npu`).

| family | model | CLI commands | served endpoints |
|---|---|---|---|
| `qwen3` | Qwen3-Embedding-0.6B | `bench`, `embed`, `gemm`, `tokenize` | OpenAI `/v1/embeddings`, `/v1/models`, `/health` |
| `intent` | Qwen3.5-0.8B intent/query-planner (`guoxuter/ov_intent_analysis_sft:v7_q8`) | `gen` | OpenAI `/v1/chat/completions`; Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` |
| `gemma` | EmbeddingGemma 2 (text/image/video/audio) | `embed`, `bench`, `tokenize` | OpenAI `/v1/embeddings` + native multimodal `/embed` |
| `gemma` | Gemma 4 E2B-it chat (text/image/audio) | `gen` | OpenAI `/v1/chat/completions`; Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` |

```
usage: rocket-inference <family> <command> [flags]
```

Serving is one command for every model: `rocket-inference serve --model-dir <dir>`
detects the checkpoint's family and exposes exactly the endpoints it supports
(see [Serving](#serving)). The families are separate model architectures and
share only generic infrastructure; every command's flags are documented below. See
`docs/experiment-log.md` (Qwen3-Embedding + intent) and
`docs/experiment-log-gemma.md` (EmbeddingGemma 2 + Gemma 4) for the full
measurements and verification results.

## Layout

| path | contents |
|---|---|
| `src/main.rs`, `src/cli.rs` | dispatch (`serve` + the four families), shared flag parser, usage |
| `src/server/` | the one server: model detection, capability routing, OpenAI + multimodal + Ollama endpoints |
| `src/util/` | shared infra: backend/device selection, RSS readout, `Proj`/`RopeCache` |
| `src/qwen3_embedding/` | Qwen3-Embedding-0.6B: model, loaders (f32/Q8/NPU), CLI |
| `src/qwen35_intent/` | Qwen3.5-0.8B intent model: model, safetensors loader + NPU pack-and-drop, CLI |
| `src/gemma/` | shared Gemma 4 building blocks: config schemas, low-RAM/NPU linear helpers (`layers`), QAT (`qat`), media decoding, USM audio frontend, vision/audio towers, input assembly |
| `src/gemma/embeddinggemma/` | EmbeddingGemma 2: text backbone + tower assembly, loaders, CLI |
| `src/gemma/gemma4/` | Gemma 4 E2B-it: causal decoder, loader (f32/f16/q8/QAT), chat template + sampling, CLI |
| `src/bin/wgpu_probe.rs` | Vulkan/wgpu GPU probe (`gpu-*` features; see §13 of the Qwen log) |
| `data/` | bench/embedding fixtures, image/audio/video inputs |
| `tools/` | HF reference scripts (`ref_embeddinggemma2.py`, `ref_gemma4.py`, `debug_audio_hf.py`) |

## Build

```sh
# dev host (flex backend; `--no-default-features` skips the CubeCL LLVM backend)
cargo build --release -p rocket-inference --no-default-features
# default build (adds the CubeCL/LLVM `cpu` backend)
cargo build --release -p rocket-inference
# aarch64 cross-build (board); `--features npu` links librocketnpu
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features
cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
    --no-default-features --features npu
```

The aarch64 binaries and models on the board live under `/root/rocket-inference/`
(binary) and `/root/models/`; run from the deploy dir (the qwen bench uses the
relative `data/bench_text.txt`). The NPU build is self-contained
(`librocketnpu` is statically linked) and needs `/dev/accel/accel0` at run time
for `--npu`.

## Serving

One command serves every model:

```sh
rocket-inference serve [--model-dir <dir>] [--family auto|qwen3|embeddinggemma|intent|gemma4] [flags]
```

`--family auto` (the default) reads `config.json` `model_type` and picks the
loader: `qwen3`, `embedding_gemma2`, `qwen3_5` or `gemma4`. Use `--family <name>`
to override (e.g. a plain generative Qwen3 checkpoint).

The routes follow the loaded model — incompatible paths are not registered
(404), and `/v1/models` reports a `capabilities` array:

| | `qwen3` | `embeddinggemma` | `intent` | `gemma4` |
|---|---|---|---|---|
| `/health`, `/v1/models` | ✓ | ✓ | ✓ | ✓ |
| `/v1/embeddings` (string/list; ids/base64 for qwen3; `dim`/`prompt` for Gemma) | ✓ | ✓ | — | — |
| `/embed` (text/image/video/audio, paths or data URIs) | — | ✓ | — | — |
| `/v1/chat/completions` (OpenAI; `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, stream=false) | — | — | ✓ | ✓ |
| `/`, `/api/version`, `/api/tags`, `/api/show`, `/api/chat`, `/api/generate` (Ollama) | — | — | ✓ | ✓ |

```sh
# embedding server (OpenAI): /v1/embeddings
$B serve --model-dir ~/models/qwen3-embedding-0.6b --backend flex --dtype f32 --quant q8 --port 8383 --max-tokens 30000
curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":"Hello world","encoding_format":"float"}'

# multimodal embedding server (OpenAI + native /embed)
$B serve --model-dir ~/models/embeddinggemma-2 --quant q8 --port 8390
curl -s localhost:8390/embed -H 'Content-Type: application/json' -d '{"image":"/path/to/cat.jpeg"}'

# query-planner server (OpenAI chat + Ollama, litellm/OpenViking-compatible)
$B serve --model-dir ~/models/ov-intent-analysis-sft --npu --npu-threads 3 --port 11434 \
    --model-name guoxuter/ov_intent_analysis_sft:v7_q8 --max-tokens 4096 --max-new-tokens 256
curl -s localhost:11434/api/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":false}'
curl -s localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'

# generation server (OpenAI chat + Ollama)
$B serve --model-dir ~/models/gemma-4-E2B-it --port 8391
curl -s localhost:8391/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64}'
```

Serving flags: `--model-dir` (default `~/models/qwen3-embedding-0.6b`),
`--family`, `--backend`, `--port` (default 8383), `--model-name` (default: the
model directory name), `--max-tokens` (prompt/context cap), `--max-new-tokens`
(default generation cap), `--temperature`, plus the loading flags of the
detected family (e.g. `--dtype`/`--quant`/`--npu`/`--npu-attn`/`--chunk`/
`--key-block`/`--attn` for qwen3; `--quant`/`--npu`/`--npu-attn`/`--attn-chunk`/
`--video-fps`/`--video-max-frames` for EmbeddingGemma 2; `--npu`/`--npu-threads`/
`--delta-chunk`/`--embed-f16`/`--pure-npu`/`--npu-decode` for the intent model;
`--f16`/`--quant`/`--npu`/`--attn-chunk` for Gemma 4). A flag that does not
apply to the detected model is rejected.

Behavior notes: `/v1/chat/completions` is non-streaming (streaming is an error,
as before); Ollama `stream: true` returns one NDJSON content line plus the final
`done` line; the Ollama endpoints accept any request `Content-Type`
(application/octet-stream included), while OpenAI chat parts accept
`image_url` / `input_audio` (one of each per request). Requests serialize on the
model; panicking forwards are contained (500) instead of poisoning the server.

Every completed inference request is logged as one `tracing` line (level
`info`, `RUST_LOG` controls it; default `info`). Embedding requests report the
prompt token count, `compute_s`, `tok_s`, the model-lock wait `queue_s` and
`rss_mib` (`RssAnon`); chat requests additionally split the model's own timings
into `prefill_s`/`prefill_tok_s` and `decode_s`/`decode_tok_s`. Failed requests
are not logged (they keep their HTTP error), and `/health`/`/v1/models` never
touch the model:

```text
INFO embedding request endpoint=embeddings model=qwen3-embedding-0.6b inputs=2 tokens=13 queue_s=0.0 compute_s=6.335 tok_s=2.1 rss_mib=968.7
INFO chat request endpoint=chat.completions model=gemma-4-E2B-it-qat-mobile prompt_tokens=15 prefill_s=1.56 prefill_tok_s=9.6 gen_tokens=2 decode_s=0.194 decode_tok_s=10.3 queue_s=0.0 compute_s=1.754 rss_mib=6783.9
```

Set `RUST_LOG=warn` to silence the per-request lines.

## Qwen3-Embedding-0.6B (`qwen3`)

Embedding-only Qwen3 (no `lm_head`): 28 layers, hidden 1024, GQA 16/8 heads,
head_dim 128, QK-RMSNorm, RoPE θ=1e6, SwiGLU, final RMSNorm, last-token
pooling. Reference: production `ik_llama.cpp` Q8_0 (`--pooling last`).

```sh
M=~/models/qwen3-embedding-0.6b
B=target/release/rocket-inference

# single-core benchmark on a fixed text
taskset -c 2 $B qwen3 bench --backend flex --dtype f32 --tokens 3633 --reps 2 --attn blocked --chunk 256 --key-block 256

# embed one text (writes a JSON float array)
$B qwen3 embed --backend flex --dtype f32 --quant q8 --text "Hello world" --out out.json
```

Flags: `--model-dir` (default `~/models/qwen3-embedding-0.6b`), `--backend cpu|flex`,
`--dtype f32|f16|bf16` (f32 is the parity mode; bf16/f16 are memory modes, slower on
flex), `--quant none|q8` (q8 = Q8-resident
low-RAM mode), `--attn fused|blocked`, `--chunk`, `--key-block` (blocked
attention only), `--tokens`, `--text`, `--text-file`, `--out`; `gemm` adds
`--m/--n/--k/--transb`. Serving uses the flags of the [Serving](#serving)
section.

Two attention paths, both causal and numerically identical: `fused` (default,
flex's tiled flash-attention kernel; best for long inputs and multi-thread) and
`blocked` (portable blocked online softmax; best single-threaded). Single-core
runs are fastest with `--attn blocked --chunk 256 --key-block 256`.

### Dev-host results

AMD Ryzen 9 5950X (Zen3, AVX2+FMA), Burn 0.22.0, all rows Q8 weights:

| Attention | Tokens | Threads | Time | Speed |
|---|---|---|---|---|
| blocked | 3633 | 1 | 51.7 s | **70.2 tok/s** |
| fused | 3633 | 1 | 79.9 s | 45.5 tok/s |
| blocked | 3633 | 4 | 25.6 s | 141.8 tok/s |
| fused | 3633 | 4 | 25.6 s | 142.1 tok/s |
| blocked | 6501 | 1 | 125.5 s | 51.8 tok/s |
| fused | 6501 | 4 | 61.6 s | **105.6 tok/s** |
| blocked (chunk 512/key 1024) | 30000 | 32 | 973.9 s | 30.8 tok/s |
| fused | 30000 | 32 | 572.8 s | **52.4 tok/s** |
| low-RAM `--quant q8` (blocked) | 3633 | 1 | 53.1 s | 68.5 tok/s |
| low-RAM `--quant q8` (fused) | 3633 | 4 | 22.4 s | 162.3 tok/s |

Numerical parity vs production (`ik_llama.cpp` Q8_0, last-token pooling,
unnormalized), cosine on newline-free text: 0.999375 (f32, 82 tokens), 0.999280
(Q8, 82), 0.999411/0.999126 (955), 0.999485/0.999209 (6501).

### Board results (rock-5b-plus, 2026-10-05)

Deployed and measured against the production server, both on cores 4-7 with
4 threads:

| Input | Production `ik_llama.cpp` | rocket-inference (flex f32, low-RAM) |
|---|---|---|
| 3,633 tokens | **46.7 s / 77.8 tok/s** | 106.2 s / 34.2 tok/s |
| 6,501 tokens | 124.2 s / 52.3 tok/s | 262.2 s / 24.8 tok/s |

- The 50 tok/s target is **not met** on the board; production is 2.28× faster
  because it uses int8 kernels (flex has no int8 GEMM). Production stays on
  `ik_llama.cpp`.
- Memory: 771 MiB resident in low-RAM mode vs ~4 GB for the production server.
- Numerics: board cosine 0.99928 at 82 tokens vs the unquantized reference,
  identical to the dev host.

### NPU offload

All 196 projections are packed into resident fp16 NPU buffers (0.82 GiB; q|k|v
and gate|up are each one segmented weight) and the CPU keeps only an f16
embedding table (**298 MiB resident**).

| Mode (3,633 tokens, cores 4-7, 4 threads) | Wall | Speed | CPU-seconds |
|---|---|---|---|
| CPU-only (f32) | 100.1 s | 36.3 tok/s | 622 |
| `--npu --npu-attn cpu` | 86.9 s | 41.8 tok/s | 409 (-34%) |
| `--npu` (NPU attention, default) | 80.8 s | 45.0 tok/s | **302 (-51%)** |

Numerics: cosine 0.999365 vs the production Q8_0 reference; the NPU attention
run is 0.999997 vs the CPU-attention NPU run. Requirements: the 600 MHz-patched
`rocket` module on the board (contained; reboot reverts). `--npu` requires
`--dtype f32` and excludes `--quant q8`.

## Qwen3.5-0.8B intent (`intent`)

`guoxuter/ov_intent_analysis_sft:v7_q8` is Qwen3.5-0.8B, a hybrid decoder
(18 Gated DeltaNet linear-attention layers + 6 gated full-attention layers) —
a different family from Qwen3. It serves OpenViking's query-planner role (both the
Ollama API and OpenAI chat; see [Serving](#serving)). The HF safetensors (`model.language_model.*`; the
vision tower is skipped) load through the family's loader; `--npu` packs the
projection groups into resident NPU weights for prefill and keeps f16 CPU copies
for single-token decode.

```sh
M=~/models/ov-intent-analysis-sft
# one-shot greedy generation (raw prompt; drop --raw to wrap in the ChatML template)
$B intent gen --model-dir $M --text "Hello!" --raw --max-new-tokens 32
```

Flags: `--model-dir` (default `~/models/ov-intent-analysis-sft`), `--backend`,
`--text`, `--text-file`, `--raw`, `--max-tokens`, `--max-new-tokens`,
`--temperature`, `--delta-chunk`, `--npu`, `--npu-threads`, `--embed-f16`,
`--pure-npu`, `--npu-decode`, `--dump-hidden`, `--dump-steps`.

The Ollama endpoints accept any request `Content-Type` (like Ollama itself):
litellm's Ollama client posts its bodies as `application/octet-stream` (litellm
1.83) and axum's `Json` extractor would otherwise reject them with 415. litellm
routes `ollama/…` through `/api/generate` and `ollama_chat/…` through
`/api/chat`.

On the board (166-token v7 planner prompt, 32 greedy tokens, cores 4-7):

| Mode | Prefill | Decode | Total | Anon RSS |
|---|---|---|---|---|
| CPU f32 | 7.91 s | 82.05 s (0.4 tok/s) | 89.96 s | ~2.4 GB |
| `--npu --npu-threads 3` | 2.94 s | 8.71 s (3.7 tok/s) | **11.65 s** | 2.06 GB |
| `--npu --embed-f16` | 2.84 s | 6.45 s (5.0 tok/s) | 9.29 s | 1.57 GB |

The CPU f32 and NPU f32-table outputs are token-identical to the HF transformers
reference (32/32 greedy tokens; per-layer hidden-state cosine 1.0).
`--embed-f16` is faster and smaller but its f16 LM head flips near-ties.
`--pure-npu` drops the CPU copies entirely and runs decode on the NPU too
(1.5 tok/s), useful only when CPU should be left alone.

## EmbeddingGemma 2 (`gemma embed|bench|tokenize|serve`)

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) is a
multimodal embedding model built from the Gemma 4 family: a 270M-parameter
bidirectional text decoder, a 170M vision tower and a 300M audio tower,
projected into one 768-d embedding space (MRL: 128/256/512/768), with per-layer
embeddings (PLE) and an 8192-token context. The full checkpoint — text, image,
video and audio — is implemented, reproducing the HF reference numerics
(`docs/experiment-log-gemma.md`).

```sh
M=~/models/embeddinggemma-2

# text embedding (768-d, L2-normalized; --dim for MRL truncation, --prompt for task prefixes)
$B gemma embed --model-dir $M --text "what is the capital of france?" --out out.json
$B gemma embed --model-dir $M --prompt query --dim 256 --text-file data/one_long.txt --out out.json

# low-RAM mode: Q8-resident projections, dequantized per call (1216 MiB vs 2845 MiB)
$B gemma embed --model-dir $M --quant q8 --text-file data/one_long.txt --out out.json

# image (the placeholder <|image|> is expanded to BOI + soft tokens + EOI)
$B gemma embed --model-dir $M --image data/cat.jpeg --out img.json
# video (1 fps, at most 32 frames) and audio (16 kHz mono WAV; else ffmpeg)
$B gemma embed --model-dir $M --video data/test.mp4 --out vid.json
$B gemma embed --model-dir $M --audio data/speech5s.wav --out audio.json

# timings and token ids
$B gemma bench --model-dir $M --text-file data/one_long.txt --reps 2
$B gemma tokenize --model-dir $M --text "hello"
```

Flags: `--model-dir` (default `~/models/embeddinggemma-2`), `--backend`,
`--dtype f32` (f16 is rejected — RMSNorm/softmax/PLE need f32), `--text`,
`--text-file`, `--image`, `--video`, `--video-fps`, `--video-max-frames`,
`--audio`, `--max-soft-tokens`, `--video-soft-tokens`, `--prompt`, `--dim`,
`--no-normalize`, `--tokens`, `--reps`, `--out`, `--attn-chunk`, `--quant none|q8`,
`--npu`, `--npu-threads`, `--npu-attn npu|cpu`, `--npu-int8`,
`--npu-int8-group` (default 32), plus `--dump-*`
debug hooks. On the NPU build, `ROCKET_CTX_CANONICAL=0` restores the legacy
pad-to-256 tiling and `ROCKET_GLUE=0/1` toggles the fused host kernels
(`--npu` enables them; `=1` forces them in CPU modes).

Task prompts (`query`, `document`, `STS`, `classification`, `clustering`,
`code`, ...) are read from `config_sentence_transformers.json`; omit `--prompt`
for no prefix (`--prompt none` is also accepted).

Server examples (and the `/embed` request shapes) are in
[Serving](#serving). Every embedding is L2-normalized; `dim` applies MRL
truncation before normalization.

### Verification and memory

Summary against the HF f32 reference (transformers 5.19; full table in the log):

| input | cosine |
|---|---|
| text (9 tok, 2594 tok; dims 768/256) | 1.00000000 |
| image (PNG/JPEG), video (4 frames), audio (5 s/30 s) | 1.00000000 |
| text + image / text + audio | 1.00000000 |
| any modality, `--quant q8` | 0.9996-0.9999 |
| any modality, `--quant q8 --npu` (board) | 0.9992-0.9999 |

Memory: f32 weights ~2.9 GB resident; `--quant q8` 1216 MiB; `--quant q8 --npu`
1068 MiB; all share a ~4.2 GiB load peak. The text backbone on the NPU packs 218
projections into 0.25 GiB of resident fp16 weights with windowed attention: on
the board (4 A76 threads, 2587-token text) `--quant q8` 34.5 s (75 tok/s, 94 s
user CPU) -> `--quant q8 --npu` 21.4 s (121 tok/s, 63 s user CPU): 1.61× faster,
33% less CPU. The vision/audio towers, norms, RoPE and the embedding table stay
on the CPU.

Performance round (2026-10-08, log §12; board timings in §12.5):
canonical tiling (short requests no longer pad every matmul to 256 rows),
resident int8 text projections (`--npu-int8`; cosine 0.99985 vs HF f32 — better
than the q8 path's 0.99965 — at 0.11 GiB resident; readback-bound, so a
numerics/memory option rather than a speed one), chunked banded attention
for the 20 sliding layers (a 7764-token run sits at 1068 MiB anon and matches
the full path's numerics, where the full-matrix path OOM-kills inside a 3 GB
cgroup), and fused host kernels for gelu·up/RMSNorm/RoPE (`ROCKET_GLUE=0/1` A/B;
q8 CPU glue on-vs-off cosine 1.0).

Known limits: audio resampling is exact for 16 kHz mono WAV (other rates go
through ffmpeg, ~0.994 cosine vs librosa/soxr); batch inputs are processed one
at a time.

## Gemma 4 E2B-it (`gemma gen`)

`google/gemma-4-E2B-it` (10.25 GB BF16 checkpoint): the causal Gemma 4 decoder
with KV sharing (layers 15-34 reuse the K/V of layer 13/14 and use double-wide
MLPs), per-layer embeddings with the token table, proportional p-RoPE on the
full layers and a soft-capped tied LM head. Greedy output is **token-identical**
to the HF 5.19 reference (see the log).

```sh
G=/mnt/hub/models/gemma-4-E2B-it   # dev host; ~/models/gemma-4-E2B-it elsewhere

# greedy generation (f32 parity mode)
$B gemma gen --gen-model-dir $G --text "What is the capital of France?" --max-new-tokens 64
# chat-style messages and sampling
$B gemma gen --gen-model-dir $G \
    --messages '[{"role":"system","content":"You are terse."},{"role":"user","content":"Hi"}]' \
    --sample --temperature 0.7 --top-k 64 --top-p 0.95
# image / audio input (the placeholder expands to the tower's soft tokens)
$B gemma gen --gen-model-dir $G --text "What is in this image? <|image|>" --image data/cat.jpeg
$B gemma gen --gen-model-dir $G --text "What do you hear? <|audio|>" --audio data/speech5s.wav
```

Flags: `--gen-model-dir` (default `/mnt/hub/models/gemma-4-E2B-it`), `--backend`,
`--text`, `--text-file`, `--image`, `--audio`, `--max-soft-tokens`,
`--video-fps`, `--video-max-frames`, `--messages`, `--max-new-tokens`,
`--sample`, `--temperature`, `--top-k`, `--top-p`, `--enable-thinking`, `--f16`,
`--quant none|q8`, `--npu`, `--npu-threads`, `--attn-chunk`, `--dump-logits`,
`--out`. Serving uses `--model-dir` and the flags of the [Serving](#serving)
section.

Precision modes (`gen` and serving), measured on the dev host (32 threads,
16-token prompt / 9-token answer):

| mode | flags | resident | decode | notes |
|---|---|---|---|---|
| f32 (default) | - | 12494 MiB | 2.8 tok/s | token-identical to HF on every tested prompt |
| f16 | `--f16` | 8923 MiB | 3.5 tok/s | token-identical on short/multi-turn; **can diverge on long prompts** |
| Q8_0 | `--quant q8` | 7290 MiB | 0.09 tok/s | memory-only: `lin` dequantizes each weight per call |

The served chat endpoints are non-streaming (OpenAI) / single-line NDJSON
(Ollama); OpenAI content parts accept `image_url` (data URI or path) and
`input_audio` (base64 + format) — one of each per request; the server inserts
the `<|image|>` / `<|audio|>` placeholder tokens itself, so do not include them
in the text part.

### QAT mobile checkpoint

`google/gemma-4-E2B-it-qat-mobile-transformers` (2.46 GB) is supported natively
— no conversion step. Weights are packed INT2/INT4/INT8 with per-output-channel
scales (per-row block scales for the token tables) and the checkpoint's SRQ
activation rounding is applied around the affected linears. `--f16` is the
useful configuration:

```sh
$B gemma gen --gen-model-dir ~/models/gemma-4-E2B-it-qat-mobile --f16 \
    --text "What is the capital of France?" --max-new-tokens 16
```

| mode | resident | short decode | notes |
|---|---|---|---|
| f32 | 12.0 GiB | 3.9 tok/s | parity mode (f32 projections) |
| `--f16` | 6.6 GiB | 4.9 tok/s | 12/12 tokens identical to the f32 reference on the 631-token prompt |
| `--quant q8` | 6.1 GiB | 0.07 tok/s | memory-only |

The loaded weights are bit-exact vs HF 5.19 (verified for a 4-bit projection, a
2-bit MLP and the 4-bit PLE table); short prompts are token-identical and the
631-token prompt is 12/12 with `--f16`. The PLE table stays packed (rows
dequantized on lookup, 1.13 GiB instead of 8.75 GiB f32). SRQ makes long-context
near-ties sensitive; `NO_SRQ=1` / `DUMP_PARAM=<substr>` are debug hooks.

### NPU prefill

`--npu` packs the used text projections into resident fp16 NPU weights and runs
prefill matmuls + `attention_causal_window` on the NPU, while decode keeps the
CPU f32 copies. Board measurement pending an idle board (the model needs
~9.5 GB host + 2.1 GiB NPU; the 2026-10-07 attempt was OOM-killed on a busy
board). Not yet done: streaming, video input.

## Cross-compiling for the board (aarch64)

`--no-default-features` drops the CubeCL/LLVM `cpu` backend, whose prebuilt LLVM
bundle is host-architecture and cannot link for aarch64; `flex` (NEON on
aarch64) is always available. The linker is configured in `.cargo/config.toml`.
`--features npu` links `librocketnpu.a`: `build.rs` uses `ROCKETNPU_DIR` when
set, else a matching `vendor/rocketnpu/`, else builds one into `$OUT_DIR` with
`scripts/build-rocketnpu.sh`.

## Notes

- `Device::sync()` is required before reading wall-clock times; `to_data()`
  alone does not wait for execution.
- Tokenization uses the HF `tokenizer.json`; the qwen path uses
  `add_special_tokens=true`, matching `llama-embedding` token counts.
- All servers serialize requests on the model (flex already parallelizes
  internally; the NPU engine is serialized by design), contain panicking
  forwards instead of poisoning the server, and keep `/health`/`/v1/models`
  lock-free.
- `--quant q8` keeps projection weights Q8_0-quantized (symmetric int8,
  32-value blocks, f16 block scales = llama.cpp's Q8_0) and dequantizes each
  weight per forward; `libc::malloc_trim` releases the freed f32 pages.
