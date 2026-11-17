# rocket-inference

Run embedding, chat and decision models on a **Rock 5B+ (RK3588)** and use its NPU
for the heavy matrix math. One binary, one HTTP server, OpenAI- and Ollama-compatible
APIs, so existing clients work unchanged. It also runs on an ordinary x86-64 or
aarch64 CPU.

It is written in Rust on [Burn](https://burn.dev) and a pure-Rust driver for the
mainline `rocket` NPU kernel driver. There is no C toolchain, vendor SDK or
kernel-header step.

> **AI disclosure.** This project was developed entirely by AI coding agents under
> human direction. Every result was checked on real hardware: numerics against
> reference implementations (cosine parity, token-identical output) and speed on a dev
> host and the board.

## Supported models

The model family is detected from the checkpoint, so `serve` is the same command for
all of them.

| Model | What it does | Endpoints |
|---|---|---|
| [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | text, image, video and audio embeddings (the first-class model) | `/v1/embeddings`, `/embed` |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | text embeddings | `/v1/embeddings` |
| Qwen3.5-0.8B intent model | query planning / chat | `/v1/chat/completions`, Ollama `/api/*` |
| Gemma 4 E2B-it | chat with text, image and audio input | `/v1/chat/completions`, Ollama `/api/*` |
| vllm-sr d3 (lite / nano) | typed decisions (`choice`, `noul`, `score` questions) | `POST /v1/systemone` |
| Clef-Omni (30B-A3B MoE) | typed decisions over text, image, audio and video | `POST /v1/systemone` |

GGUF checkpoints work too, locally or straight from the Hugging Face Hub with
`-hf <owner>/<name>:<quant>`.

## Quick start

Build (the CPU-only build works anywhere; add `--features npu` on aarch64):

```sh
cargo build --release --no-default-features
# for the board, cross-compiled:
cargo build --release --target aarch64-unknown-linux-gnu --no-default-features --features npu
```

Serve a model and ask for an embedding:

```sh
./rocket-inference serve -hf unsloth/embeddinggemma-2-GGUF:Q8_0 --port 8383

curl -s localhost:8383/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": "Hello world"}'
```

Other commands: `embed` (one embedding), `bench`, `gen` (one chat reply), `decide`
(typed decisions), `tokenize`, `gemm`. Run `rocket-inference help` or
`help <command>` for the flags.

## Using the NPU

`--device auto` (the default) applies a **per-model preset**: the NPU for every stage where
it is actually faster or smaller, the CPU where it is not (for example the token-by-token
decode of the chat models), and lazily loaded towers. `--device cpu` forces the CPU path and
`--device npu` the NPU for everything. Individual stages can be placed separately; the program
prints the resulting split, and a hint for every stage it left on the CPU on purpose, before it
loads the model. [`docs/recipes/`](docs/recipes/) explains each model's preset, what it trades
away, and how to tune it:

```
placement: text=NPU(fp16) text-attn=NPU vision=NPU(lazy) vision-attn=NPU audio=CPU
```

The NPU build needs `/dev/accel/accel0` at run time. A patched `rocket` kernel module
at 600 MHz is about three times faster than the stock 200 MHz clock.

## What to expect

Measured on a Rock 5B+, shared with other services:

| Model | Result |
|---|---|
| EmbeddingGemma 2, text only (`--defer-towers`) | 605 MiB of RAM when idle; the vision and audio towers load only when a request needs them |
| EmbeddingGemma 2, 2394-patch image | vision tower in 10.6 s on the NPU (15.2 s on CPU attention) |
| Qwen3-Embedding, 3,633 tokens | about 45 tok/s on the NPU with about half the CPU time of the CPU-only path |
| Qwen3.5 intent, 166-token prompt | 11.7 s on the NPU, 89.9 s on the CPU |
| d3-lite, text only | 347 MiB of RAM plus 966 MiB of NPU buffers |

Numerics match the reference implementations: cosine ≥ 0.9999 for the embedding models
in f32, token-identical greedy output for the chat models, and top-1 agreement with the
reference for the decision models. Per-model presets, tuning and
benchmarks are in [`docs/`](docs/).

## Containers

Every push to `main` builds images for both architectures and publishes them to GHCR:

- `ghcr.io/royalcat/rocket-inference:latest` — aarch64, NPU-enabled
- `ghcr.io/royalcat/rocket-inference:latest-amd64` — x86-64-v3 (AVX2), CPU only

Both include `ffmpeg`, so video and non-WAV audio inputs work out of the box. Build
locally with `docker build -t rocket-inference .`.

## Repository layout

| Path | What it is |
|---|---|
| `src/` | the `rocket-inference` app: CLI, HTTP server and the model families |
| `crates/burn-rocket/` | Burn backend extension that moves matmuls and attention onto the NPU |
| `crates/rocket-npu/` | pure-Rust userspace driver for the RK3588 `rocket` NPU |
| `docs/` | deployment recipes and the experiment logs |
| `tools/` | reference scripts and parity checks |
| `data/` | small test inputs (images, audio, video, text) |

## More documentation

- [`docs/recipes/`](docs/recipes/) — measured command lines for every model
- [`docs/experiments/experiment-log.md`](docs/experiments/experiment-log.md), [`docs/experiments/experiment-log-gemma.md`](docs/experiments/experiment-log-gemma.md) — measurements
- [`crates/burn-rocket/README.md`](crates/burn-rocket/README.md), [`crates/rocket-npu/README.md`](crates/rocket-npu/README.md) — the library crates
- [`AGENTS.md`](AGENTS.md) — the full developer and agent reference: all flags and endpoints, device placement, OpenTelemetry, memory behaviour, board protocol, verification commands

## License

GPL-3.0-or-later.
