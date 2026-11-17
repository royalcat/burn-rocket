# Gemma 4 E2B-it

Chat with text, image and audio input (OpenAI chat and Ollama endpoints). Best-effort model:
see [`../experiments/experiment-log-gemma.md`](../experiments/experiment-log-gemma.md) and the
root `AGENTS.md` section "Gemma 4 QAT on the NPU" for the measurements.

## Run it

```sh
rocket-inference serve --model-dir /root/models/gemma-4-E2B-it-qat-mobile --port 8391
```

The preset depends on the checkpoint.

### QAT checkpoint (`...-qat-mobile-transformers`, recommended on the board)

| Stage | Where | Why |
|---|---|---|
| all text projections, prefill **and** decode | NPU, per-channel int8 | the checkpoint is already int8 with one scale per channel, so the NPU math is exact |
| towers | CPU, loaded on first use | not NPU-packed |

E2B on the board: the model loads in 76 s with about 170 MiB of RAM and 2.5 GiB of NPU
buffers (the f16 path took 119 s plus a 81 s pack and a 6.8 GiB peak). Decode is about
1.45 tok/s (0.78 on the CPU), a 1,073-token prompt prefills in 41.5 s (49.5 s with f16).
Greedy output matched the f16 CPU run for 32 of 32 tokens.

### Other checkpoints (bf16 / f32 / GGUF)

| Stage | Where |
|---|---|
| prefill projections and causal attention | NPU, packed to fp16 |
| decode | CPU, with CPU copies of the weights |
| towers | CPU, loaded on first use |

The program prints a `hint:` line explaining why decode stays on the CPU: NPU decode of fp16
weights has not been shown to be faster. This path needs about 9 GiB for the f16 weights and
has not been run end to end on the board.

## What it trades away

- QAT int8 changes the rounding slightly: long prompts can flip a near-tie token against the
  f16 reference (SRQ activation rounding is sensitive).
- Decode is slow on every path (about 0.5–1.5 tok/s on the board); the preset favours low RAM.
- The first image or audio request pays the tower load (1.6–2.5 s).

## Tune it

| You want | Flags | Cost |
|---|---|---|
| The f16 path on a QAT checkpoint | `--dtype f16` | 4.7 GiB peak RAM text-only, 5.2 tok/s decode on the dev host |
| Everything on the CPU | `--device cpu` | no NPU; slow |
| Towers loaded at startup | `--no-defer-towers` | about 1.9 GiB more RAM, no first-request delay |
| Lowest RAM on the CPU | `--device cpu --quant q8` | 3.7 GiB but 0.09 tok/s |
| Fewer NPU workers | `--npu-threads 3` (already the QAT default) | more workers only contend for the 3 cores |
| Bit-exact f32 parity | `--device cpu --dtype f32` | 12.5 GiB; dev host only |

One-shot: `rocket-inference gen --model-dir $M --text "What is the capital of France?" --max-new-tokens 16`.
Stop other NPU services before benchmarking (they share the NPU and cores 4–7).
