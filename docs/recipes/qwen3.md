# Qwen3-Embedding-0.6B

Text embeddings, served at `/v1/embeddings`. Best-effort model: the numbers come from the
dated logs in [`../experiments/experiment-log.md`](../experiments/experiment-log.md) (§9, §17).

## Run it

```sh
rocket-inference serve --model-dir /root/models/qwen3-embedding-0.6b --port 8383
```

That is the whole recipe: `--device auto` picks the preset below.

## What the preset does

| Stage | Where | Why |
|---|---|---|
| projections (q|k|v, o, gate|up, down) | NPU | the bulk of the arithmetic |
| attention | NPU | about the same wall time as the CPU kernel but roughly 40 % fewer CPU-seconds |
| weights | the checkpoint's native bf16, packed to fp16 on the NPU | no extra conversion step to set up |

Reference run, 3,633 tokens on a Rock 5B+ (cores 4–7): about 54 tok/s, with about a half
of the CPU time of the CPU-only path.

## What it trades away

- bf16 is about 6 % slower than f32 (54.4 vs 57.8 tok/s) in exchange for a checkpoint that is
  half the size to read and to hold while loading.
- The CPU stays only lightly loaded, but the NPU is a single shared device: other NPU
  services queue behind a long request.

## Tune it

| You want | Flags | Cost |
|---|---|---|
| Fastest and most accurate | `--dtype f32` | 57.8 tok/s; about 0.8 GiB of NPU weights plus a larger load peak |
| Projections on the NPU, attention on the CPU | `--text-attn-device cpu` | 41.8 vs 45.0 tok/s, about 40 % more CPU time |
| CPU only, little RAM | `--device cpu --quant q8` | 34.2 tok/s, 771 MiB resident |
| CPU only, f32 | `--device cpu --dtype f32` | 36.3 tok/s, 622 CPU-seconds per 3,633 tokens |
| Very long inputs | `ROCKET_CAUSAL_CHUNK=512` | lower attention memory; the default 1024 is bit-identical to the dense op |
| Shrink the largest request | `ROCKET_MATMUL_CHUNK_M=1024` | smaller peak, wall time unchanged |

`--quant q8` only applies to CPU-resident weights; with NPU projections the packed weights
come from the model dtype. One-shot runs use the same flags:
`rocket-inference bench --model-dir $M --text-file data/bench_text.txt --reps 2`.
