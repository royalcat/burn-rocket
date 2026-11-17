# Qwen3.5-0.8B intent (query planner)

OpenAI chat and Ollama endpoints for the OpenViking query-planner model. Best-effort model:
numbers are from [`../experiments/experiment-log.md`](../experiments/experiment-log.md) §14
(a 166-token planner prompt, 32 generated tokens, on a busy board).

## Run it

```sh
rocket-inference serve --model-dir /root/models/ov-intent-analysis-sft --port 11434
```

## What the preset does

| Stage | Where | Why |
|---|---|---|
| prompt processing (prefill) | NPU | 7.9 s on the CPU, 2.9 s on the NPU |
| token-by-token decode | **CPU**, using f16 copies of the weights | measured 3.7 tok/s vs 1.5 tok/s with NPU decode |
| mixer attention | CPU | no NPU kernel for the delta rule |

The 166-token prompt answers in 11.7 s (89.9 s on the CPU alone). The program prints a
`hint:` line when it leaves decode on the CPU.

## What it trades away

- About 2 GB of RAM, because the CPU copies of the projections stay resident for decode.
- A busy CPU during generation. The NPU is idle while tokens are generated.
- The 1.5 tok/s NPU decode figure predates canonical tiling, so it may be better today; it
  has not been re-measured.

## Tune it

| You want | Flags | Cost |
|---|---|---|
| Fastest | `--embed-f16` | 9.3 s instead of 11.7 s and 1.57 GB instead of 2.06 GB; the f16 LM head can flip near-ties, so skip it when token identity matters |
| Leave the CPU free | `--device npu` | decode on the NPU, no CPU copies: 1.29 GB, about 1.5 tok/s |
| NPU decode and CPU copies | `--decode-device npu --keep-cpu-weights yes` | the copies stay for a later switch |
| No NPU | `--device cpu` | 90 s for the same prompt |
| Fewer NPU workers | `--npu-threads 3` | used in the measured run; smaller scratch |

`--delta-chunk` (default 64) is the delta-rule chunk width. One-shot:
`rocket-inference gen --model-dir $M --text "Hello!" --raw --max-new-tokens 32`.
