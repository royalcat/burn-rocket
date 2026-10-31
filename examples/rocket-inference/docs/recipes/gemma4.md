# Gemma 4 E2B-it recipes

Best-effort: the board has not completed a Gemma 4 run since the memory
rounds (the checkpoint is 8.9 GiB at f16 / 12.5 GiB f32, and the board shares
~15.8 GiB with its services). The numbers below are from
`docs/experiment-log-gemma.md` §13 on the dev host; the command lines use the
current unified flags. The QAT mobile checkpoint is the board-friendly one.

```sh
M=/root/models/gemma-4-E2B-it-qat-mobile

# balanced: prefill projections on the NPU, decode on the CPU, lazy towers
rocket-inference serve --model-dir $M --device auto --dtype f16 --defer-towers --port 8391

# full offload: prefill and decode on the NPU, CPU projection copies dropped
rocket-inference serve --model-dir $M --device npu --dtype f16 --defer-towers --port 8391

# speed: same as balanced for the board-safe QAT path (decode stays on the CPU)
rocket-inference serve --model-dir $M --device auto --dtype f16 --defer-towers --port 8391

# CPU-only, f32 parity (12.5 GiB: dev host only)
rocket-inference serve --model-dir /mnt/hub/models/gemma-4-E2B-it --device cpu --port 8391
```

| profile | flags | known result |
|---|---|---|
| balanced | `--device auto --dtype f16 --defer-towers` | dev host text-only peak 4.7 GiB, 5.2 tok/s decode |
| full offload | `--device npu --dtype f16 --defer-towers` | unmeasured on the board; decode matmuls on the NPU |
| speed | `--device auto --dtype f16 --defer-towers` | 631-token prompt 16/16 ids vs HF bf16 with `--f16` |
| CPU q8 | `--device cpu --quant q8` | 3.7 GiB but 0.09 tok/s (memory-only) |

Notes:

- The NPU stage is the text prefill: `--device auto` packs 218 text
  projections and runs prefill matmuls + causal attention on the NPU, decode on
  the CPU. `--device npu` also routes decode to the packed weights and drops
  the CPU copies (`--keep-cpu-weights yes` keeps them).
- The vision/audio towers are not NPU-packed for Gemma 4; the CLI rejects
  `--vision-device` on this family. `--defer-towers` loads each half on first
  use.
- `--dtype f16` is the QAT mobile mode; plain checkpoints use their native bf16
  (`--dtype auto`) or `--dtype f32` for parity.
- One-shot generation:
  `rocket-inference gen --model-dir $M --device auto --dtype f16 --defer-towers --text "What is the capital of France?" --max-new-tokens 16`.
