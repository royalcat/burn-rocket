# Qwen3.5-0.8B intent recipes

Best-effort: the numbers are from `docs/experiment-log.md` §14 (166-token v7
planner prompt, 32 generated tokens, busy board), not re-measured in the
2026-10-31 round. The command lines use the current unified flags.

```sh
M=/root/models/ov-intent-analysis-sft

# balanced: prefill projections on the NPU, decode on the CPU
# (`auto` is the default; the CPU copies of the projections are kept)
rocket-inference serve --model-dir $M --device auto --port 11434

# full offload: prefill and decode on the NPU; the CPU projection copies are
# dropped (the old --pure-npu shape)
rocket-inference serve --model-dir $M --device npu --port 11434

# speed: prefill NPU, decode CPU, f16 embedding/LM-head table
rocket-inference serve --model-dir $M --device auto --embed-f16 --port 11434

# CPU fallback
rocket-inference serve --model-dir $M --device cpu --port 11434
```

| profile | flags | prefill | decode | total | anon |
|---|---|---|---|---|---|
| balanced | `--device auto` | 2.94 s | 8.71 s (3.7 tok/s) | 11.65 s | 2.06 GB |
| full offload | `--device npu` | — | ~1.5 tok/s | — | 1.29 GB |
| speed | `--device auto --embed-f16` | 2.84 s | 6.45 s (5.0 tok/s) | 9.29 s | 1.57 GB |
| CPU f32 | `--device cpu` | 7.91 s | 82.05 s (0.4 tok/s) | 89.96 s | ~2.4 GB |

Notes:

- `--device npu` deliberately includes `--decode-device npu`; NPU decode
  single-token matmuls pay the `M >= 256` padding, so full offload is a
  CPU-relief configuration, not a speed one. Use
  `--decode-device npu --keep-cpu-weights yes` if you want NPU decode with the
  CPU copies still resident (the old `--npu-decode`).
- `--embed-f16` is faster and smaller but its f16 LM head can flip near-ties;
  keep it off when token-identity matters.
- `--delta-chunk` (default 64) is the delta-rule chunk width.
- One-shot generation is the same command set:
  `rocket-inference gen --model-dir $M --device auto --text "Hello!" --raw --max-new-tokens 32`.
