# Qwen3-Embedding-0.6B recipes

Best-effort: the numbers below are derived from the dated experiment logs
(`docs/experiment-log.md` §9/§17), not re-measured in the 2026-10-31 round.
The command lines use the current unified flags.

```sh
M=/root/models/qwen3-embedding-0.6b

# balanced: bf16 body, text + attention on the NPU
rocket-inference serve --model-dir $M --device npu --dtype bf16 --port 8383

# full offload: f32 resident fp16 NPU weights (the fastest and most accurate)
rocket-inference serve --model-dir $M --device npu --dtype f32 --port 8383

# speed: same as full offload (there is no decode or tower stage to trade)
rocket-inference serve --model-dir $M --device npu --dtype f32 --port 8383

# CPU low-RAM fallback
rocket-inference serve --model-dir $M --device cpu --quant q8 --port 8383
```

| profile | flags | board result (3,633 tokens) | resident |
|---|---|---|---|
| balanced | `--device npu --dtype bf16` | 54.4 tok/s | ~60 % of f32 |
| full offload | `--device npu --dtype f32` | 57.8 tok/s | 298 MiB anon + 0.82 GiB NPU weights |
| speed | `--device npu --dtype f32` | 57.8 tok/s | same |
| CPU f32 | `--device cpu --dtype f32` | 36.3 tok/s / 622 CPU-s | ~2.4 GB |
| CPU q8 | `--device cpu --quant q8` | 34.2 tok/s | 771 MiB |

Notes:

- Attention is a separately placeable stage. `--device cpu --text-device npu`
  keeps the projections on the NPU but the attention on the CPU: 41.8 tok/s vs
  45.0 with NPU attention in the 2026-10-05 round (CPU attention is within ~1 s
  of wall but uses ~40 % more CPU).
- The CPU attention kernel choice is `--attn-kernel fused|blocked`
  (`--chunk`/`--key-block` tune the blocked kernel).
- `--quant q8` applies to CPU-resident weights; it is not meaningful with
  `--text-device npu` (the packed weights come from the model dtype).
- Numerics: f32 NPU 0.999365 vs the production Q8_0 reference; bf16 NPU
  0.999845 vs f32.
- One-shot `embed`/`bench` are the same commands with the input flags, e.g.
  `rocket-inference bench --model-dir $M --device npu --dtype f32 --text-file data/bench_text.txt --reps 2`.
