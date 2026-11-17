# EmbeddingGemma 2 recipes

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) is the
first-class model: text, image, video and audio embeddings in one 768-d space.
The text backbone and the vision tower can run on the NPU (projections and
attention); the audio tower always runs on the CPU.

Checkpoint: `/root/models/embeddinggemma-2` (safetensors, native bf16) on the
board, or the Q8_0 GGUF via `-hf unsloth/embeddinggemma-2-GGUF:Q8_0`.

## Serve

```sh
M=/root/models/embeddinggemma-2

# balanced: every stage on the NPU, bf16 body, towers load on first use
rocket-inference serve --model-dir $M --device npu --dtype bf16 --defer-towers --port 8383

# full offload: every stage on the NPU, f32, both towers resident at startup
rocket-inference serve --model-dir $M --device npu --dtype f32 --port 8383

# speed: every stage on the NPU, f32, lazy towers (fastest text, low idle RAM)
rocket-inference serve --model-dir $M --device npu --dtype f32 --defer-towers --port 8383
```

`--device npu` is optional on the NPU build (it is what `auto` resolves to for
this model) but makes the placement explicit. The startup line reports the
resolved split:

```
placement: text=NPU(fp16) text-attn=NPU vision=NPU(lazy) vision-attn=NPU audio=CPU
```

The GGUF deployment shape (the live service's checkpoint) is the same flags
with `-hf`, plus the memory-round settings measured in log-gemma §23:

```sh
ROCKET_MATMUL_CHUNK_M=1024 rocket-inference serve \
    -hf unsloth/embeddinggemma-2-GGUF:Q8_0 --device npu \
    --defer-towers --attn-chunk 512 --port 8383 --max-tokens 8192
```

`ROCKET_MATMUL_CHUNK_M=1024` bounds the per-call matmul input pack (8k matmul
scratch 681 → 235 MiB, wall neutral or slightly better) and `--attn-chunk 512`
halves the full-layer attention scratch (424 → 245 MiB, wall neutral). The
default `ROCKET_KEEP_MB=64` then drops whatever the finished request left above
64 MiB (matmul + attention, BO and host), so an 8k request returns the process
to its weights instead of keeping ~1 GiB of BOs. Lower-memory option:
`--npu-threads 3` (−131 MiB, +12 % wall at 8k).

A text-only service never loads the towers, so `--defer-towers` costs nothing
there. Add `--quant q8` only if the CPU-resident audio tower's RAM matters:
the packed text weights always come from the full-precision weights now, and a
q8 audio tower is ~2.4x slower for audio requests.

## One-shot commands

```sh
M=/root/models/embeddinggemma-2
B=./rocket-inference
D=data

# balanced
$B embed --model-dir $M --device npu --dtype bf16 --text "what is the capital of france?" --out /tmp/e.json
$B embed --model-dir $M --device npu --dtype bf16 --image $D/cat.jpeg --out /tmp/i.json
$B embed --model-dir $M --device npu --dtype bf16 --audio $D/speech5s.wav --out /tmp/a.json
$B embed --model-dir $M --device npu --dtype bf16 --video $D/test.mp4 --out /tmp/v.json

# full offload / speed (same placement, f32; full offload also loads the towers up front)
$B embed --model-dir $M --device npu --dtype f32 --defer-towers --text "hello" --out /tmp/e32.json

# task prefixes and MRL truncation (text)
$B embed --model-dir $M --device npu --prompt query --dim 256 \
    --text-file $D/one_long.txt --out /tmp/q256.json

# throughput: repeated forwards + stage timings
$B bench --model-dir $M --device npu --dtype bf16 --text-file $D/one_long.txt --reps 2
```

## CPU fallback

```sh
rocket-inference serve --model-dir $M --device cpu --dtype f32 --defer-towers
```

Same outputs at the f32 parity level, 2x the wall and ~2.1-2.6 GiB resident;
`--quant q8` cuts RAM to ~0.9-1.2 GiB at 20-60 % more wall.
