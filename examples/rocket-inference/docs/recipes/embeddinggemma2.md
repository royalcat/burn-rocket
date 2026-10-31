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
with `-hf`:

```sh
rocket-inference serve -hf unsloth/embeddinggemma-2-GGUF:Q8_0 --device npu \
    --defer-towers --port 8383 --max-tokens 8192
```

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

## Measured (board, 2026-10-31)

Text `data/one_long.txt` = 2,587 tokens; peak is `RssAnon`; `user` is the
process CPU-seconds (the NPU time is not charged to the process, so a lower
`user` with a lower wall is the CPU-relief win).

| profile | flags | forward | tok/s | user | peak anon |
|---|---|---|---|---|---|
| balanced | `--device npu --dtype bf16 --defer-towers` | 15.2 s | 170 | 28.8 s | 723 MiB |
| full offload | `--device npu --dtype f32` (eager) | 13.6 s | 190 | 41.9 s | 2,605 MiB |
| speed | `--device npu --dtype f32 --defer-towers` | 13.3 s | 195 | 27.5 s | 1,330 MiB |
| CPU f32 | `--device cpu --dtype f32` | 27.4 s | 94 | 72.2 s | 1,457 MiB |
| CPU q8 | `--device cpu --quant q8` | 34.6 s | 75 | 83.8 s | 908 MiB |

Media (one process per run, deferred towers, so `encode` includes the tower
load; the second and later requests in a server reuse it):

| input | profile | encode | forward | wall | user | peak anon |
|---|---|---|---|---|---|---|
| image, 2,394 patches | balanced (bf16) | 14.6 s | 1.37 s | 27.5 s | 32.1 s | 1,158 MiB |
| image | speed (f32) | 13.6 s | 1.32 s | 28.8 s | 33.5 s | 1,644 MiB |
| image | CPU f32 | 23.7 s | 2.22 s | 33.1 s | 72.7 s | 2,129 MiB |
| image | CPU q8 | 28.4 s | 5.48 s | 47.0 s | 87.2 s | 1,052 MiB |
| audio, 5 s | balanced (bf16) | 4.8 s | 0.70 s | 20.8 s | 21.3 s | 1,268 MiB |
| audio | speed (f32) | 4.0 s | 0.64 s | 22.4 s | 22.6 s | 2,143 MiB |
| audio | CPU f32 | 4.0 s | 1.16 s | 19.9 s | 20.7 s | 2,612 MiB |
| audio | CPU q8 | 9.7 s | 3.90 s | 37.5 s | 42.5 s | 1,200 MiB |

Reading the table:

- **NPU text is the big lever**: 27.4 -> 13.3 s text, user CPU 72 -> 27 s,
  and image forward 2.2 -> 1.3 s.
- **NPU vision attention/weights**: image `encode` 23.7 -> 13.6 s (1.7x).
- **f32 vs bf16**: f32 is the fastest text (13.3 vs 15.2 s) and the most
  accurate; bf16 halves the peak (723 vs 1,330 MiB) and is slightly faster on
  media wall time because the checkpoint body is half the size. That is the
  balanced trade.
- **CPU q8** is a memory mode only: text is 26 % slower than CPU f32 and the
  audio tower is 2.4x slower (q8 linears dequantize per call). On the NPU path
  `--quant q8` no longer touches the packed text projections, so it is not a
  text-path knob.
- **One-shot short input is load-dominated**: 9-token text is 0.74 s CPU f32
  vs 0.26 s NPU f32 forward, but the process wall is 6.4 s (CPU load) vs
  10.1 s (NPU pack). For short text, serve once and reuse the process.

Numerics vs the same binary's CPU f32 (text 512 tokens, image 2,394 patches):

| placement | text cosine | image cosine |
|---|---|---|
| NPU f32 | 0.9999994 | 0.9999899 |
| NPU bf16 (native) | 0.9999590 | 0.9998737 |
| NPU int8 g512 | 0.9982989 | 0.9996419 |
| CPU q8 | 0.9993961 | — |

## Tuning notes

- `--attn-chunk <n>` (default 1024) is the query-chunk width of the chunked
  banded attention; keep it below the 8k class. `--vision-attn-kernel
  fused|blocked` selects the CPU vision-attention kernel when the vision
  attention is not offloaded; `--attn-kernel fused|blocked` does the same for
  text.
- `--npu-dtype int8` (with `--int8-group`) packs the text projections as
  resident W8A8. g32 is 2.9x slower than fp16 at 2,587 tokens (the group-wise
  kernel reads back one int32 partial per K-group); g512 ties fp16 on wall but
  costs accuracy (text 0.9983 vs CPU f32 at 512 tokens). It is a
  memory/fidelity option, not a speed one.
- `--defer-towers` is strongly recommended: each half loads on the first
  request of its modality. Eager f32 towers add ~1.3 GiB to the resident set
  and ~10 s to startup.
- MRL: `--dim 128|256|512|768` truncates before L2 normalization; `--prompt
  query|document|...` prepends the task prefix from
  `config_sentence_transformers.json`.

## CPU fallback

```sh
rocket-inference serve --model-dir $M --device cpu --dtype f32 --defer-towers
```

Same outputs at the f32 parity level, 2x the wall and ~2.1-2.6 GiB resident;
`--quant q8` cuts RAM to ~0.9-1.2 GiB at 20-60 % more wall.
