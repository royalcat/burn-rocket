# EmbeddingGemma 2

The first-class model: text, image, video and audio embeddings in one 768-d space, served at
`/v1/embeddings` and `/embed`. Measurements are in
[`../experiments/embeddinggemma2-board.md`](../experiments/embeddinggemma2-board.md).

## Run it

```sh
# the checkpoint the live service uses (Q8_0 GGUF, downloaded on first run)
rocket-inference serve -hf unsloth/embeddinggemma-2-GGUF:Q8_0 --port 8383 --max-tokens 8192

# or a local safetensors directory
rocket-inference serve --model-dir /root/models/embeddinggemma-2 --port 8383
```

`--device auto` picks the preset below. The program prints the `placement:` line, a `preset:`
line and any `hint:` lines before it loads.

## What the preset does

| Stage | Where | Why |
|---|---|---|
| text projections | NPU | text forward 27.4 s on the CPU, 13–15 s on the NPU (2,587 tokens) |
| text attention | NPU, in 512-token query chunks | same wall as 1024, half the attention memory |
| vision projections and attention | NPU | a 2,394-patch image's tower takes 10.6 s instead of 15.2 s with CPU attention |
| audio tower | CPU | not an NPU stage |
| towers | loaded on the first request that needs them | a text-only service idles at about 600 MiB instead of 2.6 GiB |
| weights | the checkpoint's native dtype | bf16 halves memory against f32 |

## What it trades away

- The first image, video or audio request pays the tower load: about 2.5 s for the vision half
  and about 5 s for the audio half.
- bf16 text is about 14 % slower than f32 (15.2 s vs 13.3 s for 2,587 tokens) in exchange for
  about half the RAM (723 vs 1,330 MiB).
- NPU work is serialized, so a long request delays others.

## Tune it

| You want | Flags | Cost |
|---|---|---|
| Fastest text | `--dtype f32` | 13.3 s instead of 15.2 s; 1.3 GiB instead of 0.7 GiB |
| No first-request delay | `--no-defer-towers` | about 1.3 GiB more RAM and about 10 s more startup |
| Lowest memory on long inputs | `ROCKET_MATMUL_CHUNK_M=1024` | 8k text peak 1.4 → 0.8 GiB of NPU buffers, wall unchanged |
| Even smaller attention scratch | `--attn-chunk 256` | slightly more dispatch overhead |
| Less NPU buffer memory | `--npu-threads 3` | −131 MiB, +12 % wall at 8k tokens |
| CPU only | `--device cpu` | 2× the wall, about 2.1–2.6 GiB |
| CPU only, small | `--device cpu --quant q8` | 0.9–1.2 GiB, 20–60 % more wall |
| Smaller NPU weights | `--npu-dtype int8 --int8-group 512` | ties fp16 on wall, text cosine 0.998; g32 is 2.9× slower |

Short inputs are dominated by load time (a 9-token request is 6–10 s as a one-shot command), so
serve once and reuse the process.

## One-shot commands

```sh
M=/root/models/embeddinggemma-2
rocket-inference embed --model-dir $M --text "what is the capital of france?" --out /tmp/e.json
rocket-inference embed --model-dir $M --image data/cat.jpeg --out /tmp/i.json
rocket-inference embed --model-dir $M --audio data/speech5s.wav --out /tmp/a.json
rocket-inference embed --model-dir $M --video data/test.mp4 --out /tmp/v.json
rocket-inference embed --model-dir $M --prompt query --dim 256 --text-file data/one_long.txt --out /tmp/q.json
rocket-inference bench --model-dir $M --text-file data/one_long.txt --reps 2
```

`--prompt` adds a task prefix (`query`, `document`, ...); `--dim` truncates to 128, 256, 512 or
768 before normalising. Video and non-WAV audio need `ffmpeg` (the container image has it).
