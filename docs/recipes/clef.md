# Clef-Omni recipes

[Clef-Omni](https://huggingface.co/Cloudflare/clef-omni) (Qwen3-Omni-30B-A3B, a 48-layer
mixture-of-experts model) answers the same typed questions as d3 (`choice`, `noul`,
`score`) over text, images, audio and video, through `POST /v1/systemone`. One pass over a
record answers all of its questions; nothing is generated.

It is large (66 GiB in BF16), so the weights are **streamed layer by layer** instead of
being held in memory, and a single request takes tens of seconds.

## Which checkpoint

| You pass | What runs | Where |
|---|---|---|
| the int8 AutoRound checkpoint (`-hf moeshawky/clef-omni-int8-autoround`) | int8 experts streamed to the NPU | the board (aarch64 NPU build) |
| the original BF16 checkpoint (`Cloudflare/clef-omni`) | f32 on the CPU | any host, slow |

An optional repacked copy (`--pack-dir`) reads faster; it is never created automatically.

## Serve

```sh
# board, NPU
rocket-inference serve -hf moeshawky/clef-omni-int8-autoround --device npu \
    --max-tokens 4096 --port 8385

# any host, CPU
rocket-inference serve --model-dir /path/to/clef-omni --port 8385
```

Stop other NPU services first on the board; Clef uses all three cores and about 3 GiB of
NPU buffers once the towers are open.

## One-shot

```sh
rocket-inference decide --model-dir /path/to/clef-omni \
    --requests tools/clef_fixtures.json --out answers.json
```

## Timing and memory

| | CPU (dev host, 8 threads) | NPU (Rock 5B+) |
|---|---|---|
| 300-token text record | about 21 s | about 52 s |
| 800-token text record | not measured | about 103 s |
| RAM | under 1 GiB | about 0.7–0.9 GiB, plus about 0.9 GiB of NPU buffers |

Each pass costs a fixed ~30 s floor on the board, because every expert in a layer gets
used. Batching helps: several records share one pass.

## Batching

```sh
rocket-inference serve ... --batch-window-ms 5000 --batch-max 8
```

Requests that arrive within the window are answered together. Four records took 107 s
batched versus 179 s one by one, with identical answers.

## Media

Images, audio, video and audio-in-video all work. The vision tower (~1 GiB) and audio tower
(~1.2 GiB) open on the first request that needs them; set `ROCKET_CLEF_TOWER_RELEASE=1` to
free them after each request. Limits: audio up to `ROCKET_CLEF_AUDIO_MAX_SECONDS` (300 s),
videos up to `ROCKET_CLEF_VIDEO_MAX_FRAMES` (64 sampled frames).

## More

Request and response format, every environment knob, per-phase design notes, parity gates
and all board measurements are in [`../experiments/clef-omni.md`](../experiments/clef-omni.md).
The pack format is in [`../experiments/clef-pack.md`](../experiments/clef-pack.md) and the
Python oracle in [`../experiments/clef-oracle.md`](../experiments/clef-oracle.md).
