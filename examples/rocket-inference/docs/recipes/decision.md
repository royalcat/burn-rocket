# vllm-sr d3 decision recipes

[vllm-sr Decision 3.0](https://huggingface.co/vllm-sr/d3-lite) (`d3-code-readout-v1`)
is the typed decision model: a Qwen3.5-0.8B backbone plus a 255-code readout head.
A request is a `state` and a set of typed `questions` (`choice`, `noul`, `score`);
the answer is one non-causal forward pass per question and a softmax over the
codes. Nothing is generated (`usage.output_tokens` is always 0), so both the CLI
and the server prefill only.

Checkpoint: `/root/models/d3-lite` (1.7 GB safetensors; backbone bf16, read as
f32, plus `readout.safetensors`). d3-nano runs through the same code but is
validated on the dev host only; the numbers here are d3-lite.

## Decide

```sh
M=/root/models/d3-lite
B=./rocket-inference
CASES=cases.json          # JSON array of {"name", "state", "questions"[, "images"[,
                          # "videos"]]}; local paths for media

# CPU f32 body
$B decide --model-dir $M --requests $CASES --out out-cpu.json \
    --max-state-tokens 8192 --device cpu

# text backbone projections packed as fp16 resident NPU weights
$B decide --model-dir $M --requests $CASES --out out-npu.json \
    --max-state-tokens 8192 --device npu
```

The placement line reports the split:

```
placement: text=CPU(f32) readout=CPU(f32)
placement: text=NPU(fp16) readout=CPU(f32)
```

`text` is the **only** NPU stage of this family. The vision tower used by image
and video questions, the attention inside the backbone and the readout head all
stay on the CPU in both configurations; there is no `--text-attn-device`,
`--vision-device` or `--decode-device` flag for d3. The NPU config packs the
text projections and runs the prefill GEMMs on them; it does not move the tower
or the attention.

`--max-state-tokens` is the per-question prompt cap (default 32768). The long
policy case runs three ~2,100-token prompts over the same state, so the cap
applies per question, not to a whole case; 8192 is a safe board setting.

The output is the official shape
`{"model", "answers", "usage"}`; `--requests` wraps the cases as
`{"schema": "d3-decision-v1", "model", "cases": [{"name", "response"}]}`.

## Serve

```sh
ROCKET_MEM_LOG=1 $B serve --model-dir $M --device npu --port 8399 --max-tokens 8192
```

`/v1/systemone` takes the same request object (media as `data:` URLs), `/health`
and `/v1/models` are always available. `--max-tokens` is the per-question prompt
cap. The server logs one line per request with `questions`, `tokens` and
`compute_s`, and (with `ROCKET_MEM_LOG=1`) the `npu scratch` line after load and
after every request. The CLI commands install no tracing subscriber, so
`ROCKET_MEM_LOG` prints nothing in `decide` runs; use `serve` when the BO split
matters.

## Board (d3-lite)

Board: Rock 5B+, 8 cores (A76 CPUs 4-7 via `taskset -c 4-7`), kernel
7.1.8-edge-rockchip64, patched `rocket` module at 600 MHz
(`rocket_npu_clk_hz=600000000`; dmesg reports `core N NPU clk -> 600000000 Hz
(reads back 600000000)`). Load average 7.8-10.7 from the other board services;
the live embedding service `home-llm-gateway-embed-1` was stopped for the runs
(see below). Runs are one process per config, two samples each (taken before the
service was restored); outputs are bit-identical between repeats. The deployed
copy is `/root/rocket-inference-decision/` (binary, fixtures and run outputs).
The video CPU `user` column has one valid sample (see the note below the table).

Fixtures (board paths):

- text: `choice-routing-short`, `noul-receipt-short`, `choice-policy-long` from
  `tools/decision_fixtures.json` (5 questions, 6,546 prompt tokens; the long case
  has three questions).
- image: `image-receipt-total` from `tools/decision_fixtures_images.json`, with
  `images: ["/root/models/d3-lite/assets/example-receipt.png"]` (1,029 tokens).
- video: `video-example` from `tools/decision_fixtures_videos.json`, with
  `videos: ["/root/models/d3-lite/assets/example-video.mp4"]` (1,720 tokens over
  2 questions).

`compute` is the CLI's `decided ... in` time for the whole case set (it includes
image/video decode and the vision tower); `wall` adds the model load (4-9 s) and
teardown. `user` is process CPU-seconds; peak anon is `RssAnon` sampled every
50 ms, with peak RSS in parentheses (the RSS includes the 1.7 GB checkpoint's
file pages).

| input (questions) | placement | tokens | compute | wall | user CPU | peak anon (RSS) |
|---|---|---|---|---|---|---|
| text (5) | CPU f32 | 6,546 | 197 s (194-200) | 205 s (204-206) | 453 s (384-522) | 3,999 (4,586) MiB |
| text (5) | NPU fp16 | 6,546 | 113 s (111-116) | 122 s (119-125) | 144 s (141-146) | 2,505 (3,001) MiB |
| image (1) | CPU f32 | 1,029 | 56.7 s (54.5-58.9) | 63.0 s (60.9-65.1) | 130 s (127-132) | 3,468 (4,598) MiB |
| image (1) | NPU fp16 | 1,029 | 46.3 s (45.2-47.4) | 56.2 s (53.7-58.7) | 97.3 s (97.2-97.4) | 2,497 (2,993) MiB |
| video (2) | CPU f32 | 1,720 | 62.1 s (60.2-64.0) | 68.4 s (66.6-70.3) | 129 s | 3,354 (4,584) MiB |
| video (2) | NPU fp16 | 1,720 | 40.8 s (38.5-43.1) | 49.1 s (47.0-51.3) | 67.3 s (66.6-68.0) | 2,513 (3,009) MiB |

The video CPU `user` value is one sample: a second run's rusage reported 424 s of
user time for 66.6 s of wall on a 4-CPU affinity (6.4 core-seconds per wall
second, which is impossible); its compute/wall/anon are inside the ranges above
and are kept, the CPU reading is discarded. The other configs have two readings
each; the text CPU user range (384-522 s) is the widest.

NPU memory (the same binary's `serve`, after load and after a 128-token
`/v1/systemone` request, `ROCKET_MEM_LOG=1`): 96 resident NPU weights,
**1,006.7 MiB** of weights BO; matmul BO **93.7 MiB** after load, trimmed to
**26.4 MiB** BO + 6.3 MiB host after the request (`ROCKET_KEEP_MB=64`); attention
BO/host **0** (there is no NPU attention stage). `RssAnon` is 943.6 MiB after
load and 1,034.3 MiB after the request. The BOs are not charged to the process,
so the anon peaks above exclude the ~1 GiB of NPU weights. The 128-token server
request computed in 1.98 s (64.7 tok/s) with the model already resident.

Parity against the f32 reference (`/tmp/opencode/d3/ref_*_lite_f32.json`, the
same case names; gate 2e-3). Prompt-token counts match exactly in every arm.

| input | placement | top-1 | max \|Δp\| |
|---|---|---|---|
| text | CPU f32 | 5/5 | 1.4e-06 |
| text | NPU fp16 | 5/5 | 3.4e-04 |
| image | CPU f32 | 1/1 | 7.9e-06 |
| image | NPU fp16 | 1/1 | **1.63e-03** |
| video | CPU f32 | 2/2 | 1.40e-02 |
| video | NPU fp16 | 2/2 | 1.45e-02 |

The video arms miss the 2e-3 probability gate although both top-1 answers are
correct. The reference was produced on the dev host; the board's `ffmpeg`
7.1.5 decodes the sampled frames with a per-channel difference of up to 2/255
against the dev-host `ffmpeg` n9.0.2 (verified bit-identical to the OpenCV
decode the reference used): 67 % of the RGB bytes differ, mean 0.68/255, and the
vision tower turns that into a ~1e-2 probability shift. The same-board
CPU-vs-NPU video delta is only 4.6e-4, so the miss is the decoder version, not
the Rust body or the NPU pack. The image fixture's decode matches across the two
environments, which is why its deltas stay near the gate.

Reading the table:

- **Text is the NPU win**: 1.7x the compute (197 -> 113 s) and ~3x less user CPU
  (453 -> 144 s) at 37 % lower peak anon; its prefill is projection-heavy.
- **Image/video gain less** because the vision tower and the media decode are
  CPU work in both configs: image 56.7 -> 46.3 s (1.2x) and video
  62.1 -> 40.8 s (1.5x), while user CPU still drops 1.3-1.9x.
- **Memory**: the CPU body is f32 and ~2.9 GiB resident; the NPU config keeps
  ~1.0 GiB anon (embeddings, vision tower, readout) plus the ~1.0 GiB weights
  BO. Peak anon drops from 3.3-3.9 GiB to ~2.5 GiB on every input class.
- **Board noise**: the other services keep the load at ~8-10, so wall/CPU spread
  by ±20-30 % between samples. The runs in the table were taken with the
  embedding service stopped; the two samples of each config are the pre-restore
  pair.

Service stop/restore and free memory:

- Before: `home-llm-gateway-embed-1` `running healthy` (container 3.48 GiB),
  5,517 MiB host available. Stopped with
  `docker stop home-llm-gateway-embed-1` (it stayed `exited` for the whole run;
  Komodo did not restart it, so the compose stop was not needed). Port 8383 was
  no longer listening; available memory rose to **9,068 MiB** (used
  10.0 -> 6.6 GiB).
- During the runs `/usr/bin/time` is not installed on the board, so peak
  anon/RSS was sampled from `/proc/<pid>/status`; the largest run peaked at
  3.9 GiB anon, well inside the 8.7-9.1 GiB available.
- Restored with `docker start home-llm-gateway-embed-1`; `healthy` after ~65 s
  (start period), then one `/v1/embeddings` request from inside the container
  returned 200 in 0.32 s (4 tokens, `compute_s=0.294` in the service log). The
  service then resumed its normal production traffic (49-50-frame video
  requests), so later requests queue behind it. Nothing else was stopped or
  restarted.

Notes:

- The `decide` CLI keeps one process per run: the text samples include the
  6,307-token long case, so their ~3.4 min is dominated by it. For interactive
  latency, serve once: the 128-token short case is 1.98 s of compute with the
  model resident (the one-shot CLI pays 4-9 s of load on every invocation).
- Attention is CPU in both configs (the d3 family has no NPU attention stage).
  The NPU BO line's `attn_bo_mib=0` is expected, not a disabled feature.
- Video parity is decoder-version sensitive; compare board outputs against a
  reference produced with the same `ffmpeg`, or accept the top-1-only gate for
  cross-host video checks. Text and image are stable at the 2e-3 gate.
