# Clef-Omni layer-streaming oracle (phase 1)

> **STATUS: VERIFIED 2026-11-14** (transformers 5.23.0, torch 2.15.0+cpu, 8 threads, dev host
> 62 GB RAM). Checkpoint at `/home/royalcat/models/clef-omni` (66 GiB / 70.74 GB on disk,
> 15 shards), venv at `/home/royalcat/models/venvs/clef` (1.2 GB).

`tools/clef_reference.py` computes Cloudflare/clef-omni (Qwen3-Omni-30B-A3B thinker
text decoder + `JointSchemaHead`) for text-only SystemOne records on a CPU host by
streaming ONE decoder layer at a time in float32. Peak weights in memory: one layer
(~2.4 GB f32 tensors + ~1.2 GB for the fused `gate_up_proj` copy) plus the activations.

## Setup

```sh
# venv with CPU torch + transformers >= 5 (Qwen3OmniMoe), safetensors, huggingface_hub, numpy
python3 -m venv /home/royalcat/models/venvs/clef
/home/royalcat/models/venvs/clef/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/home/royalcat/models/venvs/clef/bin/pip install 'transformers>=5.10' safetensors huggingface_hub numpy
# checkpoint: all 15 shards (each shard mixes layers) + tokenizer/config/joint head, ~70.5 GB
hf download Cloudflare/clef-omni --local-dir /home/royalcat/models/clef-omni --max-workers 2
# (do not exclude shards: each one mixes layers)
```

`tools/clef_joint_schema_model.py` is a verbatim copy of the release's
`joint_schema_model.py` (fetched 2026-11-14, 719 lines, read before use: pure model
code; its only I/O is optional media decoding/URL fetch, which this oracle never reaches).
The download dir is untrusted: the script is run with `python -I` and imports the local
copy by explicit file path; do not use the model dir as CWD.

## Run

```sh
cd /home/royalcat/projects/burn-rocket
/home/royalcat/models/venvs/clef/bin/python -I examples/rocket-inference/tools/clef_reference.py \
    --model-dir /home/royalcat/models/clef-omni \
    --requests examples/rocket-inference/tools/clef_fixtures.json \
    --out-dir /tmp/clef-dump --only checkout-errors,invoice --selfcheck
```

Flags: `--dump-layers all|0,1,23,47` (final layer always dumped; default `all`),
`--max-tokens 512` (guard; the `long-incident` fixture is 800 tokens and needs `--max-tokens 1024`),
`--threads 8` (capped at 8), `--selfcheck`, `--only ids`.

Measured cost (warm page cache after the first run; the download itself took ~28 min at
~40 MB/s with 2 workers): every record streams the whole 70 GB checkpoint once, **~1.1-1.7 s
per layer (load ~1.2 s, compute 0.2-0.5 s at 256-800 tokens)**, i.e. **~60-73 s per record**
(checkout 295 tokens: 65.6-73.1 s; invoice 256: 58.0 s; long-incident 800: 59.8 s).
Peak anonymous RAM (`RssAnon`, sampled after each layer) **1.35 GiB (295 tokens), 2.5 GiB
(800 tokens)** with the per-layer mmap handles dropped (`Shards.close`). Note: `ru_maxrss`
of an earlier version reached 43 GB because touched mmap'd file pages count toward RSS; that
is reclaimable page cache, not allocation. Dumps: 119 MB (295 tokens, all 48 layers),
15 MB (256, layers 0,1,23,47), 47 MB (800, layers 0,1,23,47). Records are processed
sequentially, each re-streaming the weights.

## What it does

1. `encode_record` (release code) -> token ids + question/option spans.
2. Embedding rows read via `safe_open().get_slice` (only the needed token rows).
3. For each of 48 layers: tensors read via `model.safetensors.index.json`, experts fused as
   HF stores them (`gate_up_proj[e] = cat(gate_proj, up_proj)` on the output dim,
   `down_proj[e]`), a real `Qwen3OmniMoeThinkerTextDecoderLayer` is built on the meta
   device and filled with `load_state_dict(assign=True)`, run in f32 with an additive
   causal mask and HF's own rotary module (text-only: all three mRoPE streams = position).
   A forward hook on the router recomputes softmax/top-8/renorm independently and asserts
   it matches the layer's routing.
4. Final RMSNorm (`thinker.model.norm.weight`), then `JointSchemaHead` unchanged in f32
   with `output_embedding_weight` = `LazyRows` over `thinker.lm_head.weight` (reads only
   the token ids used by option spans).
5. `--selfcheck`: layers 0 and 47 are also computed by `manual_layer` (plain torch, no HF
   code, documented math in its docstring: RMSNorm, per-head QK-norm, rotate-half RoPE for
   equal positions, GQA 32/4, causal softmax attention, router softmax/top-8/renorm,
   SwiGLU experts). Reported in the summary JSON under `selfcheck`.
   Measured (checkout-errors, 295 tokens, f32 HF layer vs plain-torch layer, same input):

   | layer | max abs diff | diff / max abs output | router ids equal | router weight max abs diff |
   |---|---|---|---|---|
   | 0 | 2.4e-7 | 1.8e-7 | yes | 2.4e-7 |
   | 47 | 6.1e-5 | 1.7e-7 | yes | 2.4e-7 |

   (layer 47 outputs reach ~350, so 6e-5 is f32 rounding, ~1.7e-7 relative). No divergence.

## Dump format (`<out-dir>/<record id>.npz` + `<id>.json`)

All float arrays are float32, little-endian, `np.load`-able (`np.savez`, uncompressed).
`T` = token count, `H` = 2048, `L` = layer index 0..47.

| key | shape / dtype | meaning |
|---|---|---|
| `input_ids` | `[T]` int64 | token ids from `encode_record` |
| `embeddings` | `[T,H]` | `thinker.model.embed_tokens` rows (before layer 0) |
| `hidden_L{L}` | `[T,H]` | output of decoder layer L (for L in `dump_layers`; layer 47 always) |
| `router_ids_L{L}` | `[T,8]` int32 | top-8 expert ids per token, descending router prob (ties as `torch.topk`) |
| `router_w_L{L}` | `[T,8]` | softmax-over-128 then top-8 renormalised weights (sum to 1) |
| `final_hidden` | `[T,H]` | = layer 47 output (pre final norm) |
| `final_normed` | `[T,H]` | after final `RMSNorm(thinker.model.norm.weight, eps 1e-6)`; the head input |
| `logits_{qid}` | `[n_opt]` | head logits for question `qid`, in `option_ids` order |
| `probs_{qid}` | `[n_opt]` | softmax of the logits |

Router dumps exist for all 48 layers regardless of `--dump-layers`.

`<id>.json`: `answers` (SystemOne answer format via `systemone_answer`), `usage`,
`questions` (`id`, `type` 0 noul/1 choice/2 score, `question_span`, `option_spans`
as token index half-open ranges, `option_ids`), `dump_layers`, `selfcheck`, `wall_s`,
`maxrss_mib`.

## Model math notes for the Rust port

See `manual_layer` in the script. Checkpoint layout facts: experts are stored
per-expert (`thinker.model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight`), the
router is `mlp.gate.weight [128,2048]`, no shared expert, no biases; `thinker.lm_head`
is NOT used for logits (ClefModel replaces it with Identity): its rows serve only as the
lexical option embeddings (mean over each option span's token ids).
Head: `joint_head_config.json` = hidden 2048, width 1024, routing_layers 2, layers 4,
heads 16, feedforward 4096.

## Fixtures

`tools/clef_fixtures.json`: 8 text-only records (`checkout-errors` and `invoice` are the
model-card examples; choice, score, noul, mixed, JSON state, and a ~400-token state).
Model card values for `checkout-errors` (int8 build): technical ~0.83, urgency score
~1.88, outage ~0.95. **Oracle (f32 streaming of the BF16 weights)**: technical **0.8327**
(billing 0.1673), urgency score **1.8917** (P0/1/2 = 0.0425/0.0234/0.9341), outage noul
**0.9552**: matches the card within its rounding. Two further checks:
`invoice` -> status overdue 0.9773 (paid 0.0141, draft 0.0085), large 0.9860;
`long-incident` (800 tokens) -> owner payments 0.8202, severity score 2.2751, double_charge
0.9674, owners_assigned 0.0169 (all plausible for the text).
Determinism: two independent runs of `checkout-errors` give byte-identical arrays for all
110 keys of the shared dump (the second run used the handle-dropping code and a subset of
`--dump-layers`; compared on the keys both contain).
