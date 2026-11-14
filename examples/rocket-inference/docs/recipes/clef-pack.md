# Clef pack (phase 5) — the AutoRound int8 checkpoint repacked for expert streaming

> **STATUS 2026-11-14: done and verified on the dev host. No board run.** The pack feeds
> the CPU backbone today (experts dequantized to f32); phase 6 reads the same records
> into NPU buffers.

The pack holds `moeshawky/clef-omni-int8-autoround` (weight-only 8-bit AutoRound,
group 128, symmetric) **as is** (Format A of `clef-phase0.md`): the checkpoint's int8
codes and fp16 scales, re-ordered to the `[N, K]` / `[N, K/128]` form the driver's
`I8Context::pack_weight_gw` / `matmul_gw` take, one expert per aligned record.

## Source checkpoint (measured 2026-11-14)

Download: `hf download moeshawky/clef-omni-int8-autoround --local-dir
/home/royalcat/models/clef-omni-int8-autoround --max-workers 2 --exclude "*.py"` (39 GiB
= 41.6 GB, 8 shards `model-0000N-of-00008.safetensors`, ~20 min; the `*.py` files are
not needed and not run). `joint_head.safetensors` is byte-identical to the original's.
Index sha256 `141e09a801390be0309618ef28871d2f3c6e321ee6d84b26ef2ac4dddda794b4`.

Tensor names, dtypes, shapes (layer 0; `N` = out features, `K` = in features):

| tensor | dtype | shape |
|---|---|---|
| `thinker.model.layers.L.mlp.experts.E.gate_proj.qweight` | I32 | `[512, 768]` = `[K/4, N]`, K=2048 N=768 |
| `...gate_proj.qzeros` | I32 | `[16, 192]` = `[K/128, N/4]` |
| `...gate_proj.scales` | F16 | `[16, 768]` = `[K/128, N]` |
| `...up_proj.*` | same as gate | |
| `...down_proj.qweight` / `.qzeros` / `.scales` | I32 / I32 / F16 | `[192, 2048]` / `[6, 512]` / `[6, 2048]` (K=768, N=2048) |
| `...self_attn.q_proj.*` | | `[512, 4096]` / `[16, 1024]` / `[16, 4096]` (N=4096, K=2048) |
| `...self_attn.k_proj.*`, `v_proj.*` | | `[512, 512]` / `[16, 128]` / `[16, 512]` (N=512) |
| `...self_attn.o_proj.*` | | `[1024, 2048]` / `[32, 512]` / `[32, 2048]` (K=4096, N=2048) |
| `...mlp.gate.weight` (router) | BF16 | `[128, 2048]` |
| `...input_layernorm.weight`, `post_attention_layernorm.weight` | BF16 | `[2048]` |
| `...self_attn.q_norm.weight`, `k_norm.weight` | BF16 | `[128]` |
| `thinker.model.embed_tokens.weight` / `thinker.lm_head.weight` / `thinker.model.norm.weight` | BF16 | `[152064, 2048]` / same / `[2048]` |

Facts: experts are separate modules (`mlp.experts.E.{gate,up,down}_proj`, 128 per layer, no
shared expert); there is **no `g_idx`** and no bias; attention projections are quantized
too; the router stays BF16.

Packing convention (AutoGPTQ-style, verified against the BF16 original):

* `qweight[kb, n]` is a little-endian int32 whose byte `j` is the **unsigned** code of
  `k = 4*kb + j` (`q` in `0..=255`); i.e. along K, K-major, output channel contiguous.
* `qzeros[g, n/4]`: four unsigned zero points per word along N, **stored as `zero - 1`**
  (AutoGPTQ v1); every word in the checkpoint is `0x7f7f7f7f` → zero = 128 everywhere
  (symmetric). The packer checks this for every tensor and refuses anything else.
* `scales[g, n]` fp16, may be negative.
* `w[n, k] = (q - 128) * scales[k/128, n]`; the pack stores the signed code `q - 128`
  (`(q ^ 0x80) as i8`) so `w = code * scale` with no offset.

Unpack error against the BF16 original (`clef-pack --compare-bf16`, layer 0, experts 0 and
127 and the four attention projections; every row also cross-checked against an
independent per-element reference over the packed words):

| module | max abs err / max abs w | mean abs err / mean abs w | rms rel | cosine |
|---|---|---|---|---|
| q_proj | 5.18e-2 | 1.08e-2 | 1.23e-2 | 0.999927 |
| k_proj | 3.82e-2 | 9.47e-3 | 1.11e-2 | 0.999940 |
| v_proj | 2.84e-2 | 1.03e-2 | 1.07e-2 | 0.999944 |
| o_proj | 1.59e-2 | 7.87e-3 | 8.65e-3 | 0.999963 |
| expert 0 gate / up / down | 4.86e-2 / 4.56e-2 / 1.76e-2 | 7.6e-3 / 7.6e-3 / 7.4e-3 | 9.6e-3 / 9.5e-3 / 8.0e-3 | 0.99995 / 0.99996 / 0.99997 |
| expert 127 gate / up / down | 4.22e-2 / 5.71e-2 / 2.49e-2 | 8.8e-3 / 8.5e-3 / 8.9e-3 | 1.09e-2 / 1.06e-2 / 1.05e-2 | 0.99994 / 0.99995 / 0.99995 |

That is ordinary 8-bit group-128 noise (≈1 % rms of the weight scale; the max column is the
worst single element after AutoRound's tuned rounding). A wrong zero point or byte order
shows up as 2.5-3.5 % rms (`zp = stored`) or garbage.

## Pack layout (format version 1)

`<out>/`:

| file | contents |
|---|---|
| `manifest.json` | format `clef-pack`, version, model geometry subset (layers 48, hidden 2048, experts 128, moe_intermediate 768, group 128, heads 32, kv 4, head_dim 128, vocab 152064), source (repo, index sha256, bits/group/sym/packing format), the expert record layout, every file's size + sha256, the resident tensor table (name, dtype, shape, absolute offset, bytes), the lm_head pointer, the copied files and the skipped source prefixes with tensor/byte counts |
| `layer_00.experts` … `layer_47.experts` | 128 experts per layer, see below (613,421,056 B each) |
| `resident.safetensors` | a standard safetensors file (`mmap`-able by `st::SafeFile`; dtypes BF16/F16/I8) |
| `lm_head.safetensors` | `thinker.lm_head.weight` BF16 `[152064, 2048]` (0.58 GiB; only option-token rows are read) |
| `config.json`, `tokenizer.json`, `tokenizer_config.json`, `joint_head.safetensors`, `joint_head_config.json` | verbatim copies (the pack dir is a complete model dir for `ClefModel`) |

`layer_NN.experts` (all little-endian, **everything 4096-aligned**):

```
[0, 4096)                  header: magic "CLEFXPK1", then u64 fields
                           version, layer, experts, hidden, inter, group, stride,
                           data_start, off_gu_codes, off_down_codes, off_gu_scales,
                           off_down_scales, record_bytes, reserved   (rest zero)
[4096 + e*stride, +stride) expert e's record (one aligned read per expert)
stride = round_up(record_bytes, 4096)      # 4,792,320 = 1170 pages: no padding at 2048/768/128
record (offsets relative to the record start, 64-byte aligned sections):
  0          gu_codes    i8   [1536, 2048]  gate rows 0..768 then up rows 768..1536, N-major ([N,K])
  3,145,728  down_codes  i8   [2048, 768]
  4,718,592  gu_scales   f16  [1536, 16]    [N, K/128]
  4,767,744  down_scales f16  [2048, 6]
```

`w = code * scale` with the scale of group `k/128` of row `n`. The codes are exactly the
`[N, K]` int8 matrix `pack_weight_gw` takes (gate|up fused as N = 1536; down N = 2048,
K = 768); scales are `matmul_gw`'s `b_scale`. One expert = 4.57 MiB.

`resident.safetensors`: `thinker.model.embed_tokens.weight` (BF16), `thinker.model.norm.weight`,
per layer `input_layernorm.weight`, `post_attention_layernorm.weight`,
`self_attn.{q,k}_norm.weight`, `mlp.gate.weight` (BF16, copied bytes) and for
`self_attn.{q,k,v,o}_proj` two tensors `<module>.codes` (I8 `[N, K]`) and
`<module>.scales` (F16 `[N, K/128]`). 626 tensors, 1.46 GiB.

Total on disk: 29.68 GiB in 55 files (experts 27.42 GiB, resident 1.46 GiB, lm_head
0.58 GiB, head/tokenizer/config 0.21 GiB). Skipped (not needed for text; phase 8 reads
media from the source): `thinker.visual` (351 tensors, 1.08 GB), `thinker.audio_tower`
(525, 1.30 GB), `talker.*` (8,037 tensors, 6.65 GB), `code2wav.*` (230, 0.43 GB).

## Commands

```sh
B=$CARGO_TARGET_DIR/release/rocket-inference     # hidden subcommand, not in --help
cargo build --release -p rocket-inference --no-default-features

# pack (streaming: one layer at a time, <= 2 layer buffers of 0.6 GB + mmapped pages)
$B clef-pack --src /home/royalcat/models/clef-omni-int8-autoround \
    --out /home/royalcat/models/clef-pack-g128 --threads 8
# re-read every expert O_DIRECT, compare with a fresh unpack, re-hash all files
$B clef-pack --src ... --out ... --verify-only        # or --verify after packing
# quantization error vs the BF16 original (no packing)
$B clef-pack --src ... --out x --compare-bf16 /home/royalcat/models/clef-omni --layers 0..1

# use it: --pack-dir is --model-dir (a pack is detected by its manifest.json)
$B decide --pack-dir /home/royalcat/models/clef-pack-g128 --request '{...}'
$B serve  --pack-dir /home/royalcat/models/clef-pack-g128 --port 8399
```

`--layers A..B` packs/verifies a range (a partial pack gets no manifest). `--tokenizer-from`
supplies `tokenizer.json`/`tokenizer_config.json` when the source lacks them (this
source has them).

Reader (`src/clef/pack/reader.rs`): `PackDir::open` (manifest validation, per-file sizes,
every layer header), `read_expert(layer, e, &mut AlignedBuf)` (one `pread` of `stride`
bytes, O_DIRECT with 4 KiB-aligned buffers; falls back to buffered +
`posix_fadvise(DONTNEED)` if the filesystem refuses, `CLEF_PACK_NO_DIRECT=1` forces the
fallback), `with_expert` (thread-local reusable buffer), `resident()`/`tensor()`,
`dequant_linear`. `PackedExperts` is an `ExpertSource` (codes × scale → f32);
`PackedWeights` is the backbone's `WeightSource` (attention dequantized per layer from
the resident file). `Backbone::open_pack(dir)` / `ClefModel::load(pack dir)` use them.

## Measured (dev host: 32 threads machine, `--threads 8`, btrfs NVMe, source in page cache)

* pack: **93 s** (layers ~2 s each, the writer thread overlaps hash + write with the next
  layer's unpack; resident 5.5 s), 29.68 GiB; verify-only **19-20 s** (55 sha256, 6144
  O_DIRECT expert reads vs fresh unpack, 626 resident tensors, lm_head). A cold source
  adds the 39 GiB read.
* end to end (packed experts → f32 on the CPU, 8 threads): 295 tokens 20.9 s, 256 tokens
  20.8 s, 800 tokens 27.6 s per record; 480 MiB anon after load, 759 MiB after a record.
* probabilities vs the f32 BF16-source oracle (`clef_pack_answers_match_the_oracle_top1`,
  `cargo test --release -p rocket-inference --no-default-features
  clef_pack_answers_match_the_oracle_top1 -- --ignored --nocapture`; needs the dumps of
  `clef-oracle.md`): top-1 identical for all 7 questions; checkout-errors max |dp|
  0.0100 / mean 0.0047, invoice 0.0002 / 0.0001, long-incident 0.0136 / 0.0043; all
  24 options max 0.0136, mean 0.0035 (the AutoRound card measured max 0.044, mean 0.006 on
  its own records). E.g. checkout-errors technical 0.8227 (oracle 0.8327, card 0.8298),
  urgency 2 0.9356 (0.9341), outage 0.9588 (0.9552).

## Tests (no checkpoint needed)

`clef::pack::{format,autoround,tests}`: the real record geometry and a ragged one,
header round trip + corruption rejection, int32 8-bit unpack on hand-built words
(extremes, negative scales, the stored-zero convention, non-symmetric refusal),
`PackedExperts`/`PackedWeights` vs a naive dequant of a synthetic 2-layer 4-expert
AutoRound checkpoint, the backbone on the pack bit-identical to the backbone on the
dequantized weights, `--verify-only` catching a flipped byte, manifest validation.
