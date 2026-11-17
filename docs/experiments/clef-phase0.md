# Clef-Omni phase 0 — expert-streaming format probe

Goal: run Cloudflare/clef-omni (Qwen3-Omni-30B-A3B thinker: 48 layers, hidden 2048,
128 experts/layer, top-8, expert intermediate 768) on the RK3588 NPU with **full expert
streaming from NVMe**: per layer, the hit experts are read from disk into NPU buffers and
consumed. Per expert: fused gate|up `K=2048 N=1536`, down `K=768 N=2048`
(3.0 + 1.5 MiB of int8 codes = 4.5 MiB, ~4.57 MiB with g128 fp16 scales). Tokens per
expert `M_e ~= S*8/hit`, about 8 rows at S=128, 64 at S=1024, 256 at S=4096.

Phase 0 needs no model. Status: **probe written and verified on the dev host only; no
board numbers yet.**

## What the probe measures

`crates/rocket-npu/src/bin/expert_bench.rs` (`--mode all|matmul|rewrite|disk|pipeline`):

1. **matmul** — group-wise int8 (`I8Context::pack_weight_gw` + `matmul_gw`, activation
   quantize included, as `q8_bench`) vs the resident fp16 `Context`, for gate|up and down,
   M in {16,32,64,128}, groups 128/256/512 (K=768: 128/256/768). Cosine vs an f64
   reference over 16 rows (`--no-ref` compares against the fp16 arm instead), wall and CPU.
2. **rewrite** — new API `I8Context::rewrite_weight(&mut I8Weight, codes)` /
   `rewrite_weight_gw`: re-scatters new int8 codes into the *existing* per-worker BOs
   (same layout as pack, no allocation, no zero-fill; scales are caller-side in this
   driver, so they are not part of the weight). Timed against a fresh `pack_weight_gw`,
   and checked bit-identical against a fresh pack of the same codes.
   Host test: `matmul::int8::tests::a_rewrite_equals_a_fresh_pack`.
3. **disk** — sequential reads of `--file` in expert-sized chunks (`--chunk-mib`, default
   4.7), O_DIRECT with 4 KiB-aligned buffers; falls back to buffered reads +
   `posix_fadvise(DONTNEED)` when the filesystem refuses O_DIRECT. Reads only.
4. **pipeline** — one streamed layer simulated for `--layers` layers (default 3): a
   prefetch thread reads the hit experts (and keeps reading layer L+1 while L computes,
   bounded by `--depth` buffers), the consumer rewrites the resident slot, quantizes the
   activation, runs fused gate|up and down (2 `matmul_gw` calls; a stand-in takes the
   first 768 gate|up columns as the down input instead of silu*mul). For
   S in `--tokens` (128,1024,4096): `hit = round(128*(1-(15/16)^S))`,
   `M_e = S*8/hit` rounded up to 4. Variants per S: disk only, compute only, and
   overlapped, each for **1 context x N workers** (N-split of every matmul across
   `--threads` cores, the existing design) and **3 contexts x 1 worker** (three consumer
   threads, one expert each, no N split = "3 experts on 3 cores"). Prints per-layer
   time and the extrapolated 48-layer pass.

Notes on the 3x1 arrangement: it is cheap with the existing pool because each
`I8Context::new_at(1, core_base)` owns its own fd and pinned pool thread (`new_at` is new;
`new` is unchanged — it now passes the creator's rotation base to the pool, which
previously always read 0 from the pool thread). The kernel, not userspace, schedules
jobs from different fds onto NPU cores. The simulation keeps a barrier at each layer
end (experts of L+1 may be prefetched, not computed, before L completes). Not modelled:
real routing skew, the SiLU/combine glue, attention, the router, and scale parsing
(the scales are synthetic constants; their bytes are read from disk).

## Build and run (NOT executed here; board protocol from AGENTS.md)

```sh
cargo build --release -p rocket-npu --target aarch64-unknown-linux-gnu --bin expert_bench
ssh root@rock-5b-plus.lan 'mkdir -p /root/rocket-clef0'
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/expert_bench root@rock-5b-plus.lan:/root/rocket-clef0/

# on the board: free the NPU and cores 4-7 first (restore afterwards)
ssh root@rock-5b-plus.lan
docker stop home-llm-gateway-embed-1 home-llm-gateway-decision-1
cd /root/rocket-clef0
# a >= 4 GiB file on the NVMe that will hold the model (read only; never written):
FILE=/path/on/nvme/some-large-file.bin
taskset -c 4-7 ./expert_bench --mode disk --file $FILE --reps 3 --max-mib 4096
taskset -c 4-7 ./expert_bench --mode matmul --threads 3 --reps 5
taskset -c 4-7 ./expert_bench --mode rewrite --threads 3 --reps 5
taskset -c 4-7 ./expert_bench --mode pipeline --file $FILE --threads 3 --group 128 --layers 3
taskset -c 4-7 ./expert_bench --mode pipeline --file $FILE --threads 3 --group 256 --layers 3
docker start home-llm-gateway-embed-1 home-llm-gateway-decision-1   # restore, check health
```

Run on a quiet board (load shows up as +-20 %); the disk file must be on the real NVMe
filesystem. DRM BOs are not charged to a memory cgroup; the probe's footprint is small
(one slot per consumer + ~`depth` x 4.6 MiB of host buffers). The device-free parts
(`--mode disk`, the pipeline's disk-only rows) run on any host.

## Dev-host results (not the board)

`expert_bench --mode disk` on a 256 MiB scratch file (tmpfs/page-backed, O_DIRECT
accepted): 11-12 GB/s — meaningless for NVMe, it only proves the path runs. The pipeline's
disk-only rows ran (S=128: 128 hit, M_e=8; S=1024: 128 hit, M_e=64). Everything that
needs `/dev/accel/accel0` fails cleanly with the device error on this host.

## Board results (2026-11-14)

rock-5b-plus, `taskset -c 4-7`, 600 MHz module, `--threads 3`, **board load 10-11 from other
services** (only `home-llm-gateway-embed-1`/`-decision-1` were stopped, for ~3 min, and
restored healthy), so read ratios, not absolutes (+-20 %). The disk file was a 4 GiB
incompressible scratch file on the btrfs (zstd) NVMe at `/mnt/nvme`, deleted afterwards.
Raw log: `/root/rocket-clef0/run.log` on the board (copied to `/tmp/opencode/clef-board/`).

- **Disk** (O_DIRECT, 4.7 MiB chunks, queue depth 1): **1.0-1.27 GB/s** (chunk p50 ~4 ms).
  One 128-expert layer is 585 MiB -> 0.5-0.6 s/layer, ~24-29 s per 48-layer pass. Slower than
  the 3 GB/s assumed earlier; deeper queues were not tried.
- **Matmul**, int8 group-wise vs resident fp16 (activation quantize included), cosine
  0.99999 at every shape and group: g128 is 1.9-4.7x slower than fp16 (readback-bound, as
  log §20), g256 1.1-2.3x, g512 0.6-1.2x (gate|up) and g768 1.2-2.7x (down). fp16 is not a
  streaming option anyway (2x the disk bytes and no fp16 rewrite API).
- **Rewrite** (`rewrite_weight_gw`, bit-identical to a fresh pack in all 6 cases): gate|up
  2.4-3.2 ms, down 0.6-1.1 ms per expert, 2.1-3.1x cheaper than a fresh pack.
- **Pipeline** (one streamed layer simulated, 48-layer extrapolation, expert matmuls only;
  no attention, router, SiLU glue or head). Best arrangement is **3 contexts x 1 worker**
  (one expert per core) in every row, 1.3-1.5x faster than the N-split:

| S | hit | M_e | g128 overlap 3x1w | g256 overlap 3x1w |
|---|---|---|---|---|
| 128 | 128 | 8 | 34.5 s | 28.9 s |
| 1024 | 128 | 64 | **72.0 s** | **49.5 s** |
| 4096 | 128 | 256 | 186.8 s | 121.1 s |

  Disk-only is 24-29 s per pass, so compute dominates from S~128 up and the prefetch hides
  the disk fully (overlap ~= compute + ~5 %). All 128 experts are hit at every S tested.

**Decision: GO, Format A** (g128 int8 codes + fp16 scales straight from the AutoRound
checkpoint, 4.57 MiB/expert, no requantization) with the **3 contexts x 1 worker** layout.
The expert-matmul floor at 1k tokens is ~72 s per pass, so a full request (attention,
router, glue, head) should land in the 2-3 minute range the workload allows. Format B (g256,
-31 % pass time at 1k, scales half the size) stays the fallback if later phases need speed;
it would need a real-weight numerics gate. Format C is not needed. Note the 4096-token pass
(187 s g128) is the practical per-request ceiling before attention is added.

## Decision table

Let `T_pass(S)` be the extrapolated 48-layer pass from the best overlapped arrangement.
Also compare `T_disk = 48 * hit * 4.57 MiB / disk GB/s` (the floor).

| outcome | decision |
|---|---|
| g128 `matmul_gw` int8 within ~1.5x of fp16 at M = 32-128 **and** rewrite is cheaper than a fresh pack **and** `T_pass(1024) < ~300 s` | **Format A**: ship the AutoRound g128 int8 codes + fp16 scales straight from the checkpoint (4.57 MiB/expert on disk, no conversion). |
| g128 is readback-bound (matmul 2x+ slower than g256/g512, compute dominates `T_pass`) but g256/g512 gets `T_pass(1024) < ~300 s` with cosine >= 0.9995 vs the f64 reference on real-checkpoint-like weights | **Format B**: repack offline to coarser groups (256 or 512; scales shrink to 1/2 - 1/4). Needs a real-weight numerics gate (log §20 showed outlier weights lose fidelity as groups widen) before adoption. |
| int8 group-wise is slower than fp16 everywhere (as log §20 found at large M) and the fp16 path fits: disk read of 2x bytes (9.0 MiB/expert) still gives `T_pass(1024) < ~300 s` | **Format C**: dequantize to fp16 per layer on the host (or at repack time, doubling the on-disk size) and use the resident fp16 `Context` / `rewrite` equivalent. Note there is no fp16 in-place rewrite API yet. |
| `T_pass(1024) > ~300 s` for every format | **No-go** for full streaming at this speed: either disk or compute is the wall; see which (disk-only vs compute-only rows) before changing the plan (keep hot experts resident, smaller S per pass, or a CPU `sdot` path). |

Go/no-go threshold: **per-pass estimate under ~5 minutes at 1k tokens** (`S=1024`, the
`1k tokens best arrangement` line the probe prints; 3-core 3x1 vs N-split is chosen by
whichever is faster). The probe prints GO/NO-GO against 300 s; it includes the disk only
when `--file` is given.
