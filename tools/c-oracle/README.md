# C oracle + dtype parity harness

`tools/c-oracle/` holds the host-side **C oracle** used to check the pure-Rust
`crates/rocket-npu` port against the reference `librocketnpu`, plus the
`parity.py` harness that drives it.

The oracle links the reference library but never opens the NPU device: it only
calls the pure planners (`rocket_matmul_plan*`) and the regcmd emitters
(`gen_matmul_*`), so it runs on any x86-64 or aarch64 host.

## License

The C reference library is **GPL-3.0-or-later** (`rocket-userspace`). It is
**not vendored** into this repository: only the small oracle driver
(`oracle.c`, GPL-3.0-or-later) lives here. Check the reference tree out
separately and keep its license intact.

## Pinned reference

The parity harness is pinned to:

| field | value |
|---|---|
| repo | `rocket-userspace` (the C `librocketnpu`) |
| commit | `b43a9ae1b6364ab0833747fac105b8474c1c0482` (2026-09-26) |
| local checkout | `/tmp/opencode/ref-rocket-userspace` |
| host build | `/tmp/opencode/ref-rocket-userspace/build-host` |

Sync the tree from the board (the board's copy is the one the deployed driver
was validated against):

```sh
rsync -a root@rock-5b-plus.lan:/root/npu-poc/rocket-userspace/ \
    /tmp/opencode/ref-rocket-userspace/
cd /tmp/opencode/ref-rocket-userspace
git rev-parse HEAD   # expect b43a9ae1b6364ab0833747fac105b8474c1c0482
```

Build it host-side (no NPU device needed):

```sh
cmake -S . -B build-host -DROCKETNPU_BUILD_TESTS=OFF
cmake --build build-host -j
# -> build-host/librocketnpu.a
```

## Build the oracle

```sh
tools/c-oracle/build.sh <src> <build-host> [out]
# e.g.
tools/c-oracle/build.sh /tmp/opencode/ref-rocket-userspace \
    /tmp/opencode/ref-rocket-userspace/build-host /tmp/opencode/c-oracle
# -> /tmp/opencode/c-oracle/oracle
```

`<src>` supplies the internal headers (`npu_matmul.h`, `npu_hw.h`, ...) and
`<build-host>` supplies `librocketnpu.a`. Override the compiler with `CC=`.

### Subcommands

```
oracle plan         M K N                        # fp16        rocket_matmul_plan
oracle plan-int8    M K N                        # int8 dense  rocket_matmul_plan_int8
oracle plan-int8-gw M K N [GROUP]                # int8 gw     rocket_matmul_plan_int8_gw
oracle plan-int4    M K N                        # int4        rocket_matmul_plan_int4
oracle plan-tf32    M K N                        # tf32        rocket_matmul_plan_tf32
oracle plan-bf16    M K N                        # bf16        rocket_matmul_plan_bf16
oracle emit         M K N IN WT OUT [FP32TOFP16] [ACCUMULATE] [ADD_DMA]   # gen_matmul_fp16
oracle emit-int8    ...                          # gen_matmul_int8
oracle emit-int4    ...                          # gen_matmul_int4
oracle emit-tf32    ...                          # gen_matmul_tf32
oracle emit-bf16    ...                          # gen_matmul_bf16
```

A plan prints `Mt Kt Nt jobs`, or `-1` for an unsupported shape. An emit prints
the task count, then one `%016llx` per NPUOP word, or `-1`.

**The emitters take the tile dims (`Mt Kt Nt`), not the full problem dims.** The
planner produces those dims; `parity.py` plans first and then emits the tile.
`FP32TOFP16` defaults to 1, `ACCUMULATE` to 0, `ADD_DMA` to 0 (the fp16
`emit`'s long-standing defaults). int8 group-wise shares `emit-int8`: the group
scale is applied host-side on readback, so the program is dense-int8's.

## Run the parity harness

```sh
cargo build -p rocket-npu --bin regcmd_dump     # debug profile is enough
python3 tools/c-oracle/parity.py                # oracle + newest regcmd_dump
python3 tools/c-oracle/parity.py --dump /path/to/regcmd_dump --verbose
python3 tools/c-oracle/parity.py --kind fp16,int8-gw
```

Options:

| option | meaning |
|---|---|
| `--oracle` | oracle binary (default `/tmp/opencode/c-oracle/oracle`) |
| `--dump` | `regcmd_dump` binary (default: `$CARGO_TARGET_DIR` or repo `target`, newest of `debug`/`release`) |
| `--profile` | `auto` (newest binary), `debug` or `release`; ignored with `--dump` |
| `--kind` | dtype filter, repeatable or comma-separated: `fp16`, `int8`, `int8-gw`, `int4`, `tf32`, `bf16` |
| `--verbose` | print every plan/emit case and its result |

For each dtype the harness runs a shape matrix with that dtype's alignments
(fp16 `K%32`/`N%16`, int8 `K%32`/`N%32`, int4 `K%32`/`N%64`, tf32
`K%16`/`N%16`, bf16 `K%32`/`N%16`, all `M%4`), covering small, medium and large
shapes with `K > Kt`, `N > Nt` and `M > Mt`, plus shapes that must be refused
(`M%4`, `K%32`, `N%16`, zero). int8 group-wise additionally sweeps groups
`32/64/128/256/512/2048` and the missing-group refusal. Emit is compared at the
planned tile dims with fixed DMA addresses (`0x1000/0x2000/0x3000`, accumulate
`0x4000`) so the output is stable.

Comparison is exact: plan text, task count and every NPUOP word. On a mismatch
the harness prints the first differing line with both values and exits nonzero.
Every dtype in the matrix is ported now, so a full run is six PASS lines; the
`PENDING` path (a `not implemented in the Rust port yet` refusal) is kept for a
future rung.

## Board selftest matrix (numerics)

Parity above is host-only and structural. The numeric check runs the
`crates/rocket-npu/src/bin/` self-tests on the board (aarch64, NPU device):

```sh
cargo build --release -p rocket-npu --target aarch64-unknown-linux-gnu \
    --bin selftest --bin int8_selftest --bin bf16_selftest --bin attn_selftest \
    --bin int4_selftest --bin int16_selftest --bin tf32_selftest
ssh root@rock-5b-plus.lan 'mkdir -p /root/rocket-npu-selftest'
scp $CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/{selftest,int8_selftest,bf16_selftest,attn_selftest,int4_selftest,int16_selftest,tf32_selftest} \
    root@rock-5b-plus.lan:/root/rocket-npu-selftest/
```

On the board (each binary prints cosine / max error against an f64 reference):

```sh
cd /root/rocket-npu-selftest
# fp16: tiny tile, tile-cap, all-dims-tiled, and the resident-weight path
./selftest 4 32 16
./selftest 256 1024 512
./selftest 512 2560 4096
./selftest 256 1024 768 --context
# int8 group-wise: group 32 and 128, small and all-dims-tiled
./int8_selftest 4 32 32 32
./int8_selftest 256 640 256 128
./int8_selftest 512 2560 4096 128
# bf16 stream
./bf16_selftest 4 32 16
./bf16_selftest 256 1024 512
# tf32 one-shot: no arguments runs the built-in shape list (tiny, ragged, K > Kt)
./tf32_selftest
# attention: short causal and long causal
./attn_selftest 82
./attn_selftest 2587 2587 8 8
```

The matrix mirrors `parity.py`'s shapes so a structural PASS plus these cosines
covers both the program bytes and the hardware result.

## Notes / quirks

- **Oracle CLI change (M0):** `plan-int8` used to alias the group-wise planner.
  It now means dense int8 (`rocket_matmul_plan_int8`) to match the Rust
  subcommand names; group-wise moved to `plan-int8-gw`. `plan` and `emit` are
  byte-identical to before.
- `matmul_params_t.m/k/n` are `uint16_t`, so an emit is always called with tile
  dims (max `256 x 16384 x 256`), never full problem dims.
- The C int8 emitter has an `ACCUMULATE=1` EW-add path that the Rust port does
  not implement; the C library's own comment says that encoding is not
  bit-exact and production never uses it. `parity.py` therefore emits int8 with
  the defaults only. fp16 does exercise `FP32TOFP16=0` and `ACCUMULATE=1`.
- The int4 and tf32 generators ignore `FP32TOFP16`/`ACCUMULATE`/`ADD_DMA` (plain
  conv, fixed output writer), so those flags never change their programs.
- tf32's program is the one place where the *precision* is per-stage: CNA in/proc
  and CORE proc are 7, while all three DPU fields are fp32 (5) — there is no
  tf32 code in the DPU enum.
