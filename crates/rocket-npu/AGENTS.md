# AGENTS.md — rocket-npu (crates/rocket-npu)

Developer/agent notes for the pure-Rust RK3588 NPU driver. The human summary is
[`README.md`](README.md). The long-form port history (milestones M0-M12, board
measurements, the upstream re-pin round) lives in the root [`AGENTS.md`](../../AGENTS.md)
under "C-library parity port", "Upstream re-pin round" and "Port status"; the Burn-facing
API is in [`../burn-rocket/AGENTS.md`](../burn-rocket/AGENTS.md).

## What this crate is

An interface-compatible Rust port of `librocketnpu` (reference commit `88cae6d`). Planners
and register programs are **byte-identical** to the C library for the RK3588 surface; board
numerics are cosine >= 0.99999 against f64 references. The surface that is deliberately not
ported (RK3576 entries, the C ABI, `rocket_stream_*` aliases, `_planes`/transposed variants,
the graph runner) is listed at the top of `src/lib.rs` ("Surface covered and deliberately
missing"). Start there rather than re-deriving it.

## Layout

| path | contents |
|---|---|
| `src/device/` | uAPI, buffer objects, submit, fences, sysfs, core affinity |
| `src/regcmd/` | register-program emitters: fp16/int8/int4/bf16/tf32 matmul, activation LUT, PPU pool, conv (fp16 + int8) |
| `src/matmul/` | planner (`plan`), cube layout, BO scratch, `run`, resident `Context`, int8/int4/tf32 one-shots, bf16 stream and resident bf16 |
| `src/attn/` | flash attention (per-head, chained, tiled long-context, row-major handoff) and the fast host softmax |
| `src/ops/` | activation/elementwise, reduce/softmax, norms, FFN, encoder, conv, conv-transpose, pool, resize, SAM and SigLIP |
| `src/hw.rs`, `src/knobs.rs`, `src/error.rs` | hardware constants, `ROCKET_*` environment knobs, typed errors |
| `src/bin/` | board self-tests and probes: `regcmd_dump`, `selftest`, `attn_selftest`, `int8/int4/int16/bf16/tf32_selftest`, `activation_selftest`, `reduce_selftest`, `ops_selftest`, `conv_selftest`, `encoders_selftest`, `q8_bench`, `expert_bench` |
| `tools/c-oracle/` | host C oracle + `parity.py` (plan/program byte-parity) and the board A/B scripts (`ab_board.c`, `ab_compare.py`); see its README |

## Commands

```sh
cargo test -p rocket-npu                         # 269 host tests: planner, emitters, layout, refs
cargo build -p rocket-npu --bin regcmd_dump
python3 crates/rocket-npu/tools/c-oracle/parity.py                          # must print RESULT: PASS
python3 crates/rocket-npu/tools/c-oracle/parity.py --env ROCKET_MM_STREAM=0 # legacy matrix
cargo clippy -p rocket-npu --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
```

The parity oracle needs a host build of the C library pinned at `88cae6d`; see
`tools/c-oracle/README.md`. Board self-tests are cross-built
(`--target aarch64-unknown-linux-gnu --bin <name>`), copied to the board and run with
`taskset -c 0-3`; the exact invocations are in the root AGENTS.md "Verification commands".

## Known issues

None open. (The six `needless_range_loop` clippy lints that predated the 2026-11-17 restructure were fixed the same day; clippy is clean on host and aarch64 with `-D warnings`.)

## Conventions

- Failures are typed (`Error`, with a C-compatible code via `Error::c_code`); the Burn layer
  turns them into an `OpFailure` panic payload.
- Contexts are not thread-safe. Callers (the `burn-rocket` engine) serialize access.
- Knobs keep the C library's names (`ROCKET_MM_STREAM`, `ROCKET_FA_CHAIN`, `ROCKET_FA_TILE_KV`,
  `ROCKET_ATTN_HOST_SOFTMAX`, `ROCKET_CONV_BATCH`, …) and are read per call.
- DRM buffer objects are not charged to a process memory cgroup: `systemd-run -p MemoryMax=…`
  does not contain NPU work, and a large attention allocation can push the board into a global
  OOM. Run long-context gates only with >= 5 GB free.
