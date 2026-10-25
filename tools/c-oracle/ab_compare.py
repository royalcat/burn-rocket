#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""C-vs-Rust A/B: the same raw inputs through librocketnpu and the Rust port.

Run it on the board (the C side must be built natively there, because a cross-built
librocketnpu reads an empty driver name and cannot open the device). Both binaries use
the same `--io` argument layout, so this script only has to produce inputs, run each
side and compare the raw outputs. It covers the M6b reduce/softmax family through
`reduce_selftest`, the M7 norm/vision/FFN/encoder family through `ops_selftest` and the
M8a fp16 convolution family through `conv_selftest`:

    python3 ab_compare.py --rust ./reduce_selftest --ops ./ops_selftest \
        --conv ./conv_selftest --c ./ab_board [--work DIR] [--no-wide]

For every case it prints the bit-identical element count, the cosine and the max-abs of
the C output against the Rust output, then PASS/FAIL. A mismatch exits nonzero.

What is expected to match exactly, and why:

  max/min pools   idempotent, no reciprocal       -> bit-identical
  feature reduce  same ones-matmul + fp32 accum   -> bit-identical
  cumsum          same triangular-ones matmul     -> bit-identical
  avg pool        per-pass fp16(65536/k) recip    -> ~1e-3 by design; the emit and the
                                                     host schedule are the same, so it
                                                     usually lands exact too
  softmax family  host row-max + EXP LUT + reduce -> ~1e-6 cosine; neither device path
                                                     calls libm exp/log (only the host
                                                     reference does)
  scale_rows      fp16 broadcast EW multiply      -> bit-identical

The M7 families are gated STRICTLY where nothing but EW/reduce/LUT arithmetic is
involved — the two sides then run the same program and the same host glue:

  rmsnorm/layernorm, batchnorm/groupnorm/instancenorm/l2norm   -> bit-identical
  geglu (activation LUT + EW multiply)                        -> bit-identical

and RELATIVELY where the fp16 matmul decides the last bits (the C and the Rust port run
the same planner and the same regcmd, but a K-accumulation regrouping is one ULP; the
gate is cosine >= 0.9999 and max-abs <= 5e-2, and the exact count is printed so a
regression to "not even close" is visible):

  ffn / ffn_fused / mlp_fused / encoder_block

The M8a convolution family runs the same programs and the same host scatter on both
sides, so it is gated STRICTLY (bit-identical):

  conv (direct, every stride/pad/dilation/tiling shape)   -> bit-identical
  conv depthwise                                         -> bit-identical
  conv1d                                                 -> bit-identical
  conv_transpose (deconv route and the lowering)          -> bit-identical

The fused conv->activation path adds the DPU LUT interpolation, which the C header says
to gate with a tolerance, so it is gated relatively (cosine + max-abs); the exact count
is printed either way.
"""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
from pathlib import Path


def w_f16(path: Path, values: list[float]) -> None:
    path.write_bytes(b"".join(struct.pack("<e", v) for v in values))


def w_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(b"".join(struct.pack("<f", v) for v in values))


def r_f16(path: Path) -> list[float]:
    raw = path.read_bytes()
    return [struct.unpack("<e", raw[i:i + 2])[0] for i in range(0, len(raw), 2)]


def r_f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    return [struct.unpack("<f", raw[i:i + 4])[0] for i in range(0, len(raw), 4)]


def lcg(n: int, lo: float, hi: float, seed: int) -> list[float]:
    st = (0x9E3779B9 ^ seed) & 0xFFFFFFFF
    out = []
    for _ in range(n):
        st = (st * 1664525 + 1013904223) & 0xFFFFFFFF
        u = ((st >> 8) & 0xFFFF) / 65535.0
        out.append(lo + (hi - lo) * u)
    return out


def stats(want: list[float], got: list[float]) -> tuple[int, float, float]:
    exact = sum(1 for a, b in zip(want, got) if a == b)
    dot = sum(a * b for a, b in zip(want, got))
    na = math.sqrt(sum(a * a for a in want))
    nb = math.sqrt(sum(b * b for b in got))
    cos = dot / (na * nb) if na and nb else 1.0
    max_abs = max((abs(a - b) for a, b in zip(want, got)), default=0.0)
    return exact, cos, max_abs


class Runner:
    def __init__(self, rust: Path, cbin: Path, work: Path) -> None:
        # Path("./x") would print as "x" and subprocess would then search PATH, so pin
        # both binaries to an absolute path.
        self.rust = rust.resolve()
        self.cbin = cbin.resolve()
        self.work = work
        self.fail = False
        work.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, args: list[str], src: Path, out_f32: bool,
            n_out: int) -> tuple[list[float], list[float]]:
        """Run both binaries on one input file and return (c_out, rust_out)."""
        return self.run_files(name, args, [src], out_f32, n_out)

    def run_files(self, name: str, args: list[str], inputs: list[Path], out_f32: bool,
                  n_out: int) -> tuple[list[float], list[float]]:
        """Run both binaries on `inputs` (the trailing files of the `--io` layout).

        `args` carries the family and its scalar/optional-file arguments; the binary
        appends the input file(s) and then its own output path, which is why the raw
        files go last in both `reduce_selftest --io` and `ops_selftest --io`.
        """
        c_out = self.work / f"{name}.c"
        r_out = self.work / f"{name}.rs"
        for out in (c_out, r_out):
            if out.exists():
                out.unlink()
        ins = [str(p) for p in inputs]
        r = subprocess.run([str(self.rust), "--io", *args, *ins, str(r_out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {name}: rust exited {r.returncode}: {r.stderr.strip()}")
            self.fail = True
            return [], []
        c = subprocess.run([str(self.cbin), *args, *ins, str(c_out)],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print(f"  {name}: C exited {c.returncode}: {c.stderr.strip()}")
            self.fail = True
            return [], []
        read = r_f32 if out_f32 else r_f16
        c_vals, r_vals = read(c_out), read(r_out)
        if len(c_vals) != n_out or len(r_vals) != n_out:
            print(f"  {name}: output size {len(c_vals)}/{len(r_vals)}, expected {n_out}")
            self.fail = True
        return c_vals, r_vals

    def report(self, name: str, c_vals: list[float], r_vals: list[float],
               strict: bool = False, tol: tuple[float, float] = (0.9999, 1e-3)) -> None:
        """Print the A/B for one case.

        PASS is bit-identity for the strict families (pure EW/reduce/LUT arithmetic) and
        `cos >= tol[0] and max_abs <= tol[1]` for the ones whose fp16 K-accumulation
        grouping can differ. The exact count is always printed.
        """
        if not c_vals or not r_vals:
            return
        exact, cos, max_abs = stats(c_vals, r_vals)
        n = len(c_vals)
        ok = exact == n if strict else (exact == n or (cos >= tol[0] and max_abs <= tol[1]))
        self.fail |= not ok
        tag = "bit-identical" if exact == n else f"exact={exact}/{n}"
        print(f"  {name:<34} {tag:<18} cos={cos:.9f} max_abs={max_abs:.3e} -> "
              f"{'PASS' if ok else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", required=True, type=Path,
                    help="the reduce_selftest binary (M6b families)")
    ap.add_argument("--ops", type=Path,
                    help="the ops_selftest binary (M7 families; skipped when missing)")
    ap.add_argument("--conv", type=Path,
                    help="the conv_selftest binary (M8a families; skipped when missing)")
    ap.add_argument("--c", required=True, type=Path)
    ap.add_argument("--work", default=Path("/tmp/ab_m6b"), type=Path)
    ap.add_argument("--no-wide", action="store_true",
                    help="skip the 257x1000 softmax case")
    args = ap.parse_args()

    run = Runner(args.rust, args.c, args.work)

    # ── Spatial pools ──────────────────────────────────────────────────────────
    print("spatial pools:")
    for (c, h, w) in [(8, 7, 7), (3, 56, 56), (4, 49, 49), (5, 17, 13), (2, 1, 1)]:
        vals = lcg(c * h * w, -1.0, 2.0, 0xC0 + c * 131 + h * 17 + w)
        for ch in range(c):
            off = 0.5 + 2.0 * ((ch * 37 % 100) / 100.0)
            for i in range(h * w):
                vals[ch * h * w + i] += off
        src = run.work / f"pool_{c}x{h}x{w}.in"
        w_f16(src, vals)
        for fam in ("avg", "max", "min"):
            c_vals, r_vals = run.run(f"{fam}_{c}x{h}x{w}",
                                     [fam, str(c), str(h), str(w)], src, False, c)
            run.report(f"{fam} {c}x{h}x{w}", c_vals, r_vals)

    # ── Feature reduce ─────────────────────────────────────────────────────────
    print("feature-axis reduce:")
    for (m, h) in [(4, 32), (64, 256), (264, 320)]:
        vals = lcg(m * h, -1.0, 3.0, 0xF0 + m * 31 + h)
        src = run.work / f"feat_{m}x{h}.in"
        w_f16(src, vals)
        for mean in (0, 1):
            c_vals, r_vals = run.run(f"feat_{m}x{h}_mean{mean}",
                                     ["feature", str(m), str(h), str(mean)], src, True, m)
            run.report(f"feature {m}x{h} mean={mean}", c_vals, r_vals)

    # ── Cumsum ─────────────────────────────────────────────────────────────────
    print("cumsum:")
    for (m, n) in [(4, 16), (64, 128), (128, 256)]:
        vals = lcg(m * n, -2.0, 2.0, 0x5C + m * 31 + n)
        src = run.work / f"cumsum_{m}x{n}.in"
        w_f16(src, vals)
        for excl, rev in ((0, 0), (1, 0), (0, 1), (1, 1)):
            c_vals, r_vals = run.run(f"cumsum_{m}x{n}_{excl}{rev}",
                                     ["cumsum", str(m), str(n), str(excl), str(rev)],
                                     src, False, m * n)
            run.report(f"cumsum {m}x{n} excl={excl} rev={rev}", c_vals, r_vals)

    # ── Softmax family ─────────────────────────────────────────────────────────
    print("softmax / logsoftmax / cross-entropy:")
    shapes = [(4, 8), (64, 100), (256, 512)]
    if not args.no_wide:
        shapes.append((257, 1000))
    for (m, n) in shapes:
        vals = lcg(m * n, -4.0, 4.0, 0x5F + m * 31 + n)
        src = run.work / f"sm_{m}x{n}.in"
        w_f16(src, vals)
        for fam in ("softmax", "logsoftmax"):
            c_vals, r_vals = run.run(f"{fam}_{m}x{n}", [fam, str(m), str(n)],
                                     src, False, m * n)
            run.report(f"{fam} {m}x{n}", c_vals, r_vals)
        c_vals, r_vals = run.run(f"ce_{m}x{n}", ["ce", str(m), str(n)], src, True, m)
        run.report(f"ce {m}x{n}", c_vals, r_vals)

    # ── Per-row scale ──────────────────────────────────────────────────────────
    print("scale_rows:")
    for (m, n) in [(4, 16), (64, 256)]:
        vals = lcg(m * n, -2.0, 2.0, 0x51 + m * 31 + n)
        rvals = lcg(m, 0.25, 4.0, 0x52 + m)
        src = run.work / f"scale_{m}x{n}.in"
        w_f16(src, vals)
        rpath = run.work / f"scale_{m}x{n}.r"
        w_f32(rpath, rvals)
        c_vals, r_vals = run.run(f"scale_{m}x{n}",
                                 ["scale_rows", str(m), str(n), str(rpath)],
                                 src, False, m * n)
        run.report(f"scale_rows {m}x{n}", c_vals, r_vals)

    if args.ops is not None:
        m7(run, args.ops)

    if args.conv is not None:
        m8a(run, args.conv)

    print()
    print("ab_compare:", "FAIL" if run.fail else "PASS")
    return 1 if run.fail else 0


def m7(run: Runner, ops: Path) -> None:
    """The M7 families through `ops_selftest --io`."""
    # The M7 binary is a second Rust binary; swap it in and back out so the M6b cases
    # above stay on reduce_selftest.
    m6b = run.rust
    run.rust = ops.resolve()

    def f16(path: Path, vals: list[float]) -> str:
        w_f16(path, vals)
        return str(path)

    # ── Norms ──────────────────────────────────────────────────────────────────
    print("norms (rmsnorm / layernorm):")
    for (m, h) in [(4, 256), (64, 512), (257, 1024)]:
        x = run.work / f"norm_{m}x{h}.in"
        w_f16(x, lcg(m * h, -2.5, 2.5, 0x71 + m * 31 + h))
        wpath = f16(run.work / f"norm_{m}x{h}.w", lcg(h, 0.5, 1.5, 0x72 + h))
        ones = f16(run.work / f"norm_{m}x{h}.one", [1.0] * h)
        for tag, wp in (("w", wpath), ("w=1", ones)):
            c_vals, r_vals = run.run(f"rmsnorm_{m}x{h}_{tag}",
                                     ["rmsnorm", str(m), str(h), "1e-6", wp],
                                     x, False, m * h)
            run.report(f"rmsnorm {m}x{h} {tag}", c_vals, r_vals, strict=True)
        gamma = f16(run.work / f"norm_{m}x{h}.g", lcg(h, 0.5, 1.5, 0x73 + h))
        beta = f16(run.work / f"norm_{m}x{h}.b", lcg(h, -0.5, 0.5, 0x74 + h))
        for tag, bp in (("g+b", beta), ("g", "-")):
            c_vals, r_vals = run.run(f"layernorm_{m}x{h}_{tag}",
                                     ["layernorm", str(m), str(h), "1e-5", gamma, bp],
                                     x, False, m * h)
            run.report(f"layernorm {m}x{h} {tag}", c_vals, r_vals, strict=True)

    # ── Vision norms ───────────────────────────────────────────────────────────
    print("vision norms (batchnorm / groupnorm / instancenorm / l2norm):")
    for (n, c, p) in [(2, 8, 49), (2, 16, 196)]:
        total = n * c * p
        x = run.work / f"vis_{n}x{c}x{p}.in"
        w_f16(x, lcg(total, -2.0, 2.0, 0x81 + n * 17 + c * 31 + p))
        gamma = f16(run.work / f"vis_{n}x{c}x{p}.g", lcg(c, 0.5, 1.5, 0x82 + c))
        beta = f16(run.work / f"vis_{n}x{c}x{p}.b", lcg(c, -0.5, 0.5, 0x83 + c))
        mean = f16(run.work / f"vis_{n}x{c}x{p}.m", lcg(c, -0.2, 0.2, 0x84 + c))
        var = f16(run.work / f"vis_{n}x{c}x{p}.v", lcg(c, 0.5, 1.5, 0x85 + c))
        c_vals, r_vals = run.run(
            f"bn_{n}x{c}x{p}",
            ["batchnorm", str(n), str(c), str(p), "1e-5", gamma, beta, mean, var],
            x, False, total)
        run.report(f"batchnorm {n}x{c}x{p}", c_vals, r_vals, strict=True)

        g = c // 2 if c >= 4 else 1
        c_vals, r_vals = run.run(
            f"gn_{n}x{c}x{p}",
            ["groupnorm", str(n), str(c), str(g), str(p), "1e-5", gamma, beta],
            x, False, total)
        run.report(f"groupnorm {n}x{c}x{p} G={g}", c_vals, r_vals, strict=True)

        c_vals, r_vals = run.run(
            f"in_{n}x{c}x{p}",
            ["instancenorm", str(n), str(c), str(p), "1e-5", gamma, beta],
            x, False, total)
        run.report(f"instancenorm {n}x{c}x{p}", c_vals, r_vals, strict=True)

        c_vals, r_vals = run.run(f"l2_{n}x{c}x{p}",
                                 ["l2norm", str(n * c), str(p), "1e-12"],
                                 x, False, total)
        run.report(f"l2norm {n * c}x{p}", c_vals, r_vals, strict=True)

    # ── GEGLU / FFN / MLP ──────────────────────────────────────────────────────
    print("geglu / ffn / mlp:")
    ffn_tol = (0.9999, 5e-2)
    for (m, h, i) in [(4, 256, 1024), (64, 512, 2048)]:
        x = run.work / f"ffn_{m}x{h}x{i}.x"
        w_f16(x, lcg(m * h, -1.5, 1.5, 0x91 + m * 31 + h))
        wg = f16(run.work / f"ffn_{m}x{h}x{i}.wg", lcg(i * h, -0.5, 0.5, 0x92 + h * 31 + i))
        wu = f16(run.work / f"ffn_{m}x{h}x{i}.wu", lcg(i * h, -0.5, 0.5, 0x93 + h * 31 + i))
        wd = f16(run.work / f"ffn_{m}x{h}x{i}.wd", lcg(h * i, -0.5, 0.5, 0x94 + i * 31 + h))
        b1 = f16(run.work / f"ffn_{m}x{h}x{i}.b1", lcg(i, -0.1, 0.1, 0x95 + i))

        # GEGLU on the gated pair (GELU = 5 and SiLU = 3, the two the block uses).
        gate = run.work / f"ffn_{m}x{h}x{i}.gate"
        up = run.work / f"ffn_{m}x{h}x{i}.up"
        w_f16(gate, lcg(m * i, -3.0, 3.0, 0x96 + m * 31 + i))
        w_f16(up, lcg(m * i, -2.0, 2.0, 0x97 + m * 31 + i))
        for kind, kname in ((5, "gelu"), (3, "silu")):
            c_vals, r_vals = run.run_files(f"geglu_{m}x{i}_{kname}",
                                           ["geglu", str(m * i), str(kind)],
                                           [gate, up], False, m * i)
            run.report(f"geglu {m}x{i} {kname}", c_vals, r_vals, strict=True)

        # The binary appends the input file(s) and its output path, so the weights go
        # through run_files: args = the scalars, inputs = X WG WU WD.
        for fam in ("ffn", "ffn_fused"):
            c_vals, r_vals = run.run_files(f"{fam}_{m}x{h}x{i}",
                                           [fam, str(m), str(h), str(i), "5"],
                                           [x, Path(wg), Path(wu), Path(wd)],
                                           False, m * h)
            run.report(f"{fam} {m}x{h}x{i}", c_vals, r_vals, tol=ffn_tol)

        for b1p in (b1, "-"):
            tag = "b1" if b1p != "-" else "nob1"
            c_vals, r_vals = run.run_files(
                f"mlp_{m}x{h}x{i}_{tag}",
                ["mlp_fused", str(m), str(h), str(i), str(h), "5", b1p],
                [x, Path(wg), Path(wd)],
                False, m * h)
            run.report(f"mlp_fused {m}x{h}x{i} {tag}", c_vals, r_vals, tol=ffn_tol)

    # The fused entries' fallbacks. Two shapes make the C's `goto fb` fire and both
    # sides must make the same decision (a fallback is a *different* plan set, so a
    # mismatch here would show up as a large difference, not as last-bit noise):
    #   M = 1      -> `mm_plan_init_pin` refuses M % 4; the one-shot entry's height-1
    #                 shim then serves it (M = 2 would be refused by *both* sides, so it
    #                 is not a case the APIs accept at all)
    #   M = 260    -> the KACC EW-stride floor: Mt = 256 leaves a 4-row last tile
    # (I = 128 < 256 additionally fails the Nt pin's `pin_Nt <= N`.)
    print("fused-entry fallbacks (host handoff):")
    for (m, h, i) in [(1, 64, 128), (260, 64, 256)]:
        x = run.work / f"fb_{m}x{h}x{i}.x"
        w_f16(x, lcg(m * h, -1.0, 1.0, 0xB1 + m * 31 + h))
        wg = f16(run.work / f"fb_{m}x{h}x{i}.wg", lcg(i * h, -0.4, 0.4, 0xB2 + h + i))
        wd = f16(run.work / f"fb_{m}x{h}x{i}.wd", lcg(h * i, -0.4, 0.4, 0xB3 + i + h))
        for fam in ("ffn", "ffn_fused"):
            c_vals, r_vals = run.run_files(f"fb_{fam}_{m}x{h}x{i}",
                                           [fam, str(m), str(h), str(i), "5"],
                                           [x, Path(wg), Path(wg), Path(wd)],
                                           False, m * h)
            run.report(f"{fam} fallback {m}x{h}x{i}", c_vals, r_vals, tol=ffn_tol)
        c_vals, r_vals = run.run_files(
            f"fb_mlp_{m}x{h}x{i}",
            ["mlp_fused", str(m), str(h), str(i), str(h), "5", "-"],
            [x, Path(wg), Path(wd)],
            False, m * h)
        run.report(f"mlp_fused fallback {m}x{h}x{i}", c_vals, r_vals, tol=ffn_tol)

    # ── Encoder block ──────────────────────────────────────────────────────────
    print("encoder block:")
    enc_tol = (0.9999, 5e-2)
    for (t, d, nh, dff) in [(8, 64, 2, 128), (64, 128, 4, 256)]:
        x = run.work / f"enc_{t}x{d}x{dff}.x"
        w_f16(x, lcg(t * d, -1.0, 1.0, 0xA1 + t * 31 + d))
        ln1_g = f16(run.work / f"enc_{t}x{d}x{dff}.ln1g", lcg(d, 0.5, 1.5, 0xA2 + d))
        ln1_b = f16(run.work / f"enc_{t}x{d}x{dff}.ln1b", lcg(d, -0.5, 0.5, 0xA3 + d))
        ln2_g = f16(run.work / f"enc_{t}x{d}x{dff}.ln2g", lcg(d, 0.5, 1.5, 0xA4 + d))
        ln2_b = f16(run.work / f"enc_{t}x{d}x{dff}.ln2b", lcg(d, -0.5, 0.5, 0xA5 + d))
        wq = f16(run.work / f"enc_{t}x{d}x{dff}.wq", lcg(d * d, -0.3, 0.3, 0xA6 + d))
        wk = f16(run.work / f"enc_{t}x{d}x{dff}.wk", lcg(d * d, -0.3, 0.3, 0xA7 + d))
        wv = f16(run.work / f"enc_{t}x{d}x{dff}.wv", lcg(d * d, -0.3, 0.3, 0xA8 + d))
        wo = f16(run.work / f"enc_{t}x{d}x{dff}.wo", lcg(d * d, -0.3, 0.3, 0xA9 + d))
        bq = f16(run.work / f"enc_{t}x{d}x{dff}.bq", lcg(d, -0.1, 0.1, 0xAA + d))
        bv = f16(run.work / f"enc_{t}x{d}x{dff}.bv", lcg(d, -0.1, 0.1, 0xAB + d))
        bo = f16(run.work / f"enc_{t}x{d}x{dff}.bo", lcg(d, -0.1, 0.1, 0xAC + d))
        wf1 = f16(run.work / f"enc_{t}x{d}x{dff}.wf1", lcg(dff * d, -0.3, 0.3, 0xAD + dff))
        wf2 = f16(run.work / f"enc_{t}x{d}x{dff}.wf2", lcg(d * dff, -0.3, 0.3, 0xAE + dff))
        bf1 = f16(run.work / f"enc_{t}x{d}x{dff}.bf1", lcg(dff, -0.1, 0.1, 0xAF + dff))
        bf2 = f16(run.work / f"enc_{t}x{d}x{dff}.bf2", lcg(d, -0.1, 0.1, 0xB0 + d))
        args = ["encoder", str(t), str(d), str(nh), str(dff), "1e-5",
                ln1_g, ln1_b, wq, bq, wk, "-", wv, bv, wo, bo,
                ln2_g, ln2_b, wf1, bf1, wf2, bf2]
        c_vals, r_vals = run.run(f"enc_{t}x{d}x{nh}x{dff}", args, x, False, t * d)
        run.report(f"encoder T={t} d={d} h={nh} ff={dff}", c_vals, r_vals, tol=enc_tol)

    run.rust = m6b


def conv_out_dim(i: int, k: int, s: int, p: int, d: int = 1) -> int:
    return (i + 2 * p - (d * (k - 1) + 1)) // s + 1


def m8a(run: "Runner", conv: Path) -> None:
    """The M8a fp16 convolution family through `conv_selftest --io`."""
    m6b = run.rust
    run.rust = conv.resolve()

    def tensor(name: str, n: int, lo: float, hi: float, seed: int) -> Path:
        path = run.work / name
        w_f16(path, lcg(n, lo, hi, seed))
        return path

    print("direct conv:")
    # (ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx, depthwise)
    conv_cases = [
        ("k3 p1", 32, 16, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 0),
        ("k3 s2", 32, 16, 16, 32, 3, 3, 2, 2, 0, 0, 1, 1, 0),
        ("k5 p2", 32, 16, 16, 16, 5, 5, 1, 1, 2, 2, 1, 1, 0),
        ("k5 s2 d2", 64, 16, 16, 16, 5, 5, 2, 2, 0, 0, 2, 2, 0),
        ("1x1", 64, 8, 8, 16, 1, 1, 1, 1, 0, 0, 1, 1, 0),
        ("rgb 1x1", 3, 32, 32, 16, 1, 1, 1, 1, 0, 0, 1, 1, 0),
        ("rgb stem", 3, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, 0),
        ("oc24", 32, 8, 8, 24, 5, 5, 1, 1, 2, 2, 1, 1, 0),
        ("oc tile", 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, 0),
        ("s2 tile", 128, 28, 28, 128, 3, 3, 2, 2, 1, 1, 1, 1, 0),
        ("space tile", 32, 112, 112, 32, 3, 3, 1, 1, 1, 1, 1, 1, 0),
        ("dw k3", 32, 56, 56, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1),
        ("dw s2", 64, 28, 28, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1),
        ("dw 1x1", 96, 14, 14, 96, 1, 1, 1, 1, 0, 0, 1, 1, 1),
        ("dw oc tile", 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    ]
    for case in conv_cases:
        name, ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, dy, dx, dw = case
        oh = conv_out_dim(ih, kh, sy, pt, dy)
        ow = conv_out_dim(iw, kw, sx, pl, dx)
        src = tensor(f"conv_{name}.in", ic * ih * iw, -1.5, 1.5, 0xC1 + ic * 31 + ih)
        wn = ic * kh * kw if dw else oc * ic * kh * kw
        wt = tensor(f"conv_{name}.w", wn, -0.5, 0.5, 0xC2 + oc * 31 + kh)
        args = ["conv", str(ic), str(ih), str(iw), str(oc), str(kh), str(kw),
                str(sy), str(sx), str(pt), str(pl), str(dy), str(dx), str(dw)]
        c_vals, r_vals = run.run_files(f"conv_{name}", args, [src, wt], False,
                                       oc * oh * ow)
        run.report(f"conv {name} {ic}x{ih}x{iw} oc{oc}", c_vals, r_vals, strict=True)

    print("conv1d:")
    for (ic, it, oc, kw, stride, pad) in [(80, 512, 384, 3, 1, 1), (80, 512, 384, 3, 2, 1)]:
        ot = conv_out_dim(it, kw, stride, pad)
        src = tensor(f"conv1d_{ic}x{it}.in", ic * it, -1.5, 1.5, 0xC3 + ic * 31 + it)
        wt = tensor(f"conv1d_{ic}x{it}.w", oc * ic * kw, -0.4, 0.4, 0xC4 + oc + kw)
        args = ["conv1d", str(ic), str(it), str(oc), str(kw), str(stride), str(pad)]
        c_vals, r_vals = run.run_files(f"conv1d_{ic}x{it}_s{stride}", args, [src, wt],
                                       False, oc * ot)
        run.report(f"conv1d {ic}x{it} k{kw}s{stride} p{pad} oc{oc}", c_vals, r_vals,
                   strict=True)

    print("conv -> activation fusion:")
    for kind, kname in ((3, "silu"), (4, "tanh"), (5, "gelu")):
        ic, ih, iw, oc, kh, kw = 32, 16, 16, 32, 3, 3
        oh = conv_out_dim(ih, kh, 1, 1)
        ow = conv_out_dim(iw, kw, 1, 1)
        src = tensor(f"act_{kname}.in", ic * ih * iw, -2.0, 2.0, 0xC5 + ic)
        wt = tensor(f"act_{kname}.w", oc * ic * kh * kw, -0.5, 0.5, 0xC6 + oc)
        args = ["conv_act", str(ic), str(ih), str(iw), str(oc), str(kh), str(kw),
                "1", "1", "1", "1", "1", "1", "0", str(kind)]
        c_vals, r_vals = run.run_files(f"act_{kname}", args, [src, wt], False,
                                       oc * oh * ow)
        run.report(f"conv_act {kname}", c_vals, r_vals, tol=(0.9999, 5e-3))

    print("transposed conv:")
    # (ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, opad_y, opad_x, depthwise)
    t_cases = [
        ("deconv s2", 32, 16, 16, 32, 3, 3, 2, 2, 1, 1, 1, 1, 0),
        ("deconv opad0", 32, 16, 16, 16, 3, 3, 2, 2, 1, 1, 0, 0, 0),
        ("deconv s4", 64, 8, 8, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0),
        ("lowered ic24", 24, 16, 16, 24, 3, 3, 2, 2, 1, 1, 1, 1, 0),
        ("lowered s3", 32, 16, 16, 16, 3, 3, 3, 3, 1, 1, 1, 1, 0),
        ("lowered s1", 32, 16, 16, 32, 3, 3, 1, 1, 1, 1, 0, 0, 0),
        ("dw s2", 32, 16, 16, 32, 2, 2, 2, 2, 0, 0, 0, 0, 1),
    ]
    for case in t_cases:
        (name, ic, ih, iw, oc, kh, kw, sy, sx, pt, pl, oy, ox, dw) = case
        oh = (ih - 1) * sy - 2 * pt + (kh - 1) + oy + 1
        ow = (iw - 1) * sx - 2 * pl + (kw - 1) + ox + 1
        src = tensor(f"t_{name}.in", ic * ih * iw, -1.5, 1.5, 0xC7 + ic * 31 + ih)
        wn = ic * kh * kw if dw else ic * oc * kh * kw
        wt = tensor(f"t_{name}.w", wn, -0.5, 0.5, 0xC8 + oc * 31 + kh)
        args = ["conv_t", str(ic), str(ih), str(iw), str(oc), str(kh), str(kw),
                str(sy), str(sx), str(pt), str(pl), "1", "1", str(oy), str(ox), str(dw)]
        c_vals, r_vals = run.run_files(f"t_{name}", args, [src, wt], False,
                                       oc * oh * ow)
        run.report(f"conv_t {name} {ic}x{ih}x{iw} oc{oc}", c_vals, r_vals, strict=True)

    run.rust = m6b



if __name__ == "__main__":
    sys.exit(main())
