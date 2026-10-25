#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""M6b C-vs-Rust A/B: the same raw inputs through librocketnpu and the Rust port.

Run it on the board (the C side must be built natively there, because a cross-built
librocketnpu reads an empty driver name and cannot open the device). Both binaries use
the same `--io` argument layout, so this script only has to produce inputs, run each
side and compare the raw outputs:

    python3 ab_compare.py --rust ./reduce_selftest --c ./ab_board [--work DIR]

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
        """Run both binaries on `src` and return (c_out, rust_out)."""
        c_out = self.work / f"{name}.c"
        r_out = self.work / f"{name}.rs"
        for out in (c_out, r_out):
            if out.exists():
                out.unlink()
        r = subprocess.run([str(self.rust), "--io", *args, str(src), str(r_out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {name}: rust exited {r.returncode}: {r.stderr.strip()}")
            self.fail = True
            return [], []
        c = subprocess.run([str(self.cbin), *args, str(src), str(c_out)],
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

    def report(self, name: str, c_vals: list[float], r_vals: list[float]) -> None:
        """Print the A/B for one case; PASS = identical or within 1e-3 at cos>=0.9999."""
        if not c_vals or not r_vals:
            return
        exact, cos, max_abs = stats(c_vals, r_vals)
        n = len(c_vals)
        ok = exact == n or (cos >= 0.9999 and max_abs <= 1e-3)
        self.fail |= not ok
        tag = "bit-identical" if exact == n else f"exact={exact}/{n}"
        print(f"  {name:<26} {tag:<18} cos={cos:.9f} max_abs={max_abs:.3e} -> "
              f"{'PASS' if ok else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", required=True, type=Path)
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

    print()
    print("ab_compare:", "FAIL" if run.fail else "PASS")
    return 1 if run.fail else 0


if __name__ == "__main__":
    sys.exit(main())
