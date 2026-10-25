#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dtype parity harness: the Rust rocket-npu planner/emitters vs the C oracle.

Every RK3588 matmul dtype gets a shape matrix with that dtype's alignments. For
each shape the C oracle and `regcmd_dump` are run with identical arguments and
their stdout is compared byte for byte:

  plan  -> "Mt Kt Nt jobs", or "-1"
  emit  -> task_count, then one %016llx word per NPUOP (at the planned tile dims)

Dtypes whose Rust planner/emitter is not ported yet report a visible PENDING
line instead of failing. Any mismatch exits nonzero and prints the first
differing line with both values.

Usage:
  python3 tools/c-oracle/parity.py
  python3 tools/c-oracle/parity.py --dump /path/to/regcmd_dump --verbose
  python3 tools/c-oracle/parity.py --kind fp16,int8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORACLE = "/tmp/opencode/c-oracle/oracle"

# Fixed DMA addresses so the emitted programs are stable across runs.
IN_DMA = "0x1000"
WT_DMA = "0x2000"
OUT_DMA = "0x3000"
ACC_DMA = "0x4000"


@dataclass(frozen=True)
class Dtype:
    """One matmul dtype: its oracle/Rust subcommands and shape alignment."""

    name: str
    plan_mode: str
    emit_mode: str
    n_align: int
    k_align: int
    milestone: str = ""
    groups: tuple[int | None, ...] = (None,)
    # Extra emit flag sets (FP32TOFP16, ACCUMULATE, ADD_DMA); () means defaults.
    emit_variants: tuple[tuple[str, ...], ...] = ((),)


DTYPES: tuple[Dtype, ...] = (
    Dtype("fp16", "plan", "emit", n_align=16, k_align=32,
          emit_variants=((), ("0",), ("1", "1", ACC_DMA))),
    Dtype("int8", "plan-int8", "emit-int8", n_align=32, k_align=32),
    Dtype("int8-gw", "plan-int8-gw", "emit-int8", n_align=32, k_align=32,
          groups=(None, 32, 64, 128, 256, 512, 2048)),
    Dtype("int4", "plan-int4", "emit-int4", n_align=64, k_align=32, milestone="M3"),
    Dtype("tf32", "plan-tf32", "emit-tf32", n_align=16, k_align=16, milestone="M4"),
    Dtype("bf16", "plan-bf16", "emit-bf16", n_align=16, k_align=32),
)

# (M, K, N) before per-dtype alignment: tiny, small, medium, tile-boundary and
# large shapes. K > Kt, N > Nt and M > Mt all occur at the large end.
BASE_SHAPES: tuple[tuple[int, int, int], ...] = (
    (4, 32, 16),        # one tiny tile
    (8, 48, 32),        # K%16 but not K%32: exercises tf32's K alignment
    (8, 96, 64),        # small, non-power-of-two K
    (64, 256, 128),     # medium
    (128, 512, 256),    # medium, N at the tile cap
    (256, 1024, 512),   # M at the tile cap, N > Nt
    (260, 2080, 320),   # tails in every dimension
    (512, 2560, 4096),  # M, K and N all tile
    (1024, 8192, 2048), # very large
    (4, 16384, 16),     # deep K
    (512, 32, 16),      # K at the alignment floor
)

# Shapes that must be refused identically (plan prints -1 on both sides).
INVALID_SHAPES: tuple[tuple[int, int, int], ...] = (
    (6, 64, 16),   # M % 4
    (4, 48, 16),   # K % 32
    (4, 64, 8),    # N % 16
    (0, 64, 16),   # non-positive
)


def round_up(value: int, align: int) -> int:
    return max(align, (value + align - 1) // align * align)


def shapes_for(dtype: Dtype) -> list[tuple[int, int, int]]:
    """The valid shape matrix, aligned to this dtype's geometry."""
    return [
        (m, round_up(k, dtype.k_align), round_up(n, dtype.n_align))
        for m, k, n in BASE_SHAPES
    ]


def default_dump(profile: str) -> Path:
    """`$CARGO_TARGET_DIR` (or repo `target`) + the active profile."""
    base = Path(os.environ.get("CARGO_TARGET_DIR") or REPO_ROOT / "target")
    candidates = {
        "debug": base / "debug" / "regcmd_dump",
        "release": base / "release" / "regcmd_dump",
    }
    if profile != "auto":
        return candidates[profile]
    existing = [p for p in candidates.values() if p.is_file()]
    if not existing:
        return candidates["debug"]
    # The most recently built binary is the active profile.
    return max(existing, key=lambda p: p.stat().st_mtime)


@dataclass
class Failure:
    dtype: str
    case: str
    detail: str


def run(binary: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *args], capture_output=True, text=True, check=False
    )


def is_not_implemented(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    text = (proc.stderr or "").lower()
    return "not implemented in the rust port" in text or "unknown mode" in text


def differing_lines(a: str, b: str) -> int:
    a_lines, b_lines = a.splitlines(), b.splitlines()
    diffs = sum(1 for x, y in zip(a_lines, b_lines) if x != y)
    return diffs + abs(len(a_lines) - len(b_lines))


def first_difference(a: str, b: str) -> str:
    """A compact diff: the first differing line, both values, and the count."""
    a_lines, b_lines = a.splitlines(), b.splitlines()
    for i, (x, y) in enumerate(zip(a_lines, b_lines), start=1):
        if x != y:
            return (
                f"line {i}: oracle={x!r} rust={y!r} "
                f"({differing_lines(a, b)} differing line(s))"
            )
    if len(a_lines) != len(b_lines):
        return (
            f"line count: oracle={len(a_lines)} rust={len(b_lines)} "
            f"(first extra: oracle={a_lines[len(b_lines):]!r} "
            f"rust={b_lines[len(a_lines):]!r})"
        )
    return "outputs differ only in trailing bytes"


def compare(
    failures: list[Failure],
    dtype: Dtype,
    case: str,
    oracle_proc: subprocess.CompletedProcess[str],
    rust_proc: subprocess.CompletedProcess[str],
) -> bool:
    if oracle_proc.returncode != 0:
        failures.append(
            Failure(
                dtype.name,
                case,
                f"oracle exited {oracle_proc.returncode}: "
                f"{(oracle_proc.stderr or '').strip()}",
            )
        )
        return False
    if rust_proc.returncode != 0:
        failures.append(
            Failure(
                dtype.name,
                case,
                f"regcmd_dump exited {rust_proc.returncode}: "
                f"{(rust_proc.stderr or '').strip()}",
            )
        )
        return False
    if oracle_proc.stdout != rust_proc.stdout:
        failures.append(
            Failure(dtype.name, case, first_difference(oracle_proc.stdout, rust_proc.stdout))
        )
        return False
    return True


def parse_plan(stdout: str) -> tuple[int, int, int] | None:
    line = stdout.strip()
    if line == "-1":
        return None
    parts = line.split()
    if len(parts) != 4:
        return None
    try:
        mt, kt, nt = (int(p) for p in parts[:3])
    except ValueError:
        return None
    return mt, kt, nt


def probe(
    dtype: Dtype,
    dump: Path,
    failures: list[Failure],
    verbose: bool,
) -> tuple[bool, bool]:
    """(plan_pending, emit_pending) by probing the smallest valid shape."""
    m, k = 4, round_up(32, dtype.k_align)
    n = round_up(32, dtype.n_align)
    plan_pending = emit_pending = False

    plan_args = [dtype.plan_mode, str(m), str(k), str(n)]
    if dtype.groups != (None,):
        plan_args.append("32")
    proc = run(dump, plan_args)
    if is_not_implemented(proc):
        plan_pending = True
        if verbose:
            print(f"  probe {dtype.plan_mode}: {(proc.stderr or '').strip()}")
    elif proc.returncode != 0:
        failures.append(
            Failure(
                dtype.name,
                f"{dtype.plan_mode} probe",
                f"regcmd_dump exited {proc.returncode}: {(proc.stderr or '').strip()}",
            )
        )

    proc = run(dump, [dtype.emit_mode, str(m), str(k), str(n), IN_DMA, WT_DMA, OUT_DMA])
    if is_not_implemented(proc):
        emit_pending = True
        if verbose:
            print(f"  probe {dtype.emit_mode}: {(proc.stderr or '').strip()}")
    elif proc.returncode != 0:
        failures.append(
            Failure(
                dtype.name,
                f"{dtype.emit_mode} probe",
                f"regcmd_dump exited {proc.returncode}: {(proc.stderr or '').strip()}",
            )
        )
    return plan_pending, emit_pending


def check_dtype(
    dtype: Dtype,
    oracle: Path,
    dump: Path,
    failures: list[Failure],
    verbose: bool,
) -> tuple[str, int, int]:
    """Returns (status, plan_checks, emit_checks)."""
    plan_pending, emit_pending = probe(dtype, dump, failures, verbose)
    if plan_pending or emit_pending:
        pending = []
        if plan_pending:
            pending.append(dtype.plan_mode)
        if emit_pending:
            pending.append(dtype.emit_mode)
        detail = "not implemented in the Rust port yet"
        if dtype.milestone:
            detail += f" ({dtype.milestone})"
        print(f"PENDING  {dtype.name}: {', '.join(pending)} — {detail}")
        return "pending", 0, 0

    plan_checks = 0
    emit_programs: dict[tuple[int, int, int, tuple[str, ...]], str] = {}

    def check_plan(m: int, k: int, n: int, group: int | None) -> tuple[int, int, int] | None:
        nonlocal plan_checks
        args = [dtype.plan_mode, str(m), str(k), str(n)]
        if group is not None:
            args.append(str(group))
        case = " ".join(args)
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        plan_checks += 1
        if not compare(failures, dtype, case, c_proc, r_proc):
            return None
        tile = parse_plan(c_proc.stdout)
        if verbose:
            print(f"  {case:<42} -> {c_proc.stdout.strip()}")
        return tile

    def check_emit(tile: tuple[int, int, int], variant: tuple[str, ...]) -> None:
        mt, kt, nt = tile
        key = (mt, kt, nt, variant)
        if key in emit_programs:
            return
        args = [dtype.emit_mode, str(mt), str(kt), str(nt), IN_DMA, WT_DMA, OUT_DMA, *variant]
        case = " ".join(args)
        c_proc = run(oracle, args)
        r_proc = run(dump, args)
        emit_programs[key] = case
        if compare(failures, dtype, case, c_proc, r_proc) and verbose:
            words = c_proc.stdout.splitlines()
            print(f"  {case:<42} -> {words[0] if words else '?'} words")

    for m, k, n in shapes_for(dtype):
        for group in dtype.groups:
            tile = check_plan(m, k, n, group)
            if tile is not None:
                for variant in dtype.emit_variants:
                    check_emit(tile, variant)

    for m, k, n in INVALID_SHAPES:
        # An "invalid" shape only is one for this dtype's geometry (e.g. K=48 is
        # legal for tf32 but not for fp16/int8/bf16).
        if m <= 0 or k <= 0 or n <= 0 or m % 4 or k % dtype.k_align or n % dtype.n_align:
            check_plan(m, k, n, None)

    status = "pass" if not any(f.dtype == dtype.name for f in failures) else "fail"
    return status, plan_checks, len(emit_programs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the Rust rocket-npu planner/emitters against the C oracle.",
    )
    parser.add_argument("--oracle", default=DEFAULT_ORACLE,
                        help=f"C oracle binary (default: {DEFAULT_ORACLE})")
    parser.add_argument("--dump", default=None,
                        help="regcmd_dump binary (default: $CARGO_TARGET_DIR or target/<profile>)")
    parser.add_argument("--profile", choices=("auto", "debug", "release"), default="auto",
                        help="cargo profile used when --dump is omitted (default: auto = newest)")
    parser.add_argument("--kind", action="append", default=None,
                        help="dtype filter, repeatable or comma-separated "
                             f"({', '.join(d.name for d in DTYPES)})")
    parser.add_argument("--verbose", action="store_true", help="print every case and its output")
    args = parser.parse_args()

    oracle = Path(args.oracle)
    dump = Path(args.dump) if args.dump else default_dump(args.profile)
    for label, path in (("oracle", oracle), ("regcmd_dump", dump)):
        if not path.is_file():
            print(f"error: {label} binary not found at {path}", file=sys.stderr)
            if label == "regcmd_dump":
                print("build it with: cargo build -p rocket-npu --bin regcmd_dump",
                      file=sys.stderr)
            return 2

    if args.kind:
        wanted: list[str] = []
        for chunk in args.kind:
            wanted.extend(part.strip() for part in chunk.split(",") if part.strip())
        known = {d.name for d in DTYPES}
        unknown = [k for k in wanted if k not in known]
        if unknown:
            print(f"error: unknown --kind {', '.join(unknown)}; known: {', '.join(sorted(known))}",
                  file=sys.stderr)
            return 2
        if not wanted:
            print("error: --kind selected no dtypes", file=sys.stderr)
            return 2
        selected = tuple(d for d in DTYPES if d.name in wanted)
    else:
        selected = DTYPES

    print(f"oracle: {oracle}")
    print(f"dump:   {dump}")
    print()

    failures: list[Failure] = []
    results: list[tuple[Dtype, str, int, int]] = []
    for dtype in selected:
        status, plan_checks, emit_checks = check_dtype(dtype, oracle, dump, failures, args.verbose)
        results.append((dtype, status, plan_checks, emit_checks))

    print()
    width = max(len(d.name) for d in selected)
    for dtype, status, plan_checks, emit_checks in results:
        if status == "pending":
            continue
        counts = f"plan {plan_checks}, emit {emit_checks}"
        print(f"{dtype.name:<{width}}  {status.upper():<5}  {counts}")

    if failures:
        print()
        print(f"{len(failures)} mismatch(es):")
        for failure in failures:
            print(f"  FAIL {failure.dtype}: {failure.case}")
            print(f"       {failure.detail}")
        print()
        print("RESULT: FAIL")
        return 1

    passed = sum(1 for _, status, _, _ in results if status == "pass")
    pending = sum(1 for _, status, _, _ in results if status == "pending")
    print()
    print(f"{len(results)} dtype(s): {passed} pass, {pending} pending, 0 fail")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
