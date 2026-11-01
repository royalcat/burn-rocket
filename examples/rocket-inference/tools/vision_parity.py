#!/usr/bin/env python3
"""Score the Rust d3 vision probe against the official runtime's dump.

``decision_reference.py --dump-vision DIR`` records, per image question, the
processor's ``pixel_values``/``image_grid_thw`` and the tower's pooled output
(``backbone.get_image_features(...).pooler_output``, f32).  ``decision_vision_probe``
writes the same two tensors as raw little-endian f32 plus a ``meta.json``.  This
script loads both and reports, per image:

* ``max|Δpixel|`` over the processor's ``pixel_values`` (gate ``<= 1e-6``; 0 is
  what a faithful processor gives, since both sides compute
  ``(u8 - 127.5) / 127.5``);
* the pooled cosine and ``max|Δ|`` (gate: cosine ``>= 0.999999``, and
  ``--pooled-max-abs`` when given);
* the patch grid and token count (must be exact).

The pooled gate is tight because a wrong activation in the merger (exact erf vs
the tanh approximation, which Phase 2a's review caught) moves the cosine to
0.99999996 at 1.3e-3 ``max|Δ|``; the fixed tower scores 1 - 4e-10 with a
2.4e-5 to 1.7e-4 relative ``max|Δ|``.  That residual is the attention kernel, not
this implementation: swapping torch's own SDPA backend moves the reference pooled
output by 4.4e-5 to 1.9e-4 absolute, so ``--pooled-max-abs`` is an advisory
opt-in (a useful band is 2e-3), not the gate.

Usage:
    vision_parity.py --ref DIR --probe DIR [--stem NAME] [--json OUT]
                     [--pixel-tol 1e-6] [--pooled-cosine 0.999999]
                     [--pooled-max-abs X]

``--ref`` is the reference dump directory and ``--probe`` the probe's output
directory.  Without ``--stem`` every entry of the dump's ``index.json`` is
scored, each against its own probe run: pass ``--probe`` once per entry (in
index order) if the probes live in separate directories, otherwise the probe
must have been produced for exactly one case (its ``meta.json`` records one
image set).

Exit code 0 when every checked case passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def read_f32(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size == 0:
        raise SystemExit(f"{path}: empty f32 file")
    return values.astype(np.float64)


def shape_from_meta(meta: dict) -> tuple[int, int, int, int]:
    pixel = [int(v) for v in meta["pixel_shape"]]
    pooled = [int(v) for v in meta["pooled_shape"]]
    if len(pixel) != 2 or len(pooled) != 2:
        raise SystemExit("probe meta.json: pixel_shape/pooled_shape must be 2-D")
    return pixel[0], pixel[1], pooled[0], pooled[1]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / (na * nb))


def score_case(
    ref_dir: Path,
    entry: dict,
    probe_dir: Path,
    pixel_tol: float,
    pooled_min_cosine: float,
    pooled_max_abs: float | None = None,
) -> tuple[bool, dict]:
    files = entry["files"]
    reference = np.load(ref_dir / files["pixel_values"]).astype(np.float64)
    pooled_ref = np.load(ref_dir / files["pooled"]).astype(np.float64)
    grid_ref = np.load(ref_dir / files["image_grid_thw"]).astype(np.int64)
    meta = json.loads((probe_dir / "meta.json").read_text(encoding="utf-8"))
    p_rows, p_dim, t_rows, t_dim = shape_from_meta(meta)
    pixels = read_f32(probe_dir / "pixel_values.f32").reshape(p_rows, p_dim)
    pooled = read_f32(probe_dir / "pooled.f32").reshape(t_rows, t_dim)

    report: dict = {
        "case": entry["case"],
        "question": entry["question"],
        "probe": str(probe_dir),
        "reference_pixels": list(reference.shape),
        "probe_pixels": [p_rows, p_dim],
        "reference_pooled": list(pooled_ref.shape),
        "probe_pooled": [t_rows, t_dim],
        "grids_match": True,
        "tokens_match": True,
        "problems": [],
    }

    # Grid/token agreement: the reference grid is [n, 3]; the probe records the
    # same list (one entry per image).
    probe_grids = [list(map(int, image["grid"])) for image in meta["images"]]
    reference_grids = [[int(v) for v in row] for row in grid_ref.tolist()]
    report["reference_grid"] = reference_grids
    report["probe_grid"] = probe_grids
    if reference_grids != probe_grids:
        report["grids_match"] = False
        report["problems"].append(f"grid {probe_grids} != reference {reference_grids}")
    probe_tokens = sum(int(image["tokens"]) for image in meta["images"])
    reference_tokens = int(sum(int(g) for g in entry["pooled_rows"]))
    report["tokens"] = {"reference": reference_tokens, "probe": probe_tokens}
    if reference_tokens != probe_tokens:
        report["tokens_match"] = False
        report["problems"].append(
            f"tokens {probe_tokens} != reference {reference_tokens}"
        )

    if reference.shape != pixels.shape:
        report["problems"].append(
            f"pixel_values shape {pixels.shape} != reference {reference.shape}"
        )
        report["pixel_max_abs"] = None
    else:
        delta = np.abs(pixels - reference)
        report["pixel_max_abs"] = float(delta.max()) if delta.size else 0.0
        report["pixel_nonzero"] = int(np.count_nonzero(delta))
        if report["pixel_max_abs"] > pixel_tol:
            report["problems"].append(
                f"pixel max|d| {report['pixel_max_abs']:.3e} > {pixel_tol:.1e}"
            )

    if pooled_ref.shape != pooled.shape:
        report["problems"].append(
            f"pooled shape {pooled.shape} != reference {pooled_ref.shape}"
        )
        report["pooled_cosine"] = None
    else:
        report["pooled_cosine"] = cosine(pooled.ravel(), pooled_ref.ravel())
        report["pooled_max_abs"] = float(np.abs(pooled - pooled_ref).max())
        report["pooled_scale"] = float(np.abs(pooled_ref).max())
        if not (report["pooled_cosine"] >= pooled_min_cosine):
            report["problems"].append(
                f"pooled cosine {report['pooled_cosine']:.8f} < {pooled_min_cosine}"
            )
        if pooled_max_abs is not None and not (report["pooled_max_abs"] <= pooled_max_abs):
            report["problems"].append(
                f"pooled max|d| {report['pooled_max_abs']:.3e} > {pooled_max_abs:.1e}"
            )

    return (not report["problems"]), report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ref", required=True, help="reference --dump-vision directory")
    p.add_argument(
        "--probe",
        required=True,
        action="append",
        help="probe output directory (repeat once per reference entry, in index order)",
    )
    p.add_argument("--stem", help="only the entry whose case/question stem matches this")
    p.add_argument("--json", help="write the report as JSON here")
    p.add_argument("--pixel-tol", type=float, default=1e-6)
    p.add_argument("--pooled-cosine", type=float, default=0.999999)
    p.add_argument(
        "--pooled-max-abs",
        type=float,
        default=None,
        help="also require max|Δpooled| <= this (advisory: the residual is the "
        "attention kernel, up to ~9e-4 absolute)",
    )
    args = p.parse_args(argv)

    ref_dir = Path(args.ref).expanduser()
    index_path = ref_dir / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"{ref_dir}: no index.json (not a --dump-vision directory)")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index["entries"]
    if args.stem:
        entries = [
            e
            for e in entries
            if f"{e['case']}__{e['question']}" == args.stem
            or e["case"] == args.stem
        ]
        if not entries:
            raise SystemExit(f"{index_path}: no entry for stem {args.stem!r}")
    probes = [Path(path).expanduser() for path in args.probe]
    if len(probes) == 1 and len(entries) > 1:
        raise SystemExit(
            f"{len(entries)} reference entries but one --probe; pass one --probe per entry "
            "(in index order) or select a single --stem"
        )
    if len(probes) not in (1, len(entries)):
        raise SystemExit(f"{len(probes)} --probe directories for {len(entries)} entries")

    print(
        f"reference: {ref_dir} ({len(entries)} entr{'y' if len(entries) == 1 else 'ies'}, "
        f"dtype {index.get('dtype')})"
    )
    reports = []
    failures = 0
    for i, entry in enumerate(entries):
        probe_dir = probes[0] if len(probes) == 1 else probes[i]
        ok, report = score_case(
            ref_dir,
            entry,
            probe_dir,
            args.pixel_tol,
            args.pooled_cosine,
            args.pooled_max_abs,
        )
        failures += 0 if ok else 1
        reports.append(report)
        pixel = (
            "n/a" if report["pixel_max_abs"] is None else f"{report['pixel_max_abs']:.3e}"
        )
        pooled = (
            "n/a"
            if report["pooled_cosine"] is None
            else f"{report['pooled_cosine']:.9f}"
        )
        max_abs = (
            "n/a" if "pooled_max_abs" not in report else f"{report['pooled_max_abs']:.3e}"
        )
        print(
            f"  {'PASS' if ok else 'FAIL'} {report['case']}/{report['question']}: "
            f"pixel max|d| {pixel}, pooled cos {pooled} (max|d| {max_abs}), "
            f"grid {report['probe_grid']}, tokens {report['tokens']['probe']}"
        )
        for problem in report["problems"]:
            print(f"        ! {problem}")

    summary = {
        "schema": "d3-vision-parity-v1",
        "reference": str(ref_dir),
        "pixel_tol": args.pixel_tol,
        "pooled_min_cosine": args.pooled_cosine,
        "pooled_max_abs": args.pooled_max_abs,
        "entries": len(entries),
        "failures": failures,
        "reports": reports,
    }
    if args.json:
        out = Path(args.json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    worst_pixel = max(
        (r["pixel_max_abs"] for r in reports if r["pixel_max_abs"] is not None),
        default=None,
    )
    cosines = [r["pooled_cosine"] for r in reports if r["pooled_cosine"] is not None]
    worst_cos = min(cosines) if cosines else None
    pixel_text = "n/a" if worst_pixel is None else f"{worst_pixel:.3e}"
    cos_text = "n/a" if worst_cos is None else f"{worst_cos:.9f}"
    print(
        f"RESULT: {'PASS' if failures == 0 else 'FAIL'} "
        f"({len(entries) - failures}/{len(entries)}, worst pixel max|d| {pixel_text}, "
        f"worst pooled cosine {cos_text})"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
