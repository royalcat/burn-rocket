#!/usr/bin/env python3
"""Compare our `decide` CLI against the official d3 reference, one case per process.

The reference file is the output of ``decision_reference.py``
(``schema: d3-decision-reference-v1``); the fixtures are the same JSON array both sides
ran.  For every case this script writes a one-case request file, runs

    <bin> decide --model-dir <DIR> --requests <case.json> --out <case-out.json> [--arg ...]

and compares, per question, the top option and the option probabilities.

Expected CLI output (``schema: d3-decision-v1``): a JSON object with a ``cases`` array;
each case has ``name`` and ``questions``; each question has ``type``, ``options`` (the
option labels in prompt order), ``codes`` (the prompt code per option) and
``probabilities`` (one probability per option, same order).  Missing optional fields are
filled from the reference: a question without ``options`` is assumed to use the
reference's option order, and probabilities given as an object are matched by label.
A bare top-level array of cases is accepted too.

Exit status: 0 when every question's top option matches and no |delta probability| is
above ``--max-dp``; 1 otherwise (missing cases/questions, CLI failures and code
mismatches count as failures).

Usage:
    decision_parity.py --bin target/release/rocket-inference --model-dir /models/d3-lite \
        --requests tools/decision_fixtures.json --ref /tmp/opencode/d3/ref_text_lite.json \
        --out /tmp/opencode/d3/parity_report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "d3-decision-v1"
REF_SCHEMA = "d3-decision-reference-v1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", required=True, help="our binary (runs `<bin> decide ...`)")
    p.add_argument("--requests", required=True, help="the fixtures JSON array")
    p.add_argument("--ref", required=True, help="reference JSON from decision_reference.py")
    p.add_argument("--model-dir", default=None,
                   help="checkpoint directory for the CLI (default: the reference's model.dir)")
    p.add_argument("--limit", type=int, default=None, help="only the first N cases")
    p.add_argument("--out", default=None, help="write a JSON report here")
    p.add_argument("--work", default=None, help="keep per-case files in this directory")
    p.add_argument("--keep", action="store_true", help="keep a temporary work directory")
    p.add_argument("--arg", action="append", default=[], metavar="FLAG",
                   help="extra CLI argument, repeatable (inserted before --requests)")
    p.add_argument("--command", default="decide", help="CLI subcommand (default decide)")
    p.add_argument(
        "--media-base",
        default=None,
        help="directory a case's relative images/videos paths resolve from "
        "(default: the parent of the requests file's directory, i.e. "
        "examples/rocket-inference)",
    )
    p.add_argument("--timeout", type=float, default=None, help="seconds per case (default none)")
    p.add_argument("--min-top1", type=float, default=100.0,
                   help="required top-1 match percentage (default 100)")
    p.add_argument("--max-dp", type=float, default=2e-3,
                   help="largest allowed |delta probability| (default 2e-3)")
    return p.parse_args(argv)


def media_base(args) -> Path:
    """Where relative `images`/`videos` paths are resolved from.

    The fixtures live in `examples/rocket-inference/tools/` and name their media
    relative to `examples/rocket-inference` (`data/cat.jpeg`, `data/test.mp4`);
    the CLI resolves a path against its own working directory, so the gate must
    work from any cwd. The default is the parent of the requests file's directory.
    """
    if args.media_base:
        return Path(args.media_base).expanduser().resolve()
    return Path(args.requests).expanduser().resolve().parent.parent


def resolve_media(case: dict, base: Path) -> dict:
    """A copy of one case with its relative media paths made absolute."""
    resolved = dict(case)
    for field in ("images", "videos"):
        entries = resolved.get(field)
        if not isinstance(entries, list):
            continue
        paths = []
        for entry in entries:
            if (
                isinstance(entry, str)
                and entry
                and not entry.startswith(("data:", "http://", "https://"))
                and not Path(entry).is_absolute()
            ):
                paths.append(str(base / entry))
            else:
                paths.append(entry)
        resolved[field] = paths
    return resolved


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def as_cases(blob) -> list[dict]:
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        cases = blob.get("cases")
        if isinstance(cases, list):
            return cases
    raise SystemExit("expected a JSON array of cases or an object with a cases array")


def option_order(record: dict, expected: list[str], name: str, qid: str) -> list[str]:
    if record.get("options") is None:
        return expected
    options = record["options"]
    if not isinstance(options, list) or len(options) != len(expected):
        raise SystemExit(f"{name}/{qid}: CLI options {options!r} do not match {expected!r}")
    if list(options) != list(expected):
        raise SystemExit(f"{name}/{qid}: option order {options!r} differs from {expected!r}")
    return options


def questions_of(out_case: dict, expected_case: dict) -> dict:
    """Our per-question records, falling back to a response shaped like the official one."""
    questions = out_case.get("questions")
    if isinstance(questions, dict) and questions:
        return questions
    answers = (out_case.get("response") or {}).get("answers")
    if not isinstance(answers, dict):
        answers = out_case.get("answers")
    if not isinstance(answers, dict):
        return {}
    derived = {}
    for qid, answer in answers.items():
        if not isinstance(answer, dict):
            continue
        record = {"type": answer.get("type")}
        values = answer.get("probabilities")
        if isinstance(values, dict):
            record["options"] = list(values)
            record["probabilities"] = values
        elif answer.get("type") == "noul" and isinstance(answer.get("noul"), (int, float)):
            p = float(answer["noul"])
            record["options"] = ["false", "true"]
            record["probabilities"] = [1.0 - p, p]
        else:
            continue
        derived[qid] = record
    return derived



def probabilities_of(record: dict, options: list[str], name: str, qid: str) -> list[float]:
    values = record.get("probabilities")
    if values is None:
        answer = (record.get("answer") or {})
        values = answer.get("probabilities") if isinstance(answer, dict) else None
    if isinstance(values, dict):
        missing = [k for k in options if k not in values]
        if missing:
            raise SystemExit(f"{name}/{qid}: probabilities miss options {missing}")
        values = [float(values[k]) for k in options]
    if not isinstance(values, list) or len(values) != len(options):
        raise SystemExit(f"{name}/{qid}: expected {len(options)} probabilities, got {values!r}")
    return [float(v) for v in values]


def main(argv=None) -> int:
    args = parse_args(argv)
    ref = load_json(Path(args.ref))
    if not isinstance(ref, dict) or ref.get("schema") != REF_SCHEMA:
        raise SystemExit(f"{args.ref}: not a {REF_SCHEMA} reference file")
    model_dir = args.model_dir or ref["model"]["dir"]
    binary = Path(args.bin).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"{binary}: not a file")

    fixtures = load_json(Path(args.requests))
    requests = {case["name"]: case for case in as_cases(fixtures)}
    base = media_base(args)
    print(f"media base: {base}")
    expected = ref["cases"]
    if args.limit is not None:
        expected = expected[: args.limit]
    if not expected:
        raise SystemExit("nothing to compare")

    work = Path(args.work).expanduser() if args.work else Path(tempfile.mkdtemp(prefix="d3-parity-"))
    work.mkdir(parents=True, exist_ok=True)
    cleanup = args.work is None and not args.keep

    results = []
    failures = []
    totals = {"questions": 0, "top1": 0, "max_dp": 0.0, "max_dp_where": None}
    per_type: dict[str, dict] = {}
    wall = time.perf_counter()

    for index, ref_case in enumerate(expected, 1):
        name = ref_case.get("name")
        if name not in requests:
            failures.append(f"{name}: missing from the requests file")
            print(f"! {name}: no request")
            continue
        stem = f"{index:03d}-{''.join(c if c.isalnum() else '_' for c in str(name))}"
        request_file = work / f"{stem}.json"
        output_file = work / f"{stem}.out.json"
        case = resolve_media(requests[name], base)
        request_file.write_text(json.dumps([case], ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        command = [str(binary), args.command, *args.arg, "--model-dir", str(model_dir),
                   "--requests", str(request_file), "--out", str(output_file)]
        started = time.perf_counter()
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            failures.append(f"{name}: CLI timed out after {args.timeout}s")
            print(f"! {name}: timeout")
            continue
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            failures.append(f"{name}: CLI exit {proc.returncode}: {' | '.join(tail)}")
            print(f"! {name}: CLI exit {proc.returncode}")
            continue
        if not output_file.is_file():
            failures.append(f"{name}: CLI wrote no {output_file.name}")
            print(f"! {name}: no output")
            continue
        produced = as_cases(load_json(output_file))
        if len(produced) != 1:
            failures.append(f"{name}: CLI answered {len(produced)} cases for one request")
            continue
        out_case = produced[0]
        if out_case.get("name") not in (None, name):
            failures.append(f"{name}: CLI answered case {out_case.get('name')!r}")
            print(f"! {name}: CLI answered a different case")
            continue
        out_questions = questions_of(out_case, ref_case)
        case_rows = []
        for qid, expected_q in ref_case["questions"].items():
            totals["questions"] += 1
            kind = expected_q.get("type") or "unknown"
            stats = per_type.setdefault(kind, {"questions": 0, "top1": 0, "max_dp": 0.0})
            stats["questions"] += 1
            row = {"case": name, "question": qid, "type": kind}
            if "error" in expected_q:
                row["skipped"] = "reference could not answer"
                case_rows.append(row)
                stats["questions"] -= 1
                totals["questions"] -= 1
                continue
            got = out_questions.get(qid)
            if not isinstance(got, dict):
                failures.append(f"{name}/{qid}: missing from the CLI output")
                row["error"] = "missing"
                case_rows.append(row)
                continue
            expected_options = list(expected_q["options"])
            options = option_order(got, expected_options, name, qid)
            values = probabilities_of(got, options, name, qid)
            reference = [float(p) for p in expected_q["probabilities"]]
            if len(reference) != len(values):
                failures.append(f"{name}/{qid}: {len(values)} probabilities vs {len(reference)} expected")
                row["error"] = "length"
                case_rows.append(row)
                continue
            best = max(range(len(values)), key=lambda i: (values[i], -i))
            top = options[best]
            top_ok = top == expected_q["top"]
            delta = max((abs(a - b) for a, b in zip(values, reference)), default=0.0)
            row.update({"top": top, "expected_top": expected_q["top"], "top_ok": top_ok,
                        "max_dp": delta})
            code = None
            codes = got.get("codes") or expected_q.get("codes")
            if isinstance(codes, list) and len(codes) == len(options):
                code = codes[best]
                row["code"] = code
                row["expected_code"] = expected_q["codes"][best] if expected_q.get("codes") else None
                if row["expected_code"] is not None and code != row["expected_code"]:
                    failures.append(f"{name}/{qid}: top code {code!r} != {row['expected_code']!r}")
                    row["code_ok"] = False
            totals["top1"] += int(top_ok)
            stats["top1"] += int(top_ok)
            stats["max_dp"] = max(stats["max_dp"], delta)
            results.append(row)
            case_rows.append(row)
            if totals["max_dp_where"] is None or delta > totals["max_dp"]:
                totals["max_dp"], totals["max_dp_where"] = delta, f"{name}/{qid}"
            if not top_ok or delta > args.max_dp:
                print(f"  {name}/{qid}: top {top} (expected {expected_q['top']}), "
                      f"max|dp|={delta:.3g}")
        print(f"[{index}/{len(expected)}] {name} {len(case_rows)} questions {elapsed:.2f}s")

    wall = time.perf_counter() - wall
    questions = totals["questions"]
    match = 100.0 * totals["top1"] / questions if questions else 0.0
    print()
    print(f"questions: {questions}")
    print(f"top-1 match: {totals['top1']}/{questions} ({match:.2f}%)")
    if totals["max_dp_where"]:
        print(f"max |dp|: {totals['max_dp']:.6g} at {totals['max_dp_where']}")
    else:
        print("max |dp|: n/a")
    for kind in sorted(per_type):
        stats = per_type[kind]
        if not stats["questions"]:
            continue
        rate = 100.0 * stats["top1"] / stats["questions"]
        print(f"  {kind:6s}: top-1 {stats['top1']}/{stats['questions']} ({rate:.2f}%), "
              f"max |dp| {stats['max_dp']:.6g}")
    print(f"wall: {wall:.1f}s")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for failure in failures[:40]:
            print(f"  {failure}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")

    if args.out:
        report = {
            "schema": "d3-decision-parity-v1",
            "reference": str(Path(args.ref).expanduser()),
            "requests": str(Path(args.requests).expanduser()),
            "binary": str(binary),
            "model_dir": str(model_dir),
            "limits": {"min_top1": args.min_top1, "max_dp": args.max_dp},
            "totals": {"cases": len(expected), "questions": questions, "top1": totals["top1"],
                       "top1_percent": round(match, 4), "max_dp": totals["max_dp"],
                       "max_dp_where": totals["max_dp_where"], "wall_s": round(wall, 3)},
            "per_type": {k: {**v, "top1_percent": round(100.0 * v["top1"] / v["questions"], 4)
                             if v["questions"] else None} for k, v in per_type.items()},
            "failures": failures,
            "questions": results,
        }
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")

    ok = not failures and match >= args.min_top1 and totals["max_dp"] <= args.max_dp
    print("RESULT:", "PASS" if ok else "FAIL")
    if cleanup:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
