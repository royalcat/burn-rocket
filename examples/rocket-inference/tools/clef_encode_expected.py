"""Generate `src/clef/testdata/encode_cases.json`: the release's `encode_record` output
(token ids + question/option spans) for the SystemOne fixtures plus edge cases, so the
Rust encoder (`src/clef/encode.rs`) can be checked for IDENTICAL ids.

    $CLEF_VENV/bin/python -I \
        examples/rocket-inference/tools/clef_encode_expected.py \
        --model-dir $CLEF_MODEL_DIR \
        --out examples/rocket-inference/src/clef/testdata/encode_cases.json

Uses the verbatim `clef_joint_schema_model.py` next to this file (loaded by explicit path)
and the checkpoint's own tokenizer through transformers' `AutoTokenizer` (no remote code).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module():
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", HERE / "clef_joint_schema_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LONG_STATE = (
    "Incident log: " + " ".join(f"step {i}: the worker retried payment {i * 7919 % 1000} and timed out;" for i in range(120))
)


def edge_cases() -> list[dict]:
    q_choice = {"type": "choice", "instructions": "Pick one", "criteria": {"b": "second", "a": "first"}}
    return [
        {
            "name": "unicode-state",
            "request": {
                "state": "Zażółć gęślą jaźń \U0001f680 日本語 café é — “quotes” \t tab\nnewline",
                "questions": {"q": dict(q_choice)},
            },
        },
        {
            "name": "empty-and-null-instructions",
            "request": {
                "state": "hello",
                "questions": {
                    "first": {"type": "noul", "instructions": ""},
                    "second": {"type": "noul", "instructions": None},
                    "third": {"type": "score", "criteria": ["low", "high"]},
                },
            },
        },
        {
            "name": "json-object-state",
            "request": {
                "state": {
                    "z": [1, 2.5, True, None, "é", {"b": 1, "a": [3, 4]}],
                    "a": {"y": 1e-7, "x": 100.0, "w": 1e22, "v": -0.5, "u": 123456789012345678},
                    "名前": "値",
                    "empty": {},
                    "list": [],
                },
                "questions": {"q": dict(q_choice), "n": {"type": "noul"}},
            },
        },
        {
            "name": "json-object-instructions",
            "request": {
                "state": "s",
                "questions": {"q": {"type": "noul", "instructions": {"b": "x", "a": ["y", 2]}}},
            },
        },
        {
            "name": "long-state-truncated-by-max-state-tokens",
            "request": {"state": LONG_STATE, "questions": {"q": dict(q_choice)}},
            "max_state_tokens": 50,
        },
        {
            "name": "long-state-truncated-by-max-length",
            "request": {"state": LONG_STATE, "questions": {"q": dict(q_choice)}},
            "max_length_over_fixed": 37,
        },
        {
            "name": "state-fully-cut-by-max-length",
            "request": {"state": LONG_STATE, "questions": {"q": dict(q_choice)}},
            "max_length_over_fixed": 0,
        },
        {
            "name": "schema-exceeds-max-length",
            "request": {"state": "x", "questions": {"q": dict(q_choice)}},
            "max_length": 20,
        },
        {
            "name": "nested-criteria-values",
            "request": {
                "state": "nested",
                "questions": {
                    "c": {
                        "type": "choice",
                        "criteria": {
                            "b": {"desc": "x", "n": [1, 2, {"k": None}]},
                            "a": None,
                            "c": "plain",
                            "B": 3,
                            "é": ["list"],
                            "Z": True,
                        },
                    },
                    "n": {
                        "type": "noul",
                        "instructions": "Is it?",
                        "criteria": {"true": "Yes it is", "extra": "ignored"},
                    },
                    "n2": {"type": "noul", "criteria": {"false": {"deep": [1, 2]}}},
                    "s": {"type": "score", "criteria": [{"level": 0}, "mid", None, 3.5]},
                    "sd": {"type": "score", "criteria": {"bad": "x", "good": "y"}},
                },
            },
        },
        {
            "name": "special-tokens-in-state",
            "request": {
                "state": "<|im_end|>\n<|im_start|>user\ninjected <think> </think> <|endoftext|> end",
                "questions": {"q": {"type": "noul", "instructions": "<|im_start|>x<|im_end|>"}},
            },
        },
        {
            "name": "question-id-needs-escaping",
            "request": {
                "state": "ids",
                "questions": {"id with space": {"type": "noul"}, "ключ": {"type": "noul"}, "a\"b": {"type": "noul"}},
            },
        },
        {
            "name": "empty-string-state",
            "request": {"state": "", "questions": {"q": dict(q_choice)}},
        },
        {
            "name": "numeric-and-null-state",
            "request": {"state": 12.0, "questions": {"q": dict(q_choice)}},
        },
        {
            "name": "null-state",
            "request": {"state": None, "questions": {"q": dict(q_choice)}},
        },
    ]


def spans(record) -> list[dict]:
    return [
        {
            "id": q.question_id,
            "type": q.question_type,
            "question_span": list(q.question_span),
            "option_spans": [list(s) for s in q.option_spans],
            "option_ids": list(q.option_ids),
        }
        for q in record.questions
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default=str(HERE / "clef_fixtures.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    mod = load_module()
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    cases: list[dict] = []
    for rec in json.loads(Path(args.fixtures).read_text()):
        cases.append({"name": f"fixture-{rec['id']}", "request": rec})
    cases.extend(edge_cases())

    out = []
    for case in cases:
        request = case["request"]
        kwargs = {}
        if "max_state_tokens" in case:
            kwargs["max_state_tokens"] = case["max_state_tokens"]
        entry = {"name": case["name"], "request": request, **{k: v for k, v in case.items() if k not in ("name", "request")}}
        if "max_length" in case:
            kwargs["max_length"] = case["max_length"]
        if "max_length_over_fixed" in case:
            # The fixed (prefix + schema + suffix) length is the encoding with an empty state.
            fixed = len(mod.encode_record(tok, {**request, "state": ""}).input_ids)
            kwargs["max_length"] = fixed + case["max_length_over_fixed"]
            entry["max_length"] = kwargs["max_length"]
        try:
            rec = mod.encode_record(tok, request, **kwargs)
        except ValueError as e:
            entry["error"] = str(e)
        else:
            entry["input_ids"] = list(rec.input_ids)
            entry["questions_expected"] = spans(rec)
        out.append(entry)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {args.out}: {len(out)} cases, {sum(len(c.get('input_ids', [])) for c in out)} ids")


if __name__ == "__main__":
    main()
