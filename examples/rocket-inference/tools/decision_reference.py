#!/usr/bin/env python3
"""Ground-truth d3 ("typed decision") answers from the checkpoint's official runtime.

d3 checkpoints ship their own runtime (``d3_runtime.py`` / ``d3_format.py`` with
``auto_map.AutoModel = modeling_d3.D3Model``); this script loads the checkpoint through
``AutoModel.from_pretrained(..., trust_remote_code=True)`` and calls the runtime's own
``prepare`` / ``run`` / ``respond`` -- the three calls ``D3Model.system_one`` makes -- so
the prompts, the answer codes, the masking and the reported answers are the official
ones.  Nothing of the answer logic is reimplemented here.

The ``system_one`` return value is ``{"model", "answers", "usage"}``; each answer is
``{"type": "choice", "choice": key, "probabilities": {key: p}, "confidence"}``,
``{"type": "noul", "noul": p_true}`` (a scalar: the official noul answer carries no
per-option dict) or ``{"type": "score", "score": E[level], "probabilities": {...},
"confidence", "legend"}``.  This script records that object verbatim as
``cases[].response`` and additionally the raw per-option probability vector in prompt
order, which is what a parity run compares.

Request file: a JSON array of ``{"name": str, "state": <string or JSON value>,
"questions": {qid: {"type": "choice"|"noul"|"score", "instructions": str,
"criteria": ...}}}`` -- exactly the arguments of one ``system_one`` call.  Text only:
``images``/``videos`` are not part of the request objects.

Usage:
    decision_reference.py --model /path/to/d3-lite --requests fixtures.json \
        --out ref.json [--dump DIR] [--limit N] [--batch-size 1]

Output (``schema: d3-decision-reference-v1``):
    {
      "schema": "d3-decision-reference-v1",
      "model": {"name", "dir", "backbone_dtype", "device", "batch_size", "runtime",
                "decision_config", "hidden_size", "codes", "token_ids"},
      "cases": [{
        "name", "elapsed_s", "request": {"state", "questions"}, "request_sha256",
        "response": {"model", "answers", "usage"},        # official system_one output
        "questions": {qid: {
            "type", "options": [...], "codes": [...], "tokens",
            "prompt_sha256", "probabilities": [...], "top", "answer": {...}}}
      }],
      "totals": {"cases", "questions", "tokens", "wall_s"}
    }

``--dump DIR`` writes, per question, the exact rendered prompt (``.prompt.txt``), the
input token ids (``.ids.npy``), the final-normed hidden state at the last prompt
position (``.hidden.npy``, float32) and the raw 255 code logits before masking and
temperature (``.logits.npy``, float32), plus a ``DIR/index.json`` listing them.  The
hidden is the runtime's own encoder output, recorded by wrapping it; the logits are
``hidden.float() @ readout.T`` exactly as the runtime computes them (``--check-dump``
re-runs the runtime's ``logits`` for the first dumped question and asserts equality).

dtype: the official runtime fixes BF16 for the backbone (its ``D3Model`` refuses a
``dtype`` argument).  ``--backbone-dtype f32`` keeps the official runtime and casts the
loaded backbone to float32 -- the checkpoint stores bf16, so the weights are upcast and
the body computes in f32, which is the closest reference for an f32 Rust body.  It is
also ~2x faster on CPU.  ``bf16`` (default) is the byte-faithful official path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "d3-decision-reference-v1"
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize(text: str) -> str:
    return SAFE.sub("_", text).strip("_") or "unnamed"


def canonical_sha256(value) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="local d3 checkpoint directory")
    p.add_argument("--requests", required=True, help="JSON array of {name, state, questions}")
    p.add_argument("--out", required=True, help="reference JSON to write")
    p.add_argument("--dump", help="directory for per-question prompt/ids/hidden/logits dumps")
    p.add_argument("--limit", type=int, default=None, help="only the first N cases")
    p.add_argument("--batch-size", type=int, default=1,
                   help="questions per forward pass (default 1: no left padding)")
    p.add_argument("--device", default="cpu", help="cpu or cuda:N (default cpu)")
    p.add_argument("--backbone-dtype", choices=("bf16", "f32"), default="bf16",
                   help="bf16 = the official runtime as shipped; f32 = cast the loaded backbone")
    p.add_argument("--check-dump", action="store_true",
                   help="verify the first dumped logits against the runtime's own logits()")
    p.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    p.add_argument("--verify", choices=("fast", "full", "none"), default="fast",
                   help="checkpoint manifest verification of the official loader")
    return p.parse_args(argv)


def load_requests(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"{path}: expected a nonempty JSON array of cases")
    names = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"name", "state", "questions"}:
            raise SystemExit(f"{path}: case {i} must hold exactly name, state, questions")
        name = case["name"]
        if not isinstance(name, str) or not name:
            raise SystemExit(f"{path}: case {i} has no name")
        if name in names:
            raise SystemExit(f"{path}: duplicate case name {name!r}")
        names.add(name)
        if not isinstance(case["questions"], dict) or not case["questions"]:
            raise SystemExit(f"{path}: case {name!r} has no questions")
    return cases


class Recorder:
    """Record the encoder output the runtime actually reads, without changing it."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.original_encoder = runtime.encoder
        self.original_logits = runtime.logits
        self.pending = None
        self.frames: list[tuple[list[list[int]], "object"]] = []

    def __enter__(self):
        runtime, recorder = self.runtime, self

        def logits(sequences, counts):
            recorder.pending = [list(s) for s in sequences]
            return recorder.original_logits(sequences, counts)

        def encoder(embeds, present, mrope=None):
            out = (
                recorder.original_encoder(embeds, present, mrope)
                if mrope is not None
                else recorder.original_encoder(embeds, present)
            )
            if recorder.pending is not None:
                if out.shape[0] != len(recorder.pending):
                    raise RuntimeError("encoder rows do not match the batch")
                recorder.frames.append((recorder.pending, out[:, -1].detach().float().cpu()))
                recorder.pending = None
            return out

        runtime.logits = logits
        runtime.encoder = encoder
        return self

    def __exit__(self, *exc):
        self.runtime.logits = self.original_logits
        self.runtime.encoder = self.original_encoder
        return False

    def hidden_by_sequence(self) -> dict[tuple[int, ...], "object"]:
        table = {}
        for sequences, rows in self.frames:
            for sequence, row in zip(sequences, rows):
                table[tuple(sequence)] = row
        return table


def describe_question(runtime, prepared, qid: str) -> tuple[list[str], list[str]]:
    """Option labels (report order) and the prompt codes used for them."""
    question = prepared.questions[qid]
    labels = list(question.keys)
    codes = runtime.codes[: len(labels)]
    return labels, codes


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch
    from transformers import AutoModel

    if args.threads:
        torch.set_num_threads(args.threads)

    model_dir = Path(args.model).expanduser().resolve()
    if not (model_dir / "decision_config.json").is_file():
        raise SystemExit(f"{model_dir}: not a d3 checkpoint (no decision_config.json)")

    started = time.perf_counter()
    print(f"loading {model_dir} (device={args.device}, batch_size={args.batch_size}, "
          f"backbone_dtype={args.backbone_dtype})", flush=True)
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        device=args.device,
        batch_size=args.batch_size,
        verify=args.verify,
    )
    runtime = model.runtime
    if runtime is None:
        raise SystemExit("the checkpoint did not expose the d3 runtime")
    if runtime.encoder is None:
        raise SystemExit("this checkpoint is not a noncausal d3 model; --dump/hidden needs it")
    if args.backbone_dtype == "f32":
        runtime.backbone.float()
    backbone_dtype = str(next(runtime.backbone.parameters()).dtype)
    hidden_size = runtime.backbone.config.text_config.hidden_size
    print(
        f"loaded in {time.perf_counter() - started:.1f}s: {type(model).__name__} "
        f"backbone={backbone_dtype} readout={runtime.readout.dtype} "
        f"hidden={hidden_size} temperature={runtime.temperature} "
        f"attention={runtime.attention_mode} kernels={runtime.kernels}",
        flush=True,
    )

    cases = load_requests(Path(args.requests))
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be >= 0")
        cases = cases[: args.limit]

    dump_dir = Path(args.dump).expanduser() if args.dump else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    out_cases = []
    dump_index = []
    total_tokens = 0
    total_questions = 0
    run_started = time.perf_counter()

    for index, case in enumerate(cases, 1):
        name = case["name"]
        case_started = time.perf_counter()
        try:
            prepared = runtime.prepare(case["state"], case["questions"])
        except ValueError as exc:
            raise SystemExit(f"{name}: the runtime rejected the request: {exc}") from exc
        errors = dict(prepared.errors)
        for qid, error in errors.items():
            print(f"  ! {name}/{qid}: {error.get('error')}: {error.get('message')}", flush=True)

        with Recorder(runtime) as recorder:
            probabilities, tokens = runtime.run(prepared)
        response = runtime.respond(prepared, probabilities, tokens)
        hidden = recorder.hidden_by_sequence()

        questions = {}
        for qid in prepared.keys:
            if qid in errors:
                questions[qid] = {"type": prepared.errors[qid].get("type"), "error": prepared.errors[qid]}
                continue
            labels, codes = describe_question(runtime, prepared, qid)
            sequence = prepared.sequences[qid]
            prompt = runtime.text(case["state"], prepared.questions[qid].rendered)
            encoded = runtime.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if encoded != sequence:
                raise RuntimeError(f"{name}/{qid}: rendered prompt does not re-tokenize to its ids")
            values = [float(p) for p in probabilities[qid]]
            if len(values) != len(labels):
                raise RuntimeError(f"{name}/{qid}: {len(values)} probabilities for {len(labels)} options")
            best = max(range(len(values)), key=lambda i: (values[i], -i))
            record = {
                "type": prepared.questions[qid].kind,
                "options": labels,
                "codes": codes,
                "tokens": len(sequence),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "probabilities": values,
                "top": labels[best],
                "answer": response["answers"].get(qid),
            }
            questions[qid] = record
            total_questions += 1
            if dump_dir is not None:
                row = hidden.get(tuple(sequence))
                if row is None:
                    raise RuntimeError(f"{name}/{qid}: no recorded final hidden for its forward")
                stem = f"{sanitize(name)}__{sanitize(qid)}"
                logits = (row.float() @ runtime.readout.T.cpu()).numpy().astype(np.float32)
                if args.check_dump and not dump_index:
                    reference = runtime.logits([sequence], [len(labels)])[0]
                    reference = reference[torch.isfinite(reference)].float().numpy()
                    if not np.array_equal(logits[: len(reference)], reference):
                        raise RuntimeError(f"{name}/{qid}: dumped logits differ from runtime.logits")
                    print("  dump check: hidden/logits match runtime.logits", flush=True)
                (dump_dir / f"{stem}.prompt.txt").write_text(prompt, encoding="utf-8")
                np.save(dump_dir / f"{stem}.ids.npy", np.asarray(sequence, dtype=np.int32))
                np.save(dump_dir / f"{stem}.hidden.npy", row.numpy().astype(np.float32))
                np.save(dump_dir / f"{stem}.logits.npy", logits)
                dump_index.append({
                    "case": name,
                    "question": qid,
                    "files": {
                        "prompt": f"{stem}.prompt.txt",
                        "ids": f"{stem}.ids.npy",
                        "hidden": f"{stem}.hidden.npy",
                        "logits": f"{stem}.logits.npy",
                    },
                    "tokens": len(sequence),
                    "top": labels[best],
                })

        elapsed = time.perf_counter() - case_started
        total_tokens += tokens
        tops = ", ".join(f"{qid}={questions[qid].get('top', '?')}" for qid in prepared.keys)
        print(f"[{index}/{len(cases)}] {name} q={len(case['questions'])} tokens={tokens} "
              f"{elapsed:.2f}s {tops}", flush=True)
        out_cases.append({
            "name": name,
            "elapsed_s": round(elapsed, 3),
            "request": {"state": case["state"], "questions": case["questions"]},
            "request_sha256": canonical_sha256({"state": case["state"], "questions": case["questions"]}),
            "response": response,
            "questions": questions,
        })

    wall = time.perf_counter() - run_started
    reference = {
        "schema": SCHEMA,
        "model": {
            "name": runtime.model_name,
            "dir": str(model_dir),
            "runtime": f"vllm-sr d3 runtime (transformers {__import__('transformers').__version__}, "
                       f"torch {torch.__version__})",
            "backbone_dtype": backbone_dtype,
            "readout_dtype": str(runtime.readout.dtype),
            "device": str(runtime.device),
            "batch_size": runtime.batch_size,
            "attention_mode": runtime.attention_mode,
            "pooling": runtime.config.get("pooling"),
            "temperature": runtime.temperature,
            "hidden_size": hidden_size,
            "codes": list(runtime.codes),
            "token_ids": list(runtime.token_ids),
            "decision_config": runtime.config,
        },
        "cases": out_cases,
        "totals": {
            "cases": len(out_cases),
            "questions": total_questions,
            "tokens": total_tokens,
            "wall_s": round(wall, 3),
        },
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if dump_dir is not None:
        (dump_dir / "index.json").write_text(
            json.dumps({"schema": SCHEMA, "model": reference["model"]["name"],
                        "dtype": backbone_dtype, "entries": dump_index}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {out_path} ({len(out_cases)} cases, {total_questions} questions, "
          f"{total_tokens} tokens, {wall:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
