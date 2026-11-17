#!/usr/bin/env python3
"""GGUF tokenizer parity check: the rocket-inference tokenizer vs the reference.

Compares, per input text, the token ids produced by:
  1. the repo binary (`<family> tokenize --model-dir <dir> --text <text>`),
  2. the checkpoint's reference `tokenizer.json` (HF `tokenizers`), and
  3. `llama-tokenize` on the same GGUF (when the binary is installed).

All three add the checkpoint's BOS/EOS, so the ids must match exactly. `--ref` must
be the checkpoint's *own* `tokenizer.json`: related Gemma checkpoints share the
vocab/merges but not the BOS/EOS post-processor (embeddinggemma-2 adds both, the
gemma-4-*-it chat models add neither); such a pure specials-policy difference is
reported as `OK~` (segments match) instead of a failure. Run with a python that has
`tokenizers` (on this host: /mnt/hub/venvs/emb2/bin/python).

Usage:
    python3 tools/tokenizer_parity.py --gguf MODEL.gguf --ref tokenizer.json \
        [--binary PATH] [--family gemma] [--model-dir DIR] \
        [--text TEXT]... [--text-file FILE]...

Without `--text`/`--text-file` the built-in word list runs: it covers the German
words whose merge-based segmentation a score-based Unigram path gets wrong
(the Gemma 4 GGUF regression, 2026-10-16).

The default `--binary` is `$CARGO_TARGET_DIR/release/rocket-inference` (falling
back to `target/release/rocket-inference`); `--model-dir` defaults to a temp dir
holding a symlink to `--gguf` (the binary takes a model *directory*).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Built-in corpus: words that regressed under the score-based Unigram path, a
# few passing controls, and space-prefixed forms.
DEFAULT_WORDS = [
    "Dokumenten",
    "Dokumente",
    "Dokument",
    "dokumenten",
    "Ablage",
    "Ablagen",
    "ablage",
    "asynchron",
    "Asynchron",
    "asynchronisation",
    "synchron",
    "Berechnung",
    "Berechnungen",
    "ganze",
    "Indizierung",
    "erfolgt",
    " Dateien",
    "x Dokumenten",
    "Dokumenten ",
]


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.environ.get("CARGO_TARGET_DIR", os.path.join(here, "..", "..", "..", "target"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True, help="GGUF checkpoint (for llama-tokenize)")
    ap.add_argument("--ref", required=True, help="reference tokenizer.json")
    ap.add_argument(
        "--binary",
        default=os.path.join(target, "release", "rocket-inference"),
        help="rocket-inference binary",
    )
    ap.add_argument(
        "--family", default="gemma", help="tokenize subcommand family (gemma|qwen3)"
    )
    ap.add_argument(
        "--model-dir",
        help="directory holding the GGUF (default: a temp dir with a symlink)",
    )
    ap.add_argument("--text", action="append", default=[], help="input text (repeatable)")
    ap.add_argument("--text-file", action="append", default=[], help="input text file (repeatable)")
    ap.add_argument(
        "--llama-tokenize", default="llama-tokenize", help="llama.cpp tokenize binary"
    )
    args = ap.parse_args()
    if not os.path.exists(args.binary):
        ap.error(f"binary not found: {args.binary} (build it or pass --binary)")
    if not args.text and not args.text_file:
        args.text = list(DEFAULT_WORDS)
    return args


def binary_ids(args: argparse.Namespace, model_dir: str, text: str) -> list[int]:
    out = subprocess.run(
        [args.binary, args.family, "tokenize", "--model-dir", model_dir, "--text", text],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"binary tokenize failed:\n{out.stdout}\n{out.stderr}")
    lines = out.stdout.strip().splitlines()
    ids = [int(t) for t in re.split(r"[,\s]+", lines[-1].strip().strip("[]")) if t]
    count = int(re.match(r"(\d+) tokens", lines[-2]).group(1))
    assert len(ids) == count, f"binary reported {count} tokens but printed {len(ids)}"
    return ids


def llama_ids(args: argparse.Namespace, text: str) -> list[int] | None:
    if not shutil.which(args.llama_tokenize):
        return None
    out = subprocess.run(
        [args.llama_tokenize, "-m", args.gguf, "-p", text], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise SystemExit(f"llama-tokenize failed:\n{out.stdout}\n{out.stderr}")
    ids = []
    for line in out.stdout.splitlines():
        m = re.match(r"\s*(\d+) -> '", line)
        if m:
            ids.append(int(m.group(1)))
    return ids


def first_diff(a: list[int], b: list[int]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"index {i}: {x} != {y}"
    return f"length {len(a)} != {len(b)}"


def main() -> int:
    args = parse_args()
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise SystemExit("python needs the 'tokenizers' package (e.g. /mnt/hub/venvs/emb2)")

    ref = Tokenizer.from_file(args.ref)

    tmp = None
    if args.model_dir:
        model_dir = args.model_dir
    else:
        tmp = tempfile.TemporaryDirectory()
        model_dir = os.path.join(tmp.name, "model")
        os.makedirs(model_dir)
        os.symlink(os.path.abspath(args.gguf), os.path.join(model_dir, os.path.basename(args.gguf)))

    texts: list[tuple[str, str]] = [(f"--text {t!r}", t) for t in args.text]
    for path in args.text_file:
        with open(path, encoding="utf-8") as f:
            texts.append((path, f.read()))

    has_llama = shutil.which(args.llama_tokenize) is not None
    print(f"gguf   : {args.gguf}")
    print(f"ref    : {args.ref}")
    print(f"binary : {args.binary} ({args.family} tokenize)")
    print(f"llama  : {args.llama_tokenize if has_llama else '(not installed, skipped)'}")
    print(f"inputs : {len(texts)}")

    failures = 0
    seg_only = 0
    for label, text in texts:
        ours = binary_ids(args, model_dir, text)
        want = ref.encode(text, add_special_tokens=True).ids
        status = "OK  "
        note = ""
        if ours != want:
            # Same vocab, different BOS/EOS policy in the reference file?
            seg = ref.encode(text, add_special_tokens=False).ids
            hit = None
            for lead, trail in ((1, 1), (1, 0), (0, 1)):
                if ours[lead : len(ours) - trail] == seg:
                    hit = (lead, trail)
            if hit is None:
                status = "BAD "
                failures += 1
                note = f"  [ref: {first_diff(ours, want)}]"
            else:
                status = "OK~ "
                seg_only += 1
                lead, trail = hit
                note = f"  [segments match; ref policy +{lead}/+{trail} specials]"
        llama = llama_ids(args, text) if has_llama else None
        if llama is not None and llama != ours:
            note += f"  [llama differs: {first_diff(ours, llama)}]"
        print(f"  {status} {label:44} n={len(ours)}{note}")

    summary = f"{len(texts) - failures - seg_only}/{len(texts)} exactly match the reference"
    if seg_only:
        summary += f", {seg_only} segment-only (ref BOS/EOS policy differs)"
    if failures:
        summary += f", {failures} MISMATCHED"
    if has_llama and not failures:
        summary += " (llama-tokenize agrees)"
    print(summary)
    if tmp is not None:
        tmp.cleanup()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
