#!/usr/bin/env python3
"""Video parity: our `gemma embed` vs the HF f32 EmbeddingGemma 2 reference.

Checks the documented video input end to end: the sampled frame count, the token
ids of the expanded context (`BOI + <|video|> * n + EOI` per frame, concatenated
in one shared context) and the embedding cosine.

Cases:
  video        - video only (`data/test.mp4`: 4 s at 8 fps -> 4 frames at 1 fps)
  interleaved  - text + image + video in one shared context (the model card's
                 product-listing shape)
  overflow     - a generated 40 s clip: 40 sampled frames, uniform cap to 32

The reference runs through `tools/ref_embeddinggemma2.py mixed`, so this needs
that script's venv (transformers 5.19 + torch + numpy), ffmpeg/ffprobe and a
built rocket-inference binary:

  /mnt/hub/venvs/emb2/bin/python examples/rocket-inference/tools/video_parity.py \
      --bin $CARGO_TARGET_DIR/release/rocket-inference

Exit status is non-zero if any case fails (ids differ or cosine below
`--min-cosine`). The work directory is a temp dir deleted on exit unless
`--workdir` (kept) or `--keep` is given.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
DATA = TOOLS.parent / "data"

INTERLEAVED_TEXT = (
    "Waterproof running shoes. <|image|> Featuring a breathable mesh upper. "
    "Grip test on wet rock: <|video|>"
)


def run(cmd):
    """Run a command, echo its output, and abort on failure."""
    print("$ " + " ".join(str(c) for c in cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        print(f"  | {line}")
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit(f"command failed ({out.returncode}): {' '.join(str(c) for c in cmd)}")
    return out.stdout


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / max(na * nb, 1e-30)


def ref_case(python, model, case, dump):
    """Reference ids, embedding, sampled frame count and native frame indices."""
    import numpy as np

    cmd = [
        python,
        str(TOOLS / "ref_embeddinggemma2.py"),
        "mixed",
        "--model",
        model,
        "--dump-inputs",
        str(dump),
        "--out",
        str(dump) + ".json",
    ]
    if case["text"]:
        cmd += ["--text", case["text"]]
    for image in case["images"]:
        cmd += ["--image", str(image)]
    if case["video"]:
        cmd += ["--video", str(case["video"])]
    run(cmd)

    ids = np.load(str(dump) + ".ids.npy").tolist()
    vec = json.loads(Path(str(dump) + ".json").read_text())
    indices_path = Path(str(dump) + ".indices.npy")
    indices = np.load(indices_path).tolist() if indices_path.exists() else None
    return ids, vec, indices


def our_case(binary, model, case, work):
    """Our ids, embedding and the sampled frame count from the CLI's log line."""
    ids_path = work / "ids.json"
    vec_path = work / "vec.json"
    cmd = [
        binary,
        "gemma",
        "embed",
        "--model-dir",
        model,
        "--dtype",
        "f32",
        "--dump-ids",
        str(ids_path),
        "--out",
        str(vec_path),
    ]
    if case["text"]:
        cmd += ["--text", case["text"]]
    for image in case["images"]:
        cmd += ["--image", str(image)]
    if case["video"]:
        cmd += ["--video", str(case["video"])]
    out = run(cmd)

    ids = json.loads(ids_path.read_text())
    vec = json.loads(vec_path.read_text())
    sampled = re.search(r"sampled (\d+) at", out)
    return ids, vec, int(sampled.group(1)) if sampled else None


def make_overflow_clip(work):
    """40 s clip from `data/test.mp4` (10 x 4 s), re-encoded at the source rate so
    the container metadata (8 fps, 320 frames, 40 s) is exact."""
    path = work / "overflow-40s.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-stream_loop",
            "9",
            "-i",
            str(DATA / "test.mp4"),
            "-t",
            "40",
            "-r",
            "8",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bin",
        default=os.environ.get("ROCKET_INFERENCE_BIN", ""),
        help="rocket-inference binary (default: $ROCKET_INFERENCE_BIN)",
    )
    ap.add_argument("--model", default="/mnt/hub/models/embeddinggemma-2")
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter for the reference script (needs transformers + torch)",
    )
    ap.add_argument("--workdir", default=None, help="keep artifacts here instead of a temp dir")
    ap.add_argument("--min-cosine", type=float, default=0.9999)
    ap.add_argument("--keep", action="store_true", help="keep the temporary work directory")
    args = ap.parse_args()
    if not args.bin:
        raise SystemExit("pass --bin <rocket-inference> (or set ROCKET_INFERENCE_BIN)")

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="video-parity-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {work}")

    cases = [
        {"name": "video", "text": None, "images": [], "video": DATA / "test.mp4"},
        {
            "name": "interleaved",
            "text": INTERLEAVED_TEXT,
            "images": [DATA / "cat.jpeg"],
            "video": DATA / "test.mp4",
        },
        {
            "name": "overflow",
            "text": None,
            "images": [],
            "video": make_overflow_clip(work),
        },
    ]

    rows = []
    ok = True
    for case in cases:
        name = case["name"]
        print(f"\n=== {name} ===")
        case_dir = work / name
        case_dir.mkdir(exist_ok=True)
        ref_ids, ref_vec, indices = ref_case(args.python, args.model, case, case_dir / "ref")
        our_ids, our_vec, sampled = our_case(args.bin, args.model, case, case_dir)

        ids_equal = ref_ids == our_ids
        cos = cosine(ref_vec, our_vec)
        ref_frames = len(indices) if indices is not None else None
        frames_equal = ref_frames == sampled
        passed = ids_equal and frames_equal and cos >= args.min_cosine
        ok &= passed
        rows.append((name, len(ref_ids), len(our_ids), ids_equal, ref_frames, sampled, cos, passed))
        if not passed:
            print(f"FAIL: ids_equal={ids_equal} frames {ref_frames} vs {sampled} cosine={cos:.9f}")

    print("\ncase          ids ref/our   ids equal   frames ref/our   cosine        result")
    for name, ref_n, our_n, ids_equal, ref_frames, sampled, cos, passed in rows:
        print(
            f"{name:<13} {ref_n:>5}/{our_n:<5}   {'yes' if ids_equal else 'NO ':>9}   "
            f"{str(ref_frames):>4}/{str(sampled):<4}       {cos:.9f}    "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not args.workdir and not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"artifacts kept in {work}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
