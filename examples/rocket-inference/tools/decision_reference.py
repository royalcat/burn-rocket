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
"criteria": ...}}, "images": [path, ...], "videos": [path, ...]}`` -- exactly the
arguments of one ``system_one`` call; ``images`` and ``videos`` are optional (absent =
the text path) and list local paths the request shares across all of its questions.
A case with ``videos`` takes the runtime's video path (which also processes its
``images``); video frames are decoded by OpenCV, exactly as the runtime does.

Usage:
    decision_reference.py --model /path/to/d3-lite --requests fixtures.json \
        --out ref.json [--dump DIR] [--dump-vision DIR] [--limit N] [--batch-size 1]

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

``--dump-vision DIR`` (which requires ``--batch-size 1``; a larger batch repeats a
case's images once per question, so the dumped tensors are batch-sized) writes the
image-path tensors of every question of an image case: the processor's
``pixel_values`` (``.pixel_values.npy``), ``image_grid_thw``
(``.image_grid_thw.npy``), the expanded prompt (``.expanded_text.txt``), the
processor's ``input_ids``/``mm_token_type_ids``/``attention_mask`` rows
(``.input_ids.npy``, ``.mm_token_type_ids.npy``, ``.attention_mask.npy``), the
vision tower's pooled output (``.pooled.npy``: the concatenation of
``backbone.get_image_features(...).pooler_output`` in image order, float32), the
multimodal RoPE positions (``.mrope.npy``, ``[3, 1, L]`` int64), the final-normed
hidden (``.hidden.npy``), the raw code logits (``.logits.npy``) and a
``DIR/index.json``.

A **video** case dumps the same set from its video path: the processor's
``pixel_values_videos`` (``.pixel_values_videos.npy``, ``[T*H*W, 1536]``),
``video_grid_thw`` (``.video_grid_thw.npy``, ``[n, 3]``), the request's images
(``.pixel_values.npy``/``.image_grid_thw.npy`` when it has any), the expanded prompt
(the runtime's own ``replace_image_token``/``replace_video_token`` text, including the
per-frame timestamps), the tokenized rows, the tower's pooled output
(``backbone.get_video_features(...).pooler_output``, concatenated over videos), the
positions and the rest, plus every sampled source frame as decoded RGB
(``.frames.{i}.npy``, uint8 ``[frames, height, width, 3]``) and the sampling metadata
(``.video_metadata.json``: per video ``fps``/``total_frames``/``frames_indices``).

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
    p.add_argument("--dump-vision", help="directory for per-question image-path dumps")
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
        required = {"name", "state", "questions"}
        optional = {"images", "videos"}
        if (
            not isinstance(case, dict)
            or not required <= set(case)
            or set(case) - required - optional
        ):
            raise SystemExit(
                f"{path}: case {i} must hold exactly name, state, questions and optionally "
                "images and videos"
            )
        for field in ("images", "videos"):
            media = case.get(field)
            if media is not None:
                if not isinstance(media, list) or not media:
                    raise SystemExit(f"{path}: case {i} {field} must be a nonempty list of paths")
                if not all(isinstance(p, str) and p for p in media):
                    raise SystemExit(f"{path}: case {i} {field} paths must be nonempty strings")
        name = case["name"]
        if not isinstance(name, str) or not name:
            raise SystemExit(f"{path}: case {i} has no name")
        if name in names:
            raise SystemExit(f"{path}: duplicate case name {name!r}")
        names.add(name)
        if not isinstance(case["questions"], dict) or not case["questions"]:
            raise SystemExit(f"{path}: case {name!r} has no questions")
    return cases


class ProcessorProxy:
    """Forward to the checkpoint's processor and record the tensors of every call.

    ``D3.image_logits`` calls ``self.processor(...)`` with the exact batch it feeds the
    backbone, so wrapping the call captures the tensors the reference reads (without
    changing them); the recorded ``text`` list identifies which prompt is which row.
    """

    def __init__(self, processor):
        self.processor = processor
        self.calls: list[tuple[list[str] | None, dict]] = []

    def __call__(self, *args, **kwargs):
        encoded = self.processor(*args, **kwargs)
        texts = kwargs.get("text")
        if texts is not None and not isinstance(texts, (list, tuple)):
            texts = [texts]
        self.calls.append(
            (
                None if texts is None else list(texts),
                {
                    name: value.detach().clone() if hasattr(value, "detach") else value
                    for name, value in encoded.items()
                },
            )
        )
        return encoded

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def call_for(self, text: str) -> tuple[int, dict]:
        """The recorded processor batch holding one prompt and its row index."""
        for texts, encoded in self.calls:
            if texts is not None and text in texts:
                return texts.index(text), encoded
        raise RuntimeError("no recorded processor batch holds this prompt")


class Recorder:
    """Record the encoder output the runtime actually reads, without changing it."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.original_encoder = runtime.encoder
        self.original_logits = runtime.logits
        self.original_image_logits = getattr(runtime, "image_logits", None)
        self.original_video_logits = getattr(runtime, "video_logits", None)
        self.original_processor = getattr(runtime, "processor", None)
        self.proxy: ProcessorProxy | None = None
        self.pending = None
        self.pending_texts: list[str] | None = None
        self.pending_video_texts: list[str] | None = None
        self.frames: list[tuple[list[list[int]], "object"]] = []
        self.image_frames: list[tuple[list[str], "object"]] = []
        self.video_frames: list[tuple[list[str], "object"]] = []

    def __enter__(self):
        runtime, recorder = self.runtime, self

        def logits(sequences, counts):
            recorder.pending = [list(s) for s in sequences]
            return recorder.original_logits(sequences, counts)

        def image_logits(texts, images, counts, width):
            recorder.pending_texts = list(texts)
            return recorder.original_image_logits(texts, images, counts, width)

        def video_logits(texts, prepared, features, counts, width):
            recorder.pending_video_texts = list(texts)
            return recorder.original_video_logits(texts, prepared, features, counts, width)

        def encoder(embeds, present, mrope=None):
            out = (
                recorder.original_encoder(embeds, present, mrope)
                if mrope is not None
                else recorder.original_encoder(embeds, present)
            )
            rows = out[:, -1].detach().float().cpu()
            if recorder.pending is not None:
                if rows.shape[0] != len(recorder.pending):
                    raise RuntimeError("encoder rows do not match the batch")
                recorder.frames.append((recorder.pending, rows))
                recorder.pending = None
            if recorder.pending_texts is not None:
                if rows.shape[0] != len(recorder.pending_texts):
                    raise RuntimeError("encoder rows do not match the image batch")
                recorder.image_frames.append((recorder.pending_texts, rows))
                recorder.pending_texts = None
            if recorder.pending_video_texts is not None:
                if rows.shape[0] != len(recorder.pending_video_texts):
                    raise RuntimeError("encoder rows do not match the video batch")
                recorder.video_frames.append((recorder.pending_video_texts, rows))
                recorder.pending_video_texts = None
            return out

        runtime.logits = logits
        if self.original_image_logits is not None:
            runtime.image_logits = image_logits
        if self.original_video_logits is not None:
            runtime.video_logits = video_logits
        if self.original_processor is not None:
            self.proxy = ProcessorProxy(self.original_processor)
            runtime.processor = self.proxy
        runtime.encoder = encoder
        return self

    def __exit__(self, *exc):
        self.runtime.logits = self.original_logits
        if self.original_image_logits is not None:
            self.runtime.image_logits = self.original_image_logits
        if self.original_video_logits is not None:
            self.runtime.video_logits = self.original_video_logits
        if self.original_processor is not None:
            self.runtime.processor = self.original_processor
        self.runtime.encoder = self.original_encoder
        return False

    def hidden_by_sequence(self) -> dict[tuple[int, ...], "object"]:
        table = {}
        for sequences, rows in self.frames:
            for sequence, row in zip(sequences, rows):
                table[tuple(sequence)] = row
        return table

    def hidden_by_text(self) -> dict[str, "object"]:
        table = {}
        for texts, rows in self.image_frames:
            for text, row in zip(texts, rows):
                table[text] = row
        for texts, rows in self.video_frames:
            for text, row in zip(texts, rows):
                table[text] = row
        return table


def describe_question(runtime, prepared, qid: str) -> tuple[list[str], list[str]]:
    """Option labels (report order) and the prompt codes used for them."""
    question = prepared.questions[qid]
    labels = list(question.keys)
    codes = runtime.codes[: len(labels)]
    return labels, codes


def dump_image_question(
    torch,
    runtime,
    recorder: "Recorder",
    prepared,
    qid: str,
    stem: str,
    out_dir: Path,
    row,
    options: int,
) -> dict:
    """Write the image-path tensors the runtime reads for one question.

    The processor tensors are the recorded batch ``D3.image_logits`` built (the run's
    own call, unchanged); ``pooled`` and ``mrope`` are the backbone's own
    ``get_image_features`` and ``compute_3d_position_ids`` on those tensors, i.e. the
    three calls ``D3.image_embeddings`` makes.
    """
    text = prepared.texts[qid]
    if recorder.proxy is None:
        raise RuntimeError("the runtime has no processor; cannot dump image tensors")
    row_index, batch = recorder.proxy.call_for(text)
    ids = batch["input_ids"][row_index : row_index + 1].to(torch.int64)
    mask = batch["attention_mask"][row_index : row_index + 1].to(torch.int64)
    mm = batch["mm_token_type_ids"][row_index : row_index + 1].to(torch.int64)
    grid = batch["image_grid_thw"].to(torch.int64)
    pixel_values = batch["pixel_values"].to(torch.float32)
    if pixel_values.shape[0] != int(grid.prod(-1).sum()):
        raise RuntimeError(f"{stem}: pixel_values rows do not match the grid")
    device = runtime.device
    backbone = runtime.backbone
    with torch.inference_mode():
        pooler = backbone.get_image_features(
            pixel_values.to(device), grid.to(device), return_dict=True
        ).pooler_output
        pooled = torch.cat(list(pooler), dim=0).float().cpu()
        embeds = backbone.get_input_embeddings()(ids.to(device))
        mrope = backbone.compute_3d_position_ids(
            input_ids=ids.to(device),
            image_grid_thw=grid.to(device),
            video_grid_thw=None,
            inputs_embeds=embeds,
            attention_mask=mask.to(device),
            past_key_values=None,
            mm_token_type_ids=mm.to(device),
        )
    if mrope is None:
        raise RuntimeError(f"{stem}: the backbone returned no multimodal positions")
    logits = (row.float() @ runtime.readout.T.cpu()).numpy().astype(np.float32)

    files = {
        "pixel_values": f"{stem}.pixel_values.npy",
        "image_grid_thw": f"{stem}.image_grid_thw.npy",
        "input_ids": f"{stem}.input_ids.npy",
        "mm_token_type_ids": f"{stem}.mm_token_type_ids.npy",
        "attention_mask": f"{stem}.attention_mask.npy",
        "expanded_text": f"{stem}.expanded_text.txt",
        "pooled": f"{stem}.pooled.npy",
        "mrope": f"{stem}.mrope.npy",
        "hidden": f"{stem}.hidden.npy",
        "logits": f"{stem}.logits.npy",
    }
    np.save(out_dir / files["pixel_values"], pixel_values.numpy().astype(np.float32))
    np.save(out_dir / files["image_grid_thw"], grid.numpy().astype(np.int64))
    np.save(out_dir / files["input_ids"], ids.numpy().astype(np.int64))
    np.save(out_dir / files["mm_token_type_ids"], mm.numpy().astype(np.int64))
    np.save(out_dir / files["attention_mask"], mask.numpy().astype(np.int64))
    (out_dir / files["expanded_text"]).write_text(text, encoding="utf-8")
    np.save(out_dir / files["pooled"], pooled.numpy().astype(np.float32))
    np.save(out_dir / files["mrope"], mrope.cpu().numpy().astype(np.int64))
    np.save(out_dir / files["hidden"], row.numpy().astype(np.float32))
    np.save(out_dir / files["logits"], logits)
    lengths = (grid.prod(-1) // 4).tolist()
    return {
        "case": stem.split("__")[0],
        "question": qid,
        "images": len(prepared.images),
        "files": files,
        "tokens": prepared.lengths[qid],
        "sequence_tokens": int(ids.shape[1]),
        "grid": grid.tolist(),
        "pooled_rows": lengths,
        "options": options,
    }


def video_expanded_text(runtime, prepared, text: str) -> str:
    """The prompt ``D3.video_logits`` feeds the tokenizer for one question.

    Same two calls the runtime makes: ``replace_image_token`` for every image, then
    ``get_text_with_replacements`` with the request's per-video replacement strings
    (which already carry the timestamps and the per-group vision markers).
    """
    processor = runtime.processor
    media = prepared.media
    image_texts = [
        processor.replace_image_token(media, image_idx=i)
        for i in range(len(prepared.images))
    ]
    return processor.get_text_with_replacements([text], image_texts, prepared.video_texts)[0][0]


def dump_video_question(
    torch,
    runtime,
    recorder: "Recorder",
    prepared,
    qid: str,
    stem: str,
    out_dir: Path,
    row,
    options: int,
) -> dict:
    """Write the video-path tensors the runtime reads for one question.

    ``pixel_values_videos``/``video_grid_thw``/``pixel_values``/``image_grid_thw`` are
    the processor's own outputs (``D3._prepare_videos`` keeps them in
    ``prepared.media``); the expanded prompt, ids and type ids are the exact
    ``video_logits`` steps; ``pooled``/``mrope`` are the backbone's
    ``get_video_features`` and ``compute_3d_position_ids`` on those tensors.
    """
    import json as _json

    text = prepared.texts[qid]
    if recorder.proxy is None:
        raise RuntimeError("the runtime has no processor; cannot dump video tensors")
    media = prepared.media
    expanded = video_expanded_text(runtime, prepared, text)
    tokenizer = runtime.processor.tokenizer
    encoded = tokenizer(
        expanded, padding=False, return_token_type_ids=False, return_tensors="pt"
    )
    ids = encoded["input_ids"].to(torch.int64)
    mask = encoded["attention_mask"].to(torch.int64)
    mm = torch.as_tensor(
        runtime.processor.create_mm_token_type_ids(encoded["input_ids"]),
        dtype=torch.int64,
    )
    video_grid = media["video_grid_thw"].to(torch.int64)
    pixel_values_videos = media["pixel_values_videos"].to(torch.float32)
    image_grid = media.get("image_grid_thw")
    pixel_values = media.get("pixel_values")
    device = runtime.device
    backbone = runtime.backbone
    with torch.inference_mode():
        pooler = backbone.get_video_features(
            pixel_values_videos.to(device), video_grid.to(device), return_dict=True
        ).pooler_output
        pooled = torch.cat(list(pooler), dim=0).float().cpu()
        embeds = backbone.get_input_embeddings()(ids.to(device))
        mrope = backbone.compute_3d_position_ids(
            input_ids=ids.to(device),
            image_grid_thw=(
                image_grid.to(torch.int64).to(device) if image_grid is not None else None
            ),
            video_grid_thw=video_grid.to(device),
            inputs_embeds=embeds,
            attention_mask=mask.to(device),
            past_key_values=None,
            mm_token_type_ids=mm.to(device),
        )
    if mrope is None:
        raise RuntimeError(f"{stem}: the backbone returned no multimodal positions")
    logits = (row.float() @ runtime.readout.T.cpu()).numpy().astype(np.float32)

    files = {
        "pixel_values_videos": f"{stem}.pixel_values_videos.npy",
        "video_grid_thw": f"{stem}.video_grid_thw.npy",
        "input_ids": f"{stem}.input_ids.npy",
        "mm_token_type_ids": f"{stem}.mm_token_type_ids.npy",
        "attention_mask": f"{stem}.attention_mask.npy",
        "expanded_text": f"{stem}.expanded_text.txt",
        "pooled": f"{stem}.pooled.npy",
        "mrope": f"{stem}.mrope.npy",
        "hidden": f"{stem}.hidden.npy",
        "logits": f"{stem}.logits.npy",
    }
    np.save(out_dir / files["pixel_values_videos"], pixel_values_videos.numpy().astype(np.float32))
    np.save(out_dir / files["video_grid_thw"], video_grid.numpy().astype(np.int64))
    np.save(out_dir / files["input_ids"], ids.numpy().astype(np.int64))
    np.save(out_dir / files["mm_token_type_ids"], mm.numpy().astype(np.int64))
    np.save(out_dir / files["attention_mask"], mask.numpy().astype(np.int64))
    (out_dir / files["expanded_text"]).write_text(expanded, encoding="utf-8")
    np.save(out_dir / files["pooled"], pooled.numpy().astype(np.float32))
    np.save(out_dir / files["mrope"], mrope.cpu().numpy().astype(np.int64))
    np.save(out_dir / files["hidden"], row.numpy().astype(np.float32))
    np.save(out_dir / files["logits"], logits)
    if pixel_values is not None and image_grid is not None:
        files["pixel_values"] = f"{stem}.pixel_values.npy"
        files["image_grid_thw"] = f"{stem}.image_grid_thw.npy"
        np.save(out_dir / files["pixel_values"], pixel_values.float().numpy().astype(np.float32))
        np.save(
            out_dir / files["image_grid_thw"],
            image_grid.to(torch.int64).numpy().astype(np.int64),
        )

    # The sampled source frames as decoded (OpenCV BGR -> RGB) and where they come
    # from; the Rust probe compares both against its ffmpeg decode.
    videos = []
    for index, video in enumerate(prepared.videos):
        frames = np.ascontiguousarray(video.frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise RuntimeError(f"{stem}: unexpected frame array {frames.shape} {frames.dtype}")
        name = f"{stem}.frames.{index}.npy"
        np.save(out_dir / name, frames)
        videos.append(
            {
                "frames_file": name,
                "fps": float(video.fps),
                "total_frames": int(video.total_frames),
                "frames_indices": [int(i) for i in video.indices],
                "source": [int(frames.shape[2]), int(frames.shape[1])],
                "frame_count": int(frames.shape[0]),
            }
        )
    metadata_name = f"{stem}.video_metadata.json"
    (out_dir / metadata_name).write_text(
        _json.dumps({"videos": videos}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files["video_metadata"] = metadata_name

    merge = int(runtime.processor.video_processor.merge_size) ** 2
    frame_rows = [int(g[0] * g[1] * g[2]) // merge for g in video_grid.tolist()]
    return {
        "case": stem.split("__")[0],
        "question": qid,
        "images": len(prepared.images),
        "videos": len(prepared.videos),
        "files": files,
        "tokens": prepared.lengths[qid],
        "sequence_tokens": int(ids.shape[1]),
        "grid": video_grid.tolist(),
        "video_tokens": frame_rows,
        "options": options,
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch
    from transformers import AutoModel

    if args.dump_vision and args.batch_size != 1:
        raise SystemExit(
            "--dump-vision needs --batch-size 1: a larger batch repeats a case's images "
            "once per question, so the dumped pixel_values/grid/pooled are batch-sized"
        )

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
    dump_vision_dir = Path(args.dump_vision).expanduser() if args.dump_vision else None
    if dump_vision_dir is not None:
        dump_vision_dir.mkdir(parents=True, exist_ok=True)

    out_cases = []
    dump_index = []
    vision_index = []
    total_tokens = 0
    total_questions = 0
    run_started = time.perf_counter()

    for index, case in enumerate(cases, 1):
        name = case["name"]
        images = case.get("images")
        videos = case.get("videos")
        case_started = time.perf_counter()
        try:
            if videos:
                prepared = runtime.prepare(
                    case["state"], case["questions"], images=images or [], videos=videos
                )
            elif images:
                prepared = runtime.prepare(case["state"], case["questions"], images=images)
            else:
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
        hidden_texts = recorder.hidden_by_text()

        questions = {}
        for qid in prepared.keys:
            if qid in errors:
                questions[qid] = {"type": prepared.errors[qid].get("type"), "error": prepared.errors[qid]}
                continue
            labels, codes = describe_question(runtime, prepared, qid)
            expanded = prepared.texts.get(qid)
            if expanded is None:
                sequence = prepared.sequences[qid]
                prompt = runtime.text(case["state"], prepared.questions[qid].rendered)
                encoded = runtime.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                if encoded != sequence:
                    raise RuntimeError(f"{name}/{qid}: rendered prompt does not re-tokenize to its ids")
                row = hidden.get(tuple(sequence))
                question_tokens = len(sequence)
            else:
                prompt = expanded
                row = hidden_texts.get(expanded)
                question_tokens = prepared.lengths[qid]
            values = [float(p) for p in probabilities[qid]]
            if len(values) != len(labels):
                raise RuntimeError(f"{name}/{qid}: {len(values)} probabilities for {len(labels)} options")
            best = max(range(len(values)), key=lambda i: (values[i], -i))
            record = {
                "type": prepared.questions[qid].kind,
                "options": labels,
                "codes": codes,
                "tokens": question_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "probabilities": values,
                "top": labels[best],
                "answer": response["answers"].get(qid),
            }
            if expanded is not None:
                record["images"] = len(prepared.images)
                if prepared.videos:
                    record["videos"] = len(prepared.videos)
            questions[qid] = record
            total_questions += 1
            if dump_dir is not None and expanded is None:
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
            if dump_vision_dir is not None and expanded is not None:
                if row is None:
                    raise RuntimeError(f"{name}/{qid}: no recorded final hidden for its media forward")
                stem = f"{sanitize(name)}__{sanitize(qid)}"
                dump = dump_video_question if prepared.videos else dump_image_question
                entry = dump(
                    torch, runtime, recorder, prepared, qid, stem, dump_vision_dir,
                    row, len(labels),
                )
                entry["top"] = labels[best]
                vision_index.append(entry)

        elapsed = time.perf_counter() - case_started
        total_tokens += tokens
        tops = ", ".join(f"{qid}={questions[qid].get('top', '?')}" for qid in prepared.keys)
        print(f"[{index}/{len(cases)}] {name} q={len(case['questions'])} tokens={tokens} "
              f"{elapsed:.2f}s {tops}", flush=True)
        request = {"state": case["state"], "questions": case["questions"]}
        if images:
            request["images"] = list(images)
        if videos:
            request["videos"] = list(videos)
        out_cases.append({
            "name": name,
            "elapsed_s": round(elapsed, 3),
            "request": request,
            "request_sha256": canonical_sha256(request),
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
    if dump_vision_dir is not None:
        (dump_vision_dir / "index.json").write_text(
            json.dumps({
                "schema": SCHEMA,
                "model": reference["model"]["name"],
                "dtype": backbone_dtype,
                "vision_config": runtime.backbone.config.vision_config.to_dict(),
                "entries": vision_index,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {out_path} ({len(out_cases)} cases, {total_questions} questions, "
          f"{total_tokens} tokens, {wall:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
