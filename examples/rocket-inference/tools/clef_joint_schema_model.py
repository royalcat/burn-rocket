"""Clef-Omni: the Qwen3-Omni thinker with a joint schema head for typed decisions.

A record provides a ``state`` (any JSON value), optional ``images``, ``audio``, and
``videos``, and ``questions``. Each question has a ``type`` (``noul``, ``choice``, or ``score``),
``instructions``, and, for ``choice`` and ``score``, ``criteria`` describing the
allowed options. The model returns one logit per allowed option for every question.
``systemone`` answers a Jev/SystemOne ``/v1/systemone`` request body with the same
response body.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>\n"
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>\n"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>\n"
MEDIA_BATCH_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "video_second_per_grid")
AUDIO_BATCH_KEYS = ("input_features", "feature_attention_mask")
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
DEFAULT_MAX_LENGTH = 64000
AUDIO_SAMPLE_RATE = 16_000
VIDEO_FPS = 2.0
MAX_VIDEO_FRAME_PIXELS = 256 * 32 * 32
IMAGE_SIZE = {"shortest_edge": 65_536, "longest_edge": 1_048_576}
_DATA_URL = re.compile(r"^data:[\w.+-]+/[\w.+-]+;base64,(?P<data>.+)$", re.DOTALL)


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str
    media: dict[str, Any] | None = dataclass_field(default=None, compare=False, repr=False)


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _media_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    text = str(value)
    match = _DATA_URL.match(text)
    if match is not None:
        return base64.b64decode(match["data"])
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text) as response:
            return response.read()
    return Path(text).read_bytes()


def _is_encoded(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, str, Path))


def load_image(value: Any) -> Any:
    """Return RGB pixels from a PIL image, path, URL, base64 data URL, or encoded bytes."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    with Image.open(io.BytesIO(_media_bytes(value))) as image:
        return image.convert("RGB")


def _decode_audio(data: bytes) -> np.ndarray | None:
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if stream is None:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_SAMPLE_RATE)
        chunks = []
        for frame in itertools.chain(container.decode(stream), (None,)):
            chunks.extend(resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(frame))
    if not chunks:
        return None
    samples = np.concatenate(chunks).astype(np.float32, copy=False)
    return samples if samples.size else None


def load_audio(value: Any) -> np.ndarray:
    """Return 16 kHz mono samples from an encoded clip (path, URL, data URL, or bytes) or a sample array."""
    if not _is_encoded(value):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    samples = _decode_audio(_media_bytes(value))
    if samples is None:
        raise ValueError("audio has no decodable audio stream")
    return samples


def load_video(value: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Return RGB frames sampled at ``VIDEO_FPS`` and the clip's 16 kHz audio track, if any.

    Frame arrays are used as given (already sampled at ``VIDEO_FPS``) and have no audio track.
    """
    if not _is_encoded(value):
        return np.asarray(value), None
    import av

    data = _media_bytes(value)
    frames: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if stream is None:
            raise ValueError("video has no video stream")
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < len(frames) / VIDEO_FPS:
                continue
            scale = min(1.0, math.sqrt(MAX_VIDEO_FRAME_PIXELS / (frame.width * frame.height)))
            width = max(2, round(frame.width * scale / 2) * 2)
            height = max(2, round(frame.height * scale / 2) * 2)
            frames.append(frame.to_ndarray(format="rgb24", width=width, height=height))
    if not frames:
        raise ValueError("video has no decodable frames")
    if len(frames) % 2:
        frames.append(frames[-1])
    return np.stack(frames), _decode_audio(data)


def _encode_media(processor: Any, record: dict[str, Any]) -> tuple[list[int], dict[str, Any] | None]:
    images = [load_image(image) for image in record.get("images") or []]
    audio = [load_audio(clip) for clip in record.get("audio") or []]
    decoded_videos = [load_video(video) for video in record.get("videos") or []]
    if not images and not audio and not decoded_videos:
        return [], None
    if processor is None:
        raise ValueError("records with images, audio, or videos require a processor")
    videos = [frames for frames, _ in decoded_videos]
    tracks = [track for _, track in decoded_videos]
    text = IMAGE_PLACEHOLDER * len(images) + AUDIO_PLACEHOLDER * len(audio) + VIDEO_PLACEHOLDER * len(videos)
    use_audio_in_video = bool(videos) and all(track is not None for track in tracks)
    if use_audio_in_video:
        audio.extend(tracks)
    options: dict[str, Any] = {"size": IMAGE_SIZE}
    if videos:
        options = {
            "images_kwargs": {"size": IMAGE_SIZE},
            "videos_kwargs": {"fps": VIDEO_FPS, "do_sample_frames": False, "use_audio_in_video": use_audio_in_video},
        }
    options.update(record.get("media_kwargs") or {})
    encoded = processor(
        text=text,
        images=images or None,
        audio=audio or None,
        videos=videos or None,
        add_special_tokens=False,
        return_tensors="pt",
        **options,
    )
    media = {key: encoded[key] for key in MEDIA_BATCH_KEYS + AUDIO_BATCH_KEYS if key in encoded}
    media["use_audio_in_video"] = use_audio_in_video
    return encoded["input_ids"][0].tolist(), media


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = DEFAULT_MAX_LENGTH,
    max_state_tokens: int | None = None,
    processor: Any | None = None,
) -> EncodedRecord:
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[str(question["type"])],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    media_ids, media = _encode_media(processor, record)
    prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(
            f"schema requires {fixed_length} tokens before state; maximum is {max_length}"
        )
    state_ids = state_ids[: max_length - fixed_length]
    schema_offset = len(prefix_ids) + len(state_ids)
    shifted_questions = tuple(
        EncodedQuestion(
            question_id=question.question_id,
            question_type=question.question_type,
            question_span=(
                question.question_span[0] + schema_offset,
                question.question_span[1] + schema_offset,
            ),
            option_spans=tuple(
                (start + schema_offset, end + schema_offset)
                for start, end in question.option_spans
            ),
            option_ids=question.option_ids,
        )
        for question in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted_questions:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(
        input_ids=input_ids,
        questions=shifted_questions,
        record_id=str(record.get("id", "unknown")),
        media=media,
    )


def collate_records(
    records: list[EncodedRecord],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    maximum_length = max(len(record.input_ids) for record in records)
    input_ids = torch.full(
        (len(records), maximum_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(records), maximum_length),
        dtype=torch.long,
        device=device,
    )
    for index, record in enumerate(records):
        length = len(record.input_ids)
        input_ids[index, :length] = torch.tensor(record.input_ids, device=device)
        attention_mask[index, :length] = 1
    media: dict[str, torch.Tensor] = {}
    for key in MEDIA_BATCH_KEYS:
        values = [record.media[key] for record in records if record.media and key in record.media]
        if values:
            media[key] = torch.cat(values, dim=0).to(device)
    for key in AUDIO_BATCH_KEYS:
        values = [record.media[key] for record in records if record.media and key in record.media]
        if values:
            frames = max(value.shape[-1] for value in values)
            media[key] = torch.cat([functional.pad(value, (0, frames - value.shape[-1])) for value in values]).to(device)
    video_audio = {
        record.media["use_audio_in_video"] for record in records if record.media and "pixel_values_videos" in record.media
    }
    if len(video_audio) > 1:
        raise ValueError("videos with and without audio tracks cannot share a batch")
    if video_audio == {True}:
        media["use_audio_in_video"] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "records": records,
        "media": media,
    }


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        routed, _ = self.attention(
            normalized_queries,
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [
                EvidenceRoutingLayer(
                    width=width,
                    heads=heads,
                    feedforward=feedforward,
                    dropout=dropout,
                )
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[EncodedRecord],
        output_embedding_weight: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        results: list[list[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)
        for batch_index, record in enumerate(records):
            sequence_length = int(attention_mask[batch_index].sum().item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            question_vectors = torch.stack(
                [
                    self._mean_span(sequence_hidden, question.question_span)
                    for question in record.questions
                ]
            )
            type_ids = torch.tensor(
                [question.question_type for question in record.questions],
                device=hidden_states.device,
            )
            option_contexts: list[torch.Tensor] = []
            lexical_options: list[torch.Tensor] = []
            option_counts = []
            for question in record.questions:
                context_vectors = torch.stack(
                    [
                        self._mean_span(sequence_hidden, span)
                        for span in question.option_spans
                    ]
                )
                lexical_vectors = []
                for start, end in question.option_spans:
                    token_ids = input_ids[batch_index, start:end]
                    lexical_vectors.append(output_embedding_weight[token_ids].mean(dim=0))
                lexical = torch.stack(lexical_vectors)
                option_contexts.append(context_vectors)
                lexical_options.append(lexical)
                option_counts.append(len(question.option_spans))

            option_queries = []
            for question_index, (context_vectors, lexical) in enumerate(
                zip(option_contexts, lexical_options)
            ):
                option_queries.append(
                    self.option_context_projection(context_vectors)
                    + self.option_lexical_projection(lexical)
                    + self.option_question_projection(
                        question_vectors[question_index]
                    ).unsqueeze(0)
                )
            routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)
            routed_options = routed_options[0]
            split_options = list(torch.split(routed_options, option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for field, options in zip(base_fields, split_options):
                routing_weights = torch.softmax(
                    torch.matmul(options, field) / math.sqrt(options.shape[-1]),
                    dim=0,
                )
                option_summaries.append(
                    torch.sum(routing_weights.unsqueeze(-1) * options, dim=0)
                )
            fields = (
                base_fields
                + self.option_summary_norm(torch.stack(option_summaries))
                + self.global_projection(global_vector).unsqueeze(0)
                + self.type_embedding(type_ids)
            )
            fields = fields.unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            record_logits: list[torch.Tensor] = []
            for field, question, lexical, routed in zip(
                fields,
                record.questions,
                lexical_options,
                split_options,
            ):
                anchor = functional.normalize(
                    question_vectors[len(record_logits)] + global_vector,
                    dim=-1,
                )
                lexical_anchor = functional.normalize(lexical, dim=-1)
                prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior = prior_scale * torch.matmul(lexical_anchor, anchor)
                options = self.option_norm(routed)
                repeated_field = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(repeated_field, options, dim=-1)
                features = torch.cat(
                    [
                        repeated_field,
                        options,
                        repeated_field * options,
                        torch.abs(repeated_field - options),
                    ],
                    dim=-1,
                )
                residual = self.residual_scorer(features).squeeze(-1)
                joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                joint = joint_scale * cosine + residual
                record_logits.append(
                    prior + torch.sigmoid(self.residual_gate) * joint
                )
            results.append(record_logits)
        return results


class ClefModel(torch.nn.Module):
    """The Qwen3-Omni thinker (vision and audio encoders included) with the joint schema head."""

    def __init__(self, thinker: Any, head: JointSchemaHead) -> None:
        super().__init__()
        self.output_embeddings = thinker.lm_head
        thinker.lm_head = torch.nn.Identity()
        self.thinker = thinker
        self.head = head

    def forward(self, batch: dict[str, Any]) -> list[list[torch.Tensor]]:
        """Score each record of a batch separately, so padding never changes a record's probabilities."""
        device = batch["input_ids"].device
        results: list[list[torch.Tensor]] = []
        for record in batch["records"]:
            results.extend(self._forward_unpadded(collate_records([record], 0, device)))
        return results

    def _forward_unpadded(self, batch: dict[str, Any]) -> list[list[torch.Tensor]]:
        media = dict(batch.get("media") or {})
        use_audio_in_video = bool(media.pop("use_audio_in_video", False))
        dtype = self.output_embeddings.weight.dtype
        for key in ("pixel_values", "pixel_values_videos", "input_features"):
            if key in media:
                media[key] = media[key].to(dtype)
        outputs = self.thinker(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_router_logits=False,
            use_audio_in_video=use_audio_in_video,
            return_dict=True,
            **media,
        )
        return self.head(
            outputs.logits,
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            self.output_embeddings.weight,
        )


def load_release_model(
    model_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    **from_pretrained_kwargs: Any,
) -> tuple[ClefModel, Any]:
    """Load a Clef-Omni release (merged thinker, joint schema head, and processor)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3OmniMoeForConditionalGeneration
    from transformers.utils import logging as transformers_logging

    path = Path(model_path)
    if not path.is_dir():
        path = Path(snapshot_download(str(model_path)))
    config = AutoConfig.from_pretrained(path)
    config.enable_audio_output = False
    verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        backbone, loading_info = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            path,
            config=config,
            dtype=dtype,
            device_map={"": str(device)},
            output_loading_info=True,
            **from_pretrained_kwargs,
        )
    finally:
        transformers_logging.set_verbosity(verbosity)
    problems = {
        "missing": sorted(loading_info.get("missing_keys") or ()),
        "unexpected": sorted(
            key for key in loading_info.get("unexpected_keys") or () if not key.startswith(("talker.", "code2wav."))
        ),
        "mismatched": sorted(str(key) for key in loading_info.get("mismatched_keys") or ()),
    }
    if any(problems.values()):
        raise RuntimeError(f"Clef-Omni weights did not load cleanly: {problems}")
    head_config = json.loads((path / "joint_head_config.json").read_text())
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    head = head.to(device=device, dtype=dtype)
    processor = AutoProcessor.from_pretrained(path)
    return ClefModel(backbone.thinker, head).eval(), processor


def systemone_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    """Convert per-option probabilities for one question into a SystemOne answer."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {option: round(probabilities[option], 4) for option in options},
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


@torch.inference_mode()
def systemone(
    model: ClefModel, processor: Any, request: dict[str, Any], max_length: int = DEFAULT_MAX_LENGTH
) -> dict[str, Any]:
    """Answer a Jev/SystemOne ``/v1/systemone`` request body with a SystemOne response body.

    The request has ``model``, ``state``, and ``questions``, plus optional ``images``, ``audio``, and ``videos``
    (base64 data URLs, URLs, or paths).
    """
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for question_id, question in questions.items():
        if question.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")
    encoded = encode_record(processor.tokenizer, request, max_length=max_length, processor=processor)
    device = next(model.parameters()).device
    logits = model(collate_records([encoded], processor.tokenizer.pad_token_id, device))[0]
    answers = {
        question.question_id: systemone_answer(
            questions[question.question_id],
            dict(zip(question.option_ids, question_logits.float().softmax(-1).tolist())),
        )
        for question, question_logits in zip(encoded.questions, logits)
    }
    return {
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
    }
