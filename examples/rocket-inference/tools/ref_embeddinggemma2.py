#!/usr/bin/env python3
"""Ground-truth EmbeddingGemma 2 outputs from the HF reference stack.

Runs inside the venv at /mnt/hub/venvs/emb2 (transformers>=5.18).

The processor/sentence-transformers path needs torchvision, which mismatches the
distro torch on this host, so the embedding pipeline is reproduced directly:
`prompt prefix + text` -> tokenizer -> text backbone (which projects to 768-d) ->
mean pooling -> MRL truncation -> L2 normalization (the sentence-transformers
order: pool, truncate, normalize).

Subcommands:
  tokens  - token ids for `prompt prefix + text` (add_special_tokens)
  embed   - sentence embedding, dumped as JSON
  hidden  - per-token hidden states for debugging comparison
"""

import argparse
import json


def read_text(args) -> str:
    if args.text is not None:
        return args.text
    with open(args.text_file, "r", encoding="utf-8") as f:
        return f.read()


def prompt_prefix(model_path: str, name: str | None) -> str:
    if name is None:
        return ""
    with open(f"{model_path}/config_sentence_transformers.json") as f:
        prompts = json.load(f)["prompts"]
    for key, value in prompts.items():
        if key.lower() == name.lower():
            return value
    raise SystemExit(f"unknown prompt '{name}' (available: {sorted(prompts)})")


def cmd_tokens(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    text = args.prompt_prefix + read_text(args)
    enc = tok(text, add_special_tokens=True)
    print(json.dumps({"n": len(enc["input_ids"]), "ids": enc["input_ids"]}))


def _load(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    return tok, model


def _input_ids(args, tok):
    prefix = args.prompt_prefix
    if args.prompt:
        prefix = prompt_prefix(args.model, args.prompt)
    enc = tok(prefix + read_text(args), add_special_tokens=True, return_tensors="pt")
    return enc["input_ids"]


def cmd_embed(args):
    import torch

    tok, model = _load(args)
    ids = _input_ids(args, tok)
    with torch.no_grad():
        out = model(input_ids=ids)
    h = out.last_hidden_state  # [1, S, 768]: already projected
    pooled = h.mean(dim=1)[0]
    if args.dim:
        pooled = pooled[: args.dim]
    if args.normalize:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=0, eps=1e-12)
    vec = [float(x) for x in pooled]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(vec, f)
        print(f"wrote {args.out} (len={len(vec)}, tokens={ids.shape[1]})")
    else:
        print(json.dumps(vec[:8]) + f" ... len={len(vec)}")


def cmd_hidden(args):
    import torch

    tok, model = _load(args)
    ids = _input_ids(args, tok)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states  # embedding + 24 layers
    dtype = None
    for idx in args.layers:
        h = hs[idx].reshape(-1)
        if args.out:
            torch.save({i: hs[i] for i in args.layers}, args.out)
            print(f"saved {args.out} (layers {args.layers})")
            return
        print(f"hidden[{idx}] shape={tuple(hs[idx].shape)} mean={h.mean():.6f} std={h.std():.6f}")


def cmd_image(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    img = Image.open(args.image).convert("RGB")
    text = args.text if args.text else "<|image|>"
    inputs = processor(text=text, images=[img], return_tensors="pt")
    print(
        "input_ids:",
        json.dumps(inputs["input_ids"][0].tolist()[:12]),
        f"... n={inputs['input_ids'].shape[1]}",
        "pixel_values:",
        tuple(inputs["pixel_values"].shape),
    )
    if args.dump_inputs:
        np.save(args.dump_inputs + ".ids.npy", inputs["input_ids"][0].numpy())
        np.save(args.dump_inputs + ".pos.npy", inputs["image_position_ids"][0].numpy())
        np.save(args.dump_inputs + ".pix.npy", inputs["pixel_values"][0].numpy())
        print(f"dumped processor inputs to {args.dump_inputs}.*.npy")
    with torch.no_grad():
        out = model(**inputs)
    if args.dump_inputs:
        feats = model.get_image_features(
            pixel_values=inputs["pixel_values"], image_position_ids=inputs["image_position_ids"]
        )
        np.save(
            args.dump_inputs + ".vision.npy",
            feats.last_hidden_state.detach().numpy().astype(np.float32),
        )
        np.save(
            args.dump_inputs + ".soft.npy",
            feats.pooler_output[0].detach().numpy().astype(np.float32),
        )
        print("dumped vision features to", args.dump_inputs + ".vision.npy")
    h = out.last_hidden_state.mean(dim=1)[0]
    if args.dim:
        h = h[: args.dim]
    if args.normalize:
        h = torch.nn.functional.normalize(h, p=2, dim=0, eps=1e-12)
    vec = [float(x) for x in h]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(vec, f)
        print(f"wrote {args.out} (len={len(vec)})")
    else:
        print(json.dumps(vec[:8]) + f" ... len={len(vec)}")


def cmd_audio(args):
    import json as _json

    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    text = args.text if args.text else "<|audio|>"
    # Prefer HF's own loader (librosa/ffmpeg) so the resampling matches the
    # production pipeline; fall back to raw soundfile for 16 kHz mono files.
    try:
        from transformers.audio_utils import load_audio

        wav = load_audio(args.audio, sampling_rate=16000)
        sr = 16000
    except Exception as e:  # noqa: BLE001
        print(f"load_audio unavailable ({e}); using soundfile")
        import soundfile as sf

        wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    inputs = processor(text=text, audio=[wav], sampling_rate=sr, return_tensors="pt")
    mask = inputs["input_features_mask"][0]
    ids = inputs["input_ids"][0]
    print(
        "input_ids n=",
        int(ids.shape[0]),
        "audio placeholders=",
        int((ids == processor.tokenizer.audio_token_id).sum()),
        "input_features=",
        tuple(inputs["input_features"].shape),
        "mask true=",
        int(mask.sum()),
    )
    if args.dump_inputs:
        np.save(args.dump_inputs + ".ids.npy", ids.numpy())
        np.save(args.dump_inputs + ".feat.npy", inputs["input_features"][0].numpy())
        np.save(args.dump_inputs + ".mask.npy", mask.numpy())
        print(f"dumped processor inputs to {args.dump_inputs}.*.npy")
    with torch.no_grad():
        out = model(**inputs)
    h = out.last_hidden_state.mean(dim=1)[0]
    if args.dim:
        h = h[: args.dim]
    if args.normalize:
        h = torch.nn.functional.normalize(h, p=2, dim=0, eps=1e-12)
    vec = [float(x) for x in h]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(vec, f)
        print(f"wrote {args.out} (len={len(vec)})")
    else:
        print(_json.dumps(vec[:8]) + f" ... len={len(vec)}")


def cmd_video(args):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    text = args.text if args.text else "<|video|>"
    # Decode frames with ffmpeg and hand the processor an explicit frame array +
    # metadata (torchcodec/torchvision video decoding is unavailable on this host).
    import json as _json
    import subprocess

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", args.video,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = _json.loads(probe.stdout)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    num, den = info["r_frame_rate"].split("/")
    native_fps = float(num) / float(den)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", args.video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    print(f"decoded {len(frames)} frames at {native_fps} fps, {w}x{h}")
    from transformers.video_utils import VideoMetadata

    metadata = VideoMetadata(
        fps=native_fps,
        duration=len(frames) / native_fps,
        total_num_frames=len(frames),
    )
    inputs = processor(text=text, videos=[frames], video_metadata=[metadata], return_tensors="pt")
    ids = inputs["input_ids"][0]
    print(
        "input_ids n=",
        int(ids.shape[0]),
        "video placeholders=",
        int((ids == processor.tokenizer.convert_tokens_to_ids("<|video|>")).sum()),
        "pixel_values_videos=",
        tuple(inputs["pixel_values_videos"].shape),
        "frames_per_video=",
        inputs["num_frames_per_video"],
    )
    if args.dump_inputs:
        np.save(args.dump_inputs + ".ids.npy", ids.numpy())
        np.save(args.dump_inputs + ".pix.npy", inputs["pixel_values_videos"].numpy())
        print(f"dumped processor inputs to {args.dump_inputs}.*.npy")
    with torch.no_grad():
        out = model(**inputs)
    h = out.last_hidden_state.mean(dim=1)[0]
    if args.dim:
        h = h[: args.dim]
    if args.normalize:
        h = torch.nn.functional.normalize(h, p=2, dim=0, eps=1e-12)
    vec = [float(x) for x in h]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(vec, f)
        print(f"wrote {args.out} (len={len(vec)})")
    else:
        print(json.dumps(vec[:8]) + f" ... len={len(vec)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("tokens", "embed", "hidden", "image", "video", "audio"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", default="/mnt/hub/models/embeddinggemma-2")
        sp.add_argument("--text", default=None)
        sp.add_argument("--text-file", default=None)
        sp.add_argument("--prompt-prefix", default="")
        sp.add_argument("--prompt", default=None)
        if name in ("embed", "image", "video", "audio"):
            sp.add_argument("--dim", type=int, default=None)
            sp.add_argument("--no-normalize", dest="normalize", action="store_false")
            sp.add_argument("--out", default=None)
        if name == "image":
            sp.add_argument("--image", required=True)
            sp.add_argument("--dump-inputs", default=None)
        if name == "video":
            sp.add_argument("--video", required=True)
            sp.add_argument("--dump-inputs", default=None)
        if name == "audio":
            sp.add_argument("--audio", required=True)
            sp.add_argument("--dump-inputs", default=None)
        if name == "hidden":
            sp.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24])
            sp.add_argument("--out", default=None)

    args = p.parse_args()
    {
        "tokens": cmd_tokens,
        "embed": cmd_embed,
        "hidden": cmd_hidden,
        "image": cmd_image,
        "video": cmd_video,
        "audio": cmd_audio,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
