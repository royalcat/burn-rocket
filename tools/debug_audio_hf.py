#!/usr/bin/env python3
"""Dump Gemma 4 audio tower intermediates from the HF reference for comparison
with the Rust `--dump-audio-layers` output."""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/hub/models/embeddinggemma-2")
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text", default="<|audio|>")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    import numpy as np

    try:
        from transformers.audio_utils import load_audio

        wav = load_audio(args.audio, sampling_rate=16000)
        sr = 16000
    except Exception:  # noqa: BLE001
        import soundfile as sf

        wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    inputs = processor(text=args.text, audio=[wav], sampling_rate=sr, return_tensors="pt")

    tower = model.audio_tower
    dumps = {}

    def save(name, t):
        arr = t.detach().float().cpu().numpy()
        dumps[name] = arr
        np.save(os.path.join(args.out, name + ".npy"), arr)
        print(f"{name}: {arr.shape} mean={arr.mean():.6f} std={arr.std():.6f}")

    hooks = []
    hooks.append(tower.subsample_conv_projection.register_forward_hook(lambda m, i, o: save("hf_audio_subsample", o[0])))
    layer0 = tower.layers[0]
    for name, mod in [
        ("hf_audio_l0_ff1", layer0.feed_forward1),
        ("hf_audio_l0_pre_attn", layer0.norm_pre_attn),
        ("hf_audio_l0_attn", layer0.self_attn),
        ("hf_audio_l0_post_attn", layer0.norm_post_attn),
        ("hf_audio_l0_lconv", layer0.lconv1d),
        ("hf_audio_l0_ff2", layer0.feed_forward2),
        ("hf_audio_l0_out", layer0.norm_out),
    ]:
        hooks.append(mod.register_forward_hook(lambda m, i, o, name=name: save(name, o if torch.is_tensor(o) else o[0])))
    for idx, layer in enumerate(tower.layers):
        hooks.append(layer.register_forward_hook(lambda m, i, o, idx=idx: save(f"hf_audio_layer{idx}", o)))
    hooks.append(tower.output_proj.register_forward_hook(lambda m, i, o: save("hf_audio_out", o)))
    hooks.append(model.embed_audio.register_forward_hook(lambda m, i, o: save("hf_audio_soft", o)))

    # Relative position embeddings.
    with torch.no_grad():
        pos = tower.rel_pos_enc(torch.zeros(1, 10, 1024))
    save("hf_audio_pos", pos[0])

    with torch.no_grad():
        out = model(**inputs)
    save("hf_audio_embed", out.last_hidden_state[0])
    print("done ->", args.out)


if __name__ == "__main__":
    main()
