#!/usr/bin/env python3
"""W8A8 viability probe for the EmbeddingGemma 2 text backbone.

Simulates the `librocketnpu` resident int8 matmul conventions on the HF f32
reference model and reports the embedding cosine against the unquantized f32
embedding of the same input. The simulation mirrors the library's int8
contract (see `vendor/rocketnpu/rocket_matmul.h`):

  * weights and activations are pre-quantized symmetric int8 (`max/127`);
  * the weight scale is per output channel per K-group (`b_scale[N, K/group]`),
    the activation scale per row per K-group (`a_scale[M, K/group]`);
  * products accumulate exactly in int32; each group is dequantized with
    `a_scale * b_scale` and accumulated in f32 (simulated here by a f32 GEMM of
    the dequantized operands, which differs only in f32 rounding order).

Only the `language_model` Linears are patched (the path the NPU offload
covers); the vision/audio towers stay f32.

Runs in the reference venv: /mnt/hub/venvs/emb2/bin/python.

Examples:
  w8a8_probe.py --text "what is the capital of france?" --prompt query
  w8a8_probe.py --text-file data/one_long.txt --prompt query
"""

import argparse
import json

# group spelling for --configs entries (name:a_group:b_group):
#   0  keep f32;  -1  one scale for the whole K;  N  per-N-wide K-blocks
DEFAULT_CONFIGS = ",".join(
    [
        "w16a16:0:0",
        "w8gKa16:0:-1",
        "w8g32a16:0:32",
        "w8gKa8gK:-1:-1",
        "w8g32a8g32:32:32",
        "w8g32a8gK:32:-1",
        "w8gKa8g32:-1:32",
    ]
)

# Mutated per config; `quantized_linear` reads it at call time.
PATCH = {"a_group": None, "b_group": None}


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


def quant_symmetric(x, group: int):
    """Symmetric int8 quantization along the last dim, per `group` block.

    Returns (q [..., K] in [-127, 127], scale [..., K // group]).
    """
    import torch

    *lead, k = x.shape
    assert k % group == 0, f"K={k} not divisible by group={group}"
    xg = x.reshape(*lead, k // group, group)
    scale = (xg.abs().amax(dim=-1) / 127.0).clamp_min(1e-12)
    q = torch.round(xg / scale.unsqueeze(-1)).clamp(-127, 127)
    return q.reshape(*lead, k), scale


def dequant_grouped(q, scale, group: int):
    """`q [..., K]` int8 + `scale [..., K//group]` -> f32 [..., K]."""
    *lead, k = q.shape
    qg = q.reshape(*lead, k // group, group)
    return (qg * scale.unsqueeze(-1)).reshape(*lead, k)


def norm_group(value: int, k: int) -> int:
    return k if value == -1 else value


def quantized_linear(module, x, a_group, b_group):
    """`F.linear` with int8-simulated operands (None = keep f32)."""
    import torch.nn.functional as F

    weight = module.weight
    if b_group is not None:
        g = norm_group(b_group, weight.shape[-1])
        qw, sw = quant_symmetric(weight, g)
        weight = dequant_grouped(qw, sw, g)
    if a_group is not None:
        g = norm_group(a_group, x.shape[-1])
        qa, sa = quant_symmetric(x, g)
        x = dequant_grouped(qa, sa, g)
    return F.linear(x, weight, module.bias)


def patch_language_model(model):
    """Patch every `language_model` Linear with the quantized forward.

    Returns `(module, original_forward)` pairs for restoration.
    """
    import torch.nn as nn

    saved = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "language_model" in name:
            saved.append((module, module.forward))
            module.forward = (
                lambda x, m=module: quantized_linear(
                    m, x, PATCH["a_group"], PATCH["b_group"]
                )
            )
    return saved


def restore(saved):
    for module, forward in saved:
        module.forward = forward


def run_embedding(model, ids):
    import torch

    with torch.no_grad():
        out = model(input_ids=ids)
    h = out.last_hidden_state.mean(dim=1)[0]
    return torch.nn.functional.normalize(h, p=2, dim=0, eps=1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/hub/models/embeddinggemma-2")
    p.add_argument("--text", default=None)
    p.add_argument("--text-file", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--configs", default=DEFAULT_CONFIGS)
    args = p.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    prefix = prompt_prefix(args.model, args.prompt)
    enc = tok(prefix + read_text(args), add_special_tokens=True, return_tensors="pt")
    ids = enc["input_ids"]
    print(f"tokens={ids.shape[1]} prompt={args.prompt!r}")

    ref = run_embedding(model, ids)

    saved = patch_language_model(model)
    print(f"patched {len(saved)} language_model Linears")

    for cfg in args.configs.split(","):
        name, a_s, b_s = cfg.split(":")
        PATCH["a_group"] = None if int(a_s) == 0 else int(a_s)
        PATCH["b_group"] = None if int(b_s) == 0 else int(b_s)
        v = run_embedding(model, ids)
        cos = float((v * ref).sum())
        print(
            f"{name:14s} a_group={a_s:>3s} b_group={b_s:>3s}  "
            f"cosine={cos:.9f}  max|d|={float((v - ref).abs().max()):.3e}"
        )

    restore(saved)


if __name__ == "__main__":
    main()
