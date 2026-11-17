#!/usr/bin/env python3
"""Ground-truth Gemma 4 E2B-it outputs from the HF reference stack.

Runs in /mnt/hub/venvs/emb2 (transformers 5.19.0).

Subcommands:
  tokens - chat-template rendering -> token ids
  gen    - greedy/sampled generation, prints text + token ids
  logits - prefill logits (top-k) for the first generated position
  hidden - per-layer hidden states of the text backbone (debug comparison)
"""

import argparse
import json


def messages_from_args(args):
    if args.messages:
        return json.loads(args.messages)
    text = args.text if args.text is not None else open(args.text_file).read()
    return [{"role": "user", "content": text}]


def load(args, dtype):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=dtype)
    model.eval()
    return proc, model, torch


def render(args, proc):
    msgs = messages_from_args(args)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.enable_thinking:
        kwargs["enable_thinking"] = True
    text = proc.apply_chat_template(msgs, **kwargs)
    return msgs, text


def cmd_tokens(args):
    proc, _, _ = load(args, None) if False else (None, None, None)
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    _, text = render(args, proc)
    tok = proc.tokenizer
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(json.dumps({"text": text, "n": len(ids), "ids": ids}))


def media_inputs(args, proc, text):
    """Processor inputs for a rendered conversation plus optional media."""
    from PIL import Image

    kwargs = {"text": text, "return_tensors": "pt"}
    if args.image:
        kwargs["images"] = [Image.open(p).convert("RGB") for p in args.image]
    if args.audio:
        import librosa
        import numpy as np

        wavs = []
        for p in args.audio:
            w, _ = librosa.load(p, sr=16000)
            wavs.append(w)
        kwargs["audio"] = wavs
    return proc(**kwargs)


def cmd_gen(args):
    import torch

    proc, model, torch = load(args, getattr(torch, args.dtype))
    _, text = render(args, proc)
    inputs = media_inputs(args, proc, text)
    n_prompt = inputs["input_ids"].shape[1]
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        pad_token_id=proc.tokenizer.pad_token_id,
    )
    if not args.greedy:
        gen_kwargs.update(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        torch.manual_seed(args.seed)
    dump = args.dump_logits
    if dump:
        gen_kwargs.update(output_scores=True, return_dict_in_generate=True)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    if dump:
        steps = []
        for score in out.scores:
            top = torch.topk(score[0].float(), 8)
            steps.append([[int(i), float(v)] for i, v in zip(top.indices, top.values)])
        json.dump(steps, open(dump, "w"))
        print(f"dumped {len(steps)} steps of top-8 logits to {dump}")
        out = out.sequences
    new = out[0, n_prompt:]
    toks = proc.tokenizer.convert_ids_to_tokens(new.tolist())
    print(
        json.dumps(
            {
                "n_prompt": n_prompt,
                "n_new": int(new.shape[0]),
                "ids": [int(x) for x in new],
                "tokens": toks,
                "text": proc.tokenizer.decode(new, skip_special_tokens=False),
            }
        )
    )
    if args.out:
        json.dump([int(x) for x in new], open(args.out, "w"))


def cmd_logits(args):
    import torch

    proc, model, torch = load(args, getattr(torch, args.dtype))
    _, text = render(args, proc)
    inputs = media_inputs(args, proc, text)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1].float()
    top = torch.topk(logits, args.top)
    print(
        json.dumps(
            {
                "n_prompt": int(inputs["input_ids"].shape[1]),
                "top_ids": [int(x) for x in top.indices],
                "top_vals": [float(x) for x in top.values],
                "argmax": int(logits.argmax()),
            }
        )
    )
    if args.out:
        torch.save(logits, args.out)


def cmd_hidden(args):
    import torch

    proc, model, torch = load(args, getattr(torch, args.dtype))
    _, text = render(args, proc)
    inputs = proc.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = model.model.language_model(
            input_ids=inputs["input_ids"], output_hidden_states=True
        )
    hs = out.hidden_states
    for idx in args.layers:
        h = hs[idx].reshape(-1).float()
        print(
            f"hidden[{idx}] shape={tuple(hs[idx].shape)} mean={h.mean():.6f} std={h.std():.6f} norm={h.norm():.3f}"
        )
    if args.out:
        torch.save({i: hs[i].float() for i in args.layers}, args.out)
        print(f"saved {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("tokens", "gen", "logits", "hidden"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", default="/mnt/hub/models/gemma-4-E2B-it")
        sp.add_argument("--text", default=None)
        sp.add_argument("--text-file", default=None)
        sp.add_argument("--messages", default=None, help="JSON list of chat messages")
        sp.add_argument("--enable-thinking", action="store_true")
        if name in ("gen", "logits", "tokens"):
            sp.add_argument("--image", action="append", default=None)
            sp.add_argument("--audio", action="append", default=None)
        if name == "gen":
            sp.add_argument("--dtype", default="bfloat16")
            sp.add_argument("--max-new-tokens", type=int, default=32)
            sp.add_argument("--greedy", action="store_true")
            sp.add_argument("--temperature", type=float, default=1.0)
            sp.add_argument("--top-k", type=int, default=64)
            sp.add_argument("--top-p", type=float, default=0.95)
            sp.add_argument("--seed", type=int, default=0)
            sp.add_argument("--out", default=None)
            sp.add_argument("--dump-logits", default=None)
        if name == "logits":
            sp.add_argument("--dtype", default="float32")
            sp.add_argument("--top", type=int, default=8)
            sp.add_argument("--out", default=None)
        if name == "hidden":
            sp.add_argument("--dtype", default="float32")
            sp.add_argument("--layers", type=int, nargs="+", default=[0, 15, 35])
            sp.add_argument("--out", default=None)

    args = p.parse_args()
    {"tokens": cmd_tokens, "gen": cmd_gen, "logits": cmd_logits, "hidden": cmd_hidden}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
