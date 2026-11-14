#!/usr/bin/env python3
"""Layer-streaming reference/oracle for Cloudflare/clef-omni (text-only records).

The full BF16 checkpoint is ~70 GB, so this script never loads the model: it runs the
Qwen3-Omni thinker text decoder ONE layer at a time in float32 (the layer's tensors are
read from the safetensors shards through the index, a real transformers decoder layer
module is built from them, run, and freed), then the final RMSNorm and the unchanged
`JointSchemaHead` from the release (`clef_joint_schema_model.py`, a verbatim copy of the
release's `joint_schema_model.py`).

    python -I examples/rocket-inference/tools/clef_reference.py \
        --model-dir /path/to/clef-omni --requests examples/rocket-inference/tools/clef_fixtures.json \
        --out-dir /tmp/clef-dump [--only checkout-errors] [--selfcheck] [--dump-layers 0,1,23,47]

Text-only: vision/audio tensors are never touched; records with images/audio/videos are
refused. See docs/recipes/clef-oracle.md for the dump format.

The checkpoint is untrusted data: run with `python -I`, and never use the model dir as CWD.
This script imports only the local copy of joint_schema_model.py (by explicit file path).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PEAK = [0]  # peak RssAnon sampled after every layer (MiB)


def load_joint_module():
    spec = importlib.util.spec_from_file_location(
        "clef_joint_schema_model", HERE / "clef_joint_schema_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clef_joint_schema_model"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
# Checkpoint access (lazy, via the safetensors index)
# --------------------------------------------------------------------------------------
class Shards:
    """Reads individual tensors from the sharded checkpoint; keeps one open handle/shard."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.dir = model_dir
        self.map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def _h(self, name):
        shard = self.map[name]
        if shard not in self.handles:
            self.handles[shard] = self._safe_open(str(self.dir / shard), framework="pt")
        return self.handles[shard]

    def close(self):
        """Drop the mmap handles (their touched file pages count toward maxrss otherwise)."""
        self.handles.clear()

    def get(self, name) -> torch.Tensor:
        """Whole tensor as float32."""
        return self._h(name).get_tensor(name).to(torch.float32)

    def rows(self, name, ids) -> torch.Tensor:
        """Selected rows of a 2-D tensor as float32 (one slice read per unique id)."""
        sl = self._h(name).get_slice(name)
        out = {int(i): sl[int(i) : int(i) + 1].to(torch.float32)[0] for i in sorted(set(int(x) for x in ids))}
        return torch.stack([out[int(i)] for i in ids])


class LazyRows:
    """`weight[token_ids]` without materialising [vocab, hidden] (lm_head rows for options)."""

    def __init__(self, shards: Shards, name: str):
        self.shards, self.name, self.cache = shards, name, {}

    def __getitem__(self, token_ids):
        ids = [int(i) for i in token_ids.reshape(-1).tolist()]
        need = [i for i in set(ids) if i not in self.cache]
        if need:
            got = self.shards.rows(self.name, need)
            for i, r in zip(need, got):
                self.cache[i] = r
        return torch.stack([self.cache[i] for i in ids]).reshape(*token_ids.shape, -1)


# --------------------------------------------------------------------------------------
# HF decoder layer construction
# --------------------------------------------------------------------------------------
def text_config(model_dir: Path):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir)
    tc = cfg.thinker_config.text_config
    tc._attn_implementation = "sdpa"
    return tc


def layer_tensors(sh: Shards, i: int, tc) -> dict:
    """All float32 tensors of decoder layer i, experts fused as HF's module stores them."""
    p = f"thinker.model.layers.{i}."
    t = {}
    for n in (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate.weight",
    ):
        t[n] = sh.get(p + n)
    E = tc.num_experts
    # HF fuses [gate; up] along the output dim: gate_up_proj[e] = cat(gate_proj, up_proj) -> [2*I, H]
    t["mlp.experts.gate_up_proj"] = torch.stack(
        [torch.cat([sh.get(f"{p}mlp.experts.{e}.gate_proj.weight"), sh.get(f"{p}mlp.experts.{e}.up_proj.weight")], 0) for e in range(E)]
    )
    t["mlp.experts.down_proj"] = torch.stack([sh.get(f"{p}mlp.experts.{e}.down_proj.weight") for e in range(E)])
    return t


def build_layer(tc, i: int, t: dict):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

    with torch.device("meta"):
        layer = m.Qwen3OmniMoeThinkerTextDecoderLayer(tc, i)
    sd = {k: v for k, v in t.items()}
    layer.load_state_dict(sd, strict=True, assign=True)
    return layer.eval()


def rope_cache(tc, T):
    """cos/sin from the real HF rotary module (text-only: the 3 mRoPE streams are equal)."""
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

    rot = m.Qwen3OmniMoeThinkerTextRotaryEmbedding(tc)
    pos = torch.arange(T).view(1, 1, T).expand(3, 1, T)
    dummy = torch.zeros(1, T, tc.hidden_size)
    return rot(dummy, pos)


# --------------------------------------------------------------------------------------
# Independent plain-torch implementation of one decoder layer (--selfcheck, Rust port spec)
# --------------------------------------------------------------------------------------
def manual_rmsnorm(x, w, eps=1e-6):
    # y = w * x * rsqrt(mean(x^2, last dim) + eps), computed in f32.
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def manual_layer(x, t, tc, return_routing=False):
    """x: [T, H] f32 -> [T, H].  Exactly the Qwen3-Omni-MoE thinker text layer, no HF code.

    Per layer (pre-norm residual blocks):
      h = x + Attn(RMSNorm_in(x));   y = h + MoE(RMSNorm_post(h))
    Attention (32 q heads, 4 kv heads, head_dim 128, no biases):
      q = q_norm(q_proj(n).view(T,32,128)); k = k_norm(k_proj(n).view(T,4,128))  # QK-norm: per-head RMSNorm
          over head_dim with its own [128] weight, applied BEFORE rope; v = v_proj(n).view(T,4,128)
      RoPE (theta 1e6, rotate-half layout, NOT interleaved pairs): inv_freq[j] = theta^(-2j/128), j<64,
          angle[t,j] = t*inv_freq[j]; cos/sin = cat(cos(angle), cos(angle)) over 128 dims;
          x' = x*cos + rotate_half(x)*sin, rotate_half(x) = cat(-x[64:], x[:64]).
          Qwen3-Omni's "interleaved mRoPE [24,20,20]" only mixes the T/H/W position streams per
          frequency slot; with text-only input all three streams equal t, so it is plain RoPE.
      scores = q.k^T / sqrt(128), causal mask, softmax f32, GQA: q head h uses kv head h // 8.
      out = o_proj(concat heads)
    MoE (128 experts, top-8, no shared expert):
      logits = gate(n) [T,128]; probs = softmax(logits) (f32); (w, idx) = topk(probs, 8);
      w /= sum(w)   (norm_topk_prob)
      expert e: down_e( silu(gate_e(n)) * up_e(n) ); out[t] = sum_k w[t,k] * expert_{idx[t,k]}(n[t])
    """
    T, H = x.shape
    nh, nkv, D = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    eps = tc.rms_norm_eps
    n = manual_rmsnorm(x, t["input_layernorm.weight"], eps)
    q = manual_rmsnorm((n @ t["self_attn.q_proj.weight"].T).view(T, nh, D), t["self_attn.q_norm.weight"], eps)
    k = manual_rmsnorm((n @ t["self_attn.k_proj.weight"].T).view(T, nkv, D), t["self_attn.k_norm.weight"], eps)
    v = (n @ t["self_attn.v_proj.weight"].T).view(T, nkv, D)
    theta = float(tc.rope_theta if hasattr(tc, "rope_theta") and tc.rope_theta else tc.rope_parameters["rope_theta"])
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]

    def rot(z):
        return torch.cat([-z[..., D // 2 :], z[..., : D // 2]], -1)

    q = q * cos + rot(q) * sin
    k = k * cos + rot(k) * sin
    rep = nh // nkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(D)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    scores = scores.masked_fill(causal, float("-inf"))
    a = torch.softmax(scores, -1)
    ctx = torch.einsum("hts,shd->thd", a, v).reshape(T, nh * D)
    h = x + ctx @ t["self_attn.o_proj.weight"].T

    n2 = manual_rmsnorm(h, t["post_attention_layernorm.weight"], eps)
    probs = torch.softmax(n2 @ t["mlp.gate.weight"].T, -1)
    w, idx = torch.topk(probs, tc.num_experts_per_tok, -1)
    if tc.norm_topk_prob:
        w = w / w.sum(-1, keepdim=True)
    gu, dn = t["mlp.experts.gate_up_proj"], t["mlp.experts.down_proj"]
    moe = torch.zeros_like(h)
    I = tc.moe_intermediate_size
    for e in range(tc.num_experts):
        tok, slot = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        g, u = (n2[tok] @ gu[e].T).split(I, -1)
        moe.index_add_(0, tok, (F.silu(g) * u) @ dn[e].T * w[tok, slot, None])
    y = h + moe
    return (y, idx, w) if return_routing else y


# --------------------------------------------------------------------------------------
# Streaming forward
# --------------------------------------------------------------------------------------
def run_thinker(sh: Shards, tc, input_ids, dump_layers, selfcheck, log):
    T = len(input_ids)
    x = sh.rows("thinker.model.embed_tokens.weight", input_ids)  # [T, H]
    out = {"embeddings": x.clone()}
    cos, sin = rope_cache(tc, T)
    mask = torch.full((T, T), float("-inf")).triu(1)[None, None]  # additive causal mask
    checks = {}
    for i in range(tc.num_hidden_layers):
        t0 = time.time()
        t = layer_tensors(sh, i, tc)
        t_load = time.time() - t0
        sh.close()
        layer = build_layer(tc, i, t)
        routed = {}

        def hook(mod, inp, outp):
            logits = outp[0]
            # recompute identically to the transformers router: softmax f32, top-k, renorm
            probs = torch.softmax(inp[0].reshape(-1, tc.hidden_size) @ mod.weight.T, -1)
            w, idx = torch.topk(probs, tc.num_experts_per_tok, -1)
            if tc.norm_topk_prob:
                w = w / w.sum(-1, keepdim=True)
            assert torch.equal(idx, outp[2]), f"layer {i}: router indices differ from transformers"
            torch.testing.assert_close(w, outp[1], rtol=1e-5, atol=1e-6)
            routed["idx"], routed["w"] = outp[2].to(torch.int32), outp[1].float()

        hh = layer.mlp.gate.register_forward_hook(hook)
        with torch.inference_mode():
            y = layer(x[None], attention_mask=mask, position_ids=None, position_embeddings=(cos, sin))
            y = y[0] if isinstance(y, tuple) else y
            y = y[0]
        hh.remove()
        if selfcheck and i in (0, tc.num_hidden_layers - 1):
            with torch.inference_mode():
                ym, midx, mw = manual_layer(x, t, tc, return_routing=True)
            checks[i] = {
                "max_abs_diff": float((ym - y).abs().max()),
                "max_rel_to_peak": float((ym - y).abs().max() / y.abs().max()),
                "router_ids_equal": bool(torch.equal(midx.to(torch.int32), routed["idx"])),
                "router_w_max_abs": float((mw - routed["w"]).abs().max()),
            }
            log(f"selfcheck layer {i}: {checks[i]}")
        x = y.clone()
        out[f"router_ids_L{i}"] = routed["idx"].numpy()
        out[f"router_w_L{i}"] = routed["w"].numpy()
        if i in dump_layers:
            out[f"hidden_L{i}"] = x.clone()
        del layer, t, y
        log(f"layer {i:2d} load {t_load:.1f}s total {time.time() - t0:.1f}s rss_anon {rss_anon_mib()} MiB")
        PEAK[0] = max(PEAK[0], rss_anon_mib())
    final_w = sh.get("thinker.model.norm.weight")
    xn = manual_rmsnorm(x, final_w, tc.rms_norm_eps)
    out["final_hidden"] = x  # raw output of layer 47 (before the final norm)
    out["final_normed"] = xn
    return out, checks


def rss_anon_mib():
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) // 1024
    return 0


def parse_layers(s, n):
    ls = {n - 1}
    if s == "all":
        return set(range(n))
    for p in filter(None, s.split(",")):
        ls.add(int(p))
    return ls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--requests", required=True, help="JSON list of SystemOne-style records")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default="", help="comma separated record ids")
    ap.add_argument("--dump-layers", default="all", help="comma list (final layer always) or 'all'")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selfcheck", action="store_true", help="compare layers 0 and 47 against the manual torch layer")
    args = ap.parse_args()
    torch.set_num_threads(min(args.threads, 8))
    log = lambda s: print(s, flush=True)

    jm = load_joint_module()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tc = text_config(model_dir)
    sh = Shards(model_dir)
    head_cfg = json.loads((model_dir / "joint_head_config.json").read_text())
    from safetensors.torch import load_file

    head = jm.JointSchemaHead(**head_cfg)
    head.load_state_dict(load_file(str(model_dir / "joint_head.safetensors")), strict=True)
    head = head.float().eval()
    lm = LazyRows(sh, "thinker.lm_head.weight")

    records = json.loads(Path(args.requests).read_text())
    only = set(filter(None, args.only.split(",")))
    dump_layers = parse_layers(args.dump_layers, tc.num_hidden_layers)
    for rec in records:
        rid = str(rec.get("id", "unknown"))
        if only and rid not in only:
            continue
        if any(rec.get(k) for k in ("images", "audio", "videos")):
            raise SystemExit(f"{rid}: media records are not supported by the oracle")
        enc = jm.encode_record(tok, rec)
        T = len(enc.input_ids)
        if T > args.max_tokens:
            raise SystemExit(f"{rid}: {T} tokens > --max-tokens {args.max_tokens}")
        log(f"== {rid}: {T} tokens")
        t0 = time.time()
        out, checks = run_thinker(sh, tc, list(enc.input_ids), dump_layers, args.selfcheck, log)
        ids_t = torch.tensor([enc.input_ids])
        with torch.inference_mode():
            logits = head(out["final_normed"][None], ids_t, torch.ones_like(ids_t), [enc], lm)[0]
        answers, arrays = {}, {}
        for q, lg in zip(enc.questions, logits):
            p = lg.float().softmax(-1)
            arrays[f"logits_{q.question_id}"] = lg.float().numpy()
            arrays[f"probs_{q.question_id}"] = p.numpy()
            answers[q.question_id] = jm.systemone_answer(rec["questions"][q.question_id], dict(zip(q.option_ids, p.tolist())))
        npz = {"input_ids": np.array(enc.input_ids, dtype=np.int64)}
        npz.update({k: (v.numpy() if torch.is_tensor(v) else v) for k, v in out.items()})
        npz.update(arrays)
        np.savez(out_dir / f"{rid}.npz", **npz)
        summary = {
            "id": rid,
            "model": rec.get("model", "clef-omni"),
            "answers": answers,
            "usage": {"input_tokens": T, "output_tokens": 0},
            "questions": [
                {
                    "id": q.question_id,
                    "type": q.question_type,
                    "question_span": list(q.question_span),
                    "option_spans": [list(s) for s in q.option_spans],
                    "option_ids": list(q.option_ids),
                }
                for q in enc.questions
            ],
            "dump_layers": sorted(dump_layers),
            "selfcheck": {str(k): v for k, v in checks.items()},
            "wall_s": round(time.time() - t0, 1),
            "peak_rss_anon_mib": PEAK[0],
        }
        (out_dir / f"{rid}.json").write_text(json.dumps(summary, indent=1))
        log(json.dumps(answers))
        log(f"{rid}: {summary['wall_s']}s")


if __name__ == "__main__":
    main()
