#!/usr/bin/env python3
"""Layer-streaming reference/oracle for Cloudflare/clef-omni (text, image, audio and video records).

The full BF16 checkpoint is ~70 GB, so this script never loads the model: it runs the
Qwen3-Omni thinker text decoder ONE layer at a time in float32 (the layer's tensors are
read from the safetensors shards through the index, a real transformers decoder layer
module is built from them, run, and freed), then the final RMSNorm and the unchanged
`JointSchemaHead` from the release (`clef_joint_schema_model.py`, a verbatim copy of the
release's `joint_schema_model.py`).

    python -I tools/clef_reference.py \
        --model-dir /path/to/clef-omni --requests tools/clef_fixtures.json \
        --out-dir /tmp/clef-dump [--only checkout-errors] [--selfcheck] [--dump-layers 0,1,23,47]

Records with `images`/`audio`/`videos` additionally run the vision/audio towers in f32 (clef_media.py), splice
their features into the embeddings, use the 3-D mRoPE positions of the real HF `get_rope_index`, and inject
the DeepStack features after decoder layers 0-2. Relative media paths resolve against `--media-base`
(default: examples/rocket-inference). See docs/experiments/clef-oracle.md for the dump format.

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
MEDIA = None  # clef_media module, loaded in main()
PEAK = [0]  # peak RssAnon sampled after every layer (MiB)


def load_joint_module():
    spec = importlib.util.spec_from_file_location(
        "clef_joint_schema_model", HERE / "clef_joint_schema_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clef_joint_schema_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_media_module():
    spec = importlib.util.spec_from_file_location("clef_media", HERE / "clef_media.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clef_media"] = mod
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


def rope_cache(tc, T, pos3=None):
    """cos/sin from the real HF rotary module. pos3 [3,T] = the (t,h,w) mRoPE streams; None = text-only (all equal)."""
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

    rot = m.Qwen3OmniMoeThinkerTextRotaryEmbedding(tc)
    pos = torch.arange(T).view(1, 1, T).expand(3, 1, T) if pos3 is None else pos3.view(3, 1, T).clone()
    dummy = torch.zeros(1, T, tc.hidden_size)
    return rot(dummy, pos)


# --------------------------------------------------------------------------------------
# Independent plain-torch implementation of one decoder layer (--selfcheck, Rust port spec)
# --------------------------------------------------------------------------------------
def manual_rmsnorm(x, w, eps=1e-6):
    # y = w * x * rsqrt(mean(x^2, last dim) + eps), computed in f32.
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def manual_layer(x, t, tc, return_routing=False, pos3=None):
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
          frequency slot; with text-only input all three streams equal t, so it is plain RoPE. With `pos3`
          ([3,T] positions, media records) the per-slot stream choice of clef_media.manual_mrope_cos_sin is used.
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
    if pos3 is None:
        ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    else:
        cos, sin = MEDIA.manual_mrope_cos_sin(pos3, D, theta)
        cos, sin = cos[:, None, :], sin[:, None, :]

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
def run_thinker(sh: Shards, tc, input_ids, dump_layers, selfcheck, log, embeds=None, pos3=None, ds=None):
    """embeds: spliced input embeddings (media records) else the token rows; pos3 [3,T] mRoPE positions or None;
    ds = (visual_pos_idx [n], [3 x [n,H]]) DeepStack features added to the hidden state after layers 0..2."""
    T = len(input_ids)
    x = embeds if embeds is not None else sh.rows("thinker.model.embed_tokens.weight", input_ids)  # [T, H]
    out = {"embeddings": x.clone()}
    cos, sin = rope_cache(tc, T, pos3)
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
                ym, midx, mw = manual_layer(x, t, tc, return_routing=True, pos3=pos3)
            checks[i] = {
                "max_abs_diff": float((ym - y).abs().max()),
                "max_rel_to_peak": float((ym - y).abs().max() / y.abs().max()),
                "router_ids_equal": bool(torch.equal(midx.to(torch.int32), routed["idx"])),
                "router_w_max_abs": float((mw - routed["w"]).abs().max()),
            }
            log(f"selfcheck layer {i}: {checks[i]}")
        x = y.clone()
        if ds is not None and i < len(ds[1]):
            x[ds[0]] += ds[1][i]  # DeepStack: visual positions get the i-th tower tap added after layer i
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


def _np(v):
    if torch.is_tensor(v):
        v = v.detach()
        return v.float().numpy() if v.is_floating_point() else v.numpy()
    return np.asarray(v)


def _release():
    """Drop tower weights and hand the freed pages back to the OS (a tower build transiently holds 3-5 GB)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def run_media(jm, sh, tcfg, tc, rec, enc, base, args, log):
    """Towers + splice + DeepStack + 3-D positions for one media record (all f32).

    Returns (embeds [T,H], pos3 [3,T], (visual_pos_idx, [3 x [n,H]]) or None, extra dump arrays, info dict)."""
    import gc

    M = MEDIA
    media = enc.media
    uaiv = bool(media.get("use_audio_in_video", False))
    ids = list(enc.input_ids)
    T = len(ids)
    extra, info = {}, {"use_audio_in_video": uaiv, "tower_selfcheck": {}}

    # decoded media (release decoders), so a port can gate decode and preprocessing separately
    dec = M.decode_media(jm, M.resolve_media(rec, base))
    for i, a in enumerate(dec["images"]):
        extra[f"media_image_rgb_{i}"] = a
    for i, fr in enumerate(dec["videos"]):
        extra[f"media_video_frames_{i}"] = fr
    audio_list = list(dec["audio"]) + ([t for t in dec["video_tracks"] if t is not None] if uaiv else [])
    for i, a in enumerate(audio_list):
        extra[f"media_audio_samples_{i}"] = np.asarray(a, dtype=np.float32)
    info["audio_clips"] = [int(a.shape[0]) for a in audio_list]
    info["video_frames"] = [list(v.shape) for v in dec["videos"]]
    info["image_sizes"] = [list(a.shape[:2]) for a in dec["images"]]
    for k in jm.MEDIA_BATCH_KEYS + jm.AUDIO_BATCH_KEYS:
        if k in media:
            extra[f"media_{k}"] = _np(media[k])
    extra["media_use_audio_in_video"] = np.array([uaiv], dtype=np.int8)
    pl = M.placeholder_positions(tcfg, ids)
    for k, v in pl.items():
        extra[f"placeholder_idx_{k}"] = v
    info["placeholders"] = {k: int(len(v)) for k, v in pl.items()}
    info["special_tokens_in_input"] = {
        str(t): int(sum(1 for i in ids if i == t))
        for t in (tcfg.vision_start_token_id, tcfg.vision_end_token_id, tcfg.audio_start_token_id, tcfg.audio_end_token_id)
    }

    emb = sh.rows("thinker.model.embed_tokens.weight", ids)
    extra["embeddings_text"] = emb.clone().numpy()
    vdump = [int(x) for x in args.vision_dump_blocks.split(",") if x != ""]
    adump = [int(x) for x in args.audio_dump_layers.split(",") if x != ""]
    sc = info["tower_selfcheck"]
    tower_t = {}

    if "input_features" in media:
        t0 = time.time()
        ea = M.build_audio(sh, tcfg)
        res = M.run_audio(ea, media["input_features"].float(), media["feature_attention_mask"], adump)
        tower_t["audio_s"] = round(time.time() - t0, 1)
        if args.selfcheck:
            sd = dict(ea.state_dict())
            with torch.inference_mode():
                conv, feats = M.manual_audio(sd, media["input_features"].float(), media["feature_attention_mask"], tcfg)
            sc["audio_conv_out"] = M.cmp(res["audio_conv_out"], conv)
            sc["audio_features"] = M.cmp(res["audio_features"], feats)
            log(f"selfcheck audio tower: {sc['audio_features']} conv {sc['audio_conv_out']}")
        del ea
        _release()
        feats = res["audio_features"]
        assert feats.shape[0] == len(pl["audio"]), (feats.shape, len(pl["audio"]))
        emb[pl["audio"]] = feats
        extra.update({k: _np(v) for k, v in res.items()})
        info["audio_tokens"] = int(feats.shape[0])
        info["audio_feature_lengths"] = media["feature_attention_mask"].sum(-1).tolist()
        log(f"audio tower {tower_t['audio_s']}s -> {tuple(feats.shape)} rss_anon {rss_anon_mib()} MiB")
        PEAK[0] = max(PEAK[0], rss_anon_mib())

    vis = {}
    if "pixel_values" in media or "pixel_values_videos" in media:
        ev = M.build_vision(sh, tcfg)
        sd = dict(ev.state_dict()) if args.selfcheck else None
        for prefix, pvk, gk in (("image", "pixel_values", "image_grid_thw"), ("video", "pixel_values_videos", "video_grid_thw")):
            if pvk not in media:
                continue
            t0 = time.time()
            pv, grid = media[pvk].float(), media[gk]
            res, out = M.run_vision(ev, pv, grid, vdump, prefix)
            tower_t[f"{prefix}_s"] = round(time.time() - t0, 1)
            if args.selfcheck:
                with torch.inference_mode():
                    last, merged, deep = M.manual_vision(sd, pv, grid, tcfg)
                sc[f"{prefix}_last"] = M.cmp(out.last_hidden_state, last)
                sc[f"{prefix}_merged"] = M.cmp(out.pooler_output, merged)
                for j, d in enumerate(deep):
                    sc[f"{prefix}_deepstack_{j}"] = M.cmp(out.deepstack_features[j], d)
                log(f"selfcheck {prefix} tower: last {sc[f'{prefix}_last']} merged {sc[f'{prefix}_merged']} ds2 {sc[f'{prefix}_deepstack_2']}")
            merged = out.pooler_output
            assert merged.shape[0] == len(pl[prefix]), (prefix, merged.shape, len(pl[prefix]))
            emb[pl[prefix]] = merged
            vis[prefix] = (pl[prefix], out.deepstack_features)
            extra.update({k: _np(v) for k, v in res.items()})
            extra[f"{prefix}_patch_count"] = np.array([pv.shape[0]], dtype=np.int64)
            info[f"{prefix}_grid_thw"] = grid.tolist()
            info[f"{prefix}_patches"] = int(pv.shape[0])
            info[f"{prefix}_tokens"] = int(merged.shape[0])
            log(f"{prefix} tower {tower_t[f'{prefix}_s']}s grid {grid.tolist()} -> {tuple(merged.shape)} rss_anon {rss_anon_mib()} MiB")
            PEAK[0] = max(PEAK[0], rss_anon_mib())
        del ev, sd
        _release()

    ds = None
    if vis:
        vis_idx = np.sort(np.concatenate([v[0] for v in vis.values()]))
        n_ds = len(tcfg.vision_config.deepstack_visual_indexes)
        lists = []
        for j in range(n_ds):
            joint = torch.zeros(len(vis_idx), tc.hidden_size)
            for prefix, (idx, feats) in vis.items():
                joint[torch.from_numpy(np.searchsorted(vis_idx, idx))] = feats[j]
            lists.append(joint)
            extra[f"deepstack_L{j}"] = joint.numpy()
        extra["visual_pos_idx"] = vis_idx.astype(np.int64)
        ds = (torch.from_numpy(vis_idx), lists)

    afl = media["feature_attention_mask"].sum(-1) if "feature_attention_mask" in media else None
    pos3, delta = M.position_ids(
        tcfg, ids, media.get("image_grid_thw"), media.get("video_grid_thw"), afl, uaiv, media.get("video_second_per_grid")
    )
    extra["rope_delta"] = np.array([delta], dtype=np.float32)
    info["rope_delta"] = delta
    info["position_ids_integral"] = bool(torch.equal(pos3, pos3.round()))
    info["position_max"] = float(pos3.max())
    info["position_branch"] = "3d" if ("image_grid_thw" in media or "video_grid_thw" in media) else "text-only (no image/video grid: plain cumsum positions)"
    info["tower_s"] = tower_t
    return emb, pos3, ds, extra, info


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
    ap.add_argument("--selfcheck", action="store_true", help="compare layers 0 and 47 (and the vision/audio towers of media records) against plain-torch re-implementations")
    ap.add_argument("--media-base", default=str(HERE.parent), help="base dir for relative media paths in the records")
    ap.add_argument("--vision-dump-blocks", default="0,8,16,24,26", help="vision tower block outputs to dump (pre-merge)")
    ap.add_argument("--audio-dump-layers", default="0,15,31", help="audio tower layer outputs to dump")
    args = ap.parse_args()
    torch.set_num_threads(min(args.threads, 8))
    log = lambda s: print(s, flush=True)

    global MEDIA
    jm = load_joint_module()
    MEDIA = load_media_module()
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

    tcfg = MEDIA.thinker_cfg(model_dir)
    processor = None
    records = json.loads(Path(args.requests).read_text())
    only = set(filter(None, args.only.split(",")))
    dump_layers = parse_layers(args.dump_layers, tc.num_hidden_layers)
    for rec in records:
        rid = str(rec.get("id", "unknown"))
        if only and rid not in only:
            continue
        media_rec = MEDIA.has_media(rec)
        t0 = time.time()
        extra, minfo, embeds, pos3, ds = {}, {}, None, None, None
        if media_rec:
            if processor is None:
                from transformers import AutoProcessor

                processor = MEDIA.CompatProcessor(AutoProcessor.from_pretrained(model_dir))
            enc = jm.encode_record(tok, MEDIA.resolve_media(rec, Path(args.media_base)), processor=processor)
        else:
            enc = jm.encode_record(tok, rec)
        T = len(enc.input_ids)
        if T > args.max_tokens:
            raise SystemExit(f"{rid}: {T} tokens > --max-tokens {args.max_tokens}")
        log(f"== {rid}: {T} tokens")
        if media_rec:
            embeds, pos3, ds, extra, minfo = run_media(
                jm, sh, tcfg, tc, rec, enc, Path(args.media_base), args, log
            )
        out, checks = run_thinker(sh, tc, list(enc.input_ids), dump_layers, args.selfcheck, log, embeds, pos3, ds)
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
        npz["position_ids"] = (pos3 if pos3 is not None else torch.arange(T).float().expand(3, T)).numpy().astype(np.float32)
        npz.update({k: (v.numpy() if torch.is_tensor(v) else v) for k, v in out.items()})
        npz.update(extra)
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
            "media": minfo or None,
            "selfcheck": {str(k): v for k, v in checks.items()},
            "wall_s": round(time.time() - t0, 1),
            "peak_rss_anon_mib": PEAK[0],
        }
        (out_dir / f"{rid}.json").write_text(json.dumps(summary, indent=1))
        log(json.dumps(answers))
        log(f"{rid}: {summary['wall_s']}s")


if __name__ == "__main__":
    main()
