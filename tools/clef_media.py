"""Media half of the Clef-Omni f32 oracle (images, audio, video) - imported by clef_reference.py.

Everything here runs in float32 on the CPU:

* `CompatProcessor`   - adapts the release's `encode_record` (written for transformers 5.10) to the
                        transformers 5.2x processor API (`use_audio_in_video` -> `load_audio_from_video`).
* `resolve_media`     - fixture paths relative to a base dir -> absolute (data: / http(s) kept).
* `decode_media`      - the release's own decoders (`load_image/load_audio/load_video`), for the dump.
* `run_vision`/`run_audio` - the REAL HF `Qwen3OmniMoeVisionEncoder` / `Qwen3OmniMoeAudioEncoder` built from
                        the shard weights (all tower tensors are in shard 1), freed after use.
* `manual_vision`/`manual_audio` - independent plain-torch re-implementations (no HF modelling code) used by
                        `--selfcheck`; they are also the spec for the Rust port (see the docstrings).
* `position_ids`      - the real HF `get_rope_index` (called unbound) for the 3-D mRoPE positions.
* `manual_mrope_cos_sin` - interleaved mRoPE [24,20,20] from first principles (selfcheck of the HF rotary).
"""

from __future__ import annotations

import math
import re
import time
import types

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Processor compatibility / record plumbing
# --------------------------------------------------------------------------------------
class CompatProcessor:
    """`encode_record` calls `processor(text=..., images=..., audio=..., videos=..., add_special_tokens=False,
    return_tensors='pt', **options)` and, for videos whose soundtrack is used, passes
    `videos_kwargs={'fps': 2.0, 'do_sample_frames': False, 'use_audio_in_video': True}` with the track audio
    appended to `audio` (release 5.10 semantics). transformers >= 5.23 rejects that (the track must be declared
    through `load_audio_from_video` + `audio_from_video_indices`); this proxy does exactly that translation and
    changes nothing else. The token layout it produces is the one documented in docs/experiments/clef-oracle.md."""

    def __init__(self, proc):
        self._p = proc
        self.tokenizer = proc.tokenizer

    def __call__(self, text=None, images=None, audio=None, videos=None, **kw):
        vk = dict(kw.get("videos_kwargs") or {})
        if vk.pop("use_audio_in_video", False):
            n_vid, n_aud = len(videos), len(audio)
            kw["videos_kwargs"] = vk
            kw["load_audio_from_video"] = True
            # audio = explicit clips first (placeholder in the text), then one track per video, in video order
            kw["audio_from_video_indices"] = [None] * (n_aud - n_vid) + list(range(n_vid))
        return self._p(text=text, images=images, audio=audio, videos=videos, **kw)


def _is_url_or_data(s: str) -> bool:
    return s.startswith(("data:", "http://", "https://"))


def resolve_media(rec: dict, base) -> dict:
    """Copy of the record with relative media paths made absolute against `base` (the dump keeps the originals)."""
    out = dict(rec)
    for key in ("images", "audio", "videos"):
        if rec.get(key):
            out[key] = [v if (not isinstance(v, str) or _is_url_or_data(v) or v.startswith("/")) else str(base / v) for v in rec[key]]
    return out


def has_media(rec: dict) -> bool:
    return any(rec.get(k) for k in ("images", "audio", "videos"))


def decode_media(jm, rec: dict) -> dict:
    """What the release decoders produce, for the dump (so a port can gate decode and preprocessing apart)."""
    out = {"images": [], "audio": [], "videos": [], "video_tracks": []}
    for v in rec.get("images") or []:
        out["images"].append(np.asarray(jm.load_image(v)))  # uint8 [H,W,3]
    for v in rec.get("audio") or []:
        out["audio"].append(jm.load_audio(v))  # f32 [n] 16 kHz mono
    for v in rec.get("videos") or []:
        frames, track = jm.load_video(v)
        out["videos"].append(frames)  # uint8 [F,H,W,3], sampled at 2 fps, even F
        out["video_tracks"].append(track)  # f32 [n] or None
    return out


# --------------------------------------------------------------------------------------
# Tower construction from the shards
# --------------------------------------------------------------------------------------
def thinker_cfg(model_dir):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_dir).thinker_config


def _tower_state(sh, prefix: str) -> dict:
    names = [n for n in sh.map if n.startswith(prefix)]
    return {n[len(prefix):]: sh.get(n) for n in names}


def build_vision(sh, cfg):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

    vc = cfg.vision_config
    vc._attn_implementation = "sdpa"
    enc = m.Qwen3OmniMoeVisionEncoder(vc).float().eval()
    sd = _tower_state(sh, "thinker.visual.")
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected and not [k for k in missing if "inv_freq" not in k], (missing, unexpected)
    return enc


def build_audio(sh, cfg):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

    ac = cfg.audio_config
    ac._attn_implementation = "sdpa"
    enc = m.Qwen3OmniMoeAudioEncoder(ac).float().eval()
    sd = _tower_state(sh, "thinker.audio_tower.")
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    return enc


# --------------------------------------------------------------------------------------
# HF tower runs (with block hooks for the dump)
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def run_vision(enc, pixel_values, grid_thw, dump_blocks, prefix: str):
    """pixel_values [P,1536] f32, grid_thw [n,3] -> dict of arrays; also returns the raw outputs."""
    taps = {}
    hooks = []
    for k in dump_blocks:
        hooks.append(enc.blocks[k].register_forward_hook(lambda mod, inp, outp, k=k: taps.__setitem__(k, outp.detach().clone())))
    pe = {}
    hooks.append(enc.patch_embed.register_forward_hook(lambda mod, inp, outp: pe.__setitem__("x", outp.detach().clone())))
    out = enc(pixel_values, grid_thw=grid_thw)
    for h in hooks:
        h.remove()
    res = {
        f"{prefix}_tower_last": out.last_hidden_state,  # after block 26, before the merger [P,1152]
        f"{prefix}_merged": out.pooler_output,  # merger(...)  [P/4,2048]
    }
    for j, d in enumerate(out.deepstack_features):
        res[f"{prefix}_deepstack_{j}"] = d  # [P/4,2048], taps after blocks 8/16/24
    for k, v in taps.items():
        res[f"{prefix}_block_{k}"] = v
    return res, out


@torch.inference_mode()
def run_audio(enc, input_features, feature_attention_mask, dump_layers):
    """input_features [n,128,F] f32 + mask [n,F] -> dict of arrays (features [A,2048], valid-token concat)."""
    lens = feature_attention_mask.sum(-1)
    feats = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)  # [128, sum F]
    taps, hooks = {}, []
    for k in dump_layers:
        hooks.append(enc.layers[k].register_forward_hook(lambda mod, inp, outp, k=k: taps.__setitem__(k, (outp[0] if isinstance(outp, tuple) else outp).detach().clone())))
    pre = {}
    hooks.append(enc.layers[0].register_forward_pre_hook(lambda mod, args, kwargs: pre.__setitem__("x", (args[0] if args else kwargs["hidden_states"]).detach().clone()), with_kwargs=True))
    hooks.append(enc.ln_post.register_forward_hook(lambda mod, inp, outp: pre.__setitem__("ln", outp.detach().clone())))
    out = enc(feats, feature_lens=lens, return_dict=True)
    for h in hooks:
        h.remove()
    res = {"audio_features": out.last_hidden_state, "audio_conv_out": pre["x"], "audio_ln_post": pre["ln"]}
    for k, v in taps.items():
        res[f"audio_layer_{k}"] = v
    return res


# --------------------------------------------------------------------------------------
# Independent plain-torch towers (selfcheck + spec for the Rust port)
# --------------------------------------------------------------------------------------
def _ln(x, w, b, eps):
    mu = x.mean(-1, keepdim=True)
    var = (x - mu).pow(2).mean(-1, keepdim=True)
    return (x - mu) * torch.rsqrt(var + eps) * w + b


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def _gelu_erf(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def manual_vision(sd, pixel_values, grid_thw, cfg):
    """Qwen3-Omni vision tower, no HF code. sd = tower tensors by name (without `thinker.visual.`).

    Dimensions (vision_config): hidden 1152, 16 heads (head_dim 72), depth 27, MLP 4304 (gelu tanh),
    patch 16, temporal_patch 2, spatial_merge 2, out_hidden 2048, deepstack taps after blocks {8,16,24},
    position table 48x48 (num_position_embeddings 2304).

    1. patch embed = Conv3d(3,1152,k=(2,16,16),s=same) == linear over the flattened (C,T,16,16) patch (1536 values;
       the processor already stores patches in that order): x = pv @ W.reshape(1152,-1).T + b.
       Patches are stored MERGE-BLOCK MAJOR: for each frame, 2x2 blocks in raster order, 4 patches per block
       (in-block row-major), so 4 consecutive rows form one merged token.
    2. position embedding: the 48x48 table bilinearly resampled (align_corners=True) to the frame's (h, w) grid,
       raster -> block-major reorder, repeated for every temporal group; added to x.
    3. rotary: 2-D axial, theta 10000, spatial_dim = head_dim/2 = 36, inv_freq[j] = 10000^(-2j/36) (j<18);
       freqs = [row*inv_freq (18) | col*inv_freq (18)] (36) duplicated -> 72 = head_dim; rotate-half layout
       (split halves), applied to q,k (f32) of every head. (row, col) = the patch's position in its frame.
    4. blocks: x += attn(LN1(x)); x += fc2(gelu_tanh(fc1(LN2(x)))); LayerNorm eps 1e-6.
       attn: qkv Linear (3456, bias) -> [N,3,16,72]; scale 72^-0.5; attention is FULL within one FRAME
       (cu_seqlens per temporal group of the grid: segments of h*w patches, t segments per video) and never across.
    5. merger (final): LN(1152) over each patch, then view(-1, 4608) (4 consecutive patches), Linear(4608,4608), GELU
       (erf), Linear(4608,2048). Deepstack mergers (3): view(-1,4608) FIRST, then LN(4608), same MLP.
    Returns (last_hidden [P,1152], merged [P/4,2048], [deepstack x3 [P/4,2048]]).
    """
    vc = cfg.vision_config
    H, nh, eps, m = vc.hidden_size, vc.num_heads, 1e-6, vc.spatial_merge_size
    hd = H // nh
    x = pixel_values @ sd["patch_embed.proj.weight"].reshape(H, -1).T + sd["patch_embed.proj.bias"]
    side = int(vc.num_position_embeddings**0.5)
    table = sd["pos_embed.weight"].view(side, side, H).permute(2, 0, 1)[None]  # [1,H,48,48]
    pos, rows, cols = [], [], []
    for t, h, w in grid_thw.tolist():
        interp = F.interpolate(table, size=(h, w), mode="bilinear", align_corners=True)[0]  # [H,h,w]
        e = interp.permute(1, 2, 0)  # [h,w,H] raster
        e = e.view(h // m, m, w // m, m, H).permute(0, 2, 1, 3, 4).reshape(h * w, H)  # block-major
        pos.append(e.repeat(t, 1))
        r, c = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        r = r.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).reshape(-1)
        c = c.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).reshape(-1)
        rows.append(r.repeat(t))
        cols.append(c.repeat(t))
    x = x + torch.cat(pos, 0)
    rr, cc = torch.cat(rows).float(), torch.cat(cols).float()
    sdim = hd // 2
    inv = 1.0 / (vc.rope_parameters["rope_theta"] ** (torch.arange(0, sdim, 2, dtype=torch.float32) / sdim))
    fhw = torch.cat([rr[:, None] * inv, cc[:, None] * inv], -1)  # [N,36]
    cos, sin = torch.cat([fhw, fhw], -1).cos()[:, None, :], torch.cat([fhw, fhw], -1).sin()[:, None, :]
    segs = []
    for t, h, w in grid_thw.tolist():
        segs += [h * w] * t
    bounds = np.cumsum([0] + segs).tolist()

    def rot(z):
        return torch.cat([-z[..., hd // 2:], z[..., : hd // 2]], -1)

    taps, deep = {}, []
    for b in range(vc.depth):
        p = f"blocks.{b}."
        n = _ln(x, sd[p + "norm1.weight"], sd[p + "norm1.bias"], eps)
        qkv = (n @ sd[p + "attn.qkv.weight"].T + sd[p + "attn.qkv.bias"]).view(-1, 3, nh, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        outs = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            sc = torch.einsum("thd,shd->hts", q[s:e], k[s:e]) / math.sqrt(hd)
            outs.append(torch.einsum("hts,shd->thd", torch.softmax(sc, -1), v[s:e]).reshape(e - s, H))
        x = x + torch.cat(outs, 0) @ sd[p + "attn.proj.weight"].T + sd[p + "attn.proj.bias"]
        n2 = _ln(x, sd[p + "norm2.weight"], sd[p + "norm2.bias"], eps)
        x = x + _gelu_tanh(n2 @ sd[p + "mlp.linear_fc1.weight"].T + sd[p + "mlp.linear_fc1.bias"]) @ sd[p + "mlp.linear_fc2.weight"].T + sd[p + "mlp.linear_fc2.bias"]
        if b in vc.deepstack_visual_indexes:
            j = vc.deepstack_visual_indexes.index(b)
            q4 = f"merger_list.{j}."
            d = _ln(x.reshape(-1, H * m * m), sd[q4 + "ln_q.weight"], sd[q4 + "ln_q.bias"], eps)
            d = _gelu_erf(d @ sd[q4 + "mlp.0.weight"].T + sd[q4 + "mlp.0.bias"]) @ sd[q4 + "mlp.2.weight"].T + sd[q4 + "mlp.2.bias"]
            deep.append(d)
    mg = _ln(x, sd["merger.ln_q.weight"], sd["merger.ln_q.bias"], eps).reshape(-1, H * m * m)
    mg = _gelu_erf(mg @ sd["merger.mlp.0.weight"].T + sd["merger.mlp.0.bias"]) @ sd["merger.mlp.2.weight"].T + sd["merger.mlp.2.bias"]
    return x, mg, deep


def audio_out_len(n: int, n_window: int = 50) -> int:
    """Encoder output tokens for `n` mel frames: 13 per full 100-frame chunk + 3 stride-2 convs of the tail."""
    chunk = n_window * 2
    leave = n % chunk
    f = (leave - 1) // 2 + 1
    return ((f - 1) // 2 + 1 - 1) // 2 + 1 + (n // chunk) * 13


def manual_audio(sd, input_features, feature_attention_mask, cfg):
    """Qwen3-Omni audio tower (Whisper-style, chunked), no HF code. sd = tower tensors (no `thinker.audio_tower.`).

    d_model 1280, 32 layers, 20 heads (head_dim 64), FFN 5120 (gelu erf), 128 mel bins, downsample_hidden 480,
    output_dim 2048, n_window 50 (chunk = 100 mel frames), n_window_infer 800, conv_chunksize 500.
    1. each clip's valid mel frames [128, L] are cut into chunks of 100 frames (last chunk shorter, zero-padded to 100
       on the right) -> [chunks,1,128,100].
    2. three Conv2d(k3,s2,p1) with GELU (erf): 480 channels, [480, 16, 13] per chunk (time 100->50->25->13, mel 128->16).
    3. permute to [chunks, 13, 480*16=7680] (time, then channel-major (c,f) flattening), Linear(7680,1280) no bias.
    4. add the sinusoid table [13,1280] (identical for every chunk, positions 0..12): channels = [sin(t*w_j) | cos(t*w_j)],
       w_j = exp(-ln(10000)/(640-1) * j), j<640.
    5. keep only the valid time steps of each chunk: out_len(chunk) (13 full; tail: ((((l-1)//2+1)-1)//2+1-1)//2+1),
       concatenate chunks of all clips -> [A,1280].
    6. attention windows over this flat sequence, per clip: windows of `13 * (800/100) = 104` tokens (max_len_after_cnn *
       8, the full-chunk length of the batch; a shorter last window); full (non-causal) attention inside a window only.
    7. 32 pre-LN layers: x += attn(LN(x)) (q/k/v/out Linear with bias, scale 64^-0.5); x += fc2(gelu(fc1(LN(x)))); LN eps 1e-5.
    8. ln_post (LN), proj1 (1280->1280) + GELU (erf), proj2 (1280->2048). Returns (conv_out [A,1280], out [A,2048]).
    """
    ac = cfg.audio_config
    nw, D, nh = ac.n_window, ac.d_model, ac.encoder_attention_heads
    hd = D // nh
    lens = feature_attention_mask.sum(-1).tolist()
    chunks, valid, per_clip = [], [], []
    for i, L in enumerate(lens):
        mel = input_features[i, :, :L]
        n_chunks = math.ceil(L / (2 * nw))
        cnt = 0
        for c in range(n_chunks):
            piece = mel[:, c * 2 * nw : (c + 1) * 2 * nw]
            valid.append(audio_out_len(piece.shape[1], nw))
            chunks.append(F.pad(piece, (0, 2 * nw - piece.shape[1])))
            cnt += valid[-1]
        per_clip.append(cnt)
    x = torch.stack(chunks)[:, None]  # [n,1,128,100]
    for c in (1, 2, 3):
        x = _gelu_erf(F.conv2d(x, sd[f"conv2d{c}.weight"], sd[f"conv2d{c}.bias"], stride=2, padding=1))
    n, c, f, t = x.shape
    x = x.permute(0, 3, 1, 2).reshape(n, t, c * f) @ sd["conv_out.weight"].T  # [n,13,1280]
    half = D // 2
    inc = math.log(10000.0) / (half - 1)
    inv = torch.exp(-inc * torch.arange(half).float())
    ang = torch.arange(t)[:, None].float() * inv[None, :]
    x = x + torch.cat([ang.sin(), ang.cos()], 1)[None]
    x = torch.cat([x[i, : valid[i]] for i in range(n)], 0)
    conv_out = x.clone()
    window = max(valid) * (ac.n_window_infer // (2 * nw))
    bounds, off = [0], 0
    for cnt in per_clip:
        full, rem = divmod(cnt, window)
        bounds += [off + window * (k + 1) for k in range(full)]
        if rem:
            bounds.append(off + cnt)
        off += cnt
    for li in range(ac.encoder_layers):
        p = f"layers.{li}."
        h = _ln(x, sd[p + "self_attn_layer_norm.weight"], sd[p + "self_attn_layer_norm.bias"], 1e-5)
        q = (h @ sd[p + "self_attn.q_proj.weight"].T + sd[p + "self_attn.q_proj.bias"]).view(-1, nh, hd)
        k = (h @ sd[p + "self_attn.k_proj.weight"].T + sd[p + "self_attn.k_proj.bias"]).view(-1, nh, hd)
        v = (h @ sd[p + "self_attn.v_proj.weight"].T + sd[p + "self_attn.v_proj.bias"]).view(-1, nh, hd)
        outs = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            sc = torch.einsum("thd,shd->hts", q[s:e], k[s:e]) / math.sqrt(hd)
            outs.append(torch.einsum("hts,shd->thd", torch.softmax(sc, -1), v[s:e]).reshape(e - s, D))
        x = x + torch.cat(outs, 0) @ sd[p + "self_attn.out_proj.weight"].T + sd[p + "self_attn.out_proj.bias"]
        h = _ln(x, sd[p + "final_layer_norm.weight"], sd[p + "final_layer_norm.bias"], 1e-5)
        x = x + _gelu_erf(h @ sd[p + "fc1.weight"].T + sd[p + "fc1.bias"]) @ sd[p + "fc2.weight"].T + sd[p + "fc2.bias"]
    x = _ln(x, sd["ln_post.weight"], sd["ln_post.bias"], 1e-5)
    x = _gelu_erf(x @ sd["proj1.weight"].T + sd["proj1.bias"]) @ sd["proj2.weight"].T + sd["proj2.bias"]
    return conv_out, x


# --------------------------------------------------------------------------------------
# 3-D positions (the real HF get_rope_index) + manual interleaved mRoPE
# --------------------------------------------------------------------------------------
def position_ids(cfg, input_ids, image_grid_thw, video_grid_thw, audio_feature_lengths, use_audio_in_video, video_second_per_grid):
    """[3,S] float32 positions + rope delta, from the unbound HF `get_rope_index`.

    NOTE (HF behaviour the port must copy): when the record has NO image and NO video grid (audio only, text only)
    the function takes its `else` branch: plain `cumsum(attention_mask) - 1` positions for all three streams -
    the audio placeholders get consecutive positions, exactly like text."""
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

    base = m.Qwen3OmniMoePreTrainedModelForConditionalGeneration
    r = types.SimpleNamespace(spatial_merge_size=cfg.vision_config.spatial_merge_size, config=cfg)
    for name in ("get_llm_pos_ids_for_vision", "get_chunked_index"):
        setattr(r, name, types.MethodType(getattr(base, name), r))
    ids = torch.tensor([list(input_ids)])
    mask = torch.ones_like(ids)
    pos, delta = base.get_rope_index(
        r, ids, image_grid_thw, video_grid_thw, mask, bool(use_audio_in_video), audio_feature_lengths, video_second_per_grid
    )
    return pos[:, 0, :].float(), float(delta[0, 0])


def manual_mrope_cos_sin(pos, head_dim=128, theta=1.0e6, section=(24, 20, 20)):
    """Interleaved mRoPE (Qwen3-Omni/VL), no HF code. pos [3,T] = (t,h,w) streams.

    inv_freq[j] = theta^(-2j/128), j < 64. Frequency slot j uses stream h when j%3==1 and j < 60, stream w when
    j%3==2 and j < 60, otherwise stream t (so t owns slots 0,3,...,57 and 60..63: 24, h 20, w 20; the [24,20,20] section
    sizes). angle[t,j] = pos[stream(j), t] * inv_freq[j]; cos/sin = cat(cos(angle), cos(angle)) (rotate-half layout)."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    sel = torch.zeros(half, dtype=torch.long)
    for stream, (n, off) in enumerate(zip(section[1:], (1, 2)), start=1):
        sel[off : n * 3 : 3] = stream
    ang = pos[sel, :].T * inv[None, :]  # [T,64]
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)


def placeholder_positions(cfg, input_ids):
    ids = np.asarray(input_ids)
    return {
        "image": np.nonzero(ids == cfg.image_token_id)[0].astype(np.int64),
        "video": np.nonzero(ids == cfg.video_token_id)[0].astype(np.int64),
        "audio": np.nonzero(ids == cfg.audio_token_id)[0].astype(np.int64),
    }


def cmp(a, b):
    """max abs diff and diff relative to the peak of `a` (f32 arrays/tensors)."""
    d = float((a - b).abs().max())
    return {"max_abs_diff": d, "max_rel_to_peak": d / max(float(a.abs().max()), 1e-30)}
