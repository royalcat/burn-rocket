//! Clef-Omni thinker text decoder (Qwen3-Omni-30B-A3B MoE) in plain f32 on the CPU,
//! streaming ONE decoder layer at a time from the BF16 safetensors shards.
//!
//! Reference math: `tools/clef_reference.py::manual_layer` (documented there; it equals
//! the transformers layer to ~2e-7 relative). Per layer, pre-norm residual blocks:
//! `h = x + Attn(RMSNorm_in(x))`, `y = h + MoE(RMSNorm_post(h))`.
//!
//! * Attention: 32 q heads / 4 kv heads / head_dim 128, per-head QK RMSNorm before RoPE,
//!   causal softmax, GQA (`q head h` reads `kv head h / 8`).
//! * RoPE: theta 1e6, rotate-half layout. Qwen3-Omni's *interleaved* mRoPE section
//!   `[24, 20, 20]` only chooses, per frequency slot, which of the (time, height, width)
//!   position streams drives the angle (`apply_interleaved_mrope`: slot `j` takes the
//!   height stream if `j % 3 == 1 && j < 60`, the width stream if `j % 3 == 2 && j < 60`,
//!   else the time stream). For text-only input the three streams are the same arange,
//!   so every slot's angle is `t * theta^(-2j/128)`: plain RoPE (verified against the
//!   oracle, which runs the real HF rotary module).
//! * MoE: router softmax over 128 -> top-8 -> renormalize; SwiGLU experts with
//!   intermediate 768; no shared expert.
//!
//! The expert weights come through [`ExpertSource`] (phase 6 swaps the source); this
//! module's implementation widens them from the BF16 shards one expert at a time inside
//! the expert-parallel loop, so the peak is the attention weights plus
//! `threads x 18 MiB`, not a layer's 2.4 GB.

use super::math::{dot, matmul_nt};
use super::media::{MediaFeatures, Positions};
use super::st::{SafeFile, TensorInfo};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

/// Text decoder hyper-parameters (`thinker_config.text_config` of `config.json`).
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub eps: f32,
    pub theta: f64,
    pub norm_topk: bool,
}

impl TextConfig {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&p).with_context(|| format!("read {}", p.display()))?,
        )?;
        let t = &v["thinker_config"]["text_config"];
        let u = |k: &str| -> Result<usize> {
            t[k].as_u64()
                .map(|x| x as usize)
                .with_context(|| format!("text_config.{k} missing"))
        };
        let c = Self {
            hidden: u("hidden_size")?,
            layers: u("num_hidden_layers")?,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            experts: u("num_experts")?,
            top_k: u("num_experts_per_tok")?,
            inter: u("moe_intermediate_size")?,
            eps: t["rms_norm_eps"].as_f64().context("rms_norm_eps")? as f32,
            // transformers 5 moves it under `rope_parameters` (the AutoRound export).
            theta: t["rope_theta"]
                .as_f64()
                .or_else(|| t["rope_parameters"]["rope_theta"].as_f64())
                .context("rope_theta (or rope_parameters.rope_theta)")?,
            norm_topk: t["norm_topk_prob"].as_bool().unwrap_or(true),
        };
        if t["shared_expert_intermediate_size"].as_u64().unwrap_or(0) != 0 {
            bail!("a shared expert is not supported");
        }
        if !t["mlp_only_layers"].as_array().is_none_or(|a| a.is_empty()) {
            bail!("dense (mlp_only) layers are not supported");
        }
        Ok(c)
    }
}

/// One expert's SwiGLU weights, f32.
pub struct ExpertWeights {
    /// `[inter, hidden]`
    pub gate: Vec<f32>,
    /// `[inter, hidden]`
    pub up: Vec<f32>,
    /// `[hidden, inter]`
    pub down: Vec<f32>,
}

/// Supplies a layer's expert weights, one expert at a time (called from the expert
/// parallel loop, hence `Sync`).
pub trait ExpertSource: Sync {
    fn expert(&self, layer: usize, expert: usize) -> Result<ExpertWeights>;
}

/// Index + mmaps of all shards.
pub struct Shards {
    files: Vec<SafeFile>,
    map: BTreeMap<String, usize>,
}

impl Shards {
    pub fn open(dir: &Path) -> Result<Self> {
        let p = dir.join("model.safetensors.index.json");
        let idx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&p).with_context(|| format!("read {}", p.display()))?,
        )?;
        let wm = idx["weight_map"]
            .as_object()
            .context("index has no weight_map")?;
        let mut names: Vec<String> = Vec::new();
        let mut map = BTreeMap::new();
        for (k, v) in wm {
            if !k.starts_with("thinker.model.") && k != "thinker.lm_head.weight" {
                continue;
            }
            let f = v.as_str().context("shard name")?.to_string();
            let i = match names.iter().position(|n| *n == f) {
                Some(i) => i,
                None => {
                    names.push(f);
                    names.len() - 1
                }
            };
            map.insert(k.clone(), i);
        }
        let files = names
            .iter()
            .map(|n| SafeFile::open(&dir.join(n)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { files, map })
    }

    pub fn get(&self, name: &str) -> Result<(&SafeFile, &TensorInfo)> {
        let i = *self
            .map
            .get(name)
            .with_context(|| format!("{name}: not in the shard index"))?;
        let f = &self.files[i];
        let t = f
            .tensors
            .get(name)
            .with_context(|| format!("{name}: missing from its shard"))?;
        Ok((f, t))
    }

    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (f, t) = self.get(name)?;
        if t.shape != shape {
            bail!("{name}: shape {:?}, expected {shape:?}", t.shape);
        }
        Ok(f.to_f32(t))
    }

    /// Drop every shard's resident mapped pages from this process (page cache stays).
    pub fn release(&self) {
        self.files.iter().for_each(SafeFile::release);
    }
}

/// [`ExpertSource`] over the BF16 shards (`...mlp.experts.E.{gate,up,down}_proj.weight`).
pub struct ShardExperts<'a> {
    pub shards: &'a Shards,
    pub cfg: &'a TextConfig,
}

impl ExpertSource for ShardExperts<'_> {
    fn expert(&self, layer: usize, e: usize) -> Result<ExpertWeights> {
        let (h, i) = (self.cfg.hidden, self.cfg.inter);
        let p = format!("thinker.model.layers.{layer}.mlp.experts.{e}.");
        Ok(ExpertWeights {
            gate: self.shards.f32(&format!("{p}gate_proj.weight"), &[i, h])?,
            up: self.shards.f32(&format!("{p}up_proj.weight"), &[i, h])?,
            down: self.shards.f32(&format!("{p}down_proj.weight"), &[h, i])?,
        })
    }
}

/// Everything of a decoder layer except the experts.
#[derive(Clone)]
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// Router `[experts, hidden]`.
    pub router: Vec<f32>,
}

impl LayerWeights {
    pub fn load(shards: &Shards, cfg: &TextConfig, l: usize) -> Result<Self> {
        let (h, d) = (cfg.hidden, cfg.head_dim);
        let p = format!("thinker.model.layers.{l}.");
        let g = |n: &str, shape: &[usize]| shards.f32(&format!("{p}{n}"), shape);
        Ok(Self {
            input_norm: g("input_layernorm.weight", &[h])?,
            post_norm: g("post_attention_layernorm.weight", &[h])?,
            q: g("self_attn.q_proj.weight", &[cfg.heads * d, h])?,
            k: g("self_attn.k_proj.weight", &[cfg.kv_heads * d, h])?,
            v: g("self_attn.v_proj.weight", &[cfg.kv_heads * d, h])?,
            o: g("self_attn.o_proj.weight", &[h, cfg.heads * d])?,
            q_norm: g("self_attn.q_norm.weight", &[d])?,
            k_norm: g("self_attn.k_norm.weight", &[d])?,
            router: g("mlp.gate.weight", &[cfg.experts, h])?,
        })
    }
}

/// The small per-layer tensors an engine-driven layer still needs on the host (the big
/// projections and the experts are the engine's).
#[derive(Clone)]
pub struct LayerSmall {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// Router `[experts, hidden]`.
    pub router: Vec<f32>,
}

impl From<LayerWeights> for LayerSmall {
    fn from(w: LayerWeights) -> Self {
        Self {
            input_norm: w.input_norm,
            post_norm: w.post_norm,
            q_norm: w.q_norm,
            k_norm: w.k_norm,
            router: w.router,
        }
    }
}

/// Where the backbone gets everything but the activations: per-layer dense weights, the
/// experts, the embedding rows and the final norm. Two implementations: the BF16 shards
/// ([`ShardWeights`]) and a Clef pack (`pack::PackedWeights`, int8 + scales dequantized
/// to f32 on the CPU).
pub trait WeightSource: Sync {
    /// Everything of layer `l` except the experts, widened to f32.
    fn layer_weights(&self, l: usize) -> Result<LayerWeights>;
    /// Only the small tensors of layer `l` (norms, router); sources override it to skip
    /// widening the attention projections an engine computes itself.
    fn layer_small(&self, l: usize) -> Result<LayerSmall> {
        Ok(self.layer_weights(l)?.into())
    }
    fn experts(&self) -> &dyn ExpertSource;
    /// Embedding row of token `id` into `out` (`hidden` long).
    fn embed_row(&self, id: u32, out: &mut [f32]) -> Result<()>;
    fn final_norm(&self) -> Result<Vec<f32>>;
    /// Drop this source's resident mapped pages (called after every layer).
    fn release(&self);
}

/// [`WeightSource`] over the BF16 checkpoint shards.
pub struct ShardWeights {
    shards: Shards,
    cfg: TextConfig,
}

impl ShardWeights {
    pub fn open(dir: &Path, cfg: TextConfig) -> Result<Self> {
        Ok(Self {
            shards: Shards::open(dir)?,
            cfg,
        })
    }
}

impl ExpertSource for ShardWeights {
    fn expert(&self, layer: usize, e: usize) -> Result<ExpertWeights> {
        ShardExperts {
            shards: &self.shards,
            cfg: &self.cfg,
        }
        .expert(layer, e)
    }
}

impl WeightSource for ShardWeights {
    fn layer_weights(&self, l: usize) -> Result<LayerWeights> {
        LayerWeights::load(&self.shards, &self.cfg, l)
    }

    fn experts(&self) -> &dyn ExpertSource {
        self
    }

    fn embed_row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        let (f, t) = self.shards.get("thinker.model.embed_tokens.weight")?;
        f.row_f32(t, id as usize, out)
    }

    fn final_norm(&self) -> Result<Vec<f32>> {
        self.shards
            .f32("thinker.model.norm.weight", &[self.cfg.hidden])
    }

    fn release(&self) {
        self.shards.release();
    }
}

/// `cos`/`sin` tables `[positions, head_dim / 2]` (the second half of the head repeats
/// them: rotate-half layout).
pub struct Rope {
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub fn new(cfg: &TextConfig, positions: usize) -> Self {
        let half = cfg.head_dim / 2;
        // torch: inv_freq = 1 / theta ** (arange(0, D, 2) / D) in f32, angle in f32.
        let inv: Vec<f32> = (0..half)
            .map(|j| (1.0 / cfg.theta.powf((2 * j) as f64 / cfg.head_dim as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(positions * half);
        let mut sin = Vec::with_capacity(positions * half);
        for t in 0..positions {
            for &f in &inv {
                let a = t as f32 * f;
                cos.push((a as f64).cos() as f32);
                sin.push((a as f64).sin() as f32);
            }
        }
        Self { half, cos, sin }
    }

    /// Tables for explicit 3-D mRoPE positions (one row per token): frequency slot `j`
    /// reads the stream [`Positions::stream_of_slot`] picks. With three equal streams
    /// (text) this is bit-identical to [`Rope::new`].
    pub fn from_positions(cfg: &TextConfig, pos: &Positions) -> Self {
        let half = cfg.head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|j| (1.0 / cfg.theta.powf((2 * j) as f64 / cfg.head_dim as f64)) as f32)
            .collect();
        let s = pos.len();
        let mut cos = Vec::with_capacity(s * half);
        let mut sin = Vec::with_capacity(s * half);
        for t in 0..s {
            for (j, &f) in inv.iter().enumerate() {
                let a = pos.stream(Positions::stream_of_slot(j))[t] * f;
                cos.push((a as f64).cos() as f32);
                sin.push((a as f64).sin() as f32);
            }
        }
        Self { half, cos, sin }
    }

    /// Rotate `x` (`[s, heads, head_dim]`) in place.
    pub(super) fn apply(&self, x: &mut [f32], s: usize, heads: usize) {
        let (h, d) = (self.half, self.half * 2);
        x.par_chunks_mut(heads * d)
            .take(s)
            .enumerate()
            .for_each(|(t, row)| {
                let (c, sn) = (&self.cos[t * h..(t + 1) * h], &self.sin[t * h..(t + 1) * h]);
                for head in row.chunks_mut(d) {
                    for j in 0..h {
                        let (a, b) = (head[j], head[j + h]);
                        head[j] = a * c[j] - b * sn[j];
                        head[j + h] = b * c[j] + a * sn[j];
                    }
                }
            });
    }
}

/// The rotary tables of the records of a (batched) pass: one table shared by all records
/// (plain text positions, covering the longest record) or one per record (mRoPE).
pub trait RopeSource {
    fn rope(&self, record: usize) -> &Rope;
}

impl RopeSource for Rope {
    fn rope(&self, _record: usize) -> &Rope {
        self
    }
}

impl RopeSource for Vec<Rope> {
    fn rope(&self, record: usize) -> &Rope {
        &self[record]
    }
}

/// `w * x * rsqrt(mean(x^2) + eps)` over each `n`-wide row, in place.
pub(super) fn rms_norm_rows(x: &mut [f32], n: usize, w: &[f32], eps: f32) {
    x.par_chunks_mut(n).for_each(|r| {
        let ms = r.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
        let inv = 1.0 / (ms as f32 + eps).sqrt();
        for (v, w) in r.iter_mut().zip(w) {
            *v = w * (*v * inv);
        }
    });
}

pub(super) fn rms_norm(x: &[f32], n: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    rms_norm_rows(&mut y, n, w, eps);
    y
}

/// Causal GQA attention. `q [s, heads*d]`, `k`/`v [s, kv*d]` -> context `[s, heads*d]`.
fn attention(cfg: &TextConfig, q: &[f32], k: &[f32], v: &[f32], s: usize) -> Vec<f32> {
    let (nh, nkv, d) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let rep = nh / nkv;
    let scale = 1.0 / (d as f32).sqrt();
    let mut ctx = vec![0f32; s * nh * d];
    ctx.par_chunks_mut(nh * d).enumerate().for_each(|(i, row)| {
        let mut sc = vec![0f32; i + 1];
        for h in 0..nh {
            let g = h / rep;
            let qi = &q[(i * nh + h) * d..(i * nh + h + 1) * d];
            let mut max = f32::NEG_INFINITY;
            for (j, sj) in sc.iter_mut().enumerate() {
                *sj = dot(qi, &k[(j * nkv + g) * d..(j * nkv + g + 1) * d]) * scale;
                max = max.max(*sj);
            }
            let mut sum = 0f32;
            for sj in sc.iter_mut() {
                *sj = (*sj - max).exp();
                sum += *sj;
            }
            let out = &mut row[h * d..(h + 1) * d];
            for (j, &p) in sc.iter().enumerate() {
                let p = p / sum;
                let vj = &v[(j * nkv + g) * d..(j * nkv + g + 1) * d];
                for (o, x) in out.iter_mut().zip(vj) {
                    *o += p * x;
                }
            }
        }
    });
    ctx
}

/// Router: softmax over the experts, top-k (descending, ties to the lower id as
/// `torch.topk`), optional renormalization. Returns ids `[s*k]` and weights `[s*k]`.
/// The third value is, per token, `p[k-th] - p[(k+1)-th]` of the raw softmax (the
/// near-tie margin that decides whether the selected set is stable).
pub(super) fn route(cfg: &TextConfig, logits: &[f32], s: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let (e, k) = (cfg.experts, cfg.top_k);
    let mut ids = vec![0u32; s * k];
    let mut wts = vec![0f32; s * k];
    let mut margins = vec![0f32; s];
    ids.par_chunks_mut(k)
        .zip(wts.par_chunks_mut(k))
        .zip(margins.par_iter_mut())
        .zip(logits.par_chunks(e))
        .for_each(|(((id, w), mg), l)| {
            let mut p = l.to_vec();
            super::math::softmax_row(&mut p);
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
            let sel = &order[..k];
            *mg = if k < e {
                p[order[k - 1]] - p[order[k]]
            } else {
                1.0
            };
            let sum: f32 = sel.iter().map(|&i| p[i]).sum();
            for (slot, &i) in sel.iter().enumerate() {
                id[slot] = i as u32;
                w[slot] = if cfg.norm_topk { p[i] / sum } else { p[i] };
            }
        });
    (ids, wts, margins)
}

pub(super) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// One layer's output and routing.
pub struct LayerOut {
    pub hidden: Vec<f32>,
    pub router_ids: Vec<u32>,
    pub router_weights: Vec<f32>,
    /// `[s]` raw-probability gap between the last selected and the first rejected expert.
    pub router_margins: Vec<f32>,
}

/// One decoder layer on `x [s, hidden]` (f32, causal, positions `0..s`).
#[allow(clippy::too_many_arguments)]
pub fn layer_forward(
    cfg: &TextConfig,
    w: &LayerWeights,
    src: &dyn ExpertSource,
    layer: usize,
    x: &[f32],
    s: usize,
    rope: &Rope,
    threads: usize,
) -> Result<LayerOut> {
    layer_forward_multi(cfg, w, src, layer, x, &[s], rope, threads)
}

/// One decoder layer over several records at once: `x` holds the records' rows
/// back to back (`lens[i]` rows each). Attention (projections, RoPE with positions
/// restarting per record, the causal scores) and the router run per record, so no
/// record sees another; the MoE buckets the rows of ALL records per expert, so each
/// expert's weights are widened once per layer. `rope` must cover the longest record.
/// With `lens == [s]` this is exactly [`layer_forward`].
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_multi(
    cfg: &TextConfig,
    w: &LayerWeights,
    src: &dyn ExpertSource,
    layer: usize,
    x: &[f32],
    lens: &[usize],
    rope: &dyn RopeSource,
    threads: usize,
) -> Result<LayerOut> {
    let (h, nh, nkv, d) = (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim);
    let s: usize = lens.iter().sum();
    assert_eq!(x.len(), s * h);
    let mut hid = vec![0f32; s * h];
    let mut n2 = vec![0f32; s * h];
    let (mut ids, mut wts, mut margins) = (Vec::new(), Vec::new(), Vec::new());
    let mut off = 0;
    for (ri, &sr) in lens.iter().enumerate() {
        let rope = rope.rope(ri);
        let xr = &x[off * h..(off + sr) * h];
        // ---- attention
        let n = rms_norm(xr, h, &w.input_norm, cfg.eps);
        let mut q = matmul_nt(&n, sr, h, &w.q, nh * d, threads);
        let mut k = matmul_nt(&n, sr, h, &w.k, nkv * d, threads);
        let v = matmul_nt(&n, sr, h, &w.v, nkv * d, threads);
        rms_norm_rows(&mut q, d, &w.q_norm, cfg.eps);
        rms_norm_rows(&mut k, d, &w.k_norm, cfg.eps);
        rope.apply(&mut q, sr, nh);
        rope.apply(&mut k, sr, nkv);
        let ctx = attention(cfg, &q, &k, &v, sr);
        let proj = matmul_nt(&ctx, sr, nh * d, &w.o, h, threads);
        for ((o, a), b) in hid[off * h..(off + sr) * h].iter_mut().zip(xr).zip(&proj) {
            *o = a + b;
        }
        // ---- router (per record)
        let n2r = rms_norm(&hid[off * h..(off + sr) * h], h, &w.post_norm, cfg.eps);
        let logits = matmul_nt(&n2r, sr, h, &w.router, cfg.experts, threads);
        let (i, wt, mg) = route(cfg, &logits, sr);
        ids.extend(i);
        wts.extend(wt);
        margins.extend(mg);
        n2[off * h..(off + sr) * h].copy_from_slice(&n2r);
        off += sr;
    }

    // ---- MoE over every row
    let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); cfg.experts];
    for t in 0..s {
        for slot in 0..cfg.top_k {
            buckets[ids[t * cfg.top_k + slot] as usize].push((t, wts[t * cfg.top_k + slot]));
        }
    }
    let active: Vec<usize> = (0..cfg.experts)
        .filter(|&e| !buckets[e].is_empty())
        .collect();
    let i = cfg.inter;
    // Expert-parallel: each task widens its expert, runs the three single-threaded
    // GEMMs on its own token bucket, and returns the router-scaled outputs.
    let outs: Vec<Vec<f32>> = active
        .par_iter()
        .map(|&e| -> Result<Vec<f32>> {
            let ew = src.expert(layer, e)?;
            let b = &buckets[e];
            let m = b.len();
            let mut xe = Vec::with_capacity(m * h);
            for &(t, _) in b {
                xe.extend_from_slice(&n2[t * h..(t + 1) * h]);
            }
            let mut g = matmul_nt(&xe, m, h, &ew.gate, i, 1);
            let u = matmul_nt(&xe, m, h, &ew.up, i, 1);
            for (a, b) in g.iter_mut().zip(&u) {
                *a = silu(*a) * b;
            }
            let mut y = matmul_nt(&g, m, i, &ew.down, h, 1);
            for (row, &(_, wt)) in y.chunks_mut(h).zip(b) {
                row.iter_mut().for_each(|v| *v *= wt);
            }
            Ok(y)
        })
        .collect::<Result<_>>()?;
    let mut y = hid;
    // Expert-id order (as torch's index_add_ loop) for a deterministic sum.
    for (&e, out) in active.iter().zip(&outs) {
        for (&(t, _), row) in buckets[e].iter().zip(out.chunks(h)) {
            for (a, b) in y[t * h..(t + 1) * h].iter_mut().zip(row) {
                *a += b;
            }
        }
    }
    Ok(LayerOut {
        hidden: y,
        router_ids: ids,
        router_weights: wts,
        router_margins: margins,
    })
}

#[derive(Debug, Clone)]
pub struct ForwardOpts {
    /// Worker threads of the private rayon pool (the caller's pool is not touched).
    pub threads: usize,
    /// Drop the shards' mapped pages after every layer (keeps RSS low).
    pub release_pages: bool,
}

impl Default for ForwardOpts {
    fn default() -> Self {
        Self {
            threads: 8,
            release_pages: true,
        }
    }
}

pub struct BackboneOut {
    /// `[s, hidden]`: the final RMSNorm output (the head input).
    pub final_normed: Vec<f32>,
    /// `[s, hidden]`: the last layer's output (before the final norm).
    pub final_hidden: Vec<f32>,
}

/// What a per-layer tap sees after layer `layer` ran.
pub struct Tap<'a> {
    pub layer: usize,
    /// `[s*H]` layer output.
    pub hidden: &'a [f32],
    /// `[s*k]` top-k expert ids per token (descending router probability).
    pub router_ids: &'a [u32],
    /// `[s*k]` renormalized top-k weights.
    pub router_weights: &'a [f32],
    /// `[s]` gap between the k-th and (k+1)-th raw router probability.
    pub router_margins: &'a [f32],
}

pub type LayerTap<'a> = dyn FnMut(&Tap<'_>) + Send + 'a;

pub struct Backbone {
    pub cfg: TextConfig,
    weights: Box<dyn WeightSource + Send>,
    /// When set, attention projections and the MoE run through it (the NPU / emulated
    /// placements); `None` is the f32 CPU path.
    engine: Option<Box<dyn super::engine::LayerEngine>>,
}

impl Backbone {
    /// Open the BF16 checkpoint directory.
    pub fn open(dir: &Path) -> Result<Self> {
        let cfg = TextConfig::from_model_dir(dir)?;
        let weights = Box::new(ShardWeights::open(dir, cfg.clone())?);
        Ok(Self {
            cfg,
            weights,
            engine: None,
        })
    }

    /// Open a Clef pack directory (`clef-pack` output); `config.json` is read from it.
    pub fn open_pack(dir: &Path) -> Result<Self> {
        let cfg = TextConfig::from_model_dir(dir)?;
        let weights = Box::new(super::pack::PackedWeights::open(dir, cfg.clone())?);
        Ok(Self {
            cfg,
            weights,
            engine: None,
        })
    }

    /// Open the AutoRound checkpoint (`moeshawky/clef-omni-int8-autoround`) as stored: no
    /// repack, experts read from the shards on demand.
    pub fn open_autoround(dir: &Path) -> Result<Self> {
        let cfg = TextConfig::from_model_dir(dir)?;
        let weights = Box::new(super::autoround_store::AutoRoundWeights::open(
            dir,
            cfg.clone(),
        )?);
        Ok(Self {
            cfg,
            weights,
            engine: None,
        })
    }

    /// Any weight source (tests, future placements).
    pub fn with_source(cfg: TextConfig, weights: Box<dyn WeightSource + Send>) -> Self {
        Self {
            cfg,
            weights,
            engine: None,
        }
    }

    /// Route the attention projections and the MoE through `engine` (see
    /// [`super::engine`]).
    pub fn set_engine(&mut self, engine: Box<dyn super::engine::LayerEngine>) {
        self.engine = Some(engine);
    }

    pub fn last_phases(&self) -> Option<super::engine::Phases> {
        self.engine.as_ref().and_then(|e| e.last_phases())
    }

    pub fn engine_label(&self) -> Option<String> {
        self.engine.as_ref().map(|e| e.label())
    }

    pub fn forward(&self, ids: &[u32], opts: &ForwardOpts) -> Result<BackboneOut> {
        self.forward_tapped(ids, opts, &mut |_| {})
    }

    pub fn forward_tapped(
        &self,
        ids: &[u32],
        opts: &ForwardOpts,
        on_layer: &mut LayerTap<'_>,
    ) -> Result<BackboneOut> {
        self.forward_media_tapped(ids, None, opts, on_layer)
    }

    /// [`Backbone::forward`] for a record with media: `media` supplies the mRoPE
    /// positions, the rows spliced over the placeholder embeddings and the DeepStack rows.
    pub fn forward_media(
        &self,
        ids: &[u32],
        media: Option<&MediaFeatures>,
        opts: &ForwardOpts,
    ) -> Result<BackboneOut> {
        self.forward_media_tapped(ids, media, opts, &mut |_| {})
    }

    pub fn forward_media_tapped(
        &self,
        ids: &[u32],
        media: Option<&MediaFeatures>,
        opts: &ForwardOpts,
        on_layer: &mut LayerTap<'_>,
    ) -> Result<BackboneOut> {
        if ids.is_empty() {
            bail!("empty input");
        }
        if let Some(m) = media {
            m.validate(ids.len(), self.cfg.hidden)?;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.threads.max(1))
            .build()?;
        pool.install(|| self.run(ids, media, opts, on_layer))
    }

    /// The input of layer 0 for one record: the token rows with `media`'s features
    /// spliced over the placeholder positions.
    pub fn embed_input(&self, ids: &[u32], media: Option<&MediaFeatures>) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let mut x = vec![0f32; ids.len() * h];
        for (r, &id) in x.chunks_mut(h).zip(ids) {
            self.weights.embed_row(id, r)?;
        }
        if let Some(m) = media {
            m.splice(&mut x, h);
        }
        Ok(x)
    }

    fn run(
        &self,
        ids: &[u32],
        media: Option<&MediaFeatures>,
        opts: &ForwardOpts,
        on_layer: &mut LayerTap<'_>,
    ) -> Result<BackboneOut> {
        let (cfg, s, h) = (&self.cfg, ids.len(), self.cfg.hidden);
        let mut x = self.embed_input(ids, media)?;
        let rope = match media.and_then(|m| m.positions.as_ref()) {
            Some(p) => Rope::from_positions(cfg, p),
            None => Rope::new(cfg, s),
        };
        for l in 0..cfg.layers {
            let out = if let Some(eng) = &self.engine {
                let w = self.weights.layer_small(l)?;
                let r = super::engine::engine_layer_forward(
                    cfg,
                    &w,
                    eng.as_ref(),
                    l,
                    &x,
                    s,
                    &rope,
                    opts.threads,
                );
                match r {
                    Ok(o) => o,
                    Err(e) => {
                        eng.abort_request();
                        return Err(e);
                    }
                }
            } else {
                let w = self.weights.layer_weights(l)?;
                layer_forward(
                    cfg,
                    &w,
                    self.weights.experts(),
                    l,
                    &x,
                    s,
                    &rope,
                    opts.threads,
                )?
            };
            x = out.hidden;
            if let Some(m) = media {
                m.inject_deepstack(l, &mut x, h);
            }
            on_layer(&Tap {
                layer: l,
                hidden: &x,
                router_ids: &out.router_ids,
                router_weights: &out.router_weights,
                router_margins: &out.router_margins,
            });
            if opts.release_pages {
                self.weights.release();
            }
        }
        if let Some(eng) = &self.engine {
            eng.finish_request(s);
        }
        let nw = self.weights.final_norm()?;
        let normed = rms_norm(&x, h, &nw, cfg.eps);
        Ok(BackboneOut {
            final_normed: normed,
            final_hidden: x,
        })
    }

    /// Run several records through the thinker as ONE pass per chunk: per-record
    /// attention, one MoE (one expert stream) over the rows of every record of the
    /// chunk. Records are chunked in order so a chunk holds at most `max_batch_tokens`
    /// rows (a record over the cap runs alone; `0` = no cap). Every record's output is
    /// bit-identical to [`Backbone::forward`] on it alone for the engines whose row math
    /// is independent of the other rows (see `docs/recipes/clef.md`, "Batching").
    ///
    /// An empty record fails alone; an error in a chunk's shared pass fails that chunk's
    /// members with the same message. A panic (an NPU failure) unwinds out of the call.
    pub fn forward_batch(
        &self,
        records: &[&[u32]],
        opts: &ForwardOpts,
        max_batch_tokens: usize,
    ) -> Result<BatchForward> {
        let recs: Vec<RecordRef<'_>> = records
            .iter()
            .map(|&ids| RecordRef { ids, media: None })
            .collect();
        self.forward_batch_media(&recs, opts, max_batch_tokens)
    }

    /// [`Backbone::forward_batch`] with per-record media features (each record keeps its
    /// own position stream, splice and DeepStack rows; a text-only record next to media
    /// records keeps plain positions).
    pub fn forward_batch_media(
        &self,
        records: &[RecordRef<'_>],
        opts: &ForwardOpts,
        max_batch_tokens: usize,
    ) -> Result<BatchForward> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.threads.max(1))
            .build()?;
        let mut outs: Vec<Option<Result<BackboneOut>>> = (0..records.len()).map(|_| None).collect();
        let live: Vec<usize> = (0..records.len())
            .filter(|&i| {
                let bad = if records[i].ids.is_empty() {
                    Some(anyhow::anyhow!("empty input"))
                } else {
                    records[i]
                        .media
                        .and_then(|m| m.validate(records[i].ids.len(), self.cfg.hidden).err())
                };
                if let Some(e) = bad {
                    outs[i] = Some(Err(e));
                }
                outs[i].is_none()
            })
            .collect();
        let lens: Vec<usize> = live.iter().map(|&i| records[i].ids.len()).collect();
        let mut chunks = Vec::new();
        for range in plan_batches(&lens, max_batch_tokens, usize::MAX) {
            let members: Vec<RecordRef<'_>> =
                live[range.clone()].iter().map(|&i| records[i]).collect();
            let tokens: usize = members.iter().map(|m| m.ids.len()).sum();
            let t0 = std::time::Instant::now();
            let res = pool.install(|| self.run_batch(&members, opts));
            let seconds = t0.elapsed().as_secs_f64();
            let phases = if res.is_ok() {
                self.last_phases()
            } else {
                None
            };
            chunks.push(BatchChunk {
                records: members.len(),
                tokens,
                seconds,
                phases,
            });
            match res {
                Ok(v) => {
                    for (&i, o) in live[range].iter().zip(v) {
                        outs[i] = Some(Ok(o));
                    }
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    for &i in &live[range] {
                        outs[i] = Some(Err(anyhow::anyhow!("{msg}")));
                    }
                }
            }
        }
        Ok(BatchForward {
            outs: outs
                .into_iter()
                .map(|o| o.expect("every record ran"))
                .collect(),
            chunks,
        })
    }

    /// One shared pass over `records` (all non-empty, features validated).
    fn run_batch(&self, records: &[RecordRef<'_>], opts: &ForwardOpts) -> Result<Vec<BackboneOut>> {
        let (cfg, h) = (&self.cfg, self.cfg.hidden);
        let lens: Vec<usize> = records.iter().map(|r| r.ids.len()).collect();
        let total: usize = lens.iter().sum();
        let mut x = Vec::with_capacity(total * h);
        for r in records {
            x.extend(self.embed_input(r.ids, r.media)?);
        }
        // Plain text everywhere: one shared table (unchanged); else one table per record.
        let any_positions = records
            .iter()
            .any(|r| r.media.is_some_and(|m| m.positions.is_some()));
        let ropes: Box<dyn RopeSource> = if any_positions {
            Box::new(
                records
                    .iter()
                    .map(|r| match r.media.and_then(|m| m.positions.as_ref()) {
                        Some(p) => Rope::from_positions(cfg, p),
                        None => Rope::new(cfg, r.ids.len()),
                    })
                    .collect::<Vec<Rope>>(),
            )
        } else {
            Box::new(Rope::new(cfg, lens.iter().copied().max().unwrap_or(0)))
        };
        let any_media = records.iter().any(|r| r.media.is_some());
        for l in 0..cfg.layers {
            let out = if let Some(eng) = &self.engine {
                let w = self.weights.layer_small(l)?;
                let r = super::engine::engine_layer_forward_multi(
                    cfg,
                    &w,
                    eng.as_ref(),
                    l,
                    &x,
                    &lens,
                    ropes.as_ref(),
                    opts.threads,
                );
                match r {
                    Ok(o) => o,
                    Err(e) => {
                        eng.abort_request();
                        return Err(e);
                    }
                }
            } else {
                let w = self.weights.layer_weights(l)?;
                layer_forward_multi(
                    cfg,
                    &w,
                    self.weights.experts(),
                    l,
                    &x,
                    &lens,
                    ropes.as_ref(),
                    opts.threads,
                )?
            };
            x = out.hidden;
            if any_media && l < super::media::DEEPSTACK_LAYERS {
                let mut off = 0;
                for r in records {
                    let s = r.ids.len();
                    if let Some(m) = r.media {
                        m.inject_deepstack(l, &mut x[off * h..(off + s) * h], h);
                    }
                    off += s;
                }
            }
            if opts.release_pages {
                self.weights.release();
            }
        }
        if let Some(eng) = &self.engine {
            eng.finish_request(total);
        }
        let nw = self.weights.final_norm()?;
        let normed = rms_norm(&x, h, &nw, cfg.eps);
        let mut outs = Vec::with_capacity(records.len());
        let mut off = 0;
        for &s in &lens {
            outs.push(BackboneOut {
                final_normed: normed[off * h..(off + s) * h].to_vec(),
                final_hidden: x[off * h..(off + s) * h].to_vec(),
            });
            off += s;
        }
        Ok(outs)
    }
}

/// One record of a batched pass: its token ids and optional media features.
#[derive(Clone, Copy)]
pub struct RecordRef<'a> {
    pub ids: &'a [u32],
    pub media: Option<&'a MediaFeatures>,
}

/// One shared pass of [`Backbone::forward_batch`].
#[derive(Debug, Clone)]
pub struct BatchChunk {
    pub records: usize,
    /// Rows (tokens) of the pass.
    pub tokens: usize,
    pub seconds: f64,
    /// The engine's phase totals of the pass (`None` on the plain CPU path); `experts`
    /// counts the expert records read and run over all layers.
    pub phases: Option<super::engine::Phases>,
}

/// The per-record results (input order) and the shared passes of a batch.
pub struct BatchForward {
    pub outs: Vec<Result<BackboneOut>>,
    pub chunks: Vec<BatchChunk>,
}

/// Greedy, order-preserving grouping of records (by their `lens`) into batches of at
/// most `max_records` records and `max_tokens` total rows (`0` = no row cap). A record
/// that alone exceeds the row cap forms a batch of its own. Returns index ranges.
pub fn plan_batches(
    lens: &[usize],
    max_tokens: usize,
    max_records: usize,
) -> Vec<std::ops::Range<usize>> {
    let cap = if max_tokens == 0 {
        usize::MAX
    } else {
        max_tokens
    };
    let max_records = max_records.max(1);
    let mut out = Vec::new();
    let (mut start, mut rows) = (0usize, 0usize);
    for (i, &l) in lens.iter().enumerate() {
        if i > start && (rows + l > cap || i - start >= max_records) {
            out.push(start..i);
            start = i;
            rows = 0;
        }
        rows += l;
    }
    if start < lens.len() {
        out.push(start..lens.len());
    }
    out
}

#[cfg(test)]
mod tests;
