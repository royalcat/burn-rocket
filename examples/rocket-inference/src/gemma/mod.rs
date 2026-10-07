//! Gemma family: the shared Gemma 4 building blocks used by both models here —
//! [`embeddinggemma`] (multimodal embeddings) and [`gemma4`] (E2B-it text
//! generation): config schemas, attention/weight helpers, vision and audio
//! towers, media decoding, input assembly and QAT weight handling.

pub mod audio;
pub mod audio_frontend;
pub mod cli;
pub mod config;
pub mod embeddinggemma;
pub mod gemma4;
pub mod inputs;
pub mod layers;
pub mod media;
pub mod qat;
pub mod vision;

use burn::nn::LinearConfig;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{DType, s};

/// Bias-free linear (all projections in this checkpoint family have no bias).
pub(crate) fn linear_cfg(in_features: usize, out_features: usize) -> LinearConfig {
    LinearConfig::new(in_features, out_features).with_bias(false)
}

/// `x * (mean(x^2) + eps)^-0.5` on the last dim, computed in the input dtype
/// (f32 in practice); used for value norms, which carry no scale.
pub(crate) fn rms_norm_noscale<const D: usize>(x: Tensor<D>, eps: f64) -> Tensor<D> {
    let ms = x.clone().square().mean_dim(D - 1);
    x * (ms + eps).powf_scalar(-0.5)
}

/// GQA expansion: `[B, KV, S, D] -> [B, KV * n, S, D]`.
pub(crate) fn repeat_kv(x: Tensor<4>, n: usize) -> Tensor<4> {
    if n == 1 {
        return x;
    }
    let [b, kv, s, d] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .repeat_dim(2, n)
        .reshape([b, kv * n, s, d])
}

/// Bidirectional attention with an optional symmetric sliding window
/// (`|q - kv| <= window`), chunked over queries to bound the score scratch.
/// `q`, `k`, `v` are `[B, H, S, D]` with `k`/`v` already GQA-repeated; the
/// attention scale is 1.0 (QK-RMSNorm replaces it).
pub(crate) fn chunked_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    window: Option<usize>,
    chunk: usize,
) -> Tensor<4> {
    let [b, h, s, _d] = q.dims();
    let chunk = chunk.max(1);
    let device = q.device();
    let dt = q.dtype();
    let mut outs: Vec<Tensor<4>> = Vec::new();
    let mut q0 = 0;
    while q0 < s {
        let q1 = (q0 + chunk).min(s);
        let (k0, k1) = match window {
            Some(w) => (q0.saturating_sub(w), (q1 + w).min(s)),
            None => (0, s),
        };
        let qc = q.clone().slice(s![.., .., q0..q1, ..]);
        let kc = k.clone().slice(s![.., .., k0..k1, ..]);
        let vc = v.clone().slice(s![.., .., k0..k1, ..]);

        let mut sc = qc.matmul(kc.swap_dims(2, 3));
        if sc.dtype() != DType::F32 {
            sc = sc.cast(DType::F32);
        }
        if let Some(w) = window {
            let rows = Tensor::arange(q0 as i64..q1 as i64, &device).reshape([1, 1, q1 - q0, 1]);
            let cols = Tensor::arange(k0 as i64..k1 as i64, &device).reshape([1, 1, 1, k1 - k0]);
            let w = w as i64;
            let keep = cols
                .clone()
                .lower_equal(rows.clone() + w)
                .bool_and(rows.lower_equal(cols + w));
            let fill = keep.bool_not().expand([b, h, q1 - q0, k1 - k0]);
            sc = sc.mask_fill(fill, f32::NEG_INFINITY);
        }
        let p = softmax(sc, 3);
        let p = if dt == DType::F32 { p } else { p.cast(dt) };
        outs.push(p.matmul(vc));
        q0 = q1;
    }
    Tensor::cat(outs, 2)
}
