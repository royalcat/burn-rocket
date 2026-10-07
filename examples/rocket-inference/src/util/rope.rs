//! Precomputed RoPE cos/sin tables, shared by the Qwen-family models.

use burn::prelude::*;
use burn::tensor::{DType, TensorData, s};

/// Precomputed RoPE cos/sin tables, `[max_seq, head_dim / 2]`.
pub struct RopeCache {
    cos: Tensor<2>,
    sin: Tensor<2>,
    half: usize,
}

impl RopeCache {
    pub fn new(max_seq: usize, head_dim: usize, theta: f64, dtype: DType, device: &Device) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for pos in 0..max_seq {
            for f in &inv_freq {
                let angle = pos as f32 * *f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = Tensor::<2>::from_data(TensorData::new(cos, [max_seq, half]), device).cast(dtype);
        let sin = Tensor::<2>::from_data(TensorData::new(sin, [max_seq, half]), device).cast(dtype);
        Self { cos, sin, half }
    }

    /// Applies RoPE to `x` of shape `[batch, seq, heads, head_dim]`, starting at position
    /// `seq_start`. Both q and k use the same table (no partial rotation, no freq scaling).
    pub fn apply(&self, x: Tensor<4>, seq_start: usize) -> Tensor<4> {
        let [_, s, _, d] = x.dims();
        let half = self.half;
        let cos = self
            .cos
            .clone()
            .slice(s![seq_start..seq_start + s, ..])
            .reshape([1, s, 1, half])
            .cast(x.dtype());
        let sin = self
            .sin
            .clone()
            .slice(s![seq_start..seq_start + s, ..])
            .reshape([1, s, 1, half])
            .cast(x.dtype());
        let x1 = x.clone().slice(s![.., .., .., 0..half]);
        let x2 = x.slice(s![.., .., .., half..d]);
        let o1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let o2 = x2 * cos + x1 * sin;
        Tensor::cat(vec![o1, o2], 3)
    }
}
