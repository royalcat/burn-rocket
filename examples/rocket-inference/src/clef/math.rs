//! Plain f32 kernels with PyTorch semantics (row-major, `[rows, features]`).

use rayon::prelude::*;

unsafe extern "C" {
    /// libm single-precision `erf` (what `torch.nn.GELU()` evaluates on the CPU, up to
    /// vectorized-implementation rounding).
    fn erff(x: f32) -> f32;
}

/// Exact GELU: `0.5 x (1 + erf(x / sqrt 2))`.
pub fn gelu(x: f32) -> f32 {
    // SAFETY: the C `erff` is pure, takes one f32 and returns one f32.
    let e = unsafe { erff(x * std::f32::consts::FRAC_1_SQRT_2) };
    0.5 * x * (1.0 + e)
}

pub fn gelu_in_place(x: &mut [f32]) {
    x.par_chunks_mut(4096)
        .for_each(|c| c.iter_mut().for_each(|v| *v = gelu(*v)));
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

/// `y[m, o] = x[m, k] . w[o, k]^T (+ b[o])` (`torch.nn.functional.linear`).
pub fn linear(x: &[f32], m: usize, k: usize, w: &[f32], o: usize, b: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), m * k, "linear: x shape");
    assert_eq!(w.len(), o * k, "linear: w shape");
    if let Some(b) = b {
        assert_eq!(b.len(), o, "linear: bias shape");
    }
    // [o][m] first so each task owns one weight row, then transpose.
    let mut t = vec![0f32; o * m];
    t.par_chunks_mut(m).enumerate().for_each(|(oi, col)| {
        let wr = &w[oi * k..(oi + 1) * k];
        for (mi, c) in col.iter_mut().enumerate() {
            *c = dot(&x[mi * k..(mi + 1) * k], wr);
        }
    });
    let mut y = vec![0f32; m * o];
    for oi in 0..o {
        let bias = b.map_or(0.0, |b| b[oi]);
        for mi in 0..m {
            y[mi * o + oi] = t[oi * m + mi] + bias;
        }
    }
    y
}

/// `y[m, o] = x[m, k] . w[o, k]^T` through the `gemm` crate (no bias). `threads <= 1` runs
/// on the calling thread; otherwise the call fans out over the current rayon pool.
pub fn matmul_nt(x: &[f32], m: usize, k: usize, w: &[f32], o: usize, threads: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k, "matmul_nt: x shape");
    assert_eq!(w.len(), o * k, "matmul_nt: w shape");
    let mut y = vec![0f32; m * o];
    if m == 0 || o == 0 {
        return y;
    }
    let par = if threads <= 1 {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(threads)
    };
    // SAFETY: the slices above are exactly m*k, o*k and m*o elements; strides describe
    // row-major x, w^T (rs 1, cs k) and row-major y.
    unsafe {
        gemm::gemm(
            m,
            o,
            k,
            y.as_mut_ptr(),
            1,
            o as isize,
            false,
            x.as_ptr(),
            1,
            k as isize,
            w.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            par,
        );
    }
    y
}

/// `torch.nn.LayerNorm` over the last dim: biased variance, affine.
pub fn layer_norm(x: &[f32], rows: usize, n: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * n);
    assert_eq!((w.len(), b.len()), (n, n));
    let mut y = vec![0f32; x.len()];
    y.par_chunks_mut(n)
        .zip(x.par_chunks(n))
        .for_each(|(yr, xr)| {
            let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let var = xr
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            let inv = 1.0 / (var as f32 + eps).sqrt();
            let mean = mean as f32;
            for i in 0..n {
                yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
            }
        });
    y
}

/// In-place softmax of one row.
pub fn softmax_row(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Softmax of a slice of logits (the SystemOne probabilities).
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let mut v = x.to_vec();
    softmax_row(&mut v);
    v
}

/// `torch.nn.functional.normalize(dim=-1)`: `x / max(||x||, 1e-12)`.
pub fn normalize(x: &[f32]) -> Vec<f32> {
    let n = dot(x, x).sqrt().max(1e-12);
    x.iter().map(|v| v / n).collect()
}

/// `torch.nn.functional.cosine_similarity` (eps 1e-8, each side clamped separately).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = dot(a, a).sqrt().max(1e-8);
    let nb = dot(b, b).sqrt().max(1e-8);
    let mut s = 0f32;
    for (x, y) in a.iter().zip(b) {
        s += (x / na) * (y / nb);
    }
    s
}

/// `nn.MultiheadAttention` weights (batch_first, packed projection).
pub struct Mha {
    pub in_w: Vec<f32>, // [3d, d]
    pub in_b: Vec<f32>, // [3d]
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    pub heads: usize,
}

impl Mha {
    /// `attn(q_in [nq,d], kv_in [nk,d])` -> `[nq,d]`, no masks, no dropout.
    pub fn forward(&self, q_in: &[f32], nq: usize, kv_in: &[f32], nk: usize, d: usize) -> Vec<f32> {
        let hd = d / self.heads;
        let q = linear(q_in, nq, d, &self.in_w[..d * d], d, Some(&self.in_b[..d]));
        let k = linear(
            kv_in,
            nk,
            d,
            &self.in_w[d * d..2 * d * d],
            d,
            Some(&self.in_b[d..2 * d]),
        );
        let v = linear(
            kv_in,
            nk,
            d,
            &self.in_w[2 * d * d..],
            d,
            Some(&self.in_b[2 * d..]),
        );
        let scale = 1.0 / (hd as f32).sqrt();
        // One task per head: ctx_h [nq, hd].
        let per_head: Vec<Vec<f32>> = (0..self.heads)
            .into_par_iter()
            .map(|h| {
                let mut ctx = vec![0f32; nq * hd];
                let mut s = vec![0f32; nk];
                for i in 0..nq {
                    let qi = &q[i * d + h * hd..i * d + (h + 1) * hd];
                    for (j, sj) in s.iter_mut().enumerate() {
                        *sj = dot(qi, &k[j * d + h * hd..j * d + (h + 1) * hd]) * scale;
                    }
                    softmax_row(&mut s);
                    let out = &mut ctx[i * hd..(i + 1) * hd];
                    for (j, &p) in s.iter().enumerate() {
                        let vj = &v[j * d + h * hd..j * d + (h + 1) * hd];
                        for (o, vv) in out.iter_mut().zip(vj) {
                            *o += p * vv;
                        }
                    }
                }
                ctx
            })
            .collect();
        let mut merged = vec![0f32; nq * d];
        for (h, ctx) in per_head.iter().enumerate() {
            for i in 0..nq {
                merged[i * d + h * hd..i * d + (h + 1) * hd]
                    .copy_from_slice(&ctx[i * hd..(i + 1) * hd]);
            }
        }
        linear(&merged, nq, d, &self.out_w, d, Some(&self.out_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// torch 2.15 `nn.MultiheadAttention(8, 2, batch_first=True)` output for the
    /// deterministic parameters/inputs below (f64 sin/cos, cast to f32).
    const MHA_WANT: [f32; 16] = [
        1.019_492_4,
        -0.635_100_5,
        1.563_538,
        -0.699_721_5,
        1.725_238_7,
        -0.868_995,
        1.474_872_6,
        -1.077_636_6,
        0.971_358_4,
        -0.606_101_4,
        1.554_627_5,
        -0.711_192_2,
        1.756_714,
        -0.919_439_5,
        1.542_627_3,
        -1.160_473_5,
    ];

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn gelu_matches_exact_erf_values() {
        // torch.nn.functional.gelu (exact) reference values.
        let cases = [
            (0.0f32, 0.0f32),
            (1.0, 0.841_344_8),
            (-1.0, -0.158_655_26),
            (2.0, 1.954_5),
            (-3.0, -0.004_049_867),
            (0.5, 0.345_731_2),
        ];
        for (x, y) in cases {
            assert!(close(gelu(x), y, 2e-6), "gelu({x}) = {} want {y}", gelu(x));
        }
    }

    #[test]
    fn layer_norm_matches_hand_values() {
        // x = [1,2,3,4]: mean 2.5, biased var 1.25, eps 1e-5.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 2.0, 1.0, 0.5];
        let b = [0.0f32, 0.1, 0.0, -1.0];
        let inv = 1.0 / (1.25f32 + 1e-5).sqrt();
        let y = layer_norm(&x, 1, 4, &w, &b, 1e-5);
        let want = [
            -1.5 * inv,
            -0.5 * inv * 2.0 + 0.1,
            0.5 * inv,
            1.5 * inv * 0.5 - 1.0,
        ];
        for (a, b) in y.iter().zip(want) {
            assert!(close(*a, b, 1e-6), "{a} vs {b}");
        }
        // Two rows are independent.
        let y2 = layer_norm(
            &[1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0],
            2,
            4,
            &w,
            &b,
            1e-5,
        );
        assert!(close(y2[0], y[0], 1e-7));
        assert!(y2[4..].iter().zip(&b).all(|(a, b)| close(*a, *b, 1e-6)));
    }

    #[test]
    fn linear_is_x_wt_plus_b() {
        // x [2,3], w [2,3]
        let x = [1.0f32, 2.0, 3.0, 0.0, -1.0, 1.0];
        let w = [1.0f32, 0.0, -1.0, 0.5, 0.5, 0.5];
        let y = linear(&x, 2, 3, &w, 2, Some(&[10.0, -10.0]));
        assert_eq!(y, vec![8.0, -7.0, 9.0, -10.0]);
    }

    #[test]
    fn softmax_cosine_normalize() {
        let p = softmax(&[0.0, 0.0, 0.0, 0.0]);
        assert!(p.iter().all(|v| close(*v, 0.25, 1e-7)));
        let p = softmax(&[1000.0, 1000.0]);
        assert!(close(p[0], 0.5, 1e-7));
        let n = normalize(&[3.0, 4.0]);
        assert!(close(n[0], 0.6, 1e-7) && close(n[1], 0.8, 1e-7));
        assert_eq!(normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
        assert!(close(
            cosine(&[1.0, 0.0], &[1.0, 1.0]),
            std::f32::consts::FRAC_1_SQRT_2,
            1e-6
        ));
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    fn tiny_mha() -> Mha {
        let val = |i: usize, salt: f64| ((i as f64 * 0.37 + salt).sin() * 0.5) as f32;
        Mha {
            in_w: (0..3 * 8 * 8).map(|i| val(i, 0.1)).collect(),
            in_b: (0..24).map(|i| val(i, 0.2)).collect(),
            out_w: (0..64).map(|i| val(i, 0.3)).collect(),
            out_b: (0..8).map(|i| val(i, 0.4)).collect(),
            heads: 2,
        }
    }

    #[test]
    fn mha_matches_torch_on_a_tiny_example() {
        let m = tiny_mha();
        let q: Vec<f32> = (0..2 * 8)
            .map(|i| ((i as f64) * 0.21 + 0.5).cos() as f32)
            .collect();
        let kv: Vec<f32> = (0..3 * 8)
            .map(|i| ((i as f64) * 0.13 - 0.2).sin() as f32)
            .collect();
        let y = m.forward(&q, 2, &kv, 3, 8);
        for (a, b) in y.iter().zip(MHA_WANT) {
            assert!(close(*a, b, 2e-6), "{a} vs {b}");
        }
    }
}
