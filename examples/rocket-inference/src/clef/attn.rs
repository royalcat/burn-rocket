//! Blocked causal GQA attention on the CPU for the NPU/emulated placements.
//!
//! The scores of a (head, query block) pair come from one `gemm` call `Q_b K^T`, the
//! causal softmax runs in place, and a second `gemm` multiplies by `V`. Tasks fan out over
//! the current rayon pool, so the working set is `threads x (block x keys x 4 B)`: at 4k
//! tokens and a 128-row block, 2 MiB per task (16 MiB for 8 workers), no `[s, s]` matrix.
//!
//! Moving the score/AV matmuls to the NPU (`rocket_npu::attn`, or
//! `burn_rocket::attention_causal_block`) is a follow-up; this phase keeps them here.

use super::model::TextConfig;
use rayon::prelude::*;

/// Query rows per task.
const QB: usize = 128;

/// `out[m, n] = beta * lhs[m, k] . rhs[k, n]` over strided row-major operands, one thread.
///
/// # Safety
/// The three slices must cover every element addressed by the strides.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_strided(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut f32,
    dst_rs: isize,
    lhs: *const f32,
    lhs_rs: isize,
    rhs: *const f32,
    rhs_rs: isize,
    rhs_cs: isize,
    beta: f32,
) {
    // SAFETY: forwarded from the caller.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            dst,
            1,
            dst_rs,
            false,
            lhs,
            1,
            lhs_rs,
            rhs,
            rhs_cs,
            rhs_rs,
            0.0,
            beta,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

/// Causal GQA attention. `q [s, heads*d]`, `k`/`v [s, kv*d]` -> context `[s, heads*d]`
/// (`q head h` reads `kv head h / (heads / kv)`).
pub fn causal_gqa(cfg: &TextConfig, q: &[f32], k: &[f32], v: &[f32], s: usize) -> Vec<f32> {
    let (nh, nkv, d) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    assert_eq!(q.len(), s * nh * d);
    assert_eq!(k.len(), s * nkv * d);
    assert_eq!(v.len(), s * nkv * d);
    let rep = nh / nkv;
    let scale = 1.0 / (d as f32).sqrt();
    let nblocks = s.div_ceil(QB);
    let tasks: Vec<(usize, usize)> = (0..nblocks)
        .flat_map(|b| (0..nh).map(move |h| (b, h)))
        .collect();
    let parts: Vec<Vec<f32>> = tasks
        .par_iter()
        .map(|&(b, h)| {
            let q0 = b * QB;
            let bq = QB.min(s - q0);
            let keys = q0 + bq; // keys [0, keys) are visible to some row of the block
            let g = h / rep;
            let mut sc = vec![0f32; bq * keys];
            // SAFETY: q rows q0.. at head h, k rows 0..keys at kv head g are in range;
            // sc holds bq * keys elements.
            unsafe {
                gemm_strided(
                    bq,
                    keys,
                    d,
                    sc.as_mut_ptr(),
                    keys as isize,
                    q.as_ptr().add((q0 * nh + h) * d),
                    (nh * d) as isize,
                    k.as_ptr().add(g * d),
                    1,
                    (nkv * d) as isize,
                    scale,
                );
            }
            for (i, row) in sc.chunks_mut(keys).enumerate() {
                let lim = q0 + i + 1; // visible keys: [0, lim)
                let max = row[..lim].iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
                let mut sum = 0f32;
                for x in &mut row[..lim] {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                let inv = 1.0 / sum;
                for x in &mut row[..lim] {
                    *x *= inv;
                }
                row[lim..].fill(0.0);
            }
            let mut ctx = vec![0f32; bq * d];
            // SAFETY: sc is [bq, keys]; v rows 0..keys at kv head g; ctx is [bq, d].
            unsafe {
                gemm_strided(
                    bq,
                    d,
                    keys,
                    ctx.as_mut_ptr(),
                    d as isize,
                    sc.as_ptr(),
                    keys as isize,
                    v.as_ptr().add(g * d),
                    (nkv * d) as isize,
                    1,
                    1.0,
                );
            }
            ctx
        })
        .collect();
    let mut out = vec![0f32; s * nh * d];
    for (&(b, h), ctx) in tasks.iter().zip(&parts) {
        let q0 = b * QB;
        for (i, row) in ctx.chunks(d).enumerate() {
            out[((q0 + i) * nh + h) * d..][..d].copy_from_slice(row);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clef::model::TextConfig;

    fn cfg() -> TextConfig {
        TextConfig {
            hidden: 64,
            layers: 1,
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            experts: 4,
            top_k: 2,
            inter: 32,
            eps: 1e-6,
            theta: 1e6,
            norm_topk: true,
        }
    }

    fn naive(c: &TextConfig, q: &[f32], k: &[f32], v: &[f32], s: usize) -> Vec<f32> {
        let (nh, nkv, d) = (c.heads, c.kv_heads, c.head_dim);
        let mut out = vec![0f32; s * nh * d];
        for i in 0..s {
            for h in 0..nh {
                let g = h / (nh / nkv);
                let sc: Vec<f64> = (0..=i)
                    .map(|j| {
                        (0..d)
                            .map(|x| {
                                f64::from(q[(i * nh + h) * d + x])
                                    * f64::from(k[(j * nkv + g) * d + x])
                            })
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for x in 0..d {
                    out[(i * nh + h) * d + x] = (0..=i)
                        .map(|j| e[j] / z * f64::from(v[(j * nkv + g) * d + x]))
                        .sum::<f64>() as f32;
                }
            }
        }
        out
    }

    #[test]
    fn blocked_attention_matches_a_naive_f64_reference() {
        let c = cfg();
        // 300 tokens: two full 128-row blocks and a ragged 44-row tail
        for s in [1usize, 5, 128, 129, 300] {
            let mut seed = s as u64 + 1;
            let mut r = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            };
            let q: Vec<f32> = (0..s * c.heads * c.head_dim).map(|_| r()).collect();
            let k: Vec<f32> = (0..s * c.kv_heads * c.head_dim).map(|_| r()).collect();
            let v: Vec<f32> = (0..s * c.kv_heads * c.head_dim).map(|_| r()).collect();
            let got = causal_gqa(&c, &q, &k, &v, s);
            let want = naive(&c, &q, &k, &v, s);
            let max = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max < 2e-6, "s={s}: max abs diff {max}");
        }
    }
}
