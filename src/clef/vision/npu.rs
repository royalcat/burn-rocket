//! The NPU placement of the vision tower (aarch64 `npu` builds).
//!
//! * **Linears**: every tower linear is packed ONCE as a resident fp16 weight
//!   (`RocketCtx::pack_weight`, canonical tiling, so one pack serves any M >= 4): the path the
//!   EmbeddingGemma 2 and d3 towers use. The BF16 checkpoint values convert to fp16 exactly in
//!   range (a non-finite result refuses the pack). `fc2`'s `K = 4304` is not a multiple of 32:
//!   its weight columns and the activation are zero-padded to 4320 (zeros add nothing).
//!   The bias is added on the host in f32 after the fp16 readback. ~1.07 GiB of buffer
//!   objects for the 117 weights; they live as long as the tower (it is loaded on the first
//!   image request).
//! * **Attention**: the fp16 flash-attention driver per frame (`RocketFaCtx::flash_attn`, no
//!   mask, host fast softmax). The driver wants `head_dim % 32` and `dv % 16`: the 72-wide
//!   head is zero-padded to 96 for q/k (dot products unchanged) and to 80 for v/out (the
//!   padded columns are dropped).
//! * **Range**: an activation that rounds to an fp16 infinity, or a driver score saturation
//!   (`ROCKET_E_RANGE`), is a [`Fp16Range`]: the tower reruns that image in f32 on the CPU.
//!
//! No board run was possible while this was written: the logic is covered by host tests of the
//! padding (`vision::tests`) and by `Emu16`, which reproduces the numerics.

use super::super::engine_npu::NpuFailure;
use super::super::model::Shards;
use super::compute::{Exec, Fp16Range, pad_heads, pad_heads_transposed, unpack_padded};
use super::{Lin, widen_par};
use anyhow::{Context, Result};
use burn_rocket::half::f16;
use burn_rocket::{RocketCtx, RocketFaCtx, RocketWeight};
use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Nominal M a resident weight is packed with (the canonical tiling serves every M).
const PACK_M: usize = 512;
/// `ROCKET_E_RANGE`.
const E_RANGE: i32 = -6;
/// Rows per resident matmul call (the d3/emb2 `ROCKET_MATMUL_CHUNK_M` idea, fixed here): a
/// video's `[P, 4320]` fp16 input BO would otherwise reach hundreds of MiB.
const MAX_ROWS: usize = 4096;

pub struct NpuExec {
    ctx: Mutex<RocketCtx>,
    fa: Mutex<RocketFaCtx>,
    weights: Vec<RocketWeight>,
    bytes: usize,
}

fn up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

fn failure(e: burn_rocket::Error, m: usize, k: usize, n: usize) -> anyhow::Error {
    if e.rc == E_RANGE {
        return anyhow::Error::new(Fp16Range(format!("{} (m={m} k={k} n={n})", e.op)));
    }
    anyhow::Error::new(NpuFailure {
        op: e.op,
        rc: e.rc,
        m,
        k,
        n,
    })
}

/// `[n, k]` f32 weight -> `[np, kp]` fp16 with zero padding.
fn weight_f16(w: &[f32], n: usize, k: usize, np: usize, kp: usize, name: &str) -> Result<Vec<f16>> {
    let bad = AtomicBool::new(false);
    let mut out = vec![f16::ZERO; np * kp];
    out.par_chunks_mut(kp)
        .take(n)
        .enumerate()
        .for_each(|(r, row)| {
            for (d, &s) in row[..k].iter_mut().zip(&w[r * k..(r + 1) * k]) {
                let h = f16::from_f32(s);
                if !h.is_finite() {
                    bad.store(true, Ordering::Relaxed);
                }
                *d = h;
            }
        });
    anyhow::ensure!(
        !bad.load(Ordering::Relaxed),
        "{name}: a weight is outside the fp16 range"
    );
    Ok(out)
}

impl NpuExec {
    /// Open the contexts and pack every linear of the plan.
    pub fn pack(lins: &[&Lin], shards: &Shards, threads: usize) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let mut ctx = RocketCtx::new(threads).map_err(|e| failure(e, 0, 0, 0))?;
        let fa = RocketFaCtx::new(threads).map_err(|e| failure(e, 0, 0, 0))?;
        let mut weights = Vec::with_capacity(lins.len());
        let mut bytes = 0usize;
        for &lin in lins {
            debug_assert_eq!(lin.slot, weights.len());
            let (file, info) = shards.get(&lin.name)?;
            let w = widen_par(file, info);
            let (kp, np) = (up(lin.k, 32), up(lin.n, 16));
            let h = weight_f16(&w, lin.n, lin.k, np, kp, &lin.name)?;
            drop(w);
            let packed = ctx
                .pack_weight(PACK_M, kp, np, &h)
                .map_err(|e| failure(e, PACK_M, kp, np))
                .with_context(|| format!("pack {}", lin.name))?;
            bytes += packed.bytes();
            weights.push(packed);
            shards.release();
        }
        eprintln!(
            "clef: packed a tower on the NPU: {} weights, {:.0} MiB of BOs in {:.1}s",
            weights.len(),
            bytes as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64()
        );
        Ok(Self {
            ctx: Mutex::new(ctx),
            fa: Mutex::new(fa),
            weights,
            bytes,
        })
    }
}

impl Exec for NpuExec {
    fn label(&self) -> String {
        format!(
            "fp16 resident on the NPU ({} weights, {:.0} MiB of BOs)",
            self.weights.len(),
            self.bytes as f64 / (1024.0 * 1024.0)
        )
    }

    fn matmul(
        &self,
        lin: &Lin,
        _shards: &Shards,
        x: &[f32],
        m: usize,
        _threads: usize,
    ) -> Result<Vec<f32>> {
        // A video is many thousands of rows (a 2-frame 20x28 group alone is 560): bound the
        // per-call input/output BOs. Rows are independent, so chunking is bit-identical.
        if m > MAX_ROWS {
            let mut y = Vec::with_capacity(m * lin.n);
            for c in x.chunks(MAX_ROWS * lin.k) {
                y.extend(self.matmul(lin, _shards, c, c.len() / lin.k, _threads)?);
            }
            return Ok(y);
        }
        let (k, n) = (lin.k, lin.n);
        let (kp, np) = (up(k, 32), up(n, 16));
        let mp = up(m.max(1), 4);
        let bad = AtomicBool::new(false);
        // A [mp, kp]: rows beyond m repeat the last row (ignored on readback), columns beyond k are 0
        let mut a = vec![f16::ZERO; mp * kp];
        a.par_chunks_mut(kp).enumerate().for_each(|(r, row)| {
            let src = &x[r.min(m - 1) * k..][..k];
            for (d, &s) in row[..k].iter_mut().zip(src) {
                let h = f16::from_f32(s);
                if !h.is_finite() {
                    bad.store(true, Ordering::Relaxed);
                }
                *d = h;
            }
        });
        if bad.load(Ordering::Relaxed) {
            return Err(anyhow::Error::new(Fp16Range(format!(
                "the input of {}",
                lin.name
            ))));
        }
        let mut c = vec![f16::ZERO; mp * np];
        self.ctx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .matmul_prepacked(mp, kp, np, &a, &mut c, &self.weights[lin.slot])
            .map_err(|e| failure(e, m, kp, np))?;
        drop(a);
        let mut y = vec![0f32; m * n];
        y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
            for (d, s) in row.iter_mut().zip(&c[r * np..r * np + n]) {
                let v = s.to_f32();
                if !v.is_finite() {
                    bad.store(true, Ordering::Relaxed);
                }
                *d = v;
            }
        });
        if bad.load(Ordering::Relaxed) {
            return Err(anyhow::Error::new(Fp16Range(format!(
                "the output of {}",
                lin.name
            ))));
        }
        Ok(y)
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        segs: &[(usize, usize)],
        heads: usize,
        hd: usize,
        _threads: usize,
    ) -> Result<Vec<f32>> {
        let width = heads * hd;
        let p = q.len() / width;
        let (hdp, dvp) = (up(hd, 32), up(hd, 16));
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; p * width];
        for &(s, e) in segs {
            let n = e - s;
            let (q16, k16) = (
                pad_heads(&q[s * width..e * width], n, heads, hd, hdp),
                pad_heads(&k[s * width..e * width], n, heads, hd, hdp),
            );
            let v16 = pad_heads_transposed(&v[s * width..e * width], n, heads, hd, dvp);
            let mut o16 = vec![f16::ZERO; heads * n * dvp];
            self.fa
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .flash_attn(
                    n, n, hdp, dvp, heads, heads, scale, 0.0, &q16, &k16, &v16, None, &mut o16,
                )
                .map_err(|e| failure(e, n, n, hdp))?;
            drop((q16, k16, v16));
            unpack_padded(&o16, n, heads, hd, dvp, &mut out[s * width..e * width]);
        }
        Ok(out)
    }

    fn trim(&self, keep_bytes: usize) {
        self.ctx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .trim_to_budget(keep_bytes);
        let mut fa = self.fa.lock().unwrap_or_else(|p| p.into_inner());
        let m = fa.mem_stats();
        if m.bo_bytes + m.host_bytes > keep_bytes {
            fa.clear_scratch();
        } else {
            fa.trim_scratch();
        }
    }
}
