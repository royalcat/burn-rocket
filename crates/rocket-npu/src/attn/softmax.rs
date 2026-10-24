//! Host-side softmax over the attention score matrix.
//!
//! [`Attention::flash`](super::Attention::flash) brings the full `[Tp, Kn]` f16
//! score matrix host-side (the driver has no on-NPU softmax), applies the
//! additive mask, scale and optional soft-cap, softmaxes every row and uploads
//! the f16 probabilities as the AV operand. At the vision tower's geometry
//! (`p = 2394`, 12 heads) that is 12 × 2394² = **68.8M scores per layer**; §18.4
//! estimated it at ~0.33 s of the 0.43 s NPU op, and this module's S1
//! measurements put the scalar pass at ~0.20 s of it (0.0672 s per head on one
//! A76 core × 3 heads per worker), which the fast path cuts to ~0.05 s.
//!
//! Two implementations live here, selected per [`Softmax`] value:
//!
//! - [`SoftmaxMode::Fast`] (default): a branch-free `exp` built on a degree-6
//!   polynomial for `2^f` with the exponent assembled from bits (8 lanes per
//!   step on aarch64, with hardware f16 conversion; scalar elsewhere),
//!   vectorized row passes and a single per-row scratch buffer reused across
//!   rows and calls. Rows that need neither a mask nor a soft-cap — the vision
//!   tower's whole workload — take the fully branch-free kernel; the mask and
//!   soft-cap paths keep the original semantics with the same fast `exp`.
//! - [`SoftmaxMode::Exact`]: the original scalar `f32::exp` implementation,
//!   bit-identical to the pre-spike code, kept for A/B. Selected process-wide by
//!   `SOFTMAX_EXACT=1` (or `ROCKET_SOFTMAX_EXACT=1`); the bench selects either
//!   mode with [`Softmax::with_mode`].
//!
//! Both modes apply the same arithmetic as the C library's host softmax:
//! `s = clamp(softcap·tanh((scale·s)/softcap) + mask, -30000)`, then
//! `p = exp(s - max) / Σ exp(s - max)`. A row whose scores are all clamped
//! (a fully masked query, or a padded row) yields a uniform distribution, and
//! padded columns are clamped like masked-out keys — the old path's behaviour,
//! preserved so A/B runs differ only in rounding.

use half::f16;

use crate::knobs;

/// Scores at or below this clamp to it (the C library's `-30000` padding key).
const PAD_KEY_SCORE: f32 = -30000.0;

/// `1 / ln 2`: `exp(x) = 2^(x·LOG2E)`.
const LOG2E: f32 = std::f32::consts::LOG2_E;

/// Degree-6 Chebyshev fit of `2^f` on `[-0.5, 0.5]`, `C0` pinned to 1 so
/// `exp(0)` is exact. Worst-case relative error 2.5e-9 in f64; f32 evaluation
/// adds ~1e-7 (Horner), which is the accuracy of the whole fast path.
const P2: [f32; 7] = [
    1.0,
    // ln 2: the linear term of `2^f` (clippy reads the literal as a misused
    // `LN_2` constant, so name it).
    std::f32::consts::LN_2,
    0.240_226_5,
    0.055_503_27,
    0.009_618_057,
    0.001_340_042_8,
    0.000_154_614_45,
];

/// `2^f` for `f ∈ [-0.5, 0.5]` (Horner, f32).
#[inline(always)]
fn exp2_poly(f: f32) -> f32 {
    let mut p = P2[6];
    p = p.mul_add(f, P2[5]);
    p = p.mul_add(f, P2[4]);
    p = p.mul_add(f, P2[3]);
    p = p.mul_add(f, P2[2]);
    p = p.mul_add(f, P2[1]);
    p.mul_add(f, P2[0])
}

/// `exp(x)` for `x <= 0`, branch-free.
///
/// The argument is always `score - row_max` here, so it cannot be positive; the
/// low clamp keeps the exponent-bit assembly in range (every result below
/// `2^-126` is zero at f16 output precision anyway). Accuracy is ~3e-7 relative
/// for `|x| <= 20` (the range that carries probability mass) and ~3e-6 at
/// `x = -85`, where f32 `LOG2E`'s error is amplified by `|x|`.
#[inline(always)]
fn exp_neg(x: f32) -> f32 {
    let y = (x * LOG2E).max(-126.0);
    let n = (y + 0.5).floor();
    let f = y - n;
    let scale = f32::from_bits((((n as i32) + 127) as u32) << 23);
    exp2_poly(f) * scale
}

/// Which softmax kernel a [`Softmax`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxMode {
    /// Polynomial `exp`, vectorized row passes (default).
    Fast,
    /// The original scalar `f32::exp` path, for A/B (`SOFTMAX_EXACT=1`).
    Exact,
}

/// A reusable host softmax: one row buffer, one mode.
///
/// Construct one per worker thread (the row buffer is `O(Kn)` and the attention
/// engine keeps one per worker) and call [`Softmax::run`] once per head.
#[derive(Debug, Clone)]
pub struct Softmax {
    /// f32 working row (the scores, then their exponentials).
    row: Vec<f32>,
    mode: SoftmaxMode,
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Softmax {
    /// The process default: [`SoftmaxMode::Fast`] unless `SOFTMAX_EXACT` /
    /// `ROCKET_SOFTMAX_EXACT` selects the original path.
    pub fn new() -> Self {
        let exact = knobs::boolean("ROCKET_SOFTMAX_EXACT") == Some(true)
            || knobs::boolean("SOFTMAX_EXACT") == Some(true);
        Self::with_mode(if exact {
            SoftmaxMode::Exact
        } else {
            SoftmaxMode::Fast
        })
    }

    /// A softmax with an explicit mode (benchmarks, the attention self-test).
    pub fn with_mode(mode: SoftmaxMode) -> Self {
        Softmax {
            row: Vec::new(),
            mode,
        }
    }

    /// The active mode.
    pub fn mode(&self) -> SoftmaxMode {
        self.mode
    }

    /// Softmax the `n_tokens` real rows of a padded `[tp, kn]` score matrix,
    /// writing f16 probabilities to `p`.
    ///
    /// `mask` is the additive `[n_tokens, n_kv]` f16 mask for the real rows
    /// (`None` = no mask; padded rows never take one). `scale` and `softcap`
    /// match the attention op's arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        scores: &[f16],
        p: &mut [f16],
        mask: Option<&[f16]>,
        n_tokens: usize,
        n_kv: usize,
        tp: usize,
        kn: usize,
        scale: f32,
        softcap: f32,
    ) {
        if tp == 0 || kn == 0 {
            return;
        }
        debug_assert!(n_tokens <= tp && n_kv <= kn);
        debug_assert!(scores.len() >= tp * kn && p.len() >= tp * kn);
        self.row.resize(kn, 0.0);
        match self.mode {
            SoftmaxMode::Fast => self.fast(scores, p, mask, n_tokens, n_kv, tp, kn, scale, softcap),
            SoftmaxMode::Exact => {
                self.exact(scores, p, mask, n_tokens, n_kv, tp, kn, scale, softcap)
            }
        }
    }

    /// The original scalar implementation (pre-spike), unchanged apart from the
    /// row buffer living on `self`.
    #[allow(clippy::too_many_arguments)]
    fn exact(
        &mut self,
        sc: &[f16],
        p: &mut [f16],
        mask: Option<&[f16]>,
        n_tokens: usize,
        n_kv: usize,
        tp: usize,
        kn: usize,
        scale: f32,
        softcap: f32,
    ) {
        let rowbuf = &mut self.row;
        for i in 0..tp {
            let srow = &sc[i * kn..(i + 1) * kn];
            let mrow = mask
                .filter(|_| i < n_tokens)
                .map(|m| &m[i * n_kv..i * n_kv + n_kv]);
            let mut mx = PAD_KEY_SCORE;
            for (j, slot) in rowbuf.iter_mut().enumerate().take(kn) {
                let s = if j < n_kv {
                    let mut s = srow[j].to_f32() * scale;
                    if softcap != 0.0 {
                        s = softcap * (s / softcap).tanh();
                    }
                    if let Some(m) = mrow {
                        s += m[j].to_f32();
                    }
                    s.max(PAD_KEY_SCORE)
                } else {
                    PAD_KEY_SCORE
                };
                *slot = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0f32;
            for v in rowbuf.iter_mut() {
                let e = (*v - mx).exp();
                *v = e;
                sum += e;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            let prow = &mut p[i * kn..(i + 1) * kn];
            for j in 0..kn {
                prow[j] = f16::from_f32(rowbuf[j] * inv);
            }
        }
    }

    /// Two value-space row kernels (plain / masked / soft-cap) plus a shared
    /// vectorized `exp` and f16 write.
    #[allow(clippy::too_many_arguments)]
    fn fast(
        &mut self,
        sc: &[f16],
        p: &mut [f16],
        mask: Option<&[f16]>,
        n_tokens: usize,
        n_kv: usize,
        tp: usize,
        kn: usize,
        scale: f32,
        softcap: f32,
    ) {
        let capped = softcap != 0.0;
        for i in 0..tp {
            let srow = &sc[i * kn..(i + 1) * kn];
            let mrow = mask
                .filter(|_| i < n_tokens)
                .map(|m| &m[i * n_kv..i * n_kv + n_kv]);
            let row = &mut self.row[..kn];
            let mx = match (mrow, capped) {
                (None, false) => kern::row_scores_plain(srow, row, n_kv, scale),
                (Some(m), false) => kern::row_scores_masked(srow, row, m, n_kv, scale),
                (m, true) => kern::row_scores_softcap(srow, row, m, n_kv, scale, softcap),
            };
            let sum = kern::row_exp(row, mx);
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            kern::row_write(row, &mut p[i * kn..(i + 1) * kn], inv);
        }
    }
}

// ---------------------------------------------------------------------------
// Row kernels: NEON on aarch64 (hardware f16 conversion, 8 lanes per step),
// scalar elsewhere. Both implement the same arithmetic as `Softmax::exact`.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
use neon as kern;
#[cfg(not(all(target_arch = "aarch64", target_feature = "fp16")))]
use scalar as kern;

/// Portable kernels; also the reference for the NEON ones' tails (unused on
/// aarch64, where the NEON module is selected).
#[allow(dead_code)]
mod scalar {
    use half::f16;

    use super::{PAD_KEY_SCORE, exp_neg};

    /// Fill `row` with the scaled scores (`row[n_kv..]` = padding) and return
    /// the row max (at least [`PAD_KEY_SCORE`]).
    pub(super) fn row_scores_plain(srow: &[f16], row: &mut [f32], n_kv: usize, scale: f32) -> f32 {
        let mut mx = PAD_KEY_SCORE;
        for j in 0..n_kv {
            let s = (srow[j].to_f32() * scale).max(PAD_KEY_SCORE);
            row[j] = s;
            mx = mx.max(s);
        }
        row[n_kv..].fill(PAD_KEY_SCORE);
        mx
    }

    pub(super) fn row_scores_masked(
        srow: &[f16],
        row: &mut [f32],
        mask: &[f16],
        n_kv: usize,
        scale: f32,
    ) -> f32 {
        let mut mx = PAD_KEY_SCORE;
        for j in 0..n_kv {
            let s = (srow[j].to_f32().mul_add(scale, mask[j].to_f32())).max(PAD_KEY_SCORE);
            row[j] = s;
            mx = mx.max(s);
        }
        row[n_kv..].fill(PAD_KEY_SCORE);
        mx
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn row_scores_softcap(
        srow: &[f16],
        row: &mut [f32],
        mask: Option<&[f16]>,
        n_kv: usize,
        scale: f32,
        softcap: f32,
    ) -> f32 {
        let mut mx = PAD_KEY_SCORE;
        for j in 0..n_kv {
            let mut s = srow[j].to_f32() * scale;
            s = softcap * (s / softcap).tanh();
            if let Some(m) = mask {
                s += m[j].to_f32();
            }
            let s = s.max(PAD_KEY_SCORE);
            row[j] = s;
            mx = mx.max(s);
        }
        row[n_kv..].fill(PAD_KEY_SCORE);
        mx
    }

    /// Replace `row` with `exp(row - mx)` and return the sum.
    pub(super) fn row_exp(row: &mut [f32], mx: f32) -> f32 {
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = exp_neg(*v - mx);
            *v = e;
            sum += e;
        }
        sum
    }

    /// `out = f16(row · inv)`.
    pub(super) fn row_write(row: &[f32], out: &mut [f16], inv: f32) {
        for (o, &v) in out.iter_mut().zip(row) {
            *o = f16::from_f32(v * inv);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
mod neon {
    use core::arch::aarch64::*;

    use half::f16;

    use super::P2;

    const LANES: usize = 8;

    /// `exp(x)` for `x <= 0`, 4 lanes (`2^f` polynomial, exponent from bits).
    #[inline]
    fn exp4(x: float32x4_t) -> float32x4_t {
        unsafe {
            let y = vmulq_n_f32(x, std::f32::consts::LOG2_E);
            let y = vmaxq_f32(y, vdupq_n_f32(-126.0));
            let n = vrndmq_f32(vaddq_f32(y, vdupq_n_f32(0.5)));
            let f = vsubq_f32(y, n);
            // ACLE `vfmaq_n_f32(a, b, c)` is `a + b*c`, so Horner reads
            // `p = C[k] + f*p`.
            let mut p = vdupq_n_f32(P2[6]);
            p = vfmaq_f32(vdupq_n_f32(P2[5]), f, p);
            p = vfmaq_f32(vdupq_n_f32(P2[4]), f, p);
            p = vfmaq_f32(vdupq_n_f32(P2[3]), f, p);
            p = vfmaq_f32(vdupq_n_f32(P2[2]), f, p);
            p = vfmaq_f32(vdupq_n_f32(P2[1]), f, p);
            p = vfmaq_f32(vdupq_n_f32(P2[0]), f, p);
            let e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
            vmulq_f32(p, vreinterpretq_f32_s32(e))
        }
    }

    /// Eight f16 values at `p` as two f32 vectors (hardware `fcvtl`).
    #[inline]
    fn load8(p: *const f16) -> (float32x4_t, float32x4_t) {
        unsafe {
            // `vld1q_f16`/`vcvt_f32_f16` on the raw bits: the f16-typed
            // intrinsics are still unstable (`stdarch_neon_f16`), the u16
            // reinterpretations are not and compile to the same `fcvtl`.
            let h = vreinterpretq_f16_u16(vld1q_u16(p as *const u16));
            (
                vcvt_f32_f16(vget_low_f16(h)),
                vcvt_f32_f16(vget_high_f16(h)),
            )
        }
    }

    /// Eight f32 values at `p` as two vectors.
    #[inline]
    fn load8f(p: *const f32) -> (float32x4_t, float32x4_t) {
        unsafe { (vld1q_f32(p), vld1q_f32(p.add(4))) }
    }

    #[inline]
    fn store8f(p: *mut f32, a: float32x4_t, b: float32x4_t) {
        unsafe {
            vst1q_f32(p, a);
            vst1q_f32(p.add(4), b);
        }
    }

    /// Two f32 vectors as eight f16 values (hardware `fcvtn`).
    #[inline]
    fn store8(p: *mut f16, a: float32x4_t, b: float32x4_t) {
        unsafe {
            let h = vcombine_f16(vcvt_f16_f32(a), vcvt_f16_f32(b));
            vst1q_u16(p as *mut u16, vreinterpretq_u16_f16(h));
        }
    }

    pub(super) fn row_scores_plain(srow: &[f16], row: &mut [f32], n_kv: usize, scale: f32) -> f32 {
        let mut mx;
        let mut j = 0;
        unsafe {
            let vs = vdupq_n_f32(scale);
            let floor = vdupq_n_f32(super::PAD_KEY_SCORE);
            let mut vmax = floor;
            let (sp, rp) = (srow.as_ptr(), row.as_mut_ptr());
            while j + LANES <= n_kv {
                let (a, b) = load8(sp.add(j));
                let a = vmaxq_f32(vmulq_f32(a, vs), floor);
                let b = vmaxq_f32(vmulq_f32(b, vs), floor);
                store8f(rp.add(j), a, b);
                vmax = vmaxq_f32(vmax, vmaxq_f32(a, b));
                j += LANES;
            }
            mx = vmaxvq_f32(vmax);
        }
        while j < n_kv {
            let s = (srow[j].to_f32() * scale).max(super::PAD_KEY_SCORE);
            row[j] = s;
            mx = mx.max(s);
            j += 1;
        }
        row[n_kv..].fill(super::PAD_KEY_SCORE);
        mx
    }

    pub(super) fn row_scores_masked(
        srow: &[f16],
        row: &mut [f32],
        mask: &[f16],
        n_kv: usize,
        scale: f32,
    ) -> f32 {
        let mut mx;
        let mut j = 0;
        unsafe {
            let vs = vdupq_n_f32(scale);
            let floor = vdupq_n_f32(super::PAD_KEY_SCORE);
            let mut vmax = floor;
            let (sp, mp, rp) = (srow.as_ptr(), mask.as_ptr(), row.as_mut_ptr());
            while j + LANES <= n_kv {
                let (sa, sb) = load8(sp.add(j));
                let (ma, mb) = load8(mp.add(j));
                let a = vmaxq_f32(vfmaq_f32(ma, sa, vs), floor);
                let b = vmaxq_f32(vfmaq_f32(mb, sb, vs), floor);
                store8f(rp.add(j), a, b);
                vmax = vmaxq_f32(vmax, vmaxq_f32(a, b));
                j += LANES;
            }
            mx = vmaxvq_f32(vmax);
        }
        while j < n_kv {
            let s = srow[j]
                .to_f32()
                .mul_add(scale, mask[j].to_f32())
                .max(super::PAD_KEY_SCORE);
            row[j] = s;
            mx = mx.max(s);
            j += 1;
        }
        row[n_kv..].fill(super::PAD_KEY_SCORE);
        mx
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn row_scores_softcap(
        srow: &[f16],
        row: &mut [f32],
        mask: Option<&[f16]>,
        n_kv: usize,
        scale: f32,
        softcap: f32,
    ) -> f32 {
        // Soft-cap is the rare path (`tanh` has no cheap vector form): scalar.
        let mut mx = super::PAD_KEY_SCORE;
        for j in 0..n_kv {
            let mut s = srow[j].to_f32() * scale;
            s = softcap * (s / softcap).tanh();
            if let Some(m) = mask {
                s += m[j].to_f32();
            }
            let s = s.max(super::PAD_KEY_SCORE);
            row[j] = s;
            mx = mx.max(s);
        }
        row[n_kv..].fill(super::PAD_KEY_SCORE);
        mx
    }

    pub(super) fn row_exp(row: &mut [f32], mx: f32) -> f32 {
        let n = row.len();
        let mut j = 0;
        let mut sum;
        unsafe {
            let vm = vdupq_n_f32(-mx);
            let mut vsum = vdupq_n_f32(0.0);
            let p = row.as_mut_ptr();
            while j + LANES <= n {
                let (a, b) = load8f(p.add(j));
                let a = exp4(vaddq_f32(a, vm));
                let b = exp4(vaddq_f32(b, vm));
                store8f(p.add(j), a, b);
                vsum = vaddq_f32(vsum, vaddq_f32(a, b));
                j += LANES;
            }
            sum = vaddvq_f32(vsum);
        }
        while j < n {
            let e = super::exp_neg(row[j] - mx);
            row[j] = e;
            sum += e;
            j += 1;
        }
        sum
    }

    pub(super) fn row_write(row: &[f32], out: &mut [f16], inv: f32) {
        let n = row.len();
        let mut j = 0;
        unsafe {
            let vi = vdupq_n_f32(inv);
            let (rp, op) = (row.as_ptr(), out.as_mut_ptr());
            while j + LANES <= n {
                let (a, b) = load8f(rp.add(j));
                store8(op.add(j), vmulq_f32(a, vi), vmulq_f32(b, vi));
                j += LANES;
            }
        }
        while j < n {
            out[j] = f16::from_f32(row[j] * inv);
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;

    /// The pre-spike implementation, as a standalone reference.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        sc: &[f16],
        p: &mut [f16],
        mask: Option<&[f16]>,
        n_tokens: usize,
        n_kv: usize,
        tp: usize,
        kn: usize,
        scale: f32,
        softcap: f32,
    ) {
        let mut rowbuf = vec![0.0f32; kn];
        for i in 0..tp {
            let srow = &sc[i * kn..(i + 1) * kn];
            let mrow = mask
                .filter(|_| i < n_tokens)
                .map(|m| &m[i * n_kv..i * n_kv + n_kv]);
            let mut mx = PAD_KEY_SCORE;
            for (j, slot) in rowbuf.iter_mut().enumerate().take(kn) {
                let s = if j < n_kv {
                    let mut s = srow[j].to_f32() * scale;
                    if softcap != 0.0 {
                        s = softcap * (s / softcap).tanh();
                    }
                    if let Some(m) = mrow {
                        s += m[j].to_f32();
                    }
                    s.max(PAD_KEY_SCORE)
                } else {
                    PAD_KEY_SCORE
                };
                *slot = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0f32;
            for v in rowbuf.iter_mut() {
                let e = (*v - mx).exp();
                *v = e;
                sum += e;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            let prow = &mut p[i * kn..(i + 1) * kn];
            for j in 0..kn {
                prow[j] = f16::from_f32(rowbuf[j] * inv);
            }
        }
    }

    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 60.0
    }

    fn scores(tp: usize, kn: usize, seed: u64) -> Vec<f16> {
        let mut s = seed | 1;
        (0..tp * kn).map(|_| f16::from_f32(rnd(&mut s))).collect()
    }

    fn causal_mask(t: usize, kv: usize) -> Vec<f16> {
        let mut m = vec![f16::ZERO; t * kv];
        for i in 0..t {
            for j in (i + 1)..kv {
                m[i * kv + j] = f16::NEG_INFINITY;
            }
        }
        m
    }

    fn cosine(a: &[f16], b: &[f16]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x.to_f64(), y.to_f64());
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    fn max_abs(a: &[f16], b: &[f16]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0.0, f32::max)
    }

    /// Largest distance in f16 code units between two non-negative matrices.
    fn max_ulp(a: &[f16], b: &[f16]) -> u32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| u32::from(x.to_bits()).abs_diff(u32::from(y.to_bits())))
            .max()
            .unwrap_or(0)
    }

    fn row_sums(p: &[f16], tp: usize, kn: usize) -> Vec<f32> {
        (0..tp)
            .map(|i| p[i * kn..(i + 1) * kn].iter().map(|v| v.to_f32()).sum())
            .collect()
    }

    /// Compare one configuration against the reference: fast and exact.
    #[allow(clippy::too_many_arguments)]
    fn check(
        tp: usize,
        kn: usize,
        n_kv: usize,
        scale: f32,
        softcap: f32,
        mask: Option<Vec<f16>>,
        seed: u64,
    ) {
        let n_tokens = tp.min(if mask.is_some() { tp - 4 } else { tp });
        let sc = scores(tp, kn, seed);
        let mut want = vec![f16::ZERO; tp * kn];
        reference(
            &sc,
            &mut want,
            mask.as_deref(),
            n_tokens,
            n_kv,
            tp,
            kn,
            scale,
            softcap,
        );
        for mode in [SoftmaxMode::Fast, SoftmaxMode::Exact] {
            let mut got = vec![f16::ZERO; tp * kn];
            let mut sm = Softmax::with_mode(mode);
            sm.run(
                &sc,
                &mut got,
                mask.as_deref(),
                n_tokens,
                n_kv,
                tp,
                kn,
                scale,
                softcap,
            );
            if mode == SoftmaxMode::Exact {
                assert_eq!(got, want, "exact path must stay bit-identical");
                continue;
            }
            let cos = cosine(&got, &want);
            let maxd = max_abs(&got, &want);
            let ulp = max_ulp(&got, &want);
            assert!(
                ulp <= 1,
                "max|d| {maxd} ({ulp} f16 ulps) at kn={kn} mask={}",
                mask.is_some()
            );
            assert!(
                cos > 0.999_999_9,
                "{cos} at kn={kn} mask={}",
                mask.is_some()
            );
            for (i, sum) in row_sums(&got, tp, kn).into_iter().enumerate() {
                if i < n_tokens {
                    assert!((sum - 1.0).abs() < 1e-3, "row {i} sums to {sum}");
                }
            }
        }
    }

    #[test]
    fn exp_matches_libm() {
        // Post-max arguments are <= 0. The f32 LOG2E rounding costs `|x|·1.3e-8`
        // relative, so the deep tail (~1e-30 of the row mass) is the loosest; the
        // range that carries probability is accurate to ~3e-7.
        let mut x = -85.0f32;
        while x <= 0.0 {
            let (got, want) = (exp_neg(x), x.exp());
            let rel = (got - want).abs() / want;
            let tol = if x < -20.0 { 5e-6 } else { 1e-6 };
            assert!(rel < tol, "exp({x}) = {got} vs {want} (rel {rel})");
            x += 0.001;
        }
        assert_eq!(exp_neg(0.0), 1.0);
        // Deep underflow is clamped to ~2^-126, which rounds to zero as f16.
        assert!(exp_neg(-120.0) < 1.2e-38);
        assert_eq!(f16::from_f32(exp_neg(-90.0)), f16::ZERO);
        // Monotone non-decreasing (the mask/softmax path relies on it for ties).
        let mut prev = -1.0;
        let mut x = -60.0f32;
        while x <= 0.0 {
            let v = exp_neg(x);
            assert!(v >= prev, "exp_neg not monotone at {x}");
            prev = v;
            x += 0.0005;
        }
    }

    #[test]
    fn exp2_poly_accuracy() {
        let mut f = -0.5f32;
        while f <= 0.5 {
            let want = 2.0f32.powf(f);
            assert!((exp2_poly(f) - want).abs() / want < 3e-7, "f = {f}");
            f += 0.0001;
        }
    }

    #[test]
    fn plain_rows_match_reference() {
        // Vision geometry: 2394 patches padded to 2396 x 2400, scale 1.
        check(2396, 2400, 2394, 1.0, 0.0, None, 7);
        // Text-like geometry: 64 queries over 3633 keys, with a causal mask.
        check(64, 3648, 3633, 0.088, 0.0, Some(causal_mask(60, 3633)), 11);
        check(64, 3648, 3633, 0.088, 0.0, None, 13);
    }

    #[test]
    fn masked_rows_match_reference() {
        check(70, 128, 100, 0.125, 0.0, Some(causal_mask(66, 100)), 17);
        // A sliding window (negative offsets) like EmbeddingGemma's bands.
        let mut m = vec![f16::ZERO; 66 * 100];
        for i in 0..66 {
            for j in 0..100 {
                if (j as i64 - i as i64).abs() > 20 {
                    m[i * 100 + j] = f16::NEG_INFINITY;
                }
            }
        }
        check(70, 128, 100, 0.125, 0.0, Some(m), 19);
    }

    #[test]
    fn softcap_rows_match_reference() {
        check(68, 96, 90, 0.125, 30.0, Some(causal_mask(64, 90)), 23);
        check(68, 96, 90, 0.125, 50.0, None, 29);
    }

    #[test]
    fn fully_masked_row_is_uniform() {
        // A clamped row (all -inf mask) yields 1/kn everywhere, as before.
        let (tp, kn, n_kv) = (4, 64, 60);
        let sc = scores(tp, kn, 31);
        let mask = vec![f16::NEG_INFINITY; tp * n_kv];
        for mode in [SoftmaxMode::Fast, SoftmaxMode::Exact] {
            let mut p = vec![f16::ZERO; tp * kn];
            Softmax::with_mode(mode).run(&sc, &mut p, Some(&mask), tp, n_kv, tp, kn, 0.5, 0.0);
            let want = f16::from_f32(1.0 / kn as f32);
            assert!(p.iter().all(|&v| v == want), "{mode:?}");
        }
    }

    #[test]
    fn padded_columns_and_rows_are_zero() {
        // Padded key columns must not take probability mass; padded query rows
        // are computed (not read) and must stay finite. The tiny shape exercises
        // the vector kernels' scalar tail only.
        for (tp, kn, n_kv) in [(8usize, 64usize, 33usize), (4, 32, 1), (6, 40, 39)] {
            let sc = scores(tp, kn, 37);
            let mut p = vec![f16::ZERO; tp * kn];
            Softmax::new().run(&sc, &mut p, None, 6.min(tp), n_kv, tp, kn, 1.0, 0.0);
            for i in 0..tp {
                let row = &p[i * kn..(i + 1) * kn];
                assert!(row[n_kv..].iter().all(|&v| v == f16::ZERO), "row {i}");
                let sum: f32 = row[..n_kv].iter().map(|v| v.to_f32()).sum();
                assert!((sum - 1.0).abs() < 1e-3, "row {i} sums to {sum}");
            }
        }
    }

    #[test]
    fn mode_from_env() {
        // `with_mode` is authoritative; `new` reads the knobs (tested through a
        // fresh value, the environment is process-wide).
        assert_eq!(
            Softmax::with_mode(SoftmaxMode::Exact).mode(),
            SoftmaxMode::Exact
        );
        assert_eq!(
            Softmax::with_mode(SoftmaxMode::Fast).mode(),
            SoftmaxMode::Fast
        );
        assert_eq!(Softmax::new().mode(), SoftmaxMode::Fast);
        unsafe { std::env::set_var("SOFTMAX_EXACT", "1") };
        assert_eq!(Softmax::new().mode(), SoftmaxMode::Exact);
        unsafe { std::env::remove_var("SOFTMAX_EXACT") };
        unsafe { std::env::set_var("ROCKET_SOFTMAX_EXACT", "1") };
        assert_eq!(Softmax::new().mode(), SoftmaxMode::Exact);
        unsafe { std::env::remove_var("ROCKET_SOFTMAX_EXACT") };
        assert_eq!(Softmax::new().mode(), SoftmaxMode::Fast);
    }

    /// On aarch64 the NEON kernels must agree with the portable ones (only the
    /// summation order and FMA rounding differ, so this is a tolerance check,
    /// not bit equality).
    #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
    #[test]
    fn neon_matches_scalar() {
        use super::scalar;
        let (tp, kn, n_kv) = (9, 2400, 2394);
        let sc = scores(tp, kn, 41);
        let mask = causal_mask(tp, n_kv);
        let scale = 0.37;
        for i in 0..tp {
            let srow = &sc[i * kn..(i + 1) * kn];
            let mrow = &mask[i * n_kv..(i + 1) * n_kv];
            let (mut a, mut b) = (vec![0.0f32; kn], vec![0.0f32; kn]);
            let ma = kern::row_scores_masked(srow, &mut a, mrow, n_kv, scale);
            let mb = scalar::row_scores_masked(srow, &mut b, mrow, n_kv, scale);
            assert!((ma - mb).abs() < 1e-5, "row max {ma} vs {mb}");
            for j in 0..kn {
                assert!((a[j] - b[j]).abs() < 1e-5, "row {i} col {j}");
            }
            let (sa, sb) = (kern::row_exp(&mut a, ma), scalar::row_exp(&mut b, mb));
            assert!((sa - sb).abs() / sb < 1e-6, "sum {sa} vs {sb}");
            let (mut pa, mut pb) = (vec![f16::ZERO; kn], vec![f16::ZERO; kn]);
            kern::row_write(&a, &mut pa, 1.0 / sa);
            scalar::row_write(&b, &mut pb, 1.0 / sb);
            assert!(max_abs(&pa, &pb) < 1e-6, "row {i}");
        }
    }
}
