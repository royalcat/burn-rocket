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
//! [`Online`] is the tiled long-context form (`ROCKET_FA_TILE_KV`): instead of one
//! softmax over the full key axis it folds one key tile at a time into the
//! FlashAttention-2 running state (row max `m`, denominator `l`, unnormalized output
//! `acc`), so the score matrix it touches is one tile wide. It runs the same scale /
//! mask / soft-cap row kernels and the same `exp` as [`Softmax`].
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
//!
//! # Host vs on-NPU placement (`ROCKET_ATTN_HOST_SOFTMAX`)
//!
//! The C library's attention entries can run their row softmax either on the host (its
//! flash default) or through `rocket_softmax_fp16`, the on-NPU composition this crate
//! ports as [`crate::ops::softmax::softmax`] (its `mha_self` default). [`RowSoftmax`]
//! owns both arms for one caller: the host pass and the f16 masked-score matrix the
//! on-NPU arm feeds to that composition, plus the [`host_softmax`] knob resolution
//! (`0` = on-NPU). The tiled long-context path ([`Online`]) has no such choice — its
//! running max/rescale is sequential across key tiles and inherently host-side — so it
//! always runs the host pass.

use half::f16;

use crate::Device;
use crate::error::Result;
use crate::knobs;

use super::oneshot::fa_score;

/// Scores at or below this clamp to it (the C library's `-30000` padding key).
pub(crate) const PAD_KEY_SCORE: f32 = -30000.0;

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

    /// Live bytes of the reusable row buffer.
    pub fn bytes(&self) -> usize {
        self.row.capacity() * size_of::<f32>()
    }
}

// ── Softmax placement: host pass vs the on-NPU composition ────────────────────

/// Where the materialized attention paths run their row softmax
/// (C `ROCKET_ATTN_HOST_SOFTMAX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxPlacement {
    /// The host pass ([`Softmax`]): scale, soft-cap, mask and `exp` in f32, one f16
    /// narrowing at the end (the C `host_softmax_rows` semantics).
    Host,
    /// The library's on-NPU composition ([`crate::ops::softmax::softmax`]): the masked
    /// scores narrow to f16, then the DPU `EXP` LUT, the feature-axis row sum (fp32)
    /// and the per-row scale run on the NPU, with the row max and `1/sum` on the host.
    Npu,
}

impl SoftmaxPlacement {
    /// Whether this placement is the host pass (for reporting).
    pub fn is_host(self) -> bool {
        matches!(self, SoftmaxPlacement::Host)
    }
}

/// Whether the materialized attention entries softmax on the host:
/// `ROCKET_ATTN_HOST_SOFTMAX=0` selects the on-NPU path, anything else (or unset) the
/// host one. Resolved per call, like the crate's other attention knobs, so one process
/// can A/B the two arms.
///
/// The C library reads this knob with *opposite defaults* per entry — its `mha_self`
/// (`g_attn_host_sm`) is on-NPU unless the variable is set at all, its flash entries
/// (`fa_host_softmax`) are host unless it parses to 0. **This port keeps the host pass
/// as the default everywhere** (the behaviour of every Rust attention entry before the
/// knob existed) and takes the value: `ROCKET_ATTN_HOST_SOFTMAX=0` reproduces the C's
/// on-NPU `mha_self` arm, `=1` its host arm.
pub fn host_softmax() -> bool {
    knobs::int("ROCKET_ATTN_HOST_SOFTMAX") != Some(0)
}

/// [`host_softmax`] as a [`SoftmaxPlacement`].
pub fn placement() -> SoftmaxPlacement {
    if host_softmax() {
        SoftmaxPlacement::Host
    } else {
        SoftmaxPlacement::Npu
    }
}

/// `out[0..tp][0..kn] = f16(fa_score(sc, scale, softcap, mask))`, the C
/// `fa_mask_scores` write: the real key columns take the additive mask (rows at or past
/// `n_tokens` take none, as in C) and the pad columns clamp to `-30000`, which then
/// underflows to ~0 in the softmax.
///
/// The arithmetic is the C's `fa_score`, in f32 with one f16 narrowing at the end — no
/// fused multiply-add, so the masked score is exactly the value the C library's f16 `sc`
/// buffer holds. (The host arm of [`RowSoftmax`] deliberately keeps f32 scores instead;
/// this is the arm that matches the C on the NPU.)
#[allow(clippy::too_many_arguments)]
fn masked_scores(
    sc: &[f16],
    out: &mut [f16],
    mask: Option<&[f16]>,
    n_tokens: usize,
    n_kv: usize,
    tp: usize,
    kn: usize,
    scale: f32,
    softcap: f32,
) {
    debug_assert!(sc.len() >= tp * kn && out.len() >= tp * kn && n_kv <= kn);
    let pad = f16::from_f32(PAD_KEY_SCORE);
    for i in 0..tp {
        let srow = &sc[i * kn..(i + 1) * kn];
        let mrow = mask
            .filter(|_| i < n_tokens)
            .map(|m| &m[i * n_kv..i * n_kv + n_kv]);
        let row = &mut out[i * kn..(i + 1) * kn];
        for j in 0..n_kv {
            let m = mrow.map_or(0.0, |m| m[j].to_f32());
            row[j] = f16::from_f32(fa_score(srow[j].to_f32(), scale, softcap, m));
        }
        row[n_kv..].fill(pad);
    }
}

/// One materialized attention call's row softmax, in either placement.
///
/// Holds the host pass ([`Softmax`]: its f32 row buffer and kernel mode) and the f16
/// masked-score matrix the on-NPU arm needs, so a steady request stream (one call per
/// head per layer) reallocates neither. [`RowSoftmax::run`] resolves
/// [`ROCKET_ATTN_HOST_SOFTMAX`](host_softmax) per call.
#[derive(Debug, Clone)]
pub struct RowSoftmax {
    /// The host pass (`SOFTMAX_EXACT=1` still selects its scalar kernel).
    host: Softmax,
    /// The on-NPU arm's masked f16 scores, the C `fa_mask_scores` write before the
    /// softmax. Only the on-NPU arm sizes it.
    masked: Vec<f16>,
}

impl Default for RowSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

impl RowSoftmax {
    pub fn new() -> Self {
        RowSoftmax {
            host: Softmax::new(),
            masked: Vec::new(),
        }
    }

    /// Softmax the `n_tokens` real rows of a padded `[tp, kn]` f16 score matrix into
    /// `p`, on the host or on the NPU.
    ///
    /// The mask, scale and soft-cap are applied to the scores first and the semantics
    /// match [`Softmax::run`] exactly — including the `-30000` clamp and the uniform
    /// fallback of a fully masked row — so the two arms differ only in where the
    /// `exp`/sum/scale run and in the rounding of the masked scores (f32 on the host,
    /// one f16 narrowing before the NPU, exactly as the C writes its `sc` buffer).
    ///
    /// The on-NPU arm reports the composition's own refusals (the `INT_MAX` element
    /// guard, a zero dimension); the host arm cannot fail.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        dev: &Device,
        scores: &[f16],
        p: &mut [f16],
        mask: Option<&[f16]>,
        n_tokens: usize,
        n_kv: usize,
        tp: usize,
        kn: usize,
        scale: f32,
        softcap: f32,
    ) -> Result<SoftmaxPlacement> {
        if tp == 0 || kn == 0 {
            return Ok(placement());
        }
        debug_assert!(n_tokens <= tp && n_kv <= kn);
        debug_assert!(scores.len() >= tp * kn && p.len() >= tp * kn);
        if host_softmax() {
            self.host
                .run(scores, p, mask, n_tokens, n_kv, tp, kn, scale, softcap);
            return Ok(SoftmaxPlacement::Host);
        }
        self.masked.resize(tp * kn, f16::ZERO);
        masked_scores(
            scores,
            &mut self.masked,
            mask,
            n_tokens,
            n_kv,
            tp,
            kn,
            scale,
            softcap,
        );
        crate::ops::softmax::softmax(dev, tp, kn, &self.masked, p)?;
        Ok(SoftmaxPlacement::Npu)
    }

    /// Live host bytes of both arms.
    pub fn bytes(&self) -> usize {
        self.host.bytes() + self.masked.capacity() * size_of::<f16>()
    }

    /// Release the on-NPU arm's matrix when the working shape has shrunk well below it
    /// (the 4x rule the attention scratch uses); `need` is the current `tp * kn`.
    pub fn trim(&mut self, need: usize) {
        if need == 0 {
            return;
        }
        if self.masked.len() > 4 * need {
            self.masked = Vec::new();
        }
    }
}

/// A tile row whose best visible score sits at or below this has no visible key at
/// all: the mask clamp [`PAD_KEY_SCORE`] plus one (the C `fa_heads_range_tiled` test).
const NO_VISIBLE_KEY: f32 = PAD_KEY_SCORE + 1.0;

/// The tiled long-context path's running softmax state, one head at a time.
///
/// Port of the C `fa_heads_range_tiled` fold (`fa_scratch.fm/fl/facc` plus the row
/// buffer it walks): [`Online::begin`] resets the state, [`Online::fold`] folds one
/// key tile in, [`Online::add`] accumulates that tile's `P·V`, and [`Online::finish`]
/// normalizes. All fp32 — the running max/rescale is inherently sequential across
/// tiles and needs the scores in f32, which is also why the tiled path is
/// host-softmax only.
///
/// Grow-only: the buffers keep the largest shape they have served, so a persistent
/// attention context reallocates neither `acc` (`[tp, dv]`) nor the row buffer.
#[derive(Debug)]
pub struct Online {
    mode: SoftmaxMode,
    /// f32 working row over one tile's columns.
    row: Vec<f32>,
    /// Running per-row max `m`.
    m: Vec<f32>,
    /// Running per-row denominator `l`.
    l: Vec<f32>,
    /// Running unnormalized output, `[tp, dv]`.
    acc: Vec<f32>,
}

impl Default for Online {
    fn default() -> Self {
        Self::new()
    }
}

impl Online {
    /// The process default mode (as [`Softmax::new`]).
    pub fn new() -> Self {
        Self::with_mode(Softmax::new().mode)
    }

    /// An explicit kernel mode (benchmarks, the self-test).
    pub fn with_mode(mode: SoftmaxMode) -> Self {
        Online {
            mode,
            row: Vec::new(),
            m: Vec::new(),
            l: Vec::new(),
            acc: Vec::new(),
        }
    }

    /// Grow to the tile shape and reset the running state (`m = -inf`, `l = 0`,
    /// `acc = 0`).
    pub fn begin(&mut self, tp: usize, wp: usize, dv: usize) {
        if self.row.len() < wp {
            self.row.resize(wp, 0.0);
        }
        if self.m.len() < tp {
            self.m.resize(tp, f32::NEG_INFINITY);
        }
        if self.l.len() < tp {
            self.l.resize(tp, 0.0);
        }
        if self.acc.len() < tp * dv {
            self.acc.resize(tp * dv, 0.0);
        }
        self.m[..tp].fill(f32::NEG_INFINITY);
        self.l[..tp].fill(0.0);
        self.acc[..tp * dv].fill(0.0);
    }

    /// Fold key tile `[j0, j0 + w)` into the running state, writing that tile's
    /// unnormalized `exp(s - m_new)` probabilities as f16 to `p`.
    ///
    /// `scores` is the tile's QK result `[tp, wp]`; `mask` the full additive
    /// `[n_tokens, n_kv]` mask (row `i < n_tokens`) whose columns start at `j0`;
    /// `wp >= w` with the pad columns treated like masked-out keys. A row whose every
    /// key in this tile is masked is skipped outright — its `p` row is zeroed (so the
    /// AV matmul adds nothing) and its running state is left untouched, which is what
    /// keeps a causal or sliding-window tile outside the row's range from weighting
    /// the mask clamp.
    #[allow(clippy::too_many_arguments)]
    pub fn fold(
        &mut self,
        scores: &[f16],
        p: &mut [f16],
        mask: Option<&[f16]>,
        j0: usize,
        n_tokens: usize,
        n_kv: usize,
        w: usize,
        tp: usize,
        wp: usize,
        dv: usize,
        scale: f32,
        softcap: f32,
    ) {
        if tp == 0 || wp == 0 {
            return;
        }
        debug_assert!(w <= wp && w <= n_kv && scores.len() >= tp * wp && p.len() >= tp * wp);
        debug_assert!(self.row.len() >= wp && self.m.len() >= tp && self.acc.len() >= tp * dv);
        let capped = softcap != 0.0;
        let mode = self.mode;
        for i in 0..tp {
            let srow = &scores[i * wp..(i + 1) * wp];
            let mrow = mask
                .filter(|_| i < n_tokens)
                .map(|m| &m[i * n_kv + j0..i * n_kv + j0 + w]);
            let row = &mut self.row[..wp];
            let tmax = match (mrow, capped) {
                (None, false) => kern::row_scores_plain(srow, row, w, scale),
                (Some(m), false) => kern::row_scores_masked(srow, row, m, w, scale),
                (m, true) => kern::row_scores_softcap(srow, row, m, w, scale, softcap),
            };
            let prow = &mut p[i * wp..(i + 1) * wp];
            if tmax <= NO_VISIBLE_KEY {
                prow.fill(f16::ZERO);
                continue;
            }
            let m_old = self.m[i];
            let m_new = if m_old > tmax { m_old } else { tmax };
            let corr = if m_old == f32::NEG_INFINITY {
                0.0
            } else {
                exp_mode(mode, m_old - m_new)
            };
            let rs = exp_row_mode(mode, row, m_new);
            kern::row_write(row, prow, 1.0);
            self.l[i] = self.l[i] * corr + rs;
            if corr != 1.0 {
                for v in self.acc[i * dv..(i + 1) * dv].iter_mut() {
                    *v *= corr;
                }
            }
            self.m[i] = m_new;
        }
    }

    /// `acc += ctx` for the tile just multiplied (`ctx` is the `[tp, dv]` AV result).
    pub fn add(&mut self, ctx: &[f16], tp: usize, dv: usize) {
        debug_assert!(self.acc.len() >= tp * dv && ctx.len() >= tp * dv);
        for (a, &c) in self.acc[..tp * dv].iter_mut().zip(&ctx[..tp * dv]) {
            *a += c.to_f32();
        }
    }

    /// Write `out[t, dv] = acc / l`, with a row that saw no visible key at all
    /// (`l == 0`) written as zeros — the C finalize.
    pub fn finish(&self, out: &mut [f16], t: usize, dv: usize) {
        for row in 0..t {
            let l = self.l[row];
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            let ar = &self.acc[row * dv..(row + 1) * dv];
            let orow = &mut out[row * dv..(row + 1) * dv];
            for (o, &a) in orow.iter_mut().zip(ar) {
                *o = f16::from_f32(a * inv);
            }
        }
    }

    /// Live bytes of the running state.
    pub fn bytes(&self) -> usize {
        (self.row.len() + self.m.len() + self.l.len() + self.acc.len()) * size_of::<f32>()
    }

    /// Whether the state is far larger than a `(tp, wp, dv)` call needs (the same
    /// 4x rule the attention scratch uses).
    pub fn oversized(&self, tp: usize, wp: usize, dv: usize) -> bool {
        self.row.len() > 4 * wp || self.acc.len() > 4 * tp * dv
    }

    /// Release everything; the next [`Online::begin`] reallocates.
    pub fn release(&mut self) {
        self.row = Vec::new();
        self.m = Vec::new();
        self.l = Vec::new();
        self.acc = Vec::new();
    }
}

/// `exp(x)` for `x <= 0` in `mode` (the materialized path's [`Softmax`] and the tiled
/// [`Online`] share both kernels).
fn exp_mode(mode: SoftmaxMode, x: f32) -> f32 {
    match mode {
        SoftmaxMode::Fast => exp_neg(x),
        SoftmaxMode::Exact => x.exp(),
    }
}

/// Replace `row` with `exp(row - mx)` and return the sum, in `mode`.
fn exp_row_mode(mode: SoftmaxMode, row: &mut [f32], mx: f32) -> f32 {
    match mode {
        SoftmaxMode::Fast => kern::row_exp(row, mx),
        SoftmaxMode::Exact => {
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                let e = exp_mode(mode, *v - mx);
                *v = e;
                sum += e;
            }
            sum
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

    // ── Softmax placement ──────────────────────────────────────────────────────

    /// The block has no device, so this covers the knob resolution and the masked-score
    /// write (the C `fa_mask_scores`), which is the whole on-NPU arm except the
    /// composition itself (that one is `ops::softmax::softmax`, gated on the board).
    #[test]
    fn host_softmax_knob_takes_the_value() {
        let _g = crate::knobs::TEST_ENV.lock().unwrap();
        unsafe { std::env::remove_var("ROCKET_ATTN_HOST_SOFTMAX") };
        assert!(host_softmax(), "unset is the host pass (the Rust default)");
        assert_eq!(placement(), SoftmaxPlacement::Host);
        unsafe { std::env::set_var("ROCKET_ATTN_HOST_SOFTMAX", "1") };
        assert!(host_softmax());
        unsafe { std::env::set_var("ROCKET_ATTN_HOST_SOFTMAX", "0") };
        assert!(!host_softmax(), "0 selects the on-NPU arm");
        assert_eq!(placement(), SoftmaxPlacement::Npu);
        assert!(!placement().is_host());
        // Anything else parses as "host", like the C `atoi(...) != 0`.
        unsafe { std::env::set_var("ROCKET_ATTN_HOST_SOFTMAX", "2") };
        assert!(host_softmax());
        unsafe { std::env::remove_var("ROCKET_ATTN_HOST_SOFTMAX") };
    }

    #[test]
    fn masked_scores_scales_masks_and_clamps_like_the_c() {
        // 2 padded rows x 4 padded columns, 2 real keys.
        let (tp, kn, n_kv, n_tokens) = (2usize, 4usize, 2usize, 1usize);
        let sc: Vec<f16> = [1.0f32, -2.0, 99.0, 99.0, 3.0, 4.0, 99.0, 99.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let mask: Vec<f16> = [-1.0f32, 0.0].into_iter().map(f16::from_f32).collect();
        let mut out = vec![f16::ONE; tp * kn];
        masked_scores(&sc, &mut out, Some(&mask), n_tokens, n_kv, tp, kn, 0.5, 0.0);
        let pad = f16::from_f32(PAD_KEY_SCORE);
        // Row 0 (real): scale, add the mask, then one f16 narrowing.
        assert_eq!(out[0], f16::from_f32(1.0 * 0.5 - 1.0));
        assert_eq!(out[1], f16::from_f32(-2.0 * 0.5 + 0.0));
        assert_eq!(&out[2..4], &[pad, pad], "pad columns clamp");
        // Row 1 (past n_tokens): no mask, its scores are never read.
        assert_eq!(out[4], f16::from_f32(3.0 * 0.5));
        assert_eq!(out[5], f16::from_f32(4.0 * 0.5));
        assert_eq!(&out[6..8], &[pad, pad]);
        // No mask at all, and the clamp pulls a deep negative score up to the floor.
        let sc2: Vec<f16> = [-1.0e5f32, 2.0, 0.0, 0.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let mut out2 = vec![f16::ZERO; kn];
        masked_scores(&sc2, &mut out2, None, 1, n_kv, 1, kn, 1.0, 0.0);
        assert_eq!(out2[0], pad);
        assert_eq!(out2[1], f16::from_f32(2.0));
        // A soft-cap bounds the score: 30*tanh(100/30) approaches 30 but never reaches it.
        let sc3 = vec![f16::from_f32(100.0); 2];
        let mut out3 = vec![f16::ZERO; 2];
        masked_scores(&sc3, &mut out3, None, 1, 2, 1, 2, 1.0, 30.0);
        let capped = out3[0].to_f32();
        assert!(capped < 30.0 && capped > 29.9, "{capped}");
    }

    #[test]
    fn row_softmax_trim_releases_a_stale_matrix() {
        let mut sm = RowSoftmax::new();
        sm.masked = vec![f16::ZERO; 4096];
        assert!(sm.bytes() > 0);
        sm.trim(4096);
        assert_eq!(sm.masked.len(), 4096, "the working shape stays");
        sm.trim(16);
        assert!(sm.masked.is_empty(), "a 256x smaller request releases it");
        sm.trim(0);
        assert!(sm.masked.is_empty());
    }
}
