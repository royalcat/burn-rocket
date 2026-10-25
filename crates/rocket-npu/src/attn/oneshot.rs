//! The C library's stateless attention entries and its host references.
//!
//! `rocket_attn.c` exposes the same two compositions in several forms. The persistent
//! [`Attention`] context covers `rocket_fa_ctx_*`; everything else lives here, as
//! compositions of the validated pieces (`matmul::one_shot`, [`Attention::flash`],
//! [`Softmax`]) rather than a second attention implementation:
//!
//! - [`flash_attn`] (`rocket_flash_attn_fp16`, one worker fd) and [`flash_attn_mt`]
//!   (`_mt`, the head fan-out across worker fds). Both build a context per call, so a
//!   caller running attention repeatedly should hold an [`Attention`] instead.
//! - [`flash_attn_ref`] (`rocket_flash_attn_ref_fp16`): the device-free f64 oracle.
//! - [`mha_self`] (`rocket_mha_self_fp16`): the encoder attention sublayer — the four
//!   projections and the per-head attention in one call.
//! - [`mha_self_ref`] (`rocket_mha_self_ref_fp16`): its f64 oracle.
//!
//! # Numerics and deviations from the C originals
//!
//! The references are ported line by line, including their mixed precision: the
//! flash reference accumulates the QK dot in f64, narrows it to f32 for
//! `scale`/soft-cap/mask/clamp (the C `fa_score`), then exponentiates and reduces in
//! f64; the encoder reference keeps the whole pipeline in f64 (its `scale` is an f64
//! `1/sqrt(dh)`, where the device path uses the f32 `1/sqrtf(dh)`).
//!
//! Where the device entries differ from the C library, it is because they ride the
//! crate's existing attention kernel, and the differences are the documented ones:
//!
//! - the softmax is always the **host** pass ([`Softmax`]): the fast polynomial `exp`
//!   by default (`SOFTMAX_EXACT=1` for the scalar one) instead of the C's on-NPU
//!   softmax for `mha_self` (`ROCKET_ATTN_HOST_SOFTMAX`, default off there) and its
//!   `expf` loop for the flash path;
//! - the scores stay f32 between the matmul and the softmax, so the `scale` and the
//!   pad-key clamp are not rounded back through f16 (the C writes the masked scores
//!   into its f16 `sc` buffer before softmaxing).

use std::thread;

use half::f16;

use crate::Device;
use crate::device::affinity;
use crate::error::{Error, Result};
use crate::matmul::one_shot;

use super::softmax::PAD_KEY_SCORE;
use super::{Attention, Softmax, check_flash};

/// `rocket_flash_attn_fp16`: masked GQA attention on one worker fd.
///
/// Layouts and semantics are [`Attention::flash`]'s (the C header's contracts): `Q`
/// `[n_head][n_tokens][head_dim]`, `K` `[n_kv_heads][n_kv][head_dim]`, `V`
/// `[n_kv_heads][dv][n_kv]`, `mask` `[n_tokens][n_kv]` additive f16 (none = unmasked),
/// `out` `[n_head][n_tokens][dv]`.
///
/// The context (fd + scratch) is created and dropped per call, as in C; it also means
/// the chained-batch scratch is cold every call. Use [`Attention`] for a stream of
/// calls.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn(
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    dv: usize,
    n_head: usize,
    n_kv_heads: usize,
    scale: f32,
    softcap: f32,
    q: &[f16],
    k: &[f16],
    v: &[f16],
    mask: Option<&[f16]>,
    out: &mut [f16],
) -> Result<()> {
    Attention::new(1)?.flash(
        n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads, scale, softcap, q, k, v, mask, out,
    )
}

/// `rocket_flash_attn_fp16_mt`: [`flash_attn`] with the heads fanned out across
/// `nthreads` worker fds (one scheduling entity each, so the kernel spreads the head
/// ranges over the NPU cores).
///
/// `nthreads` is clamped to `[1, min(8, n_head)]` as in C; [`Attention::new`]
/// additionally caps it at the big-core count. Heads are computed independently of how
/// they are grouped, so the result is bit-identical to [`flash_attn`] (the self-test
/// asserts it).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_mt(
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    dv: usize,
    n_head: usize,
    n_kv_heads: usize,
    scale: f32,
    softcap: f32,
    q: &[f16],
    k: &[f16],
    v: &[f16],
    mask: Option<&[f16]>,
    out: &mut [f16],
    nthreads: usize,
) -> Result<()> {
    let threads = nthreads.clamp(1, super::MAX_WORKERS).min(n_head.max(1));
    Attention::new(threads)?.flash(
        n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads, scale, softcap, q, k, v, mask, out,
    )
}

/// `rocket_flash_attn_ref_fp16`: the exact host reference of [`flash_attn`] (no device
/// access), the oracle the board self-test compares against.
///
/// Ported from the C: the QK dot accumulates in f64 and is narrowed to f32 for the
/// scale, the optional soft-cap, the additive mask and the `-30000` clamp; the
/// softmax (`exp` and the row sum) and the AV accumulation run in f64; the f16 output
/// is the only narrowing after that. The C helper returns `void` and is validated by
/// its public entry, so this port validates here (`head_dim % 32`, `dv % 16` and the
/// GQA ratio, like the device entry) — the same contract, callable directly.
///
/// One consequence of the padding, inherited from the C: a query row whose every key
/// the mask removes gets the uniform distribution over whatever the softmax iterates.
/// Here that is the real `n_kv` keys; on the device it is the padded `Kn =
/// ceil(n_kv/32)·32` columns, and the padded columns have zero values, so such a row
/// comes out `n_kv / Kn` of this reference. Causal masks never fully mask a row; a
/// band mask over a shorter key axis can (see [`fully_masked_rows`]).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_ref(
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    dv: usize,
    n_head: usize,
    n_kv_heads: usize,
    scale: f32,
    softcap: f32,
    q: &[f16],
    k: &[f16],
    v: &[f16],
    mask: Option<&[f16]>,
    out: &mut [f16],
) -> Result<()> {
    let gqa = check_flash(
        n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads, q, k, v, mask, out,
    )?;
    let mut sc = vec![0.0f64; n_kv];
    for h in 0..n_head {
        let hk = h / gqa;
        let qh = &q[h * n_tokens * head_dim..];
        let kh = &k[hk * n_kv * head_dim..];
        let vh = &v[hk * dv * n_kv..];
        let oh = &mut out[h * n_tokens * dv..];
        for t in 0..n_tokens {
            let qr = &qh[t * head_dim..t * head_dim + head_dim];
            let mr = mask.map(|m| &m[t * n_kv..t * n_kv + n_kv]);
            let mut mx = f64::NEG_INFINITY;
            for (j, slot) in sc.iter_mut().enumerate() {
                let kr = &kh[j * head_dim..j * head_dim + head_dim];
                let mut qk = 0.0f64;
                for (a, b) in qr.iter().zip(kr) {
                    qk += a.to_f64() * b.to_f64();
                }
                let m = mr.map_or(0.0f32, |m| m[j].to_f32());
                let s = f64::from(fa_score(qk as f32, scale, softcap, m));
                *slot = s;
                mx = mx.max(s);
            }
            let mut sum = 0.0f64;
            for slot in sc.iter_mut() {
                *slot = (*slot - mx).exp();
                sum += *slot;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for (c, o) in oh[t * dv..t * dv + dv].iter_mut().enumerate() {
                let mut a = 0.0f64;
                for (p, &vj) in sc.iter().zip(&vh[c * n_kv..c * n_kv + n_kv]) {
                    a += p * vj.to_f64();
                }
                *o = crate::ops::f16_from_f64(a * inv);
            }
        }
    }
    Ok(())
}

/// One pre-softmax score, the C `fa_score`: scale, optional soft-cap, additive mask and
/// the clamp that lets a masked / padded key underflow to ~0 in f16.
fn fa_score(qk: f32, scale: f32, softcap: f32, mask: f32) -> f32 {
    let mut s = qk * scale;
    if softcap != 0.0 {
        s = softcap * (s / softcap).tanh();
    }
    (s + mask).max(PAD_KEY_SCORE)
}

/// How many query rows `mask` leaves with no visible key at all (every entry at or
/// below the `-30000` clamp).
///
/// Such a row's softmax is uniform, and the device's padded columns take a share of
/// that mass with zero values, so the row differs from the unpadded reference by
/// `n_kv / Kn` — the C library's behaviour, not a port difference. Callers comparing
/// the two (the board self-test) print it when it is non-zero.
pub fn fully_masked_rows(mask: &[f16], n_tokens: usize, n_kv: usize) -> usize {
    (0..n_tokens)
        .filter(|t| {
            mask[t * n_kv..(t + 1) * n_kv]
                .iter()
                .all(|v| v.to_f32() <= PAD_KEY_SCORE)
        })
        .count()
}

/// `rocket_mha_self_fp16`: the encoder self-attention sublayer, in one call.
///
/// `q = x·Wq^T + bq`, `k = x·Wk^T + bk`, `v = x·Wv^T + bv`; per head `h` of
/// `dh = d / n_head` columns: `P = softmax(Qh·Kh^T / sqrt(dh))`, `ctx_h = P·Vh`;
/// `out = concat(ctx)·Wo^T + bo`. Every weight is row-major `[out, in]` (the
/// `C = A·B^T` convention, i.e. a PyTorch `nn.Linear`), every bias is `[d]` and may be
/// `None` (the C's `NULL` — Whisper has `bq`, `bv`, `bo` and no `bk`). It is the pure
/// attention sublayer: no LayerNorm, no residual.
///
/// The row count is padded the way the C helper pads it: `Tp = ceil(T/4)·4` query rows
/// (the matmul's `M % 4`) and `Tn = ceil(T/32)·32` keys (the matmul's `N`/`K` align).
/// The pad rows/columns are zeroed, and the pad key columns are clamped to `-30000`
/// before the softmax, so they weigh ~0; the first `T` output rows are returned.
///
/// `nthreads` fans the composition out over worker fds — the three q/k/v projections
/// run concurrently, then the head ranges do — and is clamped to `1..=8`. Every matmul
/// keeps the shape it has in the C's single-fd form (nothing is split along `N`/`K`,
/// which would move the K-tile boundaries and the last bits), so any `nthreads` is
/// bit-identical to `nthreads = 1`; the self-test asserts it.
///
/// Deviations from the C, both from reusing the crate's attention kernel: the softmax
/// is the host [`Softmax`] (fast polynomial `exp`, single f16 rounding) where the C
/// defaults to its on-NPU softmax, and the scores stay f32 between the matmul and the
/// softmax instead of being written back through f16. See the module docs.
///
/// The shape contract is the C one plus the fp16 matmul's own alignment, which the C
/// also inherits: `d % n_head == 0` here, then `d % 32 == 0` for the projections and
/// `dh % 32 == 0` for the QK contraction (the AV output needs `dh % 16 == 0`, implied),
/// reported as the planner's shape error exactly as in C.
#[allow(clippy::too_many_arguments)]
pub fn mha_self(
    n_tokens: usize,
    d: usize,
    n_head: usize,
    x: &[f16],
    wq: &[f16],
    bq: Option<&[f16]>,
    wk: &[f16],
    bk: Option<&[f16]>,
    wv: &[f16],
    bv: Option<&[f16]>,
    wo: &[f16],
    bo: Option<&[f16]>,
    out: &mut [f16],
    nthreads: usize,
) -> Result<()> {
    let (dh, tp, tn) = mha_self_shape(n_tokens, d, n_head)?;
    check_mha_buffers(n_tokens, d, x, wq, bq, wk, bk, wv, bv, wo, bo, out)?;
    let threads = nthreads.clamp(1, super::MAX_WORKERS);
    let scale = 1.0 / (dh as f32).sqrt();
    let base = affinity::base();
    // The serial phases share one fd, as the C helper does; the parallel ones open
    // their own (one drm scheduling entity per fd is what spreads them over the cores).
    let dev = Device::open()?;

    // 1. q/k/v = x·W^T (+bias). M=Tp, K=d, N=d, on zero-padded x rows. The three
    //    projections are independent, so they run concurrently up to `nthreads` at a
    //    time (one fd each); with a single worker they share one fd, as the C does.
    let xp = pad_rows(x, n_tokens, d, tp);
    let ws = [wq, wk, wv];
    let biases = [bq, bk, bv];
    let mut qkv: [Vec<f16>; 3] = Default::default();
    for slot in qkv.iter_mut() {
        *slot = vec![f16::ZERO; tp * d];
    }
    let tasks = threads.min(qkv.len());
    for (ci, chunk) in qkv.chunks_mut(tasks).enumerate() {
        let wchunk = &ws[ci * tasks..][..chunk.len()];
        if tasks == 1 {
            for (slot, &w) in chunk.iter_mut().zip(wchunk) {
                one_shot(&dev, tp, d, d, &xp, w, slot)?;
            }
            continue;
        }
        let xp = &xp;
        let results: Vec<Result<()>> = thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter_mut()
                .zip(wchunk)
                .enumerate()
                .map(|(i, (slot, &w))| {
                    scope.spawn(move || -> Result<()> {
                        affinity::pin_current(ci * tasks + i, base);
                        let dev = Device::open()?;
                        one_shot(&dev, tp, d, d, xp, w, slot)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(Error::device("mha_self/worker", 0)))
                })
                .collect()
        });
        for r in results {
            r?;
        }
    }
    for (slot, bias) in qkv.iter_mut().zip(&biases) {
        add_bias(slot, *bias, n_tokens, d);
    }
    let [q, k, v] = qkv;

    // 2. Per head: scores = q_h·k_h^T over the Tn-padded keys, host softmax, AV.
    //    The pad key columns score 0 from the zero keys, which the softmax clamps to
    //    -30000 (the C masks them explicitly); the zero pad v columns then add nothing.
    let mut ctx = vec![f16::ZERO; tp * d];
    let workers = threads.min(n_head);
    let range = |i: usize| (i * n_head / workers, (i + 1) * n_head / workers);
    if workers <= 1 {
        let mut scr = HeadScratch::new(tp, tn, dh);
        for h in 0..n_head {
            let c = scr.head(&dev, h, n_tokens, d, tp, tn, scale, &q, &k, &v)?;
            scatter_head_cols(&mut ctx, c, n_tokens, d, h * dh, dh);
        }
    } else {
        let slabs: Vec<Result<Vec<f16>>> = thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|i| {
                    let (h0, h1) = range(i);
                    let (q, k, v) = (&q, &k, &v);
                    scope.spawn(move || -> Result<Vec<f16>> {
                        affinity::pin_current(i, base);
                        let dev = Device::open()?;
                        let mut scr = HeadScratch::new(tp, tn, dh);
                        let mut slab = vec![f16::ZERO; (h1 - h0) * n_tokens * dh];
                        for (slot, h) in (h0..h1).enumerate() {
                            let c = scr.head(&dev, h, n_tokens, d, tp, tn, scale, q, k, v)?;
                            let at = slot * n_tokens * dh;
                            slab[at..at + n_tokens * dh].copy_from_slice(c);
                        }
                        Ok(slab)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(Error::device("mha_self/heads", 0)))
                })
                .collect()
        });
        for (i, slab) in slabs.into_iter().enumerate() {
            let slab = slab?;
            let (h0, h1) = range(i);
            for (slot, h) in (h0..h1).enumerate() {
                let at = slot * n_tokens * dh;
                scatter_head_cols(
                    &mut ctx,
                    &slab[at..at + n_tokens * dh],
                    n_tokens,
                    d,
                    h * dh,
                    dh,
                );
            }
        }
    }

    // 3. out = ctx·Wo^T (+bo), first T rows.
    let mut wide = vec![f16::ZERO; tp * d];
    one_shot(&dev, tp, d, d, &ctx, wo, &mut wide)?;
    add_bias(&mut wide, bo, n_tokens, d);
    out[..n_tokens * d].copy_from_slice(&wide[..n_tokens * d]);
    Ok(())
}

/// One head's host scratch for [`mha_self`]: the padded gather tiles, the score and
/// probability matrices, and the head's `[T, dh]` context.
struct HeadScratch {
    dh: usize,
    qh: Vec<f16>,
    kh: Vec<f16>,
    vh: Vec<f16>,
    sc: Vec<f16>,
    p: Vec<f16>,
    ctx: Vec<f16>,
    sm: Softmax,
}

impl HeadScratch {
    fn new(tp: usize, tn: usize, dh: usize) -> Self {
        HeadScratch {
            dh,
            qh: vec![f16::ZERO; tp * dh],
            kh: vec![f16::ZERO; tn * dh],
            vh: vec![f16::ZERO; dh * tn],
            sc: vec![f16::ZERO; tp * tn],
            p: vec![f16::ZERO; tp * tn],
            // The AV matmul's output is the padded `[tp, dh]` tile the device writes
            // (`one_shot` sizes its output exactly); only its first `n_tokens` rows are
            // the head's context.
            ctx: vec![f16::ZERO; tp * dh],
            sm: Softmax::new(),
        }
    }

    /// Gather head `h`, run its QK matmul, host softmax and AV matmul, and return the
    /// head's `[n_tokens, dh]` context.
    #[allow(clippy::too_many_arguments)]
    fn head<'a>(
        &'a mut self,
        dev: &Device,
        h: usize,
        n_tokens: usize,
        d: usize,
        tp: usize,
        tn: usize,
        scale: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
    ) -> Result<&'a [f16]> {
        let dh = self.dh;
        let off = h * dh;
        gather_head_rows(&mut self.qh, q, n_tokens, d, off, dh);
        gather_head_rows(&mut self.kh, k, n_tokens, d, off, dh);
        gather_head_t_cols(&mut self.vh, v, n_tokens, d, off, dh, tn);
        one_shot(dev, tp, dh, tn, &self.qh, &self.kh, &mut self.sc)?;
        self.sm.run(
            &self.sc,
            &mut self.p,
            None,
            n_tokens,
            n_tokens,
            tp,
            tn,
            scale,
            0.0,
        );
        one_shot(dev, tp, tn, dh, &self.p, &self.vh, &mut self.ctx)?;
        Ok(&self.ctx[..n_tokens * dh])
    }
}

/// `rocket_mha_self_ref_fp16`: the exact f64 host reference of [`mha_self`] (no device
/// access).
///
/// Ported from the C: the projections, the `1/sqrt(dh)` scale, the softmax and the
/// output projection all run in f64 over the f16 inputs; only the f16 output narrows.
/// Unpadded — the reference has no matmul alignment to satisfy, so it runs over the
/// real row count (the C's `T`, not its `Tp`).
#[allow(clippy::too_many_arguments)]
pub fn mha_self_ref(
    n_tokens: usize,
    d: usize,
    n_head: usize,
    x: &[f16],
    wq: &[f16],
    bq: Option<&[f16]>,
    wk: &[f16],
    bk: Option<&[f16]>,
    wv: &[f16],
    bv: Option<&[f16]>,
    wo: &[f16],
    bo: Option<&[f16]>,
    out: &mut [f16],
) -> Result<()> {
    let (dh, _, _) = mha_self_shape(n_tokens, d, n_head)?;
    check_mha_buffers(n_tokens, d, x, wq, bq, wk, bk, wv, bv, wo, bo, out)?;
    let scale = 1.0 / (dh as f64).sqrt();
    let td = n_tokens * d;
    let mut q = vec![0.0f64; td];
    let mut k = vec![0.0f64; td];
    let mut v = vec![0.0f64; td];
    let mut ctx = vec![0.0f64; td];
    let mut sc = vec![0.0f64; n_tokens];

    let bias = |b: Option<&[f16]>, o: usize| b.map_or(0.0f64, |b| f64::from(b[o].to_f32()));
    for t in 0..n_tokens {
        for o in 0..d {
            let (mut aq, mut ak, mut av) = (0.0f64, 0.0f64, 0.0f64);
            let xr = &x[t * d..(t + 1) * d];
            for (i, &xi) in xr.iter().enumerate() {
                let xi = f64::from(xi.to_f32());
                aq += xi * f64::from(wq[o * d + i].to_f32());
                ak += xi * f64::from(wk[o * d + i].to_f32());
                av += xi * f64::from(wv[o * d + i].to_f32());
            }
            q[t * d + o] = aq + bias(bq, o);
            k[t * d + o] = ak + bias(bk, o);
            v[t * d + o] = av + bias(bv, o);
        }
    }
    for h in 0..n_head {
        let off = h * dh;
        for i in 0..n_tokens {
            let mut mx = f64::NEG_INFINITY;
            for (j, slot) in sc.iter_mut().enumerate() {
                let mut s = 0.0f64;
                for c in 0..dh {
                    s += q[i * d + off + c] * k[j * d + off + c];
                }
                let s = s * scale;
                *slot = s;
                mx = mx.max(s);
            }
            let mut sum = 0.0f64;
            for slot in sc.iter_mut() {
                *slot = (*slot - mx).exp();
                sum += *slot;
            }
            for c in 0..dh {
                let mut a = 0.0f64;
                for (slot, vj) in sc.iter().zip(v.chunks_exact(d)) {
                    a += slot * vj[off + c];
                }
                ctx[i * d + off + c] = a / sum;
            }
        }
    }
    for t in 0..n_tokens {
        for o in 0..d {
            let mut a = 0.0f64;
            for (i, &cr) in ctx[t * d..(t + 1) * d].iter().enumerate() {
                a += cr * f64::from(wo[o * d + i].to_f32());
            }
            out[t * d + o] = crate::ops::f16_from_f64(a + bias(bo, o));
        }
    }
    Ok(())
}

/// The `mha_self` shape contract: `(dh, Tp, Tn)`. Pure, so the host tests cover the
/// rejections without a device.
fn mha_self_shape(n_tokens: usize, d: usize, n_head: usize) -> Result<(usize, usize, usize)> {
    if n_tokens == 0 || d == 0 || n_head == 0 || !d.is_multiple_of(n_head) {
        return Err(Error::shape("mha_self/dims"));
    }
    Ok((
        d / n_head,
        n_tokens.div_ceil(4) * 4,
        n_tokens.div_ceil(32) * 32,
    ))
}

/// Length checks for `mha_self` and its reference, before any allocation or device
/// access (so a caller's mistake cannot reach the driver).
#[allow(clippy::too_many_arguments)]
fn check_mha_buffers(
    n_tokens: usize,
    d: usize,
    x: &[f16],
    wq: &[f16],
    bq: Option<&[f16]>,
    wk: &[f16],
    bk: Option<&[f16]>,
    wv: &[f16],
    bv: Option<&[f16]>,
    wo: &[f16],
    bo: Option<&[f16]>,
    out: &[f16],
) -> Result<()> {
    let td = n_tokens * d;
    let w_ok = [wq, wk, wv, wo].iter().all(|w| w.len() >= d * d);
    let b_ok = [bq, bk, bv, bo]
        .iter()
        .all(|b| b.is_none_or(|b| b.len() >= d));
    if x.len() < td || out.len() < td || !w_ok || !b_ok {
        return Err(Error::shape("mha_self/buffers"));
    }
    Ok(())
}

/// Zero-pad `src`'s `rows` rows of width `d` to `padded` rows of width `d`.
fn pad_rows(src: &[f16], rows: usize, d: usize, padded: usize) -> Vec<f16> {
    let mut dst = vec![f16::ZERO; padded * d];
    dst[..rows * d].copy_from_slice(&src[..rows * d]);
    dst
}

/// `c[m][n] += b[n]` over `rows` rows, in f32 with one f16 rounding per element (the C
/// `add_bias`); `None` does nothing.
fn add_bias(c: &mut [f16], b: Option<&[f16]>, rows: usize, n: usize) {
    let Some(b) = b else { return };
    for row in c.chunks_exact_mut(n).take(rows) {
        for (v, &bias) in row.iter_mut().zip(b) {
            *v = f16::from_f32(v.to_f32() + bias.to_f32());
        }
    }
}

/// Gather `rows` rows of width `width` starting at column `off` out of a `[rows, d]`
/// row-major matrix (one head's query/key slice, the C's per-head `memcpy`); the rest
/// of `dst` (`[padded, width]`) is zeroed first.
fn gather_head_rows(dst: &mut [f16], src: &[f16], rows: usize, d: usize, off: usize, width: usize) {
    dst.fill(f16::ZERO);
    for row in 0..rows {
        let s = row * d + off;
        let o = row * width;
        dst[o..o + width].copy_from_slice(&src[s..s + width]);
    }
}

/// Transpose one head's value slice out of `[rows, d]`: `dst[c][t] = src[t][off + c]`,
/// over a `[width, cols]` destination (the C `vhT[dh,Tn]` gather); the rest of `dst`
/// is zeroed first.
#[allow(clippy::too_many_arguments)]
fn gather_head_t_cols(
    dst: &mut [f16],
    src: &[f16],
    rows: usize,
    d: usize,
    off: usize,
    width: usize,
    cols: usize,
) {
    dst.fill(f16::ZERO);
    for row in 0..rows {
        let s = row * d + off;
        for c in 0..width {
            dst[c * cols + row] = src[s + c];
        }
    }
}

/// Store one head's `[rows, width]` slice back into columns `[off, off+width)` of a
/// `[rows, d]` row-major matrix — the C's per-head `memcpy` into `ctx`.
fn scatter_head_cols(
    dst: &mut [f16],
    src: &[f16],
    rows: usize,
    d: usize,
    off: usize,
    width: usize,
) {
    for row in 0..rows {
        let s = row * width;
        let o = row * d + off;
        dst[o..o + width].copy_from_slice(&src[s..s + width]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Everything here is device-free: the two references, the shape checks and the
    // padding/bias helpers are pure host code.

    fn rnd(seed: &mut u64) -> f16 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        f16::from_f32(((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
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

    /// An independent, deliberately naive f64 attention over the real rows only (no
    /// padding, no f32 narrowing): the cross-check for the ported reference's ordering.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        t: usize,
        kv: usize,
        dk: usize,
        dv: usize,
        heads: usize,
        hkv: usize,
        scale: f32,
        softcap: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
    ) -> Vec<f64> {
        let gqa = heads / hkv;
        let mut out = vec![0.0f64; heads * t * dv];
        for h in 0..heads {
            let hk = h / gqa;
            for i in 0..t {
                let mut sc = vec![0.0f64; kv];
                let mut mx = f64::NEG_INFINITY;
                for (j, slot) in sc.iter_mut().enumerate() {
                    let mut s = 0.0f64;
                    for c in 0..dk {
                        s += q[(h * t + i) * dk + c].to_f64() * k[(hk * kv + j) * dk + c].to_f64();
                    }
                    let mut s = s as f32 * scale;
                    if softcap != 0.0 {
                        s = softcap * (s / softcap).tanh();
                    }
                    s += mask.map_or(0.0, |m| m[i * kv + j].to_f32());
                    *slot = f64::from(s);
                    mx = mx.max(*slot);
                }
                let mut sum = 0.0f64;
                for slot in sc.iter_mut() {
                    *slot = (*slot - mx).exp();
                    sum += *slot;
                }
                for c in 0..dv {
                    let mut a = 0.0f64;
                    for (j, &p) in sc.iter().enumerate() {
                        a += p * v[(hk * dv + c) * kv + j].to_f64();
                    }
                    out[(h * t + i) * dv + c] = a / sum;
                }
            }
        }
        out
    }

    #[test]
    fn fa_score_scales_caps_masks_and_clamps() {
        // No soft-cap: plain scaling plus the additive mask.
        assert_eq!(fa_score(4.0, 0.5, 0.0, 0.0), 2.0);
        assert_eq!(fa_score(4.0, 0.5, 0.0, -1.0), 1.0);
        // A soft-cap bounds the score: 30·tanh(x/30) < 30 for a mid-range x and
        // saturates at 30 for a huge one.
        let capped = fa_score(60.0, 0.5, 30.0, 0.0);
        assert!((capped - 30.0 * (1.0f32).tanh()).abs() < 1e-5, "{capped}");
        assert!(capped < 30.0);
        let saturated = fa_score(1.0e4, 0.5, 30.0, 0.0);
        assert!((saturated - 30.0).abs() < 1e-4, "{saturated}");
        // The clamp is the fp16 pad-key floor, and only downward.
        assert_eq!(fa_score(-1.0e5, 1.0, 0.0, 0.0), PAD_KEY_SCORE);
        assert_eq!(fa_score(1.0, 1.0, 0.0, -f32::INFINITY), PAD_KEY_SCORE);
    }

    #[test]
    fn flash_ref_picks_the_one_visible_key() {
        // Every key masked but one: the softmax is a one-hot, so each output row is
        // exactly that key's value column (bit-exact, no rounding in play).
        let (t, kv, dk, dv, heads, hkv) = (3usize, 5usize, 32usize, 16usize, 2usize, 1usize);
        let mut seed = 0x1234_5678_9abc_def0u64;
        let q: Vec<f16> = (0..heads * t * dk).map(|_| rnd(&mut seed)).collect();
        let k: Vec<f16> = (0..hkv * kv * dk).map(|_| rnd(&mut seed)).collect();
        let v: Vec<f16> = (0..hkv * dv * kv).map(|_| rnd(&mut seed)).collect();
        let mut mask = vec![f16::NEG_INFINITY; t * kv];
        for (row, m) in mask.chunks_exact_mut(kv).enumerate() {
            m[(row + 1) % kv] = f16::ZERO;
        }
        let mut out = vec![f16::ZERO; heads * t * dv];
        flash_attn_ref(
            t,
            kv,
            dk,
            dv,
            heads,
            hkv,
            0.125,
            0.0,
            &q,
            &k,
            &v,
            Some(&mask),
            &mut out,
        )
        .unwrap();
        for h in 0..heads {
            for row in 0..t {
                let j = (row + 1) % kv;
                for c in 0..dv {
                    assert_eq!(
                        out[(h * t + row) * dv + c],
                        v[c * kv + j],
                        "head {h} row {row} col {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn flash_ref_pads_gqa_heads_and_no_mask_like_the_naive_reference() {
        // GQA 8/2, a real key count below the 32-align, a bidirectional mask with a
        // soft-cap: every deviation knob at once, plus a mask=None run.
        let (t, kv, dk, dv, heads, hkv) = (5usize, 37usize, 64usize, 32usize, 8usize, 2usize);
        let mut seed = 0x0bad_c0de_1234_5678u64;
        let q: Vec<f16> = (0..heads * t * dk).map(|_| rnd(&mut seed)).collect();
        let k: Vec<f16> = (0..hkv * kv * dk).map(|_| rnd(&mut seed)).collect();
        let v: Vec<f16> = (0..hkv * dv * kv).map(|_| rnd(&mut seed)).collect();
        let mut mask = vec![f16::ZERO; t * kv];
        for i in 0..t {
            for j in 0..kv {
                if j > i + 2 {
                    mask[i * kv + j] = f16::NEG_INFINITY;
                }
            }
        }
        for softcap in [0.0f32, 20.0] {
            for m in [Some(mask.as_slice()), None] {
                let mut out = vec![f16::ZERO; heads * t * dv];
                flash_attn_ref(
                    t, kv, dk, dv, heads, hkv, 0.125, softcap, &q, &k, &v, m, &mut out,
                )
                .unwrap();
                let want = naive(t, kv, dk, dv, heads, hkv, 0.125, softcap, &q, &k, &v, m);
                let got: Vec<f16> = out.clone();
                let mut want16 = vec![f16::ZERO; want.len()];
                for (a, &b) in want16.iter_mut().zip(&want) {
                    *a = f16::from_f64(b);
                }
                let cos = cosine(&got, &want16);
                assert!(cos > 0.999_999, "cosine {cos} (softcap {softcap})");
                assert!(max_abs(&got, &want16) < 2e-3, "softcap {softcap}");
            }
        }
    }

    #[test]
    fn flash_ref_rejects_the_device_entry_s_contract() {
        let q = vec![f16::ZERO; 4 * 2 * 32];
        let k = vec![f16::ZERO; 2 * 2 * 32];
        let v = vec![f16::ZERO; 2 * 16 * 2];
        let mut out = vec![f16::ZERO; 4 * 2 * 16];
        let mut bad = |head_dim: usize, dv: usize, heads: usize, hkv: usize| {
            flash_attn_ref(
                2, 2, head_dim, dv, heads, hkv, 1.0, 0.0, &q, &k, &v, None, &mut out,
            )
            .is_err()
        };
        assert!(!bad(32, 16, 4, 2), "the valid shape must pass");
        assert!(bad(48, 16, 4, 2), "head_dim % 32");
        assert!(bad(32, 24, 4, 2), "dv % 16");
        assert!(bad(32, 16, 4, 3), "n_head % n_kv_heads");
        assert!(bad(0, 16, 4, 2), "an empty head dim");
        assert!(bad(32, 0, 4, 2), "an empty value dim");
        // An empty token or key count is a shape error, as the C entry returns.
        let mut empty = Vec::new();
        assert!(flash_attn_ref(0, 2, 32, 16, 4, 2, 1.0, 0.0, &q, &k, &v, None, &mut out).is_err());
        assert!(
            flash_attn_ref(2, 0, 32, 16, 4, 2, 1.0, 0.0, &q, &k, &v, None, &mut empty).is_err()
        );
        // Buffers are checked too (the output is sized for dv=16 above).
        assert!(flash_attn_ref(2, 2, 32, 32, 4, 2, 1.0, 0.0, &q, &k, &v, None, &mut out).is_err());
        // A short mask is rejected.
        assert!(
            flash_attn_ref(
                2,
                2,
                32,
                16,
                4,
                2,
                1.0,
                0.0,
                &q,
                &k,
                &v,
                Some(&[f16::ZERO; 3]),
                &mut out
            )
            .is_err()
        );
    }

    #[test]
    fn fully_masked_rows_counts_emptied_rows() {
        let (t, kv) = (3usize, 4usize);
        let mut m = vec![f16::ZERO; t * kv];
        for j in 0..kv {
            m[kv + j] = f16::NEG_INFINITY;
        }
        m[2 * kv] = f16::NEG_INFINITY;
        assert_eq!(fully_masked_rows(&m, t, kv), 1, "row 1 only");
        assert_eq!(fully_masked_rows(&m, 1, kv), 0);
        // A causal mask never empties a row (key 0 is always visible).
        let causal: Vec<f16> = (0..t * kv)
            .map(|i| {
                if i % kv > i / kv {
                    f16::NEG_INFINITY
                } else {
                    f16::ZERO
                }
            })
            .collect();
        assert_eq!(fully_masked_rows(&causal, t, kv), 0);
    }

    #[test]
    fn flash_ref_emptied_row_averages_every_real_key() {
        // All keys masked: the reference's uniform row averages V over the real n_kv
        // keys (the device path would average over the padded Kn, 1.0x here).
        let (t, kv, dk, dv) = (1usize, 3usize, 32usize, 16usize);
        let q = vec![f16::ZERO; t * dk];
        let k = vec![f16::ZERO; kv * dk];
        let mut v = vec![f16::ZERO; dv * kv];
        for c in 0..dv {
            v[c * kv + c % kv] = f16::from_f32(3.0 * (c % kv + 1) as f32);
        }
        let mask = vec![f16::NEG_INFINITY; t * kv];
        let mut out = vec![f16::ZERO; dv];
        flash_attn_ref(
            t,
            kv,
            dk,
            dv,
            1,
            1,
            0.125,
            0.0,
            &q,
            &k,
            &v,
            Some(&mask),
            &mut out,
        )
        .unwrap();
        for (c, &got) in out.iter().enumerate() {
            let want = (c % kv + 1) as f32;
            assert!(
                (got.to_f32() - want).abs() < 1e-3,
                "col {c}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn pad_rows_zero_pads_the_tail() {
        let src: Vec<f16> = (0..6).map(|i| f16::from_f32(i as f32)).collect();
        let dst = pad_rows(&src, 2, 3, 8);
        assert_eq!(&dst[..6], &src[..]);
        assert!(dst[6..].iter().all(|&v| v == f16::ZERO));
    }

    #[test]
    fn add_bias_broadcasts_and_rounds_once() {
        let b: Vec<f16> = vec![0.5, -0.25].into_iter().map(f16::from_f32).collect();
        let mut c: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        add_bias(&mut c, Some(&b), 2, 2);
        let want: Vec<f16> = vec![1.5, 1.75, 3.5, 3.75]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(c, want);
        // A None bias leaves the buffer alone.
        let mut d = c.clone();
        add_bias(&mut d, None, 2, 2);
        assert_eq!(d, c);
    }

    #[test]
    fn gather_helpers_slice_and_transpose_one_head() {
        // src[2 rows][4 cols], head offset 1, width 2.
        let src: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        let mut rows = vec![f16::ONE; 8];
        gather_head_rows(&mut rows, &src, 2, 4, 1, 2);
        let want: Vec<f16> = [1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(rows, want);
        let mut cols = vec![f16::ONE; 8];
        gather_head_t_cols(&mut cols, &src, 2, 4, 1, 2, 4);
        // dst[c][t]: rows t = 0..2, cols c = 0..2, stride 4.
        assert_eq!(cols[0], f16::from_f32(1.0));
        assert_eq!(cols[1], f16::from_f32(5.0));
        assert_eq!(cols[4], f16::from_f32(2.0));
        assert_eq!(cols[5], f16::from_f32(6.0));
        assert!(cols[2..4].iter().chain(&cols[6..]).all(|&v| v == f16::ZERO));
    }

    #[test]
    fn scatter_head_cols_is_the_inverse_of_the_gather() {
        // One head's [2,2] context lands in columns 1..3 of a [2,4] matrix.
        let src: Vec<f16> = (0..4).map(|i| f16::from_f32(i as f32)).collect();
        let mut dst = vec![f16::ONE; 8];
        scatter_head_cols(&mut dst, &src, 2, 4, 1, 2);
        let want: Vec<f16> = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(dst, want);
        // Gathering that head back out reproduces the source.
        let mut back = vec![f16::ZERO; 4];
        gather_head_rows(&mut back, &dst, 2, 4, 1, 2);
        assert_eq!(back, src);
    }

    #[test]
    fn mha_self_shape_pads_and_rejects() {
        assert_eq!(mha_self_shape(1, 64, 4).unwrap(), (16, 4, 32));
        assert_eq!(mha_self_shape(5, 64, 4).unwrap(), (16, 8, 32));
        assert_eq!(mha_self_shape(33, 128, 2).unwrap(), (64, 36, 64));
        assert!(mha_self_shape(4, 64, 0).is_err());
        assert!(mha_self_shape(4, 0, 4).is_err());
        assert!(mha_self_shape(0, 64, 4).is_err());
        // d % n_head != 0 is the C's ROCKET_E_SHAPE.
        assert!(mha_self_shape(4, 64, 3).is_err());
    }

    #[test]
    fn mha_self_rejects_bad_buffers_before_any_device_access() {
        let d = 64;
        let t = 3;
        let x = vec![f16::ZERO; t * d];
        let w = vec![f16::ZERO; d * d];
        let mut out = vec![f16::ZERO; t * d];
        // A short weight matrix: the check must fire on a host with no NPU.
        assert!(
            mha_self(
                t,
                d,
                4,
                &x,
                &w[1..],
                None,
                &w,
                None,
                &w,
                None,
                &w,
                None,
                &mut out,
                1
            )
            .is_err()
        );
        // A short output buffer.
        assert!(
            mha_self(
                t,
                d,
                4,
                &x,
                &w,
                None,
                &w,
                None,
                &w,
                None,
                &w,
                None,
                &mut out[1..],
                1
            )
            .is_err()
        );
        // A short bias.
        assert!(
            mha_self(
                t,
                d,
                4,
                &x,
                &w,
                Some(&[f16::ZERO; 4]),
                &w,
                None,
                &w,
                None,
                &w,
                None,
                &mut out,
                1
            )
            .is_err()
        );
        // d % n_head != 0.
        assert!(
            mha_self(
                t, d, 3, &x, &w, None, &w, None, &w, None, &w, None, &mut out, 1
            )
            .is_err()
        );
    }

    /// Identity weights and no bias, one token: the only key takes all the weight, so
    /// the attention output is `x` and the output projection passes it through.
    #[test]
    fn mha_self_ref_is_the_identity_for_one_token() {
        let (t, d, n_head) = (1usize, 8usize, 2usize);
        let x: Vec<f16> = (0..t * d)
            .map(|i| f16::from_f32(i as f32 * 0.25 - 1.0))
            .collect();
        let mut eye = vec![f16::ZERO; d * d];
        for i in 0..d {
            eye[i * d + i] = f16::ONE;
        }
        let mut out = vec![f16::ZERO; t * d];
        mha_self_ref(
            t, d, n_head, &x, &eye, None, &eye, None, &eye, None, &eye, None, &mut out,
        )
        .unwrap();
        assert_eq!(out, x);
    }

    /// A zero output projection leaves only its bias — and `Wo = 0` with `bo = None`
    /// leaves zeros, which pins down both the projection and the bias placement.
    #[test]
    fn mha_self_ref_output_projection_bias_only() {
        let (t, d, n_head) = (3usize, 8usize, 2usize);
        let mut seed = 0xfeed_face_dead_beefu64;
        let x: Vec<f16> = (0..t * d).map(|_| rnd(&mut seed)).collect();
        let w = vec![f16::ZERO; d * d];
        let b: Vec<f16> = (0..d).map(|_| rnd(&mut seed)).collect();
        let mut eye = vec![f16::ZERO; d * d];
        for i in 0..d {
            eye[i * d + i] = f16::ONE;
        }
        let mut out = vec![f16::ONE; t * d];
        mha_self_ref(
            t,
            d,
            n_head,
            &x,
            &eye,
            None,
            &eye,
            None,
            &eye,
            None,
            &w,
            Some(&b),
            &mut out,
        )
        .unwrap();
        for row in out.chunks_exact(d) {
            assert_eq!(row, &b[..]);
        }
        let mut zero = vec![f16::ONE; t * d];
        mha_self_ref(
            t, d, n_head, &x, &eye, None, &eye, None, &eye, None, &w, None, &mut zero,
        )
        .unwrap();
        assert!(zero.iter().all(|&v| v == f16::ZERO));
    }

    #[test]
    fn mha_self_ref_rejects_short_buffers() {
        let (t, d, n_head) = (2usize, 8usize, 2usize);
        let x = vec![f16::ZERO; t * d];
        let w = vec![f16::ZERO; d * d];
        let mut out = vec![f16::ZERO; t * d];
        assert!(
            mha_self_ref(
                t,
                d,
                n_head,
                &x,
                &w,
                None,
                &w,
                None,
                &w,
                None,
                &w,
                None,
                &mut out[..t],
            )
            .is_err()
        );
        // `mha_self_ref` is otherwise reachable on a host with no NPU: a well-formed
        // call must return Ok (not a device error).
        assert!(
            mha_self_ref(
                t, d, n_head, &x, &w, None, &w, None, &w, None, &w, None, &mut out,
            )
            .is_ok()
        );
    }
}
