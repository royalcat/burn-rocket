//! Spatial and feature-axis reductions (RK3588): port of `rocket_reduce.c`.
//!
//! Two very different engines live here, because the hardware reduces two different
//! axes.
//!
//! # Spatial reduce over `[H, W]` — the PPU pooling engine
//!
//! `out[c] = (1/(H*W)) * sum_{h,w} in[c][h][w]` (GlobalAveragePool / TFLite `MEAN` over
//! axes `[1,2]` / ONNX `ReduceMean`, the squeeze of every SE block and most classifier
//! heads).
//!
//! The PPU's kernel and stride fields are 4-bit (value-1), so a single pool window is
//! capped at **16x16** — a 56x56 global pool cannot be one pool. This op therefore
//! **decomposes** the global mean into a chain of small non-overlapping pools and
//! telescopes them:
//!
//! ```text
//! mean over k1*k2*...*kn equal groups == (...((mean_k1) mean_k2) ...) mean_kn
//! ```
//!
//! because the average of equal-sized group-averages is the grand average. Each pass
//! tiles the current extent *exactly* (kernel divides the extent, stride == kernel,
//! pad == 0), so the product of the per-pass divisors is exactly `H*W`. Intermediates
//! stay resident in the NC1HWC2 cube between passes (no host round trip): the input is
//! scattered once and the `1x1xC` result is de-scattered once.
//!
//! ## Decomposability
//!
//! [`global_avgpool_plan`] reports the path up front. The NPU path needs:
//!
//! 1. each axis **16-smooth** — every prime factor `<= 16`, so it factors into kernels
//!    in `[2,16]` (covers every power of two and all real feature-map sizes
//!    7, 14, 28, 56, 49, 98, ...);
//! 2. `H` and `W` to have the **same number of factors**, so every pass pools both axes
//!    and neither collapses to 1 ahead of the other. That holds for every square map
//!    (`H == W`, the usual GlobalAvgPool case) and equal-count rectangles.
//!
//! Factors are applied **smallest-first** so the running quotient stays `>= 4` at every
//! step: a PPU-written intermediate cube with a spatial dim `< 4` is not read back
//! correctly by the *next chained* pass (an NPU-to-NPU chaining quirk; a standalone
//! sub-4 pool is fine). A non-16-smooth axis (a prime factor 17, 19, 23, ...) or an
//! unequal factor count falls back to the exact host reduction, which is what the C
//! does for `fd < 0` as well — so this module exposes no "no device" variant: the
//! device entry points are always `&Device`, the `*_ref` functions are the host path,
//! and a non-decomposable shape is silently (and correctly) computed by `*_ref`.
//!
//! ## Precision
//!
//! Each pass divides by its kernel via the PPU's per-axis fp16 reciprocal
//! `fp16(65536/k)` ([`crate::regcmd::pool::ppu_recip_kernel_fp16`]; there is no
//! hardware divider). A symmetric kernel uses that exact validated per-axis reciprocal;
//! an asymmetric kernel (only in the tail passes of a non-square map) splits the
//! divisor geometrically across the two axes ([`recip_pair`]). Both keep the product
//! `== 1/(kh*kw)` in fp16 terms. Net error is fp16 rounding plus ~1e-3 reciprocal
//! quantization per pass, so gate an average with a **relative** tolerance, never
//! equality. `Max`/`Min` are idempotent and need no reciprocal: the decomposed result
//! is **bit-exact** against the host reduction.
//!
//! # Feature-axis reduce over the hidden axis — a ones-matmul
//!
//! `out[m] = sum_h in[m][h]` (or `/H`), the sibling the PPU cannot provide: pooling
//! reduces the spatial axes *within* a channel and never crosses channels, while the
//! contraction the transformer needs (RMSNorm/LayerNorm sum-of-squares, the softmax
//! denominator) is a reduce over the matmul's K axis. So it comes free from the
//! validated matmul:
//!
//! ```text
//! out[M,1] = in[M,H] . ones[1,H]^T
//! ```
//!
//! The contraction uses the fp32-output path ([`crate::matmul::f32out::one_shot`]), so
//! the K-partials accumulate in genuine fp32/f64 rather than being narrowed to fp16 per
//! tile — that precision matters because this primitive feeds the norm variance. `N` is
//! padded to 16 (the matmul's N-group), so 16 identical ones-columns are computed and
//! column 0 is read back; that is a tiny fixed cost on a K-dominated shape.
//!
//! # Cumsum — the same matmul with a triangular ones weight
//!
//! A cumulative sum *is* a matmul by a triangular ones matrix; the feature reduce is
//! its `N = 1` all-ones-column special case. Widening that column to the full triangle
//! makes every prefix its own output column (`L[n][k] = 1` iff `k` is in prefix `n`).
//! The fp32 result is narrowed to fp16 on read-back, so cumsum is
//! fp16-rounding-accurate (a relative tolerance), not bit-exact.
//!
//! All three matmul-backed entries share the same alignment rules: `K % 32`, `N % 16`,
//! `M % 4`, with hosts staging a zero-padded copy only when the input does not already
//! meet them (`H % 32 == 0 && M % 4 == 0` is the common LLM case).
//!
//! # The int element-count guard
//!
//! The C library hands `M*N` to the activation/EW entries as an `int` and guards it
//! with `(size_t)M*N > INT_MAX` in the softmax family (a truncating product would
//! under-process the op and return a silently partial surface). This port applies the
//! same ceiling to every entry here, cumsum and the feature reduce included, because
//! their matmul dims are `int` as well.

use half::f16;

use crate::Device;
use crate::device::Bo;
use crate::device::uapi::{DRM_ROCKET_JOB_PPU_DONE, DrmRocketTask};
use crate::error::{Error, Result};
use crate::matmul::f32out;
use crate::regcmd::desc::NpuOp;
use crate::regcmd::pool::{PoolMethod, PoolParams, gen_pool_fp16, ppu_recip_kernel_fp16};

/// Largest input `H`/`W` this op accepts (the cube dim field is 13-bit; well above any
/// real feature map, but it bounds the factor buffers).
pub const REDUCE_MAX_DIM: usize = 8192;

/// Upper bound on `factors-of-H + factors-of-W` passes.
pub const REDUCE_MAX_PASS: usize = 24;

/// DMA burst over-read slack past the cube (the C `REDUCE_BO_SLACK`).
const BO_SLACK: usize = 32768;

/// fp16 feature-cube channel atom.
const C2: usize = 8;

/// Regcmd words one pool pass reserves (the C `uint64_t regs[64]`).
const RC_WORDS: usize = 64;

/// The PPU job wait (the C passes 2 s explicitly here rather than the default).
const PASS_WAIT_NS: u64 = 2_000_000_000;

// ── Cube layout ───────────────────────────────────────────────────────────────

/// The NC1HWC2 cube index of a 1-based channel/pixel (the C `feature_data`).
///
/// `C2` is the channel atom: `[C1][H][W][C2]`, with the in-plane chroma stride
/// innermost. `c`, `row` and `col` are 1-based, exactly like the C helper, and `channels`
/// is the total channel count — unused for the index itself (only `c2` is), kept for
/// signature symmetry with the C, which does the same `(void)C`.
#[inline]
pub fn feature_data(
    channels: usize,
    pix_h: usize,
    pix_w: usize,
    c2: usize,
    ch: usize,
    row: usize,
    col: usize,
) -> usize {
    let _ = channels;
    let plane = (ch - 1) / c2;
    let offset = (ch - 1) % c2;
    plane * pix_h * pix_w * c2 + c2 * ((row - 1) * pix_w + (col - 1)) + offset
}

/// Scatter a row-major `[C][H][W]` feature into `cube` (which must be zeroed and at
/// least `C1*H*W*C2` long).
fn scatter_feature(cube: &mut [f16], c: usize, h: usize, w: usize, input: &[f16]) {
    for ch in 0..c {
        for row in 0..h {
            for col in 0..w {
                let dst = feature_data(c, h, w, C2, ch + 1, row + 1, col + 1);
                cube[dst] = input[(ch * h + row) * w + col];
            }
        }
    }
}

/// The `1x1xC` de-scatter: the channel values at pixel `(1, 1)`.
fn descatter_feature(cube: &[f16], c: usize, h: usize, w: usize, out: &mut [f16]) {
    for ch in 0..c {
        out[ch] = cube[feature_data(c, h, w, C2, ch + 1, 1, 1)];
    }
}

// ── Factorisation + plan ──────────────────────────────────────────────────────

/// `rocket_reduce_factor_axis`: decompose one axis into pooling factors.
///
/// Writes the axis's factors, **ascending**, into `f` and returns the count. Each
/// factor is in `[2, 16]` and their product is `n`; `n == 1` gives **0** factors (no
/// pooling on that axis). A non-16-smooth axis (a prime factor `> 16`) is
/// [`Error::shape`] — the C's `-1` — as is an axis smaller than 1 or a too-small `f`.
///
/// The C factorizes greedily from 16 down (largest factor first, so the axis shrinks
/// fastest), then sorts ascending. Ascending is load-bearing: after pass `i` the
/// running extent equals the product of the remaining (larger) factors, hence is at
/// least the largest factor — for any axis `> 16` that is `>= 4`, so no intermediate
/// spatial dim is `< 4` (see the module docs on the chained sub-4 quirk).
pub fn reduce_factor_axis(n: usize, f: &mut [usize]) -> Result<usize> {
    if n < 1 {
        return Err(Error::shape("rocket_reduce_factor_axis"));
    }
    let mut n = n;
    let mut c = 0usize;
    while n > 16 {
        // Largest divisor <= 16 (a composite factor keeps the count, and so the pass
        // count, small: 56 -> 14, 4 rather than 7, 4, 2).
        let d = (2..=16).rev().find(|&x| n.is_multiple_of(x));
        let Some(d) = d else {
            return Err(Error::shape("rocket_reduce_factor_axis/16-smooth")
                .with_detail(format!("{n} has a prime factor > 16")));
        };
        if c >= f.len() {
            return Err(Error::shape("rocket_reduce_factor_axis/cap"));
        }
        f[c] = d;
        c += 1;
        n /= d;
    }
    if n >= 2 {
        if c >= f.len() {
            return Err(Error::shape("rocket_reduce_factor_axis/cap"));
        }
        f[c] = n;
        c += 1;
    }
    // Ascending: smallest kernel first, so the running quotient stays large.
    f[..c].sort_unstable();
    Ok(c)
}

/// The C `rocket_global_avgpool_plan` **reason code**, verbatim.
///
/// `0` = the shape runs entirely on the NPU; otherwise, in the C's check order:
/// `-1` bad shape, `-3` a dim past [`REDUCE_MAX_DIM`], `-5` a channel count the 13-bit
/// cube field cannot encode, `-4` `H` not 16-smooth, `-5` `W` not 16-smooth, `-6`
/// unequal factor counts.
///
/// The C's `-4`/`-5`/`-6` are **local** reasons that happen to collide numerically with
/// `ROCKET_E_DEVICE`/`ROCKET_E_UNSUPPORTED`; they are kept anyway so a caller porting
/// the C's branch (or the parity harness comparing `pool C H W`) lines up exactly.
/// [`global_avgpool_plan`] is the typed form of the same decision.
pub fn global_avgpool_plan_code(c: usize, h: usize, w: usize) -> i32 {
    if c < 1 || h < 1 || w < 1 {
        return -1;
    }
    if h > REDUCE_MAX_DIM || w > REDUCE_MAX_DIM {
        return -3;
    }
    if c > 8192 {
        return -5;
    }
    let mut fh = [0usize; REDUCE_MAX_PASS];
    let mut fw = [0usize; REDUCE_MAX_PASS];
    let nh = match reduce_factor_axis(h, &mut fh) {
        Ok(n) => n,
        Err(_) => return -4, // H not 16-smooth
    };
    let nw = match reduce_factor_axis(w, &mut fw) {
        Ok(n) => n,
        Err(_) => return -5, // W not 16-smooth
    };
    if nh != nw {
        return -6; // every pass must pool both axes (see the module docs)
    }
    0
}

/// `rocket_global_avgpool_plan`: which path a `(C, H, W)` spatial reduce takes.
///
/// `Ok(())` means the shape runs entirely on the NPU (both axes 16-smooth, `H` and `W`
/// the same factor count, dims in range). A negative C result is **not** an error for
/// the runtime — it means "exact host fallback", which [`global_avgpool`] performs
/// itself — but it lets a caller (a TFLite delegate, say) decide up front.
///
/// The failure classes keep the C's: `Shape` for the C `-4`/`-5` (not 16-smooth) and
/// `-6` (unequal factor count) cases, `NoMem` past [`REDUCE_MAX_DIM`], `Unsupported`
/// for a channel count the 13-bit cube field cannot encode. The C reason itself is
/// [`global_avgpool_plan_code`].
pub fn global_avgpool_plan(c: usize, h: usize, w: usize) -> Result<()> {
    match global_avgpool_plan_code(c, h, w) {
        0 => Ok(()),
        -3 => Err(Error::nomem("rocket_global_avgpool_plan/dim")),
        // -5 is both "channels too many" and "W not 16-smooth" in the C; the channel
        // check runs first, so `c > 8192` disambiguates.
        -5 if c > 8192 => Err(Error::unsupported("rocket_global_avgpool_plan/channels")),
        code => {
            Err(Error::shape("rocket_global_avgpool_plan").with_detail(format!("reason {code}")))
        }
    }
}

/// The factors of both axes plus the planned pass schedule (the C local arrays).
struct Schedule {
    fh: [usize; REDUCE_MAX_PASS],
    fw: [usize; REDUCE_MAX_PASS],
    nh: usize,
    nw: usize,
}

impl Schedule {
    fn new(h: usize, w: usize) -> Result<Schedule> {
        let mut s = Schedule {
            fh: [0; REDUCE_MAX_PASS],
            fw: [0; REDUCE_MAX_PASS],
            nh: 0,
            nw: 0,
        };
        s.nh = reduce_factor_axis(h, &mut s.fh)?;
        s.nw = reduce_factor_axis(w, &mut s.fw)?;
        Ok(s)
    }

    fn passes(&self) -> usize {
        self.nh.max(self.nw)
    }

    fn kernels(&self, i: usize) -> (usize, usize) {
        (
            if i < self.nh { self.fh[i] } else { 1 },
            if i < self.nw { self.fw[i] } else { 1 },
        )
    }
}

/// Per-axis average reciprocals for one pass.
///
/// A symmetric kernel uses the exact validated per-axis `fp16(65536/k)`; an asymmetric
/// kernel (the tail passes of a non-square map) splits the divisor geometrically so
/// that `recip_h * recip_w * 2^-32` still equals `1/(kh*kw)` with both reciprocals
/// inside fp16 range.
pub fn recip_pair(kh: usize, kw: usize) -> (u32, u32) {
    if kh == kw {
        let r = ppu_recip_kernel_fp16(kh as u32);
        return (r, r);
    }
    let h = f16::from_f64(65536.0 / ((kh as f64) * (kw as f64)).sqrt());
    let b = h.to_bits() as u32;
    (b, b)
}

// ── Host references ───────────────────────────────────────────────────────────

/// `rocket_global_avgpool_ref_fp16`: fp64-accumulate per-channel spatial mean.
pub fn global_avgpool_ref(c: usize, h: usize, w: usize, input: &[f16], out: &mut [f16]) {
    assert!(input.len() >= c * h * w, "input shorter than C*H*W");
    assert!(out.len() >= c, "output shorter than C");
    let inv = 1.0 / (h as f64 * w as f64);
    for ch in 0..c {
        let p = &input[ch * h * w..ch * h * w + h * w];
        let mut acc = 0.0f64;
        for &v in p {
            acc += v.to_f64();
        }
        out[ch] = f16::from_f64(acc * inv);
    }
}

/// `rocket_global_maxpool_ref_fp16`: exact per-channel spatial maximum.
///
/// The comparison is `>` seeded from `p[0]`, exactly as the C: a NaN anywhere leaves
/// the running value untouched.
pub fn global_maxpool_ref(c: usize, h: usize, w: usize, input: &[f16], out: &mut [f16]) {
    assert!(input.len() >= c * h * w, "input shorter than C*H*W");
    assert!(out.len() >= c, "output shorter than C");
    for ch in 0..c {
        let p = &input[ch * h * w..ch * h * w + h * w];
        let mut m = p[0];
        for &v in &p[1..] {
            if v > m {
                m = v;
            }
        }
        out[ch] = m;
    }
}

/// `rocket_global_minpool_ref_fp16`: exact per-channel spatial minimum.
pub fn global_minpool_ref(c: usize, h: usize, w: usize, input: &[f16], out: &mut [f16]) {
    assert!(input.len() >= c * h * w, "input shorter than C*H*W");
    assert!(out.len() >= c, "output shorter than C");
    for ch in 0..c {
        let p = &input[ch * h * w..ch * h * w + h * w];
        let mut m = p[0];
        for &v in &p[1..] {
            if v < m {
                m = v;
            }
        }
        out[ch] = m;
    }
}

/// `rocket_reduce_feature_ref_fp16`: fp64-accumulate per-row sum (or mean).
pub fn reduce_feature_ref(m: usize, h: usize, input: &[f16], out: &mut [f32], mean: bool) {
    assert!(input.len() >= m * h, "input shorter than M*H");
    assert!(out.len() >= m, "output shorter than M");
    let inv = if mean { 1.0 / h as f64 } else { 1.0 };
    for row in 0..m {
        let p = &input[row * h..row * h + h];
        let mut acc = 0.0f64;
        for &v in p {
            acc += v.to_f64();
        }
        out[row] = (acc * inv) as f32;
    }
}

/// `rocket_cumsum_ref_fp16`: exact fp64 prefix/suffix scan.
///
/// `exclusive` drops the current element from the sum at its own position; `reverse`
/// grows the running total from the right. The running accumulator is fp64 and only the
/// written value narrows, so the reference is the exact scan.
pub fn cumsum_ref(
    m: usize,
    n: usize,
    input: &[f16],
    out: &mut [f16],
    exclusive: bool,
    reverse: bool,
) {
    assert!(input.len() >= m * n, "input shorter than M*N");
    assert!(out.len() >= m * n, "output shorter than M*N");
    for row in 0..m {
        let xp = &input[row * n..row * n + n];
        let op = &mut out[row * n..row * n + n];
        let mut acc = 0.0f64;
        if !reverse {
            for j in 0..n {
                if exclusive {
                    op[j] = f16::from_f64(acc);
                    acc += xp[j].to_f64();
                } else {
                    acc += xp[j].to_f64();
                    op[j] = f16::from_f64(acc);
                }
            }
        } else {
            for j in (0..n).rev() {
                if exclusive {
                    op[j] = f16::from_f64(acc);
                    acc += xp[j].to_f64();
                } else {
                    acc += xp[j].to_f64();
                    op[j] = f16::from_f64(acc);
                }
            }
        }
    }
}

// ── Device: the PPU telescoping pool chain ────────────────────────────────────

fn check_pool_io(
    name: &'static str,
    c: usize,
    h: usize,
    w: usize,
    input: &[f16],
    out: &[f16],
) -> Result<()> {
    if c < 1 || h < 1 || w < 1 {
        return Err(Error::shape(name));
    }
    if input.len() < c * h * w || out.len() < c {
        return Err(Error::shape(name).with_detail(format!(
            "in {} (need {}), out {} (need {c})",
            input.len(),
            c * h * w,
            out.len()
        )));
    }
    Ok(())
}

/// The shared telescoping engine behind the three global spatial pools.
///
/// `method` selects the PPU op; average additionally builds each pass's reciprocals.
/// The BO layout, the scatter, the ping-pong schedule, the 2 s per-pass wait and the
/// submit flag all mirror the C (`run_global_extrema_pool` shares this with the average
/// path there too, minus the reciprocals).
fn run_global_pool(
    dev: &Device,
    c: usize,
    h: usize,
    w: usize,
    method: PoolMethod,
    input: &[f16],
    out: &mut [f16],
) -> Result<()> {
    let sched = Schedule::new(h, w)?;
    let npass = sched.passes();

    let c1 = c.div_ceil(C2);
    let cube_bytes = c1 * h * w * C2 * std::mem::size_of::<f16>() + BO_SLACK;

    let _guard = dev.alloc_bo(4096)?; // keeps IOVA 0 unused, as in C
    let mut a = dev.alloc_bo(cube_bytes)?;
    let mut b = dev.alloc_bo(cube_bytes)?;
    let mut rc = dev.alloc_bo(RC_WORDS * 8)?;
    crate::matmul::bos::check_iova(&[&a, &b, &rc])?;

    // Scatter the input feature [C][H][W] into the NC1HWC2 cube in A.
    a.prep(0)?;
    {
        let words = bo_words_mut(&mut a);
        words.fill(f16::ZERO);
        scatter_feature(words, c, h, w, input);
    }
    a.fini()?;

    // Telescoping passes, ping-ponging A <-> B on the device.
    let mut ch = h;
    let mut cw = w;
    let mut ran_pass = false;
    let mut last_dst_is_a = false;
    let mut ops: Vec<NpuOp> = Vec::with_capacity(RC_WORDS);

    for i in 0..npass {
        let (kh, kw) = sched.kernels(i);
        if kh == 1 && kw == 1 {
            continue; // nothing to do this pass
        }
        ran_pass = true;
        let oh = ch / kh; // exact tiling (kernel divides the extent)
        let ow = cw / kw;

        let (src, dst) = if i & 1 == 1 {
            (&mut b, &mut a)
        } else {
            (&mut a, &mut b)
        };
        last_dst_is_a = i & 1 == 1;

        let (rh, rw) = if method == PoolMethod::Avg {
            recip_pair(kh, kw)
        } else {
            (0, 0)
        };
        let p = PoolParams {
            c: c as u16,
            ih: ch as u16,
            iw: cw as u16,
            oh: oh as u16,
            ow: ow as u16,
            kh: kh as u8,
            kw: kw as u8,
            stride_y: kh as u8, // non-overlapping tile
            stride_x: kw as u8,
            pad_top: 0,
            pad_left: 0,
            pad_bottom: 0,
            pad_right: 0,
            method,
            recip_w: rw,
            recip_h: rh,
            input_dma: src.dma_address() as u32,
            output_dma: dst.dma_address() as u32,
        };
        ops.clear();
        let count = gen_pool_fp16(&p, &mut ops)?;
        if count > RC_WORDS {
            return Err(Error::tiling("rocket_global_avgpool/regcmd"));
        }
        rc.prep(0)?;
        write_program(&mut rc, &ops)?;
        rc.fini()?;

        let task = DrmRocketTask {
            regcmd: rc.dma_address() as u32,
            regcmd_count: count as u32,
        };
        // This program is a POOL: it finishes on the PPU, not the DPU. Naming the block
        // costs a driver without PPU-done support nothing (it skips the flag) and is
        // what lets a driver that waits for the named block retire the job at all.
        let flags = if dev.ppu_done_supported() {
            DRM_ROCKET_JOB_PPU_DONE
        } else {
            0
        };
        dev.submit(
            &[task],
            &[src.handle(), rc.handle()],
            &[dst.handle()],
            flags,
        )?;
        // Wait for this pass to land before it feeds the next (RAW chain).
        //
        // Deliberately unpaired, as in C: `prep` both waits and hands `dst` to the CPU,
        // and the next iteration sends the same BO back as `src` with no `fini` between.
        // That is safe only because nothing on the CPU writes it in between — there are
        // no dirty lines for a device sync to flush. Closing the bracket would cost a
        // second full-BO cache walk per pass for nothing.
        dst.prep(PASS_WAIT_NS)?;

        ch = oh;
        cw = ow;
    }

    // A degenerate 1x1 reduce never ran a pass, so the result BO (A) is still in its
    // post-write, device-owned state: invalidate it for the CPU read here.
    if !ran_pass {
        a.prep(PASS_WAIT_NS)?;
    }
    let final_cube: &[f16] = if !ran_pass || last_dst_is_a {
        bo_words(&a)
    } else {
        bo_words(&b)
    };
    descatter_feature(final_cube, c, ch, cw, out);
    Ok(())
}

/// `rocket_global_avgpool_fp16`: per-channel spatial mean over `[H, W]`.
///
/// Non-decomposable shapes are computed by the exact host reduction
/// ([`global_avgpool_ref`]) and return `Ok`, exactly like the C's `fd < 0` path; there
/// is no separate "no device" variant.
pub fn global_avgpool(
    dev: &Device,
    c: usize,
    h: usize,
    w: usize,
    input: &[f16],
    out: &mut [f16],
) -> Result<()> {
    check_pool_io("rocket_global_avgpool_fp16", c, h, w, input, out)?;
    if global_avgpool_plan(c, h, w).is_err() {
        global_avgpool_ref(c, h, w, input, out);
        return Ok(());
    }
    run_global_pool(dev, c, h, w, PoolMethod::Avg, input, out)
}

/// `rocket_global_maxpool_fp16`: per-channel spatial maximum over `[H, W]`.
///
/// A decomposable shape is **bit-exact** against [`global_maxpool_ref`]; a
/// non-decomposable one falls back to that reference.
pub fn global_maxpool(
    dev: &Device,
    c: usize,
    h: usize,
    w: usize,
    input: &[f16],
    out: &mut [f16],
) -> Result<()> {
    check_pool_io("rocket_global_maxpool_fp16", c, h, w, input, out)?;
    if global_avgpool_plan(c, h, w).is_err() {
        global_maxpool_ref(c, h, w, input, out);
        return Ok(());
    }
    run_global_pool(dev, c, h, w, PoolMethod::Max, input, out)
}

/// `rocket_global_minpool_fp16`: per-channel spatial minimum over `[H, W]`.
pub fn global_minpool(
    dev: &Device,
    c: usize,
    h: usize,
    w: usize,
    input: &[f16],
    out: &mut [f16],
) -> Result<()> {
    check_pool_io("rocket_global_minpool_fp16", c, h, w, input, out)?;
    if global_avgpool_plan(c, h, w).is_err() {
        global_minpool_ref(c, h, w, input, out);
        return Ok(());
    }
    run_global_pool(dev, c, h, w, PoolMethod::Min, input, out)
}

// ── Device: the matmul-backed reductions ──────────────────────────────────────

/// The element-count ceiling shared by every entry in this module (see the module docs).
fn guard_elems(name: &'static str, elems: usize) -> Result<()> {
    if elems > i32::MAX as usize {
        return Err(Error::shape(name).with_detail(format!("{elems} elements > INT_MAX")));
    }
    Ok(())
}

/// `rocket_reduce_feature_fp16`: per-row sum (or mean) over the hidden axis.
///
/// `out` is `f32` so a large sum (an unclipped sum-of-squares, say) does not clip fp16
/// range; the fp16 *input* elements must still be finite. Any `H > 0` is accepted (H and
/// M are zero-padded to the matmul's `K % 32` / `M % 4` alignment internally).
pub fn reduce_feature(
    dev: &Device,
    m: usize,
    h: usize,
    input: &[f16],
    out: &mut [f32],
    mean: bool,
) -> Result<()> {
    if m < 1 || h < 1 {
        return Err(Error::shape("rocket_reduce_feature_fp16"));
    }
    if input.len() < m * h || out.len() < m {
        return Err(Error::shape("rocket_reduce_feature_fp16/buffers"));
    }
    guard_elems("rocket_reduce_feature_fp16", m * h)?;

    const N: usize = 16;
    let kpad = (h + 31) & !31;
    let mpad = (m + 3) & !3;

    // Ones weight [N, Kpad]: 1 over the real H columns, 0 over the K pad.
    let mut ones = vec![f16::ZERO; N * kpad];
    for r in 0..N {
        let row = &mut ones[r * kpad..(r + 1) * kpad];
        row[..h].fill(f16::ONE);
    }
    let mut cbuf = vec![0.0f32; mpad * N];

    // A is `input` directly when it already meets the alignment (the common LLM case:
    // H % 32 == 0, M % 4 == 0 — no host copy); else a zero-padded [Mpad, Kpad] staging copy.
    let apad;
    let a: &[f16] = if kpad != h || mpad != m {
        let mut buf = vec![f16::ZERO; mpad * kpad];
        for row in 0..m {
            buf[row * kpad..row * kpad + h].copy_from_slice(&input[row * h..row * h + h]);
        }
        apad = buf;
        &apad
    } else {
        &input[..m * h]
    };

    f32out::one_shot(dev, mpad, kpad, N, a, &ones, &mut cbuf)?;
    let inv = if mean { 1.0f32 / h as f32 } else { 1.0 };
    for row in 0..m {
        out[row] = cbuf[row * N] * inv; // column 0
    }
    Ok(())
}

/// `rocket_cumsum_fp16`: per-row prefix (or suffix) sum over the last axis.
///
/// The triangular ones weight selects the prefix columns; `Npad` output columns are
/// computed and the real `N` narrowed back to fp16.
pub fn cumsum(
    dev: &Device,
    m: usize,
    n: usize,
    input: &[f16],
    out: &mut [f16],
    exclusive: bool,
    reverse: bool,
) -> Result<()> {
    if m < 1 || n < 1 {
        return Err(Error::shape("rocket_cumsum_fp16"));
    }
    if input.len() < m * n || out.len() < m * n {
        return Err(Error::shape("rocket_cumsum_fp16/buffers"));
    }
    guard_elems("rocket_cumsum_fp16", m * n)?;

    let kpad = (n + 31) & !31;
    let npad = (n + 15) & !15;
    let mpad = (m + 3) & !3;

    // Zero pad rows and columns; only the real [N,N] block is set.
    let mut l = vec![f16::ZERO; npad * kpad];
    for row in 0..n {
        let lrow = &mut l[row * kpad..(row + 1) * kpad];
        for (k, cell) in lrow.iter_mut().enumerate().take(n) {
            let one = if reverse {
                if exclusive { k > row } else { k >= row }
            } else if exclusive {
                k < row
            } else {
                k <= row
            };
            *cell = if one { f16::ONE } else { f16::ZERO };
        }
    }
    let mut cbuf = vec![0.0f32; mpad * npad];

    let apad;
    let a: &[f16] = if kpad != n || mpad != m {
        let mut buf = vec![f16::ZERO; mpad * kpad];
        for row in 0..m {
            buf[row * kpad..row * kpad + n].copy_from_slice(&input[row * n..row * n + n]);
        }
        apad = buf;
        &apad
    } else {
        &input[..m * n]
    };

    f32out::one_shot(dev, mpad, kpad, npad, a, &l, &mut cbuf)?;
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] = f16::from_f32(cbuf[row * npad + col]); // narrow fp32 -> fp16
        }
    }
    Ok(())
}

// ── BO helpers ────────────────────────────────────────────────────────────────

/// The BO's words as `f16`s (the mapping is page-aligned, `f16` is 2-byte aligned).
fn bo_words(bo: &Bo) -> &[f16] {
    let bytes = bo.as_slice();
    // SAFETY: f16 is repr(transparent) over u16 with align 2 and every prefix of the
    // mapping is initialized.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f16>(), bytes.len() / 2) }
}

/// The BO's words as mutable `f16`s.
fn bo_words_mut(bo: &mut Bo) -> &mut [f16] {
    let len = bo.len();
    let ptr = bo.as_ptr();
    // SAFETY: exclusive borrow of the BO, page-aligned mapping, 2-byte elements.
    unsafe { std::slice::from_raw_parts_mut(ptr.cast::<f16>(), len / 2) }
}

/// Write a program into `bo` from offset 0.
fn write_program(bo: &mut Bo, ops: &[NpuOp]) -> Result<()> {
    let bytes = bo.as_mut_slice();
    if ops.len() * 8 > bytes.len() {
        return Err(Error::tiling("write_program"));
    }
    for (i, op) in ops.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&op.0.to_ne_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factors(n: usize) -> Vec<usize> {
        let mut f = [0usize; REDUCE_MAX_PASS];
        let c = reduce_factor_axis(n, &mut f).expect("decomposable");
        f[..c].to_vec()
    }

    #[test]
    fn factor_axis_is_16_smooth_ascending() {
        assert_eq!(factors(1), Vec::<usize>::new());
        assert_eq!(factors(7), vec![7]);
        assert_eq!(factors(14), vec![14]); // already <= 16: one factor
        assert_eq!(factors(16), vec![16]);
        assert_eq!(factors(28), vec![2, 14]);
        assert_eq!(factors(49), vec![7, 7]);
        assert_eq!(factors(56), vec![4, 14]); // greedy from 16 down, then sorted
        assert_eq!(factors(64), vec![4, 16]);
        assert_eq!(factors(32), vec![2, 16]); // 2*16, not 2*2*2*2*2
        for n in [2usize, 3, 5, 7, 11, 13, 14, 16, 28, 56, 98, 128, 1024, 4096] {
            let f = factors(n);
            assert_eq!(f.iter().product::<usize>(), n, "{n}: product");
            assert!(f.iter().all(|v| (2..=16).contains(v)), "{n}: range");
            let mut sorted = f.clone();
            sorted.sort_unstable();
            assert_eq!(f, sorted, "{n}: ascending");
        }
        // 17 is prime > 16: refused (the C's -1).
        assert_eq!(
            reduce_factor_axis(17, &mut [0; 8]).unwrap_err().status(),
            crate::error::Status::Shape
        );
        // 17*17 too, at the first factor.
        assert!(reduce_factor_axis(289, &mut [0; 8]).is_err());
        // 19, 23, 2*17.
        for n in [19usize, 23, 34] {
            assert!(reduce_factor_axis(n, &mut [0; 8]).is_err(), "{n}");
        }
        assert!(reduce_factor_axis(0, &mut [0; 8]).is_err());
        // A too-small factor buffer is a shape error, not a truncation.
        assert!(reduce_factor_axis(56, &mut [0; 1]).is_err());
    }

    #[test]
    fn plan_decisions() {
        // Square maps: identical factor lists, always decomposable.
        for s in [1usize, 7, 14, 16, 28, 56, 49, 98, 224] {
            assert!(global_avgpool_plan(8, s, s).is_ok(), "{s}x{s}");
        }
        // Equal-count rectangles are decomposable (the tail pass is asymmetric).
        assert!(global_avgpool_plan(16, 28, 20).is_ok()); // [2,14] vs [2,10]
        assert!(global_avgpool_plan(8, 16, 8).is_ok()); // [16] vs [8]
        // Unequal counts fall back (the C's -6): 56 -> [4,14] but 8 -> [8].
        let e = global_avgpool_plan(16, 56, 8).unwrap_err();
        assert_eq!(e.status(), crate::error::Status::Shape);
        // Not 16-smooth falls back (-4 / -5).
        assert!(global_avgpool_plan(8, 17, 17).is_err());
        assert!(global_avgpool_plan(8, 7, 19).is_err());
        // Bounds and channel field.
        assert_eq!(
            global_avgpool_plan(8, REDUCE_MAX_DIM + 1, 8)
                .unwrap_err()
                .status(),
            crate::error::Status::NoMem
        );
        assert_eq!(
            global_avgpool_plan(8193, 8, 8).unwrap_err().status(),
            crate::error::Status::Unsupported
        );
        assert!(global_avgpool_plan(8192, 8, 8).is_ok());
        assert!(global_avgpool_plan(0, 8, 8).is_err());
    }

    #[test]
    fn feature_data_round_trips() {
        // The C feature_data(C,H,W,8,c,h,w) with 1-based c,h,w.
        let (c, h, w) = (19usize, 5usize, 7usize);
        let c1 = c.div_ceil(C2);
        let mut cube = vec![f16::ZERO; c1 * h * w * C2];
        let input: Vec<f16> = (0..c * h * w).map(|i| f16::from_f32(i as f32)).collect();
        scatter_feature(&mut cube, c, h, w, &input);
        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let idx = feature_data(c, h, w, C2, ch + 1, row + 1, col + 1);
                    assert_eq!(cube[idx], input[(ch * h + row) * w + col]);
                }
            }
        }
        // Every slot the scatter writes is unique, and the untouched pad channel of the
        // last C-plane stays zero.
        let mut seen = vec![false; cube.len()];
        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let idx = feature_data(c, h, w, C2, ch + 1, row + 1, col + 1);
                    assert!(!seen[idx], "duplicate slot {idx}");
                    seen[idx] = true;
                }
            }
        }
        assert_eq!(seen.iter().filter(|v| **v).count(), c * h * w);
        for ch in c..c1 * C2 {
            assert_eq!(cube[feature_data(c, h, w, C2, ch + 1, 1, 1)], f16::ZERO);
        }
        // The 1x1xC de-scatter reads back the input's spatial pixel (1,1).
        let mut out = vec![f16::ZERO; c];
        descatter_feature(&cube, c, h, w, &mut out);
        for ch in 0..c {
            assert_eq!(out[ch], input[ch * h * w]);
        }
        // It matches the matmul layout helper for the same geometry.
        assert_eq!(
            feature_data(c, h, w, C2, 3, 2, 4),
            crate::matmul::layout::feat_idx(h * w, 3, w + (4 - 1) + 1)
        );
    }

    #[test]
    fn references_on_known_rows() {
        // Average: a constant channel averages to itself; fp64 accumulate.
        let mut x = vec![f16::from_f32(0.25); 3 * 4 * 4];
        x[..16].fill(f16::from_f32(1.5));
        let mut out = vec![f16::ZERO; 3];
        global_avgpool_ref(3, 4, 4, &x, &mut out);
        assert_eq!(out[0], f16::from_f32(1.5));
        assert_eq!(out[1], f16::from_f32(0.25));

        // Max/min pick the extrema per channel, ignoring NaN (the C's `>` seed).
        let mut y: Vec<f16> = (0..12).map(|i| f16::from_f32(i as f32 - 4.0)).collect();
        y[1] = f16::NAN;
        y[5] = f16::from_f32(100.0);
        y[7] = f16::from_f32(-100.0);
        let mut mx = vec![f16::ZERO; 2];
        let mut mn = vec![f16::ZERO; 2];
        global_maxpool_ref(2, 2, 3, &y, &mut mx);
        global_minpool_ref(2, 2, 3, &y, &mut mn);
        assert_eq!(mx[0], f16::from_f32(100.0));
        assert_eq!(mn[0], f16::from_f32(-4.0)); // NaN at index 1 never wins
        assert_eq!(mx[1], f16::from_f32(7.0));
        assert_eq!(mn[1], f16::from_f32(-100.0));

        // Feature reduce: sum and mean, in f32 out.
        let z: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        let mut s = vec![0.0f32; 2];
        reduce_feature_ref(2, 4, &z, &mut s, false);
        assert_eq!(s, vec![6.0, 22.0]);
        reduce_feature_ref(2, 4, &z, &mut s, true);
        assert_eq!(s, vec![1.5, 5.5]);
    }

    #[test]
    fn cumsum_reference_covers_all_four_variants() {
        let x: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let expect = |exclusive: bool, reverse: bool| -> Vec<f32> {
            let mut out = vec![f16::ZERO; 4];
            cumsum_ref(1, 4, &x, &mut out, exclusive, reverse);
            out.iter().map(|v| v.to_f32()).collect()
        };
        assert_eq!(expect(false, false), vec![1.0, 3.0, 6.0, 10.0]); // inclusive forward
        assert_eq!(expect(true, false), vec![0.0, 1.0, 3.0, 6.0]); // exclusive forward
        assert_eq!(expect(false, true), vec![10.0, 9.0, 7.0, 4.0]); // inclusive reverse
        assert_eq!(expect(true, true), vec![9.0, 7.0, 4.0, 0.0]); // exclusive reverse
        // Two rows are independent.
        let y: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        let mut out = vec![f16::ZERO; 8];
        cumsum_ref(2, 4, &y, &mut out, false, false);
        let got: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, vec![0.0, 1.0, 3.0, 6.0, 4.0, 9.0, 15.0, 22.0]);
    }

    #[test]
    fn reciprocal_pairs_keep_the_divisor_product() {
        // Symmetric: both axes get the validated per-axis fp16(65536/k).
        for k in 2..=16usize {
            let (rh, rw) = recip_pair(k, k);
            assert_eq!(rh, ppu_recip_kernel_fp16(k as u32));
            assert_eq!(rw, rh);
        }
        // Asymmetric: both get fp16(65536/sqrt(kh*kw)), which keeps the product at
        // 1/(kh*kw) in the 2^-32 convention while staying inside fp16 range.
        for (kh, kw) in [(14usize, 10usize), (2, 7), (16, 2), (3, 5)] {
            let (rh, rw) = recip_pair(kh, kw);
            assert_eq!(rh, rw);
            let v = f16::from_bits(rh as u16).to_f32() as f64;
            let want = 1.0 / (kh as f64 * kw as f64);
            let got = (v * v) / (65536.0 * 65536.0);
            // fp16 rounding of the geometrically split reciprocal: ~1e-3 relative.
            assert!(
                (got - want).abs() / want < 6e-3,
                "{kh}x{kw}: {got} vs {want}"
            );
        }
        // The symmetric case is exact enough that the product error is ~1e-3.
        let (rh, rw) = recip_pair(7, 7);
        let v = f16::from_bits(rh as u16).to_f32() as f64;
        let got = (v * v) / (65536.0 * 65536.0);
        assert!((got - 1.0 / 49.0).abs() / (1.0 / 49.0) < 2e-3);
        assert_eq!(rw, rh);
    }

    #[test]
    fn schedule_never_collapses_below_four() {
        // Ascending factors keep every PPU-written intermediate >= 4.
        for (h, w) in [(56usize, 56usize), (49, 49), (64, 64), (28, 20), (7, 7)] {
            let s = Schedule::new(h, w).unwrap();
            let (mut ch, mut cw) = (h, w);
            for i in 0..s.passes() {
                let (kh, kw) = s.kernels(i);
                ch /= kh;
                cw /= kw;
                if i + 1 < s.passes() || ch > 1 {
                    assert!(ch >= 4, "{h}x{w} pass {i}: H -> {ch}");
                    assert!(cw >= 4, "{h}x{w} pass {i}: W -> {cw}");
                }
            }
            assert_eq!((ch, cw), (1, 1));
        }
    }
}
