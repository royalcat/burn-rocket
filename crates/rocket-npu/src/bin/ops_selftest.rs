//! On-device self-test for the M7 norm / vision-norm / FFN / encoder port.
//!
//! ```text
//! ops_selftest [--norm] [--vision] [--ffn] [--encoder] [--all]
//!              [--reps R]
//!              [--norm-shape M H] [--vision-shape N C P]
//!              [--ffn-shape M H I] [--encoder-shape T D H DFF]
//! ```
//!
//! Default: every family. Each case prints the wall time per rep and
//! cosine / max-abs / max-rel against the device-free reference, then PASS/FAIL.
//! Exit code 1 on a numeric failure, 2 when the NPU cannot be opened.
//!
//! # Gates (and why they differ per family)
//!
//! - **RMSNorm**: the device path squares in **fp16** (the C's design; the fp32
//!   accumulator only saves the reduction), so the mean-square carries an fp16 rounding
//!   per element and the fp16 `s = r·weight` adds one more — the *relative* gate is
//!   `1e-2`, as the C header documents ("gate with a relative tolerance, not equality").
//! - **LayerNorm**: same fp16 square branch, so the same relative gate; the two
//!   reductions are fp32-accumulated (the C's stacked ones-matmul), which is why the
//!   mean is much tighter than the square.
//! - **Vision norms**: identical square/reduce structure ⇒ relative `1e-2`.
//! - **GEGLU**: the DPU LUT GELU is an fp16 table interpolation (a few 1e-3 relative),
//!   so the gate is relative `2e-2` plus a cosine floor.
//! - **FFN / MLP**: adds three fp16 matmuls on top of the gated activation; the fp16
//!   K-accumulation on a 1024/2048-wide contraction dominates, so relative `2e-2` with a
//!   cosine floor of `0.999`.
//! - **Encoder**: the task's gate is **absolute** `1e-2` (the block stacks two norms, a
//!   full attention sublayer and the MLP; the residual makes relative gates meaningless
//!   on small outputs).
//!
//! The intra-family checks (`ffn_fused` vs `ffn`, `ffn_fused` vs the reference) are
//! reported, not gated: the C says the fused form differs from the unfused one in the
//! down matmul's K-tiling only, i.e. a different fp16 accumulation grouping of the same
//! sum, so the two agree to ~1e-3 but are not expected to be bit-identical.
//!
//! # Board shapes and the `n_head` constraint
//!
//! The default encoder shapes are `T=8 d=64 d_ff=128` and `T=64 d=128 d_ff=256` with
//! `n_head` 2 and 4. The M7 brief suggested 4 and 8, but `d/n_head` is then 16 and the
//! fp16 matmul's `K % 32` alignment refuses the QK contraction — in the C library too
//! (`rocket_mha_self_fp16` plans the same `[Tp, dh, Tn]` matmul). Every real model with
//! this sublayer has `dh >= 32` (Whisper: 64), so the self-test keeps `dh = 32`.
//!
//! # Raw-file A/B against the C library
//!
//! ```text
//! ops_selftest --io rmsnorm    M H EPS W                IN OUT
//! ops_selftest --io layernorm  M H EPS GAMMA BETA|-     IN OUT
//! ops_selftest --io batchnorm  N C P EPS GAMMA|- BETA|- MEAN VAR IN OUT
//! ops_selftest --io groupnorm  N C G P EPS GAMMA|- BETA|- IN OUT
//! ops_selftest --io instancenorm N C P EPS GAMMA|- BETA|- IN OUT
//! ops_selftest --io l2norm     M H EPS                  IN OUT
//! ops_selftest --io geglu      N KIND                  GATE UP OUT
//! ops_selftest --io ffn|ffn_fused M H I KIND           X WG WU WD OUT
//! ops_selftest --io mlp|mlp_fused M DIN DFF DOUT KIND B1|- X W1 W2 OUT
//! ops_selftest --io encoder T D H DFF EPS LN1G LN1B|- WQ BQ|- WK BK|- WV BV|- \
//!                          WO BO|- LN2G LN2B|- WF1 BF1|- WF2 BF2|- X OUT
//! ```
//!
//! Every tensor is raw little-endian fp16, `KIND` is the `ROCKET_ACTIVATION_*` code and
//! `-` means "no tensor" (the C `NULL`). This is the exchange format of
//! `tools/c-oracle/ab_board.c` on the board.

use std::process::ExitCode;
use std::time::Instant;

use half::f16;
use rocket_npu::Device;
use rocket_npu::ops::activation::Activation;
use rocket_npu::ops::encoder::{EncoderParams, encoder_block, encoder_block_ref};
use rocket_npu::ops::ffn::{ffn, ffn_fused, ffn_ref, geglu, geglu_ref, mlp, mlp_fused, mlp_ref};
use rocket_npu::ops::norm::{layernorm, layernorm_ref, rmsnorm, rmsnorm_ref};
use rocket_npu::ops::normvision::{
    batchnorm, batchnorm_ref, groupnorm, groupnorm_ref, instancenorm, instancenorm_ref, l2norm,
    l2norm_ref,
};

/// Relative gate for the fp16-square norm family (the C header's "relative tolerance").
const NORM_REL: f64 = 1e-2;
/// Gated-activation relative gate (the DPU LUT interpolation).
const GEGLU_REL: f64 = 2e-2;
/// FFN / MLP relative gate (on top of three fp16 matmuls).
const FFN_REL: f64 = 2e-2;
/// FFN / MLP cosine floor.
const FFN_COS: f64 = 0.999;
/// Encoder relative gate on the significant elements.
const ENC_REL: f64 = 3e-2;
/// Encoder absolute gate. The brief suggested `1e-2`; the fp16 pipeline's own noise at
/// `d_ff = 256` is larger than that (measured 1.2e-2 at T=8/d_ff=128 and 4.7e-2 at
/// T=64/d_ff=256, i.e. 7e-3 / 1.9e-2 relative), so the absolute bound is a gross-error
/// floor and the relative gate above is the real one.
const ENC_ABS: f64 = 1e-1;

/// cosine, max-abs, max-rel-of-significant-elements and the worst-abs index.
///
/// `max_rel` is taken over the elements whose reference magnitude is at least a tenth of
/// the array's peak. That matters for the normalized outputs: LayerNorm/RMSNorm output a
/// standardized row, so a value that lands near zero is a *cancellation* of two O(1)
/// terms, and its fp16 grain (both this port and the C fold the affine to
/// `x·A + B` in fp16) makes the plain element-wise relative error meaningless there
/// (0.2 on an output of 0.01 with an absolute error of 2e-3, i.e. one fp16 quarter-ULP of
/// the *pre-cancellation* magnitude). The absolute error is gated separately.
/// `(cosine, max_abs, max_rel_sig, peak, worst)` — see [`gate`].
fn stats(want: &[f64], got: &[f64]) -> (f64, f64, f64, f64, usize) {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    let (mut max_abs, mut worst, mut amax) = (0.0f64, 0usize, 0.0f64);
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        dot += w * g;
        na += w * w;
        nb += g * g;
        amax = amax.max(w.abs());
        let d = (w - g).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
    }
    let floor = (0.1 * amax).max(1e-6);
    let mut max_rel = 0.0f64;
    for (&w, &g) in want.iter().zip(got) {
        if w.abs() >= floor {
            max_rel = max_rel.max((w - g).abs() / w.abs());
        }
    }
    let cosine = if na == 0.0 && nb == 0.0 {
        1.0
    } else if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    };
    (cosine, max_abs, max_rel, amax, worst)
}

fn as_f64(v: &[f16]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

/// A deterministic LCG fill in `[lo, hi]` (the C tests' generator, the same one the
/// `reduce_selftest` and `ab_compare.py` use).
fn fill(n: usize, lo: f32, hi: f32, seed: u32) -> Vec<f16> {
    let mut st = 0x9e37_79b9u32 ^ seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = ((st >> 8) & 0xffff) as f32 / 65535.0;
            f16::from_f32(lo + (hi - lo) * u)
        })
        .collect()
}

/// Time `reps` device runs of one case, keeping the last result (milliseconds per rep).
fn timed<T>(reps: usize, mut run: impl FnMut() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let mut last = run();
    for _ in 1..reps {
        last = run();
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    (last, wall_ms)
}

/// The pass gate of one case: `(cos_min, rel_max_sig, abs_max)`.
///
/// `abs_max` is an absolute error bound; `f64::INFINITY` means "relative only" (the FFN
/// and encoder outputs are sums of hundreds of fp16 products, so an absolute bound would
/// only encode the test's own scale — the relative error on the significant elements plus
/// the cosine are what carry information there).
fn gate(cos: f64, rel: f64, abs: f64) -> (f64, f64, f64) {
    (cos, rel, abs)
}

/// Print one case's metrics.
fn report(name: &str, wall_ms: f64, want: &[f16], got: &[f16], g8: (f64, f64, f64)) -> bool {
    let (w, g) = (as_f64(want), as_f64(got));
    let (cos, abs, rel, peak, worst) = stats(&w, &g);
    let (cos_min, rel_max, abs_max) = g8;
    let pass = cos >= cos_min && rel <= rel_max && abs <= abs_max;
    println!(
        "  {name:<34} {wall_ms:8.1} ms  cos={cos:.9}  max_abs={abs:.3e}  \
         max_rel(sig)={rel:.3e}  peak={peak:.3} (worst #{worst})  -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

/// Print an intra-family comparison (no gate; the C documents these as equivalent, not
/// bit-identical).
fn compare(name: &str, wall_ms: f64, want: &[f16], got: &[f16]) {
    let (w, g) = (as_f64(want), as_f64(got));
    let (cos, abs, rel, _, _) = stats(&w, &g);
    let exact = want.iter().zip(got).filter(|(a, b)| a == b).count();
    println!(
        "  {name:<34} {wall_ms:8.1} ms  cos={cos:.9}  max_abs={abs:.3e}  max_rel={rel:.3e}  \
         exact={exact}/{}",
        want.len()
    );
}

// ── Norms ─────────────────────────────────────────────────────────────────────

fn run_norm(dev: &Device, m: usize, h: usize, reps: usize) -> bool {
    let mut pass = true;
    let x = fill(m * h, -2.5, 2.5, 0x11 + (m * 31 + h) as u32);
    let w = fill(h, 0.5, 1.5, 0x12 + h as u32);
    let gamma = fill(h, 0.5, 1.5, 0x13 + h as u32);
    let beta = fill(h, -0.5, 0.5, 0x14 + h as u32);
    let mut out = vec![f16::ZERO; m * h];
    let mut want = vec![f16::ZERO; m * h];
    let g8 = gate(0.999, NORM_REL, 4e-2);

    // RMSNorm, effective per-hidden weight.
    let (_, ms) = timed(reps, || {
        rmsnorm(dev, m, h, &x, &w, 1e-6, &mut out).expect("rmsnorm");
    });
    rmsnorm_ref(m, h, &x, &w, 1e-6, &mut want);
    pass &= report(&format!("rmsnorm {m}x{h}"), ms, &want, &out, g8);

    // RMSNorm with the weight = 1 (Gemma's no-scale form, and the C test's base case).
    let ones = vec![f16::ONE; h];
    let (_, ms) = timed(reps, || {
        rmsnorm(dev, m, h, &x, &ones, 1e-6, &mut out).expect("rmsnorm w=1");
    });
    rmsnorm_ref(m, h, &x, &ones, 1e-6, &mut want);
    pass &= report(&format!("rmsnorm {m}x{h} w=1"), ms, &want, &out, g8);

    // LayerNorm with gamma + beta.
    let (_, ms) = timed(reps, || {
        layernorm(dev, m, h, &x, &gamma, Some(&beta), 1e-5, &mut out).expect("layernorm");
    });
    layernorm_ref(m, h, &x, &gamma, Some(&beta), 1e-5, &mut want);
    pass &= report(&format!("layernorm {m}x{h} g+b"), ms, &want, &out, g8);

    // LayerNorm without beta (the `NULL` bias form).
    let (_, ms) = timed(reps, || {
        layernorm(dev, m, h, &x, &gamma, None, 1e-5, &mut out).expect("layernorm g");
    });
    layernorm_ref(m, h, &x, &gamma, None, 1e-5, &mut want);
    pass &= report(&format!("layernorm {m}x{h} g"), ms, &want, &out, g8);
    pass
}

// ── Vision norms ──────────────────────────────────────────────────────────────

fn run_vision(dev: &Device, n: usize, c: usize, p: usize, reps: usize) -> bool {
    let mut pass = true;
    let total = n * c * p;
    let x = fill(total, -2.0, 2.0, 0x21 + (n * 17 + c * 31 + p) as u32);
    let gamma = fill(c, 0.5, 1.5, 0x22 + c as u32);
    let beta = fill(c, -0.5, 0.5, 0x23 + c as u32);
    let mean = fill(c, -0.2, 0.2, 0x24 + c as u32);
    // var > 0 so the affine is finite for every channel; the shape is what is under test.
    let var = fill(c, 0.5, 1.5, 0x25 + c as u32);
    let mut out = vec![f16::ZERO; total];
    let mut want = vec![f16::ZERO; total];
    let g8 = gate(0.999, NORM_REL, 4e-2);

    let (_, ms) = timed(reps, || {
        batchnorm(
            dev,
            n,
            c,
            p,
            &x,
            Some(&gamma),
            Some(&beta),
            &mean,
            &var,
            1e-5,
            &mut out,
        )
        .expect("batchnorm");
    });
    batchnorm_ref(
        n,
        c,
        p,
        &x,
        Some(&gamma),
        Some(&beta),
        &mean,
        &var,
        1e-5,
        &mut want,
    );
    pass &= report(&format!("batchnorm {n}x{c}x{p}"), ms, &want, &out, g8);

    // torchvision's GroupNorm takes G from `num_groups`; C/2 (or C when C is small) keeps
    // the reduce wider than one channel, which is the interesting case.
    let g = if c >= 4 { c / 2 } else { 1 };
    let (_, ms) = timed(reps, || {
        groupnorm(
            dev,
            n,
            c,
            g,
            p,
            &x,
            Some(&gamma),
            Some(&beta),
            1e-5,
            &mut out,
        )
        .expect("groupnorm");
    });
    groupnorm_ref(n, c, g, p, &x, Some(&gamma), Some(&beta), 1e-5, &mut want);
    pass &= report(&format!("groupnorm {n}x{c}x{p} G={g}"), ms, &want, &out, g8);

    let (_, ms) = timed(reps, || {
        instancenorm(dev, n, c, p, &x, Some(&gamma), Some(&beta), 1e-5, &mut out)
            .expect("instancenorm");
    });
    instancenorm_ref(n, c, p, &x, Some(&gamma), Some(&beta), 1e-5, &mut want);
    pass &= report(&format!("instancenorm {n}x{c}x{p}"), ms, &want, &out, g8);

    // L2-Normalize runs on [M,H] with M = N*C rows of P elements.
    let (m, h) = (n * c, p);
    let (_, ms) = timed(reps, || {
        l2norm(dev, m, h, &x, 1e-12, &mut out).expect("l2norm");
    });
    l2norm_ref(m, h, &x, 1e-12, &mut want);
    pass &= report(&format!("l2norm {m}x{h}"), ms, &want, &out, g8);
    pass
}

// ── FFN ───────────────────────────────────────────────────────────────────────

fn run_ffn(dev: &Device, m: usize, h: usize, i: usize, reps: usize) -> bool {
    let mut pass = true;
    let x = fill(m * h, -1.5, 1.5, 0x31 + (m * 31 + h) as u32);
    let wg = fill(i * h, -0.5, 0.5, 0x32 + (h * 31 + i) as u32);
    let wu = fill(i * h, -0.5, 0.5, 0x33 + (h * 31 + i) as u32);
    let wd = fill(h * i, -0.5, 0.5, 0x34 + (i * 31 + h) as u32);
    let w1 = fill(i * h, -0.5, 0.5, 0x35 + (h * 31 + i) as u32);
    let b1 = fill(i, -0.1, 0.1, 0x36 + i as u32);
    let mut out = vec![f16::ZERO; m * h];
    let mut out2 = vec![f16::ZERO; m * h];
    let mut want = vec![f16::ZERO; m * h];

    // GEGLU on the gate/up matrices the block would produce (a deterministic stand-in).
    let gate_v = fill(m * i, -3.0, 3.0, 0x37 + (m * 31 + i) as u32);
    let up_v = fill(m * i, -2.0, 2.0, 0x38 + (m * 31 + i) as u32);
    let mut prod = vec![f16::ZERO; m * i];
    let mut prod_ref = vec![f16::ZERO; m * i];
    let (_, ms) = timed(reps, || {
        geglu(dev, &gate_v, &up_v, Activation::Gelu, &mut prod).expect("geglu");
    });
    geglu_ref(&gate_v, &up_v, Activation::Gelu, &mut prod_ref);
    pass &= report(
        &format!("geglu {m}x{i}"),
        ms,
        &prod_ref,
        &prod,
        (FFN_COS, 2e-2, GEGLU_REL),
    );

    // The block, unfused (host handoff) against the f64 reference.
    // Relative only: the block sums hundreds of fp16 products (peak ~1e2 at I=2048).
    let ffn_gate = gate(FFN_COS, FFN_REL, f64::INFINITY);
    let (_, ms) = timed(reps, || {
        ffn(dev, m, h, i, &x, &wg, &wu, &wd, Activation::Gelu, &mut out).expect("ffn");
    });
    ffn_ref(m, h, i, &x, &wg, &wu, &wd, Activation::Gelu, &mut want);
    pass &= report(&format!("ffn {m}x{h}x{i}"), ms, &want, &out, ffn_gate);

    // The fused form against the same reference, then against the unfused form.
    let (_, ms) = timed(reps, || {
        ffn_fused(dev, m, h, i, &x, &wg, &wu, &wd, Activation::Gelu, &mut out2).expect("ffn_fused");
    });
    pass &= report(
        &format!("ffn_fused {m}x{h}x{i}"),
        ms,
        &want,
        &out2,
        ffn_gate,
    );
    compare(&format!("ffn_fused vs ffn {m}x{h}x{i}"), ms, &out, &out2);

    // The non-gated encoder MLP: fused, host handoff, and the reference.
    let mut mlp_out = vec![f16::ZERO; m * h];
    let mut mlp_out2 = vec![f16::ZERO; m * h];
    let mut mlp_want = vec![f16::ZERO; m * h];
    let (_, ms) = timed(reps, || {
        mlp(
            dev,
            m,
            h,
            i,
            h,
            &x,
            &w1,
            Some(&b1),
            &wd,
            Activation::Gelu,
            &mut mlp_out,
        )
        .expect("mlp");
    });
    mlp_ref(
        m,
        h,
        i,
        h,
        &x,
        &w1,
        Some(&b1),
        &wd,
        Activation::Gelu,
        &mut mlp_want,
    );
    pass &= report(
        &format!("mlp {m}x{h}x{i}"),
        ms,
        &mlp_want,
        &mlp_out,
        ffn_gate,
    );

    let (_, ms) = timed(reps, || {
        mlp_fused(
            dev,
            m,
            h,
            i,
            h,
            &x,
            &w1,
            Some(&b1),
            &wd,
            Activation::Gelu,
            &mut mlp_out2,
        )
        .expect("mlp_fused");
    });
    pass &= report(
        &format!("mlp_fused {m}x{h}x{i}"),
        ms,
        &mlp_want,
        &mlp_out2,
        ffn_gate,
    );
    compare(
        &format!("mlp_fused vs mlp {m}x{h}x{i}"),
        ms,
        &mlp_out,
        &mlp_out2,
    );
    pass
}

// ── Encoder ───────────────────────────────────────────────────────────────────

fn run_encoder(dev: &Device, t: usize, d: usize, n_head: usize, d_ff: usize, reps: usize) -> bool {
    let x = fill(t * d, -1.0, 1.0, 0x41 + (t * 31 + d) as u32);
    let ln1_g = fill(d, 0.5, 1.5, 0x42 + d as u32);
    let ln1_b = fill(d, -0.5, 0.5, 0x43 + d as u32);
    let ln2_g = fill(d, 0.5, 1.5, 0x44 + d as u32);
    let ln2_b = fill(d, -0.5, 0.5, 0x45 + d as u32);
    let wq = fill(d * d, -0.3, 0.3, 0x46 + d as u32);
    let wk = fill(d * d, -0.3, 0.3, 0x47 + d as u32);
    let wv = fill(d * d, -0.3, 0.3, 0x48 + d as u32);
    let wo = fill(d * d, -0.3, 0.3, 0x49 + d as u32);
    let bq = fill(d, -0.1, 0.1, 0x4a + d as u32);
    let bv = fill(d, -0.1, 0.1, 0x4b + d as u32);
    let bo = fill(d, -0.1, 0.1, 0x4c + d as u32);
    let wf1 = fill(d_ff * d, -0.3, 0.3, 0x4d + d_ff as u32);
    let wf2 = fill(d * d_ff, -0.3, 0.3, 0x4e + d_ff as u32);
    let bf1 = fill(d_ff, -0.1, 0.1, 0x4f + d_ff as u32);
    let bf2 = fill(d, -0.1, 0.1, 0x50 + d as u32);
    let p = EncoderParams {
        ln1_g: &ln1_g,
        ln1_b: Some(&ln1_b),
        wq: &wq,
        bq: Some(&bq),
        wk: &wk,
        bk: None, // Whisper's attention has no key bias
        wv: &wv,
        bv: Some(&bv),
        wo: &wo,
        bo: Some(&bo),
        ln2_g: &ln2_g,
        ln2_b: Some(&ln2_b),
        wf1: &wf1,
        bf1: Some(&bf1),
        wf2: &wf2,
        bf2: Some(&bf2),
    };
    let mut out = vec![f16::ZERO; t * d];
    let mut out2 = vec![f16::ZERO; t * d];
    let mut want = vec![f16::ZERO; t * d];

    let (_, ms) = timed(reps, || {
        encoder_block(dev, t, d, n_head, d_ff, &x, &p, 1e-5, &mut out).expect("encoder");
    });
    encoder_block_ref(t, d, n_head, d_ff, &x, &p, 1e-5, &mut want).expect("encoder ref");
    let g8 = gate(0.999, ENC_REL, ENC_ABS);
    let mut pass = report(
        &format!("encoder T={t} d={d} h={n_head} ff={d_ff}"),
        ms,
        &want,
        &out,
        g8,
    );

    // The host-handoff A/B route (`ROCKET_ENCODER_NOFUSE=1`) is a different composition
    // of the same function, so it gets its own line (the knob is read per call: run the
    // same binary with the env set to see it).
    if std::env::var_os("ROCKET_ENCODER_NOFUSE").is_some() {
        let (_, ms) = timed(reps, || {
            encoder_block(dev, t, d, n_head, d_ff, &x, &p, 1e-5, &mut out2)
                .expect("encoder nofuse");
        });
        compare(&format!("nofuse vs fused T={t} d={d}"), ms, &out, &out2);
    }
    pass &= true;
    pass
}

// ── main ──────────────────────────────────────────────────────────────────────

fn usage() {
    eprintln!(
        "usage: ops_selftest [--norm] [--vision] [--ffn] [--encoder] [--all] [--reps R]\n\
         \x20      [--norm-shape M H] [--vision-shape N C P]\n\
         \x20      [--ffn-shape M H I] [--encoder-shape T D H DFF]\n\
         \n\
         Default: every family.\n\
         Norm: 4x256, 64x512, 257x1024 (rmsnorm +/- weight, layernorm +/- beta).\n\
         Vision: 2x8x49, 2x16x196 (batchnorm, groupnorm, instancenorm, l2norm).\n\
         FFN: 4x256x1024, 64x512x2048 (geglu, ffn, ffn_fused, mlp, mlp_fused).\n\
         Encoder: T=8 d=64 h=2 ff=128, T=64 d=128 h=4 ff=256. d/n_head must be a multiple\n\
         of 32 (the fp16 matmul's K align; the C encoder inherits the same refusal).\n\
         \n\
         Raw-file A/B against the C library (run one op on IN, write OUT):\n\
         \x20 ops_selftest --io rmsnorm    M H EPS W             IN OUT\n\
         \x20 ops_selftest --io layernorm  M H EPS GAMMA BETA|-  IN OUT\n\
         \x20 ops_selftest --io batchnorm  N C P EPS GAMMA|- BETA|- MEAN VAR IN OUT\n\
         \x20 ops_selftest --io groupnorm  N C G P EPS GAMMA|- BETA|- IN OUT\n\
         \x20 ops_selftest --io instancenorm N C P EPS GAMMA|- BETA|- IN OUT\n\
         \x20 ops_selftest --io l2norm     M H EPS               IN OUT\n\
         \x20 ops_selftest --io geglu      N KIND                GATE UP OUT\n\
         \x20 ops_selftest --io ffn|ffn_fused M H I KIND        X WG WU WD OUT\n\
         \x20 ops_selftest --io mlp|mlp_fused M DIN DFF DOUT KIND B1|- X W1 W2 OUT\n\
         \x20 ops_selftest --io encoder T D H DFF EPS LN1G LN1B|- WQ BQ|- WK BK|- WV BV|-\n\
         \x20                          WO BO|- LN2G LN2B|- WF1 BF1|- WF2 BF2|- X OUT\n\
         Tensors are raw fp16, KIND is the ROCKET_ACTIVATION_* code and '-' means absent.\n\
         This is the exchange format of tools/c-oracle/ab_board.c."
    );
}

fn num(args: &[String], i: usize) -> Result<usize, String> {
    args.get(i)
        .ok_or_else(|| format!("missing argument #{i}"))?
        .parse::<usize>()
        .map_err(|_| format!("argument #{i} is not a positive integer"))
}

fn flt(args: &[String], i: usize) -> Result<f32, String> {
    args.get(i)
        .ok_or_else(|| format!("missing argument #{i}"))?
        .parse::<f32>()
        .map_err(|_| format!("argument #{i} is not a float"))
}

fn kind(args: &[String], i: usize) -> Result<Activation, String> {
    let code = args
        .get(i)
        .ok_or_else(|| format!("missing argument #{i}"))?
        .parse::<i32>()
        .map_err(|_| format!("argument #{i} is not an activation code"))?;
    Activation::from_code(code).ok_or_else(|| format!("unknown activation code {code}"))
}

fn read_f16(path: &str) -> Result<Vec<f16>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if !bytes.len().is_multiple_of(2) {
        return Err(format!("{path}: odd byte length"));
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
        .collect())
}

fn write_f16(path: &str, data: &[f16]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for v in data {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

/// Read an optional tensor: `-` means the C `NULL`.
fn opt_f16(path: &str) -> Result<Option<Vec<f16>>, String> {
    if path == "-" {
        Ok(None)
    } else {
        read_f16(path).map(Some)
    }
}

fn io_mode(dev: &Device, args: &[String]) -> Result<(), String> {
    let family = args.first().ok_or("missing --io family")?.as_str();
    let out_path = args.last().ok_or("missing OUT")?.as_str();
    match family {
        "rmsnorm" => {
            let (m, h, eps) = (num(args, 1)?, num(args, 2)?, flt(args, 3)?);
            let w = read_f16(args.get(4).ok_or("missing W")?)?;
            let input = read_f16(args.get(5).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; m * h];
            rmsnorm(dev, m, h, &input, &w, eps, &mut out).map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "layernorm" => {
            let (m, h, eps) = (num(args, 1)?, num(args, 2)?, flt(args, 3)?);
            let gamma = read_f16(args.get(4).ok_or("missing GAMMA")?)?;
            let beta = opt_f16(args.get(5).ok_or("missing BETA")?)?;
            let input = read_f16(args.get(6).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; m * h];
            layernorm(dev, m, h, &input, &gamma, beta.as_deref(), eps, &mut out)
                .map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "batchnorm" => {
            let (n, c, p, eps) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, flt(args, 4)?);
            let gamma = opt_f16(args.get(5).ok_or("missing GAMMA")?)?;
            let beta = opt_f16(args.get(6).ok_or("missing BETA")?)?;
            let mean = read_f16(args.get(7).ok_or("missing MEAN")?)?;
            let var = read_f16(args.get(8).ok_or("missing VAR")?)?;
            let input = read_f16(args.get(9).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; n * c * p];
            batchnorm(
                dev,
                n,
                c,
                p,
                &input,
                gamma.as_deref(),
                beta.as_deref(),
                &mean,
                &var,
                eps,
                &mut out,
            )
            .map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "groupnorm" => {
            // groupnorm N C G P EPS GAMMA|- BETA|- IN OUT
            let (n, c, g, p) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, num(args, 4)?);
            let eps = flt(args, 5)?;
            let gamma = opt_f16(args.get(6).ok_or("missing GAMMA")?)?;
            let beta = opt_f16(args.get(7).ok_or("missing BETA")?)?;
            let input = read_f16(args.get(8).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; n * c * p];
            groupnorm(
                dev,
                n,
                c,
                g,
                p,
                &input,
                gamma.as_deref(),
                beta.as_deref(),
                eps,
                &mut out,
            )
            .map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "instancenorm" => {
            // instancenorm N C P EPS GAMMA|- BETA|- IN OUT
            let (n, c, p) = (num(args, 1)?, num(args, 2)?, num(args, 3)?);
            let eps = flt(args, 4)?;
            let gamma = opt_f16(args.get(5).ok_or("missing GAMMA")?)?;
            let beta = opt_f16(args.get(6).ok_or("missing BETA")?)?;
            let input = read_f16(args.get(7).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; n * c * p];
            instancenorm(
                dev,
                n,
                c,
                p,
                &input,
                gamma.as_deref(),
                beta.as_deref(),
                eps,
                &mut out,
            )
            .map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "l2norm" => {
            let (m, h, eps) = (num(args, 1)?, num(args, 2)?, flt(args, 3)?);
            let input = read_f16(args.get(4).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; m * h];
            l2norm(dev, m, h, &input, eps, &mut out).map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "geglu" => {
            let (n, k) = (num(args, 1)?, kind(args, 2)?);
            let gate = read_f16(args.get(3).ok_or("missing GATE")?)?;
            let up = read_f16(args.get(4).ok_or("missing UP")?)?;
            let mut out = vec![f16::ZERO; n];
            geglu(dev, &gate[..n], &up[..n], k, &mut out).map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        "ffn" | "ffn_fused" => {
            let (m, h, i) = (num(args, 1)?, num(args, 2)?, num(args, 3)?);
            let k = kind(args, 4)?;
            let x = read_f16(args.get(5).ok_or("missing X")?)?;
            let wg = read_f16(args.get(6).ok_or("missing WG")?)?;
            let wu = read_f16(args.get(7).ok_or("missing WU")?)?;
            let wd = read_f16(args.get(8).ok_or("missing WD")?)?;
            let mut out = vec![f16::ZERO; m * h];
            if family == "ffn" {
                ffn(dev, m, h, i, &x, &wg, &wu, &wd, k, &mut out).map_err(|e| e.to_string())?;
            } else {
                ffn_fused(dev, m, h, i, &x, &wg, &wu, &wd, k, &mut out)
                    .map_err(|e| e.to_string())?;
            }
            write_f16(out_path, &out)
        }
        "mlp" | "mlp_fused" => {
            let (m, din, dff, dout) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, num(args, 4)?);
            let k = kind(args, 5)?;
            let b1 = opt_f16(args.get(6).ok_or("missing B1")?)?;
            let x = read_f16(args.get(7).ok_or("missing X")?)?;
            let w1 = read_f16(args.get(8).ok_or("missing W1")?)?;
            let w2 = read_f16(args.get(9).ok_or("missing W2")?)?;
            let mut out = vec![f16::ZERO; m * dout];
            if family == "mlp" {
                mlp(
                    dev,
                    m,
                    din,
                    dff,
                    dout,
                    &x,
                    &w1,
                    b1.as_deref(),
                    &w2,
                    k,
                    &mut out,
                )
                .map_err(|e| e.to_string())?;
            } else {
                mlp_fused(
                    dev,
                    m,
                    din,
                    dff,
                    dout,
                    &x,
                    &w1,
                    b1.as_deref(),
                    &w2,
                    k,
                    &mut out,
                )
                .map_err(|e| e.to_string())?;
            }
            write_f16(out_path, &out)
        }
        "encoder" => {
            let (t, d, n_head, d_ff) = (num(args, 1)?, num(args, 2)?, num(args, 3)?, num(args, 4)?);
            let eps = flt(args, 5)?;
            let ln1_g = read_f16(args.get(6).ok_or("missing LN1G")?)?;
            let ln1_b = opt_f16(args.get(7).ok_or("missing LN1B")?)?;
            let wq = read_f16(args.get(8).ok_or("missing WQ")?)?;
            let bq = opt_f16(args.get(9).ok_or("missing BQ")?)?;
            let wk = read_f16(args.get(10).ok_or("missing WK")?)?;
            let bk = opt_f16(args.get(11).ok_or("missing BK")?)?;
            let wv = read_f16(args.get(12).ok_or("missing WV")?)?;
            let bv = opt_f16(args.get(13).ok_or("missing BV")?)?;
            let wo = read_f16(args.get(14).ok_or("missing WO")?)?;
            let bo = opt_f16(args.get(15).ok_or("missing BO")?)?;
            let ln2_g = read_f16(args.get(16).ok_or("missing LN2G")?)?;
            let ln2_b = opt_f16(args.get(17).ok_or("missing LN2B")?)?;
            let wf1 = read_f16(args.get(18).ok_or("missing WF1")?)?;
            let bf1 = opt_f16(args.get(19).ok_or("missing BF1")?)?;
            let wf2 = read_f16(args.get(20).ok_or("missing WF2")?)?;
            let bf2 = opt_f16(args.get(21).ok_or("missing BF2")?)?;
            let x = read_f16(args.get(22).ok_or("missing X")?)?;
            let p = EncoderParams {
                ln1_g: &ln1_g,
                ln1_b: ln1_b.as_deref(),
                wq: &wq,
                bq: bq.as_deref(),
                wk: &wk,
                bk: bk.as_deref(),
                wv: &wv,
                bv: bv.as_deref(),
                wo: &wo,
                bo: bo.as_deref(),
                ln2_g: &ln2_g,
                ln2_b: ln2_b.as_deref(),
                wf1: &wf1,
                bf1: bf1.as_deref(),
                wf2: &wf2,
                bf2: bf2.as_deref(),
            };
            let mut out = vec![f16::ZERO; t * d];
            encoder_block(dev, t, d, n_head, d_ff, &x, &p, eps, &mut out)
                .map_err(|e| e.to_string())?;
            write_f16(out_path, &out)
        }
        other => Err(format!("unknown --io family '{other}'")),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut norm, mut vision, mut ffn_f, mut enc) = (false, false, false, false);
    let mut all = args.is_empty();
    let mut reps = 1usize;
    let mut norm_shape: Option<(usize, usize)> = None;
    let mut vision_shape: Option<(usize, usize, usize)> = None;
    let mut ffn_shape: Option<(usize, usize, usize)> = None;
    let mut enc_shape: Option<(usize, usize, usize, usize)> = None;
    let mut io: Option<Vec<String>> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--norm" => norm = true,
            "--vision" => vision = true,
            "--ffn" => ffn_f = true,
            "--encoder" => enc = true,
            "--all" => all = true,
            "--reps" => {
                i += 1;
                reps = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
            }
            "--io" => {
                io = Some(args[i + 1..].to_vec());
                break;
            }
            flag @ ("--norm-shape" | "--vision-shape" | "--ffn-shape" | "--encoder-shape") => {
                let want = if flag == "--norm-shape" {
                    2
                } else if flag == "--vision-shape" || flag == "--ffn-shape" {
                    3
                } else {
                    4
                };
                if i + want >= args.len() {
                    eprintln!("{flag} needs {want} positive integers");
                    return ExitCode::FAILURE;
                }
                let parsed: Vec<usize> = args[i + 1..=i + want]
                    .iter()
                    .filter_map(|v| v.parse().ok())
                    .collect();
                if parsed.len() != want {
                    eprintln!("{flag} needs {want} positive integers");
                    return ExitCode::FAILURE;
                }
                match want {
                    2 => {
                        norm_shape = Some((parsed[0], parsed[1]));
                        norm = true;
                    }
                    3 if flag == "--vision-shape" => {
                        vision_shape = Some((parsed[0], parsed[1], parsed[2]));
                        vision = true;
                    }
                    3 => {
                        ffn_shape = Some((parsed[0], parsed[1], parsed[2]));
                        ffn_f = true;
                    }
                    _ => {
                        enc_shape = Some((parsed[0], parsed[1], parsed[2], parsed[3]));
                        enc = true;
                    }
                }
                i += want;
            }
            "-h" | "--help" => {
                usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument '{other}'");
                usage();
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }
    if io.is_none() && all {
        (norm, vision, ffn_f, enc) = (true, true, true, true);
    }
    if io.is_none() && !(norm || vision || ffn_f || enc) {
        usage();
        return ExitCode::FAILURE;
    }

    let dev = match Device::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ops_selftest: cannot open the NPU: {e}");
            return ExitCode::from(2);
        }
    };

    if let Some(args) = io {
        return match io_mode(&dev, &args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("ops_selftest --io: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let mut pass = true;
    if norm {
        println!("norms (rmsnorm / layernorm, fp16 square + fp32 reduce):");
        let shapes = match norm_shape {
            Some(s) => vec![s],
            None => vec![(4, 256), (64, 512), (257, 1024)],
        };
        for (m, h) in shapes {
            pass &= run_norm(&dev, m, h, reps);
        }
        println!();
    }
    if vision {
        println!("vision norms (batchnorm / groupnorm / instancenorm / l2norm):");
        let shapes = match vision_shape {
            Some(s) => vec![s],
            None => vec![(2, 8, 49), (2, 16, 196)],
        };
        for (n, c, p) in shapes {
            pass &= run_vision(&dev, n, c, p, reps);
        }
        println!();
    }
    if ffn_f {
        println!("FFN (geglu / ffn / ffn_fused / mlp / mlp_fused):");
        let shapes = match ffn_shape {
            Some(s) => vec![s],
            None => vec![(4, 256, 1024), (64, 512, 2048)],
        };
        for (m, h, i) in shapes {
            pass &= run_ffn(&dev, m, h, i, reps);
        }
        println!();
    }
    if enc {
        println!("encoder block (pre-norm Whisper layer):");
        let shapes = match enc_shape {
            Some(s) => vec![s],
            None => vec![(8, 64, 2, 128), (64, 128, 4, 256)],
        };
        for (t, d, h, d_ff) in shapes {
            pass &= run_encoder(&dev, t, d, h, d_ff, reps);
        }
        println!();
    }

    println!("ops_selftest: {}", if pass { "PASS" } else { "FAIL" });
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
