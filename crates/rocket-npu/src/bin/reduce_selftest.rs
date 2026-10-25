//! On-device reduce / softmax self-test: the port of `rocket_reduce.c`,
//! `rocket_softmax.c` and `rocket_scale_rows_fp16` against their host references.
//!
//! ```text
//! reduce_selftest [--pool] [--feature] [--cumsum] [--softmax] [--all]
//!                 [--shape C H W] [--reps R]
//! ```
//!
//! Default: every family. Each case prints the wall time per rep and
//! cosine / max-abs / max-rel against the device-free reference, then PASS/FAIL.
//! Exit code 1 on a numeric failure, 2 when the NPU cannot be opened.
//!
//! # Gates (and why they differ per family)
//!
//! - **Spatial pools, average**: the PPU divides by a per-axis fp16 reciprocal
//!   (`fp16(65536/k)`), so every pass carries ~1e-3 of reciprocal quantization plus one
//!   fp16 rounding. Gated at `max_rel <= 2e-3` (plus an absolute floor of 4e-3, exactly
//!   the C test's `0.004 + 0.004*|ref|`).
//! - **Spatial pools, max/min**: idempotent, no reciprocal — the telescoped device result
//!   must be **bit-exact** against the host reduction.
//! - **Feature reduce**: the ones-matmul uses the genuine fp32 accumulator
//!   (`matmul::one_shot_f32out`), so the error against the fp64 reference is the fp32
//!   rounding of the sum; gated at `max_rel <= 1e-5`.
//! - **Cumsum**: the fp32 accumulator is narrowed to fp16 on read-back, so the gate is
//!   relative at `2e-3` (the C header's "fp16-rounding-accurate, not bit-exact").
//! - **Softmax**: the C test's two-metric gate — absolute `2e-3` for tiny
//!   probabilities, relative `3%` where `p > 1e-3` — plus a row-sum-to-1 check at `1%`.
//! - **LogSoftmax**: all-additive, so **absolute** error (the C's `0.05`, dominated by
//!   the EXP-LUT bias in `log(s)` and fp16 storage of the large-magnitude entries).
//! - **Cross-entropy**: all-additive in fp32, absolute `1e-3` (the task's gate).
//!
//! The non-decomposable pool shape (`5x17x13`, 17 is prime > 16) must report the host
//! fallback from `global_avgpool_plan` and then match the fp64 reference exactly, since
//! the fallback *is* the reference.

use std::process::ExitCode;
use std::time::Instant;

use half::f16;
use rocket_npu::Device;
use rocket_npu::ops::reduce as red;
use rocket_npu::ops::softmax as sm;

/// Average-pool relative gate (the documented per-pass reciprocal quantization).
const AVG_REL: f64 = 2e-3;
/// Average-pool absolute floor (zero-mean channels must still pass).
const AVG_ABS: f64 = 4e-3;
/// Feature-reduce relative gate (fp32 accumulator vs the fp64 reference).
const FEATURE_REL: f64 = 1e-5;
/// Cumsum relative gate (fp16 read-back of an exact fp32 scan).
const CUMSUM_REL: f64 = 2e-3;
/// Softmax element gates (the C test's pair).
const SOFTMAX_ABS: f64 = 2e-3;
const SOFTMAX_REL: f64 = 0.03;
/// Softmax row-sum deviation gate.
const SOFTMAX_ROWSUM: f64 = 0.01;
/// LogSoftmax absolute gate (the C's).
const LOGSOFTMAX_ABS: f64 = 0.05;
/// Cross-entropy absolute gate.
const CE_ABS: f64 = 1e-3;

/// cosine, max-abs and max-rel (floor `1e-3`) of `got` against `want`.
fn stats(want: &[f64], got: &[f64]) -> (f64, f64, f64, usize) {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    let (mut max_abs, mut max_rel, mut worst) = (0.0f64, 0.0f64, 0usize);
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        dot += w * g;
        na += w * w;
        nb += g * g;
        let d = (w - g).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        let rel = d / w.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let cosine = if na == 0.0 && nb == 0.0 {
        1.0
    } else if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    };
    (cosine, max_abs, max_rel, worst)
}

fn as_f64(v: &[f16]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

/// A deterministic LCG fill in `[lo, hi)` (the C tests' generator).
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

// ── Spatial pools ─────────────────────────────────────────────────────────────

fn run_pool(dev: &Device, c: usize, h: usize, w: usize, reps: usize) -> bool {
    let n = c * h * w;
    // Offset per channel in [0.5, 2.5] plus noise: every mean is O(1), channels differ,
    // and a divisor off by a few percent misses (the C test's reasoning).
    let mut input = vec![f16::ZERO; n];
    for ch in 0..c {
        let off = 0.5 + 2.0 * ((ch * 37 % 100) as f32) / 100.0;
        let chan = fill(h * w, -1.0, 1.0, 0x3ea7u32 ^ (ch as u32));
        for (i, v) in chan.iter().enumerate() {
            input[ch * h * w + i] = f16::from_f32(off + v.to_f32());
        }
    }

    let plan = red::global_avgpool_plan(c, h, w);
    let path = match &plan {
        Ok(()) => "npu",
        Err(_) => "host-fallback",
    };

    let mut pass = true;

    // Average.
    let mut got = vec![f16::ZERO; c];
    let (_, wall) = timed(reps, || {
        red::global_avgpool(dev, c, h, w, &input, &mut got).expect("avgpool");
    });
    let mut want = vec![f16::ZERO; c];
    red::global_avgpool_ref(c, h, w, &input, &mut want);
    let (cos, max_abs, max_rel, worst) = stats(&as_f64(&want), &as_f64(&got));
    let mut bad = 0;
    for (wv, gv) in want.iter().zip(&got) {
        let (wv, gv) = (wv.to_f64(), gv.to_f64());
        let d = (wv - gv).abs();
        // NaN (a pool that produced nothing) must fail this test, so compare the
        // other way round rather than negating.
        if d > AVG_ABS + AVG_REL * wv.abs() || d.is_nan() {
            bad += 1;
        }
    }
    // A host fallback runs the reference itself, so it is exact by construction.
    let ok = bad == 0;
    println!(
        "  avg   C={c:<4} {h}x{w} ({path}) {wall:>8.2} ms cos={cos:.9} max_abs={max_abs:.2e} \
         max_rel={max_rel:.2e} bad={bad} worst=c{worst} -> {}",
        if ok { "PASS" } else { "FAIL" }
    );
    pass &= ok;

    // Extrema: bit-exact on the NPU path, and (trivially) on the fallback.
    for (name, run, reference) in [
        (
            "max",
            red::global_maxpool as fn(&Device, usize, usize, usize, &[f16], &mut [f16]) -> _,
            red::global_maxpool_ref as fn(usize, usize, usize, &[f16], &mut [f16]),
        ),
        ("min", red::global_minpool, red::global_minpool_ref),
    ] {
        let mut got = vec![f16::ZERO; c];
        let (_, wall) = timed(reps, || {
            run(dev, c, h, w, &input, &mut got).expect("extrema");
        });
        let mut want = vec![f16::ZERO; c];
        reference(c, h, w, &input, &mut want);
        let bad = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        println!(
            "  {name}   C={c:<4} {h}x{w} ({path}) {wall:>8.2} ms exact bad={bad} -> {}",
            if bad == 0 { "PASS" } else { "FAIL" }
        );
        pass &= bad == 0;
    }
    pass
}

// ── Feature reduce ────────────────────────────────────────────────────────────

fn run_feature(dev: &Device, m: usize, h: usize, reps: usize) -> bool {
    // A zero-mean and a unit-mean fill exercise both the sum and the mean scale, and a
    // sum that can be near zero (the relative gate falls back to the 1e-3 floor).
    let mut pass = true;
    for mean in [false, true] {
        let input = fill(m * h, -1.0, 3.0, 0x51ed ^ ((m as u32) << 8) ^ (h as u32));
        let mut want = vec![0.0f32; m];
        red::reduce_feature_ref(m, h, &input, &mut want, mean);
        let mut got = vec![0.0f32; m];
        let (_, wall) = timed(reps, || {
            red::reduce_feature(dev, m, h, &input, &mut got, mean).expect("feature");
        });
        let want64: Vec<f64> = want.iter().map(|v| *v as f64).collect();
        let got64: Vec<f64> = got.iter().map(|v| *v as f64).collect();
        let (cos, max_abs, max_rel, worst) = stats(&want64, &got64);
        let ok = max_rel <= FEATURE_REL;
        println!(
            "  feat  M={m:<5} H={h:<5} mean={:<5} {wall:>8.2} ms cos={cos:.9} \
             max_abs={max_abs:.2e} max_rel={max_rel:.2e} worst={worst} -> {}",
            mean as u8,
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    pass
}

// ── Cumsum ────────────────────────────────────────────────────────────────────

fn run_cumsum(dev: &Device, m: usize, n: usize, reps: usize) -> bool {
    let input = fill(m * n, -2.0, 2.0, 0xc5_0000 ^ ((m as u32) << 8) ^ (n as u32));
    let mut pass = true;
    for (exclusive, reverse) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut want = vec![f16::ZERO; m * n];
        red::cumsum_ref(m, n, &input, &mut want, exclusive, reverse);
        let mut got = vec![f16::ZERO; m * n];
        let (_, wall) = timed(reps, || {
            red::cumsum(dev, m, n, &input, &mut got, exclusive, reverse).expect("cumsum");
        });
        let (cos, max_abs, max_rel, worst) = stats(&as_f64(&want), &as_f64(&got));
        let exact = want.iter().zip(&got).filter(|(a, b)| a == b).count();
        let ok = max_rel <= CUMSUM_REL;
        println!(
            "  cumsum M={m:<4} N={n:<4} excl={} rev={} {wall:>8.2} ms cos={cos:.9} \
             max_abs={max_abs:.2e} max_rel={max_rel:.2e} exact={exact}/{} worst={worst} -> {}",
            exclusive as u8,
            reverse as u8,
            m * n,
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    pass
}

// ── Softmax / logsoftmax / cross-entropy ──────────────────────────────────────

fn softmax_case_ok(want: &[f16], got: &[f16], m: usize, n: usize) -> (bool, f64) {
    let mut bad = 0;
    for (w, g) in want.iter().zip(got) {
        let (rf, gt) = (w.to_f64(), g.to_f64());
        let ad = (gt - rf).abs();
        let rd = ad / (rf + 1e-6);
        if ad > SOFTMAX_ABS && (rf <= 1e-3 || rd > SOFTMAX_REL) {
            bad += 1;
        }
    }
    let mut worst_dev = 0.0f64;
    for row in 0..m {
        let s: f64 = got[row * n..row * n + n].iter().map(|v| v.to_f64()).sum();
        worst_dev = worst_dev.max((s - 1.0).abs());
    }
    (bad == 0 && worst_dev <= SOFTMAX_ROWSUM, worst_dev)
}

fn run_softmax(dev: &Device, m: usize, n: usize, reps: usize) -> bool {
    let input = fill(m * n, -4.0, 4.0, 0x5f_0000 ^ ((m as u32) << 8) ^ (n as u32));
    let mut pass = true;

    // Softmax.
    let mut want = vec![f16::ZERO; m * n];
    sm::softmax_ref(m, n, &input, &mut want);
    let mut got = vec![f16::ZERO; m * n];
    let (_, wall) = timed(reps, || {
        sm::softmax(dev, m, n, &input, &mut got).expect("softmax");
    });
    let (cos, max_abs, max_rel, worst) = stats(&as_f64(&want), &as_f64(&got));
    let (ok, rowdev) = softmax_case_ok(&want, &got, m, n);
    println!(
        "  softmax M={m:<4} N={n:<6} {wall:>9.2} ms cos={cos:.9} max_abs={max_abs:.2e} \
         max_rel={max_rel:.2e} rowsum_dev={rowdev:.2e} worst={worst} -> {}",
        if ok { "PASS" } else { "FAIL" }
    );
    pass &= ok;

    // LogSoftmax: all-additive, so absolute error (and exp(out) sums to 1).
    let mut want = vec![f16::ZERO; m * n];
    sm::logsoftmax_ref(m, n, &input, &mut want);
    let mut got = vec![f16::ZERO; m * n];
    let (_, wall) = timed(reps, || {
        sm::logsoftmax(dev, m, n, &input, &mut got).expect("logsoftmax");
    });
    let (cos, max_abs, max_rel, worst) = stats(&as_f64(&want), &as_f64(&got));
    let mut exp_dev = 0.0f64;
    for row in 0..m {
        let s: f64 = got[row * n..row * n + n]
            .iter()
            .map(|v| v.to_f64().exp())
            .sum();
        exp_dev = exp_dev.max((s - 1.0).abs());
    }
    let ok = max_abs <= LOGSOFTMAX_ABS;
    println!(
        "  logsft M={m:<4} N={n:<6} {wall:>9.2} ms cos={cos:.9} max_abs={max_abs:.2e} \
         max_rel={max_rel:.2e} sum_exp_dev={exp_dev:.2e} worst={worst} -> {}",
        if ok { "PASS" } else { "FAIL" }
    );
    pass &= ok;

    // Cross-entropy with pseudo-random targets (including an out-of-range one, which the
    // reference treats as "no logit" and must not panic).
    let target: Vec<i32> = (0..m)
        .map(|row| {
            let t = ((row * 7919 + 13) % n) as i32;
            if row == 0 { -1 } else { t }
        })
        .collect();
    let mut want = vec![0.0f32; m];
    sm::cross_entropy_ref(m, n, &input, &target, &mut want);
    let mut got = vec![0.0f32; m];
    let (_, wall) = timed(reps, || {
        sm::cross_entropy(dev, m, n, &input, &target, &mut got).expect("cross-entropy");
    });
    let (cos, max_abs, max_rel, worst) = stats(
        &want.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &got.iter().map(|v| *v as f64).collect::<Vec<_>>(),
    );
    let ok = max_abs <= CE_ABS;
    println!(
        "  ce     M={m:<4} N={n:<6} {wall:>9.2} ms cos={cos:.9} max_abs={max_abs:.2e} \
         max_rel={max_rel:.2e} worst_row={worst} -> {}",
        if ok { "PASS" } else { "FAIL" }
    );
    pass &= ok;
    pass
}

/// `MemAvailable` from `/proc/meminfo`, in bytes.
fn mem_available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

// ── Raw-file A/B exchange ─────────────────────────────────────────────────────

/// Read raw little-endian fp16 words.
fn read_f16(path: &str) -> Result<Vec<f16>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
        .collect())
}

/// Read raw little-endian fp32 words.
fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn write_f16(path: &str, v: &[f16]) -> Result<(), String> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("write {path}: {e}"))
}

fn write_f32(path: &str, v: &[f32]) -> Result<(), String> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("write {path}: {e}"))
}

/// `--io FAMILY ...`: run one op on a raw input file and write the raw output.
///
/// This is the exchange format shared with `tests/ab_board.c`, so the C library and the
/// port can consume the same bytes and their outputs can be compared (bit-for-bit where
/// the algorithm allows it) instead of only reading each side's own PASS line.
fn io_mode(dev: &Device, args: &[String]) -> Result<(), String> {
    let num = |i: usize| -> Result<usize, String> {
        args.get(i)
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(|| format!("missing/invalid integer argument #{} ({args:?})", i + 1))
    };
    let flag = |i: usize| -> Result<bool, String> { Ok(num(i)? != 0) };
    let family = args.first().map(String::as_str).unwrap_or("");
    // The positional layout is family-specific; IN/OUT sit after the numbers.
    match family {
        "avg" | "max" | "min" => {
            let (c, h, w) = (num(1)?, num(2)?, num(3)?);
            let input = read_f16(args.get(4).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; c];
            let r = match family {
                "avg" => red::global_avgpool(dev, c, h, w, &input, &mut out),
                "max" => red::global_maxpool(dev, c, h, w, &input, &mut out),
                _ => red::global_minpool(dev, c, h, w, &input, &mut out),
            };
            r.map_err(|e| e.to_string())?;
            write_f16(args.get(5).ok_or("missing OUT")?, &out)
        }
        "feature" => {
            let (m, h, mean) = (num(1)?, num(2)?, flag(3)?);
            let input = read_f16(args.get(4).ok_or("missing IN")?)?;
            let mut out = vec![0.0f32; m];
            red::reduce_feature(dev, m, h, &input, &mut out, mean).map_err(|e| e.to_string())?;
            write_f32(args.get(5).ok_or("missing OUT")?, &out)
        }
        "cumsum" => {
            let (m, n, excl, rev) = (num(1)?, num(2)?, flag(3)?, flag(4)?);
            let input = read_f16(args.get(5).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; m * n];
            red::cumsum(dev, m, n, &input, &mut out, excl, rev).map_err(|e| e.to_string())?;
            write_f16(args.get(6).ok_or("missing OUT")?, &out)
        }
        "softmax" | "logsoftmax" => {
            let (m, n) = (num(1)?, num(2)?);
            let input = read_f16(args.get(3).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; m * n];
            if family == "softmax" {
                sm::softmax(dev, m, n, &input, &mut out).map_err(|e| e.to_string())?;
            } else {
                sm::logsoftmax(dev, m, n, &input, &mut out).map_err(|e| e.to_string())?;
            }
            write_f16(args.get(4).ok_or("missing OUT")?, &out)
        }
        "ce" => {
            let (m, n) = (num(1)?, num(2)?);
            let input = read_f16(args.get(3).ok_or("missing IN")?)?;
            // The shared default target rule (an out-of-range row 0, so both sides check
            // the guard too); ab_board.c uses the same one.
            let target: Vec<i32> = (0..m)
                .map(|row| {
                    let t = ((row * 7919 + 13) % n) as i32;
                    if row == 0 { -1 } else { t }
                })
                .collect();
            let mut out = vec![0.0f32; m];
            sm::cross_entropy(dev, m, n, &input, &target, &mut out).map_err(|e| e.to_string())?;
            write_f32(args.get(4).ok_or("missing OUT")?, &out)
        }
        "scale_rows" => {
            let (m, n) = (num(1)?, num(2)?);
            let r = read_f32(args.get(3).ok_or("missing RFILE")?)?;
            let input = read_f16(args.get(4).ok_or("missing IN")?)?;
            let mut out = vec![f16::ZERO; m * n];
            rocket_npu::ops::norm::scale_rows(dev, m, n, &input, &r, &mut out)
                .map_err(|e| e.to_string())?;
            write_f16(args.get(5).ok_or("missing OUT")?, &out)
        }
        other => Err(format!("unknown --io family '{other}'")),
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn usage() {
    eprintln!(
        "usage: reduce_selftest [--pool] [--feature] [--cumsum] [--softmax] [--all]\n\
         \x20      [--shape C H W] [--reps R]\n\
         \n\
         Default: every family. --shape restricts the spatial pools to one C H W.\n\
         Pool shapes: 8x7x7, 8x14x14, 3x56x56, 4x49x49, 5x17x13 (host fallback), 2x1x1.\n\
         Feature: 4x32, 64x256, 264x320, 512x1024 (sum and mean).\n\
         Cumsum: 4x16, 64x128, 128x256 x exclusive/reverse.\n\
         Softmax: 4x8, 64x1000, 257x4096 (plus the 151936-wide vocabulary shape if it fits).\n\
         \n\
         Raw-file A/B against the C library (run one op on IN, write OUT):\n\
         \x20 reduce_selftest --io avg|max|min C H W      IN OUT\n\
         \x20 reduce_selftest --io feature M H MEAN       IN OUT\n\
         \x20 reduce_selftest --io cumsum M N EXCL REV    IN OUT\n\
         \x20 reduce_selftest --io softmax|logsoftmax M N IN OUT\n\
         \x20 reduce_selftest --io ce M N                 IN OUT\n\
         \x20 reduce_selftest --io scale_rows M N RFILE   IN OUT\n\
         Pools/cumsum/softmax read and write raw fp16, feature and ce raw f32 (MEAN, EXCL and\n\
         REV are 0/1). This is the exchange format of tools/c-oracle/ab_board.c on the board."
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut pool, mut feature, mut cumsum, mut softmax) = (false, false, false, false);
    let mut all = args.is_empty();
    let mut reps = 1usize;
    let mut shape: Option<(usize, usize, usize)> = None;
    let mut io: Option<Vec<String>> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => pool = true,
            "--feature" => feature = true,
            "--cumsum" => cumsum = true,
            "--softmax" => softmax = true,
            "--all" => all = true,
            "--reps" => {
                i += 1;
                reps = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
            }
            "--io" => {
                // Everything after --io belongs to the raw-file mode.
                io = Some(args[i + 1..].to_vec());
                break;
            }
            "--shape" => {
                if i + 3 >= args.len() {
                    usage();
                    return ExitCode::FAILURE;
                }
                let parse = |s: &String| s.parse::<usize>().ok();
                match (
                    parse(&args[i + 1]),
                    parse(&args[i + 2]),
                    parse(&args[i + 3]),
                ) {
                    (Some(c), Some(h), Some(w)) => {
                        shape = Some((c, h, w));
                        pool = true;
                    }
                    _ => {
                        eprintln!("--shape needs three positive integers: C H W");
                        return ExitCode::FAILURE;
                    }
                }
                i += 3;
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
    if io.is_some() {
        // The raw-file mode owns the whole command line: no --all expansion, no summary.
    } else if all {
        pool = true;
        feature = true;
        cumsum = true;
        softmax = true;
    }
    if io.is_none() && !(pool || feature || cumsum || softmax) {
        usage();
        return ExitCode::FAILURE;
    }

    let dev = match Device::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("reduce_selftest: cannot open the NPU: {e}");
            return ExitCode::from(2);
        }
    };

    if let Some(args) = io {
        return match io_mode(&dev, &args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("reduce_selftest --io: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let mut pass = true;
    if pool {
        println!("spatial pools (PPU telescoping decomposition):");
        let shapes: Vec<(usize, usize, usize)> = match shape {
            Some(s) => vec![s],
            None => vec![
                (8, 7, 7),
                (8, 14, 14),
                (3, 56, 56),
                (4, 49, 49),
                (5, 17, 13), // 17 is prime > 16: the exact host fallback
                (2, 1, 1),   // degenerate identity
            ],
        };
        for (c, h, w) in shapes {
            pass &= run_pool(&dev, c, h, w, reps);
        }
        println!();
    }
    if feature {
        println!("feature-axis reduce (ones-matmul, fp32 accumulator):");
        for (m, h) in [(4usize, 32usize), (64, 256), (264, 320), (512, 1024)] {
            pass &= run_feature(&dev, m, h, reps);
        }
        println!();
    }
    if cumsum {
        println!("cumsum (triangular-ones matmul):");
        for (m, n) in [(4usize, 16usize), (64, 128), (128, 256)] {
            pass &= run_cumsum(&dev, m, n, reps);
        }
        println!();
    }
    if softmax {
        println!("softmax / logsoftmax / cross-entropy:");
        for (m, n) in [(4usize, 8usize), (64, 1000), (257, 4096)] {
            pass &= run_softmax(&dev, m, n, reps);
        }
        // The LM-head shape (a 151936-wide vocabulary) is ~78 MB of fp16 per matrix, and
        // the case holds three host matrices plus the NPU-side scratch. Run it only when
        // the board reports the headroom (and never on a busy board's swap), so the test
        // skips cleanly instead of being OOM-killed.
        let (m, n) = (257usize, 151936usize);
        let need = (m * n * 2 * 4 + (1 << 30)) as u64; // host matrices + 1 GiB of slack
        match mem_available() {
            Some(avail) if avail >= need => {
                println!("  (vocabulary shape {m}x{n}, {} MiB needed)", need >> 20);
                pass &= run_softmax(&dev, m, n, reps);
            }
            Some(avail) => println!(
                "  (skipping the {m}x{n} vocabulary shape: {} MiB available, {} MiB needed)",
                avail >> 20,
                need >> 20
            ),
            None => println!("  (skipping the {m}x{n} vocabulary shape: no /proc/meminfo)"),
        }
        println!();
    }

    println!("reduce_selftest: {}", if pass { "PASS" } else { "FAIL" });
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
