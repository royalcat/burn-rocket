//! On-device int4 self-tests: the dense one-shot, the group-wise one-shot, and the
//! resident [`I4Context`] in both modes.
//!
//!   `int4_selftest [MODE] [--kt-cap N] [--extreme] [M K N]`
//!
//! `MODE` defaults to `--dense`; `--gw-one-shot GROUP` and `--prepacked-gw GROUP` take
//! the quant group as their argument.
//!
//! - `--dense` — [`one_shot_int4_ex`]. `--kt-cap` maps to its int16-saturation cap: `0`
//!   (the default) is the C `rocket_matmul_int4`, `480` the C's in-model value, which
//!   keeps `|partial| <= 64*Kt < 32767` for `[-8,7]` operands. `--extreme` fills both
//!   operands with `-8` (worst case: every K-tile partial is `64*Kt`), which forces the
//!   saturation regime at `kt_cap=0` and shows the tiled reference reproducing it.
//! - `--prepacked` — [`I4Context::pack_weight`] + [`I4Context::matmul`], raw int32.
//! - `--gw-one-shot GROUP` — [`one_shot_int4_groupwise`], compared with an f64
//!   dequantized reference and with the resident path (they must agree bit for bit).
//! - `--prepacked-gw GROUP` — [`I4Context::pack_weight_gw`] + [`I4Context::matmul_gw`].
//!
//! Two references are checked for the integer modes. The plain int64 sum is only
//! reachable when no K-tile partial overflows int16 — the conv writes int16, so a bigger
//! `Kt` saturates and the host cannot recover it. The second reference reproduces the
//! tiling (the plan's `Kt`) and clamps each tile's partial to int16 first, which is
//! exact in *both* regimes and therefore also validates that the tiling the call used is
//! the planned one.
//!
//! Both resident modes pack at the canonical tile height and then compute at the
//! requested `M`, and repeat the compute at a second `M` with the same weight: that is
//! the M-independence contract of `rk4_worker_alloc` (one pack serves every row count).

use std::time::Instant;

use rocket_npu::Device;
use rocket_npu::hw;
use rocket_npu::matmul::{
    I4Context, I4Weight, one_shot_int4_ex, one_shot_int4_groupwise, plan_int4,
};

/// A xorshift64 -> uniform f32 in [-0.5, 0.5), the int8_selftest generator.
fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

/// Symmetric int4 codes, one per `i8`, in `[-8,7]`.
fn int4_codes(x: &[f32]) -> Vec<i8> {
    x.iter()
        .map(|v| ((v * 16.0).round() as i32).clamp(-8, 7) as i8)
        .collect()
}

/// Per-group symmetric int4 quantization: one scale per `group` K elements, codes in
/// `[-7,7]` (the range the C's `49*group < 32767` saturation bound assumes).
fn quant_rows(x: &[f32], rows: usize, k: usize, group: usize) -> (Vec<i8>, Vec<f32>) {
    let ng = k / group;
    let mut q = vec![0i8; rows * k];
    let mut s = vec![0f32; rows * ng];
    for r in 0..rows {
        for g in 0..ng {
            let base = r * k + g * group;
            let mx = x[base..base + group]
                .iter()
                .fold(0f32, |a, v| a.max(v.abs()));
            let sc = if mx > 0.0 { mx / 7.0 } else { 1.0 };
            s[r * ng + g] = sc;
            for j in 0..group {
                let v = (x[base + j] / sc).round().clamp(-7.0, 7.0);
                q[base + j] = v as i8;
            }
        }
    }
    (q, s)
}

/// `C = A * B^T` summed exactly in int64 over the whole K.
fn exact_reference(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i64> {
    let mut want = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for p in 0..k {
                sum += a[i * k + p] as i64 * b[j * k + p] as i64;
            }
            want[i * n + j] = sum;
        }
    }
    want
}

/// The same sum with each K-tile's partial saturated to int16 — what the hardware's
/// int16 output writer produces, per `kt`.
fn tiled_reference(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, kt: usize) -> Vec<i64> {
    let mut want = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            let mut k0 = 0;
            while k0 < k {
                let k1 = (k0 + kt).min(k);
                let mut tile = 0i64;
                for p in k0..k1 {
                    tile += a[i * k + p] as i64 * b[j * k + p] as i64;
                }
                sum += tile.clamp(i16::MIN as i64, i16::MAX as i64);
                k0 = k1;
            }
            want[i * n + j] = sum;
        }
    }
    want
}

/// The dequantized group-wise reference, in f64.
fn dequant_reference(
    a: &[i8],
    b: &[i8],
    a_scale: &[f32],
    b_scale: &[f32],
    dims: (usize, usize, usize, usize),
) -> Vec<f64> {
    let (m, k, n, group) = dims;
    let ng = k / group;
    let mut want = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                let g = p / group;
                sum += (a[i * k + p] as f64)
                    * (b[j * k + p] as f64)
                    * (a_scale[i * ng + g] as f64)
                    * (b_scale[j * ng + g] as f64);
            }
            want[i * n + j] = sum;
        }
    }
    want
}

/// Max |difference| and exact equality of two int64 vectors.
fn compare_i64(got: &[i64], want: &[i64]) -> (i64, bool) {
    let max = got
        .iter()
        .zip(want)
        .fold(0i64, |a, (g, w)| a.max((g - w).abs()));
    let exact = got.iter().zip(want).all(|(g, w)| g == w);
    (max, exact)
}

/// Cosine, max |difference| and exact equality of two f64 vectors.
fn compare_f64(got: &[f64], want: &[f64]) -> (f64, f64, bool) {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut max = 0f64;
    let mut exact = true;
    for (g, w) in got.iter().zip(want) {
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
        exact &= g == w;
    }
    (dot / (na.sqrt() * nb.sqrt()), max, exact)
}

/// Compare a raw int32 result with the exact and the int16-clamped references, print
/// which one matched, and fail if neither did.
fn report_int32(got: &[i32], a: &[i8], b: &[i8], m: usize, k: usize, n: usize, kt: usize) -> bool {
    let got: Vec<i64> = got.iter().map(|v| *v as i64).collect();
    let want = exact_reference(a, b, m, k, n);
    let (max, exact) = compare_i64(&got, &want);
    let tiled = tiled_reference(a, b, m, k, n, kt);
    let (tmax, texact) = compare_i64(&got, &tiled);
    println!("  exact {exact} max_abs {max} | tiled(Kt={kt})-exact {texact} max_abs {tmax}");
    if exact {
        println!("  OK: int32-exact vs the int64 reference");
        true
    } else if texact {
        println!(
            "  OK: int32-exact vs the int16-clamped tiled reference; the plain int64 sum \
             differs because Kt={kt} lets a tile partial saturate int16"
        );
        true
    } else {
        println!("  FAIL: neither reference matches");
        false
    }
}

fn report_bytes(w: &I4Weight, k: usize, n: usize) {
    let (pm, pk, pn, group) = w.shape();
    println!(
        "  weight packed at M={pm} K={pk} N={pn} group={group}: {} bytes (codes {} B)",
        w.bytes(),
        n * k / 2
    );
    // Half a byte per element plus one CBUF-bank pad per worker BO.
    let cbuf = rocket_npu::regcmd::desc::CBUF_BANK_SIZE;
    assert!(
        w.bytes() >= n * k / 2,
        "a codes-only weight cannot be smaller"
    );
    assert!(
        w.bytes() <= n * k / 2 + 8 * cbuf,
        "at most one bank of pad per worker"
    );
}

/// `want` rows of `src` (row stride `stride`), taken cyclically so a small `M` can still
/// exercise a bigger row count.
fn rows_of<T: Copy>(src: &[T], stride: usize, want: usize) -> Vec<T> {
    let rows = src.len() / stride;
    let mut out = Vec::with_capacity(want * stride);
    for r in 0..want {
        let base = (r % rows) * stride;
        out.extend_from_slice(&src[base..base + stride]);
    }
    out
}

/// A genuine pack/call tiling mismatch must be reported as the C's -2 rather than
/// miscompute. Forces one with a `ROCKET_MM_*` override on a row count this context has
/// not planned yet (the scratch cache is keyed by `(M,K,N,group)`, so an already-built
/// shape keeps the tiling it was packed against). The weight stays in this context: BO
/// handles are only valid on the fd that allocated them, so a second context cannot be
/// used.
fn check_tiling_mismatch(ctx: &mut I4Context, threads: usize, k: usize, n: usize, w: &I4Weight) {
    let group = w.shape().3;
    // Group-wise pins Kt to the group, so only an N-tile change can move its layout; the
    // dense mode moves Kt instead (which always binds when K > 32).
    let nsub = n.div_ceil(threads).div_ceil(64) * 64;
    let (knob, value) = if group == 0 {
        if k == 32 {
            println!("  tiling-mismatch guard: skipped (K = 32 is one K tile)");
            return;
        }
        ("ROCKET_MM_KT", "32")
    } else {
        if nsub <= 64 {
            println!("  tiling-mismatch guard: skipped (one 64-column N tile per worker)");
            return;
        }
        ("ROCKET_MM_NT", "64")
    };
    unsafe { std::env::set_var(knob, value) };
    let probe_m = 12usize;
    let result = {
        let a = vec![1i8; probe_m * k];
        if group == 0 {
            let mut out = vec![0i32; probe_m * n];
            ctx.matmul(probe_m, k, n, &a, &mut out, w)
        } else {
            let ng = k.div_ceil(group);
            let mut out = vec![0f32; probe_m * n];
            ctx.matmul_gw(
                probe_m,
                k,
                n,
                &a,
                &vec![1.0f32; probe_m * ng],
                &vec![1.0f32; n * ng],
                &mut out,
                w,
            )
        }
    };
    unsafe { std::env::remove_var(knob) };
    match result {
        Err(e) if e.status() == rocket_npu::Status::Tiling => println!(
            "  tiling-mismatch guard ({knob}={value}): {e} (status {}, the C's -2)",
            e.status().code()
        ),
        Err(e) => println!("  tiling-mismatch guard: unexpected {e}"),
        Ok(()) => println!("  FAIL: a re-tiled context accepted the weight"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut kt_cap = 0usize;
    let mut extreme = false;
    let mut mode = "dense";
    let mut group = 32usize;
    let mut dims: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--kt-cap" => {
                kt_cap = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--kt-cap needs a value")?
            }
            "--extreme" => extreme = true,
            "--dense" => mode = "dense",
            "--prepacked" => mode = "prepacked",
            "--gw-one-shot" => {
                mode = "gw-one-shot";
                group = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--gw-one-shot needs a group")?;
            }
            "--prepacked-gw" => {
                mode = "prepacked-gw";
                group = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--prepacked-gw needs a group")?;
            }
            other => dims.push(other.to_string()),
        }
    }
    let num = |i: usize, d: usize| dims.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let m: usize = num(0, 64);
    let k: usize = num(1, 256);
    let n: usize = num(2, 128);
    println!("int4 {mode} M={m} K={k} N={n} group={group}");
    assert_eq!(k % 32, 0, "K must be a multiple of 32");
    assert_eq!(n % 64, 0, "N must be a multiple of 64");
    if mode == "dense" {
        assert!(
            m == 1 || m.is_multiple_of(4),
            "M must be 1 or a multiple of 4"
        );
    } else {
        assert!(m.is_multiple_of(4), "M must be a multiple of 4");
    }
    let max_tile = hw::current().max_tile as usize;

    if mode == "dense" {
        let (mt, kt, nt) = plan_int4(m.max(4), k, n)?;
        let capped = if kt_cap >= 32 && kt > kt_cap {
            ((kt_cap / 32) * 32).max(32)
        } else {
            kt
        };
        println!(
            "plan: Mt={mt} Kt={kt} Nt={nt} jobs={} | capped Kt={capped} nKt={}",
            m.div_ceil(mt) * n.div_ceil(nt) * k.div_ceil(kt),
            k.div_ceil(capped)
        );

        let (a, b): (Vec<i8>, Vec<i8>) = if extreme {
            (vec![-8i8; m * k], vec![-8i8; n * k])
        } else {
            let mut seed = 0x1234_5678_9abc_def1u64;
            let af: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
            let bf: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();
            (int4_codes(&af), int4_codes(&bf))
        };

        let dev = Device::open()?;
        let mut got = vec![0i32; m * n];
        let t = Instant::now();
        one_shot_int4_ex(&dev, m, k, n, &a, &b, &mut got, kt_cap)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("one-shot {ms:.2} ms");
        if report_int32(&got, &a, &b, m, k, n, capped) {
            Ok(())
        } else {
            Err("int4 mismatch: neither reference matches".into())
        }
    } else {
        run_resident(mode, group, m, k, n, max_tile, extreme)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_resident(
    mode: &str,
    group: usize,
    m: usize,
    k: usize,
    n: usize,
    max_tile: usize,
    extreme: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Mode-specific contracts first, so a bad group fails before any NPU work.
    if mode != "prepacked" {
        if extreme {
            return Err(
                "--extreme is a dense-mode check ([-8,7] codes leave the group bound)".into(),
            );
        }
        assert_eq!(group % 32, 0, "group must be a multiple of 32");
        assert_eq!(k % group, 0, "K must be a multiple of group");
        assert!(49 * group < 32767, "49*group must stay inside int16");
    }

    // The canonical (M-independent) plan, for reference: a resident weight is scattered
    // at `Mt = max_tile`, so the same pack serves every M. This is the whole-N plan; a
    // worker plans its own N slice, so its `Kt` can differ for a K that is not
    // CBUF-bound anyway (`report_int32` checks both references).
    let (cmt, ckt, cnt) = plan_int4(max_tile, k, n)?;
    println!("canonical plan (full N, M={max_tile}): Mt={cmt} Kt={ckt} Nt={cnt}");

    let mut seed = 0x1234_5678_9abc_def1u64;
    let af: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
    let bf: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();

    if mode == "prepacked" {
        let (a, b) = if extreme {
            (vec![-8i8; m * k], vec![-8i8; n * k])
        } else {
            (int4_codes(&af), int4_codes(&bf))
        };
        let mut ctx = I4Context::new(5)?;
        let t = Instant::now();
        let w = ctx.pack_weight(max_tile, k, n, &b)?;
        let pack_ms = t.elapsed().as_secs_f64() * 1e3;
        report_bytes(&w, k, n);
        let mut got = vec![0i32; m * n];
        let t = Instant::now();
        ctx.matmul(m, k, n, &a, &mut got, &w)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("pack {pack_ms:.2} ms | resident matmul {ms:.2} ms");
        let ok = report_int32(&got, &a, &b, m, k, n, ckt);

        // M-independence: the same weight, a different row count, no re-pack.
        let m2 = if m == 4 { 8 } else { 4 };
        let a2 = rows_of(&a, k, m2);
        let mut small = vec![0i32; m2 * n];
        let t = Instant::now();
        ctx.matmul(m2, k, n, &a2, &mut small, &w)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("M-independence: packed at M={max_tile}, computed at M={m2} ({ms:.2} ms)");
        let ok2 = report_int32(&small, &a2, &b, m2, k, n, ckt);

        check_tiling_mismatch(&mut ctx, 5, k, n, &w);
        return if ok && ok2 {
            println!("OK: resident int4 exact at both row counts");
            Ok(())
        } else {
            Err("prepacked int4 mismatch".into())
        };
    }

    // Group-wise modes.
    let (a, a_scale) = quant_rows(&af, m, k, group);
    let (b, b_scale) = quant_rows(&bf, n, k, group);
    let want = dequant_reference(&a, &b, &a_scale, &b_scale, (m, k, n, group));

    let mut ctx = I4Context::new(5)?;
    let t = Instant::now();
    let w = ctx.pack_weight_gw(max_tile, k, n, &b, group)?;
    let pack_ms = t.elapsed().as_secs_f64() * 1e3;
    report_bytes(&w, k, n);
    assert_eq!(w.shape().3, group);
    let mut resident = vec![0f32; m * n];
    let t = Instant::now();
    ctx.matmul_gw(m, k, n, &a, &a_scale, &b_scale, &mut resident, &w)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!("pack {pack_ms:.2} ms | resident matmul_gw {ms:.2} ms");

    let mut ok = true;
    if mode == "gw-one-shot" {
        let dev = Device::open()?;
        let mut one = vec![0f32; m * n];
        let t = Instant::now();
        one_shot_int4_groupwise(&dev, m, k, n, &a, &b, &a_scale, &b_scale, &mut one, group)?;
        let one_ms = t.elapsed().as_secs_f64() * 1e3;
        let same = one == resident;
        let max = one
            .iter()
            .zip(&resident)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        println!(
            "one-shot {one_ms:.2} ms | one-shot vs resident: bit-identical {same} max_abs {max:e}"
        );
        ok &= same;
    }

    let got64: Vec<f64> = resident.iter().map(|v| *v as f64).collect();
    let (cos, max, _) = compare_f64(&got64, &want);
    println!("resident vs the f64 dequant reference: cosine {cos:.8} max_abs {max:.6}");
    ok &= max < 1e-3;

    // M-independence for the group-wise weight too.
    let m2 = if m == 4 { 8 } else { 4 };
    let ng = k / group;
    let a2 = rows_of(&a, k, m2);
    let a_scale2 = rows_of(&a_scale, ng, m2);
    let mut small = vec![0f32; m2 * n];
    ctx.matmul_gw(m2, k, n, &a2, &a_scale2, &b_scale, &mut small, &w)?;
    let want2 = dequant_reference(&a2, &b, &a_scale2, &b_scale, (m2, k, n, group));
    let small64: Vec<f64> = small.iter().map(|v| *v as f64).collect();
    let (cos2, max2, _) = compare_f64(&small64, &want2);
    println!(
        "M-independence: packed at M={max_tile}, computed at M={m2}: cosine {cos2:.8} max_abs {max2:.6}"
    );
    ok &= max2 < 1e-3;

    check_tiling_mismatch(&mut ctx, 5, k, n, &w);
    if ok {
        println!("OK: group-wise int4 matches its dequantized reference");
        Ok(())
    } else {
        Err("group-wise int4 mismatch".into())
    }
}
