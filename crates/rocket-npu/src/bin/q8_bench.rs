//! Q8_0-native projection probe (P2b): the resident f16 `Context` vs the
//! group-wise int8 `I8Context` at qwen3's projection shapes.
//!
//!   q8_bench [--reps R] [--threads T] [--groups 32,128,512] [--rows S]
//!            [--only f16|int8] [M K N]...
//!
//! Defaults: (82,1024,1024) (512,1024,1024) (2777,1024,1024) (2777,1024,3072)
//! (2777,4096,1024) — qwen3-embedding-0.6B's q/k/v/o (1024x1024), gate/up
//! (1024x3072) and down (4096x1024) projections at 82 / 512 / 2777 tokens — with
//! `--groups 32` (llama.cpp Q8_0's block width) and 4 threads.
//!
//! Both arms are timed the way the app's `burn_rocket::matmul` extension runs them
//! (`src/ext.rs` in the root crate): the activation arrives as f32, the arm's own
//! input conversion is included (f32->f16 plus the `M % 4` pad copy for the
//! resident f16 path; the per-row per-32-group f32->i8 quantize for the int8
//! path), the matmul is timed on its own, and the output is read back as f32. The
//! int8 arm's output is already f32 (`matmul_gw` applies the group scales on
//! readback), so it has no narrow; the f16 arm pays the f16->f32 readback
//! conversion. qwen3's projections have no bias, so neither arm adds one. `M` is
//! padded to `M % 4` (rows are independent; the pad is dropped before comparison). Every group in `--groups` is measured back to back inside
//! one process, each with its own f16 baseline arm, so the reported ratios do not
//! mix board windows.
//!
//! The weight is one synthetic `[N, K]` f32 matrix per shape, stored the way a
//! Q8_0 GGUF stores it (34-byte blocks: f16 scale + 32 int8 codes) and fed to
//! both arms: the f16 arm packs the *dequantized* blocks, the int8 arm packs the
//! raw codes + widened scales, so the two arms see the identical weight and the
//! int8 arm carries no requantization error of its own. The gate is the int8 arm
//! against the f64 reference product with the dequantized weight (what the app's
//! `--quant q8` dequant path computes in f32): cosine >= 0.9995, where the only
//! error left is the activation's own int8 quantization.
//!
//! A group > 32 requantizes the dequantized weight into that width — not
//! Q8_0-native, but it is the tradeoff curve: the driver reads back one int32
//! partial per K-tile, so the readback volume is `N*M*(K/group)*4` bytes
//! (group 32 = the Q8_0 block, i.e. the most readbacks).

use std::time::Instant;

use half::f16;
use rocket_npu::matmul::{Context, I8Context};

/// Pack-time `M` (the app's `PACK_M`): canonical tiling makes the resident weight
/// valid for every live `M >= 4`.
const PACK_M: usize = 512;

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

/// `(user + sys)` CPU seconds of this process.
fn cpu_s() -> f64 {
    // SAFETY: getrusage fills the zeroed struct on success, which is the only case read.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return 0.0;
    }
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// Cosine, max |difference| and exact equality of two f32 vectors (scored in f64).
fn compare(got: &[f32], want: &[f32]) -> (f64, f64, bool) {
    let (mut dot, mut na, mut nb, mut max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut exact = true;
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
        exact &= g == w;
    }
    (dot / (na.sqrt() * nb.sqrt()), max, exact)
}

/// Split `m` rows of `x`/`q`/`s` into at most `threads` contiguous blocks.
struct RowBlock<'a> {
    x: &'a [f32],
    q: &'a mut [i8],
    s: &'a mut [f32],
}

fn row_blocks<'a>(
    x: &'a [f32],
    q: &'a mut [i8],
    s: &'a mut [f32],
    m: usize,
    k: usize,
    ng: usize,
    threads: usize,
) -> Vec<RowBlock<'a>> {
    let step = m.div_ceil(threads.max(1)).max(1);
    let mut out = Vec::new();
    let (mut x, mut q, mut s) = (x, q, s);
    let mut left = m;
    while left > 0 {
        let take = left.min(step);
        let (xh, xt) = x.split_at(take * k);
        let (qh, qt) = q.split_at_mut(take * k);
        let (sh, st) = s.split_at_mut(take * ng);
        out.push(RowBlock {
            x: xh,
            q: qh,
            s: sh,
        });
        x = xt;
        q = qt;
        s = st;
        left -= take;
    }
    out
}

/// One row-group quantize, the same symmetric rule as `host::quantize_row_groups`
/// in the root crate (int8 resident W8A8): `scale = max|v| / 127`, `round(v/scale)`.
fn quantize_group(seg: &[f32], q: &mut [i8], s: &mut f32) {
    let amax = seg.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = (amax / 127.0).max(1e-12);
    *s = scale;
    let inv = 1.0 / scale;
    for (qi, &v) in q.iter_mut().zip(seg) {
        *qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
}

/// Per-row per-group f32 -> int8 quantize, parallel over `threads` row blocks.
fn quantize_rows_par(
    x: &[f32],
    m: usize,
    k: usize,
    group: usize,
    q: &mut [i8],
    s: &mut [f32],
    threads: usize,
) {
    let ng = k / group;
    let blocks = row_blocks(x, q, s, m, k, ng, threads);
    std::thread::scope(|scope| {
        for b in blocks {
            scope.spawn(move || {
                for (xr, (qr, sr)) in b.x.chunks(k).zip(b.q.chunks_mut(k).zip(b.s.chunks_mut(ng))) {
                    for g in 0..ng {
                        quantize_group(
                            &xr[g * group..(g + 1) * group],
                            &mut qr[g * group..(g + 1) * group],
                            &mut sr[g],
                        );
                    }
                }
            });
        }
    });
}

/// llama.cpp Q8_0 quantization: 32-value blocks of `{f16 scale, int8 codes}`,
/// `scale = max|v| / 127` (the same rule the W8A8 group path uses at group 32).
/// Returns `(codes [N,K], scales [N,K/32] widened to f32)`.
fn q8_0_quantize(w: &[f32], n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let ng = k / 32;
    let mut q = vec![0i8; n * k];
    let mut s = vec![0f32; n * ng];
    for r in 0..n {
        for g in 0..ng {
            let seg = &w[r * k + g * 32..r * k + (g + 1) * 32];
            let amax = seg.iter().fold(0f32, |a, &v| a.max(v.abs()));
            // Q8_0 stores the scale as f16; the codes are quantized against the
            // *stored* (f16-rounded) scale, exactly as llama.cpp does.
            let d = f16::from_f32(amax / 127.0);
            let scale = d.to_f32();
            s[r * ng + g] = scale;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for (i, &v) in seg.iter().enumerate() {
                q[r * k + g * 32 + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (q, s)
}

/// Requantize a dequantized `[N,K]` weight into `group`-wide symmetric int8 groups.
fn requantize(w: &[f32], n: usize, k: usize, group: usize) -> (Vec<i8>, Vec<f32>) {
    let ng = k / group;
    let mut q = vec![0i8; n * k];
    let mut s = vec![0f32; n * ng];
    for r in 0..n {
        for g in 0..ng {
            quantize_group(
                &w[r * k + g * group..r * k + (g + 1) * group],
                &mut q[r * k + g * group..r * k + (g + 1) * group],
                &mut s[r * ng + g],
            );
        }
    }
    (q, s)
}

/// `codes * scales` back to f32, in the stored group order.
fn dequantize(q: &[i8], s: &[f32], n: usize, k: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for g in 0..ng {
            let sc = s[r * ng + g];
            for i in 0..group {
                let idx = r * k + g * group + i;
                out[idx] = q[idx] as f32 * sc;
            }
        }
    }
    out
}

/// Pad `rows` f16 rows of width `k` to `padded_rows` by repeating the last row
/// (rows are independent; the pad is dropped on readback) — the app's `pad_rows`.
fn pad_rows_f16(src: &[f16], rows: usize, k: usize, padded_rows: usize) -> Vec<f16> {
    assert!(rows > 0, "pad_rows needs at least one row");
    if padded_rows == rows {
        return src.to_vec();
    }
    let mut out = Vec::with_capacity(padded_rows * k);
    out.extend_from_slice(src);
    let last = &src[(rows - 1) * k..rows * k];
    for _ in rows..padded_rows {
        out.extend_from_slice(last);
    }
    out
}

fn f32_to_f16_par(src: &[f32], threads: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; src.len()];
    let chunk = src.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (o, s) in out.chunks_mut(chunk).zip(src.chunks(chunk)) {
            scope.spawn(move || {
                for (d, &x) in o.iter_mut().zip(s) {
                    *d = f16::from_f32(x);
                }
            });
        }
    });
    out
}

fn f16_to_f32_par(src: &[f16], threads: usize) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    let chunk = src.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| {
        for (o, s) in out.chunks_mut(chunk).zip(src.chunks(chunk)) {
            scope.spawn(move || {
                for (d, &x) in o.iter_mut().zip(s) {
                    *d = x.to_f32();
                }
            });
        }
    });
    out
}

/// f64 reference product `C[sample,N] = A[sample,K] * Bq[N,K]^T` over the first
/// `sample` rows — the app's dequantized-weight f32 GEMM, accumulated in f64 so
/// the comparison is not limited by the reference's own rounding.
fn reference(a: &[f32], bq: &[f32], k: usize, n: usize, sample: usize) -> Vec<f32> {
    let mut out = vec![0f32; sample * n];
    for i in 0..sample {
        for j in 0..n {
            let mut sum = 0f64;
            for p in 0..k {
                sum += a[i * k + p] as f64 * bq[j * k + p] as f64;
            }
            out[i * n + j] = sum as f32;
        }
    }
    out
}

struct Shape {
    m: usize,
    k: usize,
    n: usize,
}

/// One measured (shape, group) pair, for the summary table.
struct Row {
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    f16_wall: f64,
    f16_cpu: f64,
    i8_wall: f64,
    i8_cpu: f64,
    i8_quant: f64,
    f16_bo: usize,
    i8_bo: usize,
    cos: f64,
    max: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reps = 3usize;
    let mut threads = 4usize;
    let mut groups = vec![32usize];
    let mut sample_rows = 256usize;
    let mut only = "all".to_string();
    let mut dims: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--reps" => reps = args.next().and_then(|v| v.parse().ok()).unwrap_or(reps),
            "--threads" => threads = args.next().and_then(|v| v.parse().ok()).unwrap_or(threads),
            "--group" | "--groups" => {
                let list = args.next().unwrap_or_default();
                groups = list
                    .split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect();
            }
            "--rows" => {
                sample_rows = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(sample_rows)
            }
            "--only" => only = args.next().unwrap_or(only),
            other => dims.push(other.to_string()),
        }
    }
    let shapes: Vec<Shape> = if dims.is_empty() {
        [
            (82, 1024, 1024),
            (512, 1024, 1024),
            (2777, 1024, 1024),
            (2777, 1024, 3072),
            (2777, 4096, 1024),
        ]
        .into_iter()
        .map(|(m, k, n)| Shape { m, k, n })
        .collect()
    } else {
        dims.chunks(3)
            .map(|c| {
                let v = |i: usize| c.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                Shape {
                    m: v(0),
                    k: v(1),
                    n: v(2),
                }
            })
            .collect()
    };
    if groups.is_empty() {
        groups = vec![32];
    }
    for &g in &groups {
        assert!(
            g >= 32 && g.is_multiple_of(32),
            "group {g} must be a multiple of 32"
        );
    }

    let mut ctx = Context::new(threads)?;
    let mut ictx = I8Context::new(threads)?;
    let run_f16 = only != "int8";
    let run_i8 = only != "f16";

    println!(
        "q8_bench: threads={threads} reps={reps} groups={groups:?} only={only} \
         (f16 resident vs int8 group-wise, activation quantize included)"
    );
    let mut table: Vec<Row> = Vec::new();

    for sh in shapes {
        let Shape { m, k, n } = sh;
        let pm = (m.div_ceil(4) * 4).max(4);
        let sample = m.min(sample_rows);
        println!("\n== M={m} (pad {pm}) K={k} N={n} ==");
        let mut seed =
            0x9e37_79b9_7f4a_7c15u64 ^ ((m as u64) << 40) ^ ((n as u64) << 16) ^ k as u64;
        let a32: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed) * 2.0).collect();
        let w32: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();

        // The stored weight: Q8_0 blocks (the deployed GGUF format) at group 32.
        let (wq8, ws8) = q8_0_quantize(&w32, n, k);
        let wdq8 = dequantize(&wq8, &ws8, n, k, 32);
        let (q8cos, _, _) = compare(&wdq8, &w32);
        println!(
            "  weight: f16 {:.2} MiB | Q8_0 g32 {:.2} MiB ({:.2}x smaller) | \
             Q8_0 weight cosine vs exact f32 {q8cos:.8}",
            (n * k * 2) as f64 / (1u64 << 20) as f64,
            (n * k / 32 * 34) as f64 / (1u64 << 20) as f64,
            (n * k * 2) as f64 / (n * k / 32 * 34) as f64,
        );

        for &group in &groups {
            // The group's stored weight: group 32 is the Q8_0 file itself, wider
            // groups are a requant of the same dequantized values.
            let (bq, bs, bf16) = if group == 32 {
                (wq8.clone(), ws8.clone(), wdq8.clone())
            } else {
                let (q, s) = requantize(&wdq8, n, k, group);
                let dq = dequantize(&q, &s, n, k, group);
                (q, s, dq)
            };
            let bq16: Vec<f16> = bf16.iter().map(|&v| f16::from_f32(v)).collect();
            let bdq: Vec<f32> = bq16.iter().map(|v| v.to_f32()).collect();
            let (wcos, _, _) = compare(&bdq, &w32);

            let t = Instant::now();
            let cref = reference(&a32, &bdq, k, n, sample);
            println!(
                "  -- group {group}: stored weight cosine {wcos:.8} | f64 reference over \
                 {sample}/{m} rows in {:.2}s (readback {:.1} MiB/row)",
                t.elapsed().as_secs_f64(),
                (k / group) as f64 * n as f64 * 4.0 / (1u64 << 20) as f64
            );

            // ---- f16 resident arm (the deployed `--npu` projection path) ----
            let mut c16_out: Vec<f32> = Vec::new();
            let f16_arm = if run_f16 {
                let t = Instant::now();
                let w = ctx.pack_weight(PACK_M, k, n, &bq16)?;
                let f16_pack = t.elapsed().as_secs_f64() * 1e3;
                let f16_bo = w.bytes();
                let mut walls = Vec::with_capacity(reps);
                let mut cpus = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let c0 = cpu_s();
                    let t = Instant::now();
                    let a16 = f32_to_f16_par(&a32, threads);
                    let a16 = pad_rows_f16(&a16, m, k, pm);
                    let mut c16 = vec![f16::ZERO; pm * n];
                    ctx.matmul_prepacked(pm, k, n, &a16, &mut c16, &w)?;
                    c16_out = f16_to_f32_par(&c16, threads);
                    walls.push(t.elapsed().as_secs_f64() * 1e3);
                    cpus.push((cpu_s() - c0) * 1e3);
                }
                let f16_wall = median(walls);
                let f16_cpu = median(cpus);
                let (cos, max, _) = compare(&c16_out[..sample * n], &cref);
                println!(
                    "  f16  : pack {f16_pack:8.2} ms | wall {f16_wall:8.2} ms cpu {f16_cpu:8.2} ms \
                     | bo {:.1} MiB | cosine {cos:.8} max|d| {max:.2e}",
                    f16_bo as f64 / (1u64 << 20) as f64
                );
                Some((f16_wall, f16_cpu, f16_bo))
            } else {
                None
            };
            let (f16_wall, f16_cpu, f16_bo) = f16_arm.unwrap_or((f64::NAN, f64::NAN, 0));

            // ---- int8 group-wise arm ----
            if run_i8 {
                let t = Instant::now();
                let w = ictx.pack_weight_gw(PACK_M, k, n, &bq, group)?;
                let i8_pack = t.elapsed().as_secs_f64() * 1e3;
                let i8_bo = w.bytes();
                let ng = k / group;
                let mut walls = Vec::with_capacity(reps);
                let mut cpus = Vec::with_capacity(reps);
                let mut quants = Vec::with_capacity(reps);
                let mut npus = Vec::with_capacity(reps);
                let mut out: Vec<f32> = Vec::new();
                for _ in 0..reps {
                    let c0 = cpu_s();
                    let t = Instant::now();
                    let mut q = vec![0i8; pm * k];
                    let mut sa = vec![1f32; pm * ng];
                    quantize_rows_par(&a32, m, k, group, &mut q, &mut sa, threads);
                    let tq = t.elapsed().as_secs_f64() * 1e3;

                    let t = Instant::now();
                    let mut cf = vec![0f32; pm * n];
                    ictx.matmul_gw(pm, k, n, &q, &sa, &bs, &mut cf, &w)?;
                    let npu = t.elapsed().as_secs_f64() * 1e3;
                    quants.push(tq);
                    npus.push(npu);
                    walls.push(tq + npu);
                    cpus.push((cpu_s() - c0) * 1e3);
                    out = cf;
                }
                let i8_wall = median(walls);
                let i8_cpu = median(cpus);
                let i8_quant = median(quants);
                let i8_npu = median(npus);
                let (cos, max, _) = compare(&out[..sample * n], &cref);
                let cross = if c16_out.is_empty() {
                    String::new()
                } else {
                    format!(
                        " | vs f16 arm {:.8}",
                        compare(&out[..sample * n], &c16_out[..sample * n]).0
                    )
                };
                println!(
                    "  int8 : pack {i8_pack:8.2} ms | wall {i8_wall:8.2} ms cpu {i8_cpu:8.2} ms \
                     (quantize {i8_quant:.2} + npu {i8_npu:.2}) | bo {:.1} MiB | \
                     cosine {cos:.8} max|d| {max:.2e} | gate {}{cross}",
                    i8_bo as f64 / (1u64 << 20) as f64,
                    if cos >= 0.9995 { "PASS" } else { "FAIL" }
                );
                if f16_wall.is_finite() {
                    println!(
                        "  ratio int8/f16: wall {:.2}x | cpu {:.2}x",
                        i8_wall / f16_wall,
                        i8_cpu / f16_cpu
                    );
                }
                table.push(Row {
                    m,
                    k,
                    n,
                    group,
                    f16_wall,
                    f16_cpu,
                    i8_wall,
                    i8_cpu,
                    i8_quant,
                    f16_bo,
                    i8_bo,
                    cos,
                    max,
                });
            } else if run_f16 {
                table.push(Row {
                    m,
                    k,
                    n,
                    group,
                    f16_wall,
                    f16_cpu,
                    i8_wall: f64::NAN,
                    i8_cpu: f64::NAN,
                    i8_quant: f64::NAN,
                    f16_bo,
                    i8_bo: 0,
                    cos: f64::NAN,
                    max: f64::NAN,
                });
            }
        }
    }

    println!("\n== summary (ms, median of {reps}) ==");
    println!(
        "      M     K     N | grp |   f16 wall  f16 cpu |  int8 wall i8 cpu i8 quant | ratio | \
         int8 cosine     max|d| | bo MiB f16/i8"
    );
    for r in &table {
        let ratio = if r.i8_wall.is_finite() && r.f16_wall.is_finite() {
            format!("{:.2}x", r.i8_wall / r.f16_wall)
        } else {
            "-".to_string()
        };
        println!(
            "  {:>5} {:>5} {:>5} | {:>3} | {:>10.2} {:>7.2} | {:>10.2} {:>6.2} {:>8.2} | {:>6} | \
             {:>12.8} {:>9.2e} | {:.1}/{:.1}",
            r.m,
            r.k,
            r.n,
            r.group,
            r.f16_wall,
            r.f16_cpu,
            r.i8_wall,
            r.i8_cpu,
            r.i8_quant,
            ratio,
            r.cos,
            r.max,
            r.f16_bo as f64 / (1u64 << 20) as f64,
            r.i8_bo as f64 / (1u64 << 20) as f64,
        );
    }
    Ok(())
}
