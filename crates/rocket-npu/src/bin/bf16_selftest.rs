//! On-device native-bf16 self-test: the streaming engine (`Bf16Stream`, the C
//! library's shape — both operands re-packed per call) against the resident-weight
//! engine (`Bf16Context`/`Bf16Weight`, the Rust extension) and an f64 reference.
//!
//!   bf16_selftest [--stream] [--canon] [--resident] [--f16] [--no-ref]
//!                 [--threads N] [--reps R] [M K N]
//!
//! With no shape argument the built-in ladder runs `M = 4, 64, 256, 512` at
//! `K = 1024, N = 768` and packs the resident weight **once** for the whole ladder;
//! `M K N` runs that single shape. With no arm flag every arm runs.
//!
//! The arms:
//! - `--stream` — [`Bf16Stream`]: the C planner follows the call's `M`, so `Kt` (and
//!   with it the weight cube layout) changes with `M` and the weight is re-scattered
//!   every call.
//! - `--canon`  — [`Bf16Stream::new_canonical`]: the same stream over the pinned
//!   canonical tiling, i.e. the resident weight's layout. This is the arm that isolates
//!   the resident weight from the tiling choice: it must be **bit-identical** to the
//!   resident arm at every `M`.
//! - `--resident` — [`Bf16Context`] + [`Bf16Weight`]: pack once per `(K,N)`, then only A
//!   is packed per call. Bit-identical to `--canon`; against the plain stream it is the
//!   same product within the K-partition reassociation (identical too when
//!   `M >= max_tile`, where the two plans coincide).
//! - `--f16` — the fp16 [`Context`] (the app's default resident path) at the same shape,
//!   reported as a cosine only: different numerics by construction (11-bit vs 8-bit
//!   mantissa), so no bit comparison.

use std::time::Instant;

use half::f16;
use rocket_npu::matmul::plan::{BF16, plan_geom};
use rocket_npu::matmul::{Bf16Context, Bf16Stream, Bf16Weight, Context, Plan, resident_plan};

/// The one-line synopsis (also printed by `--help`).
const USAGE: &str = "bf16_selftest [--stream] [--canon] [--resident] [--f16] [--no-ref] \
                     [--threads N] [--reps R] [M [K [N]]]";

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn bf16(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
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

/// Cosine, max |difference| and the count of differing bits, over f32 outputs.
fn compare(got: &[f32], want: &[f32]) -> (f64, f64, usize) {
    let (mut dot, mut na, mut nb, mut max, mut diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
        diff += (g != w) as usize;
    }
    (dot / (na.sqrt() * nb.sqrt()), max, diff)
}

struct Arms {
    stream: bool,
    canon: bool,
    resident: bool,
    f16: bool,
    reference: bool,
    threads: usize,
    reps: usize,
}

/// A callable arm: the engine call over one shape's operands.
type Arm<'a> = (
    &'a str,
    Box<dyn FnMut(&mut [f32]) -> rocket_npu::Result<()> + 'a>,
);

/// One measured arm: the last output, the median wall/CPU of its calls.
struct Timing {
    out: Vec<f32>,
    wall_ms: f64,
    cpu_ms: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut want = Arms {
        stream: false,
        canon: false,
        resident: false,
        f16: false,
        reference: true,
        threads: 5,
        reps: 2,
    };
    let mut picked = false;
    let mut positional: Vec<usize> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--stream" => {
                want.stream = true;
                picked = true;
            }
            "--canon" => {
                want.canon = true;
                picked = true;
            }
            "--resident" => {
                want.resident = true;
                picked = true;
            }
            "--f16" => {
                want.f16 = true;
                picked = true;
            }
            "--no-ref" => want.reference = false,
            "--help" | "-h" => {
                println!(
                    "{}\n\narms: --stream (C tiling) --canon (canonical tiling) \
                     --resident (Bf16Context) --f16 (fp16 Context, cosine only)\n\
                     opts: --no-ref --threads N --reps R\n\
                     shape: [M [K [N]]] (default: the M = 4,64,256,512 ladder at \
                     K=1024 N=768 and K=4096 N=1024)\n\
                     no arm flag = every arm",
                    USAGE
                );
                return Ok(());
            }
            "--threads" => {
                want.threads = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--threads needs a value")?
            }
            "--reps" => {
                want.reps = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--reps needs a value")?
            }
            other => positional.push(other.parse::<usize>()?),
        }
    }
    if !picked {
        want.stream = true;
        want.canon = true;
        want.resident = true;
        want.f16 = true;
    }

    // (K, N) -> the M ladder. The resident weight is packed once per (K, N).
    let groups: Vec<(usize, usize, Vec<usize>)> = match positional.len() {
        0 => vec![
            (1024, 768, vec![4, 64, 256, 512]),
            (4096, 1024, vec![4, 64, 256, 512]),
        ],
        1 => vec![(1024, 768, vec![positional[0]])],
        _ => {
            let k = *positional.get(1).unwrap_or(&1024);
            let n = *positional.get(2).unwrap_or(&768);
            vec![(k, n, vec![positional[0]])]
        }
    };

    println!(
        "bf16 selftest: threads={} reps={} arms stream={} canon={} resident={} f16={} ref={}",
        want.threads, want.reps, want.stream, want.canon, want.resident, want.f16, want.reference
    );

    let mut stream = want
        .stream
        .then(|| Bf16Stream::new(want.threads))
        .transpose()?;
    let mut canon = want
        .canon
        .then(|| Bf16Stream::new_canonical(want.threads))
        .transpose()?;
    let mut resident = want
        .resident
        .then(|| Bf16Context::new(want.threads))
        .transpose()?;
    let mut f16ctx = want.f16.then(|| Context::new(want.threads)).transpose()?;

    let mut totals = [0.0f64; 4]; // stream calls, canon calls, resident pack, resident calls
    let mut calls = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (k, n, ladder) in &groups {
        let (k, n) = (*k, *n);
        let plan = resident_plan(k, n)?;
        println!(
            "\n=== bf16 K={k} N={n}: resident plan Mt={} Kt={} Nt={} nMt={} nKt={} nNt={} wt_slot={}",
            plan.mt, plan.kt, plan.nt, plan.nmt, plan.nkt, plan.nnt, plan.wt_slot
        );

        let mut seed = 0x1234_5678_9abc_def1u64 ^ ((k as u64) << 32) ^ n as u64;
        let b: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();
        let b_bf: Vec<f32> = b.iter().map(|v| bf16(*v)).collect();

        let pack_t0 = Instant::now();
        let weight: Option<Bf16Weight> = match resident.as_mut() {
            Some(ctx) => Some(ctx.pack_weight(k, n, &b)?),
            None => None,
        };
        let pack_ms = pack_t0.elapsed().as_secs_f64() * 1e3;
        totals[2] += pack_ms;
        if let Some(w) = &weight {
            println!(
                "resident weight packed once in {pack_ms:.2} ms: {:.3} MiB BO (host f32 weight {:.3} MiB)",
                w.bytes() as f64 / (1 << 20) as f64,
                (n * k * 4) as f64 / (1 << 20) as f64
            );
        }

        let mut g_stream = 0.0f64;
        let mut g_resident = 0.0f64;
        let mut g_calls = 0usize;

        // The fp16 resident arm (the app's default path) packs once per (K, N) as well.
        let f16w = match f16ctx.as_mut() {
            Some(ctx) => {
                let b16: Vec<f16> = b.iter().map(|v| f16::from_f32(*v)).collect();
                let pack_m = ladder.iter().copied().max().unwrap_or(256).max(256);
                Some((ctx.pack_weight(pack_m, k, n, &b16)?, pack_m))
            }
            None => None,
        };

        for &m in ladder {
            let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
            let call = Plan::canonical_geom(&rocket_npu::matmul::plan::BF16, m, k, n)?;
            println!(
                "--- M={m}: call plan Mt={} nMt={} (pinned Kt={} nKt={})",
                call.mt, call.nmt, call.kt, call.nkt
            );

            // The arms are sampled interleaved (ABBA) and reported by median: this
            // board's run-to-run scatter on a shared NPU is +-40 %, so two sequential
            // arms would not be comparable.
            let mut arms: Vec<Arm> = Vec::new();
            let (ar, br) = (&a, &b);
            if let Some(s) = stream.as_mut() {
                arms.push((
                    "stream",
                    Box::new(move |o: &mut [f32]| s.matmul(m, k, n, ar, br, o)),
                ));
            }
            if let Some(s) = canon.as_mut() {
                arms.push((
                    "canon",
                    Box::new(move |o: &mut [f32]| s.matmul(m, k, n, ar, br, o)),
                ));
            }
            if let (Some(ctx), Some(w)) = (resident.as_mut(), weight.as_ref()) {
                arms.push((
                    "resident",
                    Box::new(move |o: &mut [f32]| ctx.matmul(m, k, n, ar, w, o)),
                ));
            }
            let mut outs: Vec<Vec<f32>> = arms.iter().map(|_| vec![0f32; m * n]).collect();
            for (i, (_, call)) in arms.iter_mut().enumerate() {
                call(&mut outs[i])?; // warmup: the shape's scratch set + first plan
            }
            let mut walls: Vec<Vec<f64>> = arms.iter().map(|_| Vec::new()).collect();
            let mut cpus: Vec<Vec<f64>> = arms.iter().map(|_| Vec::new()).collect();
            for rep in 0..want.reps.max(1) {
                let order: Vec<usize> = if rep % 2 == 0 {
                    (0..arms.len()).collect()
                } else {
                    (0..arms.len()).rev().collect()
                };
                for i in order {
                    let c0 = cpu_s();
                    let t = Instant::now();
                    arms[i].1(&mut outs[i])?;
                    walls[i].push(t.elapsed().as_secs_f64() * 1e3);
                    cpus[i].push((cpu_s() - c0) * 1e3);
                }
            }
            let rows: Vec<(&str, Timing)> = arms
                .iter()
                .map(|(name, _)| *name)
                .zip(outs)
                .zip(walls.into_iter().zip(cpus))
                .map(|((name, out), (w, c))| {
                    (
                        name,
                        Timing {
                            out,
                            wall_ms: median(w),
                            cpu_ms: median(c),
                        },
                    )
                })
                .collect();
            g_calls += 1;
            for (name, row) in &rows {
                match *name {
                    "stream" => {
                        totals[0] += row.wall_ms;
                        g_stream += row.wall_ms;
                    }
                    "canon" => totals[1] += row.wall_ms,
                    _ => {
                        totals[3] += row.wall_ms;
                        g_resident += row.wall_ms;
                    }
                }
            }

            if want.reference {
                let t0 = Instant::now();
                let want_ref = f64_reference(m, k, n, &a, &b_bf);
                let ref_ms = t0.elapsed().as_secs_f64() * 1e3;
                for (name, row) in &rows {
                    let (cos, max, _) = compare(&row.out, &want_ref);
                    let verdict = if cos >= 0.99999 { "ok" } else { "FAIL" };
                    if cos < 0.99999 {
                        failures.push(format!("M={m} {name} vs f64 ref cos={cos:.8}"));
                    }
                    println!(
                        "  {name:<9} {:>8.2} ms  cpu {:>8.2} ms  cos {cos:.8}  max_abs {max:.3e}  {verdict}",
                        row.wall_ms, row.cpu_ms
                    );
                }
                println!("  (f64 reference {ref_ms:.0} ms)");
            } else {
                for (name, row) in &rows {
                    println!(
                        "  {name:<9} {:>8.2} ms  cpu {:>8.2} ms",
                        row.wall_ms, row.cpu_ms
                    );
                }
            }

            // The bit-identity gates: the resident weight against the canonical stream
            // (same tiling, same programs) must match exactly, and against the C-tiling
            // stream it is the same product within the K-partition reassociation.
            let get = |name: &str| rows.iter().find(|(n, _)| *n == name).map(|(_, t)| &t.out);
            if let (Some(res), Some(can)) = (get("resident"), get("canon")) {
                let (cos, max, diff) = compare(res, can);
                let bit = diff == 0;
                println!(
                    "  resident vs canon:  {} ({}/{} values, cos {cos:.9}, max_abs {max:.3e})",
                    if bit { "BIT-IDENTICAL" } else { "DIFFERS" },
                    res.len() - diff,
                    res.len()
                );
                if !bit {
                    failures.push(format!("M={m} resident vs canon-stream not bit-identical"));
                }
            }
            if let (Some(res), Some(stream)) = (get("resident"), get("stream")) {
                let (cos, max, diff) = compare(res, stream);
                println!(
                    "  resident vs stream: {} differing values, cos {cos:.9}, max_abs {max:.3e}{}",
                    diff,
                    if diff == 0 {
                        " (identical plans at this M)"
                    } else {
                        " (K-partition reassociation)"
                    }
                );
            }

            if let (Some(ctx), Some((w, _)), Some(res)) =
                (f16ctx.as_mut(), f16w.as_ref(), get("resident"))
            {
                let a16: Vec<f16> = a.iter().map(|v| f16::from_f32(*v)).collect();
                let mut out = vec![f16::ZERO; m * n];
                ctx.matmul_prepacked(m, k, n, &a16, &mut out, w)?;
                let out: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
                let (cos, max, _) = compare(res, &out);
                println!(
                    "  f16 Context vs resident: cos {cos:.8} max_abs {max:.3e} (numerics only)"
                );
            }
            calls += 1;
        }

        if let Some(w) = &weight {
            let per_shape: Vec<String> = ladder
                .iter()
                .map(|m| {
                    format!(
                        "M={m}: {:.2} MiB",
                        stream_weight_bytes(*m, k, n, want.threads) as f64 / (1 << 20) as f64
                    )
                })
                .collect();
            let total: usize = ladder
                .iter()
                .map(|m| stream_weight_bytes(*m, k, n, want.threads))
                .sum();
            println!(
                "  weight memory: resident {:.3} MiB for every M | stream {} = {:.2} MiB over the ladder",
                w.bytes() as f64 / (1 << 20) as f64,
                per_shape.join(", "),
                total as f64 / (1 << 20) as f64
            );
        }

        if weight.is_some() && stream.is_some() && g_calls > 0 {
            let (per_stream, per_res) = (g_stream / g_calls as f64, g_resident / g_calls as f64);
            let save = per_stream - per_res;
            println!(
                "  K={k} N={n} per call: stream {per_stream:.2} ms vs resident {per_res:.2} ms ({:.2}x){}",
                per_res / per_stream,
                if save > 0.0 {
                    format!(
                        "; the one-time {pack_ms:.1} ms pack breaks even after {:.1} calls",
                        pack_ms / save
                    )
                } else {
                    String::from("; no per-call gain")
                }
            );
        }
    }

    if want.stream && want.resident {
        let per_call = totals[3] / calls as f64;
        println!(
            "\ntotals: stream {:.2} ms ({:.2}/call) | resident pack {:.2} ms + {:.2} ms calls = {:.2} ms total",
            totals[0],
            totals[0] / calls as f64,
            totals[2],
            totals[3],
            totals[2] + totals[3]
        );
        println!(
            "resident vs stream (pack included): {:.2}x  |  per call {:.2} vs {:.2} ms",
            (totals[2] + totals[3]) / totals[0],
            per_call,
            totals[0] / calls as f64
        );
    }
    if want.canon && want.resident {
        println!(
            "canonical stream {:.2} ms vs resident {:.2} ms per call ({:.2}x)",
            totals[1] / calls as f64,
            totals[3] / calls as f64,
            totals[3] / totals[1]
        );
    }
    if !failures.is_empty() {
        eprintln!("FAILURES:");
        for f in &failures {
            eprintln!("  {f}");
        }
        std::process::exit(1);
    }
    println!("RESULT: PASS");
    Ok(())
}

/// The streaming arm's weight BOs for one `(M,K,N)` shape: the stream keeps one
/// per-worker weight scratch *per shape*, so a ladder of `M` values holds one set each —
/// what the resident weight replaces with a single set for all `M`.
fn stream_weight_bytes(m: usize, k: usize, n: usize, threads: usize) -> usize {
    let step = (n.div_ceil(threads.max(1)).div_ceil(16) * 16).max(16);
    let mut total = 0;
    for t in 0..threads.max(1) {
        let n0 = t * step;
        if n0 >= n {
            break;
        }
        let nsub = (n - n0).min(step);
        let Ok((mt, kt, nt)) = plan_geom(&BF16, m, k, nsub, 0) else {
            continue;
        };
        let p = Plan::derive_geom(&BF16, m, k, nsub, mt, kt, nt);
        total += p.nnt * p.nkt * p.wt_slot * 2 + 32 * 1024;
    }
    total
}

/// The exact reference: f64 accumulation of the bf16-truncated products.
fn f64_reference(m: usize, k: usize, n: usize, a: &[f32], b_bf: &[f32]) -> Vec<f32> {
    let a_bf: Vec<f32> = a.iter().map(|v| bf16(*v)).collect();
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        let arow = &a_bf[i * k..][..k];
        for j in 0..n {
            let brow = &b_bf[j * k..][..k];
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += arow[p] as f64 * brow[p] as f64;
            }
            want[i * n + j] = sum as f32;
        }
    }
    want
}
