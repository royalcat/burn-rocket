//! On-device numeric self-test for the one-shot matmul surface.
//!
//!   selftest [M K N] [--context] [--f32out] [--bf16] [--mt N] [--batch N]
//!
//! The fp16 one-shot check (the original CLI) always runs: it is the single-fd
//! baseline every other mode is compared against. In addition:
//!
//! - `--context` exercises the resident-weight path at a canonical pack M (256) and
//!   multiple call Ms;
//! - `--mt N` runs the multi-fd `_mt` fan-out (N column slices);
//! - `--f32out` runs the fp32-output path, so the fp16 reference shows its per-tile
//!   rounding and this one the true fp16-input dot product;
//! - `--bf16` runs the bf16 one-shot (`_mt` when `--mt` is given) and reports its
//!   fidelity against the exact f32 product of the same f32 operands;
//! - `--batch N` runs `N` independent items of the same shape through the batched
//!   matmul — once as the stateless `batch()` (create/run/free) and twice through a
//!   persistent `MmBatch` — and gates every item bit-for-bit against `N` independent
//!   `one_shot` calls. The batch path is CPU fp32 accumulation by definition, so this
//!   mode forces `ROCKET_NO_KACC=1` for the reference as well (KACC rounds in fp16 on
//!   purpose); `ROCKET_BATCH_SUBMIT=0|1` selects the gapped vs chained layout.
//!
//! Every mode compares against a reference accumulated in f64 from the same inputs and
//! prints the cosine / max error plus wall time.

use std::time::Instant;

use half::f16;
use rocket_npu::Device;
use rocket_npu::matmul::{
    Context, MmBatch, batch, one_shot, one_shot_bf16, one_shot_bf16_mt, one_shot_f32out,
    one_shot_mt,
};

fn rand_f32(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn rand_f16(seed: &mut u64) -> f16 {
    f16::from_f32(rand_f32(seed))
}

/// The exact `A[M,K] * B[N,K]^T` product in f64 (f16 and f32 operands both widen).
fn reference<T: Copy + Into<f64>>(m: usize, k: usize, n: usize, a: &[T], b: &[T]) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p].into() * b[j * k + p].into();
            }
            out[i * n + j] = sum;
        }
    }
    out
}

fn report(name: &str, got: &[f64], want: &[f64], ms: f64) {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let g = *g;
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    let ms = if ms.is_finite() {
        format!("{ms:.2} ms")
    } else {
        "-".to_string()
    };
    println!("{name}: cosine {cos:.8} max_abs {max:.6} {ms}");
}

fn report_f16(name: &str, got: &[f16], want: &[f64], ms: f64) {
    let got: Vec<f64> = got.iter().map(|v| v.to_f64()).collect();
    report(name, &got, want, ms);
}

fn report_f32(name: &str, got: &[f32], want: &[f64], ms: f64) {
    let got: Vec<f64> = got.iter().map(|v| *v as f64).collect();
    report(name, &got, want, ms);
}

/// Compare a batch of per-item outputs against a batch of references both bitwise (the
/// gate: every item must be byte-identical to an independent `one_shot`) and by cosine.
fn report_batch(name: &str, got: &[Vec<f16>], want: &[Vec<f16>], ms: f64) {
    let total: usize = want.iter().map(Vec::len).sum();
    let mut differ = 0usize;
    let mut first: Option<(usize, usize)> = None;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max = 0.0f64;
    for (item, (g, w)) in got.iter().zip(want).enumerate() {
        for (j, (gv, wv)) in g.iter().zip(w).enumerate() {
            if gv.to_bits() != wv.to_bits() {
                differ += 1;
                first.get_or_insert((item, j));
            }
            let (g, w) = (gv.to_f64(), wv.to_f64());
            dot += g * w;
            na += g * g;
            nb += w * w;
            max = max.max((g - w).abs());
        }
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    let verdict = match first {
        None => "bit-identical".to_string(),
        Some((item, j)) => format!("differs: {differ}/{total} (first item {item} elem {j})"),
    };
    println!("{name}: cosine {cos:.8} max_abs {max:.6} {verdict} {ms:.2} ms");
}

struct Args {
    dims: Vec<usize>,
    context: bool,
    f32out: bool,
    bf16: bool,
    mt: Option<usize>,
    batch: Option<usize>,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        dims: Vec::new(),
        context: false,
        f32out: false,
        bf16: false,
        mt: None,
        batch: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--context" => args.context = true,
            "--f32out" => args.f32out = true,
            "--bf16" => args.bf16 = true,
            "--mt" => {
                let v = it.next().ok_or("--mt needs a thread count")?;
                args.mt = Some(v.parse()?);
            }
            "--batch" => {
                let v = it.next().ok_or("--batch needs an item count")?;
                args.batch = Some(v.parse()?);
            }
            _ => args.dims.push(arg.parse()?),
        }
    }
    if args.dims.len() > 3 {
        return Err("at most three positional dimensions (M K N)".into());
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if args.batch.is_some() {
        // The batched path is CPU fp32 accumulation by definition, so put the per-item
        // reference on the same path; otherwise `one_shot` may take the KACC route and
        // the bit-for-bit gate would compare two deliberately different rounding orders.
        // SAFETY: single-threaded, before any engine call reads the knob.
        unsafe { std::env::set_var("ROCKET_NO_KACC", "1") };
        println!(
            "--batch: forcing ROCKET_NO_KACC=1 (the batch path accumulates in fp32 on the CPU)"
        );
    }
    let m = args.dims.first().copied().unwrap_or(64);
    let k = args.dims.get(1).copied().unwrap_or(256);
    let n = args.dims.get(2).copied().unwrap_or(128);
    println!("shape M={m} K={k} N={n}");

    let mut seed = 0x1234_5678_9abc_def1u64;
    let a: Vec<f16> = (0..m * k).map(|_| rand_f16(&mut seed)).collect();
    let b: Vec<f16> = (0..n * k).map(|_| rand_f16(&mut seed)).collect();
    let want = reference(m, k, n, &a, &b);

    let dev = Device::open()?;
    println!("driver {} caps {:?}", dev.driver_name(), dev.caps());
    if args.batch.is_some() {
        // Resolved once per process, so this one line reports both batch passes below.
        println!(
            "batch submit: {}",
            if rocket_npu::matmul::chain_enabled() {
                "chained (contiguous regcmd, one IRQ per group)"
            } else {
                "gapped per-task"
            }
        );
    }
    let mut c = vec![f16::ZERO; m * n];
    let t = Instant::now();
    one_shot(&dev, m, k, n, &a, &b, &mut c)?;
    report_f16("one_shot", &c, &want, t.elapsed().as_secs_f64() * 1e3);

    if let Some(threads) = args.mt {
        let t = Instant::now();
        one_shot_mt(m, k, n, &a, &b, &mut c, threads)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        report_f16(&format!("one_shot_mt({threads})"), &c, &want, ms);
    }

    if args.f32out {
        let mut cf = vec![0.0f32; m * n];
        let t = Instant::now();
        one_shot_f32out(&dev, m, k, n, &a, &b, &mut cf)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        report_f32("f32out", &cf, &want, ms);
        if m == 1 {
            println!("f32out: M=1 went through the height-4 pad (row 0 returned)");
        }
    }

    if args.bf16 {
        let a32: Vec<f32> = (0..m * k).map(|_| rand_f32(&mut seed)).collect();
        let b32: Vec<f32> = (0..n * k).map(|_| rand_f32(&mut seed)).collect();
        let want32 = reference(m, k, n, &a32, &b32);
        let mut c32 = vec![0.0f32; m * n];
        let mut name = "bf16".to_string();
        let t = Instant::now();
        match args.mt {
            Some(threads) => {
                one_shot_bf16_mt(m, k, n, &a32, &b32, &mut c32, threads)?;
                name = format!("bf16_mt({threads})");
            }
            None => one_shot_bf16(m, k, n, &a32, &b32, &mut c32)?,
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        // The reference is the exact f32 product, so this line reports the bf16
        // truncation error as well as any engine error.
        report_f32(&format!("{name} vs f32"), &c32, &want32, ms);
    }

    if args.context {
        let threads = std::env::var("ROCKET_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let mut ctx = Context::new(threads)?;
        let t = Instant::now();
        let w = ctx.pack_weight(256, k, n, &b)?;
        println!(
            "pack_weight(256): {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        ctx.matmul_prepacked(m, k, n, &a, &mut c, &w)?;
        rocket_npu::device::submit::reset_counters();
        let t = Instant::now();
        ctx.matmul_prepacked(m, k, n, &a, &mut c, &w)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        report_f16("prepacked", &c, &want, ms);
        println!(
            "submits {} tasks {}",
            rocket_npu::device::submit::ioctl_submit_count(),
            rocket_npu::device::submit::submitted_task_count()
        );
        if m != 4 {
            let a4 = &a[..4 * k];
            let want4 = reference(4, k, n, a4, &b);
            let mut c4 = vec![f16::ZERO; 4 * n];
            ctx.matmul_prepacked(4, k, n, a4, &mut c4, &w)?;
            let t = Instant::now();
            ctx.matmul_prepacked(4, k, n, a4, &mut c4, &w)?;
            report_f16("prepacked@4", &c4, &want4, t.elapsed().as_secs_f64() * 1e3);
        }
    }
    if let Some(nbatch) = args.batch {
        if nbatch == 0 {
            return Err("--batch needs at least one item".into());
        }
        if m == 1 {
            return Err(
                "--batch needs M >= 4: the batched path has no GEMV pad (only an \
                        nbatch of 1 falls through to the padded one-shot)"
                    .into(),
            );
        }
        let a_items: Vec<Vec<f16>> = (0..nbatch)
            .map(|_| (0..m * k).map(|_| rand_f16(&mut seed)).collect())
            .collect();
        let b_items: Vec<Vec<f16>> = (0..nbatch)
            .map(|_| (0..n * k).map(|_| rand_f16(&mut seed)).collect())
            .collect();
        let a_refs: Vec<&[f16]> = a_items.iter().map(Vec::as_slice).collect();
        let b_refs: Vec<&[f16]> = b_items.iter().map(Vec::as_slice).collect();

        // The gate: `N` independent single-matmul calls.
        let mut want: Vec<Vec<f16>> = vec![vec![f16::ZERO; m * n]; nbatch];
        let t = Instant::now();
        for (w, (a, b)) in want.iter_mut().zip(a_items.iter().zip(&b_items)) {
            one_shot(&dev, m, k, n, a, b, w)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "{nbatch} x one_shot: {ms:.2} ms ({:.2} ms/item)",
            ms / nbatch as f64
        );

        let mut got: Vec<Vec<f16>> = vec![vec![f16::ZERO; m * n]; nbatch];
        let ms = {
            let mut outs: Vec<&mut [f16]> = got.iter_mut().map(Vec::as_mut_slice).collect();
            let t = Instant::now();
            batch(&dev, m, k, n, &a_refs, &b_refs, &mut outs)?;
            t.elapsed().as_secs_f64() * 1e3
        };
        report_batch("batch()", &got, &want, ms);
        // Item 0 against the exact product in f64 (the reference above is only f16).
        report_f16(
            "batch()[0] vs f64",
            &got[0],
            &reference(m, k, n, &a_items[0], &b_items[0]),
            f64::NAN,
        );

        // The persistent context: the same layout twice, so the second run reclaims the
        // per-call BO zeroing that `batch()` pays for its transient context.
        let mut ctx = MmBatch::new(&dev)?;
        for pass in 0..2 {
            for v in got.iter_mut() {
                v.fill(f16::ZERO);
            }
            let ms = {
                let mut outs: Vec<&mut [f16]> = got.iter_mut().map(Vec::as_mut_slice).collect();
                let t = Instant::now();
                ctx.run(m, k, n, &a_refs, &b_refs, &mut outs)?;
                t.elapsed().as_secs_f64() * 1e3
            };
            report_batch(
                &format!("mm_batch run{pass} ({} MiB resident)", ctx.bytes() >> 20),
                &got,
                &want,
                ms,
            );
        }
        println!(
            "submits {} tasks {}",
            rocket_npu::device::submit::ioctl_submit_count(),
            rocket_npu::device::submit::submitted_task_count()
        );
    }
    Ok(())
}
