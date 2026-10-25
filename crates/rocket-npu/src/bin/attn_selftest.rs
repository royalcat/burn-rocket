//! On-device attention self-test against an f64 reference, plus the chained-batch A/B.
//!
//! Usage: `attn_selftest [tokens] [kv] [heads] [kv_heads] [--dk N] [--dv N]
//! [--chain 0|1] [--reps N] [--ref 0|1]`
//!
//! `ROCKET_FA_CHAIN` (the `--chain` flag) selects the per-head path (`0`) or the chained
//! group batching (`1`). With no `--chain` the test runs BOTH — interleaved, so a drifting
//! board cannot bias one mode — and asserts the outputs are bit-identical (the M1d
//! contract), reporting the median wall and CPU time of each. `--dk`/`--dv` pick the head
//! geometry (default 128/128; the embedding model's vision tower is 64/64 at 2394
//! patches).

use std::time::Instant;

use half::f16;
use rocket_npu::attn::Attention;

fn rnd(seed: &mut u64) -> f16 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    f16::from_f32(((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
}

#[allow(clippy::too_many_arguments)]
fn ref_attn(
    t: usize,
    kv: usize,
    dk: usize,
    dv: usize,
    h: usize,
    hkv: usize,
    scale: f32,
    q: &[f16],
    k: &[f16],
    v: &[f16],
    mask: Option<&[f16]>,
) -> Vec<f64> {
    let gqa = h / hkv;
    let mut out = vec![0.0f64; h * t * dv];
    for head in 0..h {
        let hk = head / gqa;
        for row in 0..t {
            let mut sc = vec![0.0f64; kv];
            let mut mx = f64::NEG_INFINITY;
            for j in 0..kv {
                let mut s = 0.0f64;
                for c in 0..dk {
                    s += q[(head * t + row) * dk + c].to_f64() * k[(hk * kv + j) * dk + c].to_f64();
                }
                let mut s = (s as f32) * scale;
                if let Some(m) = mask {
                    s += m[row * kv + j].to_f32();
                }
                if s < -30000.0 {
                    s = -30000.0;
                }
                sc[j] = s as f64;
                mx = mx.max(sc[j]);
            }
            let mut sum = 0.0f64;
            for x in &mut sc {
                *x = (*x - mx).exp();
                sum += *x;
            }
            for c in 0..dv {
                let mut a = 0.0f64;
                for j in 0..kv {
                    a += sc[j] * v[(hk * dv + c) * kv + j].to_f64();
                }
                out[(head * t + row) * dv + c] = a / sum;
            }
        }
    }
    out
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

/// Cosine and max abs deviation against the f64 reference.
fn compare(got: &[f16], want: &[f64]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let g = g.to_f64();
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), max)
}

/// One attention call with the test's fixed arguments (softcap 0, the causal mask).
struct Case<'a> {
    t: usize,
    kv: usize,
    dk: usize,
    dv: usize,
    heads: usize,
    hkv: usize,
    scale: f32,
    q: &'a [f16],
    k: &'a [f16],
    v: &'a [f16],
    mask: &'a [f16],
}

impl Case<'_> {
    fn run(&self, attn: &mut Attention, out: &mut [f16]) -> rocket_npu::Result<()> {
        attn.flash(
            self.t,
            self.kv,
            self.dk,
            self.dv,
            self.heads,
            self.hkv,
            self.scale,
            0.0,
            self.q,
            self.k,
            self.v,
            Some(self.mask),
            out,
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Positional `tokens kv heads kv_heads`, then the flags.
    let mut pos = Vec::new();
    let mut chain: Option<bool> = None;
    let mut dk = 128usize;
    let mut dv = 0usize;
    let mut reps = 3usize;
    let mut want_ref: Option<bool> = None;
    let mut mem = false;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        let mut value = || -> Result<String, Box<dyn std::error::Error>> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value").into())
        };
        match arg {
            "--chain" => chain = Some(value()?.trim() != "0"),
            "--dk" => dk = value()?.parse()?,
            "--dv" => dv = value()?.parse()?,
            "--reps" => reps = value()?.parse::<usize>()?.max(1),
            "--ref" => want_ref = Some(value()?.trim() != "0"),
            "--mem" => mem = true,
            "--help" | "-h" => {
                println!(
                    "usage: attn_selftest [tokens] [kv] [heads] [kv_heads] \
                     [--dk N] [--dv N] [--chain 0|1] [--reps N] [--ref 0|1] [--mem]"
                );
                return Ok(());
            }
            other => pos.push(other.to_string()),
        }
        i += 1;
    }
    let num = |n: usize, d: usize| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(match pos.get(n) {
            Some(s) => s.parse()?,
            None => d,
        })
    };
    let t = num(0, 82)?;
    let kv = num(1, t)?;
    let heads = num(2, 8)?;
    let hkv = num(3, heads)?;
    if dv == 0 {
        dv = dk;
    }
    let scale = 1.0 / (dk as f32).sqrt();
    println!("attn T={t} kv={kv} heads={heads} kv_heads={hkv} dk={dk} dv={dv} reps={reps}");

    let mut seed = 0xdead_beef_1234_5678u64;
    let q: Vec<f16> = (0..heads * t * dk).map(|_| rnd(&mut seed)).collect();
    let k: Vec<f16> = (0..hkv * kv * dk).map(|_| rnd(&mut seed)).collect();
    let v: Vec<f16> = (0..hkv * dv * kv).map(|_| rnd(&mut seed)).collect();
    let mut mask = vec![f16::ZERO; t * kv];
    for row in 0..t {
        for j in 0..kv {
            if j > row {
                mask[row * kv + j] = f16::NEG_INFINITY;
            }
        }
    }
    let case = Case {
        t,
        kv,
        dk,
        dv,
        heads,
        hkv,
        scale,
        q: &q,
        k: &k,
        v: &v,
        mask: &mask,
    };

    // The f64 oracle is O(heads * t * kv * dk); skip it when it would dominate the run
    // (the chained A/B does not need it).
    let ref_work = (heads * t * kv * dk) as f64;
    let do_ref = want_ref.unwrap_or(ref_work <= 2e9);
    if !do_ref {
        println!("reference: skipped ({ref_work:.1e} MACs; --ref 1 forces it)");
    }
    let want = if do_ref {
        Some(ref_attn(
            t,
            kv,
            dk,
            dv,
            heads,
            hkv,
            scale,
            &q,
            &k,
            &v,
            Some(&mask),
        ))
    } else {
        None
    };

    let threads = std::env::var("ROCKET_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let mut attn = Attention::new(threads)?;
    let modes: Vec<bool> = match chain {
        Some(c) => vec![c],
        None => vec![false, true],
    };

    // The knob is resolved per call, so one context A/Bs both paths. Warm both first
    // (opens the worker fds, sizes the scratch, packs the batch contexts), then
    // interleave the timed reps so a drifting board hits both modes alike.
    let mut outs: Vec<Vec<f16>> = modes
        .iter()
        .map(|_| vec![f16::ZERO; heads * t * dv])
        .collect();
    let mut walls: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
    let mut cpus: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
    for (m, &mode) in modes.iter().enumerate() {
        unsafe { std::env::set_var("ROCKET_FA_CHAIN", if mode { "1" } else { "0" }) };
        case.run(&mut attn, &mut outs[m])?;
    }
    for _ in 0..reps {
        for (m, &mode) in modes.iter().enumerate() {
            unsafe { std::env::set_var("ROCKET_FA_CHAIN", if mode { "1" } else { "0" }) };
            let cpu0 = cpu_s();
            let t0 = Instant::now();
            case.run(&mut attn, &mut outs[m])?;
            walls[m].push(t0.elapsed().as_secs_f64() * 1e3);
            cpus[m].push((cpu_s() - cpu0) * 1e3);
        }
    }
    for (m, &mode) in modes.iter().enumerate() {
        println!(
            "chain={} : wall {:9.2} ms  cpu {:9.2} ms  (median of {reps})",
            u8::from(mode),
            median(walls[m].clone()),
            median(cpus[m].clone())
        );
        if let Some(want) = &want {
            let (cos, max) = compare(&outs[m], want);
            println!("            cosine {cos:.8} max_abs {max:.6}");
        }
    }

    if modes.len() == 2 {
        let (per_head, chained) = (&outs[0], &outs[1]);
        let diff = per_head
            .iter()
            .zip(chained)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = per_head
            .iter()
            .zip(chained)
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0.0f32, f32::max);
        println!(
            "bit-identical: {} ({diff} differing of {}, max_abs {max:.6})",
            if diff == 0 { "YES" } else { "NO" },
            per_head.len()
        );
        let ratio = |v: &[Vec<f64>]| {
            let (a, b) = (median(v[0].clone()), median(v[1].clone()));
            (a, b, if b > 0.0 { a / b } else { 0.0 })
        };
        let (wa, wb, wr) = ratio(&walls);
        let (ca, cb, cr) = ratio(&cpus);
        println!("wall: per-head {wa:.2} ms, chained {wb:.2} ms, {wr:.2}x");
        println!("cpu:  per-head {ca:.2} ms, chained {cb:.2} ms, {cr:.2}x");
        if diff != 0 {
            return Err("chained attention is not bit-identical to the per-head path".into());
        }
    }

    if mem {
        report_mem(&attn, "after the timed runs");
        // A tiny request drops the working shape, which is what `trim_scratch` keys on.
        let (t3, kv3) = (3usize, 3usize);
        let q3: Vec<f16> = (0..heads * t3 * dk).map(|_| rnd(&mut seed)).collect();
        let k3: Vec<f16> = (0..hkv * kv3 * dk).map(|_| rnd(&mut seed)).collect();
        let v3: Vec<f16> = (0..hkv * dv * kv3).map(|_| rnd(&mut seed)).collect();
        let mask3 = vec![f16::ZERO; t3 * kv3];
        let tiny = Case {
            t: t3,
            kv: kv3,
            dk,
            dv,
            heads,
            hkv,
            scale,
            q: &q3,
            k: &k3,
            v: &v3,
            mask: &mask3,
        };
        let mut out3 = vec![f16::ZERO; heads * t3 * dv];
        tiny.run(&mut attn, &mut out3)?;
        report_mem(&attn, "after a 3-token request, before trim");
        attn.trim_scratch();
        report_mem(&attn, "after trim_scratch");
    }
    Ok(())
}

/// One attention context's live scratch, split the way `burn_rocket::mem_stats` reports it.
fn report_mem(attn: &Attention, when: &str) {
    let m = attn.mem_stats();
    println!(
        "mem {when}: bo {:.1} MiB, host {:.1} MiB",
        m.bo_bytes as f64 / (1 << 20) as f64,
        m.host_bytes as f64 / (1 << 20) as f64
    );
    for (i, w) in m.workers.iter().enumerate() {
        println!(
            "  worker {i}: Tp {} Kn {} bo {:.1} MiB host {:.1} MiB",
            w.shape.0,
            w.shape.1,
            w.bo_bytes as f64 / (1 << 20) as f64,
            w.host_bytes as f64 / (1 << 20) as f64
        );
    }
}
