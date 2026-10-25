//! On-device attention self-tests: the stateless entries against the f64 host
//! reference, the encoder `mha_self` pair, the chained-batch A/B and the tiled
//! long-context A/B.
//!
//! Usage: `attn_selftest [tokens] [kv] [heads] [kv_heads] [--dk N] [--dv N]
//! [--chain 0|1] [--reps N] [--ref 0|1] [--mem] [--oneshot] [--mt N] [--mha]
//! [--softcap F] [--mask none|causal|window] [--tile KBP] [--tile-min KV] [--tile-only]`
//!
//! `ROCKET_FA_CHAIN` (the `--chain` flag) selects the per-head path (`0`) or the chained
//! group batching (`1`). With no `--chain` the default mode runs BOTH — interleaved, so
//! a drifting board cannot bias one mode — and asserts the outputs are bit-identical
//! (the M1d contract), reporting the median wall and CPU time of each.
//!
//! `--tile KBP` switches to the tiled long-context A/B instead: it sets
//! `ROCKET_FA_TILE_KV=KBP` (and `ROCKET_FA_TILE_MIN_KV`, `--tile-min`, default 8192),
//! runs the case materialized (tiling off) and tiled, reports each arm's wall/CPU,
//! its cosine/max-abs against the f64 reference and against the other arm, and the
//! scratch footprint each arm leaves behind (`--mem` adds the trim / 3-token release
//! check). The tiled arm is also run with `ROCKET_FA_CHAIN=0` to confirm the chained
//! path never engages under tiling (bit-identical output). `--tile-only` skips the
//! materialized arm entirely — use it at 16k+ keys, where the `[Tp,Kn]` score matrix
//! the tiled path exists to avoid is what would OOM.
//!
//! `--oneshot` instead runs the stateless entries (`rocket_flash_attn_fp16` and its
//! `_mt` fan-out, one context per call) and compares both with
//! [`flash_attn_ref`], asserting the fan-out is
//! bit-identical to the single-fd call. `--mha` runs the encoder composition
//! (`rocket_mha_self_fp16`, with `d = dk * heads`) against its f64 reference and
//! asserts the `nthreads` projection fan-out changes no bits. Both flags may be given
//! together; either one replaces the chained-batch bench.
//!
//! `--host-sm 0|1` sets `ROCKET_ATTN_HOST_SOFTMAX` (`0` = the on-NPU softmax,
//! `1`/unset = the host pass, the port's default) for every mode below. `--sm-ab`
//! instead runs BOTH placements of `flash_attn` and `mha_self` on one case, interleaved,
//! and reports each arm's wall/CPU and its cosine/max-abs against the f64 reference plus
//! the two arms against each other — the on-NPU `EXP` LUT is approximate, so that pair
//! is expected to differ by ~1e-3 in max-abs, not bit-wise.
//!
//! `--dk`/`--dv` pick the head geometry (default 128/128; the embedding model's vision
//! tower is 64/64 at 2394 patches). `--softcap` and `--mask` pick the score post-
//! processing (`window` is a bidirectional band of `tokens/8` keys).

use std::time::Instant;

use half::f16;
use rocket_npu::attn::{
    Attention, flash_attn, flash_attn_mt, flash_attn_ref, host_softmax, mha_self, mha_self_ref,
    tile_kv_width,
};

fn rnd(seed: &mut u64) -> f16 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    f16::from_f32(((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
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

/// Cosine and max abs deviation of two f16 matrices.
fn compare(got: &[f16], want: &[f16]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (g.to_f64(), w.to_f64());
        dot += g * w;
        na += g * g;
        nb += w * w;
        max = max.max((g - w).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), max)
}

/// `(differing elements, max |difference|)` by f16 bit pattern.
fn bit_diff(a: &[f16], b: &[f16]) -> (usize, f32) {
    let diff = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0f32, f32::max);
    (diff, max)
}

fn verdict(diff: usize, total: usize, max: f32) -> String {
    format!(
        "{} ({diff} differing of {total}, max_abs {max:.6})",
        if diff == 0 { "YES" } else { "NO" }
    )
}

/// One attention call with the test's fixed arguments.
struct Case<'a> {
    t: usize,
    kv: usize,
    dk: usize,
    dv: usize,
    heads: usize,
    hkv: usize,
    scale: f32,
    softcap: f32,
    q: &'a [f16],
    k: &'a [f16],
    v: &'a [f16],
    mask: Option<&'a [f16]>,
}

impl Case<'_> {
    fn out_len(&self) -> usize {
        self.heads * self.t * self.dv
    }

    fn run(&self, attn: &mut Attention, out: &mut [f16]) -> rocket_npu::Result<()> {
        attn.flash(
            self.t,
            self.kv,
            self.dk,
            self.dv,
            self.heads,
            self.hkv,
            self.scale,
            self.softcap,
            self.q,
            self.k,
            self.v,
            self.mask,
            out,
        )
    }

    fn run_oneshot(&self, out: &mut [f16]) -> rocket_npu::Result<()> {
        flash_attn(
            self.t,
            self.kv,
            self.dk,
            self.dv,
            self.heads,
            self.hkv,
            self.scale,
            self.softcap,
            self.q,
            self.k,
            self.v,
            self.mask,
            out,
        )
    }

    fn run_mt(&self, out: &mut [f16], threads: usize) -> rocket_npu::Result<()> {
        flash_attn_mt(
            self.t,
            self.kv,
            self.dk,
            self.dv,
            self.heads,
            self.hkv,
            self.scale,
            self.softcap,
            self.q,
            self.k,
            self.v,
            self.mask,
            out,
            threads,
        )
    }

    fn reference(&self) -> rocket_npu::Result<Vec<f16>> {
        let mut out = vec![f16::ZERO; self.out_len()];
        flash_attn_ref(
            self.t,
            self.kv,
            self.dk,
            self.dv,
            self.heads,
            self.hkv,
            self.scale,
            self.softcap,
            self.q,
            self.k,
            self.v,
            self.mask,
            &mut out,
        )?;
        Ok(out)
    }
}

/// The stateless entries: single fd, the head fan-out, and both against the reference.
fn oneshot_block(
    case: &Case,
    want: Option<&[f16]>,
    threads: usize,
    reps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // A row the mask empties is uniform over the *padded* key count on the device, so it
    // reads n_kv/Kn of the unpadded reference (the C behaviour). Say so when it happens,
    // instead of letting it look like a numerics failure.
    if let Some(mask) = case.mask {
        let empty = rocket_npu::attn::fully_masked_rows(mask, case.t, case.kv);
        if empty > 0 {
            let kn = case.kv.div_ceil(32) * 32;
            println!(
                "note: {empty} of {} rows are fully masked; the device spreads them over \
                 Kn={kn} columns (V=0 there), so those rows are {:.4}x the reference",
                case.t,
                case.kv as f64 / kn as f64
            );
        }
    }
    let (mut one, mut many) = (
        vec![f16::ZERO; case.out_len()],
        vec![f16::ZERO; case.out_len()],
    );
    // Warm both (opens the worker fds, sizes the scratch) before timing.
    case.run_oneshot(&mut one)?;
    case.run_mt(&mut many, threads)?;
    let (mut w1, mut wm, mut c1, mut cm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        let cpu = cpu_s();
        let t0 = Instant::now();
        case.run_oneshot(&mut one)?;
        w1.push(t0.elapsed().as_secs_f64() * 1e3);
        c1.push((cpu_s() - cpu) * 1e3);

        let cpu = cpu_s();
        let t0 = Instant::now();
        case.run_mt(&mut many, threads)?;
        wm.push(t0.elapsed().as_secs_f64() * 1e3);
        cm.push((cpu_s() - cpu) * 1e3);
    }
    for (label, out, walls, cpus) in [("single-fd", &one, &w1, &c1), ("mt", &many, &wm, &cm)] {
        let head = if label == "mt" {
            format!("mt({threads})")
        } else {
            label.to_string()
        };
        println!(
            "oneshot {head:<10}: wall {:9.2} ms  cpu {:9.2} ms  (median of {reps})",
            median(walls.clone()),
            median(cpus.clone())
        );
        if let Some(want) = want {
            let (cos, max) = compare(out, want);
            println!("                         cosine {cos:.8} max_abs {max:.6}");
        }
    }
    let (diff, max) = bit_diff(&one, &many);
    println!(
        "oneshot bit-identical single-fd vs mt: {}",
        verdict(diff, one.len(), max)
    );
    if diff != 0 {
        return Err("flash_attn_mt is not bit-identical to flash_attn".into());
    }
    Ok(())
}

/// The `mha_self` fixture: `x`, the four `[d, d]` projections and the C's Whisper biases
/// (`bq`, `bv`, `bo` present, `bk` NULL).
struct MhaCase {
    t: usize,
    d: usize,
    dh: usize,
    heads: usize,
    x: Vec<f16>,
    wq: Vec<f16>,
    wk: Vec<f16>,
    wv: Vec<f16>,
    wo: Vec<f16>,
    bq: Vec<f16>,
    bv: Vec<f16>,
    bo: Vec<f16>,
}

impl MhaCase {
    fn new(t: usize, dh: usize, heads: usize) -> Self {
        let d = dh * heads;
        let mut seed = 0x005e_ed0f_0a7e_5713_u64;
        let x: Vec<f16> = (0..t * d).map(|_| rnd(&mut seed)).collect();
        let mat = |n: usize, seed: &mut u64| -> Vec<f16> { (0..n).map(|_| rnd(seed)).collect() };
        let (wq, wk, wv, wo) = (
            mat(d * d, &mut seed),
            mat(d * d, &mut seed),
            mat(d * d, &mut seed),
            mat(d * d, &mut seed),
        );
        let (bq, bv, bo) = (mat(d, &mut seed), mat(d, &mut seed), mat(d, &mut seed));
        MhaCase {
            t,
            d,
            dh,
            heads,
            x,
            wq,
            wk,
            wv,
            wo,
            bq,
            bv,
            bo,
        }
    }

    fn run(&self, out: &mut [f16], threads: usize) -> rocket_npu::Result<()> {
        mha_self(
            self.t,
            self.d,
            self.heads,
            &self.x,
            &self.wq,
            Some(&self.bq),
            &self.wk,
            None,
            &self.wv,
            Some(&self.bv),
            &self.wo,
            Some(&self.bo),
            out,
            threads,
        )
    }

    fn reference(&self) -> rocket_npu::Result<Vec<f16>> {
        let mut want = vec![f16::ZERO; self.t * self.d];
        mha_self_ref(
            self.t,
            self.d,
            self.heads,
            &self.x,
            &self.wq,
            Some(&self.bq),
            &self.wk,
            None,
            &self.wv,
            Some(&self.bv),
            &self.wo,
            Some(&self.bo),
            &mut want,
        )?;
        Ok(want)
    }

    /// The f64 oracle's MAC count, for the skip decision.
    fn ref_work(&self) -> f64 {
        4.0 * self.t as f64 * (self.d * self.d) as f64
            + 2.0 * self.heads as f64 * (self.t * self.t * self.dh) as f64
    }
}

/// The encoder composition: `mha_self` with one and with `threads` projection workers,
/// both against the f64 reference.
fn mha_block(
    t: usize,
    dh: usize,
    heads: usize,
    threads: usize,
    reps: usize,
    force_ref: Option<bool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let case = MhaCase::new(t, dh, heads);
    let (d, t) = (case.d, case.t);
    println!("mha_self T={t} d={d} n_head={heads} dh={dh} threads={threads}");

    // The f64 oracle is O(4·T·d² + 2·n_head·T²·dh); skip it when it would dominate.
    let work = case.ref_work();
    let do_ref = force_ref.unwrap_or(work <= 2e9);
    let want = if do_ref {
        let t0 = Instant::now();
        let want = case.reference()?;
        println!(
            "mha_self_ref     : wall {:9.2} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
        Some(want)
    } else {
        println!("mha_self_ref     : skipped ({work:.1e} MACs; --ref 1 forces it)");
        None
    };

    let mut outs = [vec![f16::ZERO; t * d], vec![f16::ZERO; t * d]];
    let threads_of = [1usize, threads];
    // Warm both (packs the projections once, sizes the scratch).
    for (o, &n) in outs.iter_mut().zip(&threads_of) {
        case.run(o, n)?;
    }
    let mut walls = vec![Vec::new(); 2];
    for _ in 0..reps {
        for (i, (o, &n)) in outs.iter_mut().zip(&threads_of).enumerate() {
            let t0 = Instant::now();
            case.run(o, n)?;
            walls[i].push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    for (i, &n) in threads_of.iter().enumerate() {
        println!(
            "mha_self(threads={n:<2}): wall {:9.2} ms  (median of {reps})",
            median(walls[i].clone())
        );
        if let Some(want) = &want {
            let (cos, max) = compare(&outs[i], want);
            println!("                    cosine {cos:.8} max_abs {max:.6}");
        }
    }
    let (diff, max) = bit_diff(&outs[0], &outs[1]);
    println!(
        "mha_self bit-identical threads=1 vs {threads}: {}",
        verdict(diff, outs[0].len(), max)
    );
    if diff != 0 {
        return Err("mha_self with a projection fan-out is not bit-identical".into());
    }
    Ok(())
}

/// One arm of the placement A/B: what it produced and what it cost.
struct SmArm {
    out: Vec<f16>,
    walls: Vec<f64>,
    cpus: Vec<f64>,
}

/// Run one closure under both softmax placements, interleaved, and report each arm.
///
/// `ROCKET_ATTN_HOST_SOFTMAX` is set in the environment before every call (`1` = host,
/// `0` = on-NPU; the knobs are resolved per call), and the warm-up call per arm sizes each
/// arm's scratch. Interleaved reps keep a drifting board from biasing one arm.
fn placement_ab<F>(
    tag: &str,
    out_len: usize,
    want: Option<&[f16]>,
    reps: usize,
    run: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(&mut [f16]) -> rocket_npu::Result<()>,
{
    let placements = [("host", "1"), ("npu", "0")];
    let mut outs: [Vec<f16>; 2] = std::array::from_fn(|_| vec![f16::ZERO; out_len]);
    let mut walls: [Vec<f64>; 2] = Default::default();
    let mut cpus: [Vec<f64>; 2] = Default::default();
    // Warm both (opens the fds, sizes the scratch, allocates the on-NPU arm's matrix).
    for (i, (_, v)) in placements.iter().enumerate() {
        unsafe { std::env::set_var("ROCKET_ATTN_HOST_SOFTMAX", v) };
        run(&mut outs[i])?;
    }
    for _ in 0..reps {
        for (i, (_, v)) in placements.iter().enumerate() {
            unsafe { std::env::set_var("ROCKET_ATTN_HOST_SOFTMAX", v) };
            let cpu = cpu_s();
            let t0 = Instant::now();
            run(&mut outs[i])?;
            walls[i].push(t0.elapsed().as_secs_f64() * 1e3);
            cpus[i].push((cpu_s() - cpu) * 1e3);
        }
    }
    let labels = ["host", "npu"];
    let mut arms: Vec<SmArm> = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        let arm = SmArm {
            out: std::mem::take(&mut outs[i]),
            walls: std::mem::take(&mut walls[i]),
            cpus: std::mem::take(&mut cpus[i]),
        };
        println!(
            "{tag} softmax={label:<4}: wall {:9.2} ms  cpu {:9.2} ms  (median of {reps})",
            median(arm.walls.clone()),
            median(arm.cpus.clone())
        );
        if let Some(want) = want {
            let (cos, max) = compare(&arm.out, want);
            println!("                         cosine {cos:.8} max_abs {max:.6}  (vs f64 ref)");
        }
        arms.push(arm);
    }
    let (cos, max) = compare(&arms[1].out, &arms[0].out);
    let (diff, _) = bit_diff(&arms[1].out, &arms[0].out);
    println!("{tag} npu vs host: cosine {cos:.8} max_abs {max:.6} ({diff} differing bits)");
    let ratio = |a: &[f64], b: &[f64]| {
        let (a, b) = (median(a.to_vec()), median(b.to_vec()));
        if b > 0.0 { a / b } else { 0.0 }
    };
    println!(
        "{tag} npu/host: wall {:.2}x  cpu {:.2}x",
        ratio(&arms[1].walls, &arms[0].walls),
        ratio(&arms[1].cpus, &arms[0].cpus)
    );
    Ok(())
}

/// `--sm-ab`: the host pass against the on-NPU composition for both the flash primitive
/// and the encoder sublayer. The on-NPU `EXP` LUT is approximate, so this reports the
/// deviation instead of gating it: the C library's own gate accepts cosine >= 0.9995.
fn sm_ab_block(
    case: &Case,
    want: Option<&[f16]>,
    threads: usize,
    reps: usize,
    mha_dh: usize,
    mha_heads: usize,
    force_ref: Option<bool>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "softmax placement A/B: flash T={} kv={} dk={} dv={} heads={} (ref {})",
        case.t,
        case.kv,
        case.dk,
        case.dv,
        case.heads,
        if want.is_some() { "on" } else { "off" }
    );
    // Single fd: the stateless entry, one context per call (the C's shape).
    placement_ab("flash single-fd", case.out_len(), want, reps, |o| {
        case.run_oneshot(o)
    })?;
    placement_ab(
        &format!("flash mt({threads})"),
        case.out_len(),
        want,
        reps,
        |o| case.run_mt(o, threads),
    )?;

    // The encoder sublayer, whose C default is the on-NPU arm.
    let mha = MhaCase::new(case.t, mha_dh, mha_heads);
    let work = mha.ref_work();
    let do_ref = force_ref.unwrap_or(work <= 2e9);
    let want_mha = if do_ref {
        let t0 = Instant::now();
        let want = mha.reference()?;
        println!(
            "mha_self_ref     : wall {:9.2} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
        Some(want)
    } else {
        println!("mha_self_ref     : skipped ({work:.1e} MACs; --ref 1 forces it)");
        None
    };
    placement_ab("mha_self", mha.t * mha.d, want_mha.as_deref(), reps, |o| {
        mha.run(o, threads)
    })?;
    Ok(())
}

// ── Raw-file mode (`--io`), the C A/B layout ────────────────────────────────
//
// Shared with `tools/c-oracle/ab_board.c` (the same argument layout on both sides),
// driven by `tools/c-oracle/ab_compare.py --attn`. Floats travel as f32 bit patterns in
// hex so neither side parses a decimal differently; `-` is a NULL tensor.

fn read_f16(path: &str, n: usize) -> Result<Vec<f16>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != n * 2 {
        return Err(format!("{path}: {} bytes, want {}", bytes.len(), n * 2).into());
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
        .collect())
}

fn read_f16_opt(path: &str, n: usize) -> Result<Option<Vec<f16>>, Box<dyn std::error::Error>> {
    if path == "-" {
        Ok(None)
    } else {
        Ok(Some(read_f16(path, n)?))
    }
}

fn write_f16(path: &str, v: &[f16]) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(v.len() * 2);
    for x in v {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// An f32 argument as a hex bit pattern (`0x3db504f3`), so both sides read the same bits.
fn arg_f32(s: &str) -> Result<f32, Box<dyn std::error::Error>> {
    let t = s.trim();
    let t = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .unwrap_or(t);
    Ok(f32::from_bits(u32::from_str_radix(t, 16)?))
}

/// `--io flash ...` / `--io mha ...`: one op on raw files, the C A/B layout.
fn io_mode(a: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let num = |i: usize| -> Result<usize, Box<dyn std::error::Error>> {
        a.get(i)
            .ok_or_else(|| "missing argument".into())
            .and_then(|s| Ok(s.parse::<usize>()?))
    };
    let file = |i: usize| -> Result<&str, Box<dyn std::error::Error>> {
        a.get(i)
            .map(String::as_str)
            .ok_or_else(|| "missing file".into())
    };
    match a.first().map(String::as_str) {
        Some("flash") => {
            // flash T KV DK DV HEADS HKV SCALE_BITS SOFTCAP_BITS Q K V MASK OUT
            let (t, kv, dk, dv, heads, hkv) =
                (num(1)?, num(2)?, num(3)?, num(4)?, num(5)?, num(6)?);
            let (scale, softcap) = (arg_f32(file(7)?)?, arg_f32(file(8)?)?);
            let q = read_f16(file(9)?, heads * t * dk)?;
            let k = read_f16(file(10)?, hkv * kv * dk)?;
            let v = read_f16(file(11)?, hkv * dv * kv)?;
            let mask = read_f16_opt(file(12)?, t * kv)?;
            let mut out = vec![f16::ZERO; heads * t * dv];
            flash_attn(
                t,
                kv,
                dk,
                dv,
                heads,
                hkv,
                scale,
                softcap,
                &q,
                &k,
                &v,
                mask.as_deref(),
                &mut out,
            )?;
            write_f16(file(13)?, &out)?;
        }
        Some("mha") => {
            // mha T D HEADS X WQ BQ WK BK WV BV WO BO OUT
            let (t, d, heads) = (num(1)?, num(2)?, num(3)?);
            let x = read_f16(file(4)?, t * d)?;
            let wq = read_f16(file(5)?, d * d)?;
            let bq = read_f16_opt(file(6)?, d)?;
            let wk = read_f16(file(7)?, d * d)?;
            let bk = read_f16_opt(file(8)?, d)?;
            let wv = read_f16(file(9)?, d * d)?;
            let bv = read_f16_opt(file(10)?, d)?;
            let wo = read_f16(file(11)?, d * d)?;
            let bo = read_f16_opt(file(12)?, d)?;
            let mut out = vec![f16::ZERO; t * d];
            mha_self(
                t,
                d,
                heads,
                &x,
                &wq,
                bq.as_deref(),
                &wk,
                bk.as_deref(),
                &wv,
                bv.as_deref(),
                &wo,
                bo.as_deref(),
                &mut out,
                1,
            )?;
            write_f16(file(13)?, &out)?;
        }
        other => {
            return Err(format!("--io needs flash|mha (got {})", other.unwrap_or("<none>")).into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("--io") {
        return io_mode(&argv[1..]);
    }
    // Positional `tokens kv heads kv_heads`, then the flags.
    let mut pos = Vec::new();
    let mut chain: Option<bool> = None;
    let mut dk = 128usize;
    let mut dv = 0usize;
    let mut reps = 3usize;
    let mut want_ref: Option<bool> = None;
    let mut mem = false;
    let mut oneshot = false;
    let mut mha = false;
    let mut threads = 4usize;
    let mut softcap = 0.0f32;
    let mut mask_kind = "causal".to_string();
    let mut tile = 0usize;
    let mut tile_min = 8192i64;
    let mut tile_only = false;
    let mut host_sm: Option<bool> = None;
    let mut sm_ab = false;
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
            "--oneshot" => oneshot = true,
            "--mha" => mha = true,
            "--mt" => threads = value()?.parse::<usize>()?.max(1),
            "--softcap" => softcap = value()?.parse()?,
            "--mask" => mask_kind = value()?,
            "--tile" => tile = value()?.parse()?,
            "--tile-min" => tile_min = value()?.parse()?,
            "--tile-only" => tile_only = true,
            "--host-sm" => host_sm = Some(value()?.trim() != "0"),
            "--sm-ab" => sm_ab = true,
            "--help" | "-h" => {
                println!(
                    "usage: attn_selftest [tokens] [kv] [heads] [kv_heads] \
                     [--dk N] [--dv N] [--chain 0|1] [--reps N] [--ref 0|1] [--mem] \
                     [--oneshot] [--mt N] [--mha] [--softcap F] [--mask none|causal|window] \
                     [--tile KBP] [--tile-min KV] [--tile-only] [--host-sm 0|1] [--sm-ab]"
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
    println!(
        "attn T={t} kv={kv} heads={heads} kv_heads={hkv} dk={dk} dv={dv} reps={reps} \
         mask={mask_kind} softcap={softcap}"
    );

    let mut seed = 0xdead_beef_1234_5678u64;
    let q: Vec<f16> = (0..heads * t * dk).map(|_| rnd(&mut seed)).collect();
    let k: Vec<f16> = (0..hkv * kv * dk).map(|_| rnd(&mut seed)).collect();
    let v: Vec<f16> = (0..hkv * dv * kv).map(|_| rnd(&mut seed)).collect();
    let mask: Option<Vec<f16>> = match mask_kind.as_str() {
        "none" => None,
        // Causal: key j is visible to row i iff j <= i.
        "causal" => {
            let mut m = vec![f16::ZERO; t * kv];
            for row in 0..t {
                for j in 0..kv {
                    if j > row {
                        m[row * kv + j] = f16::NEG_INFINITY;
                    }
                }
            }
            Some(m)
        }
        // A bidirectional band of t/8 keys (the sliding-layer shape).
        "window" => {
            let w = (t / 8).max(1) as i64;
            let mut m = vec![f16::ZERO; t * kv];
            for row in 0..t {
                for j in 0..kv {
                    if (j as i64 - row as i64).abs() > w {
                        m[row * kv + j] = f16::NEG_INFINITY;
                    }
                }
            }
            Some(m)
        }
        other => return Err(format!("--mask {other}: expected none|causal|window").into()),
    };
    let case = Case {
        t,
        kv,
        dk,
        dv,
        heads,
        hkv,
        scale,
        softcap,
        q: &q,
        k: &k,
        v: &v,
        mask: mask.as_deref(),
    };

    // The knobs are resolved per call, so setting them here covers every mode below —
    // the stateless `--oneshot` / `--mha` entries ride the same dispatcher, and
    // `tile_block` re-states them for the A/B.
    if tile > 0 {
        unsafe {
            std::env::set_var("ROCKET_FA_TILE_KV", tile.to_string());
            std::env::set_var("ROCKET_FA_TILE_MIN_KV", tile_min.to_string());
        }
    }
    // `--host-sm` selects the materialized paths' softmax placement process-wide (the
    // knob is resolved per call); `--sm-ab` sets it per arm itself.
    if let Some(host) = host_sm {
        unsafe { std::env::set_var("ROCKET_ATTN_HOST_SOFTMAX", if host { "1" } else { "0" }) };
    }
    println!(
        "softmax placement: {} (ROCKET_ATTN_HOST_SOFTMAX={})",
        if host_softmax() { "host" } else { "on-NPU" },
        if host_softmax() { "1" } else { "0" }
    );

    // The f64 oracle is O(heads * t * kv * dk); skip it when it would dominate the run
    // (the chained A/B does not need it).
    let ref_work = (heads * t * kv * dk) as f64;
    let do_ref = want_ref.unwrap_or(ref_work <= 2e9);
    let want = if do_ref {
        Some(case.reference()?)
    } else {
        println!("reference: skipped ({ref_work:.1e} MACs; --ref 1 forces it)");
        None
    };

    if oneshot {
        oneshot_block(&case, want.as_deref(), threads, reps)?;
    }
    if mha {
        mha_block(t, dk, heads, threads, reps, want_ref)?;
    }
    if sm_ab {
        sm_ab_block(&case, want.as_deref(), threads, reps, dk, heads, want_ref)?;
    }
    if oneshot || mha || sm_ab {
        return Ok(());
    }

    let attn_threads = std::env::var("ROCKET_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let mut attn = Attention::new(attn_threads)?;
    if tile > 0 {
        let opts = TileOpts {
            kbp: tile,
            min_kv: tile_min,
            materialized: !tile_only,
            chain: chain.unwrap_or(true),
        };
        tile_block(&case, want.as_deref(), &mut attn, reps, &opts)?;
        if mem {
            mem_block(&case, &mut attn)?;
        }
        return Ok(());
    }
    let modes: Vec<bool> = match chain {
        Some(c) => vec![c],
        None => vec![false, true],
    };

    // The knob is resolved per call, so one context A/Bs both paths. Warm both first
    // (opens the worker fds, sizes the scratch, packs the batch contexts), then
    // interleave the timed reps so a drifting board hits both modes alike.
    let mut outs: Vec<Vec<f16>> = modes
        .iter()
        .map(|_| vec![f16::ZERO; case.out_len()])
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
        let (diff, max) = bit_diff(per_head, chained);
        println!("bit-identical: {}", verdict(diff, per_head.len(), max));
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
        mem_block(&case, &mut attn)?;
    }
    Ok(())
}

/// The `--tile` A/B: materialized (tiling off) against tiled on the same case.
struct TileOpts {
    kbp: usize,
    min_kv: i64,
    materialized: bool,
    chain: bool,
}

/// One arm of the tiled A/B: its output and the timed walls/CPU seconds.
struct Arm {
    label: &'static str,
    out: Vec<f16>,
    walls: Vec<f64>,
    cpus: Vec<f64>,
}

/// Run one case materialized and tiled, compare both with the f64 reference and each
/// other, and report the wall/CPU and the scratch each arm leaves behind.
///
/// The arms run one after the other, not interleaved: a tiled `prepare` releases the
/// full-key-axis scratch of a materialized call at the same length (that is the memory
/// the path exists to save), so interleaving them would re-allocate ~`Tp*Kn` per rep
/// and measure the allocator instead of the kernels.
fn tile_block(
    case: &Case,
    want: Option<&[f16]>,
    attn: &mut Attention,
    reps: usize,
    opts: &TileOpts,
) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        std::env::set_var("ROCKET_FA_TILE_KV", opts.kbp.to_string());
        std::env::set_var("ROCKET_FA_TILE_MIN_KV", opts.min_kv.to_string());
    }
    let kn = case.kv.div_ceil(32) * 32;
    // The worker resolves tiling from the padded key count alone (any `dk`/`dv`), so
    // report what the path will actually do.
    let kbp = tile_kv_width(kn);
    println!(
        "tile: ROCKET_FA_TILE_KV={} ROCKET_FA_TILE_MIN_KV={} -> tile_kv_width(Kn={kn}) = {kbp}{}",
        opts.kbp,
        opts.min_kv,
        if case.dv != case.dk {
            format!(
                " (MLA: dk {} dv {} — the tiled path carries the two separately)",
                case.dk, case.dv
            )
        } else {
            String::new()
        }
    );
    if kbp == 0 {
        println!(
            "note: the tiled path does not engage at Kn={kn} (needs Kn > min_kv and \
             kbp < Kn); both arms run materialized"
        );
    }
    println!(
        "tile: scaled score elements held at once: materialized {} MiB/head, tiled {} MiB/head",
        (case.t.div_ceil(4) * kn * 2) as f64 / (1 << 20) as f64,
        if kbp > 0 {
            (case.t.div_ceil(4) * kbp * 2) as f64 / (1 << 20) as f64
        } else {
            (case.t.div_ceil(4) * kn * 2) as f64 / (1 << 20) as f64
        }
    );

    let mut arms: Vec<Arm> = Vec::new();
    if opts.materialized {
        unsafe { std::env::set_var("ROCKET_FA_TILE_KV", "0") };
        unsafe {
            std::env::set_var("ROCKET_FA_CHAIN", if opts.chain { "1" } else { "0" });
        }
        let mut out = vec![f16::ZERO; case.out_len()];
        case.run(attn, &mut out)?;
        report_mem(attn, "materialized, after a warm run");
        let (mut walls, mut cpus) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            let cpu = cpu_s();
            let t0 = Instant::now();
            case.run(attn, &mut out)?;
            walls.push(t0.elapsed().as_secs_f64() * 1e3);
            cpus.push((cpu_s() - cpu) * 1e3);
        }
        arms.push(Arm {
            label: "materialized",
            out,
            walls,
            cpus,
        });
        unsafe { std::env::set_var("ROCKET_FA_TILE_KV", opts.kbp.to_string()) };
    }

    // The tiled arm: warmed first (sizes the tile scratch), then timed with the chained
    // knob left wherever the materialized arm put it — the tiled path ignores it, which
    // the bit-identity check below proves.
    let mut out = vec![f16::ZERO; case.out_len()];
    case.run(attn, &mut out)?;
    report_mem(attn, "tiled, after a warm run");
    let (mut walls, mut cpus) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        let cpu = cpu_s();
        let t0 = Instant::now();
        case.run(attn, &mut out)?;
        walls.push(t0.elapsed().as_secs_f64() * 1e3);
        cpus.push((cpu_s() - cpu) * 1e3);
    }
    let mut out_unchained = vec![f16::ZERO; case.out_len()];
    unsafe { std::env::set_var("ROCKET_FA_CHAIN", "0") };
    case.run(attn, &mut out_unchained)?;
    unsafe { std::env::set_var("ROCKET_FA_CHAIN", "1") };
    arms.push(Arm {
        label: "tiled",
        out,
        walls,
        cpus,
    });

    for arm in &arms {
        let (label, out) = (arm.label, &arm.out);
        println!(
            "{label:<13}: wall {:9.2} ms  cpu {:9.2} ms  (median of {reps})",
            median(arm.walls.clone()),
            median(arm.cpus.clone())
        );
        if let Some(want) = want {
            let (cos, max) = compare(out, want);
            println!("               cosine {cos:.8} max_abs {max:.6}  (vs flash_attn_ref)");
        }
    }
    if let Some(mat) = arms.iter().find(|a| a.label == "materialized") {
        let tiled = arms
            .iter()
            .find(|a| a.label == "tiled")
            .expect("the tiled arm always runs");
        let (cos, max) = compare(&tiled.out, &mat.out);
        println!(
            "tiled vs materialized: cosine {cos:.8} max_abs {max:.6} \
             (online reassociation, not bit-identical by design)"
        );
        let ratio = |a: &[f64], b: &[f64]| {
            let (a, b) = (median(a.to_vec()), median(b.to_vec()));
            if b > 0.0 { a / b } else { 0.0 }
        };
        println!(
            "tiled/materialized: wall {:.2}x  cpu {:.2}x",
            ratio(&tiled.walls, &mat.walls),
            ratio(&tiled.cpus, &mat.cpus)
        );
    }
    let tiled = arms
        .iter()
        .find(|a| a.label == "tiled")
        .expect("the tiled arm always runs");
    let (diff, max) = bit_diff(&tiled.out, &out_unchained);
    println!(
        "tiled bit-identical ROCKET_FA_CHAIN=0 vs 1: {}",
        verdict(diff, tiled.out.len(), max)
    );
    if diff != 0 {
        return Err("the tiled path must not depend on ROCKET_FA_CHAIN".into());
    }
    // Mark this request as the resident generation, so `--mem`'s 3-token request is
    // the one that can release it (the trim keeps the just-finished request's class).
    attn.trim_scratch();
    report_mem(attn, "tiled, after the timed runs and a trim");
    Ok(())
}

/// The `--mem` report: live scratch after the load, after a 3-token request and after
/// `trim_scratch`, so the 4x / request-generation rules can be seen to fire.
fn mem_block(case: &Case, attn: &mut Attention) -> Result<(), Box<dyn std::error::Error>> {
    report_mem(attn, "after the timed runs");
    // A tiny request drops the working shape, which is what `trim_scratch` keys on.
    let (t3, kv3) = (3usize, 3usize);
    let mut seed = 0xfeed_face_dead_beefu64;
    let q3: Vec<f16> = (0..case.heads * t3 * case.dk)
        .map(|_| rnd(&mut seed))
        .collect();
    let k3: Vec<f16> = (0..case.hkv * kv3 * case.dk)
        .map(|_| rnd(&mut seed))
        .collect();
    let v3: Vec<f16> = (0..case.hkv * case.dv * kv3)
        .map(|_| rnd(&mut seed))
        .collect();
    let mask3 = vec![f16::ZERO; t3 * kv3];
    let tiny = Case {
        t: t3,
        kv: kv3,
        dk: case.dk,
        dv: case.dv,
        heads: case.heads,
        hkv: case.hkv,
        scale: case.scale,
        softcap: case.softcap,
        q: &q3,
        k: &k3,
        v: &v3,
        mask: Some(&mask3),
    };
    let mut out3 = vec![f16::ZERO; tiny.out_len()];
    tiny.run(attn, &mut out3)?;
    report_mem(attn, "after a 3-token request, before trim");
    attn.trim_scratch();
    report_mem(attn, "after trim_scratch");
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
            "  worker {i}: Tp {} Kn {} tile {} bo {:.1} MiB host {:.1} MiB",
            w.shape.0,
            w.shape.1,
            w.tile,
            w.bo_bytes as f64 / (1 << 20) as f64,
            w.host_bytes as f64 / (1 << 20) as f64
        );
    }
}
