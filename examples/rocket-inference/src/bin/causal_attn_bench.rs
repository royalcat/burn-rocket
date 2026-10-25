//! Causal block attention probe (P2a): the chunked NPU causal path vs the dense
//! masked op and the f64 host oracle.
//!
//! The chunked path is what makes long causal prompts fit on the board: per
//! chunk, `burn_rocket::attention_causal_block` runs a query chunk against only
//! its key prefix, so the additive mask and the score matrix stay `[C, q1]`
//! instead of the model's `[n, n]`. This binary pins the numerics:
//!
//! * `dense` — `burn_rocket::attention(.., is_causal = true)` or
//!   `attention_causal_window(.., window)` over the whole sequence;
//! * `chunk-N` — the app helper (`examples/rocket-inference/src/util/causal.rs`,
//!   compiled from the same source file) with a chunk width of `N`;
//! * `ref` — `rocket_npu::attn::flash_attn_ref`, the device-free f64 oracle, run
//!   over the whole sequence with a locally built causal mask (`--ref`).
//!
//! Every arm is scored against `dense` (cosine / max|d| / differing bits) and,
//! with `--ref`, against the f64 oracle. Each arm also runs twice so
//! within-process repeatability is visible separately from the cross-process
//! scatter the shared board shows.
//!
//! Usage (aarch64 NPU build):
//!   causal_attn_bench [n] [chunk[,chunk...]] [reps]
//!   causal_attn_bench 512 256 2 --ref          # against the f64 oracle
//!   causal_attn_bench 1024 256 --window 512    # sliding layers (Gemma 4)
//!   causal_attn_bench 30000 1024 --chunked-only
//!   causal_attn_bench 1024 1024 --kv 30000     # one 30k-context chunk
//!
//! Defaults: `n = 1024`, chunks `256,1024`, `reps = 1`, pure causal, qwen3's
//! geometry (16 heads / 8 KV heads / 128). `--chunked-only` skips the dense arm
//! (its `[n, n]` mask is what a 30k run cannot afford) and the oracle, so the
//! chunked path's footprint can be measured at lengths where the dense one dies.

#[path = "../util/causal.rs"]
#[allow(dead_code)]
mod causal;

use std::time::Instant;

use burn::prelude::*;
use burn_rocket::half::f16;

/// qwen3-embedding-0.6b's attention geometry.
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

fn main() {
    let mut n = 1024usize;
    let mut chunks = vec![256usize, 1024];
    let mut reps = 1usize;
    let mut window: Option<usize> = None;
    let mut with_ref = false;
    let mut chunked_only = false;
    let mut kv_len: Option<usize> = None;
    let mut positional = 0;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--ref" => with_ref = true,
            "--chunked-only" => chunked_only = true,
            "--kv" => {
                i += 1;
                kv_len = Some(args[i].parse().expect("--kv N"));
            }
            "--window" => {
                i += 1;
                window = Some(args[i].parse().expect("--window N"));
            }
            other => match positional {
                0 => {
                    n = other.parse().expect("n");
                    positional += 1;
                }
                1 => {
                    chunks = other
                        .split(',')
                        .map(|v| v.parse().expect("chunk"))
                        .collect();
                    positional += 1;
                }
                2 => {
                    reps = other.parse().expect("reps");
                    positional += 1;
                }
                _ => panic!("unexpected argument {other}"),
            },
        }
        i += 1;
    }

    let (h, kv, d) = (HEADS, KV_HEADS, HEAD_DIM);
    // `--kv N`: one query chunk of `n` rows against a longer key axis (the shape
    // a real 30k-context chunk runs, without holding the whole query sequence).
    let kn = kv_len.unwrap_or(n);
    if kn != n {
        chunked_only = true;
    }
    let scale = 1.0 / (d as f64).sqrt();
    let device = Device::flex();
    // The app's NPU path: 5 worker threads (the engine serializes them; the
    // attention op fans heads out over this many device contexts).
    burn_rocket::init(5).expect("rocket init");
    println!(
        "causal block attention: n {n}, keys {kn}, heads {h} (kv heads {kv}), head_dim {d}, \
         window {window:?}, chunks {chunks:?}, reps {reps}"
    );
    println!("start RssAnon {} MiB", rss_anon_mib());

    // qwen3's q/k arrive post-RoPE with unit-ish rows; [-0.5, 0.5) is the right
    // order and keeps the softmax away from saturation.
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let q: Vec<f32> = (0..n * h * d).map(|_| rnd()).collect();
    let k: Vec<f32> = (0..kn * kv * d).map(|_| rnd()).collect();
    let v: Vec<f32> = (0..kn * kv * d).map(|_| rnd()).collect();
    let t = |data: Vec<f32>, rows: usize, w: usize| {
        Tensor::<3>::from_data(TensorData::new(data, [1, rows, w]), &device)
    };
    let (qf, kf, vf) = (
        t(q.clone(), n, h * d),
        t(k.clone(), kn, kv * d),
        t(v.clone(), kn, kv * d),
    );

    let run_dense = || {
        if let Some(w) = window {
            burn_rocket::attention_causal_window(
                qf.clone(),
                kf.clone(),
                vf.clone(),
                h,
                kv,
                d,
                scale,
                None,
                w as i64,
            )
        } else {
            burn_rocket::attention(
                qf.clone(),
                kf.clone(),
                vf.clone(),
                h,
                kv,
                d,
                scale,
                None,
                true,
            )
        }
    };
    let q_pos = kn.saturating_sub(n);
    let run_chunked = |chunk: usize| {
        causal::causal_attention(
            qf.clone(),
            kf.clone(),
            vf.clone(),
            h,
            kv,
            d,
            scale,
            None,
            window,
            q_pos,
            chunk,
        )
    };

    let time = |f: &dyn Fn() -> Tensor<3>| -> (Vec<f32>, f64) {
        let mut total = 0.0;
        let mut out = Vec::new();
        for rep in 0..=reps {
            let t0 = Instant::now();
            let o = f();
            device.sync().expect("flex sync");
            let dt = t0.elapsed().as_secs_f64();
            if rep > 0 {
                total += dt;
            }
            if rep == reps {
                out = o.into_data().try_to_vec().expect("f32");
            }
        }
        (out, total / reps as f64)
    };

    let mut dense: Vec<f32> = Vec::new();
    if chunked_only {
        println!("dense        skipped (--chunked-only; its [n, n] mask is the 30k blocker)");
    } else {
        let (d, dense_s) = time(&run_dense);
        let (d2, _) = time(&run_dense);
        let (rep_cos, rep_max, rep_bits) = compare_meta(&d, &d2);
        println!(
            "dense        {dense_s:8.3} s   repeat cos {rep_cos:.9} max|d| {rep_max:.2e} \
             diff_bits {rep_bits}   attention {} MiB",
            attn_mib()
        );
        dense = d;
    }

    for &chunk in &chunks {
        let (out, s) = time(&|| run_chunked(chunk));
        let (out2, _) = time(&|| run_chunked(chunk));
        let (rc, rm, rb) = compare_meta(&out, &out2);
        if chunked_only {
            println!(
                "chunk-{chunk:<6} {s:8.3} s   repeat cos {rc:.9} max|d| {rm:.2e} \
                 diff_bits {rb}   attention {} MiB",
                attn_mib()
            );
        } else {
            let (cos, maxd, bits) = compare_meta(&dense, &out);
            println!(
                "chunk-{chunk:<6} {s:8.3} s   vs dense cos {cos:.9} max|d| {maxd:.2e} \
                 diff_bits {bits}/{}   repeat cos {rc:.9} max|d| {rm:.2e} diff_bits {rb}   \
                 attention {} MiB",
                out.len(),
                attn_mib()
            );
        }
    }

    if with_ref && !chunked_only {
        // The oracle is O(n^2 * d) in f64 on one thread; keep it small.
        let t0 = Instant::now();
        let (mask, keep) = full_causal_mask(n, window);
        let ref_out = oracle(&q, &k, &v, n, h, kv, d, scale, &mask);
        println!(
            "ref          {:8.3} s   f64 flash_attn_ref, {} of {} mask entries visible",
            t0.elapsed().as_secs_f64(),
            keep,
            n * n
        );
        let (cos, maxd, _) = compare_meta(&dense, &ref_out);
        println!("dense        vs ref cos {cos:.9} max|d| {maxd:.2e}");
        for &chunk in &chunks {
            let (out, _) = time(&|| run_chunked(chunk));
            let (cos, maxd, _) = compare_meta(&ref_out, &out);
            println!("chunk-{chunk:<6} vs ref cos {cos:.9} max|d| {maxd:.2e}");
        }
    }

    let st = burn_rocket::stats();
    println!(
        "npu stats: {:.1} ms convert, {:.1} ms device, {} calls; RssAnon {} MiB",
        st.convert_s * 1e3,
        st.npu_s * 1e3,
        st.calls,
        rss_anon_mib()
    );
}

/// `(cosine, max|d|, differing value count)` — the last one over f32 bits.
fn compare_meta(a: &[f32], b: &[f32]) -> (f64, f32, usize) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxd) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    let mut bits = 0;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
        maxd = maxd.max((x - y).abs());
        bits += usize::from(x.to_bits() != y.to_bits());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxd, bits)
}

/// The whole-sequence causal (optionally windowed) mask, and its visible count.
fn full_causal_mask(n: usize, window: Option<usize>) -> (Vec<f16>, usize) {
    let mut mask = vec![f16::NEG_INFINITY; n * n];
    let mut keep = 0;
    for t in 0..n {
        let lo = match window {
            Some(w) => (t + 1).saturating_sub(w),
            None => 0,
        };
        for j in lo..=t {
            mask[t * n + j] = f16::ZERO;
            keep += 1;
        }
    }
    (mask, keep)
}

/// `flash_attn_ref` in the driver's host layouts (`[h, n, d]` q/out, `[kv, n, d]`
/// k, `[kv, d, n]` v) after narrowing the f32 operands to f16, i.e. the same
/// inputs the device arm receives.
#[allow(clippy::too_many_arguments)]
fn oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    h: usize,
    kv: usize,
    d: usize,
    scale: f64,
    mask: &[f16],
) -> Vec<f32> {
    let mut qh = vec![f16::ZERO; h * n * d];
    let mut kh = vec![f16::ZERO; kv * n * d];
    let mut vh = vec![f16::ZERO; kv * d * n];
    for hi in 0..h {
        for s in 0..n {
            for di in 0..d {
                qh[hi * n * d + s * d + di] = f16::from_f32(q[s * h * d + hi * d + di]);
            }
        }
    }
    for hi in 0..kv {
        for s in 0..n {
            for di in 0..d {
                kh[hi * n * d + s * d + di] = f16::from_f32(k[s * kv * d + hi * d + di]);
                vh[hi * d * n + di * n + s] = f16::from_f32(v[s * kv * d + hi * d + di]);
            }
        }
    }
    let mut oh = vec![f16::ZERO; h * n * d];
    burn_rocket::npu::attn::flash_attn_ref(
        n,
        n,
        d,
        d,
        h,
        kv,
        scale as f32,
        0.0,
        &qh,
        &kh,
        &vh,
        Some(mask),
        &mut oh,
    )
    .expect("flash_attn_ref");
    let mut out = vec![0f32; n * h * d];
    for hi in 0..h {
        for s in 0..n {
            for di in 0..d {
                out[s * h * d + hi * d + di] = oh[hi * n * d + s * d + di].to_f32();
            }
        }
    }
    out
}

/// The engine's live attention BO+host bytes (MiB), for the footprint claim.
fn attn_mib() -> u64 {
    burn_rocket::mem_stats()
        .map(|m| ((m.attention_bo_bytes + m.attention_host_bytes) / (1024 * 1024)) as u64)
        .unwrap_or(0)
}

/// Current `RssAnon` from `/proc/self/status` (MiB).
fn rss_anon_mib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
            return kb / 1024;
        }
    }
    0
}
