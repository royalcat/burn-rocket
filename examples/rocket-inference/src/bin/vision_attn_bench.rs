//! Microbenchmark: the vision tower's attention geometry on flex vs the NPU.
//!
//! One layer of EmbeddingGemma 2's vision tower is bidirectional attention over
//! `p` patches with 12 heads of width 64 and scale 1.0 (QK-RMSNorm replaces the
//! usual `1/sqrt(head_dim)`). This binary times that single op in the shapes the
//! tower uses:
//!
//! * `fused-f32` — `burn::tensor::module::attention` on `[1, 12, p, 64]` f32 (the
//!   deployed `--vision-attn fused` path);
//! * `fused-f16` / `fused-bf16` — the same call with 16-bit operands, i.e. the
//!   question of whether feeding the kernel the model dtype is faster (flex
//!   dispatches by dtype but casts f16/bf16 to f32 for the computation);
//! * `npu` — `burn_rocket::attention_window` (`window = -1`, no mask) on
//!   `[1, p, 768]`, including its host conversion/layout work;
//! * `npu-chunk-N` — the same through `attention_window_block` over query chunks
//!   of `N` (keys start at 0, `window = -1`), i.e. the bounded-scratch variant.
//!
//! Numerics: cosine and max|d| of each arm against the fused-f32 output.
//!
//! Two further modes isolate the primitives that frame batching (log-gemma §18.6)
//! changes, on the tower's real shapes:
//!
//! * `matmul [reps]` — the layer's packed projections (q|k|v 2304x768,
//!   o 768x768, gate|up 6144x768, down 768x3072) at `M` = 1170, 2340, 4680,
//!   9360 rows: ms per call and ns per row. A flat ns/row means the NPU time is
//!   proportional to the row count (no per-call cost to amortize by batching);
//! * `attn-batch [patches] [reps]` — fused attention at batch 1, 2, 4, 8 with
//!   the same per-frame work: whether the kernel's `batch * heads` parallelism
//!   helps at a given frame count (12 heads already fill 4 threads).
//!
//! `softmax [kn] [reps] [tp]` times the host softmax alone (S1 spike: the host
//! score pass was the §18.4 blocker, log-gemma §19): a synthetic f16 `[tp, kn]`
//! score matrix (padded like the driver: `tp` to 4, `kn` to 32) through
//! `burn_rocket::npu::attn::Softmax`, exact vs fast, with and without a causal
//! mask, plus the fast path's deviation from exact.
//!
//! Build (aarch64, NPU; the bin is gated on the `npu` feature):
//!   cargo build --release -p rocket-inference --target aarch64-unknown-linux-gnu \
//!       --no-default-features --features npu --bin vision_attn_bench
//! then scp `$CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/vision_attn_bench`
//! to the board and run it from any directory with `taskset -c 4-7`:
//!   ./vision_attn_bench [patches] [reps] [chunk,chunk,...]
//!   ./vision_attn_bench matmul [reps]
//!   ./vision_attn_bench attn-batch [patches] [reps]
//!   ./vision_attn_bench softmax [kn] [reps] [tp]

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn_rocket::half::f16;
use burn_rocket::npu::attn::{Softmax, SoftmaxMode};

const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
/// Vision hidden size / MLP width of both Gemma 4 towers (EmbeddingGemma 2 and
/// E2B-it share the vision config).
const HIDDEN: usize = 768;
const INTER: usize = 3072;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("matmul") => {
            let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
            matmul_scale(reps);
        }
        Some("attn-batch") => {
            let p: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1170);
            let reps: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);
            attention_batch(p, reps);
        }
        Some("softmax") => {
            let kn: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2394);
            let reps: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
            let tp: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(kn);
            softmax_bench(kn, reps, tp);
        }
        _ => attention_arms(&args),
    }
}

fn attention_arms(args: &[String]) {
    let p: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(2394);
    let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let chunks: Vec<usize> = args
        .get(3)
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1024, 512]);
    let (h, d) = (HEADS, HEAD_DIM);
    let device = Device::flex();

    println!(
        "vision attention: patches {p}, heads {h}, head_dim {d}, scale 1.0, bidirectional, reps {reps}"
    );
    println!("start RssAnon {} MiB", rss_anon_mib());

    // Deterministic pseudo-random q/k/v in [-0.5, 0.5) (the tower's q/k arrive
    // RMS-normalized, so these are the right order of magnitude). The kernel and
    // host-softmax cost do not depend on the values.
    let n = p * h * d;
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let q: Vec<f32> = (0..n).map(|_| rnd()).collect();
    let k: Vec<f32> = (0..n).map(|_| rnd()).collect();
    let v: Vec<f32> = (0..n).map(|_| rnd()).collect();
    let flat =
        |data: Vec<f32>| Tensor::<3>::from_data(TensorData::new(data, [1, p, h * d]), &device);
    let (qf, kf, vf) = (flat(q), flat(k), flat(v));

    // `[1, p, h*d]` -> `[1, h, p, d]` (what the fused kernel wants).
    let to_kernel = |t: Tensor<3>| t.reshape([p, h, d]).swap_dims(0, 1).unsqueeze_dim::<4>(0);

    // ---- fused CPU kernel, operands in each float dtype --------------------
    // The tower feeds the kernel in its activation dtype: f32 in the parity
    // arms, and (per this round's question) f16/bf16 if that kernel path were
    // faster. flex's `attention` dispatches by dtype and computes f32 internally
    // for 16-bit operands, so the 16-bit arms measure conversion + the same f32
    // kernel.
    let mut fused_f32: Vec<f32> = Vec::new();
    let fused_dtypes = [
        ("fused-f32", DType::F32),
        ("fused-f16", DType::F16),
        ("fused-bf16", DType::BF16),
    ];
    for (name, dt) in fused_dtypes {
        let cast = |t: Tensor<3>| {
            if dt == DType::F32 { t } else { t.cast(dt) }
        };
        let (q, k, v) = (cast(qf.clone()), cast(kf.clone()), cast(vf.clone()));
        let (q4, k4, v4) = (to_kernel(q), to_kernel(k), to_kernel(v));
        let mut total_s = 0.0f64;
        let mut out: Vec<f32> = Vec::new();
        for rep in 0..=reps {
            let t0 = Instant::now();
            let o = attention(
                q4.clone(),
                k4.clone(),
                v4.clone(),
                None,
                None,
                AttentionModuleOptions {
                    scale: Some(1.0),
                    softcap: None,
                    is_causal: false,
                },
            );
            device.sync().expect("flex sync");
            let dt_s = t0.elapsed().as_secs_f64();
            if rep > 0 {
                total_s += dt_s;
            }
            if rep == reps {
                // Back to token-major `[1, p, h*d]` f32 for the comparison.
                out = o
                    .reshape([h, p, d])
                    .swap_dims(0, 1)
                    .reshape([1, p, h * d])
                    .cast(DType::F32)
                    .into_data()
                    .try_to_vec()
                    .expect("f32");
            }
        }
        let per_rep = total_s / reps as f64;
        if name == "fused-f32" {
            fused_f32 = out;
            println!(
                "{name:<12} {:8.3} s/rep   RssAnon {} MiB  (reference)",
                per_rep,
                rss_anon_mib()
            );
        } else {
            let (cos, maxd) = compare(&fused_f32, &out);
            println!(
                "{name:<12} {:8.3} s/rep   RssAnon {} MiB  vs fused-f32 cos {cos:.9} max|d| {maxd:.2e}",
                per_rep,
                rss_anon_mib()
            );
        }
    }
    let fused_out = fused_f32;

    // ---- NPU attention ----------------------------------------------------
    burn_rocket::init(4).expect("rocket init");
    let mut npu_out: Vec<f32> = Vec::new();
    for chunk in std::iter::once(0usize).chain(chunks.iter().copied()) {
        let name = if chunk == 0 {
            "npu".to_string()
        } else {
            format!("npu-chunk-{chunk}")
        };
        let mut total_s = 0.0f64;
        let (mut conv, mut npu_s, mut calls) = (0.0, 0.0, 0);
        for rep in 0..=reps {
            burn_rocket::stats_reset();
            let t0 = Instant::now();
            let o = if chunk == 0 {
                burn_rocket::attention_window(
                    qf.clone(),
                    kf.clone(),
                    vf.clone(),
                    h,
                    h,
                    d,
                    1.0,
                    None,
                    -1,
                )
            } else {
                let mut outs: Vec<Tensor<3>> = Vec::new();
                let mut q0 = 0;
                while q0 < p {
                    let q1 = (q0 + chunk).min(p);
                    outs.push(burn_rocket::attention_window_block(
                        qf.clone().slice(s![.., q0..q1, ..]),
                        kf.clone(),
                        vf.clone(),
                        h,
                        h,
                        d,
                        1.0,
                        None,
                        -1,
                        q0 as i64,
                        0,
                    ));
                    q0 = q1;
                }
                Tensor::cat(outs, 1)
            };
            let dt = t0.elapsed().as_secs_f64();
            let st = burn_rocket::stats();
            if rep > 0 {
                total_s += dt;
                conv += st.convert_s;
                npu_s += st.npu_s;
                calls += st.calls;
            }
            if rep == reps {
                npu_out = o.into_data().try_to_vec().expect("f32");
            }
        }
        let (cos, maxd) = compare(&fused_out, &npu_out);
        println!(
            "{name:<12} {:8.3} s/rep   convert {:6.3} s  npu {:6.3} s  {} calls  RssAnon {} MiB  vs fused cos {cos:.9} max|d| {maxd:.2e}",
            total_s / reps as f64,
            conv / reps as f64,
            npu_s / reps as f64,
            calls / reps as u64,
            rss_anon_mib()
        );
    }

    if let Some(m) = burn_rocket::mem_stats() {
        println!(
            "npu scratch: attention BO {} MiB + host {} MiB, matmul BO {} MiB",
            m.attention_bo_bytes as f64 / 1048576.0,
            m.attention_host_bytes as f64 / 1048576.0,
            m.matmul_bo_bytes as f64 / 1048576.0
        );
        if let Some(w) = m.attn_workers.first() {
            println!(
                "npu attention worker shape {}x{} over {} workers",
                w.tp,
                w.kn,
                m.attn_workers.len()
            );
        }
    }
}

/// The layer's packed projections at several row counts (log-gemma §18.6).
///
/// Frame batching turns `frames` calls of `M = patches` rows into one call of
/// `M = frames * patches`. If `ns/row` is flat in `M`, the NPU work (and the
/// host conversion around it) is proportional to the rows and batching buys only
/// the per-call overhead.
fn matmul_scale(reps: usize) {
    let device = Device::flex();
    burn_rocket::init(4).expect("rocket init");
    println!(
        "vision tower projections on the NPU: hidden {HIDDEN}, mlp {INTER}, reps {reps} (1 warmup each)"
    );
    println!("start RssAnon {} MiB", rss_anon_mib());
    let w = |n: usize, k: usize, seed: u64| {
        Tensor::<2>::from_data(TensorData::new(rnd_vec(n * k, seed), [n, k]), &device)
    };
    let qkv = burn_rocket::pack3(
        w(HIDDEN, HIDDEN, 1),
        w(HIDDEN, HIDDEN, 2),
        w(HIDDEN, HIDDEN, 3),
    );
    let o = burn_rocket::pack(w(HIDDEN, HIDDEN, 4));
    let gateup = burn_rocket::pack2(w(INTER, HIDDEN, 5), w(INTER, HIDDEN, 6));
    let down = burn_rocket::pack(w(HIDDEN, INTER, 7));
    let arms: [(&str, &burn_rocket::WeightId, usize, usize); 4] = [
        ("q|k|v", &qkv, 3 * HIDDEN, HIDDEN),
        ("o", &o, HIDDEN, HIDDEN),
        ("gate|up", &gateup, 2 * INTER, HIDDEN),
        ("down", &down, HIDDEN, INTER),
    ];
    println!("start+packed RssAnon {} MiB", rss_anon_mib());
    for m in [1170usize, 2340, 4680, 9360] {
        for (name, id, n, k) in arms {
            let x = Tensor::<2>::from_data(TensorData::new(rnd_vec(m * k, 11), [m, k]), &device);
            let mut total = 0.0;
            for rep in 0..=reps {
                let t0 = Instant::now();
                let y = burn_rocket::matmul(x.clone(), id);
                let _ = y.into_data();
                if rep > 0 {
                    total += t0.elapsed().as_secs_f64();
                }
            }
            let ms = total / reps as f64 * 1e3;
            println!(
                "{name:<8} M={m:>5} N={n:>4} K={k:>4} {ms:8.2} ms/call {:6.1} ns/row  RssAnon {} MiB",
                ms * 1e6 / m as f64,
                rss_anon_mib()
            );
        }
    }
}

/// Fused attention with `batch` frames of the same per-frame work: does the
/// kernel's `batch * heads` parallelism help beyond the 12 heads of one frame?
fn attention_batch(p: usize, reps: usize) {
    let device = Device::flex();
    println!(
        "fused vision attention, batch scaling: patches {p}, heads {HEADS}, head_dim {HEAD_DIM}, scale 1.0, reps {reps}"
    );
    println!("start RssAnon {} MiB", rss_anon_mib());
    let one = rnd_vec(p * HEADS * HEAD_DIM, 42);
    for b in [1usize, 2, 4, 8] {
        let mut data = Vec::with_capacity(b * one.len());
        for _ in 0..b {
            data.extend_from_slice(&one);
        }
        let t = Tensor::<4>::from_data(TensorData::new(data, [b, HEADS, p, HEAD_DIM]), &device);
        let mut total = 0.0;
        for rep in 0..=reps {
            let t0 = Instant::now();
            let o = attention(
                t.clone(),
                t.clone(),
                t.clone(),
                None,
                None,
                AttentionModuleOptions {
                    scale: Some(1.0),
                    softcap: None,
                    is_causal: false,
                },
            );
            let _ = o.into_data();
            if rep > 0 {
                total += t0.elapsed().as_secs_f64();
            }
        }
        let per = total / reps as f64;
        println!(
            "B={b} {per:8.3} s/rep  {:8.3} s/frame  RssAnon {} MiB",
            per / b as f64,
            rss_anon_mib()
        );
    }
}

/// The host softmax alone: exact vs fast through `Softmax::run`, with and
/// without a causal mask (the text backbone's mask mode), on a padded score
/// matrix shaped like the attention driver's (`tp % 4`, `kn % 32`).
///
/// One `exp` per element, so `Gelem/s` is the `exp/s` rate; `GB/s` counts the
/// f16 traffic the op pays (read the scores, write the probabilities).
fn softmax_bench(kn: usize, reps: usize, tp: usize) {
    let (kn_real, tp_real) = (kn, tp);
    let kn = kn.div_ceil(32) * 32;
    let tp = tp.div_ceil(4) * 4;
    let n_tokens = tp.saturating_sub(4);
    let kernel = if cfg!(all(target_arch = "aarch64", target_feature = "fp16")) {
        "NEON 8-lane, hardware f16"
    } else {
        "scalar"
    };
    println!(
        "host softmax: tp {tp} ({tp_real} rows), kn {kn} ({kn_real} keys), {kernel}, reps {reps}",
    );
    println!("start RssAnon {} MiB", rss_anon_mib());

    let mut state = 0x5eed_1234_9abc_def0u64;
    let mut rnd = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 32.0
    };
    let scores: Vec<f16> = (0..tp * kn).map(|_| f16::from_f32(rnd())).collect();
    let mut causal = vec![f16::ZERO; n_tokens * kn_real];
    for i in 0..n_tokens {
        for j in (i + 1)..kn_real {
            causal[i * kn_real + j] = f16::NEG_INFINITY;
        }
    }

    let elems = (tp * kn) as f64;
    for (mask_name, mask) in [("none", None), ("causal", Some(&causal[..]))] {
        let mut arms = [
            (
                "exact",
                SoftmaxMode::Exact,
                vec![f16::ZERO; tp * kn],
                0.0f64,
            ),
            ("fast", SoftmaxMode::Fast, vec![f16::ZERO; tp * kn], 0.0f64),
        ];
        // Warm up both arms (first-touch pages, row buffer growth).
        for (_, mode, out, _) in arms.iter_mut() {
            Softmax::with_mode(*mode).run(&scores, out, mask, n_tokens, kn_real, tp, kn, 1.0, 0.0);
        }
        for rep in 0..reps {
            let order: [usize; 2] = if rep % 2 == 0 { [0, 1] } else { [1, 0] };
            for a in order {
                let (_, mode, out, total) = &mut arms[a];
                let t0 = Instant::now();
                Softmax::with_mode(*mode)
                    .run(&scores, out, mask, n_tokens, kn_real, tp, kn, 1.0, 0.0);
                *total += t0.elapsed().as_secs_f64();
            }
        }
        let mut s = [0.0f64; 2];
        for (i, (name, _, _, total)) in arms.iter().enumerate() {
            s[i] = total / reps as f64;
            println!(
                "mask {mask_name:<6} {name:<5} {:8.4} s/rep  {:6.3} GB/s  {:6.3} Gexp/s",
                s[i],
                elems * 4.0 / s[i] / 1e9,
                elems / s[i] / 1e9
            );
        }
        let (cos, maxd, ulps) = compare_f16(&arms[1].2, &arms[0].2);
        println!(
            "  fast/exact {:.2}x  cos {cos:.9}  max|d| {maxd:.2e}  max {ulps} f16 ulp  RssAnon {} MiB",
            s[0] / s[1],
            rss_anon_mib()
        );
    }
}

/// Cosine, max|d| and max f16-code distance between two non-negative f16
/// matrices.
fn compare_f16(a: &[f16], b: &[f16]) -> (f64, f32, u32) {
    let (mut dot, mut na, mut nb, mut maxd) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    let mut ulps = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        let (xf, yf) = (x.to_f64(), y.to_f64());
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
        maxd = maxd.max((xf - yf).abs() as f32);
        // Probabilities are non-negative, so the f16 code order is monotone.
        ulps = ulps.max(u32::from(x.to_bits()).abs_diff(u32::from(y.to_bits())));
    }
    (dot / (na.sqrt() * nb.sqrt()), maxd, ulps)
}

/// Deterministic pseudo-random f32 values in [-0.05, 0.05).
fn rnd_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 33) as f32 / (1u64 << 31) as f32;
            (u - 0.5) * 0.1
        })
        .collect()
}

/// Cosine and max|d| between two equal-length f32 vectors.
fn compare(a: &[f32], b: &[f32]) -> (f64, f32) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxd) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
        maxd = maxd.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxd)
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
