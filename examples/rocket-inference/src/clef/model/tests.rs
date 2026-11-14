use super::*;
use crate::clef::head::{ClefHead, LmHeadFile, QuestionSpans};
use crate::clef::math::{cosine, softmax};
use crate::clef::testutil::{f32s, read_npz};
use serde_json::Value;
use std::path::PathBuf;

include!("mini_expected.in");

const MODEL_DIR: &str = "/home/royalcat/models/clef-omni";
const DUMP_DIR: &str = "/tmp/opencode/clef/dump";

// ---- tiny synthetic model (weights = closed formula, mirrored by tools/clef_mini_gen.py)

fn val(n: usize, salt: f64, scale: f64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f64 * 0.37 + salt).sin() * scale) as f32)
        .collect()
}

fn one_plus(n: usize, salt: f64, scale: f64) -> Vec<f32> {
    val(n, salt, scale).into_iter().map(|v| 1.0 + v).collect()
}

fn mini_cfg() -> TextConfig {
    TextConfig {
        hidden: 64,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        experts: 4,
        top_k: 2,
        inter: 32,
        eps: 1e-6,
        theta: 1e6,
        norm_topk: true,
    }
}

fn mini_layer(l: usize) -> (LayerWeights, MemExperts) {
    let c = mini_cfg();
    let (h, d, i) = (c.hidden, c.head_dim, c.inter);
    let s = 10.0 * l as f64;
    let w = LayerWeights {
        input_norm: one_plus(h, s + 1.0, 0.1),
        post_norm: one_plus(h, s + 2.0, 0.1),
        q: val(c.heads * d * h, s + 3.0, 0.3),
        k: val(c.kv_heads * d * h, s + 4.0, 0.3),
        v: val(c.kv_heads * d * h, s + 5.0, 0.3),
        o: val(h * c.heads * d, s + 6.0, 0.3),
        q_norm: one_plus(d, s + 7.0, 0.1),
        k_norm: one_plus(d, s + 8.0, 0.1),
        router: val(c.experts * h, s + 9.0, 0.5),
    };
    let ex = (0..c.experts)
        .map(|e| {
            (
                val(i * h, s + 20.0 + e as f64, 0.3),
                val(i * h, s + 30.0 + e as f64, 0.3),
                val(h * i, s + 40.0 + e as f64, 0.3),
            )
        })
        .collect();
    (w, MemExperts(ex))
}

struct MemExperts(Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>);

impl ExpertSource for MemExperts {
    fn expert(&self, _layer: usize, e: usize) -> Result<ExpertWeights> {
        let (g, u, d) = &self.0[e];
        Ok(ExpertWeights {
            gate: g.clone(),
            up: u.clone(),
            down: d.clone(),
        })
    }
}

#[test]
fn the_mini_decoder_matches_the_torch_manual_layer() {
    let c = mini_cfg();
    let s = 5;
    let rope = Rope::new(&c, s);
    let mut x = val(s * c.hidden, 99.0, 1.0);
    let want_h = [&HIDDEN_L0[..], &HIDDEN_L1[..]];
    let want_i = [&IDS_L0[..], &IDS_L1[..]];
    let want_w = [&WTS_L0[..], &WTS_L1[..]];
    for l in 0..2 {
        let (w, ex) = mini_layer(l);
        let out = layer_forward(&c, &w, &ex, l, &x, s, &rope, 2).unwrap();
        assert_eq!(out.router_ids, want_i[l], "layer {l} router ids");
        for (a, b) in out.router_weights.iter().zip(want_w[l]) {
            assert!((a - b).abs() < 1e-5, "layer {l} router weight {a} vs {b}");
        }
        let peak = want_h[l].iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = out
            .hidden
            .iter()
            .zip(want_h[l])
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst / peak < 2e-6,
            "layer {l}: max diff {worst} (peak {peak})"
        );
        assert!(cosine(&out.hidden, want_h[l]) > 0.999_999);
        x = out.hidden;
    }
}

#[test]
fn single_thread_and_pooled_layers_agree() {
    let c = mini_cfg();
    let s = 5;
    let rope = Rope::new(&c, s);
    let x = val(s * c.hidden, 99.0, 1.0);
    let (w, ex) = mini_layer(0);
    let a = layer_forward(&c, &w, &ex, 0, &x, s, &rope, 1).unwrap();
    let b = layer_forward(&c, &w, &ex, 0, &x, s, &rope, 4).unwrap();
    for (p, q) in a.hidden.iter().zip(&b.hidden) {
        assert!((p - q).abs() < 1e-4);
    }
    assert_eq!(a.router_ids, b.router_ids);
}

#[test]
fn rope_is_identity_at_position_zero_and_a_pure_rotation() {
    let c = mini_cfg();
    let rope = Rope::new(&c, 3);
    let d = c.head_dim;
    let base = val(3 * d, 1.0, 1.0);
    let mut x = base.clone();
    rope.apply(&mut x, 3, 1);
    assert_eq!(&x[..d], &base[..d]);
    for t in 0..3 {
        let n0: f32 = base[t * d..(t + 1) * d].iter().map(|v| v * v).sum();
        let n1: f32 = x[t * d..(t + 1) * d].iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-5 * n0);
    }
    // slot 0 has inv_freq 1: position 2 rotates pair (0, d/2) by exactly 2 rad.
    let (a, b) = (base[2 * d], base[2 * d + d / 2]);
    let want = a * 2f32.cos() - b * 2f32.sin();
    assert!((x[2 * d] - want).abs() < 1e-6);
}

#[test]
fn routing_breaks_ties_to_the_lower_id_and_renormalizes() {
    let mut c = mini_cfg();
    c.experts = 4;
    c.top_k = 2;
    let (ids, w, margin) = route(&c, &[1.0, 1.0, 1.0, 0.0], 1);
    assert_eq!(ids, vec![0, 1]);
    assert!((w[0] - 0.5).abs() < 1e-6 && (w[1] - 0.5).abs() < 1e-6);
    assert!(margin[0].abs() < 1e-7, "tie: tiny margin {}", margin[0]);
    let (ids, w, _) = route(&c, &[0.0, 3.0, 1.0, 2.0], 1);
    assert_eq!(ids, vec![1, 3]);
    let p1 = 3f32.exp() / (3f32.exp() + 2f32.exp());
    assert!((w[0] - p1).abs() < 1e-6);
}

#[test]
fn matmul_nt_matches_a_naive_product() {
    let (m, k, o) = (7, 33, 5);
    let x = val(m * k, 0.3, 1.0);
    let w = val(o * k, 0.7, 1.0);
    for threads in [1, 3] {
        let y = matmul_nt(&x, m, k, &w, o, threads);
        for i in 0..m {
            for j in 0..o {
                let want: f64 = (0..k)
                    .map(|t| x[i * k + t] as f64 * w[j * k + t] as f64)
                    .sum();
                assert!((y[i * o + j] as f64 - want).abs() < 1e-4);
            }
        }
    }
}

// ---- parity with the Phase-1 oracle dumps (needs the 66 GiB checkpoint) -------------

/// Cosine accumulated in f64 (the f32 `math::cosine` loses ~1e-3 over 600k elements).
fn cosine64(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

fn peak_rss_mib() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let f = |k: &str| {
        s.lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            / 1024
    };
    (f("VmHWM:"), f("RssAnon:"))
}

fn span(v: &Value) -> (usize, usize) {
    (
        v[0].as_u64().unwrap() as usize,
        v[1].as_u64().unwrap() as usize,
    )
}

#[test]
#[ignore = "streams the 66 GiB checkpoint per record; run with --ignored (CLEF_IDS=a,b to select)"]
fn clef_backbone_parity() {
    let dir = PathBuf::from(MODEL_DIR);
    let dump = PathBuf::from(DUMP_DIR);
    if !dir.join("model.safetensors.index.json").exists() || !dump.join("invoice.npz").exists() {
        eprintln!("skipping: checkpoint or oracle dumps absent ({MODEL_DIR}, {DUMP_DIR})");
        return;
    }
    let ids_env =
        std::env::var("CLEF_IDS").unwrap_or("checkout-errors,invoice,long-incident".into());
    let bb = Backbone::open(&dir).unwrap();
    let head = ClefHead::load(&dir).unwrap();
    let lm = LmHeadFile::open(&dir).unwrap();
    for id in ids_env.split(',') {
        let Ok(_) = std::fs::metadata(dump.join(format!("{id}.npz"))) else {
            eprintln!("skipping {id}: no dump");
            continue;
        };
        let npz = read_npz(&dump.join(format!("{id}.npz")));
        let meta: Value =
            serde_json::from_slice(&std::fs::read(dump.join(format!("{id}.json"))).unwrap())
                .unwrap();
        let ids: Vec<u32> = npz["input_ids"]
            .2
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as u32)
            .collect();
        let (s, h, k) = (ids.len(), bb.cfg.hidden, bb.cfg.top_k);
        let t0 = std::time::Instant::now();
        let mut rows = Vec::new();
        let mut layer_times = Vec::new();
        let mut last = std::time::Instant::now();
        let out = bb
            .forward_tapped(&ids, &ForwardOpts::default(), &mut |t: &Tap| {
                layer_times.push(last.elapsed().as_secs_f64());
                last = std::time::Instant::now();
                let l = t.layer;
                let (mut cos, mut rel) = (f32::NAN, f32::NAN);
                if let Some(want) = npz.get(&format!("hidden_L{l}")) {
                    let want = f32s(&want.2);
                    cos = cosine64(t.hidden, &want);
                    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let worst = t
                        .hidden
                        .iter()
                        .zip(&want)
                        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                    rel = worst / peak;
                }
                let wid: Vec<i32> = npz[&format!("router_ids_L{l}")]
                    .2
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c))
                    .collect();
                let (mut set_diff, mut order_only) = (0usize, 0usize);
                let mut min_margin = f32::INFINITY;
                for tok in 0..s {
                    let mut a: Vec<i32> = t.router_ids[tok * k..(tok + 1) * k]
                        .iter()
                        .map(|&v| v as i32)
                        .collect();
                    let mut b = wid[tok * k..(tok + 1) * k].to_vec();
                    if a != b {
                        let same_order = a.clone();
                        a.sort();
                        b.sort();
                        if a != b {
                            set_diff += 1;
                            min_margin = min_margin.min(t.router_margins[tok]);
                        } else {
                            let _ = same_order;
                            order_only += 1;
                        }
                    }
                }
                rows.push((l, cos, rel, set_diff, order_only, min_margin));
            })
            .unwrap();
        let wall = t0.elapsed().as_secs_f64();
        let (hwm, anon) = peak_rss_mib();
        eprintln!("== {id}: {s} tokens, wall {wall:.1}s, VmHWM {hwm} MiB, RssAnon {anon} MiB");
        eprintln!(
            "   layer times: min {:.2}s max {:.2}s",
            layer_times.iter().cloned().fold(f64::MAX, f64::min),
            layer_times.iter().cloned().fold(0.0, f64::max)
        );
        let mut worst_cos = (1f32, 0usize);
        let mut worst_rel = (0f32, 0usize);
        let (mut sd, mut oo) = (0, 0);
        for &(l, c, r, d, o, m) in &rows {
            if !c.is_nan() {
                if c < worst_cos.0 {
                    worst_cos = (c, l);
                }
                if r > worst_rel.0 {
                    worst_rel = (r, l);
                }
            }
            sd += d;
            oo += o;
            if d > 0 {
                eprintln!(
                    "   layer {l}: {d} tokens with a different expert SET (min 8th-9th margin {m:.3e}), {o} order-only"
                );
            }
        }
        eprintln!(
            "   worst cosine {:.9} (layer {}), worst max|d|/peak {:.3e} (layer {}), router set mismatches {sd}, order-only {oo}",
            worst_cos.0, worst_cos.1, worst_rel.0, worst_rel.1
        );
        let want_n = f32s(&npz["final_normed"].2);
        let fcos = cosine64(&out.final_normed, &want_n);
        let peak = want_n.iter().fold(0f32, |m, v| m.max(v.abs()));
        let frel = out
            .final_normed
            .iter()
            .zip(&want_n)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
            / peak;
        eprintln!("   final_normed cosine {fcos:.9}, max|d|/peak {frel:.3e}");
        // end to end through the Phase-2 head
        let qs = meta["questions"].as_array().unwrap();
        let questions: Vec<QuestionSpans> = qs
            .iter()
            .map(|q| QuestionSpans {
                question_type: q["type"].as_u64().unwrap() as usize,
                question_span: span(&q["question_span"]),
                option_spans: q["option_spans"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(span)
                    .collect(),
            })
            .collect();
        let logits = head
            .forward(&out.final_normed, &ids, &questions, &lm)
            .unwrap();
        let mut dp_max = 0f32;
        for (q, got) in qs.iter().zip(&logits) {
            let qid = q["id"].as_str().unwrap();
            let wantp = f32s(&npz[&format!("probs_{qid}")].2);
            let gotp = softmax(got);
            let dp = wantp
                .iter()
                .zip(&gotp)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("   {qid}: max|dp| {dp:.3e}  probs {gotp:?}");
            dp_max = dp_max.max(dp);
        }
        assert_eq!(sd, 0, "{id}: router expert-set mismatches");
        assert!(
            worst_cos.0 >= 0.99999,
            "{id}: layer {} cosine {}",
            worst_cos.1,
            worst_cos.0
        );
        assert!(fcos >= 0.99999, "{id}: final cosine {fcos}");
        assert!(dp_max <= 1e-4, "{id}: max |dp| {dp_max}");
        let _ = h;
    }
}

#[test]
fn text_positions_give_exactly_the_plain_rope_tables() {
    let cfg = mini_cfg();
    let a = Rope::new(&cfg, 37);
    let b = Rope::from_positions(&cfg, &Positions::text(37));
    assert_eq!(a.cos, b.cos);
    assert_eq!(a.sin, b.sin);
    // different streams: slot 1 follows the height stream, slot 2 the width, slot 0 time
    let p = Positions {
        t: vec![0.0, 5.0],
        h: vec![0.0, 7.0],
        w: vec![0.0, 9.0],
    };
    let r = Rope::from_positions(&mini_cfg(), &p);
    let inv = |j: usize| (1.0 / 1e6f64.powf((2 * j) as f64 / 16.0)) as f32;
    for (j, pos) in [(0usize, 5.0f32), (1, 7.0), (2, 9.0), (3, 5.0)] {
        let want = ((pos * inv(j)) as f64).cos() as f32;
        assert_eq!(r.cos[8 + j], want, "slot {j}");
    }
}
