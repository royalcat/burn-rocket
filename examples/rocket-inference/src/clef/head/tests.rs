use super::*;
use crate::clef::answer;
use crate::clef::testutil::{f32s, model_dir, read_npz};
use serde_json::Value;
use std::path::PathBuf;

const DUMP_DIR: &str = "/tmp/opencode/clef/dump";

// ---- strict loader (synthetic checkpoint) ----------------------------------------

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>)]) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape) in tensors {
        let n: usize = shape.iter().product();
        let start = data.len();
        for i in 0..n {
            let v = ((i % 7) as f32 - 3.0) * 0.01;
            data.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let h = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

fn tiny_names(c: &HeadConfig) -> Vec<(String, Vec<usize>)> {
    let (h, w, ff) = (c.hidden_size, c.width, c.feedforward);
    let mut v: Vec<(String, Vec<usize>)> = Vec::new();
    let norm = |v: &mut Vec<_>, p: &str| {
        v.push((format!("{p}.weight"), vec![w]));
        v.push((format!("{p}.bias"), vec![w]));
    };
    let mha = |v: &mut Vec<_>, p: &str| {
        v.push((format!("{p}.in_proj_weight"), vec![3 * w, w]));
        v.push((format!("{p}.in_proj_bias"), vec![3 * w]));
        v.push((format!("{p}.out_proj.weight"), vec![w, w]));
        v.push((format!("{p}.out_proj.bias"), vec![w]));
    };
    v.push(("hidden_norm.weight".into(), vec![h]));
    v.push(("hidden_norm.bias".into(), vec![h]));
    for p in [
        "memory_projection",
        "question_projection",
        "option_question_projection",
        "global_projection",
        "option_context_projection",
        "option_lexical_projection",
    ] {
        v.push((format!("{p}.weight"), vec![w, h]));
    }
    v.push(("type_embedding.weight".into(), vec![3, w]));
    for i in 0..c.routing_layers {
        let p = format!("evidence_layers.{i}");
        norm(&mut v, &format!("{p}.query_norm"));
        norm(&mut v, &format!("{p}.memory_norm"));
        mha(&mut v, &format!("{p}.attention"));
        norm(&mut v, &format!("{p}.feedforward_norm"));
        v.push((format!("{p}.feedforward.0.weight"), vec![ff, w]));
        v.push((format!("{p}.feedforward.0.bias"), vec![ff]));
        v.push((format!("{p}.feedforward.3.weight"), vec![w, ff]));
        v.push((format!("{p}.feedforward.3.bias"), vec![w]));
    }
    norm(&mut v, "option_summary_norm");
    for i in 0..c.layers {
        let p = format!("layers.{i}");
        for n in ["norm1", "norm2", "norm3"] {
            norm(&mut v, &format!("{p}.{n}"));
        }
        mha(&mut v, &format!("{p}.self_attn"));
        mha(&mut v, &format!("{p}.multihead_attn"));
        v.push((format!("{p}.linear1.weight"), vec![ff, w]));
        v.push((format!("{p}.linear1.bias"), vec![ff]));
        v.push((format!("{p}.linear2.weight"), vec![w, ff]));
        v.push((format!("{p}.linear2.bias"), vec![w]));
    }
    norm(&mut v, "field_norm");
    norm(&mut v, "option_norm");
    v.push(("residual_scorer.0.weight".into(), vec![w, 4 * w]));
    v.push(("residual_scorer.0.bias".into(), vec![w]));
    v.push(("residual_scorer.3.weight".into(), vec![1, w]));
    v.push(("residual_scorer.3.bias".into(), vec![1]));
    for s in ["prior_logit_scale", "joint_logit_scale", "residual_gate"] {
        v.push((s.into(), vec![]));
    }
    v
}

fn tiny_dir(tag: &str, edit: impl FnOnce(&mut Vec<(String, Vec<usize>)>)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("clef-head-test-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("joint_head_config.json"),
        r#"{"hidden_size":8,"width":4,"routing_layers":1,"layers":2,"heads":2,"feedforward":8}"#,
    )
    .unwrap();
    let cfg = HeadConfig {
        hidden_size: 8,
        width: 4,
        routing_layers: 1,
        layers: 2,
        heads: 2,
        feedforward: 8,
    };
    let mut names = tiny_names(&cfg);
    edit(&mut names);
    write_safetensors(&dir.join("joint_head.safetensors"), &names);
    dir
}

struct Rows;
impl LmHeadRows for Rows {
    fn row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        for (i, o) in out.iter_mut().enumerate() {
            *o = ((id as usize * 3 + i) % 5) as f32 - 2.0;
        }
        Ok(())
    }
}

#[test]
fn loader_is_strict_and_a_tiny_head_runs() {
    let dir = tiny_dir("ok", |_| {});
    let head = ClefHead::load(&dir).unwrap();
    let hidden: Vec<f32> = (0..6 * 8)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
        .collect();
    let q = QuestionSpans {
        question_type: 1,
        question_span: (1, 3),
        option_spans: vec![(3, 4), (4, 6)],
    };
    let ids: Vec<u32> = (0..6).collect();
    let out = head.forward(&hidden, &ids, &[q.clone(), q], &Rows).unwrap();
    assert_eq!(out.len(), 2);
    assert!(
        out.iter()
            .all(|l| l.len() == 2 && l.iter().all(|v| v.is_finite()))
    );
    // Same inputs -> same logits; the two identical questions agree up to the
    // cross-field self-attention (they are identical rows, so exactly).
    assert_eq!(out[0], out[1]);
    // Bad spans are errors, not panics.
    let bad = QuestionSpans {
        question_type: 0,
        question_span: (4, 4),
        option_spans: vec![(0, 1)],
    };
    assert!(head.forward(&hidden, &ids, &[bad], &Rows).is_err());
    std::fs::remove_dir_all(dir).ok();

    let dir = tiny_dir("missing", |n| n.retain(|(k, _)| k != "field_norm.bias"));
    let e = ClefHead::load(&dir).err().unwrap().to_string();
    assert!(e.contains("missing tensor `field_norm.bias`"), "{e}");
    std::fs::remove_dir_all(dir).ok();

    let dir = tiny_dir("extra", |n| n.push(("surprise.weight".into(), vec![2])));
    let e = ClefHead::load(&dir).err().unwrap().to_string();
    assert!(
        e.contains("unexpected tensors") && e.contains("surprise"),
        "{e}"
    );
    std::fs::remove_dir_all(dir).ok();

    let dir = tiny_dir("shape", |n| {
        for (k, s) in n.iter_mut() {
            if k == "type_embedding.weight" {
                *s = vec![2, 4];
            }
        }
    });
    let e = ClefHead::load(&dir).err().unwrap().to_string();
    assert!(
        e.contains("type_embedding.weight") && e.contains("shape"),
        "{e}"
    );
    std::fs::remove_dir_all(dir).ok();
}

// ---- parity with the Phase-1 oracle dumps -----------------------------------------

/// Returns (max |dlogit|, max |dp|) over all questions of one dump.
fn parity(head: &ClefHead, lm: &LmHeadFile, id: &str) -> (f32, f32) {
    let npz = read_npz(&PathBuf::from(DUMP_DIR).join(format!("{id}.npz")));
    let meta: Value = serde_json::from_slice(
        &std::fs::read(PathBuf::from(DUMP_DIR).join(format!("{id}.json"))).unwrap(),
    )
    .unwrap();
    let (descr, shape, bytes) = &npz["final_normed"];
    assert_eq!(descr, "<f4");
    let hidden = f32s(bytes);
    assert_eq!(shape[1], head.config.hidden_size);
    let (idescr, _, ibytes) = &npz["input_ids"];
    assert_eq!(idescr, "<i8");
    let ids: Vec<u32> = ibytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c) as u32)
        .collect();
    let span = |v: &Value| {
        (
            v[0].as_u64().unwrap() as usize,
            v[1].as_u64().unwrap() as usize,
        )
    };
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
    let t = std::time::Instant::now();
    let logits = head.forward(&hidden, &ids, &questions, lm).unwrap();
    eprintln!(
        "clef parity {id}: head forward {:.2}s",
        t.elapsed().as_secs_f64()
    );
    let (mut dl, mut dp) = (0f32, 0f32);
    for (q, got) in qs.iter().zip(&logits) {
        let qid = q["id"].as_str().unwrap();
        let want = f32s(&npz[&format!("logits_{qid}")].2);
        let wantp = f32s(&npz[&format!("probs_{qid}")].2);
        assert_eq!(want.len(), got.len(), "{id}/{qid}");
        let gotp = crate::clef::math::softmax(got);
        for i in 0..want.len() {
            dl = dl.max((want[i] - got[i]).abs());
            dp = dp.max((wantp[i] - gotp[i]).abs());
        }
    }
    // The answers built from our logits equal the oracle's `answers` (needs the
    // fixture requests for the criteria order, so compare the numbers it carries).
    for (q, got) in qs.iter().zip(&logits) {
        let qid = q["id"].as_str().unwrap();
        let ids_: Vec<String> = q["option_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect();
        let probs = answer::probabilities(&ids_, got);
        let want = &meta["answers"][qid];
        match want["type"].as_str().unwrap() {
            "noul" => assert_eq!(
                answer::round4(probs["true"]),
                want["noul"].as_f64().unwrap(),
                "{id}/{qid}"
            ),
            "choice" => {
                for (k, v) in want["probabilities"].as_object().unwrap() {
                    assert_eq!(
                        answer::round4(probs[k]),
                        v.as_f64().unwrap(),
                        "{id}/{qid}/{k}"
                    );
                }
            }
            _ => {
                let s: f64 = (0..ids_.len())
                    .map(|i| i as f64 * probs[&i.to_string()])
                    .sum();
                assert_eq!(
                    answer::round4(s),
                    want["score"].as_f64().unwrap(),
                    "{id}/{qid}"
                );
            }
        }
    }
    (dl, dp)
}

#[test]
fn head_matches_the_oracle_dumps() {
    let dir = PathBuf::from(model_dir());
    let dump = PathBuf::from(DUMP_DIR);
    if !dir.join("joint_head.safetensors").exists() || !dump.join("invoice.npz").exists() {
        eprintln!(
            "skipping: Clef-Omni checkpoint or oracle dumps absent ({}, {DUMP_DIR})",
            model_dir()
        );
        return;
    }
    let head = ClefHead::load(&dir).unwrap();
    let lm = LmHeadFile::open(&dir).unwrap();
    for id in ["checkout-errors", "invoice", "long-incident"] {
        if !dump.join(format!("{id}.npz")).exists() {
            eprintln!("skipping {id}: no dump");
            continue;
        }
        let (dl, dp) = parity(&head, &lm, id);
        eprintln!("clef parity {id}: max|dlogit| = {dl:.3e}, max|dp| = {dp:.3e}");
        assert!(dl <= 1e-3, "{id}: max |dlogit| {dl}");
        assert!(dp <= 1e-4, "{id}: max |dp| {dp}");
    }
}
