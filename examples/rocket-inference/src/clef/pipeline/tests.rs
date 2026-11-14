use super::*;
use crate::clef::encode::EncodedQuestion;
use crate::clef::testutil::{f32s, read_npz};
use serde_json::json;
use std::path::PathBuf;

const MODEL_DIR: &str = "/home/royalcat/models/clef-omni";
const DUMP_DIR: &str = "/tmp/opencode/clef/dump";

fn eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| eq(v, w)))
        }
        _ => a == b,
    }
}

fn two_question_request() -> ClefRequest {
    ClefRequest::from_value(json!({
        "model": "clef-omni",
        "state": "x",
        "questions": {
            "zeta": {"type": "choice", "criteria": {"paid": "p", "draft": "d"}},
            "alpha": {"type": "noul"},
        },
    }))
    .unwrap()
}

fn encoded_for(req: &ClefRequest) -> EncodedRecord {
    let questions = req
        .questions
        .iter()
        .map(|(id, q)| {
            let option_ids: Vec<String> = crate::clef::answer::option_ids(q).unwrap();
            EncodedQuestion {
                id: id.clone(),
                question_type: 0,
                question_span: (0, 1),
                option_spans: option_ids.iter().map(|_| (0, 1)).collect(),
                option_ids,
            }
        })
        .collect();
    EncodedRecord {
        input_ids: vec![0; 42],
        questions,
    }
}

#[test]
fn the_response_has_the_systemone_shape_in_request_order() {
    let req = two_question_request();
    let enc = encoded_for(&req);
    // zeta scores [draft, paid]; alpha scores [true, false].
    let a = assemble(&req, &enc, &[vec![0.0, 2.0], vec![1.0, 1.0]]).unwrap();
    let keys: Vec<&String> = a.response["answers"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["zeta", "alpha"]);
    assert_eq!(a.response["model"], "clef-omni");
    assert_eq!(
        a.response["usage"],
        json!({"input_tokens": 42, "output_tokens": 0})
    );
    let z = &a.response["answers"]["zeta"];
    assert_eq!(z["choice"], "paid");
    assert_eq!(z["confidence"], 0.8808);
    // `probabilities` follow the request's criteria order (paid, draft), not the scored one.
    let order: Vec<&String> = z["probabilities"].as_object().unwrap().keys().collect();
    assert_eq!(order, ["paid", "draft"]);
    assert_eq!(
        a.response["answers"]["alpha"],
        json!({"type": "noul", "noul": 0.5})
    );
    assert_eq!((a.tokens, a.questions), (42, 2));
    assert_eq!(a.probabilities[0].0, "zeta");
}

#[test]
fn a_mismatched_encoding_is_an_error_not_a_panic() {
    let req = two_question_request();
    let mut enc = encoded_for(&req);
    enc.questions[0].id = "nope".into();
    assert!(assemble(&req, &enc, &[vec![0.0, 1.0], vec![0.0, 1.0]]).is_err());
}

fn fixture(id: &str) -> ClefRequest {
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("../../../tools/clef_fixtures.json")).unwrap();
    let rec = fixtures.into_iter().find(|f| f["id"] == id).unwrap();
    ClefRequest::from_value(rec).unwrap()
}

/// End to end against the f32 oracle dumps: probabilities within 1e-4 and the rounded
/// answers equal. Streams the whole 66 GiB checkpoint per record (~1-2 min each).
#[test]
#[ignore = "streams the 66 GiB checkpoint per record; run with --ignored (CLEF_IDS=a,b to select)"]
fn clef_answers_match_the_oracle() {
    let dir = PathBuf::from(MODEL_DIR);
    if !dir.join("joint_head.safetensors").exists() {
        eprintln!("skipping: {MODEL_DIR} is absent");
        return;
    }
    let ids = std::env::var("CLEF_IDS").unwrap_or_else(|_| "checkout-errors,invoice".into());
    let model = ClefModel::load(&dir).unwrap();
    for id in ids.split(',') {
        let dump = PathBuf::from(DUMP_DIR);
        let Ok(text) = std::fs::read_to_string(dump.join(format!("{id}.json"))) else {
            eprintln!("skipping {id}: no oracle dump in {DUMP_DIR} (docs/recipes/clef-oracle.md)");
            continue;
        };
        let oracle: Value = serde_json::from_str(&text).unwrap();
        let npz = read_npz(&dump.join(format!("{id}.npz")));
        let t0 = std::time::Instant::now();
        let got = model.answer(&fixture(id), 64000).unwrap();
        eprintln!(
            "{id}: {} tokens in {:.1}s",
            got.tokens,
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(
            got.response["usage"], oracle["usage"],
            "{id}: usage (token count)"
        );
        let mut worst = 0f64;
        for (qid, probs) in &got.probabilities {
            let (_, _, bytes) = &npz[&format!("probs_{qid}")];
            let want = f32s(bytes);
            let q = oracle["questions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|q| q["id"] == qid.as_str())
                .unwrap();
            for (opt, w) in q["option_ids"].as_array().unwrap().iter().zip(&want) {
                let p = probs[opt.as_str().unwrap()];
                worst = worst.max((p - f64::from(*w)).abs());
            }
        }
        eprintln!("{id}: max |dp| = {worst:.3e}");
        assert!(worst <= 1e-4, "{id}: max |dp| {worst:.3e}");
        assert!(
            eq(&got.response["answers"], &oracle["answers"]),
            "{id}: answers differ\n got {}\nwant {}",
            got.response["answers"],
            oracle["answers"]
        );
        assert_eq!(got.response["model"], oracle["model"]);
    }
}

const PACK_DIR: &str = "/home/royalcat/models/clef-pack-g128";

/// End to end through the int8 pack (CPU, packed experts dequantized to f32) against the
/// f32 BF16-source oracle dumps. The int8 quantization moves probabilities (the
/// AutoRound card measured max 0.044, mean 0.006 on its own records); every question's
/// top option must agree. Prints per-question probabilities, max and mean |dp|.
#[test]
#[ignore = "needs the full clef pack (CLEF_PACK_DIR) and oracle dumps; ~1 min per record"]
fn clef_pack_answers_match_the_oracle_top1() {
    let dir = PathBuf::from(std::env::var("CLEF_PACK_DIR").unwrap_or_else(|_| PACK_DIR.into()));
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no clef pack at {}", dir.display());
        return;
    }
    let ids = std::env::var("CLEF_IDS")
        .unwrap_or_else(|_| "checkout-errors,invoice,long-incident".into());
    let model = ClefModel::load(&dir).unwrap();
    let (mut worst, mut sum, mut n) = (0f64, 0f64, 0usize);
    for id in ids.split(',') {
        let dump = PathBuf::from(DUMP_DIR);
        let Ok(text) = std::fs::read_to_string(dump.join(format!("{id}.json"))) else {
            eprintln!("skipping {id}: no oracle dump in {DUMP_DIR} (docs/recipes/clef-oracle.md)");
            continue;
        };
        let oracle: Value = serde_json::from_str(&text).unwrap();
        let npz = read_npz(&dump.join(format!("{id}.npz")));
        let t0 = std::time::Instant::now();
        let got = model.answer(&fixture(id), 64000).unwrap();
        eprintln!(
            "{id}: {} tokens in {:.1}s",
            got.tokens,
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(got.response["usage"], oracle["usage"], "{id}: token count");
        let (mut rec_worst, mut rec_sum, mut rec_n) = (0f64, 0f64, 0usize);
        for (qid, probs) in &got.probabilities {
            let want = f32s(&npz[&format!("probs_{qid}")].2);
            let q = oracle["questions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|q| q["id"] == qid.as_str())
                .unwrap();
            let opts: Vec<&str> = q["option_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o.as_str().unwrap())
                .collect();
            let mut top = (0usize, f64::MIN);
            let mut top_want = (0usize, f32::MIN);
            for (i, (opt, w)) in opts.iter().zip(&want).enumerate() {
                let p = probs[*opt];
                rec_worst = rec_worst.max((p - f64::from(*w)).abs());
                rec_sum += (p - f64::from(*w)).abs();
                rec_n += 1;
                if p > top.1 {
                    top = (i, p);
                }
                if *w > top_want.1 {
                    top_want = (i, *w);
                }
                eprintln!("  {id}/{qid}/{opt}: pack {p:.4} oracle {w:.4}");
            }
            assert_eq!(
                top.0, top_want.0,
                "{id}/{qid}: top-1 differs from the oracle"
            );
        }
        eprintln!(
            "{id}: max |dp| = {rec_worst:.4}, mean |dp| = {:.4} over {rec_n} options",
            rec_sum / rec_n as f64
        );
        worst = worst.max(rec_worst);
        sum += rec_sum;
        n += rec_n;
    }
    eprintln!(
        "all records: max |dp| = {worst:.4}, mean |dp| = {:.4} over {n} options",
        sum / n.max(1) as f64
    );
    assert!(n > 0, "no record ran");
    // The card's own measurement was 0.044 max / 0.006 mean on other records; far above
    // it would point at an unpacking or dequantization bug.
    assert!(worst < 0.1, "max |dp| {worst:.4}");
}
