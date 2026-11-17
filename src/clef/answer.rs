//! SystemOne answers from per-option probabilities (`systemone_answer` in the
//! release's `joint_schema_model.py`) and the option ordering the head scores in
//! (`question_options`).

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

/// Python `round(x, 4)`: correctly rounded to the exact decimal value of the double,
/// ties to even (Rust's `{:.4}` formatting is exact in the same way).
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

/// Option ids of a question in the order the model scores them (`question_options`):
/// noul = `true`,`false`; choice = criteria keys sorted by code point; score =
/// `"0".."n-1"`.
pub fn option_ids(question: &Value) -> Result<Vec<String>> {
    let ty = question["type"].as_str().context("question has no type")?;
    match ty {
        "noul" => Ok(vec!["true".into(), "false".into()]),
        "choice" => {
            let crit = question["criteria"]
                .as_object()
                .context("choice criteria must be an object")?;
            // Python sorts (str(key), value) tuples; keys are unique so the key decides.
            let mut keys: Vec<String> = crit.keys().cloned().collect();
            keys.sort_by(|a, b| a.chars().cmp(b.chars()));
            Ok(keys)
        }
        "score" => {
            let n = match &question["criteria"] {
                Value::Array(a) => a.len(),
                Value::Object(o) => o.len(),
                _ => bail!("score criteria must be a list"),
            };
            Ok((0..n).map(|i| i.to_string()).collect())
        }
        other => bail!("type must be noul, choice, or score (got {other})"),
    }
}

/// `dict(zip(option_ids, softmax(logits)))`, widened to f64 like `Tensor.tolist()`.
pub fn probabilities(ids: &[String], logits: &[f32]) -> BTreeMap<String, f64> {
    let p = super::math::softmax(logits);
    ids.iter()
        .cloned()
        .zip(p.into_iter().map(f64::from))
        .collect()
}

fn get(p: &BTreeMap<String, f64>, k: &str) -> Result<f64> {
    p.get(k)
        .copied()
        .with_context(|| format!("no probability for option `{k}`"))
}

/// Convert per-option probabilities for one question into a SystemOne answer.
///
/// The choice answer iterates `criteria` in *request* order (ties go to the first
/// maximal option) even though the model scores the options in sorted order.
pub fn systemone_answer(question: &Value, probs: &BTreeMap<String, f64>) -> Result<Value> {
    let ty = question["type"].as_str().context("question has no type")?;
    match ty {
        "noul" => Ok(json!({"type": "noul", "noul": round4(get(probs, "true")?)})),
        "choice" => {
            let options: Vec<&String> = question["criteria"]
                .as_object()
                .context("choice criteria must be an object")?
                .keys()
                .collect();
            let mut best: Option<(&String, f64)> = None;
            for o in &options {
                let p = get(probs, o)?;
                if best.is_none_or(|(_, bp)| p > bp) {
                    best = Some((o, p));
                }
            }
            let (choice, p) = best.context("choice has no options")?;
            let mut all = Map::new();
            for o in &options {
                all.insert((*o).clone(), json!(round4(get(probs, o)?)));
            }
            Ok(json!({
                "type": "choice",
                "choice": choice,
                "confidence": round4(p),
                "probabilities": all,
            }))
        }
        "score" => {
            let (legend_vals, n): (Vec<Value>, usize) = match &question["criteria"] {
                Value::Array(a) => (a.clone(), a.len()),
                Value::Object(o) => (o.keys().cloned().map(Value::from).collect(), o.len()),
                _ => bail!("score criteria must be a list"),
            };
            let levels: Vec<String> = (0..n).map(|i| i.to_string()).collect();
            let mut score = 0f64; // Python: sum() over index * p, left to right
            let mut conf = f64::NEG_INFINITY;
            let mut legend = Map::new();
            let mut all = Map::new();
            for (i, level) in levels.iter().enumerate() {
                let p = get(probs, level)?;
                score += i as f64 * p;
                if p > conf {
                    conf = p;
                }
                legend.insert(level.clone(), legend_vals[i].clone());
                all.insert(level.clone(), json!(round4(p)));
            }
            Ok(json!({
                "type": "score",
                "score": round4(score),
                "confidence": round4(conf),
                "legend": legend,
                "probabilities": all,
            }))
        }
        other => bail!("type must be noul, choice, or score (got {other})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probs(items: &[(&str, f64)]) -> BTreeMap<String, f64> {
        items.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn round4_matches_python_round() {
        assert_eq!(round4(0.97734), 0.9773);
        assert_eq!(round4(0.123450000001), 0.1235);
        // 0.00005 is below the exact tie in binary -> rounds down, like Python.
        assert_eq!(round4(0.00005), 0.0001);
        assert_eq!(round4(2.5), 2.5);
        assert_eq!(round4(0.9), 0.9);
        // Python: round(0.12345, 4) == 0.1235 (0.12345 is stored just above the tie).
        assert_eq!(round4(0.12345), 0.1235);
        // Python: round(0.5, 0) ties-to-even is not reachable at 4 digits; 0.00125 is
        // stored as 0.001250000000000000026 -> 0.0013.
        assert_eq!(round4(0.00125), 0.0013);
    }

    #[test]
    fn options_are_sorted_for_choice_and_indexed_for_score() {
        let q = json!({"type": "choice", "criteria": {"paid": "a", "draft": "b", "overdue": "c"}});
        assert_eq!(option_ids(&q).unwrap(), ["draft", "overdue", "paid"]);
        let q = json!({"type": "score", "criteria": ["x", "y", "z"]});
        assert_eq!(option_ids(&q).unwrap(), ["0", "1", "2"]);
        let q = json!({"type": "noul"});
        assert_eq!(option_ids(&q).unwrap(), ["true", "false"]);
        assert!(option_ids(&json!({"type": "weird"})).is_err());
    }

    #[test]
    fn noul_answer_is_the_rounded_true_probability() {
        let a = systemone_answer(
            &json!({"type": "noul"}),
            &probs(&[("true", 0.955_234), ("false", 0.044_766)]),
        )
        .unwrap();
        assert_eq!(a, json!({"type": "noul", "noul": 0.9552}));
    }

    #[test]
    fn choice_answer_follows_request_order_and_ties_pick_the_first() {
        // Request order: paid, draft, overdue (the model scored draft, overdue, paid).
        let q = json!({"type": "choice", "criteria": {"paid": "p", "draft": "d", "overdue": "o"}});
        let a = systemone_answer(
            &q,
            &probs(&[("draft", 0.0085), ("overdue", 0.97734), ("paid", 0.01416)]),
        )
        .unwrap();
        assert_eq!(a["choice"], "overdue");
        assert_eq!(a["confidence"], 0.9773);
        let keys: Vec<&String> = a["probabilities"].as_object().unwrap().keys().collect();
        assert_eq!(keys, ["paid", "draft", "overdue"]);
        assert_eq!(a["probabilities"]["paid"], 0.0142);
        assert_eq!(a["probabilities"]["draft"], 0.0085);
        // An exact tie goes to the first option in request order.
        let t = systemone_answer(
            &q,
            &probs(&[("draft", 0.4), ("overdue", 0.2), ("paid", 0.4)]),
        )
        .unwrap();
        assert_eq!(t["choice"], "paid");
    }

    #[test]
    fn score_answer_is_the_expectation_with_a_legend() {
        let q = json!({"type": "score", "criteria": ["P0", "P1", "P2"]});
        let a =
            systemone_answer(&q, &probs(&[("0", 0.0425), ("1", 0.0234), ("2", 0.9341)])).unwrap();
        // 0*0.0425 + 1*0.0234 + 2*0.9341 = 1.8916
        assert_eq!(a["score"], 1.8916);
        assert_eq!(a["confidence"], 0.9341);
        assert_eq!(a["legend"], json!({"0": "P0", "1": "P1", "2": "P2"}));
        assert_eq!(a["probabilities"]["1"], 0.0234);
    }

    #[test]
    fn missing_probabilities_are_errors() {
        let q = json!({"type": "choice", "criteria": {"a": 1}});
        assert!(systemone_answer(&q, &probs(&[("b", 1.0)])).is_err());
    }

    #[test]
    fn probabilities_are_a_softmax_over_the_sorted_ids() {
        let ids = vec!["a".to_string(), "b".to_string()];
        let p = probabilities(&ids, &[0.0, 0.0]);
        assert!((p["a"] - 0.5).abs() < 1e-7 && (p["b"] - 0.5).abs() < 1e-7);
    }
}
