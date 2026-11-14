use super::*;
use serde_json::json;
use std::path::Path;

const MODEL_DIR: &str = "/home/royalcat/models/clef-omni";

fn tokenizer() -> Option<Tokenizer> {
    let p = Path::new(MODEL_DIR).join("tokenizer.json");
    if !p.exists() {
        eprintln!("skipping: {} is absent", p.display());
        return None;
    }
    Some(Tokenizer::from_file(p).unwrap())
}

#[test]
fn floats_print_like_python_repr() {
    for (f, s) in [
        (2.5, "2.5"),
        (100.0, "100.0"),
        (1e-7, "1e-07"),
        (1.5e-7, "1.5e-07"),
        (1e22, "1e+22"),
        (1e16, "1e+16"),
        (1e15, "1000000000000000.0"),
        (123456789012345680.0, "1.2345678901234568e+17"),
        (0.0001, "0.0001"),
        (0.00001, "1e-05"),
        (0.5, "0.5"),
        (-0.5, "-0.5"),
        (0.1, "0.1"),
        (7.31828, "7.31828"),
        (1234.5678, "1234.5678"),
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (f64::INFINITY, "Infinity"),
        (f64::NEG_INFINITY, "-Infinity"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
    ] {
        assert_eq!(py_float_repr(f), s, "{f:e}");
    }
}

#[test]
fn render_matches_python_json_dumps() {
    // sort_keys + compact separators + ensure_ascii=False; integers keep their digits,
    // floats go through repr.
    let v: Value = serde_json::from_str(
        r#"{"z":[1,2.5,true,null,"é",{"b":1,"a":[3,4]}],"a":{"y":1e-7,"x":100.0,"w":1e22,
        "v":-0.5,"u":123456789012345678901234567890,"t":-0,"s":1E2},"名前":"値","e":{},"l":[]}"#,
    )
    .unwrap();
    assert_eq!(
        render(&v),
        r#"{"a":{"s":100.0,"t":0,"u":123456789012345678901234567890,"v":-0.5,"w":1e+22,"x":100.0,"y":1e-07},"e":{},"l":[],"z":[1,2.5,true,null,"é",{"a":[3,4],"b":1}],"名前":"値"}"#
    );
    // Strings are verbatim at the top level and JSON-escaped when nested.
    assert_eq!(render(&json!("a \"q\" é\n")), "a \"q\" é\n");
    assert_eq!(
        render(&json!(["a \"q\" é\n\u{1}\u{7f}"])),
        "[\"a \\\"q\\\" é\\n\\u0001\u{7f}\"]"
    );
    assert_eq!(render(&Value::Null), "null");
    assert_eq!(render(&json!(true)), "true");
}

#[test]
fn options_follow_question_options() {
    let o = |q: Value| question_options(&q).unwrap();
    // noul defaults, user overrides for true/false only, null removes the description.
    let n = o(json!({"type": "noul"}));
    assert_eq!(n[0].0, "true");
    assert_eq!(n[1].0, "false");
    assert_eq!(
        n[0].1,
        Some(json!("The proposition is true or the answer is yes."))
    );
    let n = o(json!({"type": "noul", "criteria": {"true": "T", "extra": "x", "false": null}}));
    assert_eq!(n[0].1, Some(json!("T")));
    assert_eq!(n[1].1, None);
    // choice: sorted by key (code point), null description kept as None.
    let c = o(json!({"type": "choice", "criteria": {"b": 1, "a": null, "B": 2, "é": 3}}));
    let ids: Vec<&str> = c.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(ids, ["B", "a", "b", "é"]);
    assert_eq!(c[1].1, None);
    // score: index ids; a dict walks its keys.
    let s = o(json!({"type": "score", "criteria": ["x", null, 3]}));
    assert_eq!(s[0], ("0".into(), Some(json!("x"))));
    assert_eq!(s[1], ("1".into(), None));
    let s = o(json!({"type": "score", "criteria": {"lo": 1, "hi": 2}}));
    assert_eq!(s[1], ("1".into(), Some(json!("hi"))));
}

fn request_of(case: &Value) -> ClefRequest {
    let mut r = case["request"].clone();
    if r.get("model").is_none() {
        r["model"] = json!("clef-omni");
    }
    ClefRequest::from_value(r).unwrap()
}

/// Token ids and spans must be IDENTICAL to the release's Python `encode_record`
/// (`tools/clef_encode_expected.py` regenerates `testdata/encode_cases.json`).
#[test]
fn the_encoder_is_identical_to_the_python_release_on_every_case() {
    let Some(tok) = tokenizer() else { return };
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../testdata/encode_cases.json")).unwrap();
    assert!(cases.len() >= 20);
    let mut with_ids = 0;
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let req = request_of(case);
        let max_length = case["max_length"]
            .as_u64()
            .map_or(DEFAULT_MAX_LENGTH, |v| v as usize);
        let max_state = case["max_state_tokens"].as_u64().map(|v| v as usize);
        let got = encode_record(&tok, &req, max_length, max_state);
        if let Some(expected) = case["error"].as_str() {
            let err = got.expect_err(name);
            let invalid = err
                .downcast_ref::<Invalid>()
                .unwrap_or_else(|| panic!("{name}: {err:#}"));
            assert_eq!(invalid.0, expected, "{name}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let want: Vec<u32> = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(got.input_ids.len(), want.len(), "{name}: length");
        assert!(got.input_ids == want, "{name}: ids differ");
        let exp_q = case["questions_expected"].as_array().unwrap();
        assert_eq!(got.questions.len(), exp_q.len(), "{name}");
        let span = |v: &Value| {
            (
                v[0].as_u64().unwrap() as usize,
                v[1].as_u64().unwrap() as usize,
            )
        };
        for (g, e) in got.questions.iter().zip(exp_q) {
            assert_eq!(g.id, e["id"].as_str().unwrap(), "{name}");
            assert_eq!(
                g.question_type,
                e["type"].as_u64().unwrap() as usize,
                "{name}"
            );
            assert_eq!(
                g.question_span,
                span(&e["question_span"]),
                "{name} {}",
                g.id
            );
            let es: Vec<(usize, usize)> = e["option_spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(span)
                .collect();
            assert_eq!(g.option_spans, es, "{name} {}", g.id);
            let eo: Vec<&str> = e["option_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(g.option_ids, eo, "{name} {}", g.id);
        }
        with_ids += 1;
    }
    assert!(with_ids >= 20, "{with_ids} cases compared");
}

#[test]
fn the_three_oracle_dumps_have_the_same_ids() {
    // The phase-1 oracle dump records the ids the f32 reference actually ran.
    let Some(tok) = tokenizer() else { return };
    let dump = Path::new("/tmp/opencode/clef/dump");
    if !dump.exists() {
        eprintln!("skipping: {} is absent", dump.display());
        return;
    }
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("../../../tools/clef_fixtures.json")).unwrap();
    for id in ["checkout-errors", "invoice", "long-incident"] {
        let npz = crate::clef::testutil::read_npz(&dump.join(format!("{id}.npz")));
        let (descr, _shape, bytes) = &npz["input_ids"];
        assert_eq!(descr, "<i8");
        let want: Vec<u32> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as u32)
            .collect();
        let rec = fixtures.iter().find(|f| f["id"] == id).unwrap().clone();
        let req = ClefRequest::from_value(rec).unwrap();
        let got = encode_record(&tok, &req, DEFAULT_MAX_LENGTH, None).unwrap();
        assert!(got.input_ids == want, "{id}");
    }
}

#[test]
fn a_schema_that_does_not_fit_is_a_client_error() {
    let Some(tok) = tokenizer() else { return };
    let req = ClefRequest::from_value(json!({
        "model": "m", "state": "x",
        "questions": {"q": {"type": "noul"}},
    }))
    .unwrap();
    let err = encode_record(&tok, &req, 10, None).unwrap_err();
    assert!(err.downcast_ref::<Invalid>().is_some());
    assert!(err.to_string().starts_with("schema requires "), "{err}");
    // Exactly the fixed length leaves no state tokens and still encodes.
    let empty = ClefRequest::from_value(json!({
        "model": "m", "state": "",
        "questions": {"q": {"type": "noul"}},
    }))
    .unwrap();
    let fixed = encode_record(&tok, &empty, DEFAULT_MAX_LENGTH, None)
        .unwrap()
        .input_ids
        .len();
    let cut = encode_record(&tok, &req, fixed, None).unwrap();
    assert_eq!(cut.input_ids.len(), fixed);
    let full = encode_record(&tok, &req, fixed + 1000, None).unwrap();
    assert!(full.input_ids.len() > fixed);
}
