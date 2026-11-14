//! The Clef-Omni SystemOne request (`systemone` in the release's
//! `joint_schema_model.py`) and its validation.
//!
//! The Python entry point is lenient about extra top-level keys (it only reads
//! `model`, `state`, `questions`, `images`, `audio`, `videos` and `media_kwargs`), and so
//! is this parser. The validation messages are the Python `ValueError`s verbatim.

use serde_json::{Map, Value};
use std::fmt;

/// Why a request is refused (HTTP 422): the client's fault, not the model's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invalid(pub String);

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Invalid {}

/// What a media request is answered with while the image/audio/video path is missing.
pub const MEDIA_NOT_IMPLEMENTED: &str = "Clef-Omni image/audio/video input is not implemented yet \
     (this build answers text-only requests): remove `images`, `audio` and `videos` from the request";

/// `QUESTION_TYPES` of the release (`noul`, `choice`, `score` -> 0, 1, 2).
pub fn question_type_index(ty: &str) -> Option<usize> {
    match ty {
        "noul" => Some(0),
        "choice" => Some(1),
        "score" => Some(2),
        _ => None,
    }
}

/// Python truthiness of a JSON value (`not question.get("criteria")`).
fn falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !*b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

#[derive(Debug, Clone)]
pub struct ClefRequest {
    /// Echoed in the response (`request["model"]`).
    pub model: String,
    /// Any JSON value; a string is used verbatim, anything else as compact sorted JSON.
    pub state: Value,
    /// Question id -> question object, request order.
    pub questions: Map<String, Value>,
    /// `images`/`audio`/`videos` entries present (any non-empty list).
    pub has_media: bool,
}

impl ClefRequest {
    /// Validate one request object (`POST /v1/systemone`, a `decide` case).
    pub fn from_value(value: Value) -> Result<Self, Invalid> {
        let Value::Object(mut object) = value else {
            return Err(Invalid(
                "expected a JSON object such as {\"model\": ..., \"state\": ..., \"questions\": {...}}"
                    .into(),
            ));
        };
        let model = match object.get("model") {
            Some(Value::String(m)) if object.contains_key("state") => m.clone(),
            _ => return Err(Invalid("model and state are required".into())),
        };
        let state = object.remove("state").unwrap_or(Value::Null);
        let questions = match object.remove("questions") {
            Some(Value::Object(q)) if !q.is_empty() => q,
            _ => return Err(Invalid("at least one question is required".into())),
        };
        for (id, question) in &questions {
            let Value::Object(q) = question else {
                return Err(Invalid(format!("{id}: a question must be a JSON object")));
            };
            let ty = q.get("type").and_then(Value::as_str);
            let Some(ty) = ty.filter(|t| question_type_index(t).is_some()) else {
                return Err(Invalid(format!(
                    "{id}: type must be noul, choice, or score"
                )));
            };
            let criteria = q.get("criteria").unwrap_or(&Value::Null);
            if ty != "noul" && falsy(criteria) {
                return Err(Invalid(format!("{id}: criteria must not be empty")));
            }
            // Shapes the Python code would crash on (AttributeError/TypeError): a clear 422.
            match (ty, criteria) {
                ("choice", Value::Object(_)) => {}
                ("choice", _) => {
                    return Err(Invalid(format!("{id}: choice criteria must be an object")));
                }
                ("score", Value::Array(_) | Value::Object(_)) => {}
                ("score", _) => {
                    return Err(Invalid(format!("{id}: score criteria must be a list")));
                }
                ("noul", Value::Null | Value::Object(_)) => {}
                ("noul", v) if falsy(v) => {}
                _ => return Err(Invalid(format!("{id}: noul criteria must be an object"))),
            }
        }
        let has_media = ["images", "audio", "videos"]
            .iter()
            .any(|k| object.get(*k).is_some_and(|v| !falsy(v)));
        Ok(Self {
            model,
            state,
            questions,
            has_media,
        })
    }

    /// The refusal for a media request (this build is text-only).
    pub fn media_refusal(&self) -> Option<Invalid> {
        self.has_media
            .then(|| Invalid(MEDIA_NOT_IMPLEMENTED.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: Value) -> Result<ClefRequest, String> {
        ClefRequest::from_value(v).map_err(|e| e.0)
    }

    fn ok() -> Value {
        json!({
            "model": "clef-omni",
            "state": "s",
            "questions": {
                "z": {"type": "choice", "criteria": {"a": "x"}},
                "a": {"type": "score", "criteria": ["l", "h"]},
                "m": {"type": "noul"},
            },
        })
    }

    #[test]
    fn a_valid_request_keeps_the_question_order() {
        let r = parse(ok()).unwrap();
        assert_eq!(r.model, "clef-omni");
        let ids: Vec<&String> = r.questions.keys().collect();
        assert_eq!(ids, ["z", "a", "m"]);
        assert!(!r.has_media);
        assert!(r.media_refusal().is_none());
    }

    #[test]
    fn model_and_state_are_required_like_the_python_check() {
        let mut v = ok();
        v.as_object_mut().unwrap().remove("state");
        assert_eq!(parse(v).unwrap_err(), "model and state are required");
        let mut v = ok();
        v["model"] = json!(3);
        assert_eq!(parse(v).unwrap_err(), "model and state are required");
        let mut v = ok();
        v.as_object_mut().unwrap().remove("model");
        assert_eq!(parse(v).unwrap_err(), "model and state are required");
        // `state: null` is present, hence accepted (Python: `"state" not in request`).
        let mut v = ok();
        v["state"] = Value::Null;
        assert!(parse(v).is_ok());
        assert!(parse(json!([1])).is_err());
    }

    #[test]
    fn questions_must_be_a_nonempty_object() {
        for q in [json!({}), json!([]), json!(null), json!("q")] {
            let mut v = ok();
            v["questions"] = q;
            assert_eq!(parse(v).unwrap_err(), "at least one question is required");
        }
    }

    #[test]
    fn question_types_and_criteria_are_validated() {
        let mut v = ok();
        v["questions"]["z"]["type"] = json!("rank");
        assert_eq!(
            parse(v).unwrap_err(),
            "z: type must be noul, choice, or score"
        );
        let mut v = ok();
        v["questions"]["z"].as_object_mut().unwrap().remove("type");
        assert_eq!(
            parse(v).unwrap_err(),
            "z: type must be noul, choice, or score"
        );
        for empty in [json!({}), json!(null), json!([]), json!("")] {
            let mut v = ok();
            v["questions"]["z"]["criteria"] = empty;
            assert_eq!(parse(v).unwrap_err(), "z: criteria must not be empty");
        }
        let mut v = ok();
        v["questions"]["a"]["criteria"] = json!([]);
        assert_eq!(parse(v).unwrap_err(), "a: criteria must not be empty");
        // A noul question needs no criteria; a non-empty one must be an object.
        let mut v = ok();
        v["questions"]["m"]["criteria"] = json!({"true": "yes"});
        assert!(parse(v).is_ok());
        let mut v = ok();
        v["questions"]["m"]["criteria"] = json!(["yes"]);
        assert_eq!(parse(v).unwrap_err(), "m: noul criteria must be an object");
        let mut v = ok();
        v["questions"]["z"]["criteria"] = json!(["a"]);
        assert_eq!(
            parse(v).unwrap_err(),
            "z: choice criteria must be an object"
        );
        let mut v = ok();
        v["questions"]["a"]["criteria"] = json!("high");
        assert_eq!(parse(v).unwrap_err(), "a: score criteria must be a list");
        let mut v = ok();
        v["questions"]["a"] = json!(7);
        assert_eq!(parse(v).unwrap_err(), "a: a question must be a JSON object");
    }

    #[test]
    fn media_is_detected_and_refused_with_a_clear_message() {
        for key in ["images", "audio", "videos"] {
            let mut v = ok();
            v[key] = json!(["data:image/png;base64,AAAA"]);
            let r = parse(v).unwrap();
            assert!(r.has_media, "{key}");
            let refusal = r.media_refusal().unwrap();
            assert!(refusal.0.contains("not implemented yet"), "{refusal}");
            assert!(refusal.0.contains("image/audio/video"), "{refusal}");
        }
        // Empty lists (what the Python `or []` accepts) are not media.
        let mut v = ok();
        v["images"] = json!([]);
        v["audio"] = Value::Null;
        assert!(!parse(v).unwrap().has_media);
    }
}
