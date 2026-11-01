//! The `/v1/systemone` request envelope and the CLI's case inputs.
//!
//! ```json
//! {"state": <any JSON>, "questions": {"<id>": {"type": "...", ...}, ...},
//!  "images": [...], "videos": [...]}
//! ```
//!
//! The envelopes are parsed from a [`serde_json::Value`] rather than with a
//! `Deserialize` derive: the reference HTTP server checks the fields itself, so
//! the client-facing messages (`unknown field`, `images takes a list of base64
//! data URLs`, ...) and the accepted field set are part of the contract. A
//! question that cannot be answered is reported inside the response's `answers`
//! (see [`crate::decision::model`]); only envelope problems fail the request.
//!
//! Media is phase 2: a non-empty `images`/`videos` list is refused up front (the
//! server maps that to 422) instead of being silently ignored.

use serde_json::{Map, Value};

/// The fields a request may carry (the reference's `ACCEPTED_FIELDS`).
pub const ACCEPTED_FIELDS: [&str; 5] = ["state", "questions", "images", "videos", "model"];

/// One System One request.
#[derive(Debug, Clone)]
pub struct DecisionRequest {
    /// The state the questions are about (`null` when absent).
    pub state: Value,
    /// Question id -> question object, request order.
    pub questions: Map<String, Value>,
}

/// One case of a `--requests` array: a request plus the name reported back.
#[derive(Debug, Clone)]
pub struct CaseInput {
    pub name: Option<String>,
    pub request: DecisionRequest,
}

impl DecisionRequest {
    /// Parse one request object (`POST /v1/systemone`, `decide --request`).
    ///
    /// Unknown fields are refused, exactly like the reference server; a JSON
    /// value that is not an object is refused too.
    pub fn from_value(value: Value) -> Result<Self, String> {
        Self::parse(value, true)
    }

    fn parse(value: Value, strict: bool) -> Result<Self, String> {
        let Value::Object(mut object) = value else {
            return Err(
                "expected a JSON object such as {\"questions\": {...}, \"state\": ...}".to_string(),
            );
        };
        if strict {
            let mut unknown: Vec<String> = object
                .keys()
                .filter(|key| !ACCEPTED_FIELDS.contains(&key.as_str()))
                .cloned()
                .collect();
            if !unknown.is_empty() {
                unknown.sort();
                let named = unknown
                    .iter()
                    .map(|key| format!("'{key}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "{named} not understood; a request may carry {}",
                    ACCEPTED_FIELDS.join(", ")
                ));
            }
        }
        let state = object.remove("state").unwrap_or(Value::Null);
        let questions = object.remove("questions").unwrap_or(Value::Null);
        let images = object.remove("images").unwrap_or(Value::Null);
        let videos = object.remove("videos").unwrap_or(Value::Null);
        // Accepted and ignored: the answer always names the served model.
        let _model = object.remove("model");

        let Value::Object(questions) = questions else {
            return Err(
                "questions is required: an object of one or more question IDs, each to a question"
                    .to_string(),
            );
        };
        if questions.is_empty() {
            return Err(
                "questions is required: an object of one or more question IDs, each to a question"
                    .to_string(),
            );
        }
        if questions.keys().any(|key| key.is_empty()) {
            return Err("question IDs must be nonempty strings".to_string());
        }
        check_media("images", &images)?;
        check_media("videos", &videos)?;
        Ok(Self { state, questions })
    }
}

/// `images`/`videos` must be absent, `null` or an empty list; anything else is
/// either the phase-2 media path or a type error.
fn check_media(name: &str, value: &Value) -> Result<(), String> {
    match value {
        Value::Null => Ok(()),
        Value::Array(items) if items.is_empty() => Ok(()),
        Value::Array(_) => Err("images/videos are not supported yet".to_string()),
        _ => Err(format!("{name} takes a list of base64 data URLs")),
    }
}

impl CaseInput {
    /// Parse one `--requests` array entry: `name` is read (and stripped) here,
    /// the rest is the request; extra keys are ignored.
    pub fn from_value(value: Value) -> Result<Self, String> {
        let Value::Object(mut object) = value else {
            return Err("a case must be a JSON object".to_string());
        };
        let name = match object.remove("name") {
            None | Some(Value::Null) => None,
            Some(Value::String(name)) => Some(name),
            Some(_) => return Err("a case name must be a string".to_string()),
        };
        let request = DecisionRequest::parse(Value::Object(object), false)?;
        Ok(Self { name, request })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: serde_json::Value) -> Result<DecisionRequest, String> {
        DecisionRequest::from_value(value)
    }

    #[test]
    fn a_request_keeps_question_order_and_defaults_the_state() {
        let req = parse(json!({
            "questions": {"z": {"type": "noul"}, "a": {"type": "noul"}},
        }))
        .unwrap();
        assert!(req.state.is_null());
        let ids: Vec<&String> = req.questions.keys().collect();
        assert_eq!(ids, ["z", "a"], "serde_json keeps insertion order");
    }

    #[test]
    fn unknown_fields_are_refused_like_the_reference_server() {
        let err =
            parse(json!({"questions": {"q": {"type": "noul"}}, "temperature": 1})).unwrap_err();
        assert_eq!(
            err,
            "'temperature' not understood; a request may carry state, questions, images, videos, model"
        );
        let err = parse(json!({"questions": {"q": {}}, "b": 1, "a": 2})).unwrap_err();
        assert_eq!(
            err,
            "'a', 'b' not understood; a request may carry state, questions, images, videos, model"
        );
        // `model` is accepted and ignored.
        let req = parse(json!({"questions": {"q": {"type": "noul"}}, "model": "ignored"})).unwrap();
        assert_eq!(req.questions.len(), 1);
    }

    #[test]
    fn media_is_refused_in_phase_1() {
        let err =
            parse(json!({"questions": {"q": {"type": "noul"}}, "images": ["x"]})).unwrap_err();
        assert_eq!(err, "images/videos are not supported yet");
        let err =
            parse(json!({"questions": {"q": {"type": "noul"}}, "videos": ["x"]})).unwrap_err();
        assert_eq!(err, "images/videos are not supported yet");
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "images": 5})).unwrap_err();
        assert_eq!(err, "images takes a list of base64 data URLs");
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "videos": {}})).unwrap_err();
        assert_eq!(err, "videos takes a list of base64 data URLs");
        // An empty media list is the same as none.
        assert!(
            parse(json!({"questions": {"q": {"type": "noul"}}, "images": [], "videos": []}))
                .is_ok()
        );
    }

    #[test]
    fn bad_envelopes_are_refused() {
        const MISSING: &str =
            "questions is required: an object of one or more question IDs, each to a question";
        assert_eq!(
            parse(json!([])).unwrap_err(),
            "expected a JSON object such as {\"questions\": {...}, \"state\": ...}"
        );
        for bad in [
            json!({}),
            json!({"questions": {}}),
            json!({"questions": []}),
            json!({"questions": "q"}),
            json!({"questions": null}),
        ] {
            assert_eq!(
                parse(bad.clone()).unwrap_err(),
                MISSING,
                "message for {bad}"
            );
        }
        assert_eq!(
            parse(json!({"questions": {"": {"type": "noul"}}})).unwrap_err(),
            "question IDs must be nonempty strings"
        );
    }

    #[test]
    fn a_case_carries_its_name_and_request() {
        let case = CaseInput::from_value(json!({"name": "one", "state": "s", "questions": {}}));
        assert!(case.is_err(), "an empty questions object is refused");
        let case =
            CaseInput::from_value(json!({"name": "one", "state": "s", "questions": {"q": {}}}))
                .unwrap();
        assert_eq!(case.name.as_deref(), Some("one"));
        assert_eq!(case.request.state, json!("s"));
        // Extra case-level keys are ignored (the CLI's own format).
        let case = CaseInput::from_value(
            json!({"name": null, "state": 1, "questions": {"q": {}}, "elapsed_s": 1.0}),
        )
        .unwrap();
        assert!(case.name.is_none());
        assert!(CaseInput::from_value(json!(["not", "an", "object"])).is_err());
    }
}
