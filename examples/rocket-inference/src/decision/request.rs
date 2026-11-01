//! The `/v1/systemone` request envelope and the CLI's case inputs.
//!
//! ```json
//! {"state": <any JSON>, "questions": {"<id>": {"type": "...", ...}, ...},
//!  "images": [...], "videos": [...]}
//! ```
//!
//! `state` is optional (and may be any JSON); `questions` must be a nonempty
//! object of nonempty ids. A question that cannot be answered is reported inside
//! the response's `answers` (see [`crate::decision::model`]); only envelope
//! problems fail the whole request.
//!
//! Media is phase 2: a non-empty `images`/`videos` list is refused up front (the
//! server maps that to 400) instead of being silently ignored.

use anyhow::{Result, bail};
use serde_json::{Map, Value};

/// One System One request.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DecisionRequest {
    /// The state the questions are about (`null` when absent).
    #[serde(default)]
    pub state: Value,
    /// Question id -> question object. Typed as a `Value` so a malformed value
    /// gets our own message instead of a serde error.
    #[serde(default)]
    pub questions: Value,
    /// Refused in phase 1 (text-only).
    #[serde(default)]
    pub images: Option<Vec<Value>>,
    /// Refused in phase 1 (text-only).
    #[serde(default)]
    pub videos: Option<Vec<Value>>,
    /// Accepted and ignored: the response always names the served model.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<Value>,
}

/// One case of a `--requests` array: a request plus the name reported back.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CaseInput {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(flatten)]
    pub request: DecisionRequest,
}

impl DecisionRequest {
    /// Validate the request envelope and return the questions in request order.
    pub fn questions(&self) -> Result<&Map<String, Value>> {
        if self.images.as_ref().is_some_and(|v| !v.is_empty())
            || self.videos.as_ref().is_some_and(|v| !v.is_empty())
        {
            bail!("images/videos are not supported yet");
        }
        let Some(questions) = self.questions.as_object() else {
            bail!("questions must be a nonempty mapping of question IDs to questions");
        };
        if questions.is_empty() {
            bail!("questions must be a nonempty mapping of question IDs to questions");
        }
        if questions.keys().any(|key| key.is_empty()) {
            bail!("question IDs must be nonempty strings");
        }
        Ok(questions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: serde_json::Value) -> DecisionRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn a_request_keeps_question_order_and_defaults_the_state() {
        let req = parse(json!({
            "questions": {"z": {"type": "noul"}, "a": {"type": "noul"}},
        }));
        assert!(req.state.is_null());
        let ids: Vec<&String> = req.questions().unwrap().keys().collect();
        assert_eq!(ids, ["z", "a"], "serde_json keeps insertion order");
    }

    #[test]
    fn media_and_bad_envelopes_are_refused() {
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "images": ["x"]}))
            .questions()
            .unwrap_err();
        assert!(err.to_string().contains("not supported yet"), "{err}");
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "videos": ["x"]}))
            .questions()
            .unwrap_err();
        assert!(err.to_string().contains("not supported yet"), "{err}");
        // An empty media list is the same as none.
        assert!(
            parse(json!({"questions": {"q": {"type": "noul"}}, "images": []}))
                .questions()
                .is_ok()
        );
        for bad in [
            json!({"questions": {}}),
            json!({"questions": []}),
            json!({"questions": "q"}),
            json!({}),
            json!({"questions": {"": {"type": "noul"}}}),
        ] {
            assert!(parse(bad).questions().is_err());
        }
    }

    #[test]
    fn a_case_carries_its_name_and_request() {
        let case: CaseInput =
            serde_json::from_value(json!({"name": "one", "state": "s", "questions": {}})).unwrap();
        assert_eq!(case.name.as_deref(), Some("one"));
        assert_eq!(case.request.state, json!("s"));
        // The name is not part of the request itself.
        assert!(case.request.questions().is_err());
    }
}
