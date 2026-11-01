//! `POST /v1/systemone` (the d3 decision family): one request, one typed answer
//! per question.
//!
//! The contract is the checkpoint's own `d3_server.py`:
//!
//! - the body must be a JSON object carrying only `state`, `questions`, `images`,
//!   `videos` and `model`; anything else is `422 {"detail": "<reason>"}`;
//! - `questions` must be a nonempty object of nonempty ids (422 otherwise);
//! - `images`/`videos` are phase 2: a non-empty list is 422 (data URLs only when
//!   the vision tower lands);
//! - a question that fails validation refuses the whole request (422
//!   `{"detail": {"invalid_questions": {qid: message}}}`) and nothing runs;
//! - a question over the prompt cap is **not** a refusal: it keeps its
//!   `max_length_exceeded` entry in a 200 answer, and `usage.input_tokens` counts
//!   the answered questions only;
//! - a valid request answers `{"model", "answers", "usage"}`.

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::Value;

use super::AppState;
use super::engine::DecisionReply;
use super::error::ApiError;
use super::log;
use crate::decision::DecisionRequest;

/// A `/v1/systemone` failure in the official `{"detail": ...}` shape.
pub(crate) struct DecisionError {
    status: StatusCode,
    detail: Value,
}

impl DecisionError {
    /// 422 with a plain message (`d3_server._unprocessable`).
    fn unprocessable(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: Value::String(detail.into()),
        }
    }

    /// 422 with the per-question validation messages.
    fn invalid_questions(map: serde_json::Map<String, Value>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: serde_json::json!({"invalid_questions": map}),
        }
    }

    /// A compute failure keeps its status; the body stays in the detail shape.
    fn from_api(error: ApiError) -> Self {
        Self {
            status: error.status,
            detail: Value::String(error.message),
        }
    }

    /// The response body (`{"detail": ...}`).
    fn body(&self) -> Value {
        serde_json::json!({"detail": self.detail})
    }
}

impl IntoResponse for DecisionError {
    fn into_response(self) -> Response {
        let body = self.body();
        (self.status, Json(body)).into_response()
    }
}

pub(crate) async fn systemone(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<Value>, DecisionError> {
    // Axum's own JSON rejection would be a 400 without the detail shape; parse
    // the bytes here so a malformed body is a 422 like the reference server.
    let value: Value = serde_json::from_slice(&body).map_err(|error| {
        DecisionError::unprocessable(format!("could not read the body as JSON ({error})"))
    })?;
    let request = DecisionRequest::from_value(value).map_err(DecisionError::unprocessable)?;
    let model_name = state.settings.model_name.clone();
    let (reply, timing) = state
        .compute(move |engine| engine.decisions(&model_name, request))
        .await
        .map_err(DecisionError::from_api)?;
    match reply {
        DecisionReply::Refused(invalid) => Err(DecisionError::invalid_questions(invalid)),
        DecisionReply::Answered(outcome) => {
            log::decision(&state, outcome.questions, outcome.tokens, &timing);
            Ok(Json(outcome.response))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_error_is_a_422_detail_string() {
        let error = DecisionError::unprocessable("images/videos are not supported yet");
        assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            error.body(),
            serde_json::json!({"detail": "images/videos are not supported yet"})
        );
    }

    #[test]
    fn invalid_questions_are_a_422_detail_map() {
        let mut invalid = serde_json::Map::new();
        invalid.insert(
            "q".to_string(),
            serde_json::json!("unsupported question type 'rank' (choice, noul or score)"),
        );
        let error = DecisionError::invalid_questions(invalid);
        assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            error.body(),
            serde_json::json!({"detail": {"invalid_questions": {
                "q": "unsupported question type 'rank' (choice, noul or score)"
            }}})
        );
    }
}
