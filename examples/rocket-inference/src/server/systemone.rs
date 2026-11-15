//! `POST /v1/systemone` (the d3 decision family): one request, one typed answer
//! per question.
//!
//! The contract is the checkpoint's own `d3_server.py`:
//!
//! - the body must be a JSON object carrying only `state`, `questions`, `images`,
//!   `videos` and `model`; anything else is `422 {"detail": "<reason>"}`;
//! - `questions` must be a nonempty object of nonempty ids (422 otherwise);
//! - `images` is a list of `data:image/{png,jpeg,webp};base64,...` URLs shared by
//!   every question (`d3_runtime.MAX_IMAGE_BYTES` 8,000,000 decoded bytes,
//!   `MAX_IMAGE_SOURCE_PIXELS` 16,000,000 pixels, PNG/JPEG/WebP); a local path, a
//!   remote URL, another subtype or an undecodable payload is a 422 whose detail
//!   is the decoder's message (`images[i]: ...`);
//! - `videos` is a list of `data:video/{mp4,webm,quicktime,x-matroska};base64,...`
//!   URLs with the reference's strict limits (32,000,000 decoded bytes, 8,294,400
//!   pixels per frame, 300 seconds), decoded at 2 frames per second; every refusal
//!   is `videos[i]: ...`, and all videos of one request together may not exceed
//!   16,384 input tokens;
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
use super::clef_batch::ClefJob;
use super::engine::{DecisionReply, clef_api_error};
use super::error::{ApiError, blocking};
use super::log;
use crate::clef::ClefRequest;
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

/// `POST /v1/systemone` for Clef-Omni: the release's `systemone()` contract. The body is
/// validated like the Python entry point (`model` and `state` required, a nonempty
/// `questions` object, types and criteria); a refusal (including a request carrying
/// audio (not implemented yet), videos whose soundtracks the model would use (audio-in-video,
/// not implemented yet), an image or video the server cannot read or decode, and a record, media
/// included, over the prompt cap) is `422 {"detail": "<reason>"}` immediately and runs nothing. A valid
/// request is encoded, queued, answered together with whatever shares its batch
/// (`--batch-window-ms`, `--batch-max`, `--batch-max-tokens`) and returns
/// `{"model", "answers", "usage"}`; the answer does not depend on its batch mates.
pub(crate) async fn systemone_clef(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<Value>, DecisionError> {
    let value: Value = serde_json::from_slice(&body).map_err(|error| {
        DecisionError::unprocessable(format!("could not read the body as JSON ({error})"))
    })?;
    let request = ClefRequest::from_value(value).map_err(|e| DecisionError::unprocessable(e.0))?;
    if let Some(refusal) = request.media_refusal() {
        return Err(DecisionError::unprocessable(refusal.0));
    }
    let front = state
        .clef
        .clone()
        .ok_or_else(|| DecisionError::from_api(ApiError::internal("no Clef model is loaded")))?;
    // Encode before queueing: a record over the cap is a 422 now, and the batcher knows
    // each job's exact row count.
    if (!request.images.is_empty() || !request.videos.is_empty()) && !front.images {
        return Err(DecisionError::unprocessable(
            "this deployment serves a clef-pack, which has no vision tower: image and video \
             input is unavailable (serve the BF16 checkpoint or the AutoRound shards)"
                .to_string(),
        ));
    }
    let encoder = front.clone();
    // Image fetch/decode/resize happen here, on the blocking pool (the tower itself runs in
    // the batch worker under the model lock).
    let (request, enc) = blocking(move || {
        match crate::clef::encode_request(&encoder.tokenizer, &request, encoder.max_tokens) {
            Ok(enc) => Ok((request, enc)),
            Err(error) => Err(clef_api_error(&error)),
        }
    })
    .await
    .map_err(DecisionError::from_api)?;
    let rows = enc.input_ids.len();
    let (reply, answer) = tokio::sync::oneshot::channel();
    front
        .batcher
        .submit(
            ClefJob {
                req: request,
                enc,
                reply,
            },
            rows,
        )
        .map_err(|_| {
            DecisionError::from_api(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "the server is shutting down",
            ))
        })?;
    let done = answer
        .await
        .map_err(|_| DecisionError::from_api(ApiError::internal("the batch worker stopped")))?
        .map_err(DecisionError::from_api)?;
    log::clef_decision(
        &state,
        done.outcome.questions,
        done.outcome.tokens,
        done.queue_s,
        done.compute_s,
        done.batch_records,
        done.batch_tokens,
    );
    Ok(Json(done.outcome.response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_error_is_a_422_detail_string() {
        let error = DecisionError::unprocessable(
            "images[0]: strict mode: images must be given as a data: URL string, not a file \
             path, web address or PIL image",
        );
        assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            error.body(),
            serde_json::json!({"detail": "images[0]: strict mode: images must be given as a \
                data: URL string, not a file path, web address or PIL image"})
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
