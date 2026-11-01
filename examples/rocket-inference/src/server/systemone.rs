//! `POST /v1/systemone` (the d3 decision family): one request, one typed answer
//! per question.
//!
//! Envelope problems (malformed `questions`, media) are the request's error (400).
//! A question that cannot be answered is reported inside `answers` with its own
//! `error` field and the rest of the request still runs, so those stay 200.

use axum::{Json, extract::State};

use super::AppState;
use super::error::OpenAiError;
use super::log;
use crate::decision::DecisionRequest;

pub(crate) async fn systemone(
    State(state): State<AppState>,
    Json(req): Json<DecisionRequest>,
) -> Result<Json<serde_json::Value>, OpenAiError> {
    let model_name = state.settings.model_name.clone();
    let (outcome, timing) = state
        .compute(move |engine| engine.decisions(&model_name, req))
        .await?;
    log::decision(&state, outcome.questions, outcome.tokens, &timing);
    Ok(Json(outcome.response))
}
