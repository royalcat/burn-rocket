//! Native multimodal `/embed` (EmbeddingGemma 2): text and/or one media input,
//! L2-normalized with optional MRL truncation.

use std::path::PathBuf;

use axum::{Json, extract::State};
use serde::Deserialize;

use super::AppState;
use super::engine::EmbedMediaRequest;
use super::error::{ApiError, OpenAiError};
use super::log;
use crate::gemma::media::resolve_media;

#[derive(Deserialize, Default)]
pub(crate) struct NativeRequest {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    video: Option<String>,
    #[serde(default)]
    audio: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    dim: Option<usize>,
}

pub(crate) async fn embed(
    State(state): State<AppState>,
    Json(req): Json<NativeRequest>,
) -> Result<Json<serde_json::Value>, OpenAiError> {
    let model_name = state.settings.model_name.clone();
    let resolve = |value: Option<String>, ext: &str| -> Result<Option<PathBuf>, ApiError> {
        value
            .as_deref()
            .map(|v| resolve_media(v, ext))
            .transpose()
            .map_err(ApiError::bad_request)
    };
    let request = EmbedMediaRequest {
        text: req.text,
        image: resolve(req.image, "img")?,
        video: resolve(req.video, "mp4")?,
        audio: resolve(req.audio, "wav")?,
        prompt: req.prompt,
        dim: req.dim,
    };
    let (result, timing) = state
        .compute(move |engine| engine.embed_media(request))
        .await?;
    log::embedding(&state, "embed", 1, result.tokens, &timing);
    let vector = result.vectors.into_iter().next().unwrap_or_default();
    Ok(Json(serde_json::json!({
        "model": model_name,
        "dim": vector.len(),
        "embedding": vector,
    })))
}
