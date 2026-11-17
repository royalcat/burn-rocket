//! Native multimodal `/embed` (EmbeddingGemma 2): text and/or one media input,
//! L2-normalized with optional MRL truncation.

use axum::{Json, extract::State};
use serde::Deserialize;

use super::AppState;
use super::engine::EmbedMediaRequest;
use super::error::{ApiError, OpenAiError};
use super::log;
use crate::gemma::media::{ResolvedMedia, resolve_media};

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
    let resolve = |value: Option<String>, ext: &str| -> Result<Option<ResolvedMedia>, ApiError> {
        value
            .as_deref()
            .map(|v| resolve_media(v, ext))
            .transpose()
            .map_err(ApiError::bad_request)
    };
    // The guards keep base64 uploads alive across the compute await and delete
    // them afterwards.
    let image = resolve(req.image, "img")?;
    let video = resolve(req.video, "mp4")?;
    let audio = resolve(req.audio, "wav")?;
    let request = EmbedMediaRequest {
        text: req.text,
        image: image.as_ref().map(|m| m.path().to_owned()),
        video: video.as_ref().map(|m| m.path().to_owned()),
        audio: audio.as_ref().map(|m| m.path().to_owned()),
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
