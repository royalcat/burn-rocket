//! One HTTP server for the loaded model.
//!
//! The route set follows the model's [`Capabilities`]: `/health` and
//! `/v1/models` always; `/v1/embeddings` (plus `/embed` for multimodal models)
//! for embedding models; `/v1/chat/completions` and the Ollama API (`/`,
//! `/api/*`) for chat models. Incompatible paths are not registered (404).
//!
//! The model sits behind one mutex (a forward is CPU-bound and single-flight);
//! panics are contained so the server survives them.

pub mod cli;
pub mod engine;
pub mod error;
mod multimodal;
mod ollama;
mod openai;

use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};

use crate::util::http::lock_or_recover;
use engine::{Capabilities, Engine, Family};

/// Immutable settings, kept outside the model lock so `/health`, `/v1/models`
/// and request routing never block behind a running forward.
pub struct Settings {
    pub model_name: String,
    pub family: Family,
    pub capabilities: Capabilities,
    /// Prompt-token cap (embedding truncation, chat context default).
    pub max_tokens: usize,
    /// Default cap on generated tokens for chat requests.
    pub max_new_tokens: usize,
    /// Default sampling temperature (0 = greedy).
    pub temperature: f32,
}

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) inner: Arc<Mutex<Engine>>,
    pub(crate) settings: Arc<Settings>,
}

impl AppState {
    pub(crate) fn lock(&self) -> MutexGuard<'_, Engine> {
        lock_or_recover(&self.inner)
    }
}

/// Serve one loaded model on `addr` until the process is stopped.
pub fn serve(addr: &str, engine: Engine, settings: Settings) -> Result<()> {
    let capabilities = settings.capabilities;
    let model_name = settings.model_name.clone();
    let state = AppState {
        inner: Arc::new(Mutex::new(engine)),
        settings: Arc::new(settings),
    };

    let mut app: Router<AppState> = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models));
    if capabilities.embeddings {
        app = app.route("/v1/embeddings", post(openai::embeddings));
        if capabilities.multimodal {
            app = app.route("/embed", post(multimodal::embed));
        }
    }
    if capabilities.chat {
        app = app.route("/v1/chat/completions", post(openai::chat_completions));
        app = app.merge(ollama::router());
    }
    let app = app.with_state(state);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("tokio runtime")?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("bind {addr}"))?;
        println!("listening on http://{addr} (model '{model_name}')");
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.settings.model_name,
            "object": "model",
            "owned_by": "local",
            "capabilities": state.settings.capabilities.names(),
        }],
    }))
}
