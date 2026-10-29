//! One HTTP server for the loaded model.
//!
//! The route set follows the model's [`Capabilities`]: `/health` and
//! `/v1/models` always; `/v1/embeddings` (plus `/embed` for multimodal models)
//! for embedding models; `/v1/chat/completions` and the Ollama API (`/`,
//! `/api/*`) for chat models. Incompatible paths are not registered (404).
//!
//! The model sits behind one mutex (a forward is CPU-bound and single-flight);
//! panics are contained so the server survives them, and every inference
//! request is timed and logged through [`log`].

pub mod cli;
pub mod engine;
pub mod error;
mod log;
mod multimodal;
mod ollama;
mod openai;
pub(crate) mod telemetry;

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use crate::util::http::lock_or_recover;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    middleware,
    routing::{get, post},
};
use engine::{Capabilities, Engine, Family};
use error::{ApiError, blocking};
use log::RequestTiming;

/// Immutable settings, kept outside the model lock so `/health`, `/v1/models`
/// and request routing never block behind a running forward.
#[derive(Debug)]
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

#[derive(Clone, Debug)]
pub(crate) struct AppState {
    pub(crate) inner: Arc<Mutex<Engine>>,
    pub(crate) settings: Arc<Settings>,
}

impl AppState {
    fn lock(&self) -> MutexGuard<'_, Engine> {
        lock_or_recover(&self.inner)
    }

    /// Run one inference call: wait for the model lock, time the wait and the
    /// call itself, and contain panics (through [`blocking`]). Every inference
    /// endpoint goes through this so the request logs carry consistent timings.
    ///
    /// The `inference` span and its children (`prepare`, `towers.load`,
    /// `forward`/`prefill`/`decode`, ...) are only created while trace export
    /// is on: the stdout layer keeps events only, so the disabled path costs a
    /// metadata check and leaves the log output unchanged.
    #[telemetry::instrument(
        name = "inference",
        skip_all,
        fields(
            model = %self.settings.model_name,
            queue_s = tracing::field::Empty,
            compute_s = tracing::field::Empty,
            encode_s = tracing::field::Empty,
        )
    )]
    pub(crate) async fn compute<T, F>(&self, f: F) -> Result<(T, RequestTiming), ApiError>
    where
        T: Send + 'static,
        F: FnOnce(&mut Engine) -> Result<T, ApiError> + Send + 'static,
    {
        let state = self.clone();
        let arrived = Instant::now();
        // `blocking` carries the current span (this one) onto the worker thread.
        blocking(move || {
            let mut engine = state.lock();
            let queue_s = arrived.elapsed().as_secs_f64();
            let span = tracing::Span::current();
            span.record("queue_s", queue_s);

            let t0 = Instant::now();
            crate::gemma::vision::vision_stats_reset();
            let out = f(&mut engine)?;
            let compute_s = t0.elapsed().as_secs_f64();
            let encode_s = crate::gemma::vision::vision_stats().total_s;
            span.record("compute_s", compute_s);
            if encode_s > 0.0 {
                span.record("encode_s", encode_s);
            }
            let timing = RequestTiming {
                queue_s,
                compute_s,
                encode_s,
            };
            // Long requests leave their transient buffers in the allocator's
            // arenas; hand them back before the next request (`ROCKET_TRIM=0`
            // disables it). The driver trim keeps the just-used shape set and
            // drops per-shape scratch from earlier requests.
            crate::util::trim_after_request();
            crate::util::log_npu_mem();
            Ok((out, timing))
        })
        .await
    }
}

/// Serve one loaded model on `addr` until the process is stopped.
pub fn serve(addr: &str, engine: Engine, settings: Settings) -> Result<()> {
    // Installs the log subscriber; with `OTEL_SDK_ENABLED=true` it also builds
    // the OTLP exporter and installs the trace layer. The guard flushes the
    // queued spans on the way out.
    let telemetry = telemetry::Telemetry::init();
    // Release what loading and tokenizing left in the allocator's arenas before
    // the first request (the loaders trim their own transients; the tokenizer
    // parse and the engine construction happen after those trims).
    crate::util::trim_heap();
    // Mark the load-time matmul scratch as older than the first request: the
    // per-request trim then keeps only the request's own shape set.
    crate::util::trim_npu_scratch();
    crate::util::log_npu_mem();
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
    // Base64 images (data URIs) and multi-image batches exceed axum's 2 MiB
    // default JSON body limit.
    let app = app.layer(DefaultBodyLimit::max(32 * 1024 * 1024));
    // Outermost layer (added last), so every route and fallback is covered.
    let app = if telemetry::enabled() {
        app.layer(middleware::from_fn_with_state(
            state.clone(),
            telemetry::trace_http,
        ))
    } else {
        app
    };
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
        tracing::info!(addr = %addr, model = %model_name, "listening");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        Ok::<(), anyhow::Error>(())
    })?;
    // In-flight requests are drained above; export what is still queued.
    drop(telemetry);
    Ok(())
}

/// Resolve when the process is asked to stop. SIGTERM is what docker stop and
/// systemd send; Ctrl+C covers interactive runs.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "failed to listen for Ctrl+C");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::error!(%error, "failed to listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutdown signal received; draining in-flight requests");
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
