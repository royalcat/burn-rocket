//! Clef `/v1/systemone` batching: validated, encoded records queue up and one worker
//! thread answers them in shared thinker passes (one expert stream per layer for the
//! whole batch). See [`super::batcher`] for when a batch is dispatched.
//!
//! The model lock discipline is unchanged: the worker takes the engine mutex for the
//! duration of a batch, so `/health` and `/v1/models` (which never touch it) stay
//! responsive, and a panic in the shared forward is contained and mapped like any
//! other compute failure (every member of the batch gets the error).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokenizers::Tokenizer;
use tokio::sync::oneshot;

use super::batcher::{BatchLimits, Batcher, Queued};
use super::engine::{DecisionOutcome, Engine};
use super::error::{ApiError, catch_compute};
use super::telemetry;
use crate::clef::{ClefRequest, EncodedRecord};
use crate::util::http::lock_or_recover;

/// `--batch-window-ms`, `--batch-max`, `--batch-max-tokens` (Clef serve only).
#[derive(Debug, Clone, Copy)]
pub struct BatchSettings {
    pub window_ms: u64,
    pub max_records: usize,
    pub max_tokens: usize,
}

impl Default for BatchSettings {
    fn default() -> Self {
        Self {
            window_ms: 0,
            max_records: 8,
            max_tokens: 4096,
        }
    }
}

/// What a finished request reports.
pub(crate) struct ClefDone {
    pub outcome: DecisionOutcome,
    /// Arrival -> the batch got the model lock.
    pub queue_s: f64,
    /// The shared batch compute (the same for every member).
    pub compute_s: f64,
    pub batch_records: usize,
    pub batch_tokens: usize,
}

pub(crate) struct ClefJob {
    pub req: ClefRequest,
    pub enc: EncodedRecord,
    pub reply: oneshot::Sender<Result<ClefDone, ApiError>>,
}

/// The handler-side half: the tokenizer the handlers encode with, and the queue.
pub(crate) struct ClefFront {
    pub tokenizer: Tokenizer,
    /// The checkpoint holds the vision tower (a clef-pack does not): `images` are served.
    pub images: bool,
    /// Per-record prompt cap (`--max-tokens`).
    pub max_tokens: usize,
    pub batcher: Batcher<ClefJob>,
}

impl std::fmt::Debug for ClefFront {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClefFront")
            .field("max_tokens", &self.max_tokens)
            .finish_non_exhaustive()
    }
}

impl ClefFront {
    pub fn start(
        engine: Arc<Mutex<Engine>>,
        tokenizer: Tokenizer,
        images: bool,
        max_tokens: usize,
        settings: BatchSettings,
    ) -> Self {
        let limits = BatchLimits {
            window: std::time::Duration::from_millis(settings.window_ms),
            max_jobs: settings.max_records.max(1),
            max_cost: settings.max_tokens.max(1),
        };
        // the engine's row cap per shared pass equals the batch's token cap
        let cap = settings.max_tokens;
        let batcher = Batcher::start(limits, move |batch| run_batch(&engine, cap, batch));
        Self {
            tokenizer,
            images,
            max_tokens,
            batcher,
        }
    }
}

/// Answer one batch: drop jobs whose client already left, run the shared pass under the
/// model lock, reply to every member.
fn run_batch(engine: &Mutex<Engine>, max_batch_tokens: usize, batch: Vec<Queued<ClefJob>>) {
    let total = batch.len();
    let live: Vec<Queued<ClefJob>> = batch
        .into_iter()
        .filter(|q| !q.job.reply.is_closed())
        .collect();
    if live.len() < total {
        tracing::debug!(
            dropped = total - live.len(),
            "skipped requests whose client left"
        );
    }
    if live.is_empty() {
        return;
    }
    let rows: usize = live.iter().map(|q| q.cost).sum();
    run_locked(engine, max_batch_tokens, live.len(), rows, live);
}

#[telemetry::instrument(
    name = "inference",
    skip_all,
    fields(
        batch = records,
        batch_tokens = rows,
        queue_s = tracing::field::Empty,
        compute_s = tracing::field::Empty,
    )
)]
fn run_locked(
    engine: &Mutex<Engine>,
    max_batch_tokens: usize,
    records: usize,
    rows: usize,
    live: Vec<Queued<ClefJob>>,
) {
    let arrived: Vec<Instant> = live.iter().map(|q| q.arrived).collect();
    let (reqs, encs, replies): (Vec<_>, Vec<_>, Vec<_>) = live.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut a, mut b, mut c), q| {
            a.push(q.job.req);
            b.push(q.job.enc);
            c.push(q.job.reply);
            (a, b, c)
        },
    );
    let mut guard = lock_or_recover(engine);
    let locked = Instant::now();
    let queue: Vec<f64> = arrived
        .iter()
        .map(|a| (locked - *a).as_secs_f64())
        .collect();
    let span = tracing::Span::current();
    span.record("queue_s", queue.iter().copied().fold(0.0, f64::max));
    let refs: Vec<&ClefRequest> = reqs.iter().collect();
    let outcome = catch_compute(|| guard.clef_batch(&refs, encs, max_batch_tokens));
    let compute_s = locked.elapsed().as_secs_f64();
    span.record("compute_s", compute_s);
    // Long requests leave their transient buffers in the allocator's arenas, and the
    // NPU scratch policy applies per request (`ROCKET_KEEP_MB`): once per batch.
    crate::util::trim_after_request();
    crate::util::log_npu_mem();
    drop(guard);

    let clone_err = |e: &ApiError| ApiError::new(e.status, e.message.clone());
    match outcome {
        Ok(done) => {
            tracing::info!(
                batch = records,
                batch_tokens = done.tokens,
                passes = done.passes,
                experts_read = done.experts_read,
                compute_s = (compute_s * 1000.0).round() / 1000.0,
                "clef batch"
            );
            for ((result, reply), queue_s) in done.results.into_iter().zip(replies).zip(queue) {
                let msg = result.map(|outcome| ClefDone {
                    outcome,
                    queue_s,
                    compute_s,
                    batch_records: records,
                    batch_tokens: done.tokens,
                });
                // a client that disconnected meanwhile is simply not answered
                let _ = reply.send(msg);
            }
        }
        Err(e) => {
            tracing::error!(batch = records, status = %e.status, error = %e.message, "clef batch failed");
            for reply in replies {
                let _ = reply.send(Err(clone_err(&e)));
            }
        }
    }
}
