//! A request batcher: one worker thread forms batches out of a queue and hands them to a
//! runner, so records that arrive together share ONE thinker pass (Clef `/v1/systemone`).
//!
//! A batch is dispatched when the first queued job has waited `window` (counted from its
//! arrival, so a job that queued behind a running batch is not delayed again), or as soon
//! as `max_jobs` jobs / `max_cost` cost units are queued. A job that would push a started
//! batch over `max_cost` starts the next batch instead (a single job over the cap runs
//! alone). With `window == 0` nothing is waited for: a batch is whatever is queued when
//! the worker looks, which keeps the latency of an isolated request unchanged.
//!
//! Shutdown drops the queue's sender: the worker still receives every queued job (running
//! them without waiting for the window), then exits and is joined.

use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchLimits {
    /// How long the first queued job waits for company.
    pub window: Duration,
    /// Most jobs per batch.
    pub max_jobs: usize,
    /// Most cost units (tokens) per batch.
    pub max_cost: usize,
}

/// A queued job with its cost and arrival time.
pub struct Queued<J> {
    pub job: J,
    pub cost: usize,
    pub arrived: Instant,
}

pub struct Batcher<J: Send + 'static> {
    tx: Mutex<Option<Sender<Queued<J>>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl<J: Send + 'static> std::fmt::Debug for Batcher<J> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batcher").finish_non_exhaustive()
    }
}

impl<J: Send + 'static> Batcher<J> {
    /// Start the worker; `run` receives each batch (never empty) in arrival order.
    pub fn start(
        limits: BatchLimits,
        mut run: impl FnMut(Vec<Queued<J>>) + Send + 'static,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<Queued<J>>();
        let worker = std::thread::Builder::new()
            .name("clef-batcher".into())
            .spawn(move || worker_loop(limits, &rx, &mut run))
            .expect("spawn the batcher");
        Self {
            tx: Mutex::new(Some(tx)),
            worker: Mutex::new(Some(worker)),
        }
    }

    /// Queue a job; hands it back when the batcher is shut down.
    pub fn submit(&self, job: J, cost: usize) -> Result<(), J> {
        let guard = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(tx) => tx
                .send(Queued {
                    job,
                    cost,
                    arrived: Instant::now(),
                })
                .map_err(|e| e.0.job),
            None => Err(job),
        }
    }

    /// Stop accepting jobs, run what is queued, join the worker. Idempotent.
    pub fn shutdown(&self) {
        drop(self.tx.lock().unwrap_or_else(|e| e.into_inner()).take());
        if let Some(h) = self.worker.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = h.join();
        }
    }
}

impl<J: Send + 'static> Drop for Batcher<J> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop<J>(
    limits: BatchLimits,
    rx: &Receiver<Queued<J>>,
    run: &mut impl FnMut(Vec<Queued<J>>),
) {
    let max_jobs = limits.max_jobs.max(1);
    // A job pulled for the batch that was already full: it opens the next one.
    let mut carry: Option<Queued<J>> = None;
    loop {
        let first = match carry.take() {
            Some(q) => q,
            None => match rx.recv() {
                Ok(q) => q,
                Err(_) => return, // sender dropped and the queue is empty
            },
        };
        let deadline = first.arrived + limits.window;
        let mut cost = first.cost;
        let mut batch = vec![first];
        while batch.len() < max_jobs && cost < limits.max_cost {
            let next = match rx.try_recv() {
                Ok(q) => q,
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    match rx.recv_timeout(deadline - now) {
                        Ok(q) => q,
                        Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
                    }
                }
            };
            if cost + next.cost > limits.max_cost {
                carry = Some(next);
                break;
            }
            cost += next.cost;
            batch.push(next);
        }
        run(batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    type Log = Arc<Mutex<Vec<Vec<u32>>>>;

    fn start(window_ms: u64, max_jobs: usize, max_cost: usize, run_ms: u64) -> (Batcher<u32>, Log) {
        let log: Log = Arc::default();
        let l = log.clone();
        let b = Batcher::start(
            BatchLimits {
                window: Duration::from_millis(window_ms),
                max_jobs,
                max_cost,
            },
            move |batch| {
                std::thread::sleep(Duration::from_millis(run_ms));
                l.lock()
                    .unwrap()
                    .push(batch.into_iter().map(|q| q.job).collect());
            },
        );
        (b, log)
    }

    #[test]
    fn a_window_collects_late_arrivals_into_one_batch() {
        let (b, log) = start(300, 8, 1000, 0);
        for i in 0..3 {
            b.submit(i, 10).unwrap();
            std::thread::sleep(Duration::from_millis(40));
        }
        b.shutdown();
        assert_eq!(*log.lock().unwrap(), [vec![0, 1, 2]]);
    }

    #[test]
    fn a_zero_window_runs_an_isolated_job_at_once() {
        let (b, log) = start(0, 8, 1000, 0);
        let t0 = Instant::now();
        b.submit(7, 10).unwrap();
        while log.lock().unwrap().is_empty() {
            assert!(t0.elapsed() < Duration::from_secs(2), "never ran");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(t0.elapsed() < Duration::from_millis(250));
        b.shutdown();
        assert_eq!(*log.lock().unwrap(), [vec![7]]);
    }

    #[test]
    fn a_zero_window_still_batches_what_queued_behind_a_running_batch() {
        let (b, log) = start(0, 8, 1000, 150);
        b.submit(0, 1).unwrap();
        std::thread::sleep(Duration::from_millis(30)); // 0 is running
        for i in 1..4 {
            b.submit(i, 1).unwrap();
        }
        b.shutdown();
        assert_eq!(*log.lock().unwrap(), [vec![0], vec![1, 2, 3]]);
    }

    #[test]
    fn the_job_count_triggers_dispatch_without_waiting() {
        let (b, log) = start(60_000, 3, 1000, 0);
        let t0 = Instant::now();
        for i in 0..7 {
            b.submit(i, 1).unwrap();
        }
        // 3 + 3 dispatch at once; the last job would wait out the window until shutdown
        while log.lock().unwrap().len() < 2 {
            assert!(t0.elapsed() < Duration::from_secs(5), "never ran");
            std::thread::sleep(Duration::from_millis(2));
        }
        b.shutdown();
        assert!(t0.elapsed() < Duration::from_secs(10));
        assert_eq!(
            *log.lock().unwrap(),
            [vec![0, 1, 2], vec![3, 4, 5], vec![6]]
        );
    }

    #[test]
    fn the_cost_cap_splits_batches_and_a_big_job_runs_alone() {
        let (b, log) = start(200, 8, 100, 0);
        for (i, c) in [(0, 60), (1, 30), (2, 30), (3, 500), (4, 10)] {
            b.submit(i, c).unwrap();
        }
        b.shutdown();
        // 60+30 fits, +30 would not -> carried; 30 alone (500 next is over) ; 500 alone; 10
        assert_eq!(
            *log.lock().unwrap(),
            [vec![0, 1], vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn shutdown_drains_the_queue_in_order_and_refuses_new_jobs() {
        let (b, log) = start(60_000, 2, 1000, 20);
        for i in 0..5 {
            b.submit(i, 1).unwrap();
        }
        b.shutdown(); // must not wait out the 60 s window
        assert_eq!(*log.lock().unwrap(), [vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(b.submit(9, 1), Err(9));
        b.shutdown(); // idempotent
    }

    #[test]
    fn dropped_receivers_do_not_stall_the_worker() {
        // The runner ignores a failed reply send, like the server's runner does.
        let (reply_tx, reply_rx) = mpsc::channel::<u32>();
        let b = Batcher::start(
            BatchLimits {
                window: Duration::ZERO,
                max_jobs: 8,
                max_cost: 100,
            },
            move |batch: Vec<Queued<(u32, tokio::sync::oneshot::Sender<u32>)>>| {
                for q in batch {
                    let (n, tx) = q.job;
                    if tx.send(n * 2).is_err() {
                        reply_tx.send(n).ok(); // reply to a client that went away
                    }
                }
            },
        );
        let (tx1, rx1) = tokio::sync::oneshot::channel();
        let (tx2, rx2) = tokio::sync::oneshot::channel();
        drop(rx1); // the first client disconnected while queued
        b.submit((1, tx1), 1).unwrap();
        b.submit((2, tx2), 1).unwrap();
        b.shutdown();
        assert_eq!(rx2.blocking_recv().unwrap(), 4);
        assert_eq!(reply_rx.try_recv().unwrap(), 1);
    }
}
