//! Phase 6: the layer engine behind the NPU / emulated Clef placements.
//!
//! A [`LayerEngine`] computes the heavy parts of one thinker layer: the attention
//! projections (`qkv`, `o`) and the MoE. The backbone keeps the small host work (norms,
//! RoPE, the router, causal attention scores/softmax/AV, residuals).
//!
//! The MoE streams its experts from the pack ([`ExpertReader`]):
//!
//! * a **reader thread** takes `(layer, experts)` commands and reads the experts one by
//!   one into a small pool of aligned buffers (`depth` ahead, O_DIRECT in the pack
//!   reader), handing `(layer, expert, buffer)` items to the consumers through a bounded
//!   queue;
//! * one **consumer thread per [`ExpertWorker`]** (three NPU contexts with one worker
//!   each, or N host threads for the emulation) takes the next arrived expert of the
//!   current layer, gathers that expert's token rows, runs gate|up, SiLU*mul and down,
//!   scales the rows by the router weights and files the result under the expert id;
//! * the layer ends with a barrier (the scope join); the per-expert results are then
//!   combined **per token in expert-id order**, which makes the layer output independent
//!   of the order in which the consumers finished (bit-identical for any schedule).
//!
//! When `S * top_k >= SPEC_MIN_SLOTS` essentially every expert is hit, so the reader is
//! also told to read the *whole next layer* as soon as the current layer's commands are
//! drained (and the layer itself at its start, before attention): reads overlap attention
//! and the combine. Correctness never depends on it: consumers discard items that are not
//! in the layer's hit set and items of earlier layers.

use super::attn::causal_gqa;
use super::int8::{emu_matmul_gw, f16_scales_to_f32, quantize_rows};
use super::math::matmul_nt;
use super::model::{LayerOut, LayerSmall, Rope, TextConfig, rms_norm, rms_norm_rows, route, silu};
use super::pack::format::{Dims, ExpertRecord};
use super::pack::reader::{AlignedBuf, PackDir};
use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// `S * top_k` at or above which the reader is told to read a whole layer speculatively.
pub const SPEC_MIN_SLOTS: usize = 512;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------------------
// phases

/// Which host phase a timing belongs to (for [`LayerEngine::note`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Attn,
    Router,
}

/// Aggregated phase times of one request. The `*_thread` fields sum over all consumer
/// threads (so they can exceed the wall time of the MoE); the others are wall seconds.
#[derive(Debug, Clone, Default)]
pub struct Phases {
    pub read_wait_thread: f64,
    pub rewrite_thread: f64,
    pub quant_thread: f64,
    pub npu_thread: f64,
    pub glue_thread: f64,
    pub read_io_thread: f64,
    pub combine: f64,
    pub moe: f64,
    pub qkv: f64,
    pub oproj: f64,
    pub attn: f64,
    pub router: f64,
    pub experts: usize,
    pub rows: usize,
    pub layers: usize,
    pub peak_anon_mib: f64,
    pub layer_moe: Vec<f64>,
}

/// Per-expert times a worker reports (seconds).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertTimes {
    pub rewrite: f64,
    pub quant: f64,
    pub npu: f64,
    pub glue: f64,
}

impl ExpertTimes {
    fn add(&mut self, o: &ExpertTimes) {
        self.rewrite += o.rewrite;
        self.quant += o.quant;
        self.npu += o.npu;
        self.glue += o.glue;
    }
}

pub fn phase_log_level() -> u8 {
    match std::env::var("ROCKET_PHASE_LOG").ok().as_deref() {
        None | Some("") | Some("0") => 0,
        Some("2") => 2,
        Some(_) => 1,
    }
}

// ---------------------------------------------------------------------------------------
// traits

/// The heavy parts of a thinker layer. `Sync` because the backbone shares it with its
/// rayon pool; every method is called from the request thread in layer order.
pub trait LayerEngine: Send + Sync {
    fn label(&self) -> String;
    /// A layer is about to start (the engine may start reading its experts).
    fn hint_layer(&self, _layer: usize, _s: usize) {}
    /// `q [s, heads*d]`, `k`/`v [s, kv*d]` for the normalized input `n [s, hidden]`.
    fn qkv(&self, layer: usize, n: &[f32], s: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)>;
    /// `[s, hidden]` output projection of the attention context `[s, heads*d]`.
    fn o_proj(&self, layer: usize, ctx: &[f32], s: usize) -> Result<Vec<f32>>;
    /// `y += MoE(n2)` for the router output `ids`/`wts` (`[s * top_k]`).
    fn moe_add(
        &self,
        layer: usize,
        n2: &[f32],
        s: usize,
        ids: &[u32],
        wts: &[f32],
        y: &mut [f32],
    ) -> Result<()>;
    fn note(&self, _phase: Phase, _secs: f64) {}
    /// The forward finished: log, trim scratch, drop the reader.
    fn finish_request(&self, _tokens: usize) {}
    /// The forward failed: tear everything per-request down.
    fn abort_request(&self) {}
    /// The phase totals of the last finished request (for telemetry).
    fn last_phases(&self) -> Option<Phases> {
        None
    }
}

/// Where experts come from: aligned reads of one record per expert.
pub trait ExpertReader: Send + Sync {
    fn dims(&self) -> Dims;
    fn layers(&self) -> usize;
    fn experts(&self) -> usize;
    fn new_buffer(&self) -> AlignedBuf;
    fn read(&self, layer: usize, expert: usize, buf: &mut AlignedBuf) -> Result<()>;
    /// `layer`'s experts were consumed: drop what the page cache holds of them.
    fn release_layer(&self, _layer: usize) {}
    /// Drop mapped resident pages.
    fn release_resident(&self) {}
}

impl ExpertReader for PackDir {
    fn dims(&self) -> Dims {
        self.dims
    }
    fn layers(&self) -> usize {
        PackDir::layers(self)
    }
    fn experts(&self) -> usize {
        PackDir::experts(self)
    }
    fn new_buffer(&self) -> AlignedBuf {
        PackDir::new_buffer(self)
    }
    fn read(&self, layer: usize, expert: usize, buf: &mut AlignedBuf) -> Result<()> {
        self.read_expert(layer, expert, buf)
    }
    fn release_layer(&self, layer: usize) {
        self.drop_layer_cache(layer);
    }
    fn release_resident(&self) {
        PackDir::release(self);
    }
}

/// One expert's computation on a consumer: `out[m, hidden] = down(silu(gate x) * (up x))`
/// for the gathered rows `x [m, hidden]` (no router weighting).
pub trait ExpertWorker: Send {
    fn run(
        &mut self,
        rec: &ExpertRecord<'_>,
        x: &[f32],
        m: usize,
        out: &mut [f32],
        t: &mut ExpertTimes,
    ) -> Result<()>;
    /// Release cached scratch down to `keep_bytes` (the NPU workers' shape cache).
    fn trim(&mut self, _keep_bytes: usize) {}
}

/// The attention projections.
pub trait Dense: Send + Sync {
    fn label(&self) -> String;
    fn qkv(&self, layer: usize, n: &[f32], s: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)>;
    fn o_proj(&self, layer: usize, ctx: &[f32], s: usize) -> Result<Vec<f32>>;
    fn trim(&self, _keep_bytes: usize) {}
}

// ---------------------------------------------------------------------------------------
// token bucketing

/// Tokens grouped by the expert that serves them.
pub struct Buckets {
    /// Per expert: `(token, router weight)` in ascending token order.
    pub rows: Vec<Vec<(u32, f32)>>,
    /// `[s * top_k]`: for token `t`, slot `j`: `(expert, row within the expert's bucket)`.
    pub slots: Vec<(u32, u32)>,
}

impl Buckets {
    pub fn build(ids: &[u32], wts: &[f32], s: usize, top_k: usize, experts: usize) -> Self {
        assert_eq!(ids.len(), s * top_k);
        assert_eq!(wts.len(), s * top_k);
        let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); experts];
        let mut slots = Vec::with_capacity(s * top_k);
        for t in 0..s {
            for j in 0..top_k {
                let e = ids[t * top_k + j] as usize;
                slots.push((e as u32, rows[e].len() as u32));
                rows[e].push((t as u32, wts[t * top_k + j]));
            }
        }
        Self { rows, slots }
    }

    /// Experts with at least one token, ascending.
    pub fn hit(&self) -> Vec<usize> {
        (0..self.rows.len())
            .filter(|&e| !self.rows[e].is_empty())
            .collect()
    }
}

/// `y[t] += sum over the token's experts (ascending expert id) of outs[e][row]`; the
/// expert-id order makes the sum independent of how the experts were scheduled.
pub fn combine(
    y: &mut [f32],
    h: usize,
    top_k: usize,
    b: &Buckets,
    outs: &[Option<Vec<f32>>],
) -> Result<()> {
    y.par_chunks_mut(h)
        .enumerate()
        .try_for_each(|(t, yr)| -> Result<()> {
            let mut sl: Vec<(u32, u32)> = b.slots[t * top_k..(t + 1) * top_k].to_vec();
            sl.sort_unstable_by_key(|&(e, _)| e);
            for (e, row) in sl {
                let out = outs[e as usize]
                    .as_ref()
                    .ok_or_else(|| anyhow!("expert {e} produced no output"))?;
                let r = &out[row as usize * h..(row as usize + 1) * h];
                for (a, v) in yr.iter_mut().zip(r) {
                    *a += v;
                }
            }
            Ok(())
        })
}

// ---------------------------------------------------------------------------------------
// the reader stream

struct Cmd {
    layer: usize,
    experts: Vec<usize>,
}

struct Item {
    layer: usize,
    expert: usize,
    buf: std::result::Result<AlignedBuf, String>,
}

/// A request's reader thread plus its queues.
struct Stream {
    cmd_tx: Option<Sender<Cmd>>,
    full_rx: Option<Mutex<Receiver<Item>>>,
    free_tx: Option<Sender<AlignedBuf>>,
    reader: Option<std::thread::JoinHandle<()>>,
    /// The next layer that has not been queued yet (layers are queued in order).
    next_queue: usize,
    read_ns: Arc<AtomicU64>,
}

impl Stream {
    fn start(reader: Arc<dyn ExpertReader>, depth: usize, consumers: usize) -> Self {
        let depth = depth.max(1);
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        let (full_tx, full_rx): (SyncSender<Item>, Receiver<Item>) = mpsc::sync_channel(depth);
        let (free_tx, free_rx) = mpsc::channel::<AlignedBuf>();
        for _ in 0..depth + consumers {
            free_tx.send(reader.new_buffer()).ok();
        }
        let read_ns = Arc::new(AtomicU64::new(0));
        let ns = read_ns.clone();
        let handle = std::thread::Builder::new()
            .name("clef-reader".into())
            .spawn(move || {
                for cmd in cmd_rx {
                    for e in cmd.experts {
                        let Ok(mut buf) = free_rx.recv() else { return };
                        let t = Instant::now();
                        let r = reader.read(cmd.layer, e, &mut buf);
                        ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        let item = Item {
                            layer: cmd.layer,
                            expert: e,
                            buf: r.map(|()| buf).map_err(|e| format!("{e:#}")),
                        };
                        if full_tx.send(item).is_err() {
                            return;
                        }
                    }
                }
            })
            .expect("spawn the expert reader");
        Self {
            cmd_tx: Some(cmd_tx),
            full_rx: Some(Mutex::new(full_rx)),
            free_tx: Some(free_tx),
            reader: Some(handle),
            next_queue: 0,
            read_ns,
        }
    }

    fn queue(&mut self, layer: usize, experts: Vec<usize>) {
        debug_assert_eq!(layer, self.next_queue);
        if let Some(tx) = &self.cmd_tx {
            tx.send(Cmd { layer, experts }).ok();
        }
        self.next_queue = layer + 1;
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Unblock the reader wherever it waits (queue full / no free buffer / no command).
        drop(self.full_rx.take());
        drop(self.free_tx.take());
        drop(self.cmd_tx.take());
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}

// ---------------------------------------------------------------------------------------
// the MoE engine

/// Per-expert weighted outputs, summed worker times, summed read-wait seconds, experts run.
type LayerRun = (Vec<Option<Vec<f32>>>, ExpertTimes, f64, usize);

pub struct PackEngine {
    label: String,
    cfg: TextConfig,
    reader: Arc<dyn ExpertReader>,
    dense: Box<dyn Dense>,
    workers: Mutex<Vec<Box<dyn ExpertWorker>>>,
    stream: Mutex<Option<Stream>>,
    depth: usize,
    speculate: bool,
    keep_bytes: usize,
    phases: Mutex<Phases>,
    last: Mutex<Option<Phases>>,
}

impl PackEngine {
    pub fn new(
        label: String,
        cfg: TextConfig,
        reader: Arc<dyn ExpertReader>,
        dense: Box<dyn Dense>,
        workers: Vec<Box<dyn ExpertWorker>>,
        depth: usize,
        speculate: bool,
    ) -> Self {
        assert!(!workers.is_empty());
        Self {
            label,
            cfg,
            reader,
            dense,
            workers: Mutex::new(workers),
            stream: Mutex::new(None),
            depth,
            speculate,
            keep_bytes: crate::util::mem::keep_scratch_mib() * 1024 * 1024,
            phases: Mutex::new(Phases::default()),
            last: Mutex::new(None),
        }
    }

    fn speculating(&self, s: usize) -> bool {
        self.speculate && s * self.cfg.top_k >= SPEC_MIN_SLOTS
    }

    fn ensure_stream<'a>(&self, g: &'a mut Option<Stream>) -> &'a mut Stream {
        g.get_or_insert_with(|| {
            let consumers = lock(&self.workers).len();
            Stream::start(self.reader.clone(), self.depth, consumers)
        })
    }

    /// Run the layer's experts on the consumers; returns the per-expert weighted outputs.
    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &self,
        st: &Stream,
        workers: &mut [Box<dyn ExpertWorker>],
        layer: usize,
        b: &Buckets,
        n2: &[f32],
    ) -> Result<LayerRun> {
        let h = self.cfg.hidden;
        let dims = self.reader.dims();
        let need = b.hit().len();
        let taken = AtomicUsize::new(0);
        let abort = AtomicBool::new(false);
        let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        let panic_payload: Mutex<Option<Box<dyn Any + Send>>> = Mutex::new(None);
        let outs: Mutex<Vec<(usize, Vec<f32>)>> = Mutex::new(Vec::with_capacity(need));
        let totals: Mutex<(ExpertTimes, f64)> = Mutex::new((ExpertTimes::default(), 0.0));
        let rx = st.full_rx.as_ref().expect("stream receiver");
        let free_tx = st.free_tx.as_ref().expect("stream free list");
        let fail = |e: anyhow::Error| {
            let mut g = lock(&first_err);
            if g.is_none() {
                *g = Some(e);
            }
            abort.store(true, Ordering::SeqCst);
        };
        std::thread::scope(|sc| {
            for w in workers.iter_mut() {
                let (taken, abort, outs, totals, fail, panic_payload) =
                    (&taken, &abort, &outs, &totals, &fail, &panic_payload);
                sc.spawn(move || {
                    let body = || {
                        let mut times = ExpertTimes::default();
                        let mut wait = 0f64;
                        loop {
                            if abort.load(Ordering::SeqCst) {
                                break;
                            }
                            let t0 = Instant::now();
                            let item = {
                                let g = lock(rx);
                                loop {
                                    if taken.load(Ordering::SeqCst) >= need {
                                        break None;
                                    }
                                    match g.recv() {
                                        Err(_) => {
                                            fail(anyhow!("the expert reader stopped"));
                                            break None;
                                        }
                                        Ok(it) => {
                                            if it.layer < layer || b.rows[it.expert].is_empty() {
                                                if let Ok(buf) = it.buf {
                                                    free_tx.send(buf).ok();
                                                }
                                                continue;
                                            }
                                            if it.layer > layer {
                                                fail(anyhow!(
                                                    "expert stream out of order: got layer {} \
                                                     while running {layer}",
                                                    it.layer
                                                ));
                                                break None;
                                            }
                                            taken.fetch_add(1, Ordering::SeqCst);
                                            break Some(it);
                                        }
                                    }
                                }
                            };
                            wait += t0.elapsed().as_secs_f64();
                            let Some(it) = item else { break };
                            let buf = match it.buf {
                                Ok(b) => b,
                                Err(msg) => {
                                    fail(anyhow!("read layer {layer} expert {}: {msg}", it.expert));
                                    break;
                                }
                            };
                            let r = (|| -> Result<(usize, Vec<f32>)> {
                                let tg = Instant::now();
                                let rec = ExpertRecord::parse(dims, buf.as_slice())?;
                                let rows = &b.rows[it.expert];
                                let m = rows.len();
                                let mut xe = Vec::with_capacity(m * h);
                                for &(t, _) in rows {
                                    xe.extend_from_slice(&n2[t as usize * h..(t as usize + 1) * h]);
                                }
                                let mut out = vec![0f32; m * h];
                                times.glue += tg.elapsed().as_secs_f64();
                                w.run(&rec, &xe, m, &mut out, &mut times)?;
                                let tg = Instant::now();
                                for (row, &(_, wt)) in out.chunks_mut(h).zip(rows) {
                                    row.iter_mut().for_each(|v| *v *= wt);
                                }
                                times.glue += tg.elapsed().as_secs_f64();
                                Ok((it.expert, out))
                            })();
                            free_tx.send(buf).ok();
                            match r {
                                Ok(o) => lock(outs).push(o),
                                Err(e) => {
                                    fail(e);
                                    break;
                                }
                            }
                        }
                        let mut g = lock(totals);
                        g.0.add(&times);
                        g.1 += wait;
                    };
                    if let Err(p) = catch_unwind(AssertUnwindSafe(body)) {
                        *lock(panic_payload) = Some(p);
                        abort.store(true, Ordering::SeqCst);
                    }
                });
            }
        });
        if let Some(p) = lock(&panic_payload).take() {
            resume_unwind(p);
        }
        if let Some(e) = lock(&first_err).take() {
            return Err(e);
        }
        let mut by_expert: Vec<Option<Vec<f32>>> = vec![None; self.cfg.experts];
        let got = lock(&outs).len();
        for (e, o) in std::mem::take(&mut *lock(&outs)) {
            by_expert[e] = Some(o);
        }
        if got != need {
            bail!("layer {layer}: {got} of {need} experts were processed");
        }
        let (times, wait) = *lock(&totals);
        Ok((by_expert, times, wait, need))
    }
}

/// Device failures must reach the server as a typed panic payload (it maps NOMEM to 503
/// and exits on a dead device); everything else stays an ordinary error.
pub fn raise_if_device_failure(e: &anyhow::Error) {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if let Some(f) = e.downcast_ref::<super::engine_npu::NpuFailure>() {
        f.raise();
    }
    let _ = e;
}

impl LayerEngine for PackEngine {
    fn label(&self) -> String {
        format!(
            "{}; attention projections {}",
            self.label,
            self.dense.label()
        )
    }

    fn hint_layer(&self, layer: usize, s: usize) {
        if !self.speculating(s) || layer >= self.reader.layers() {
            return;
        }
        let mut g = lock(&self.stream);
        let st = self.ensure_stream(&mut g);
        if st.next_queue == layer {
            st.queue(layer, (0..self.cfg.experts).collect());
        }
    }

    fn qkv(&self, layer: usize, n: &[f32], s: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let t = Instant::now();
        let r = self.dense.qkv(layer, n, s);
        lock(&self.phases).qkv += t.elapsed().as_secs_f64();
        if let Err(e) = &r {
            raise_if_device_failure(e);
        }
        r
    }

    fn o_proj(&self, layer: usize, ctx: &[f32], s: usize) -> Result<Vec<f32>> {
        let t = Instant::now();
        let r = self.dense.o_proj(layer, ctx, s);
        lock(&self.phases).oproj += t.elapsed().as_secs_f64();
        if let Err(e) = &r {
            raise_if_device_failure(e);
        }
        r
    }

    fn moe_add(
        &self,
        layer: usize,
        n2: &[f32],
        s: usize,
        ids: &[u32],
        wts: &[f32],
        y: &mut [f32],
    ) -> Result<()> {
        let t_all = Instant::now();
        let (h, k) = (self.cfg.hidden, self.cfg.top_k);
        let b = Buckets::build(ids, wts, s, k, self.cfg.experts);
        let hit = b.hit();
        let res = {
            let mut g = lock(&self.stream);
            let st = self.ensure_stream(&mut g);
            if st.next_queue <= layer {
                debug_assert_eq!(st.next_queue, layer);
                st.queue(layer, hit.clone());
            }
            let mut workers = lock(&self.workers);
            let r = self.run_layer(st, &mut workers, layer, &b, n2);
            if r.is_err() {
                *g = None; // tear the reader down; the next request starts afresh
            }
            r
        };
        let (outs, times, wait, nexp) = match res {
            Ok(r) => r,
            Err(e) => {
                raise_if_device_failure(&e);
                return Err(e);
            }
        };
        let t_c = Instant::now();
        combine(y, h, k, &b, &outs)?;
        let combine_s = t_c.elapsed().as_secs_f64();
        drop(outs);
        self.reader.release_layer(layer);
        self.reader.release_resident();
        // the next layer's reads overlap the combine's tail, the tap and the attention
        self.hint_layer(layer + 1, s);
        let moe = t_all.elapsed().as_secs_f64();
        let mut p = lock(&self.phases);
        p.read_wait_thread += wait;
        p.rewrite_thread += times.rewrite;
        p.quant_thread += times.quant;
        p.npu_thread += times.npu;
        p.glue_thread += times.glue;
        p.combine += combine_s;
        p.moe += moe;
        p.experts += nexp;
        p.rows += s * k;
        p.layers += 1;
        p.layer_moe.push(moe);
        p.peak_anon_mib = p.peak_anon_mib.max(crate::util::mem::rss_mib());
        Ok(())
    }

    fn note(&self, phase: Phase, secs: f64) {
        let mut p = lock(&self.phases);
        match phase {
            Phase::Attn => p.attn += secs,
            Phase::Router => p.router += secs,
        }
    }

    fn finish_request(&self, tokens: usize) {
        let read_io = {
            let mut g = lock(&self.stream);
            let ns = g.as_ref().map_or(0, |s| s.read_ns.load(Ordering::Relaxed));
            *g = None;
            ns as f64 * 1e-9
        };
        let mut p = std::mem::take(&mut *lock(&self.phases));
        p.read_io_thread = read_io;
        *lock(&self.last) = Some(p.clone());
        {
            let mut ws = lock(&self.workers);
            let each = self.keep_bytes / ws.len().max(1);
            for w in ws.iter_mut() {
                w.trim(each);
            }
        }
        self.dense.trim(self.keep_bytes);
        self.reader.release_resident();
        if phase_log_level() > 0 {
            eprintln!(
                "clef phases [{}] tokens={tokens} layers={} experts={} rows={} | wall: moe={:.2}s \
                 qkv={:.2}s oproj={:.2}s attn={:.2}s router={:.2}s combine={:.2}s | thread-s: \
                 read_wait={:.2} rewrite={:.2} quant={:.2} npu={:.2} glue={:.2} read_io={:.2} | \
                 peak_anon={:.0} MiB",
                self.label,
                p.layers,
                p.experts,
                p.rows,
                p.moe,
                p.qkv,
                p.oproj,
                p.attn,
                p.router,
                p.combine,
                p.read_wait_thread,
                p.rewrite_thread,
                p.quant_thread,
                p.npu_thread,
                p.glue_thread,
                p.read_io_thread,
                p.peak_anon_mib,
            );
            if phase_log_level() > 1 {
                for (l, m) in p.layer_moe.iter().enumerate() {
                    eprintln!("clef layer {l}: moe={m:.3}s");
                }
            }
            crate::util::mem::log_npu_mem();
        }
    }

    fn last_phases(&self) -> Option<Phases> {
        lock(&self.last).clone()
    }

    fn abort_request(&self) {
        *lock(&self.stream) = None;
        *lock(&self.phases) = Phases::default();
    }
}

// ---------------------------------------------------------------------------------------
// one engine layer

/// One decoder layer with the engine doing the projections and the MoE.
#[allow(clippy::too_many_arguments)]
pub fn engine_layer_forward(
    cfg: &TextConfig,
    w: &LayerSmall,
    eng: &dyn LayerEngine,
    layer: usize,
    x: &[f32],
    s: usize,
    rope: &Rope,
    threads: usize,
) -> Result<LayerOut> {
    engine_layer_forward_multi(cfg, w, eng, layer, x, &[s], rope, threads)
}

/// [`engine_layer_forward`] over several records whose rows are concatenated in `x`
/// (`lens[i]` rows each). The row-independent steps (norms, the q|k|v and o
/// projections, the residual) run over all rows at once; RoPE (positions restart per
/// record), the causal scores and the router run per record, so no record attends to
/// another; ONE `moe_add` buckets the rows of every record per expert, so each expert
/// of the layer is read and run once. `rope` must cover the longest record.
#[allow(clippy::too_many_arguments)]
pub fn engine_layer_forward_multi(
    cfg: &TextConfig,
    w: &LayerSmall,
    eng: &dyn LayerEngine,
    layer: usize,
    x: &[f32],
    lens: &[usize],
    rope: &Rope,
    threads: usize,
) -> Result<LayerOut> {
    let (h, nh, nkv, d) = (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim);
    let s: usize = lens.iter().sum();
    assert_eq!(x.len(), s * h);
    // reads of this layer's experts start before the attention
    eng.hint_layer(layer, s);
    let n = rms_norm(x, h, &w.input_norm, cfg.eps);
    let (mut q, mut k, v) = eng.qkv(layer, &n, s)?;
    drop(n);
    let t = Instant::now();
    rms_norm_rows(&mut q, d, &w.q_norm, cfg.eps);
    rms_norm_rows(&mut k, d, &w.k_norm, cfg.eps);
    let mut ctx = Vec::with_capacity(s * nh * d);
    let mut off = 0;
    for &sr in lens {
        let (qr, kr) = (
            &mut q[off * nh * d..(off + sr) * nh * d],
            &mut k[off * nkv * d..(off + sr) * nkv * d],
        );
        rope.apply(qr, sr, nh);
        rope.apply(kr, sr, nkv);
        ctx.extend(causal_gqa(
            cfg,
            qr,
            kr,
            &v[off * nkv * d..(off + sr) * nkv * d],
            sr,
        ));
        off += sr;
    }
    eng.note(Phase::Attn, t.elapsed().as_secs_f64());
    drop((q, k, v));
    let proj = eng.o_proj(layer, &ctx, s)?;
    drop(ctx);
    let hid: Vec<f32> = x.iter().zip(&proj).map(|(a, b)| a + b).collect();
    drop(proj);

    let n2 = rms_norm(&hid, h, &w.post_norm, cfg.eps);
    let t = Instant::now();
    let (mut ids, mut wts, mut margins) = (Vec::new(), Vec::new(), Vec::new());
    let mut off = 0;
    for &sr in lens {
        let logits = matmul_nt(
            &n2[off * h..(off + sr) * h],
            sr,
            h,
            &w.router,
            cfg.experts,
            threads,
        );
        let (i, wt, mg) = route(cfg, &logits, sr);
        ids.extend(i);
        wts.extend(wt);
        margins.extend(mg);
        off += sr;
    }
    eng.note(Phase::Router, t.elapsed().as_secs_f64());
    let mut y = hid;
    eng.moe_add(layer, &n2, s, &ids, &wts, &mut y)?;
    Ok(LayerOut {
        hidden: y,
        router_ids: ids,
        router_weights: wts,
        router_margins: margins,
    })
}

// ---------------------------------------------------------------------------------------
// workers and dense projections for the host emulation (`Int8Emu`) and the CPU fallback

/// Host emulation of exactly what the NPU worker computes: W8A8 group-wise matmuls with
/// int32-exact group partials (see `int8.rs`), SiLU*mul and the quantization of the
/// intermediate on the host.
pub struct EmuWorker {
    pub dims: Dims,
}

impl EmuWorker {
    pub fn new(dims: Dims) -> Self {
        Self { dims }
    }
}

impl ExpertWorker for EmuWorker {
    fn run(
        &mut self,
        rec: &ExpertRecord<'_>,
        x: &[f32],
        m: usize,
        out: &mut [f32],
        t: &mut ExpertTimes,
    ) -> Result<()> {
        let Dims {
            hidden: h,
            inter: i,
            group: g,
        } = self.dims;
        let t0 = Instant::now();
        let (mut aq, mut a_s) = (vec![0i8; m * h], vec![0f32; m * h / g]);
        quantize_rows(x, h, g, &mut aq, &mut a_s, false);
        let mut gu_s = vec![0f32; rec.gu_scales.len()];
        f16_scales_to_f32(rec.gu_scales, &mut gu_s);
        t.quant += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let mut gu = vec![0f32; m * 2 * i];
        emu_matmul_gw(
            &aq,
            &a_s,
            m,
            h,
            rec.gu_codes,
            &gu_s,
            2 * i,
            g,
            &mut gu,
            false,
        );
        t.npu += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let mut act = vec![0f32; m * i];
        for (a, row) in act.chunks_mut(i).zip(gu.chunks(2 * i)) {
            for j in 0..i {
                a[j] = silu(row[j]) * row[i + j];
            }
        }
        t.glue += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let (mut aq2, mut a_s2) = (vec![0i8; m * i], vec![0f32; m * i / g]);
        quantize_rows(&act, i, g, &mut aq2, &mut a_s2, false);
        let mut dn_s = vec![0f32; rec.down_scales.len()];
        f16_scales_to_f32(rec.down_scales, &mut dn_s);
        t.quant += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        emu_matmul_gw(&aq2, &a_s2, m, i, rec.down_codes, &dn_s, h, g, out, false);
        t.npu += t0.elapsed().as_secs_f64();
        Ok(())
    }
}

/// `[n, k]` int8 codes and `[n, k/group]` f32 scales of a resident `<module>.codes/.scales`.
pub fn resident_linear<'a>(
    pack: &'a PackDir,
    module: &str,
    n: usize,
    k: usize,
) -> Result<(&'a [i8], Vec<f32>)> {
    use super::st::Dtype;
    let g = pack.dims.group;
    let ct = pack.tensor(&format!("{module}.codes"))?;
    let st = pack.tensor(&format!("{module}.scales"))?;
    if ct.dtype != Dtype::I8
        || ct.shape != [n, k]
        || st.dtype != Dtype::F16
        || st.shape != [n, k / g]
    {
        bail!(
            "{module}: unexpected codes/scales shapes {:?} / {:?}",
            ct.shape,
            st.shape
        );
    }
    let bytes = pack.resident().bytes(ct);
    // SAFETY: i8 and u8 share a layout; the slice is the tensor's exact byte range.
    let codes = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) };
    let sb = pack.resident().bytes(st);
    let scales: Vec<f32> = sb
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| super::st::f16_to_f32(u16::from_le_bytes(*c)))
        .collect();
    Ok((codes, scales))
}

/// Attention projections computed by the host emulation of the W8A8 path.
pub struct EmuDense {
    pack: Arc<PackDir>,
    cfg: TextConfig,
}

impl EmuDense {
    pub fn new(pack: Arc<PackDir>, cfg: TextConfig) -> Self {
        Self { pack, cfg }
    }
}

impl Dense for EmuDense {
    fn label(&self) -> String {
        "int8 g128 W8A8 (host emulation)".into()
    }

    fn qkv(&self, layer: usize, n: &[f32], s: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (h, d) = (self.cfg.hidden, self.cfg.head_dim);
        let g = self.pack.dims.group;
        let (mut aq, mut a_s) = (vec![0i8; s * h], vec![0f32; s * h / g]);
        quantize_rows(n, h, g, &mut aq, &mut a_s, true);
        let pre = format!("thinker.model.layers.{layer}.self_attn.");
        let one = |name: &str, rows: usize| -> Result<Vec<f32>> {
            let (codes, scales) = resident_linear(&self.pack, &format!("{pre}{name}"), rows, h)?;
            let mut out = vec![0f32; s * rows];
            emu_matmul_gw(&aq, &a_s, s, h, codes, &scales, rows, g, &mut out, true);
            Ok(out)
        };
        let q = one("q_proj", self.cfg.heads * d)?;
        let k = one("k_proj", self.cfg.kv_heads * d)?;
        let v = one("v_proj", self.cfg.kv_heads * d)?;
        Ok((q, k, v))
    }

    fn o_proj(&self, layer: usize, ctx: &[f32], s: usize) -> Result<Vec<f32>> {
        let (h, kk) = (self.cfg.hidden, self.cfg.heads * self.cfg.head_dim);
        let g = self.pack.dims.group;
        let (mut aq, mut a_s) = (vec![0i8; s * kk], vec![0f32; s * kk / g]);
        quantize_rows(ctx, kk, g, &mut aq, &mut a_s, true);
        let name = format!("thinker.model.layers.{layer}.self_attn.o_proj");
        let (codes, scales) = resident_linear(&self.pack, &name, h, kk)?;
        let mut out = vec![0f32; s * h];
        emu_matmul_gw(&aq, &a_s, s, kk, codes, &scales, h, g, &mut out, true);
        Ok(out)
    }
}

/// Attention projections on the CPU in f32 from the dequantized int8 weights
/// (`ROCKET_CLEF_ATTN_DEVICE=cpu`).
pub struct CpuDense {
    pack: Arc<PackDir>,
    cfg: TextConfig,
    threads: usize,
}

impl CpuDense {
    pub fn new(pack: Arc<PackDir>, cfg: TextConfig, threads: usize) -> Self {
        Self { pack, cfg, threads }
    }
}

impl Dense for CpuDense {
    fn label(&self) -> String {
        "f32 on the CPU (dequantized int8)".into()
    }

    fn qkv(&self, layer: usize, n: &[f32], s: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (h, d) = (self.cfg.hidden, self.cfg.head_dim);
        let pre = format!("thinker.model.layers.{layer}.self_attn.");
        let one = |name: &str, rows: usize| -> Result<Vec<f32>> {
            let w = self.pack.dequant_linear(&format!("{pre}{name}"), rows, h)?;
            Ok(matmul_nt(n, s, h, &w, rows, self.threads))
        };
        let q = one("q_proj", self.cfg.heads * d)?;
        let k = one("k_proj", self.cfg.kv_heads * d)?;
        let v = one("v_proj", self.cfg.kv_heads * d)?;
        Ok((q, k, v))
    }

    fn o_proj(&self, layer: usize, ctx: &[f32], s: usize) -> Result<Vec<f32>> {
        let (h, kk) = (self.cfg.hidden, self.cfg.heads * self.cfg.head_dim);
        let name = format!("thinker.model.layers.{layer}.self_attn.o_proj");
        let w = self.pack.dequant_linear(&name, h, kk)?;
        Ok(matmul_nt(ctx, s, kk, &w, h, self.threads))
    }
}

/// What the placement env vars select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnDevice {
    Cpu,
    /// The NPU on an NPU build, the host emulation of it otherwise.
    Npu,
}

/// `ROCKET_CLEF_ATTN_DEVICE=cpu|npu` (default npu).
pub fn attn_device_from_env() -> Result<AttnDevice> {
    match std::env::var("ROCKET_CLEF_ATTN_DEVICE").ok().as_deref() {
        None | Some("") | Some("npu") => Ok(AttnDevice::Npu),
        Some("cpu") => Ok(AttnDevice::Cpu),
        Some(o) => bail!("ROCKET_CLEF_ATTN_DEVICE={o}: expected cpu or npu"),
    }
}

/// `ROCKET_CLEF_EMU=1`: the host emulation of the NPU path on any host.
pub fn emu_from_env() -> bool {
    std::env::var("ROCKET_CLEF_EMU").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// `ROCKET_CLEF_DEPTH` read-ahead buffers (default 8, 4.57 MiB each).
pub fn depth_from_env() -> usize {
    std::env::var("ROCKET_CLEF_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&d: &usize| d >= 1)
        .unwrap_or(8)
}

/// Build the host-emulation engine over an opened pack.
pub fn emu_engine(
    pack: Arc<PackDir>,
    cfg: TextConfig,
    threads: usize,
    attn: AttnDevice,
) -> PackEngine {
    let dims = pack.dims;
    let dense: Box<dyn Dense> = match attn {
        AttnDevice::Npu => Box::new(EmuDense::new(pack.clone(), cfg.clone())),
        AttnDevice::Cpu => Box::new(CpuDense::new(pack.clone(), cfg.clone(), threads)),
    };
    let workers: Vec<Box<dyn ExpertWorker>> = (0..threads.saturating_sub(1).max(1))
        .map(|_| Box::new(EmuWorker::new(dims)) as Box<dyn ExpertWorker>)
        .collect();
    PackEngine::new(
        format!("int8 g{} W8A8 host emulation, streamed experts", dims.group),
        cfg,
        pack,
        dense,
        workers,
        depth_from_env(),
        true,
    )
}

#[cfg(test)]
mod tests;
