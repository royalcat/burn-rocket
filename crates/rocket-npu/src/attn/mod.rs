//! Masked grouped-query flash attention on the fp16 matmul engine.
//!
//! The layout and semantics match the C library's `rocket_flash_attn_fp16`:
//!
//! - `Q`: `[n_head][n_tokens][head_dim]`
//! - `K`: `[n_kv_heads][n_kv][head_dim]`
//! - `V`: `[n_kv_heads][dv][n_kv]`
//! - `mask`: `[n_tokens][n_kv]` additive f16 (or none)
//! - `out`: `[n_head][n_tokens][dv]`
//!
//! Scores are brought host-side for the mandatory additive mask, scale, optional
//! soft-cap and softmax, then uploaded as the AV operand (the C default,
//! `ROCKET_ATTN_HOST_SOFTMAX=1`). Heads split into contiguous ranges across worker
//! fds, one per NPU scheduling entity.

use std::thread;

use half::f16;

use crate::Device;
use crate::device::affinity;
use crate::error::{Error, Result};

use crate::matmul::Config;
use crate::matmul::bos::Bos;
use crate::matmul::plan::Plan;
use crate::matmul::run::{compute_cpu, compute_kacc};

const PAD_KEY_SCORE: f32 = -30000.0;
const MAX_WORKERS: usize = 8;

struct Scratch {
    plan: Plan,
    bos: Bos,
}

impl Scratch {
    fn new(dev: &Device, m: usize, k: usize, n: usize) -> Result<Self> {
        let plan = Plan::new(m, k, n)?;
        let bos = Bos::new(dev, &plan, Config::from_env())?;
        Ok(Scratch { plan, bos })
    }

    fn get(&mut self, dev: &Device, m: usize, k: usize, n: usize) -> Result<()> {
        if self.plan.m != m || self.plan.k != k || self.plan.n != n {
            *self = Scratch::new(dev, m, k, n)?;
        }
        Ok(())
    }

    fn matmul(&mut self, a: &[f16], b: &[f16], c: &mut [f16]) -> Result<()> {
        self.bos.pack_input(&self.plan, a)?;
        self.bos.pack_weights(&self.plan, b)?;
        let wt = self.bos.wt_all.mem_ref();
        if self.bos.has_pong {
            match compute_kacc(&self.plan, &mut self.bos, wt, c) {
                Ok(()) => return Ok(()),
                Err(e) if e.status() == crate::Status::Tiling => {}
                Err(e) => return Err(e),
            }
        }
        compute_cpu(&self.plan, &mut self.bos, wt, c)
    }
}

struct Worker {
    dev: Device,
    qk: Scratch,
    av: Scratch,
    qh: Vec<f16>,
    kh: Vec<f16>,
    vh: Vec<f16>,
    p: Vec<f16>,
    scores: Vec<f16>,
    ctx: Vec<f16>,
    out: Vec<f16>,
    head_dim: usize,
    dv: usize,
    tp: usize,
    kn: usize,
}

impl Worker {
    fn new(dev: Device) -> Result<Self> {
        // Small placeholder shapes; replaced on the first real call.
        let qk = Scratch::new(&dev, 4, 32, 16)?;
        let av = Scratch::new(&dev, 4, 32, 16)?;
        Ok(Worker {
            dev,
            qk,
            av,
            qh: Vec::new(),
            kh: Vec::new(),
            vh: Vec::new(),
            p: Vec::new(),
            scores: Vec::new(),
            ctx: Vec::new(),
            out: Vec::new(),
            head_dim: 0,
            dv: 0,
            tp: 0,
            kn: 0,
        })
    }

    fn prepare(&mut self, t: usize, kv: usize, dk: usize, dv: usize) -> Result<()> {
        self.tp = t.div_ceil(4) * 4;
        self.kn = kv.div_ceil(32) * 32;
        self.head_dim = dk;
        self.dv = dv;
        if self.kn == 0 || self.tp == 0 {
            return Err(Error::shape("flash_attn/empty"));
        }
        self.qh.resize(self.tp * dk, f16::ZERO);
        self.kh.resize(self.kn * dk, f16::ZERO);
        self.vh.resize(dv * self.kn, f16::ZERO);
        self.scores.resize(self.tp * self.kn, f16::ZERO);
        self.p.resize(self.tp * self.kn, f16::ZERO);
        self.ctx.resize(self.tp * dv, f16::ZERO);
        self.qk.get(&self.dev, self.tp, dk, self.kn)?;
        self.av.get(&self.dev, self.tp, self.kn, dv)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        t: usize,
        kv: usize,
        heads: &[usize],
        gqa: usize,
        scale: f32,
        softcap: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
    ) -> Result<()> {
        let dk = self.head_dim;
        let dv = self.dv;
        let tp = self.tp;
        let kn = self.kn;
        self.out.clear();
        self.out.resize(heads.len() * t * dv, f16::ZERO);
        for (oi, &h) in heads.iter().enumerate() {
            let hk = h / gqa;
            self.qh.fill(f16::ZERO);
            for row in 0..t {
                let src = (h * t + row) * dk;
                let dst = row * dk;
                self.qh[dst..dst + dk].copy_from_slice(&q[src..src + dk]);
            }
            self.kh.fill(f16::ZERO);
            for row in 0..kv {
                let src = (hk * kv + row) * dk;
                let dst = row * dk;
                self.kh[dst..dst + dk].copy_from_slice(&k[src..src + dk]);
            }
            self.vh.fill(f16::ZERO);
            for c in 0..dv {
                let src = (hk * dv + c) * kv;
                let dst = c * kn;
                self.vh[dst..dst + kv].copy_from_slice(&v[src..src + kv]);
            }

            self.qk.matmul(&self.qh, &self.kh, &mut self.scores)?;
            softmax(
                &self.scores,
                &mut self.p,
                mask,
                t,
                kv,
                tp,
                kn,
                scale,
                softcap,
            );
            self.av.matmul(&self.p, &self.vh, &mut self.ctx)?;

            for row in 0..t {
                let dst = (oi * t + row) * dv;
                let src = row * dv;
                self.out[dst..dst + dv].copy_from_slice(&self.ctx[src..src + dv]);
            }
        }
        Ok(())
    }
}

/// A persistent attention context: one device fd per worker, per-shape scratch.
pub struct Attention {
    workers: Vec<Worker>,
}

impl Attention {
    pub fn new(threads: usize) -> Result<Attention> {
        let big = affinity::num_big_cores();
        let cap = if big > 0 { big } else { MAX_WORKERS };
        let n = threads.clamp(1, MAX_WORKERS).min(cap);
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            workers.push(Worker::new(Device::open()?)?);
        }
        Ok(Attention { workers })
    }

    /// `out = softmax(scale * Q K^T + mask) V`, per head, GQA-aware.
    #[allow(clippy::too_many_arguments)]
    pub fn flash(
        &mut self,
        n_tokens: usize,
        n_kv: usize,
        head_dim: usize,
        dv: usize,
        n_head: usize,
        n_kv_heads: usize,
        scale: f32,
        softcap: f32,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mask: Option<&[f16]>,
        out: &mut [f16],
    ) -> Result<()> {
        if n_head == 0 || n_kv_heads == 0 || !n_head.is_multiple_of(n_kv_heads) {
            return Err(Error::shape("flash_attn/heads"));
        }
        if !head_dim.is_multiple_of(32) || !dv.is_multiple_of(16) {
            return Err(Error::shape("flash_attn/dims"));
        }
        let gqa = n_head / n_kv_heads;
        if q.len() < n_head * n_tokens * head_dim
            || k.len() < n_kv_heads * n_kv * head_dim
            || v.len() < n_kv_heads * dv * n_kv
            || out.len() < n_head * n_tokens * dv
        {
            return Err(Error::shape("flash_attn/buffers"));
        }
        if let Some(m) = mask
            && m.len() < n_tokens * n_kv
        {
            return Err(Error::shape("flash_attn/mask"));
        }

        let threads = self.workers.len().min(n_head);
        // Balanced contiguous ranges: 16 heads over 5 workers is 3/3/3/3/4, not
        // 4/4/4/4 with the fifth idle.
        let range = |i: usize| (i * n_head / threads, (i + 1) * n_head / threads);
        let base = affinity::base();
        let results: Vec<Result<()>> = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for (i, w) in self.workers.iter_mut().take(threads).enumerate() {
                let (h0, h1) = range(i);
                if h0 >= h1 {
                    continue;
                }
                handles.push(scope.spawn(move || -> Result<()> {
                    affinity::pin_current(i, base);
                    w.prepare(n_tokens, n_kv, head_dim, dv)?;
                    let heads: Vec<usize> = (h0..h1).collect();
                    w.run(n_tokens, n_kv, &heads, gqa, scale, softcap, q, k, v, mask)
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(Error::device("attn/worker", 0)))
                })
                .collect()
        });
        if let Some(e) = results.into_iter().find_map(|r| r.err()) {
            return Err(e);
        }

        // Copy each worker's contiguous head range into the caller's output.
        for (i, w) in self.workers.iter().take(threads).enumerate() {
            let (h0, h1) = range(i);
            if h0 >= h1 {
                continue;
            }
            let n = (h1 - h0) * n_tokens * dv;
            out[h0 * n_tokens * dv..h0 * n_tokens * dv + n].copy_from_slice(&w.out[..n]);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn softmax(
    sc: &[f16],
    p: &mut [f16],
    mask: Option<&[f16]>,
    n_tokens: usize,
    n_kv: usize,
    tp: usize,
    kn: usize,
    scale: f32,
    softcap: f32,
) {
    let mut rowbuf = vec![0.0f32; kn];
    for i in 0..tp {
        let srow = &sc[i * kn..(i + 1) * kn];
        let mrow = mask
            .filter(|_| i < n_tokens)
            .map(|m| &m[i * n_kv..i * n_kv + n_kv]);
        let mut mx = PAD_KEY_SCORE;
        for (j, slot) in rowbuf.iter_mut().enumerate().take(kn) {
            let s = if j < n_kv {
                let mut s = srow[j].to_f32() * scale;
                if softcap != 0.0 {
                    s = softcap * (s / softcap).tanh();
                }
                if let Some(m) = mrow {
                    s += m[j].to_f32();
                }
                s.max(PAD_KEY_SCORE)
            } else {
                PAD_KEY_SCORE
            };
            *slot = s;
            if s > mx {
                mx = s;
            }
        }
        let mut sum = 0.0f32;
        for v in rowbuf.iter_mut() {
            let e = (*v - mx).exp();
            *v = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let prow = &mut p[i * kn..(i + 1) * kn];
        for j in 0..kn {
            prow[j] = f16::from_f32(rowbuf[j] * inv);
        }
    }
}

/// A held-open attention context, matching the C `rocket_fa_ctx` lifetime.
pub type FaContext = Attention;
