//! Native bf16 streaming matmul (f32 operands, bf16 hardware, f64 host accumulation).
//!
//! Port of `rocket_bf16_stream.c`: N is split across worker fds, A and B are
//! truncated to bf16 during the scatter, the NPU writes fp32 partials, and the host
//! accumulates K-tiles in f64 before narrowing to f32. There is no resident bf16
//! weight variant, so the weight is re-packed on every call.

use std::collections::{HashMap, VecDeque};
use std::thread;

use crate::Device;
use crate::device::uapi::DrmRocketTask;
use crate::device::{Bo, Handle};
use crate::error::{Error, Result};
use crate::regcmd::bf16::emit;
use crate::regcmd::desc::{BATCH, CBUF_BANK_SIZE, RC_STRIDE};
use crate::regcmd::fp16::MatmulParams;

use super::layout;
use super::plan::{BF16, Plan, plan_geom};
use super::run::wait_ns;

const MAX_WORKERS: usize = 8;

struct Scratch {
    workers: Vec<Worker>,
    a_buf: Option<Vec<u16>>,
}

struct Worker {
    n0: usize,
    nsub: usize,
    plan: Plan,
    guard: Bo,
    regcmd: Bo,
    in_all: Bo,
    wt_all: Bo,
    out_all: Bo,
    acc: Vec<f64>,
    csub: Vec<f32>,
    in_zeroed: bool,
    wt_zeroed: bool,
}

/// A persistent bf16 streaming context (fds + per-shape scratch).
pub struct Bf16Stream {
    devs: Vec<Device>,
    threads: usize,
    scratches: HashMap<(usize, usize, usize), Scratch>,
    order: VecDeque<(usize, usize, usize)>,
}

impl Bf16Stream {
    pub fn new(nthreads: usize) -> Result<Bf16Stream> {
        let threads = nthreads.clamp(1, MAX_WORKERS);
        let mut devs = Vec::with_capacity(threads);
        for _ in 0..threads {
            devs.push(Device::open()?);
        }
        Ok(Bf16Stream {
            devs,
            threads,
            scratches: HashMap::new(),
            order: VecDeque::new(),
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    fn build(&self, m: usize, k: usize, n: usize) -> Result<Scratch> {
        let nstep = fanout_step(n, self.threads, 16);
        let mut workers = Vec::with_capacity(self.threads);
        for t in 0..self.threads {
            let n0 = t * nstep;
            if n0 >= n {
                break;
            }
            let nsub = (n - n0).min(nstep);
            let plan = plan_for(m, k, nsub)?;
            let dev = &self.devs[t];
            let guard = dev.alloc_bo(4096)?;
            let regcmd = dev.alloc_bo(BATCH * RC_STRIDE * 8)?;
            let in_all = dev.alloc_bo(plan.nmt * plan.nkt * plan.in_slot * 2 + CBUF_BANK_SIZE)?;
            let wt_all = dev.alloc_bo(plan.nnt * plan.nkt * plan.wt_slot * 2 + CBUF_BANK_SIZE)?;
            let out_all = dev.alloc_bo(BATCH * plan.out_slot * 4 + CBUF_BANK_SIZE)?;
            workers.push(Worker {
                n0,
                nsub,
                plan,
                guard,
                regcmd,
                in_all,
                wt_all,
                out_all,
                acc: Vec::new(),
                csub: Vec::new(),
                in_zeroed: false,
                wt_zeroed: false,
            });
        }
        if workers.is_empty() {
            return Err(Error::shape("bf16/split"));
        }
        Ok(Scratch {
            workers,
            a_buf: None,
        })
    }

    fn scratch(&mut self, m: usize, k: usize, n: usize) -> Result<&mut Scratch> {
        let key = (m, k, n);
        if !self.scratches.contains_key(&key) {
            let workers = self.build(m, k, n)?;
            if self.scratches.len() >= 32
                && let Some(old) = self.order.pop_front()
            {
                self.scratches.remove(&old);
            }
            self.scratches.insert(key, workers);
            self.order.push_back(key);
        }
        Ok(self.scratches.get_mut(&key).unwrap())
    }

    /// `C[M,N] = A[M,K] * B[N,K]^T`, f32 in and out, bf16 device math.
    pub fn matmul(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<()> {
        if a.len() != m * k || b.len() != n * k || c.len() != m * n {
            return Err(Error::shape("bf16/buffers"));
        }
        let scratch = self.scratch(m, k, n)?;
        let Scratch { workers, a_buf } = scratch;
        let packed = prepare_shared(workers, a_buf, a);
        let base = crate::device::affinity::base();
        let results: Vec<Result<()>> = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers.len());
            for (i, worker) in workers.iter_mut().enumerate() {
                handles.push(scope.spawn(move || -> Result<()> {
                    crate::device::affinity::pin_current(i, base);
                    match packed {
                        Some(p) => worker.load_input(p)?,
                        None => worker.pack_input(m, k, a)?,
                    }
                    worker.pack_weights(k, b)?;
                    worker.compute(m, k)
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(Error::device("bf16/worker", 0)))
                })
                .collect()
        });
        if let Some(e) = results.into_iter().find_map(|r| r.err()) {
            return Err(e);
        }
        for worker in workers.iter() {
            for row in 0..m {
                c[row * n + worker.n0..][..worker.nsub]
                    .copy_from_slice(&worker.csub[row * worker.nsub..][..worker.nsub]);
            }
        }
        Ok(())
    }

    /// Per-call multi-core form, opening its own fds (the old `rocket_matmul_bf16_mt`).
    pub fn matmul_mt(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        nthreads: usize,
    ) -> Result<()> {
        let mut s = Bf16Stream::new(nthreads)?;
        s.matmul(m, k, n, a, b, c)
    }
}

impl Worker {
    fn load_input(&mut self, packed: &[u16]) -> Result<()> {
        let n = self.plan.input_elems();
        self.in_all.prep(0)?;
        {
            let bytes = self.in_all.as_mut_slice();
            let words = unsafe {
                std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u16>(), bytes.len() / 2)
            };
            words[..n].copy_from_slice(&packed[..n]);
        }
        self.in_all.fini()
    }

    fn pack_input(&mut self, _m: usize, k: usize, a: &[f32]) -> Result<()> {
        self.in_all.prep(0)?;
        {
            let bytes = self.in_all.as_mut_slice();
            let words = unsafe {
                std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u16>(), bytes.len() / 2)
            };
            if !self.in_zeroed {
                words.fill(0);
                self.in_zeroed = true;
            }
            for mi in 0..self.plan.nmt {
                let m0 = mi * self.plan.mt;
                let mtile = self.plan.m_tile(mi);
                for ki in 0..self.plan.nkt {
                    let k0 = ki * self.plan.kt;
                    let ktile = self.plan.k_tile(ki);
                    let off = (mi * self.plan.nkt + ki) * self.plan.in_slot;
                    layout::feat_scatter_bf16(
                        &mut words[off..off + self.plan.in_slot],
                        a,
                        k,
                        m0,
                        k0,
                        mtile,
                        ktile,
                    );
                }
            }
        }
        self.in_all.fini()
    }

    fn pack_weights(&mut self, k: usize, b: &[f32]) -> Result<()> {
        self.wt_all.prep(0)?;
        {
            let bytes = self.wt_all.as_mut_slice();
            let words = unsafe {
                std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u16>(), bytes.len() / 2)
            };
            if !self.wt_zeroed {
                words.fill(0);
                self.wt_zeroed = true;
            }
            for ni in 0..self.plan.nnt {
                let n0 = ni * self.plan.nt;
                let ntile = self.plan.n_tile(ni);
                for ki in 0..self.plan.nkt {
                    let k0 = ki * self.plan.kt;
                    let ktile = self.plan.k_tile(ki);
                    let off = (ni * self.plan.nkt + ki) * self.plan.wt_slot;
                    layout::wt_scatter_tile_bf16(
                        &mut words[off..off + self.plan.wt_slot],
                        b,
                        k,
                        self.n0 + n0,
                        k0,
                        ntile,
                        ktile,
                    );
                }
            }
        }
        self.wt_all.fini()
    }

    fn compute(&mut self, m: usize, k: usize) -> Result<()> {
        let plan = &self.plan;
        self.acc.clear();
        self.acc.resize(m * self.nsub, 0.0);
        let mut tasks: Vec<DrmRocketTask> = Vec::with_capacity(BATCH);
        let mut meta: Vec<(usize, usize, usize, usize, usize)> = Vec::with_capacity(BATCH);
        let mut ops = Vec::with_capacity(160);
        let mut done = 0usize;
        let total = plan.jobs();
        let mut nb = 0usize;

        for mi in 0..plan.nmt {
            let m0 = mi * plan.mt;
            let mtile = plan.m_tile(mi);
            for ni in 0..plan.nnt {
                let n0 = ni * plan.nt;
                let ntile = plan.n_tile(ni);
                for ki in 0..plan.nkt {
                    if nb == 0 {
                        self.regcmd.prep(0)?;
                    }
                    let ktile = plan.k_tile(ki);
                    let out_off = nb * plan.out_slot;
                    let input = self.in_all.dma_address()
                        + ((mi * plan.nkt + ki) * plan.in_slot * 2) as u64;
                    let weights = self.wt_all.dma_address()
                        + ((ni * plan.nkt + ki) * plan.wt_slot * 2) as u64;
                    let output = self.out_all.dma_address() + (out_off * 4) as u64;
                    let p = MatmulParams::new(
                        mtile as u32,
                        ktile as u32,
                        ntile as u32,
                        input as u32,
                        weights as u32,
                        output as u32,
                    );
                    ops.clear();
                    let count = emit(&p, &mut ops)?;
                    if ops.len() > RC_STRIDE {
                        return Err(Error::tiling("bf16/regcmd"));
                    }
                    {
                        let bytes = self.regcmd.as_mut_slice();
                        let base = nb * RC_STRIDE * 8;
                        for (j, op) in ops.iter().enumerate() {
                            bytes[base + j * 8..base + j * 8 + 8]
                                .copy_from_slice(&op.0.to_ne_bytes());
                        }
                    }
                    let regcmd = (self.regcmd.dma_address() + (nb * RC_STRIDE * 8) as u64) as u32;
                    tasks.push(DrmRocketTask {
                        regcmd,
                        regcmd_count: count as u32,
                    });
                    meta.push((m0, n0, mtile, ntile, out_off));
                    nb += 1;
                    done += 1;

                    if nb == BATCH || done == total {
                        self.regcmd.fini()?;
                        self.out_all.prep(0)?;
                        self.out_all.fini()?;
                        let in_h: [Handle; 3] = [
                            self.in_all.handle(),
                            self.wt_all.handle(),
                            self.regcmd.handle(),
                        ];
                        let out_h = [self.out_all.handle()];
                        self.guard.device().submit(&tasks, &in_h, &out_h, 0)?;
                        self.out_all.prep(wait_ns())?;
                        {
                            let bytes = self.out_all.as_slice();
                            let words = unsafe {
                                std::slice::from_raw_parts(
                                    bytes.as_ptr().cast::<f32>(),
                                    bytes.len() / 4,
                                )
                            };
                            for &(m0, n0, mtile, ntile, off) in &meta {
                                let slot = &words[off..off + plan.out_slot];
                                layout::detile_accum_bf16(
                                    &mut self.acc,
                                    self.nsub,
                                    slot,
                                    m0,
                                    n0,
                                    mtile,
                                    ntile,
                                );
                            }
                        }
                        self.out_all.fini()?;
                        nb = 0;
                        tasks.clear();
                        meta.clear();
                    }
                }
            }
        }
        let _ = k;
        self.csub.clear();
        self.csub.extend(self.acc.iter().map(|v| *v as f32));
        Ok(())
    }
}

fn prepare_shared<'a>(
    workers: &[Worker],
    buf: &'a mut Option<Vec<u16>>,
    a: &[f32],
) -> Option<&'a [u16]> {
    let p0 = &workers[0].plan;
    let shared = workers.len() > 1
        && !crate::knobs::present("ROCKET_NO_SHARED_PACK")
        && workers[1..].iter().all(|w| {
            w.plan.mt == p0.mt
                && w.plan.kt == p0.kt
                && w.plan.nmt == p0.nmt
                && w.plan.nkt == p0.nkt
                && w.plan.in_slot == p0.in_slot
        });
    if !shared {
        return None;
    }
    if buf.is_none() {
        let mut v = vec![0u16; p0.input_elems()];
        scatter_shared(p0, &mut v, a);
        *buf = Some(v);
    } else {
        let v = buf.as_mut().unwrap();
        v.fill(0);
        scatter_shared(p0, v, a);
    }
    let v = buf.as_ref().unwrap();
    Some(&v[..p0.input_elems()])
}

fn scatter_shared(plan: &Plan, dst: &mut [u16], a: &[f32]) {
    for mi in 0..plan.nmt {
        let m0 = mi * plan.mt;
        let mtile = plan.m_tile(mi);
        for ki in 0..plan.nkt {
            let k0 = ki * plan.kt;
            let ktile = plan.k_tile(ki);
            let off = (mi * plan.nkt + ki) * plan.in_slot;
            layout::feat_scatter_bf16(
                &mut dst[off..off + plan.in_slot],
                a,
                plan.k,
                m0,
                k0,
                mtile,
                ktile,
            );
        }
    }
}

fn plan_for(m: usize, k: usize, n: usize) -> Result<Plan> {
    let (mt, kt, nt) = plan_geom(&BF16, m, k, n, 0)?;
    Ok(Plan::derive(m, k, n, mt, kt, nt))
}

fn fanout_step(n: usize, threads: usize, align: usize) -> usize {
    let threads = threads.max(1);
    let align = align.max(1);
    n.div_ceil(threads).div_ceil(align) * align
}
