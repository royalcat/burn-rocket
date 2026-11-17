//! The NPU side of the Clef engine (aarch64 `npu` builds): `I8Context` based expert
//! workers (three contexts, one worker each, one expert per core) and the resident
//! attention projections.
//!
//! The matmuls are the group-wise W8A8 path of the driver (`I8Context::matmul_gw`): the
//! activation is quantized per row per group on the host (`int8::quantize_rows`), the NPU
//! computes exact int32 group partials, the host applies `a_scale * w_scale` and sums in
//! f32. `Int8Emu` (engine.rs) is the host twin of exactly this.

use super::engine::{
    AttnDevice, ClefStore, CpuDense, Dense, ExpertTimes, ExpertWorker, PackEngine, depth_from_env,
};
use super::int8::{f16_scales_to_f32, m_chunks, quantize_rows};
use super::model::{TextConfig, silu};
use super::pack::format::{Dims, ExpertRecord};
use anyhow::Result;
use burn_rocket::npu::matmul::{I8Context, I8Weight};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The pack's resident weights are packed with this nominal M (the canonical tiling makes
/// them valid for every M).
const PACK_M: usize = 128;

/// A device failure with the shape that triggered it; [`NpuFailure::raise`] turns it into
/// the typed panic payload the server maps (NOMEM -> 503, dead device -> exit).
#[derive(Debug)]
pub struct NpuFailure {
    pub op: &'static str,
    pub rc: i32,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl std::fmt::Display for NpuFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} failed (rc={}, m={} k={} n={})",
            self.op, self.rc, self.m, self.k, self.n
        )
    }
}

impl std::error::Error for NpuFailure {}

impl NpuFailure {
    pub fn raise(&self) -> ! {
        eprintln!("clef: NPU op {self}");
        std::panic::panic_any(burn_rocket::OpFailure {
            error: burn_rocket::Error {
                op: self.op,
                rc: self.rc,
            },
            m: self.m,
            k: self.k,
            n: self.n,
        })
    }
}

fn fail(e: burn_rocket::npu::Error, m: usize, k: usize, n: usize) -> anyhow::Error {
    anyhow::Error::new(NpuFailure {
        op: e.op(),
        rc: e.status().code(),
        m,
        k,
        n,
    })
}

/// `out[m, n] = x[m, k] . W^T` through `matmul_gw`, chunked and padded to shape classes.
#[allow(clippy::too_many_arguments)]
fn gw_matmul(
    ctx: &mut I8Context,
    w: &I8Weight,
    w_scales: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    parallel: bool,
    out: &mut [f32],
    t: &mut ExpertTimes,
) -> Result<()> {
    let ng = k / group;
    for (start, rows, mp) in m_chunks(m) {
        let t0 = Instant::now();
        let mut q = vec![0i8; mp * k];
        let mut sc = vec![1f32; mp * ng];
        quantize_rows(
            &x[start * k..(start + rows) * k],
            k,
            group,
            &mut q[..rows * k],
            &mut sc[..rows * ng],
            parallel,
        );
        t.quant += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let mut c = vec![0f32; mp * n];
        ctx.matmul_gw(mp, k, n, &q, &sc, w_scales, &mut c, w)
            .map_err(|e| fail(e, rows, k, n))?;
        t.npu += t0.elapsed().as_secs_f64();
        out[start * n..(start + rows) * n].copy_from_slice(&c[..rows * n]);
    }
    Ok(())
}

/// One NPU core's expert slot: a one-worker context with a resident gate|up weight
/// (`K=hidden, N=2*inter`) and a resident down weight, rewritten per expert.
pub struct NpuWorker {
    ctx: I8Context,
    gu: I8Weight,
    dn: I8Weight,
    dims: Dims,
    gu_s: Vec<f32>,
    dn_s: Vec<f32>,
}

impl NpuWorker {
    pub fn new(core: usize, dims: Dims) -> Result<Self> {
        let Dims {
            hidden: h,
            inter: i,
            group: g,
        } = dims;
        let mut ctx = I8Context::new_at(1, core).map_err(|e| fail(e, 0, h, 2 * i))?;
        let gu = ctx
            .pack_weight_gw(PACK_M, h, 2 * i, &vec![0i8; 2 * i * h], g)
            .map_err(|e| fail(e, PACK_M, h, 2 * i))?;
        let dn = ctx
            .pack_weight_gw(PACK_M, i, h, &vec![0i8; h * i], g)
            .map_err(|e| fail(e, PACK_M, i, h))?;
        Ok(Self {
            ctx,
            gu,
            dn,
            dims,
            gu_s: vec![0.0; 2 * i * (h / g)],
            dn_s: vec![0.0; h * (i / g)],
        })
    }
}

impl ExpertWorker for NpuWorker {
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
        self.ctx
            .rewrite_weight_gw(&mut self.gu, rec.gu_codes)
            .map_err(|e| fail(e, 0, h, 2 * i))?;
        self.ctx
            .rewrite_weight_gw(&mut self.dn, rec.down_codes)
            .map_err(|e| fail(e, 0, i, h))?;
        f16_scales_to_f32(rec.gu_scales, &mut self.gu_s);
        f16_scales_to_f32(rec.down_scales, &mut self.dn_s);
        t.rewrite += t0.elapsed().as_secs_f64();
        let mut gu = vec![0f32; m * 2 * i];
        gw_matmul(
            &mut self.ctx,
            &self.gu,
            &self.gu_s,
            x,
            m,
            h,
            2 * i,
            g,
            false,
            &mut gu,
            t,
        )?;
        let t0 = Instant::now();
        let mut act = vec![0f32; m * i];
        for (a, row) in act.chunks_mut(i).zip(gu.chunks(2 * i)) {
            for j in 0..i {
                a[j] = silu(row[j]) * row[i + j];
            }
        }
        t.glue += t0.elapsed().as_secs_f64();
        gw_matmul(
            &mut self.ctx,
            &self.dn,
            &self.dn_s,
            &act,
            m,
            i,
            h,
            g,
            false,
            out,
            t,
        )
    }

    fn trim(&mut self, keep_bytes: usize) {
        if self.ctx.scratch_bytes() > keep_bytes {
            self.ctx.clear_scratch();
        }
    }
}

struct LayerProj {
    qkv: I8Weight,
    qkv_s: Vec<f32>,
    o: I8Weight,
    o_s: Vec<f32>,
}

/// Attention projections resident on the NPU: per layer a fused q|k|v weight
/// (`K=hidden, N=(heads+2*kv)*d`) and `o` (`K=heads*d, N=hidden`), N-split over three
/// workers of one context (~0.9 GB of BOs for the 48 layers).
pub struct NpuDense {
    ctx: Mutex<I8Context>,
    layers: Vec<LayerProj>,
    cfg: TextConfig,
    group: usize,
}

impl NpuDense {
    pub fn new(pack: &dyn ClefStore, cfg: &TextConfig) -> Result<Self> {
        let t0 = Instant::now();
        let (h, d) = (cfg.hidden, cfg.head_dim);
        let g = pack.dims().group;
        let (qn, kn) = (cfg.heads * d, cfg.kv_heads * d);
        let mut ctx = I8Context::new_at(3, 0).map_err(|e| fail(e, 0, h, qn))?;
        let mut layers = Vec::with_capacity(cfg.layers);
        let mut bytes = 0usize;
        for l in 0..cfg.layers {
            let pre = format!("thinker.model.layers.{l}.self_attn.");
            let mut codes: Vec<i8> = Vec::with_capacity((qn + 2 * kn) * h);
            let mut qkv_s: Vec<f32> = Vec::new();
            for (name, rows) in [("q_proj", qn), ("k_proj", kn), ("v_proj", kn)] {
                let (c, s) = pack.resident_linear(&format!("{pre}{name}"), rows, h)?;
                codes.extend_from_slice(&c);
                qkv_s.extend_from_slice(&s);
            }
            let qkv = ctx
                .pack_weight_gw(PACK_M, h, qn + 2 * kn, &codes, g)
                .map_err(|e| fail(e, PACK_M, h, qn + 2 * kn))?;
            let (oc, o_s) = pack.resident_linear(&format!("{pre}o_proj"), h, qn)?;
            let o = ctx
                .pack_weight_gw(PACK_M, qn, h, &oc, g)
                .map_err(|e| fail(e, PACK_M, qn, h))?;
            bytes += qkv.bytes() + o.bytes();
            layers.push(LayerProj { qkv, qkv_s, o, o_s });
            pack.release_resident();
        }
        eprintln!(
            "clef: packed the attention projections on the NPU: {} layers, {:.0} MiB of BOs in {:.1}s",
            cfg.layers,
            bytes as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64()
        );
        Ok(Self {
            ctx: Mutex::new(ctx),
            layers,
            cfg: cfg.clone(),
            group: g,
        })
    }
}

impl Dense for NpuDense {
    fn label(&self) -> String {
        format!("int8 g{} W8A8 on the NPU (resident)", self.group)
    }

    fn qkv(&self, layer: usize, n: &[f32], s: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (h, d) = (self.cfg.hidden, self.cfg.head_dim);
        let (qn, kn) = (self.cfg.heads * d, self.cfg.kv_heads * d);
        let total = qn + 2 * kn;
        let lp = &self.layers[layer];
        let mut all = vec![0f32; s * total];
        let mut ctx = self.ctx.lock().unwrap_or_else(|e| e.into_inner());
        gw_matmul(
            &mut ctx,
            &lp.qkv,
            &lp.qkv_s,
            n,
            s,
            h,
            total,
            self.group,
            true,
            &mut all,
            &mut ExpertTimes::default(),
        )?;
        let (mut q, mut k, mut v) = (
            Vec::with_capacity(s * qn),
            Vec::with_capacity(s * kn),
            Vec::with_capacity(s * kn),
        );
        for row in all.chunks(total) {
            q.extend_from_slice(&row[..qn]);
            k.extend_from_slice(&row[qn..qn + kn]);
            v.extend_from_slice(&row[qn + kn..]);
        }
        Ok((q, k, v))
    }

    fn o_proj(&self, layer: usize, ctx_in: &[f32], s: usize) -> Result<Vec<f32>> {
        let (h, kk) = (self.cfg.hidden, self.cfg.heads * self.cfg.head_dim);
        let lp = &self.layers[layer];
        let mut out = vec![0f32; s * h];
        let mut ctx = self.ctx.lock().unwrap_or_else(|e| e.into_inner());
        gw_matmul(
            &mut ctx,
            &lp.o,
            &lp.o_s,
            ctx_in,
            s,
            kk,
            h,
            self.group,
            true,
            &mut out,
            &mut ExpertTimes::default(),
        )?;
        Ok(out)
    }

    fn trim(&self, keep_bytes: usize) {
        let mut ctx = self.ctx.lock().unwrap_or_else(|e| e.into_inner());
        if ctx.scratch_bytes() > keep_bytes {
            ctx.clear_scratch();
        }
    }
}

/// `ROCKET_CLEF_CONSUMERS` expert consumers (default 3 = one per NPU core).
fn consumers_from_env() -> usize {
    std::env::var("ROCKET_CLEF_CONSUMERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| (1..=6).contains(&n))
        .unwrap_or(3)
}

/// The NPU engine: three one-worker expert contexts + resident attention projections.
pub fn npu_engine(
    pack: Arc<dyn ClefStore>,
    cfg: TextConfig,
    threads: usize,
    attn: AttnDevice,
) -> Result<PackEngine> {
    let dims = pack.dims();
    let dense: Box<dyn Dense> = match attn {
        AttnDevice::Npu => Box::new(NpuDense::new(&*pack, &cfg)?),
        AttnDevice::Cpu => Box::new(CpuDense::new(pack.clone(), cfg.clone(), threads)),
    };
    let n = consumers_from_env();
    let mut workers: Vec<Box<dyn ExpertWorker>> = Vec::with_capacity(n);
    for core in 0..n {
        workers.push(Box::new(NpuWorker::new(core, dims)?));
    }
    Ok(PackEngine::new(
        format!(
            "int8 g{} on the NPU, streamed experts ({n} contexts x 1 worker)",
            dims.group
        ),
        cfg,
        pack,
        dense,
        workers,
        depth_from_env(),
        true,
    ))
}
