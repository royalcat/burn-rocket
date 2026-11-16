//! Compute placements of the vision tower: f32 on the CPU, the host emulation of the fp16 NPU
//! numerics, and (aarch64 `npu` builds) the NPU itself (`npu.rs`).

use super::super::attn::gemm_strided;
use super::super::math::matmul_nt;
use super::super::model::Shards;
use super::{Lin, widen_par};
use anyhow::{Result, bail};
use half::f16;
use rayon::prelude::*;

/// Where the tower's matmuls and attention run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TowerDevice {
    /// f32 on the CPU (gemm).
    Cpu,
    /// Host emulation of the NPU numerics: fp16 operands and results, f32 accumulation.
    Emu16,
    /// Resident fp16 weights + fp16 flash attention on the NPU (aarch64 `npu` builds).
    Npu,
}

impl TowerDevice {
    /// `ROCKET_CLEF_TOWER_DEVICE=cpu|emu16|npu` overrides `default`.
    pub fn resolve(default: TowerDevice) -> Result<TowerDevice> {
        match std::env::var("ROCKET_CLEF_TOWER_DEVICE").ok().as_deref() {
            None | Some("") => Ok(default),
            Some("cpu") => Ok(TowerDevice::Cpu),
            Some("emu16") => Ok(TowerDevice::Emu16),
            Some("npu") => Ok(TowerDevice::Npu),
            Some(other) => bail!("ROCKET_CLEF_TOWER_DEVICE={other}: expected cpu, emu16 or npu"),
        }
    }
}

/// A value left the fp16 range (or the driver reported a saturated score): the caller reruns on
/// the CPU in f32.
#[derive(Debug)]
pub struct Fp16Range(pub String);

impl std::fmt::Display for Fp16Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fp16 range exceeded in {}", self.0)
    }
}

impl std::error::Error for Fp16Range {}

/// One compute placement.
pub trait Exec: Send + Sync {
    fn label(&self) -> String;
    /// `x [m, k] . W^T -> [m, n]` for the linear's stored `[n, k]` weight (no bias).
    fn matmul(
        &self,
        lin: &Lin,
        shards: &Shards,
        x: &[f32],
        m: usize,
        threads: usize,
    ) -> Result<Vec<f32>>;
    /// Full (non-causal) attention inside every `(start, end)` segment of the `[P, heads*hd]`
    /// q/k/v rows, scale `hd^-0.5`; returns `[P, heads*hd]`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        segs: &[(usize, usize)],
        heads: usize,
        hd: usize,
        threads: usize,
    ) -> Result<Vec<f32>>;
    /// Drop scratch above `keep_bytes` (NPU placements).
    fn trim(&self, _keep_bytes: usize) {}
}

fn widened(lin: &Lin, shards: &Shards) -> Result<Vec<f32>> {
    let (file, info) = shards.get(&lin.name)?;
    Ok(widen_par(file, info))
}

pub struct CpuExec;

impl Exec for CpuExec {
    fn label(&self) -> String {
        "f32 on the CPU".into()
    }

    fn matmul(
        &self,
        lin: &Lin,
        shards: &Shards,
        x: &[f32],
        m: usize,
        threads: usize,
    ) -> Result<Vec<f32>> {
        let w = widened(lin, shards)?;
        Ok(matmul_nt(x, m, lin.k, &w, lin.n, threads))
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        segs: &[(usize, usize)],
        heads: usize,
        hd: usize,
        _threads: usize,
    ) -> Result<Vec<f32>> {
        Ok(full_attention(q, k, v, segs, heads, hd))
    }
}

/// Round to fp16 and back; an overflow to infinity is a [`Fp16Range`].
fn round16(v: &mut [f32], what: &str) -> Result<()> {
    let bad = std::sync::atomic::AtomicBool::new(false);
    v.par_chunks_mut(1 << 14).for_each(|c| {
        for x in c {
            let r = f16::from_f32(*x).to_f32();
            if r.is_infinite() && x.is_finite() {
                bad.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            *x = r;
        }
    });
    if bad.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(anyhow::Error::new(Fp16Range(what.to_string())));
    }
    Ok(())
}

/// The fp16 numerics of the NPU path on the host: operands and results rounded to fp16, f32
/// accumulation (the device accumulates in fp32 too), attention on fp16 q/k/v/out.
pub struct Emu16Exec;

impl Exec for Emu16Exec {
    fn label(&self) -> String {
        "fp16 emulation on the CPU".into()
    }

    fn matmul(
        &self,
        lin: &Lin,
        shards: &Shards,
        x: &[f32],
        m: usize,
        threads: usize,
    ) -> Result<Vec<f32>> {
        let mut w = widened(lin, shards)?;
        round16(&mut w, &lin.name)?;
        let mut xr = x.to_vec();
        round16(&mut xr, &lin.name)?;
        let mut y = matmul_nt(&xr, m, lin.k, &w, lin.n, threads);
        round16(&mut y, &lin.name)?;
        Ok(y)
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        segs: &[(usize, usize)],
        heads: usize,
        hd: usize,
        _threads: usize,
    ) -> Result<Vec<f32>> {
        let (mut q, mut k, mut v) = (q.to_vec(), k.to_vec(), v.to_vec());
        for (a, n) in [(&mut q, "q"), (&mut k, "k"), (&mut v, "v")] {
            round16(a, n)?;
        }
        let mut out = full_attention(&q, &k, &v, segs, heads, hd);
        round16(&mut out, "attention")?;
        Ok(out)
    }
}

/// Query rows per task.
const QB: usize = 128;

/// Non-causal attention inside every segment; `q/k/v [P, heads*hd]` -> `[P, heads*hd]`.
pub fn full_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    segs: &[(usize, usize)],
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let width = heads * hd;
    let p = q.len() / width;
    assert_eq!(k.len(), p * width);
    assert_eq!(v.len(), p * width);
    let scale = 1.0 / (hd as f32).sqrt();
    let tasks: Vec<(usize, usize, usize, usize)> = segs
        .iter()
        .flat_map(|&(s, e)| {
            (0..(e - s).div_ceil(QB))
                .flat_map(move |b| (0..heads).map(move |h| (s, e, s + b * QB, h)))
        })
        .collect();
    let parts: Vec<Vec<f32>> = tasks
        .par_iter()
        .map(|&(s, e, q0, h)| {
            let bq = QB.min(e - q0);
            let keys = e - s;
            let mut sc = vec![0f32; bq * keys];
            // SAFETY: q rows q0.. and k/v rows s..e at head h are inside the [P, width] buffers;
            // `sc` holds bq*keys, `ctx` bq*hd elements.
            unsafe {
                gemm_strided(
                    bq,
                    keys,
                    hd,
                    sc.as_mut_ptr(),
                    keys as isize,
                    q.as_ptr().add(q0 * width + h * hd),
                    width as isize,
                    k.as_ptr().add(s * width + h * hd),
                    1,
                    width as isize,
                    scale,
                );
            }
            for row in sc.chunks_mut(keys) {
                super::super::math::softmax_row(row);
            }
            let mut ctx = vec![0f32; bq * hd];
            // SAFETY: as above.
            unsafe {
                gemm_strided(
                    bq,
                    hd,
                    keys,
                    ctx.as_mut_ptr(),
                    hd as isize,
                    sc.as_ptr(),
                    keys as isize,
                    v.as_ptr().add(s * width + h * hd),
                    width as isize,
                    1,
                    1.0,
                );
            }
            ctx
        })
        .collect();
    let mut out = vec![0f32; p * width];
    for (&(_, _, q0, h), ctx) in tasks.iter().zip(&parts) {
        for (i, row) in ctx.chunks(hd).enumerate() {
            out[(q0 + i) * width + h * hd..][..hd].copy_from_slice(row);
        }
    }
    out
}

/// `src [n, heads*hd]` -> `[heads][n][hdp]` fp16, columns `hd..hdp` zero.
pub(crate) fn pad_heads(src: &[f32], n: usize, heads: usize, hd: usize, hdp: usize) -> Vec<f16> {
    let width = heads * hd;
    let mut dst = vec![f16::ZERO; heads * n * hdp];
    dst.par_chunks_mut(n * hdp).enumerate().for_each(|(h, dh)| {
        for (i, row) in dh.chunks_mut(hdp).enumerate() {
            for (d, &s) in row[..hd].iter_mut().zip(&src[i * width + h * hd..][..hd]) {
                *d = f16::from_f32(s);
            }
        }
    });
    dst
}

/// `src [n, heads*hd]` -> the per-head transpose `[heads][dvp][n]` fp16, rows `hd..dvp` zero (the
/// AV B-operand layout).
pub(crate) fn pad_heads_transposed(
    src: &[f32],
    n: usize,
    heads: usize,
    hd: usize,
    dvp: usize,
) -> Vec<f16> {
    let width = heads * hd;
    let mut dst = vec![f16::ZERO; heads * dvp * n];
    dst.par_chunks_mut(dvp * n).enumerate().for_each(|(h, dh)| {
        for i in 0..n {
            for (d, &s) in src[i * width + h * hd..][..hd].iter().enumerate() {
                dh[d * n + i] = f16::from_f32(s);
            }
        }
    });
    dst
}

/// `o16 [heads][n][dvp]` (the driver's output) -> `out [n, heads*hd]` f32, dropping the padded
/// columns `hd..dvp`.
pub(crate) fn unpack_padded(
    o16: &[f16],
    n: usize,
    heads: usize,
    hd: usize,
    dvp: usize,
    out: &mut [f32],
) {
    out.par_chunks_mut(heads * hd)
        .enumerate()
        .for_each(|(i, row)| {
            for h in 0..heads {
                let src = &o16[(h * n + i) * dvp..][..hd];
                for (d, x) in row[h * hd..(h + 1) * hd].iter_mut().zip(src) {
                    *d = x.to_f32();
                }
            }
        });
}

/// Build the placement for `device`. `lins` are the tower's linears in slot order (the NPU
/// placement packs them as resident weights; `Lin::slot` is the index into this list).
pub fn make_exec(
    device: TowerDevice,
    lins: &[&Lin],
    shards: &Shards,
    threads: usize,
) -> Result<Box<dyn Exec>> {
    let _ = (lins, shards, threads);
    match device {
        TowerDevice::Cpu => Ok(Box::new(CpuExec)),
        TowerDevice::Emu16 => Ok(Box::new(Emu16Exec)),
        TowerDevice::Npu => {
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            {
                Ok(Box::new(super::npu::NpuExec::pack(lins, shards, threads)?))
            }
            #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
            {
                bail!("this build has no NPU support (aarch64 + the `npu` feature)")
            }
        }
    }
}
