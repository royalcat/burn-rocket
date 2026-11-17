//! RoPE probe (P2c): the fused host kernel vs the composite tensor-op path vs an
//! on-NPU elementwise composition, at qwen3-embedding's attention shapes.
//!
//! Every decoder layer rotates q and k after their QK-norm. The app has two CPU
//! forms and this probe measures both against a possible NPU form:
//!
//! * `host-fused` — `burn_rocket::rope_apply`, the deployed one-pass rayon kernel
//!   (`src/host.rs::rope_row`): read both halves, two multiplies, one add, write.
//! * `host-composite` — `RopeCache::apply`, the flex tensor-op form used when the
//!   fused glue is off: 15 tensor ops per call (2 tables sliced/reshaped/cast,
//!   2 slices of the input, 4 multiplies, 1 subtract, 1 add, 1 `cat`).
//! * `npu-ew` — the same math through the driver's conv-main EW datapath
//!   (`rocket_npu::ops::activation::ew_mul`/`ew_add`): host narrow to f16, host
//!   rotate-half gather, `ew_mul(x, cos)`, `ew_mul(rot, sin)`, `ew_add`, host
//!   widen. The EW op has no broadcast, so the cos/sin tables are replicated to
//!   the full `[S, H, D]` element layout (built once per forward, inside the timed
//!   region), and the op tiles at `[1020, 32]` elements per submit — which is what
//!   makes the arm submit-bound.
//!
//! Timings are wall + process CPU (`getrusage`), interleaved inside each rep, and
//! reported as the median of the measured reps (the first rep warms up). Each arm
//! runs the whole model's RoPE work (`--layers` 28 layers x q and k), which is the
//! unit an integration decision applies to. `npu-ew` additionally reports its
//! internal split (tables / host glue / device) and the exact submit count
//! (`ceil(n / 32640)` per EW call).
//!
//! Usage (aarch64 NPU build, board):
//!   rope_bench [--tokens 82,2777] [--dtypes f32,bf16] [--layers 28] [--reps 3]
//!              [--skip-npu]
//!
//! `--skip-npu` runs the two host arms only (a board without the accel node, or
//! a quick CPU-only check).

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use burn_rocket::half::{bf16, f16};
use burn_rocket::npu::Device as NpuDevice;
use burn_rocket::npu::ops::activation;

#[path = "../util/rope.rs"]
mod rope;
use rope::RopeCache;

/// qwen3-embedding-0.6b's attention geometry.
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const LAYERS: usize = 28;
const THETA: f64 = 1_000_000.0;
/// The EW path's per-submit element cap: `[1020, 32]` rows (`ops::activation`).
const EW_TILE: usize = 1020 * 32;
/// The composite `RopeCache::apply` op count per call (documented in the report).
const COMPOSITE_OPS: usize = 15;

fn main() {
    let mut tokens = vec![82usize, 2777];
    let mut dtypes = vec![DType::F32, DType::BF16];
    let mut layers = LAYERS;
    let mut reps = 3usize;
    let mut want_npu = true;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--tokens" => {
                i += 1;
                tokens = args[i]
                    .split(',')
                    .map(|v| v.parse().expect("tokens"))
                    .collect();
            }
            "--dtypes" => {
                i += 1;
                dtypes = args[i]
                    .split(',')
                    .map(|v| match v {
                        "f32" => DType::F32,
                        "f16" => DType::F16,
                        "bf16" => DType::BF16,
                        other => panic!("unknown dtype {other}"),
                    })
                    .collect();
            }
            "--layers" => {
                i += 1;
                layers = args[i].parse().expect("layers");
            }
            "--reps" => {
                i += 1;
                reps = args[i].parse().expect("reps");
            }
            "--skip-npu" => want_npu = false,
            other => panic!("unexpected argument {other}"),
        }
        i += 1;
    }

    let dev_flex = Device::flex();
    let dev_npu = if want_npu {
        match NpuDevice::open() {
            Ok(d) => Some(d),
            Err(e) => {
                println!("NPU device unavailable ({e}); running the host arms only");
                None
            }
        }
    } else {
        None
    };

    println!(
        "rope probe: qwen3 geometry (heads {HEADS}, kv heads {KV_HEADS}, head_dim {HEAD_DIM}), \
         {} layers, {} reps (median), dtypes {:?}, npu {}",
        layers,
        reps,
        dtypes,
        if dev_npu.is_some() { "on" } else { "off" }
    );
    println!("start RssAnon {} MiB", rss_anon_mib());

    for dtype in dtypes {
        for &m in &tokens {
            run_case(m, dtype, layers, reps, &dev_flex, dev_npu.as_ref());
        }
    }

    println!("\nend RssAnon {} MiB", rss_anon_mib());
}

/// `(user + sys)` CPU seconds of this process.
fn cpu_s() -> f64 {
    // SAFETY: getrusage fills the zeroed struct on success, which is the only case read.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return 0.0;
    }
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// Current `RssAnon` from `/proc/self/status` (MiB).
fn rss_anon_mib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("RssAnon:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
            return kb / 1024;
        }
    }
    0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    HostFused,
    HostComposite,
    NpuEw,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::HostFused => "host-fused",
            Arm::HostComposite => "host-composite",
            Arm::NpuEw => "npu-ew",
        }
    }
}

/// One arm's internal time split (zero for the host arms).
#[derive(Clone, Copy, Debug, Default)]
struct Split {
    /// cos/sin tables (the NPU arm's replicated `[S, H, D]` f16 copies).
    tables: f64,
    /// Host narrow + rotate-half gather + widen, i.e. everything but the EW ops.
    glue: f64,
    /// The EW ops themselves (submit + fence per tile, including their BO churn).
    device: f64,
}

struct ArmOut {
    q: Tensor<4>,
    k: Tensor<4>,
    split: Split,
}

/// Reused f16 buffers for the NPU arm (an integration would keep them per length).
#[derive(Default)]
struct Scratch {
    x: Vec<f16>,
    rot: Vec<f16>,
    m1: Vec<f16>,
    m2: Vec<f16>,
    out: Vec<f16>,
}

impl Scratch {
    fn ensure(&mut self, n: usize) {
        // `resize` keeps the capacity; the fill value is irrelevant (overwritten).
        for b in [
            &mut self.x,
            &mut self.rot,
            &mut self.m1,
            &mut self.m2,
            &mut self.out,
        ] {
            b.resize(n, f16::ZERO);
        }
    }
}

/// The NPU arm's cos/sin tables, replicated to the full element layout of q and k
/// (the EW op takes two equal-length flat arrays: no broadcast).
struct Tables {
    cos_q: Vec<f16>,
    sin_q: Vec<f16>,
    cos_k: Vec<f16>,
    sin_k: Vec<f16>,
}

impl Tables {
    /// Built once per forward (the length is per request), like the app builds its
    /// `RopeCache` once per process.
    fn build(m: usize, d: usize) -> Self {
        let half = d / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / THETA.powf(2.0 * i as f64 / d as f64)) as f32)
            .collect();
        let mut cos_q = vec![f16::ZERO; m * HEADS * d];
        let mut sin_q = vec![f16::ZERO; m * HEADS * d];
        let mut cos_k = vec![f16::ZERO; m * KV_HEADS * d];
        let mut sin_k = vec![f16::ZERO; m * KV_HEADS * d];
        let mut c = vec![f16::ZERO; half];
        let mut s = vec![f16::ZERO; half];
        for pos in 0..m {
            for j in 0..half {
                let angle = pos as f32 * inv_freq[j];
                c[j] = f16::from_f32(angle.cos());
                s[j] = f16::from_f32(angle.sin());
            }
            for h in 0..HEADS {
                let base = (pos * HEADS + h) * d;
                let (cq1, cq2) = cos_q[base..base + d].split_at_mut(half);
                cq1.copy_from_slice(&c);
                cq2.copy_from_slice(&c);
                let (sq1, sq2) = sin_q[base..base + d].split_at_mut(half);
                sq1.copy_from_slice(&s);
                sq2.copy_from_slice(&s);
            }
            for h in 0..KV_HEADS {
                let base = (pos * KV_HEADS + h) * d;
                let (ck1, ck2) = cos_k[base..base + d].split_at_mut(half);
                ck1.copy_from_slice(&c);
                ck2.copy_from_slice(&c);
                let (sk1, sk2) = sin_k[base..base + d].split_at_mut(half);
                sk1.copy_from_slice(&s);
                sk2.copy_from_slice(&s);
            }
        }
        Self {
            cos_q,
            sin_q,
            cos_k,
            sin_k,
        }
    }
}

/// The NPU arm for one tensor: host narrow + rotate-half gather, three EW ops
/// (`x*cos`, `rot*sin`, `+`), host widen. `x` is `[1, m, heads, d]` row-major and
/// `cos`/`sin` are the same layout. Returns the f32 values plus the time split.
fn npu_rope(
    dev: &NpuDevice,
    x: &[f32],
    cos: &[f16],
    sin: &[f16],
    d: usize,
    sc: &mut Scratch,
) -> (Vec<f32>, Split) {
    let n = x.len();
    let half = d / 2;
    sc.ensure(n);
    let mut split = Split::default();

    // Host glue: narrow + `rot = [-x2, x1]` (the rotate-half the kernel uses).
    let t = Instant::now();
    for (d, &v) in sc.x.iter_mut().zip(x) {
        *d = f16::from_f32(v);
    }
    for row in 0..n / d {
        let base = row * d;
        for j in 0..half {
            sc.rot[base + j] = -sc.x[base + half + j];
            sc.rot[base + half + j] = sc.x[base + j];
        }
    }
    split.glue += t.elapsed().as_secs_f64();

    // Device: three ew calls, each tiling the flat array at 32640 elements/submit.
    let t = Instant::now();
    activation::ew_mul(dev, &sc.x, cos, &mut sc.m1).expect("ew_mul x*cos");
    activation::ew_mul(dev, &sc.rot, sin, &mut sc.m2).expect("ew_mul rot*sin");
    activation::ew_add(dev, &sc.m1, &sc.m2, &mut sc.out).expect("ew_add");
    split.device += t.elapsed().as_secs_f64();

    // Host glue: widen back to f32 for the tensor the next op consumes.
    let t = Instant::now();
    let out: Vec<f32> = sc.out.iter().map(|v| v.to_f32()).collect();
    split.glue += t.elapsed().as_secs_f64();
    (out, split)
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    arm: Arm,
    q0: &Tensor<4>,
    k0: &Tensor<4>,
    cache: &RopeCache,
    m: usize,
    layers: usize,
    device: &Device,
    dev_npu: Option<&NpuDevice>,
    sc: &mut Scratch,
) -> ArmOut {
    let mut split = Split::default();
    match arm {
        Arm::HostFused => {
            let mut q = q0.clone();
            let mut k = k0.clone();
            for _ in 0..layers {
                // A fresh projection per layer: the same work, one rotation per input.
                q = burn_rocket::rope_apply(q0.clone(), cache.cos_rows(0, m), cache.sin_rows(0, m));
                k = burn_rocket::rope_apply(k0.clone(), cache.cos_rows(0, m), cache.sin_rows(0, m));
            }
            ArmOut { q, k, split }
        }
        Arm::HostComposite => {
            let mut q = q0.clone();
            let mut k = k0.clone();
            for _ in 0..layers {
                q = cache.apply(q0.clone(), 0);
                k = cache.apply(k0.clone(), 0);
            }
            ArmOut { q, k, split }
        }
        Arm::NpuEw => {
            let dev = dev_npu.expect("npu device");
            let t = Instant::now();
            let tables = Tables::build(m, HEAD_DIM);
            split.tables = t.elapsed().as_secs_f64();
            let dtype = q0.dtype();
            let mut q = q0.clone();
            let mut k = k0.clone();
            for _ in 0..layers {
                let t = Instant::now();
                let qv = to_f32_vec(q0.clone());
                let kv = to_f32_vec(k0.clone());
                split.glue += t.elapsed().as_secs_f64();

                let (qo, s_q) = npu_rope(dev, &qv, &tables.cos_q, &tables.sin_q, HEAD_DIM, sc);
                let (ko, s_k) = npu_rope(dev, &kv, &tables.cos_k, &tables.sin_k, HEAD_DIM, sc);
                split.glue += s_q.glue + s_k.glue;
                split.device += s_q.device + s_k.device;

                let t = Instant::now();
                q = from_f32_vec(qo, [1, m, HEADS, HEAD_DIM], dtype, device);
                k = from_f32_vec(ko, [1, m, KV_HEADS, HEAD_DIM], dtype, device);
                split.glue += t.elapsed().as_secs_f64();
            }
            ArmOut { q, k, split }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    m: usize,
    dtype: DType,
    layers: usize,
    reps: usize,
    dev_flex: &Device,
    dev_npu: Option<&NpuDevice>,
) {
    println!(
        "\n== M {m} tokens, dtype {dtype:?}, {layers} layers (q {HEADS} heads + k {KV_HEADS} \
         heads x {HEAD_DIM}), {reps} measured reps =="
    );
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    };
    // qwen3's q/k arrive post-QK-norm (unit-rms rows); [-0.5, 0.5) is the right order.
    let qv: Vec<f32> = (0..m * HEADS * HEAD_DIM).map(|_| 0.5 * rnd()).collect();
    let kv: Vec<f32> = (0..m * KV_HEADS * HEAD_DIM).map(|_| 0.5 * rnd()).collect();
    let q0 = from_f32_vec(qv, [1, m, HEADS, HEAD_DIM], dtype, dev_flex);
    let k0 = from_f32_vec(kv, [1, m, KV_HEADS, HEAD_DIM], dtype, dev_flex);
    let cache = RopeCache::new(m, HEAD_DIM, THETA, dtype, dev_flex);

    let mut arms = vec![Arm::HostFused, Arm::HostComposite];
    if dev_npu.is_some() {
        arms.push(Arm::NpuEw);
    }

    let mut sc = Scratch::default();
    let mut walls: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut cpus: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut last: Vec<ArmOut> = Vec::new();
    for rep in 0..=reps {
        let mut outs = Vec::with_capacity(arms.len());
        for (ai, &arm) in arms.iter().enumerate() {
            let c0 = cpu_s();
            let t0 = Instant::now();
            let out = run_arm(arm, &q0, &k0, &cache, m, layers, dev_flex, dev_npu, &mut sc);
            dev_flex.sync().expect("flex sync");
            let wall = t0.elapsed().as_secs_f64();
            let cpu = cpu_s() - c0;
            if rep > 0 {
                walls[ai].push(wall);
                cpus[ai].push(cpu);
            }
            outs.push(out);
        }
        last = outs;
    }

    let base_wall = median(walls[0].clone());
    let base_cpu = median(cpus[0].clone());
    for (ai, &arm) in arms.iter().enumerate() {
        let wall = median(walls[ai].clone());
        let cpu = median(cpus[ai].clone());
        let wmin = walls[ai].iter().copied().fold(f64::INFINITY, f64::min);
        let wmax = walls[ai].iter().copied().fold(0.0f64, f64::max);
        let split = last[ai].split;
        let split_note = if arm == Arm::NpuEw {
            format!(
                "   (tables {:.3} s, glue {:.3} s, device {:.3} s)",
                split.tables, split.glue, split.device
            )
        } else {
            String::new()
        };
        println!(
            "{:<15} {:8.3} s wall {:.2}x   {:8.3} s CPU {:.2}x   {:7.1} ms/layer   \
             spread {:.3}-{:.3} s{}",
            arm.name(),
            wall,
            wall / base_wall,
            cpu,
            if base_cpu > 0.0 { cpu / base_cpu } else { 0.0 },
            wall * 1e3 / layers as f64,
            wmin,
            wmax,
            split_note
        );
    }

    // Numerics against the fused host kernel (the deployed path), where the NPU
    // arm's only error is the f16 narrowing of x/tables and the f16 arithmetic.
    let qref = to_f32_vec(last[0].q.clone());
    let kref = to_f32_vec(last[0].k.clone());
    for (ai, &arm) in arms.iter().enumerate().skip(1) {
        let q = to_f32_vec(last[ai].q.clone());
        let k = to_f32_vec(last[ai].k.clone());
        let (qc, qm, qb) = compare(&q, &qref);
        let (kc, km, kb) = compare(&k, &kref);
        println!(
            "{:<15} vs host-fused: q cos {qc:.9} max|d| {qm:.2e} diff_f16 {qb}/{}   \
             k cos {kc:.9} max|d| {km:.2e} diff_f16 {kb}/{}",
            arm.name(),
            q.len(),
            k.len()
        );
    }

    if let Some(dev) = dev_npu {
        // One EW call on the full q layout: the submit-bound unit the arm multiplies.
        let n = m * HEADS * HEAD_DIM;
        let s = ew_single(dev, n);
        println!(
            "single ew_mul: {n} elem, {} submits, {:.3} s ({:.2} GB/s of the 2x2-byte move)",
            n.div_ceil(EW_TILE),
            s,
            (2 * n * 2) as f64 / s / 1e9
        );
    }

    // Operation counts (per forward), the point of the exercise.
    let q_elems = m * HEADS * HEAD_DIM;
    let k_elems = m * KV_HEADS * HEAD_DIM;
    let q_tiles = q_elems.div_ceil(EW_TILE);
    let k_tiles = k_elems.div_ceil(EW_TILE);
    println!(
        "ops/forward: host-fused {n} passes over q+k (1 kernel each); host-composite {nc} flex \
         ops; npu-ew {ns} EW calls / {nsub} submits ({qt} q tiles + {kt} k tiles per call)",
        n = 2 * layers,
        nc = 2 * layers * COMPOSITE_OPS,
        ns = 6 * layers,
        nsub = 3 * layers * (q_tiles + k_tiles),
        qt = q_tiles,
        kt = k_tiles
    );
}

/// One full-length EW call as the arm pays it (the diagnostic for the submit cost).
fn ew_single(dev: &NpuDevice, n: usize) -> f64 {
    let a = vec![f16::ONE; n];
    let b = vec![f16::ONE; n];
    let mut o = vec![f16::ZERO; n];
    let t = Instant::now();
    activation::ew_mul(dev, &a, &b, &mut o).expect("ew_mul");
    t.elapsed().as_secs_f64()
}

/// `(cosine, max|d|, differing f16 bit patterns)` of two f32 vectors.
fn compare(a: &[f32], b: &[f32]) -> (f64, f32, usize) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxd, mut bits) = (0.0f64, 0.0f64, 0.0f64, 0.0f32, 0usize);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
        maxd = maxd.max((x - y).abs());
        bits += usize::from(f16::from_f32(x).to_bits() != f16::from_f32(y).to_bits());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxd, bits)
}

fn from_f32_vec(v: Vec<f32>, dims: [usize; 4], dtype: DType, device: &Device) -> Tensor<4> {
    match dtype {
        DType::F32 => Tensor::from_data(TensorData::new(v, dims), device),
        DType::F16 => {
            let d: Vec<f16> = v.into_iter().map(f16::from_f32).collect();
            Tensor::from_data(TensorData::new(d, dims), device)
        }
        DType::BF16 => {
            let d: Vec<bf16> = v.into_iter().map(bf16::from_f32).collect();
            Tensor::from_data(TensorData::new(d, dims), device)
        }
        other => panic!("unsupported dtype {other:?}"),
    }
}

fn to_f32_vec(t: Tensor<4>) -> Vec<f32> {
    match t.dtype() {
        DType::F32 => t.into_data().try_to_vec().expect("f32 data"),
        DType::F16 => t
            .into_data()
            .try_to_vec::<f16>()
            .expect("f16 data")
            .into_iter()
            .map(f16::to_f32)
            .collect(),
        DType::BF16 => t
            .into_data()
            .try_to_vec::<bf16>()
            .expect("bf16 data")
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
        other => panic!("unsupported dtype {other:?}"),
    }
}
