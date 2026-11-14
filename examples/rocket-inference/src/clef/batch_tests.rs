//! Phase 7: records batched through one thinker pass equal the same records run alone,
//! bit for bit, on a small synthetic model (CPU f32 path and the host emulation of the
//! W8A8 streaming engine), including the row cap, the error isolation and the expert
//! read counts.

use super::engine::{Dense, EmuWorker, ExpertReader, ExpertWorker, PackEngine};
use super::int8::{emu_matmul_gw, quantize_rows};
use super::model::{
    Backbone, ExpertSource, ExpertWeights, ForwardOpts, LayerWeights, TextConfig, WeightSource,
    plan_batches,
};
use super::pack::f32_to_f16_bits;
use super::pack::format::{Dims, ExpertRecordMut};
use super::pack::reader::AlignedBuf;
use anyhow::Result;
use std::sync::Arc;

const DIMS: Dims = Dims {
    hidden: 64,
    inter: 32,
    group: 32,
};
const VOCAB: usize = 50;

fn cfg() -> TextConfig {
    TextConfig {
        hidden: DIMS.hidden,
        layers: 3,
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        experts: 8,
        top_k: 2,
        inter: DIMS.inter,
        eps: 1e-6,
        theta: 1e6,
        norm_topk: true,
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
    fn one_plus(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| 1.0 + self.next() * scale).collect()
    }
}

/// f32 weights for every part of the mini model.
struct Mini {
    cfg: TextConfig,
    layers: Vec<LayerWeights>,
    experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    embed: Vec<f32>,
    norm: Vec<f32>,
}

impl Mini {
    fn new() -> Self {
        let c = cfg();
        let (h, d, i) = (c.hidden, c.head_dim, c.inter);
        let mut r = Lcg(7);
        let layers = (0..c.layers)
            .map(|_| LayerWeights {
                input_norm: r.one_plus(h, 0.1),
                post_norm: r.one_plus(h, 0.1),
                q: r.vec(c.heads * d * h, 0.3),
                k: r.vec(c.kv_heads * d * h, 0.3),
                v: r.vec(c.kv_heads * d * h, 0.3),
                o: r.vec(h * c.heads * d, 0.3),
                q_norm: r.one_plus(d, 0.1),
                k_norm: r.one_plus(d, 0.1),
                router: r.vec(c.experts * h, 0.8),
            })
            .collect();
        let experts = (0..c.layers * c.experts)
            .map(|_| (r.vec(i * h, 0.3), r.vec(i * h, 0.3), r.vec(h * i, 0.3)))
            .collect();
        Self {
            embed: r.vec(VOCAB * h, 1.0),
            norm: r.one_plus(h, 0.1),
            layers,
            experts,
            cfg: c,
        }
    }
}

struct MiniSource(Arc<Mini>);

impl ExpertSource for MiniSource {
    fn expert(&self, layer: usize, e: usize) -> Result<ExpertWeights> {
        let (g, u, d) = &self.0.experts[layer * self.0.cfg.experts + e];
        Ok(ExpertWeights {
            gate: g.clone(),
            up: u.clone(),
            down: d.clone(),
        })
    }
}

impl WeightSource for MiniSource {
    fn layer_weights(&self, l: usize) -> Result<LayerWeights> {
        Ok(self.0.layers[l].clone())
    }
    fn experts(&self) -> &dyn ExpertSource {
        self
    }
    fn embed_row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        let h = self.0.cfg.hidden;
        out.copy_from_slice(&self.0.embed[id as usize * h..(id as usize + 1) * h]);
        Ok(())
    }
    fn final_norm(&self) -> Result<Vec<f32>> {
        Ok(self.0.norm.clone())
    }
    fn release(&self) {}
}

fn cpu_backbone(m: &Arc<Mini>) -> Backbone {
    Backbone::with_source(m.cfg.clone(), Box::new(MiniSource(m.clone())))
}

// ---- the emulated W8A8 engine over in-memory int8 experts

struct MemReader {
    layers: usize,
    recs: Vec<Vec<u8>>,
}

impl MemReader {
    fn new(layers: usize, experts: usize) -> Arc<Self> {
        let l = DIMS.layout();
        let recs = (0..layers * experts)
            .map(|i| {
                let mut region = vec![0u8; l.stride];
                let mut r = Lcg(0x1234 + i as u64);
                let rec = ExpertRecordMut::split(DIMS, &mut region);
                for c in rec.gu_codes.iter_mut().chain(rec.down_codes.iter_mut()) {
                    *c = (r.next() * 100.0) as i8;
                }
                for sc in rec
                    .gu_scales
                    .chunks_mut(2)
                    .chain(rec.down_scales.chunks_mut(2))
                {
                    let bits = f32_to_f16_bits(0.005 + 0.02 * r.next().abs());
                    sc.copy_from_slice(&bits.to_le_bytes());
                }
                region
            })
            .collect();
        Arc::new(Self { layers, recs })
    }
}

impl ExpertReader for MemReader {
    fn dims(&self) -> Dims {
        DIMS
    }
    fn layers(&self) -> usize {
        self.layers
    }
    fn experts(&self) -> usize {
        cfg().experts
    }
    fn new_buffer(&self) -> AlignedBuf {
        AlignedBuf::new(DIMS.layout().stride)
    }
    fn read(&self, layer: usize, expert: usize, buf: &mut AlignedBuf) -> Result<()> {
        let r = &self.recs[layer * cfg().experts + expert];
        buf.as_mut_slice()[..r.len()].copy_from_slice(r);
        Ok(())
    }
}

/// codes, scales, n, k
type Proj = (Vec<i8>, Vec<f32>, usize, usize);

/// int8 q|k|v|o projections (per-channel-group scales) through the emulated matmul.
struct MiniDense {
    /// per layer: (codes, scales) of q, k, v, o
    w: Vec<[Proj; 4]>,
}

impl MiniDense {
    fn new(m: &Mini) -> Self {
        let c = &m.cfg;
        let g = DIMS.group;
        let quant = |w: &[f32], n: usize, k: usize| {
            let (mut q, mut s) = (vec![0i8; n * k], vec![0f32; n * k / g]);
            quantize_rows(w, k, g, &mut q, &mut s, false);
            (q, s, n, k)
        };
        let (h, d) = (c.hidden, c.head_dim);
        Self {
            w: m.layers
                .iter()
                .map(|l| {
                    [
                        quant(&l.q, c.heads * d, h),
                        quant(&l.k, c.kv_heads * d, h),
                        quant(&l.v, c.kv_heads * d, h),
                        quant(&l.o, h, c.heads * d),
                    ]
                })
                .collect(),
        }
    }

    fn mm(&self, layer: usize, which: usize, x: &[f32], s: usize) -> Vec<f32> {
        let (codes, scales, n, k) = &self.w[layer][which];
        let g = DIMS.group;
        let (mut aq, mut a_s) = (vec![0i8; s * k], vec![0f32; s * k / g]);
        quantize_rows(x, *k, g, &mut aq, &mut a_s, true);
        let mut out = vec![0f32; s * n];
        emu_matmul_gw(&aq, &a_s, s, *k, codes, scales, *n, g, &mut out, true);
        out
    }
}

impl Dense for MiniDense {
    fn label(&self) -> String {
        "mini int8".into()
    }
    fn qkv(&self, layer: usize, n: &[f32], s: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        Ok((
            self.mm(layer, 0, n, s),
            self.mm(layer, 1, n, s),
            self.mm(layer, 2, n, s),
        ))
    }
    fn o_proj(&self, layer: usize, ctx: &[f32], s: usize) -> Result<Vec<f32>> {
        Ok(self.mm(layer, 3, ctx, s))
    }
}

fn emu_backbone(m: &Arc<Mini>, consumers: usize) -> Backbone {
    let mut bb = cpu_backbone(m);
    let workers: Vec<Box<dyn ExpertWorker>> = (0..consumers)
        .map(|_| Box::new(EmuWorker::new(DIMS)) as Box<dyn ExpertWorker>)
        .collect();
    bb.set_engine(Box::new(PackEngine::new(
        "mini emu".into(),
        m.cfg.clone(),
        MemReader::new(m.cfg.layers, m.cfg.experts),
        Box::new(MiniDense::new(m)),
        workers,
        4,
        true,
    )));
    bb
}

// ---- helpers

fn record(seed: u64, len: usize) -> Vec<u32> {
    let mut r = Lcg(seed);
    (0..len)
        .map(|_| ((r.next() * 0.5 + 0.5) * (VOCAB - 1) as f32) as u32)
        .collect()
}

fn opts() -> ForwardOpts {
    ForwardOpts {
        threads: 4,
        release_pages: false,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Run `records` alone and as one batch (with the row cap) and require identical bits.
/// Returns (experts read by the singles, experts read by the batch passes).
fn assert_batch_equals_singles(bb: &Backbone, records: &[Vec<u32>], cap: usize) -> (usize, usize) {
    let mut singles = Vec::new();
    let mut single_experts = 0;
    for r in records {
        singles.push(bb.forward(r, &opts()).unwrap());
        single_experts += bb.last_phases().map_or(0, |p| p.experts);
    }
    let refs: Vec<&[u32]> = records.iter().map(Vec::as_slice).collect();
    let fwd = bb.forward_batch(&refs, &opts(), cap).unwrap();
    assert_eq!(fwd.outs.len(), records.len());
    for (i, (want, got)) in singles.iter().zip(&fwd.outs).enumerate() {
        let got = got.as_ref().unwrap();
        assert_eq!(
            bits(&got.final_normed),
            bits(&want.final_normed),
            "record {i} (len {}): final_normed differs",
            records[i].len()
        );
        assert_eq!(
            bits(&got.final_hidden),
            bits(&want.final_hidden),
            "record {i}: final_hidden differs"
        );
    }
    let batch_experts = fwd
        .chunks
        .iter()
        .map(|c| c.phases.as_ref().map_or(0, |p| p.experts))
        .sum();
    (single_experts, batch_experts)
}

fn mixed_records() -> Vec<Vec<u32>> {
    // different lengths, a one-token record and a duplicate
    vec![
        record(1, 7),
        record(2, 1),
        record(3, 23),
        record(4, 40),
        record(1, 7),
    ]
}

/// The plain f32 CPU path (`ROCKET`-less, no engine) is NOT bit-identical under batching:
/// the expert GEMMs run `gemm` on `m = rows of the expert's bucket`, and the `gemm`
/// crate's microkernel/blocking choice (hence the f32 summation order) depends on `m`.
/// Everything else (attention, router, combine order) is per record. The difference is
/// float-rounding noise; this pins that it stays at that level.
#[test]
fn the_cpu_f32_path_batches_to_within_float_rounding() {
    let m = Arc::new(Mini::new());
    let bb = cpu_backbone(&m);
    let records = mixed_records();
    let refs: Vec<&[u32]> = records.iter().map(Vec::as_slice).collect();
    let fwd = bb.forward_batch(&refs, &opts(), 0).unwrap();
    let mut worst = 0f32;
    for (r, got) in records.iter().zip(&fwd.outs) {
        let want = bb.forward(r, &opts()).unwrap();
        let got = got.as_ref().unwrap();
        let peak = want.final_normed.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let max = want
            .final_normed
            .iter()
            .zip(&got.final_normed)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        worst = worst.max(max / peak);
    }
    eprintln!("cpu f32 batch vs single: worst max|d|/peak = {worst:.2e}");
    assert!(worst < 1e-4, "{worst}");
}

#[test]
fn the_emulated_engine_batches_bit_identically_and_reads_each_expert_once() {
    let m = Arc::new(Mini::new());
    let bb = emu_backbone(&m, 3);
    let (singles, batch) = assert_batch_equals_singles(&bb, &mixed_records(), 0);
    // one pass: at most every expert of every layer once; the singles pay per record
    assert!(batch <= m.cfg.layers * m.cfg.experts, "batch read {batch}");
    assert!(
        batch > 0 && singles > batch,
        "singles {singles} vs batch {batch}"
    );
    eprintln!("expert reads: singles {singles}, one batch {batch}");
}

#[test]
fn the_row_cap_chunks_records_without_changing_a_bit() {
    let m = Arc::new(Mini::new());
    let bb = emu_backbone(&m, 2);
    let records = mixed_records(); // 7 1 23 40 7
    let refs: Vec<&[u32]> = records.iter().map(Vec::as_slice).collect();
    let fwd = bb.forward_batch(&refs, &opts(), 30).unwrap();
    let shape: Vec<(usize, usize)> = fwd.chunks.iter().map(|c| (c.records, c.tokens)).collect();
    // [7 1] (31 > 30 with 23), [23], [40 alone, over the cap], [7]
    assert_eq!(shape, [(2, 8), (1, 23), (1, 40), (1, 7)]);
    assert_batch_equals_singles(&bb, &records, 30);
    assert_batch_equals_singles(&bb, &records, 64);
}

#[test]
fn plan_batches_is_greedy_ordered_and_lets_a_big_record_run_alone() {
    assert_eq!(
        plan_batches(&[], 10, 8),
        Vec::<std::ops::Range<usize>>::new()
    );
    assert_eq!(plan_batches(&[3, 3, 3], 10, 8), vec![0..3]);
    assert_eq!(plan_batches(&[4, 4, 4], 10, 8), [0..2, 2..3]);
    assert_eq!(plan_batches(&[3, 25, 3], 10, 8), [0..1, 1..2, 2..3]);
    assert_eq!(plan_batches(&[1, 1, 1, 1, 1], 100, 2), [0..2, 2..4, 4..5]);
    assert_eq!(plan_batches(&[50, 50], 0, 8), vec![0..2]); // 0 = no row cap
}

#[test]
fn an_empty_record_fails_alone() {
    let m = Arc::new(Mini::new());
    let bb = emu_backbone(&m, 2);
    let (a, b): (Vec<u32>, Vec<u32>) = (record(5, 9), record(6, 4));
    let refs: [&[u32]; 3] = [&a, &[], &b];
    let fwd = bb.forward_batch(&refs, &opts(), 0).unwrap();
    assert!(
        fwd.outs[1]
            .as_ref()
            .is_err_and(|e| e.to_string().contains("empty"))
    );
    for (i, rec) in [(0usize, &a), (2, &b)] {
        let want = bb.forward(rec, &opts()).unwrap();
        assert_eq!(
            bits(&fwd.outs[i].as_ref().unwrap().final_normed),
            bits(&want.final_normed)
        );
    }
}

#[test]
fn a_batch_of_one_is_the_single_record_path() {
    let m = Arc::new(Mini::new());
    for bb in [cpu_backbone(&m), emu_backbone(&m, 3)] {
        assert_batch_equals_singles(&bb, &[record(9, 31)], 0);
    }
}
