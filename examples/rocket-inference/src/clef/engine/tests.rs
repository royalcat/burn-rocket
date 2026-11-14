use super::*;
use crate::clef::model::{LayerWeights, layer_forward};
use crate::clef::pack::format::ExpertRecordMut;
use crate::clef::pack::{f32_to_f16_bits, reader::dequant_expert};
use std::time::Duration;

const DIMS: Dims = Dims {
    hidden: 64,
    inter: 32,
    group: 32,
};
const EXPERTS: usize = 8;
const TOPK: usize = 2;

fn cfg() -> TextConfig {
    TextConfig {
        hidden: DIMS.hidden,
        layers: 3,
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        experts: EXPERTS,
        top_k: TOPK,
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
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize % n
    }
}

fn make_record(seed: u64) -> Vec<u8> {
    let l = DIMS.layout();
    let mut region = vec![0u8; l.stride];
    let mut r = Lcg(seed);
    let rec = ExpertRecordMut::split(DIMS, &mut region);
    for c in rec.gu_codes.iter_mut().chain(rec.down_codes.iter_mut()) {
        *c = (r.next() * 100.0) as i8;
    }
    for sc in rec
        .gu_scales
        .chunks_mut(2)
        .chain(rec.down_scales.chunks_mut(2))
    {
        let bits = f32_to_f16_bits(0.004 + 0.02 * r.next().abs() + 0.001);
        sc.copy_from_slice(&bits.to_le_bytes());
    }
    region
}

/// In-memory stand-in for a pack: `layers x EXPERTS` records, optional injected failure.
struct MemReader {
    layers: usize,
    recs: Vec<Vec<u8>>,
    fail: Mutex<Option<(usize, usize)>>,
    reads: AtomicUsize,
}

impl MemReader {
    fn new(layers: usize) -> Arc<Self> {
        Arc::new(Self {
            layers,
            recs: (0..layers * EXPERTS)
                .map(|i| make_record(0x1234 + i as u64))
                .collect(),
            fail: Mutex::new(None),
            reads: AtomicUsize::new(0),
        })
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
        EXPERTS
    }
    fn new_buffer(&self) -> AlignedBuf {
        AlignedBuf::new(DIMS.layout().stride)
    }
    fn read(&self, layer: usize, expert: usize, buf: &mut AlignedBuf) -> Result<()> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if *lock(&self.fail) == Some((layer, expert)) {
            bail!("injected read failure");
        }
        let r = &self.recs[layer * EXPERTS + expert];
        buf.as_mut_slice()[..r.len()].copy_from_slice(r);
        Ok(())
    }
}

/// The f32 math of the CPU path for one expert (dequantized weights).
struct RefWorker;

impl ExpertWorker for RefWorker {
    fn run(
        &mut self,
        rec: &ExpertRecord<'_>,
        x: &[f32],
        m: usize,
        out: &mut [f32],
        _t: &mut ExpertTimes,
    ) -> Result<()> {
        let (h, i) = (DIMS.hidden, DIMS.inter);
        let ew = dequant_expert(rec);
        let mut g = matmul_nt(x, m, h, &ew.gate, i, 1);
        let u = matmul_nt(x, m, h, &ew.up, i, 1);
        for (a, b) in g.iter_mut().zip(&u) {
            *a = silu(*a) * b;
        }
        out.copy_from_slice(&matmul_nt(&g, m, i, &ew.down, h, 1));
        Ok(())
    }
}

fn tag_of_codes(codes: &[i8]) -> usize {
    codes[..8]
        .iter()
        .fold(0usize, |a, &c| a.wrapping_mul(31).wrapping_add(c as usize))
}

fn tag_of(rec: &[u8]) -> usize {
    // gu codes start at offset 0 of the record
    tag_of_codes(&rec[..8].iter().map(|&b| b as i8).collect::<Vec<i8>>())
}

/// Wraps a worker: records which experts it ran and sleeps a jittered time.
struct Jitter<W: ExpertWorker> {
    inner: W,
    id: usize,
    log: Arc<Mutex<Vec<usize>>>,
    fail_expert: Option<usize>,
}

impl<W: ExpertWorker> ExpertWorker for Jitter<W> {
    fn run(
        &mut self,
        rec: &ExpertRecord<'_>,
        x: &[f32],
        m: usize,
        out: &mut [f32],
        t: &mut ExpertTimes,
    ) -> Result<()> {
        let tag = tag_of_codes(rec.gu_codes);
        lock(&self.log).push(tag);
        std::thread::sleep(Duration::from_micros(((tag ^ self.id) % 7) as u64 * 400));
        if self.fail_expert.is_some_and(|e| tag == e) {
            bail!("injected worker failure");
        }
        self.inner.run(rec, x, m, out, t)
    }
}

struct NoDense;

impl Dense for NoDense {
    fn label(&self) -> String {
        "none".into()
    }
    fn qkv(&self, _: usize, _: &[f32], _: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        bail!("no dense in this test")
    }
    fn o_proj(&self, _: usize, _: &[f32], _: usize) -> Result<Vec<f32>> {
        bail!("no dense in this test")
    }
}

/// Random distinct top-k experts per token (from `only` when given).
fn routing(s: usize, seed: u64, only: Option<&[usize]>) -> (Vec<u32>, Vec<f32>) {
    let mut r = Lcg(seed);
    let pool: Vec<usize> = only.map_or_else(|| (0..EXPERTS).collect(), |o| o.to_vec());
    let (mut ids, mut wts) = (Vec::new(), Vec::new());
    for _ in 0..s {
        let mut p = pool.clone();
        let mut picked = Vec::new();
        for _ in 0..TOPK {
            picked.push(p.remove(r.below(p.len())));
        }
        let w: Vec<f32> = (0..TOPK).map(|_| 0.1 + r.next().abs()).collect();
        let z: f32 = w.iter().sum();
        for (e, w) in picked.into_iter().zip(w) {
            ids.push(e as u32);
            wts.push(w / z);
        }
    }
    (ids, wts)
}

fn activations(s: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg(seed);
    (0..s * DIMS.hidden).map(|_| r.next()).collect()
}

/// The CPU path's MoE loop (`layer_forward`), written independently of `Buckets`.
fn reference_moe(
    reader: &MemReader,
    layer: usize,
    n2: &[f32],
    s: usize,
    ids: &[u32],
    wts: &[f32],
    y: &mut [f32],
) {
    let (h, i) = (DIMS.hidden, DIMS.inter);
    let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); EXPERTS];
    for t in 0..s {
        for j in 0..TOPK {
            buckets[ids[t * TOPK + j] as usize].push((t, wts[t * TOPK + j]));
        }
    }
    for (e, b) in buckets.iter().enumerate() {
        if b.is_empty() {
            continue;
        }
        let rec = ExpertRecord::parse(DIMS, &reader.recs[layer * EXPERTS + e]).unwrap();
        let ew = dequant_expert(&rec);
        let m = b.len();
        let mut xe = Vec::new();
        for &(t, _) in b {
            xe.extend_from_slice(&n2[t * h..(t + 1) * h]);
        }
        let mut g = matmul_nt(&xe, m, h, &ew.gate, i, 1);
        let u = matmul_nt(&xe, m, h, &ew.up, i, 1);
        for (a, c) in g.iter_mut().zip(&u) {
            *a = silu(*a) * c;
        }
        let mut o = matmul_nt(&g, m, i, &ew.down, h, 1);
        for (row, &(_, wt)) in o.chunks_mut(h).zip(b) {
            row.iter_mut().for_each(|v| *v *= wt);
        }
        for (row, &(t, _)) in o.chunks(h).zip(b) {
            for (a, c) in y[t * h..(t + 1) * h].iter_mut().zip(row) {
                *a += c;
            }
        }
    }
}

fn engine(
    reader: Arc<MemReader>,
    consumers: usize,
    depth: usize,
    spec: bool,
    log: &Arc<Mutex<Vec<usize>>>,
) -> PackEngine {
    let workers: Vec<Box<dyn ExpertWorker>> = (0..consumers)
        .map(|id| {
            Box::new(Jitter {
                inner: RefWorker,
                id,
                log: log.clone(),
                fail_expert: None,
            }) as Box<dyn ExpertWorker>
        })
        .collect();
    PackEngine::new(
        "test".into(),
        cfg(),
        reader,
        Box::new(NoDense),
        workers,
        depth,
        spec,
    )
}

fn run_moe(
    eng: &PackEngine,
    layer: usize,
    s: usize,
    ids: &[u32],
    wts: &[f32],
    y0: &[f32],
) -> Result<Vec<f32>> {
    let n2 = activations(s, 99);
    let mut y = y0.to_vec();
    eng.moe_add(layer, &n2, s, ids, wts, &mut y)?;
    Ok(y)
}

#[test]
fn buckets_scatter_every_slot_to_its_row() {
    let (s, k) = (11, TOPK);
    let (ids, wts) = routing(s, 5, None);
    let b = Buckets::build(&ids, &wts, s, k, EXPERTS);
    assert_eq!(b.rows.iter().map(Vec::len).sum::<usize>(), s * k);
    for t in 0..s {
        for j in 0..k {
            let (e, row) = b.slots[t * k + j];
            assert_eq!(e, ids[t * k + j]);
            let (tok, w) = b.rows[e as usize][row as usize];
            assert_eq!((tok as usize, w), (t, wts[t * k + j]));
        }
    }
    // tokens ascend inside every bucket
    for r in &b.rows {
        assert!(r.windows(2).all(|w| w[0].0 < w[1].0));
    }
    let hit = b.hit();
    assert!(hit.windows(2).all(|w| w[0] < w[1]));
    assert!(hit.iter().all(|&e| !b.rows[e].is_empty()));
}

#[test]
fn the_streamed_moe_equals_the_cpu_loop_bit_for_bit_for_any_schedule() {
    let reader = MemReader::new(3);
    for s in [3usize, 40, 300] {
        let (ids, wts) = routing(s, 7 + s as u64, None);
        let y0 = activations(s, 1);
        let n2 = activations(s, 99);
        let mut want = y0.clone();
        reference_moe(&reader, 0, &n2, s, &ids, &wts, &mut want);
        // 300 tokens x top-2 = 600 slots turns the speculative whole-layer read on
        for (consumers, depth, spec) in [(1, 1, false), (3, 2, true), (5, 8, true), (2, 1, true)] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let eng = engine(reader.clone(), consumers, depth, spec, &log);
            let got = run_moe(&eng, 0, s, &ids, &wts, &y0).unwrap();
            assert_eq!(
                got, want,
                "s={s} consumers={consumers} depth={depth} spec={spec}"
            );
            // every hit expert exactly once, nothing else
            let mut hit: Vec<usize> = Buckets::build(&ids, &wts, s, TOPK, EXPERTS)
                .hit()
                .into_iter()
                .map(|e| tag_of(&reader.recs[e]))
                .collect();
            let mut ran = lock(&log).clone();
            hit.sort_unstable();
            ran.sort_unstable();
            assert_eq!(ran, hit, "s={s} consumers={consumers}");
            eng.finish_request(s);
        }
    }
}

#[test]
fn a_tiny_request_reads_only_the_hit_experts_and_a_big_one_discards_the_rest() {
    let reader = MemReader::new(3);
    // S=3, top-2: at most 6 experts are hit; below SPEC_MIN_SLOTS only those are queued
    let (ids, wts) = routing(3, 3, None);
    let hit = Buckets::build(&ids, &wts, 3, TOPK, EXPERTS).hit();
    assert!(hit.len() <= 6);
    let log = Arc::new(Mutex::new(Vec::new()));
    let eng = engine(reader.clone(), 3, 4, true, &log);
    run_moe(&eng, 0, 3, &ids, &wts, &activations(3, 1)).unwrap();
    eng.finish_request(3);
    assert_eq!(reader.reads.load(Ordering::SeqCst), hit.len());

    // 300 tokens routed only to experts {1,3}: the reader speculates all 8 of the layer,
    // the consumers must run exactly two and discard the rest, across two layers
    let only = [1usize, 3];
    let log = Arc::new(Mutex::new(Vec::new()));
    let eng = engine(reader.clone(), 3, 2, true, &log);
    let mut y = activations(300, 1);
    for layer in 0..2 {
        let (ids, wts) = routing(300, 40 + layer as u64, Some(&only));
        let n2 = activations(300, 99);
        let mut want = y.clone();
        reference_moe(&reader, layer, &n2, 300, &ids, &wts, &mut want);
        eng.hint_layer(layer, 300);
        eng.moe_add(layer, &n2, 300, &ids, &wts, &mut y).unwrap();
        assert_eq!(y, want, "layer {layer}");
    }
    let mut ran = lock(&log).clone();
    ran.sort_unstable();
    let mut want: Vec<usize> = [0usize, 1]
        .iter()
        .flat_map(|&l| only.iter().map(move |&e| (l, e)))
        .map(|(l, e)| tag_of(&reader.recs[l * EXPERTS + e]))
        .collect();
    want.sort_unstable();
    assert_eq!(ran, want);
    eng.finish_request(300);
}

#[test]
fn read_and_worker_failures_surface_and_the_engine_recovers() {
    let reader = MemReader::new(3);
    let (ids, wts) = routing(40, 11, None);
    let y0 = activations(40, 1);
    let log = Arc::new(Mutex::new(Vec::new()));
    let eng = engine(reader.clone(), 3, 2, true, &log);

    *lock(&reader.fail) = Some((0, 5));
    let err = format!("{:#}", run_moe(&eng, 0, 40, &ids, &wts, &y0).unwrap_err());
    assert!(err.contains("read layer 0 expert 5"), "{err}");
    eng.abort_request();

    // the next request on the same engine works and is exact
    *lock(&reader.fail) = None;
    let n2 = activations(40, 99);
    let mut want = y0.clone();
    reference_moe(&reader, 0, &n2, 40, &ids, &wts, &mut want);
    assert_eq!(run_moe(&eng, 0, 40, &ids, &wts, &y0).unwrap(), want);
    eng.finish_request(40);

    // a worker that fails on one expert
    let bad = tag_of(&reader.recs[2]);
    let workers: Vec<Box<dyn ExpertWorker>> = (0..3)
        .map(|id| {
            Box::new(Jitter {
                inner: RefWorker,
                id,
                log: log.clone(),
                fail_expert: Some(bad),
            }) as Box<dyn ExpertWorker>
        })
        .collect();
    let eng = PackEngine::new(
        "t".into(),
        cfg(),
        reader,
        Box::new(NoDense),
        workers,
        2,
        false,
    );
    let err = format!("{:#}", run_moe(&eng, 0, 40, &ids, &wts, &y0).unwrap_err());
    assert!(err.contains("injected worker failure"), "{err}");
}

/// The emulated expert against a naive f64 evaluation of the same W8A8 pipeline.
#[test]
fn the_emulated_expert_matches_a_naive_w8a8_reference() {
    let rec_bytes = make_record(77);
    let rec = ExpertRecord::parse(DIMS, &rec_bytes).unwrap();
    let (h, i, g) = (DIMS.hidden, DIMS.inter, DIMS.group);
    let m = 5;
    let x = activations(m, 8);
    let mut out = vec![0f32; m * h];
    EmuWorker::new(DIMS)
        .run(&rec, &x, m, &mut out, &mut ExpertTimes::default())
        .unwrap();

    let f16 = |b: u16| f64::from(crate::clef::st::f16_to_f32(b));
    let naive =
        |aq: &[i8], a_s: &[f32], k: usize, codes: &[i8], sc: &[u16], n: usize| -> Vec<f64> {
            let ng = k / g;
            let mut o = vec![0f64; m * n];
            for r in 0..m {
                for j in 0..n {
                    for c in 0..k {
                        o[r * n + j] += f64::from(aq[r * k + c])
                            * f64::from(a_s[r * ng + c / g])
                            * f64::from(codes[j * k + c])
                            * f16(sc[j * ng + c / g]);
                    }
                }
            }
            o
        };
    let (mut aq, mut a_s) = (vec![0i8; m * h], vec![0f32; m * h / g]);
    quantize_rows(&x, h, g, &mut aq, &mut a_s, false);
    let gu = naive(&aq, &a_s, h, rec.gu_codes, rec.gu_scales, 2 * i);
    let mut act = vec![0f32; m * i];
    for r in 0..m {
        for j in 0..i {
            act[r * i + j] = silu(gu[r * 2 * i + j] as f32) * gu[r * 2 * i + i + j] as f32;
        }
    }
    let (mut aq2, mut a_s2) = (vec![0i8; m * i], vec![0f32; m * i / g]);
    quantize_rows(&act, i, g, &mut aq2, &mut a_s2, false);
    let want = naive(&aq2, &a_s2, i, rec.down_codes, rec.down_scales, h);
    let peak = want.iter().fold(0f64, |a, &v| a.max(v.abs()));
    for (a, b) in out.iter().zip(&want) {
        assert!(
            (f64::from(*a) - b).abs() <= 2e-3 * peak,
            "{a} vs {b} (peak {peak})"
        );
    }
    // and it tracks the f32 expert within the W8A8 error
    let mut f32out = vec![0f32; m * h];
    RefWorker
        .run(&rec, &x, m, &mut f32out, &mut ExpertTimes::default())
        .unwrap();
    let num: f64 = out
        .iter()
        .zip(&f32out)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let den: f64 = f32out.iter().map(|b| f64::from(*b).powi(2)).sum();
    assert!(
        (num / den).sqrt() < 0.05,
        "relative error {}",
        (num / den).sqrt()
    );
}

/// `engine_layer_forward` with f32 stand-ins equals the legacy `layer_forward`
/// (blocked attention vs the row-wise one: float-close).
#[test]
fn the_engine_layer_matches_the_legacy_layer() {
    use crate::clef::model::{ExpertSource, ExpertWeights, LayerSmall, Rope};
    let c = cfg();
    let (h, d) = (c.hidden, c.head_dim);
    let mut r = Lcg(5);
    let mut v = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|_| r.next() * s).collect() };
    let one_plus = |x: Vec<f32>| -> Vec<f32> { x.iter().map(|x| 1.0 + x).collect() };
    let w = LayerWeights {
        input_norm: one_plus(v(h, 0.1)),
        post_norm: one_plus(v(h, 0.1)),
        q: v(c.heads * d * h, 0.3),
        k: v(c.kv_heads * d * h, 0.3),
        v: v(c.kv_heads * d * h, 0.3),
        o: v(h * c.heads * d, 0.3),
        q_norm: one_plus(v(d, 0.1)),
        k_norm: one_plus(v(d, 0.1)),
        router: v(EXPERTS * h, 0.5),
    };
    let reader = MemReader::new(1);
    struct Src(Arc<MemReader>);
    impl ExpertSource for Src {
        fn expert(&self, layer: usize, e: usize) -> Result<ExpertWeights> {
            let rec = ExpertRecord::parse(DIMS, &self.0.recs[layer * EXPERTS + e])?;
            Ok(dequant_expert(&rec))
        }
    }
    struct F32Dense {
        w: LayerWeights,
        cfg: TextConfig,
    }
    impl Dense for F32Dense {
        fn label(&self) -> String {
            "f32".into()
        }
        fn qkv(&self, _: usize, n: &[f32], s: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
            let (h, d) = (self.cfg.hidden, self.cfg.head_dim);
            Ok((
                matmul_nt(n, s, h, &self.w.q, self.cfg.heads * d, 1),
                matmul_nt(n, s, h, &self.w.k, self.cfg.kv_heads * d, 1),
                matmul_nt(n, s, h, &self.w.v, self.cfg.kv_heads * d, 1),
            ))
        }
        fn o_proj(&self, _: usize, ctx: &[f32], s: usize) -> Result<Vec<f32>> {
            Ok(matmul_nt(
                ctx,
                s,
                self.cfg.heads * self.cfg.head_dim,
                &self.w.o,
                self.cfg.hidden,
                1,
            ))
        }
    }
    let eng = PackEngine::new(
        "t".into(),
        c.clone(),
        reader.clone(),
        Box::new(F32Dense {
            w: w.clone(),
            cfg: c.clone(),
        }),
        vec![
            Box::new(RefWorker),
            Box::new(RefWorker),
            Box::new(RefWorker),
        ],
        3,
        true,
    );
    for s in [1usize, 17, 300] {
        let x = activations(s, 21);
        let rope = Rope::new(&c, s);
        let want = layer_forward(&c, &w, &Src(reader.clone()), 0, &x, s, &rope, 1).unwrap();
        let small: LayerSmall = w.clone().into();
        let got = engine_layer_forward(&c, &small, &eng, 0, &x, s, &rope, 1).unwrap();
        assert_eq!(got.router_ids, want.router_ids, "s={s}");
        let max = got
            .hidden
            .iter()
            .zip(&want.hidden)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let peak = want.hidden.iter().fold(0f32, |a, &v| a.max(v.abs()));
        assert!(max < 1e-5 * peak, "s={s}: max |d| {max} (peak {peak})");
        eng.finish_request(s);
    }
}
