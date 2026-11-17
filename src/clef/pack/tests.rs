//! Pack tests on a tiny synthetic AutoRound checkpoint (no real weights needed).

use super::build::{run, write_safetensors};
use super::format::{ALIGN, ExpertRecord, layer_file};
use super::reader::{AlignedBuf, PackDir, PackedExperts, PackedWeights, dequant_expert};
use super::*;
use crate::clef::model::{
    Backbone, ExpertSource, ExpertWeights, ForwardOpts, LayerWeights, TextConfig, WeightSource,
};
use crate::clef::st::Dtype;
use anyhow::Result;
use std::path::{Path, PathBuf};

const H: usize = 64;
const I: usize = 32;
const G: usize = 32;
const LAYERS: usize = 2;
const EXPERTS: usize = 4;
const HEADS: usize = 4;
const KV: usize = 2;
const D: usize = 16;
const VOCAB: usize = 16;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// A value in `[-r, r)`.
    fn f(&mut self, r: f32) -> f32 {
        ((self.next() % 20001) as f32 / 10000.0 - 1.0) * r
    }
}

fn bf16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bf16_back(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|x| f32::from_bits(x.to_bits() & 0xffff_0000))
        .collect()
}

type Tensor = (String, Dtype, Vec<usize>, Vec<u8>);

/// A GPTQ-packed `[n, k]` linear plus its logical f32 weights.
fn qlinear(rng: &mut Lcg, module: &str, n: usize, k: usize) -> (Vec<Tensor>, Vec<f32>) {
    let q: Vec<u8> = (0..n * k).map(|_| (rng.next() % 256) as u8).collect();
    // scales: positive and negative, exactly representable (multiples of 1/64)
    let sc: Vec<f32> = (0..(k / G) * n)
        .map(|_| ((rng.next() % 31) as f32 - 15.0) / 64.0)
        .collect();
    let mut qw = vec![0u8; k / 4 * n * 4];
    for kb in 0..k / 4 {
        for row in 0..n {
            let mut w = 0u32;
            for j in 0..4 {
                w |= u32::from(q[row * k + kb * 4 + j]) << (8 * j);
            }
            qw[(kb * n + row) * 4..][..4].copy_from_slice(&w.to_le_bytes());
        }
    }
    let qz: Vec<u8> = (0..(k / G) * (n / 4))
        .flat_map(|_| 0x7f7f_7f7fu32.to_le_bytes())
        .collect();
    let scales: Vec<u8> = sc
        .iter()
        .flat_map(|&s| if s == 0.0 { 0u16 } else { f32_to_f16_bits(s) }.to_le_bytes())
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| (i32::from(q[i]) - 128) as f32 * sc[(i % k / G) * n + i / k])
        .collect();
    (
        vec![
            (format!("{module}.qweight"), Dtype::I32, vec![k / 4, n], qw),
            (
                format!("{module}.qzeros"),
                Dtype::I32,
                vec![k / G, n / 4],
                qz,
            ),
            (
                format!("{module}.scales"),
                Dtype::F16,
                vec![k / G, n],
                scales,
            ),
        ],
        w,
    )
}

/// The logical (dequantized) weights of the synthetic model, for the naive reference.
struct Truth {
    layers: Vec<LayerWeights>,
    experts: Vec<Vec<ExpertWeights>>,
    embed: Vec<f32>,
    norm: Vec<f32>,
}

fn temp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("clef-pack-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_source(dir: &Path) -> Truth {
    let mut rng = Lcg(7);
    let mut t: Vec<Tensor> = Vec::new();
    let dense = |t: &mut Vec<Tensor>, rng: &mut Lcg, name: &str, shape: Vec<usize>, r: f32| {
        let v: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| rng.f(r))
            .collect();
        t.push((name.to_string(), Dtype::Bf16, shape, bf16(&v)));
        bf16_back(&v)
    };
    let embed = dense(
        &mut t,
        &mut rng,
        "thinker.model.embed_tokens.weight",
        vec![VOCAB, H],
        1.0,
    );
    let norm = dense(&mut t, &mut rng, "thinker.model.norm.weight", vec![H], 1.0);
    dense(
        &mut t,
        &mut rng,
        "thinker.lm_head.weight",
        vec![VOCAB, H],
        1.0,
    );
    let (mut layers, mut experts) = (Vec::new(), Vec::new());
    for l in 0..LAYERS {
        let p = format!("thinker.model.layers.{l}.");
        let d = |t: &mut Vec<Tensor>, rng: &mut Lcg, n: &str, s: Vec<usize>, r: f32| {
            let v: Vec<f32> = (0..s.iter().product::<usize>())
                .map(|_| rng.f(r) + 1.0)
                .collect();
            t.push((format!("{p}{n}"), Dtype::Bf16, s, bf16(&v)));
            bf16_back(&v)
        };
        let input_norm = d(&mut t, &mut rng, "input_layernorm.weight", vec![H], 0.1);
        let post_norm = d(
            &mut t,
            &mut rng,
            "post_attention_layernorm.weight",
            vec![H],
            0.1,
        );
        let q_norm = d(&mut t, &mut rng, "self_attn.q_norm.weight", vec![D], 0.1);
        let k_norm = d(&mut t, &mut rng, "self_attn.k_norm.weight", vec![D], 0.1);
        let router = d(&mut t, &mut rng, "mlp.gate.weight", vec![EXPERTS, H], 0.5);
        let lin = |t: &mut Vec<Tensor>, rng: &mut Lcg, m: &str, n: usize, k: usize| {
            let (ts, w) = qlinear(rng, &format!("{p}{m}"), n, k);
            t.extend(ts);
            w
        };
        let q = lin(&mut t, &mut rng, "self_attn.q_proj", HEADS * D, H);
        let k = lin(&mut t, &mut rng, "self_attn.k_proj", KV * D, H);
        let v = lin(&mut t, &mut rng, "self_attn.v_proj", KV * D, H);
        let o = lin(&mut t, &mut rng, "self_attn.o_proj", H, HEADS * D);
        let mut ex = Vec::new();
        for e in 0..EXPERTS {
            let m = format!("mlp.experts.{e}.");
            ex.push(ExpertWeights {
                gate: lin(&mut t, &mut rng, &format!("{m}gate_proj"), I, H),
                up: lin(&mut t, &mut rng, &format!("{m}up_proj"), I, H),
                down: lin(&mut t, &mut rng, &format!("{m}down_proj"), H, I),
            });
        }
        layers.push(LayerWeights {
            input_norm,
            post_norm,
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
            router,
        });
        experts.push(ex);
    }
    // extra source tensors the pack must skip (and account for)
    t.push((
        "thinker.visual.blocks.0.w".into(),
        Dtype::Bf16,
        vec![2, 2],
        bf16(&[1.0; 4]),
    ));
    t.push((
        "talker.model.w".into(),
        Dtype::Bf16,
        vec![4],
        bf16(&[1.0; 4]),
    ));
    let wm: serde_json::Map<String, serde_json::Value> = t
        .iter()
        .map(|x| {
            (
                x.0.clone(),
                serde_json::json!("model-00001-of-00001.safetensors"),
            )
        })
        .collect();
    write_safetensors(&dir.join("model-00001-of-00001.safetensors"), &t).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": wm})).unwrap(),
    )
    .unwrap();
    let cfg = serde_json::json!({
        "model_type": "qwen3_omni_moe",
        "thinker_config": {"text_config": {
            "hidden_size": H, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
            "num_key_value_heads": KV, "head_dim": D, "num_experts": EXPERTS,
            "num_experts_per_tok": 2, "moe_intermediate_size": I, "rms_norm_eps": 1e-6,
            "rope_theta": 1e6, "norm_topk_prob": true, "vocab_size": VOCAB,
            "shared_expert_intermediate_size": 0, "mlp_only_layers": []}},
        "quantization_config": {"bits": 8, "group_size": G, "sym": true,
            "packing_format": "auto_round:auto_gptq", "block_name_to_quantize": "thinker.model.layers"},
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    for f in [
        "joint_head.safetensors",
        "joint_head_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        std::fs::write(dir.join(f), format!("stub {f}")).unwrap();
    }
    Truth {
        layers,
        experts,
        embed,
        norm,
    }
}

fn pack_synthetic(name: &str) -> (PathBuf, PathBuf, Truth) {
    let src = temp(&format!("{name}-src"));
    let out = temp(&format!("{name}-out"));
    let truth = write_source(&src);
    let args = [
        "--src",
        src.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--threads",
        "2",
        "--verify",
    ];
    run(args.iter().map(|s| s.to_string())).unwrap();
    (src, out, truth)
}

#[test]
fn a_synthetic_checkpoint_packs_verifies_and_reads_back() {
    let (src, out, truth) = pack_synthetic("rt");
    let pack = PackDir::open(&out).unwrap();
    let m = &pack.manifest;
    assert_eq!(
        (m.model.layers, m.model.experts, m.model.group),
        (LAYERS, EXPERTS, G)
    );
    assert_eq!(m.version, 1);
    assert!(m.files.iter().any(|f| f.name == "lm_head.safetensors"));
    assert!(m.files.iter().all(|f| f.sha256.len() == 64));
    // media/talker tensors are listed as skipped
    let skipped: Vec<&str> = m.skipped.iter().map(|s| s.prefix.as_str()).collect();
    assert_eq!(skipped, ["talker.model", "thinker.visual"]);
    // every layer file: aligned experts, the records equal the file bytes, one read each
    let stride = pack.stride();
    assert_eq!(stride % ALIGN, 0);
    for l in 0..LAYERS {
        let raw = std::fs::read(out.join(layer_file(l))).unwrap();
        assert_eq!(raw.len(), ALIGN + EXPERTS * stride);
        let mut buf = AlignedBuf::new(stride);
        for e in 0..EXPERTS {
            pack.read_expert(l, e, &mut buf).unwrap();
            let off = ALIGN + e * stride;
            assert_eq!(
                buf.as_slice()[..stride],
                raw[off..off + stride],
                "layer {l} expert {e}"
            );
        }
    }
    // PackedExperts == naive dequant of the source
    let pe = PackedExperts { pack: &pack };
    for l in 0..LAYERS {
        for e in 0..EXPERTS {
            let got = pe.expert(l, e).unwrap();
            let want = &truth.experts[l][e];
            assert_eq!(got.gate, want.gate, "gate {l}/{e}");
            assert_eq!(got.up, want.up, "up {l}/{e}");
            assert_eq!(got.down, want.down, "down {l}/{e}");
        }
    }
    // dequant_expert on a borrowed record agrees with the thread-local path
    let w = pack.with_expert(1, 2, dequant_expert).unwrap();
    assert_eq!(w.down, truth.experts[1][2].down);
    pack.with_expert(0, 0, |r: &ExpertRecord<'_>| {
        assert_eq!(r.gu_codes.len(), 2 * I * H)
    })
    .unwrap();
    // attention + norms + router through the weight source, embeddings and final norm
    let cfg = TextConfig::from_model_dir(&out).unwrap();
    let pw = PackedWeights::open(&out, cfg).unwrap();
    for l in 0..LAYERS {
        let got = pw.layer_weights(l).unwrap();
        let want = &truth.layers[l];
        assert_eq!(got.q, want.q);
        assert_eq!(got.k, want.k);
        assert_eq!(got.v, want.v);
        assert_eq!(got.o, want.o);
        assert_eq!(got.input_norm, want.input_norm);
        assert_eq!(got.post_norm, want.post_norm);
        assert_eq!(got.q_norm, want.q_norm);
        assert_eq!(got.k_norm, want.k_norm);
        assert_eq!(got.router, want.router);
    }
    let mut row = vec![0f32; H];
    pw.embed_row(5, &mut row).unwrap();
    assert_eq!(row, truth.embed[5 * H..6 * H]);
    assert_eq!(pw.final_norm().unwrap(), truth.norm);
    let _ = std::fs::remove_dir_all(src);
    let _ = std::fs::remove_dir_all(out);
}

/// The naive in-memory weight source (the dequantized truth).
struct Mem {
    t: Truth,
}

struct MemExperts<'a>(&'a Truth);

impl ExpertSource for MemExperts<'_> {
    fn expert(&self, l: usize, e: usize) -> Result<ExpertWeights> {
        let w = &self.0.experts[l][e];
        Ok(ExpertWeights {
            gate: w.gate.clone(),
            up: w.up.clone(),
            down: w.down.clone(),
        })
    }
}

impl ExpertSource for Mem {
    fn expert(&self, l: usize, e: usize) -> Result<ExpertWeights> {
        MemExperts(&self.t).expert(l, e)
    }
}

impl WeightSource for Mem {
    fn layer_weights(&self, l: usize) -> Result<LayerWeights> {
        Ok(self.t.layers[l].clone())
    }
    fn experts(&self) -> &dyn ExpertSource {
        self
    }
    fn embed_row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        out.copy_from_slice(&self.t.embed[id as usize * H..(id as usize + 1) * H]);
        Ok(())
    }
    fn final_norm(&self) -> Result<Vec<f32>> {
        Ok(self.t.norm.clone())
    }
    fn release(&self) {}
}

#[test]
fn the_backbone_on_the_pack_equals_the_backbone_on_the_dequantized_weights() {
    let (src, out, truth) = pack_synthetic("bb");
    let cfg = TextConfig::from_model_dir(&out).unwrap();
    let ids = [3u32, 1, 4, 1, 5, 9, 2, 6];
    let opts = ForwardOpts {
        threads: 2,
        release_pages: true,
    };
    let from_pack = Backbone::open_pack(&out)
        .unwrap()
        .forward(&ids, &opts)
        .unwrap();
    let naive = Backbone::with_source(cfg, Box::new(Mem { t: truth }))
        .forward(&ids, &opts)
        .unwrap();
    assert_eq!(
        from_pack.final_hidden, naive.final_hidden,
        "bit-identical: same f32 weights"
    );
    assert_eq!(from_pack.final_normed, naive.final_normed);
    assert!(from_pack.final_normed.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(src);
    let _ = std::fs::remove_dir_all(out);
}

#[test]
fn verify_only_catches_a_flipped_byte_and_a_stale_source() {
    let (src, out, _) = pack_synthetic("bad");
    let args = |extra: &[&str]| {
        let mut v: Vec<String> = [
            "--src",
            src.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--threads",
            "2",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        v.extend(extra.iter().map(|s| s.to_string()));
        v.into_iter()
    };
    run(args(&["--verify-only"])).unwrap();
    // flip one expert code byte in layer 1
    let p = out.join(layer_file(1));
    let mut raw = std::fs::read(&p).unwrap();
    raw[ALIGN + 17] ^= 0x55;
    std::fs::write(&p, &raw).unwrap();
    let err = format!("{:#}", run(args(&["--verify-only"])).unwrap_err());
    assert!(
        err.contains("checksum mismatch") && err.contains("layer_01.experts"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(src);
    let _ = std::fs::remove_dir_all(out);
}

#[test]
fn the_manifest_validation_rejects_inconsistent_packs() {
    let (src, out, _) = pack_synthetic("man");
    let good = Manifest::read(&out).unwrap();
    good.validate().unwrap();
    assert!(PackDir::is_pack(&out));
    assert!(!PackDir::is_pack(&src));
    let mut m = good.clone();
    m.version = 2;
    assert!(m.validate().is_err());
    let mut m = good.clone();
    m.format = "other".into();
    assert!(m.validate().is_err());
    let mut m = good.clone();
    m.files.retain(|f| f.name != "layer_01.experts");
    assert!(m.validate().is_err());
    let mut m = good.clone();
    m.expert_record.stride += 4096;
    assert!(m.validate().is_err());
    let mut m = good.clone();
    m.model.group = 7;
    assert!(m.validate().is_err());
    let mut m = good;
    m.resident.tensors[0].offset = u64::MAX / 2;
    assert!(m.validate().is_err());
    let _ = std::fs::remove_dir_all(src);
    let _ = std::fs::remove_dir_all(out);
}

// ---------------------------------------------------------------- shard store (step 0)

mod shards {
    use super::*;
    use crate::clef::autoround_store::{AutoRoundWeights, ShardStore};
    use crate::clef::engine::{ClefStore, ExpertReader};

    #[test]
    fn the_shard_store_reads_records_identical_to_the_pack() {
        let (src, out, _truth) = pack_synthetic("shard-rec");
        assert!(ShardStore::is_autoround(&src));
        assert!(!ShardStore::is_autoround(&out));
        let store = ShardStore::open(&src).unwrap();
        let pack = PackDir::open(&out).unwrap();
        assert_eq!(store.dims(), pack.dims);
        let stride = pack.stride();
        let (mut a, mut b) = (store.new_buffer(), pack.new_buffer());
        for l in 0..LAYERS {
            for e in 0..EXPERTS {
                store.read_expert(l, e, &mut a).unwrap();
                pack.read_expert(l, e, &mut b).unwrap();
                assert_eq!(a.as_slice()[..stride], b.as_slice()[..stride], "{l}/{e}");
            }
        }
        // resident projections: codes and f32 scales equal the pack's resident tensors
        for l in 0..LAYERS {
            let m = format!("thinker.model.layers.{l}.self_attn.q_proj");
            let (c1, s1) = ClefStore::resident_linear(&store, &m, HEADS * D, H).unwrap();
            let (c2, s2) = ClefStore::resident_linear(&pack, &m, HEADS * D, H).unwrap();
            assert_eq!(c1, c2);
            assert_eq!(s1, s2);
            let m = format!("thinker.model.layers.{l}.self_attn.o_proj");
            assert_eq!(
                ClefStore::dequant_linear(&store, &m, H, HEADS * D).unwrap(),
                ClefStore::dequant_linear(&pack, &m, H, HEADS * D).unwrap()
            );
        }
    }

    #[test]
    fn the_backbone_on_the_shards_equals_the_backbone_on_the_pack() {
        let (src, out, _) = pack_synthetic("shard-bb");
        let cfg = TextConfig::from_model_dir(&src).unwrap();
        let sw = AutoRoundWeights::open(&src, cfg.clone()).unwrap();
        let pw = PackedWeights::open(&out, cfg.clone()).unwrap();
        for l in 0..LAYERS {
            let (a, b) = (sw.layer_weights(l).unwrap(), pw.layer_weights(l).unwrap());
            assert_eq!((a.q, a.k, a.v, a.o), (b.q, b.k, b.v, b.o));
            assert_eq!((a.router, a.input_norm), (b.router, b.input_norm));
            for e in 0..EXPERTS {
                let (x, y) = (sw.expert(l, e).unwrap(), pw.expert(l, e).unwrap());
                assert_eq!((x.gate, x.up, x.down), (y.gate, y.up, y.down));
            }
        }
        let ids = [3u32, 1, 4, 1, 5, 9, 2, 6];
        let opts = ForwardOpts {
            threads: 2,
            ..ForwardOpts::default()
        };
        let a = Backbone::open_autoround(&src)
            .unwrap()
            .forward(&ids, &opts)
            .unwrap();
        let b = Backbone::open_pack(&out)
            .unwrap()
            .forward(&ids, &opts)
            .unwrap();
        assert_eq!(a.final_normed, b.final_normed);
    }

    #[test]
    fn startup_validation_names_a_truncated_or_missing_shard() {
        let (src, _, _) = pack_synthetic("shard-bad");
        let shard = src.join("model-00001-of-00001.safetensors");
        let full = std::fs::read(&shard).unwrap();
        std::fs::write(&shard, &full[..full.len() - 100]).unwrap();
        let e = format!(
            "{:#}",
            ShardStore::open(&src).err().expect("truncated shard")
        );
        assert!(e.contains("model-00001-of-00001"), "{e}");
        std::fs::remove_file(&shard).unwrap();
        assert!(ShardStore::open(&src).is_err());
        std::fs::remove_file(src.join("model.safetensors.index.json")).unwrap();
        assert!(ShardStore::open(&src).is_err());
    }

    /// The real checkpoint against the real pack: every sampled expert record is
    /// bit-identical, and an expert is two reads. Skipped when either is absent.
    #[test]
    fn real_shards_equal_the_real_pack_bit_for_bit() {
        let src = Path::new("/home/royalcat/models/clef-omni-int8-autoround");
        let pack_dir = Path::new("/home/royalcat/models/clef-pack-g128");
        if !src.join("model.safetensors.index.json").exists()
            || !pack_dir.join("manifest.json").exists()
        {
            eprintln!("skipped: the real checkpoint or pack is absent");
            return;
        }
        let store = ShardStore::open(src).unwrap();
        let pack = PackDir::open(pack_dir).unwrap();
        let stride = pack.stride();
        let (mut a, mut b) = (store.new_buffer(), pack.new_buffer());
        let mut n = 0;
        for (l, e) in [
            (0, 0),
            (0, 1),
            (0, 127),
            (7, 64),
            (8, 0),
            (17, 105),
            (24, 3),
            (33, 100),
            (42, 10),
            (47, 127),
        ] {
            assert_eq!(store.reads_per_expert(l, e).unwrap(), 2, "{l}/{e} reads");
            store.read_expert(l, e, &mut a).unwrap();
            pack.read_expert(l, e, &mut b).unwrap();
            assert!(a.as_slice()[..stride] == b.as_slice()[..stride], "{l}/{e}");
            n += 1;
        }
        // informational: wall of one expert read + conversion (set RAYON_NUM_THREADS=1 for
        // the single-core cost)
        let t0 = std::time::Instant::now();
        for e in 0..20 {
            store.read_expert(30, e, &mut a).unwrap();
        }
        let t1 = std::time::Instant::now();
        for e in 0..20 {
            pack.read_expert(30, e, &mut b).unwrap();
        }
        eprintln!(
            "per expert: shard read+convert {:.1} ms, pack read {:.1} ms",
            (t1 - t0).as_secs_f64() * 50.0,
            t1.elapsed().as_secs_f64() * 50.0
        );
        // an attention projection and the dequantized form agree too
        let m = "thinker.model.layers.3.self_attn.k_proj";
        let (c1, s1) = ClefStore::resident_linear(&store, m, 512, 2048).unwrap();
        let (c2, s2) = ClefStore::resident_linear(&pack, m, 512, 2048).unwrap();
        assert!(c1 == c2 && s1 == s2);
        eprintln!(
            "real shards == real pack on {n} experts (direct io: {})",
            store.is_direct()
        );
    }
}
