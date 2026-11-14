use super::*;
use crate::clef::encode::{encode_record_with_media, newline_token, prefix_tokens};
use crate::clef::head::QuestionSpans;
use crate::clef::math::softmax;
use crate::clef::model::{ForwardOpts, Tap};
use crate::clef::pipeline::{ClefModel, EngineKind};
use crate::clef::request::ClefRequest;
use crate::clef::testutil::{Npz, f32s, read_npz_keys};
use serde_json::Value;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const MODEL_DIR: &str = "/home/royalcat/models/clef-omni";
const AUTOROUND_DIR: &str = "/home/royalcat/models/clef-omni-int8-autoround";
const DUMP_DIR: &str = "/tmp/opencode/clef/dump";
const NL: u32 = 198;
const ALL_FIXTURES: &[&str] = &[
    "media-image-small",
    "media-image-cat",
    "media-images-two",
    "media-audio-5s",
    "media-video-silent",
    "media-video-audio-small",
    "media-video-test",
    "media-mixed",
    "media-mixed-av",
];

// ---- hand-checked units -----------------------------------------------------------

#[test]
fn audio_lengths_follow_the_chunked_conv_stack() {
    for (frames, want) in [
        (0, 0),
        (1, 1),
        (100, 13),
        (200, 26),
        (400, 52),
        (500, 65),
        (150, 13 + 7),
    ] {
        assert_eq!(audio_out_len(frames), want, "{frames} frames");
    }
}

#[test]
fn smart_resize_matches_the_documented_sizes() {
    const IMG: (usize, usize) = (65536, 1048576);
    const VID: (usize, usize) = (131072, 786432);
    // [H, W] -> pixels
    assert_eq!(smart_resize(64, 96, 32, IMG.0, IMG.1).unwrap(), (224, 320));
    assert_eq!(
        smart_resize(686, 960, 32, IMG.0, IMG.1).unwrap(),
        (672, 960)
    );
    assert_eq!(
        smart_resize(144, 192, 32, VID.0, VID.1).unwrap(),
        (320, 448)
    );
    assert_eq!(
        smart_resize(180, 240, 32, VID.0, VID.1).unwrap(),
        (320, 448)
    );
    assert_eq!(
        smart_resize(360, 480, 32, VID.0, VID.1).unwrap(),
        (352, 480)
    );
    assert!(smart_resize(10, 3000, 32, IMG.0, IMG.1).is_err());
}

#[test]
fn the_audio_in_video_interleave_alternates_one_group_per_second() {
    let v = interleave_video_audio([2, 20, 28], 1.0, 26);
    let mut runs: Vec<(bool, usize)> = Vec::new();
    for &b in &v {
        match runs.last_mut() {
            Some((x, n)) if *x == b => *n += 1,
            _ => runs.push((b, 1)),
        }
    }
    // measured on the oracle: V*140 A*13 V*140 A*13
    assert_eq!(runs, [(true, 140), (false, 13), (true, 140), (false, 13)]);
}

fn img_layout() -> MediaLayout {
    MediaLayout {
        image_grids: vec![[1, 14, 20]],
        ..Default::default()
    }
}

#[test]
fn an_image_gets_3d_positions_and_text_resumes_after_the_max() {
    let mut ids = vec![5, 6, 7];
    ids.extend(build_media_ids(&img_layout(), NL).unwrap());
    ids.extend([8, 9]);
    let p = rope_index(&ids, &img_layout()).unwrap();
    assert_eq!(p.len(), ids.len());
    // text 0..2, <vs> 3, image tokens from 4: t constant, h 4..10, w 4..13, <ve> 14, \n 15
    assert_eq!(&p.t[..3], &[0., 1., 2.]);
    assert_eq!(p.t[3], 3.0);
    assert!(p.t[4..74].iter().all(|&v| v == 4.0));
    assert_eq!((p.h[4], p.h[73], p.w[4], p.w[73]), (4.0, 10.0, 4.0, 13.0));
    assert_eq!(
        (p.h[74], p.t[75], p.t[76], p.t[77]),
        (14.0, 15.0, 16.0, 17.0)
    );
    // the structure checks of the oracle: rope delta = max + 1 - S
    assert_eq!(p.rope_delta(), 17.0 + 1.0 - ids.len() as f32);
}

#[test]
fn interleaved_mrope_gives_24_20_20_slots() {
    let mut n = [0usize; 3];
    for j in 0..64 {
        n[Positions::stream_of_slot(j)] += 1;
    }
    assert_eq!(n, MROPE_SECTION);
}

#[test]
fn audio_only_and_text_records_keep_plain_positions() {
    let l = MediaLayout {
        audio_frames: vec![500],
        explicit_audio: 1,
        ..Default::default()
    };
    let mut ids = vec![1, 2];
    ids.extend(build_media_ids(&l, NL).unwrap());
    assert_eq!(ids.len(), 2 + 1 + 65 + 1 + 1);
    assert_eq!(rope_index(&ids, &l).unwrap(), Positions::text(ids.len()));
    assert_eq!(
        rope_index(&[1, 2, 3], &MediaLayout::default()).unwrap(),
        Positions::text(3)
    );
}

#[test]
fn a_layout_that_disagrees_with_its_lengths_is_an_error() {
    let mut ids = vec![1];
    ids.extend(build_media_ids(&img_layout(), NL).unwrap());
    ids.pop(); // drop the "\n": still consistent
    assert!(rope_index(&ids, &img_layout()).is_ok());
    ids.insert(3, IMAGE_PAD); // one pad too many
    assert!(rope_index(&ids, &img_layout()).is_err());
    // audio count mismatch in the layout
    let bad = MediaLayout {
        audio_frames: vec![],
        explicit_audio: 1,
        ..Default::default()
    };
    assert!(build_media_ids(&bad, NL).is_err());
}

#[test]
fn features_line_up_with_pads_splice_and_deepstack() {
    let h = 4;
    let ids = [
        1,
        VISION_START,
        IMAGE_PAD,
        IMAGE_PAD,
        VISION_END,
        AUDIO_START,
        AUDIO_PAD,
        AUDIO_END,
        2,
    ];
    let row = |v: f32| vec![v; h];
    let cat = |a: &[f32], b: &[f32]| [a, b].concat();
    let t = TowerOutputs {
        image: cat(&row(10.0), &row(11.0)),
        audio: row(20.0),
        image_deepstack: [
            cat(&row(100.0), &row(101.0)),
            cat(&row(200.0), &row(201.0)),
            cat(&row(300.0), &row(301.0)),
        ],
        ..Default::default()
    };
    let f = MediaFeatures::assemble(&ids, None, &t, h).unwrap();
    assert_eq!(f.splice_idx, [2, 3, 6]);
    assert_eq!(f.deepstack_idx, [2, 3]);
    let mut x = vec![0f32; ids.len() * h];
    f.splice(&mut x, h);
    assert_eq!(&x[2 * h..3 * h], &row(10.0)[..]);
    assert_eq!(&x[6 * h..7 * h], &row(20.0)[..]);
    f.inject_deepstack(1, &mut x, h);
    assert_eq!(&x[2 * h..3 * h], &row(210.0)[..]);
    assert_eq!(&x[3 * h..4 * h], &row(212.0)[..]);
    assert_eq!(
        &x[6 * h..7 * h],
        &row(20.0)[..],
        "audio rows get no DeepStack"
    );
    f.inject_deepstack(3, &mut x, h); // beyond the taps: a no-op
    assert_eq!(&x[2 * h..3 * h], &row(210.0)[..]);
    // a count mismatch is refused
    let mut short = t.clone();
    short.audio.clear();
    assert!(MediaFeatures::assemble(&ids, None, &short, h).is_err());
}

// ---- oracle-backed ------------------------------------------------------------------

fn dump_ok(id: &str) -> bool {
    std::path::Path::new(DUMP_DIR)
        .join(format!("{id}.npz"))
        .exists()
}

fn meta(id: &str) -> Value {
    serde_json::from_slice(&std::fs::read(format!("{DUMP_DIR}/{id}.json")).unwrap()).unwrap()
}

fn fixture_record(id: &str) -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tools/clef_fixtures_media.json"
    );
    let all: Vec<Value> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    all.into_iter().find(|r| r["id"] == id).unwrap()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_u64().unwrap() as usize).collect())
        .unwrap_or_default()
}

fn grids(v: &Value) -> Vec<[usize; 3]> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|g| {
                    let g = usizes(g);
                    [g[0], g[1], g[2]]
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The layout the preprocessing would report for fixture `id` (from the oracle's json).
fn layout_of(id: &str) -> MediaLayout {
    let m = &meta(id)["media"];
    MediaLayout {
        image_grids: grids(&m["image_grid_thw"]),
        video_grids: grids(&m["video_grid_thw"]),
        audio_frames: usizes(&m["audio_feature_lengths"]),
        explicit_audio: fixture_record(id)["audio"]
            .as_array()
            .map_or(0, |a| a.len()),
        use_audio_in_video: m["use_audio_in_video"].as_bool().unwrap(),
        second_per_grid: vec![],
    }
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

fn load(id: &str, want: impl Fn(&str) -> bool) -> Npz {
    read_npz_keys(&PathBuf::from(DUMP_DIR).join(format!("{id}.npz")), want)
}

#[test]
fn placeholder_layout_and_positions_match_the_oracle_dumps() {
    let mut n = 0;
    for id in ALL_FIXTURES {
        if !dump_ok(id) {
            eprintln!("skipping {id}: no oracle dump in {DUMP_DIR}");
            continue;
        }
        let npz = load(id, |k| {
            k == "input_ids" || k == "position_ids" || k == "rope_delta"
        });
        let ids: Vec<u32> = i64s(&npz["input_ids"].2)
            .iter()
            .map(|&v| v as u32)
            .collect();
        let want = Positions::from_flat(&f32s(&npz["position_ids"].2));
        let layout = layout_of(id);
        // the media block follows the 35-token prefix
        let block = build_media_ids(&layout, NL).unwrap();
        assert_eq!(&ids[35..35 + block.len()], &block[..], "{id}: media block");
        let got = rope_index(&ids, &layout).unwrap();
        assert_eq!(got, want, "{id}: position ids");
        assert_eq!(
            got.rope_delta(),
            f32s(&npz["rope_delta"].2)[0],
            "{id}: rope delta"
        );
        eprintln!(
            "{id}: {} tokens, media block {}, positions equal",
            ids.len(),
            block.len()
        );
        n += 1;
    }
    if n == 0 {
        eprintln!("no oracle dumps found: only the unit cases ran");
    }
}

#[test]
fn encode_with_media_reproduces_the_oracle_input_ids() {
    let tok_path = PathBuf::from(MODEL_DIR).join("tokenizer.json");
    if !tok_path.exists() {
        eprintln!("skipping: no tokenizer at {}", tok_path.display());
        return;
    }
    let tok = Tokenizer::from_file(&tok_path).unwrap();
    assert_eq!(newline_token(&tok).unwrap(), NL);
    assert_eq!(prefix_tokens(&tok).unwrap().len(), 35);
    for id in ALL_FIXTURES {
        if !dump_ok(id) {
            continue;
        }
        let npz = load(id, |k| k == "input_ids");
        let want: Vec<u32> = i64s(&npz["input_ids"].2)
            .iter()
            .map(|&v| v as u32)
            .collect();
        let block = build_media_ids(&layout_of(id), newline_token(&tok).unwrap()).unwrap();
        let req = ClefRequest::from_value(fixture_record(id)).unwrap();
        assert!(req.has_media);
        let enc = encode_record_with_media(&tok, &req, 64000, None, &block).unwrap();
        assert_eq!(enc.input_ids, want, "{id}: input ids");
        let m = meta(id);
        for (q, w) in enc.questions.iter().zip(m["questions"].as_array().unwrap()) {
            let span = |v: &Value| {
                (
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                )
            };
            assert_eq!(q.question_span, span(&w["question_span"]), "{id}/{}", q.id);
            let ws: Vec<_> = w["option_spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(span)
                .collect();
            assert_eq!(q.option_spans, ws, "{id}/{}", q.id);
        }
        // the text-only encoder is the empty-media case
        let _ = encode_record_with_media(&tok, &req, 64000, None, &[]).unwrap();
        eprintln!("{id}: {} ids, spans equal", want.len());
    }
}

// ---- the end-to-end gates (ignored: stream the checkpoint) -----------------------------

fn cosine64(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt())
}

fn max_rel(a: &[f32], want: &[f32]) -> f32 {
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(want)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
        / peak
}

fn opt(npz: &Npz, k: &str) -> Vec<f32> {
    npz.get(k).map(|v| f32s(&v.2)).unwrap_or_default()
}

/// Run the media fixtures through `model` with the oracle's tower outputs and compare to
/// the oracle dump. `tight`: BF16 CPU path (hard numeric gates), else top-1 + bounded dp.
fn run_gate(model: &ClefModel, ids_env: &str, tight: bool) {
    let h = model.backbone.cfg.hidden;
    for id in ids_env.split(',') {
        if !dump_ok(id) {
            eprintln!("skipping {id}: no oracle dump in {DUMP_DIR}");
            continue;
        }
        let npz = load(id, |k| {
            !k.starts_with("router_") && !k.starts_with("media_") && !k.contains("_block_")
        });
        let m = meta(id);
        let ids: Vec<u32> = i64s(&npz["input_ids"].2)
            .iter()
            .map(|&v| v as u32)
            .collect();
        let layout = layout_of(id);
        let pos = rope_index(&ids, &layout).unwrap();
        assert_eq!(
            pos,
            Positions::from_flat(&f32s(&npz["position_ids"].2)),
            "{id}: positions"
        );
        let towers = TowerOutputs {
            image: opt(&npz, "image_merged"),
            video: opt(&npz, "video_merged"),
            audio: opt(&npz, "audio_features"),
            image_deepstack: [0, 1, 2].map(|k| opt(&npz, &format!("image_deepstack_{k}"))),
            video_deepstack: [0, 1, 2].map(|k| opt(&npz, &format!("video_deepstack_{k}"))),
        };
        let media = MediaFeatures::assemble(&ids, Some(pos), &towers, h).unwrap();
        // the assembled DeepStack positions and rows are the oracle's, bit for bit
        // (audio-only dumps carry no visual keys: then there must be no DeepStack rows)
        let vis: Vec<usize> = npz
            .get("visual_pos_idx")
            .map(|v| i64s(&v.2).iter().map(|&v| v as usize).collect())
            .unwrap_or_default();
        assert_eq!(media.deepstack_idx, vis, "{id}: visual_pos_idx");
        for k in 0..3 {
            assert_eq!(
                media.deepstack[k],
                opt(&npz, &format!("deepstack_L{k}")),
                "{id}: deepstack_L{k}"
            );
        }
        let emb = model.backbone.embed_input(&ids, Some(&media)).unwrap();
        assert_eq!(emb, opt(&npz, "embeddings"), "{id}: spliced embeddings");

        let k = model.backbone.cfg.top_k;
        let s = ids.len();
        let t0 = std::time::Instant::now();
        let mut rows: Vec<(usize, f64, f32)> = Vec::new();
        let mut router_sets = 0usize;
        let routers = load(id, |k| k.starts_with("router_ids_L"));
        let out = model
            .backbone
            .forward_media_tapped(
                &ids,
                Some(&media),
                &ForwardOpts::default(),
                &mut |t: &Tap| {
                    let l = t.layer;
                    if let Some(want) = npz.get(&format!("hidden_L{l}")) {
                        let want = f32s(&want.2);
                        rows.push((l, cosine64(t.hidden, &want), max_rel(t.hidden, &want)));
                    }
                    if let Some(r) = routers.get(&format!("router_ids_L{l}")) {
                        let want: Vec<i32> =
                            r.2.as_chunks::<4>()
                                .0
                                .iter()
                                .map(|c| i32::from_le_bytes(*c))
                                .collect();
                        for tok in 0..s {
                            let mut a: Vec<i32> = t.router_ids[tok * k..(tok + 1) * k]
                                .iter()
                                .map(|&v| v as i32)
                                .collect();
                            let mut b = want[tok * k..(tok + 1) * k].to_vec();
                            a.sort();
                            b.sort();
                            router_sets += usize::from(a != b);
                        }
                    }
                },
            )
            .unwrap();
        let wall = t0.elapsed().as_secs_f64();
        let by_layer = |l: usize| rows.iter().find(|r| r.0 == l).copied();
        for l in [0, 1, 2] {
            let (_, c, r) = by_layer(l).expect("layers 0-2 are dumped");
            eprintln!("  {id} layer {l} (DeepStack applied): cosine {c:.9} max|d|/peak {r:.2e}");
            if tight {
                assert!(c >= 0.999999, "{id}: layer {l} cosine {c}");
            }
        }
        let worst = rows.iter().fold(
            (1f64, 0usize),
            |m, r| if r.1 < m.0 { (r.1, r.0) } else { m },
        );
        let fcos = cosine64(&out.final_normed, &f32s(&npz["final_normed"].2));
        let frel = max_rel(&out.final_normed, &f32s(&npz["final_normed"].2));
        eprintln!(
            "== {id}: {s} tokens, {wall:.1}s, worst layer cosine {:.9} (layer {}), final_normed cosine {fcos:.9} max|d|/peak {frel:.2e}, router set mismatches {router_sets}",
            worst.0, worst.1
        );
        let qs = m["questions"].as_array().unwrap();
        let spans: Vec<QuestionSpans> = qs
            .iter()
            .map(|q| {
                let span = |v: &Value| {
                    (
                        v[0].as_u64().unwrap() as usize,
                        v[1].as_u64().unwrap() as usize,
                    )
                };
                QuestionSpans {
                    question_type: q["type"].as_u64().unwrap() as usize,
                    question_span: span(&q["question_span"]),
                    option_spans: q["option_spans"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(span)
                        .collect(),
                }
            })
            .collect();
        let logits = model
            .head
            .forward(&out.final_normed, &ids, &spans, &model.lm_head)
            .unwrap();
        let mut dp_max = 0f32;
        for (q, got) in qs.iter().zip(&logits) {
            let qid = q["id"].as_str().unwrap();
            let want = f32s(&npz[&format!("probs_{qid}")].2);
            let p = softmax(got);
            let dp = want
                .iter()
                .zip(&p)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let top = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
            eprintln!(
                "  {id}/{qid}: max|dp| {dp:.3e} top-1 {} vs oracle {}",
                top(&p),
                top(&want)
            );
            assert_eq!(top(&p), top(&want), "{id}/{qid}: top-1");
            dp_max = dp_max.max(dp);
        }
        if tight {
            assert!(
                worst.0 >= 0.99999,
                "{id}: layer {} cosine {}",
                worst.1,
                worst.0
            );
            assert!(fcos >= 0.9999, "{id}: final cosine {fcos}");
            assert!(dp_max <= 1e-3, "{id}: max |dp| {dp_max}");
        } else {
            assert!(dp_max <= 0.1, "{id}: max |dp| {dp_max}");
        }
    }
}

#[test]
#[ignore = "streams the 66 GiB BF16 checkpoint per record; needs the oracle dumps (CLEF_IDS=a,b)"]
fn media_backbone_matches_the_oracle_on_the_bf16_cpu_path() {
    let dir = PathBuf::from(std::env::var("CLEF_MODEL_DIR").unwrap_or_else(|_| MODEL_DIR.into()));
    if !dir.join("model.safetensors.index.json").exists() {
        eprintln!("skipping: no BF16 checkpoint at {}", dir.display());
        return;
    }
    let model = ClefModel::load_kind(&dir, 8, EngineKind::Cpu).unwrap();
    let ids = std::env::var("CLEF_IDS")
        .unwrap_or_else(|_| "media-image-small,media-audio-5s,media-video-audio-small".into());
    run_gate(&model, &ids, true);
}

#[test]
#[ignore = "streams the AutoRound shards through the host emulation (ROCKET_CLEF_EMU semantics); needs the oracle dumps"]
fn media_backbone_matches_the_oracle_on_the_emulated_int8_path() {
    let dir =
        PathBuf::from(std::env::var("CLEF_AUTOROUND_DIR").unwrap_or_else(|_| AUTOROUND_DIR.into()));
    if !dir.join("model.safetensors.index.json").exists() {
        eprintln!("skipping: no AutoRound checkpoint at {}", dir.display());
        return;
    }
    let model = ClefModel::load_kind(&dir, 8, EngineKind::Emu).unwrap();
    let ids = std::env::var("CLEF_IDS")
        .unwrap_or_else(|_| "media-image-small,media-audio-5s,media-video-audio-small".into());
    run_gate(&model, &ids, false);
}
