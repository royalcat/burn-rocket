use super::*;
use crate::clef::Invalid;
use crate::clef::encode::EncodedQuestion;
use crate::clef::encode::encode_record;
use crate::clef::testutil::{autoround_dir, f32s, model_dir, read_npz};
use serde_json::json;
use std::path::PathBuf;

const DUMP_DIR: &str = "/tmp/opencode/clef/dump";

fn eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| eq(v, w)))
        }
        _ => a == b,
    }
}

fn two_question_request() -> ClefRequest {
    ClefRequest::from_value(json!({
        "model": "clef-omni",
        "state": "x",
        "questions": {
            "zeta": {"type": "choice", "criteria": {"paid": "p", "draft": "d"}},
            "alpha": {"type": "noul"},
        },
    }))
    .unwrap()
}

fn encoded_for(req: &ClefRequest) -> EncodedRecord {
    let questions = req
        .questions
        .iter()
        .map(|(id, q)| {
            let option_ids: Vec<String> = crate::clef::answer::option_ids(q).unwrap();
            EncodedQuestion {
                id: id.clone(),
                question_type: 0,
                question_span: (0, 1),
                option_spans: option_ids.iter().map(|_| (0, 1)).collect(),
                option_ids,
            }
        })
        .collect();
    EncodedRecord {
        input_ids: vec![0; 42],
        questions,
        media: None,
    }
}

#[test]
fn the_response_has_the_systemone_shape_in_request_order() {
    let req = two_question_request();
    let enc = encoded_for(&req);
    // zeta scores [draft, paid]; alpha scores [true, false].
    let a = assemble(&req, &enc, &[vec![0.0, 2.0], vec![1.0, 1.0]]).unwrap();
    let keys: Vec<&String> = a.response["answers"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["zeta", "alpha"]);
    assert_eq!(a.response["model"], "clef-omni");
    assert_eq!(
        a.response["usage"],
        json!({"input_tokens": 42, "output_tokens": 0})
    );
    let z = &a.response["answers"]["zeta"];
    assert_eq!(z["choice"], "paid");
    assert_eq!(z["confidence"], 0.8808);
    // `probabilities` follow the request's criteria order (paid, draft), not the scored one.
    let order: Vec<&String> = z["probabilities"].as_object().unwrap().keys().collect();
    assert_eq!(order, ["paid", "draft"]);
    assert_eq!(
        a.response["answers"]["alpha"],
        json!({"type": "noul", "noul": 0.5})
    );
    assert_eq!((a.tokens, a.questions), (42, 2));
    assert_eq!(a.probabilities[0].0, "zeta");
}

#[test]
fn a_mismatched_encoding_is_an_error_not_a_panic() {
    let req = two_question_request();
    let mut enc = encoded_for(&req);
    enc.questions[0].id = "nope".into();
    assert!(assemble(&req, &enc, &[vec![0.0, 1.0], vec![0.0, 1.0]]).is_err());
}

fn fixture(id: &str) -> ClefRequest {
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("../../../tools/clef_fixtures.json")).unwrap();
    let rec = fixtures.into_iter().find(|f| f["id"] == id).unwrap();
    ClefRequest::from_value(rec).unwrap()
}

/// End to end against the f32 oracle dumps: probabilities within 1e-4 and the rounded
/// answers equal. Streams the whole 66 GiB checkpoint per record (~1-2 min each).
#[test]
#[ignore = "streams the 66 GiB checkpoint per record; run with --ignored (CLEF_IDS=a,b to select)"]
fn clef_answers_match_the_oracle() {
    let dir = PathBuf::from(model_dir());
    if !dir.join("joint_head.safetensors").exists() {
        eprintln!("skipping: {} is absent", model_dir());
        return;
    }
    let ids = std::env::var("CLEF_IDS").unwrap_or_else(|_| "checkout-errors,invoice".into());
    let model = ClefModel::load(&dir).unwrap();
    for id in ids.split(',') {
        let dump = PathBuf::from(DUMP_DIR);
        let Ok(text) = std::fs::read_to_string(dump.join(format!("{id}.json"))) else {
            eprintln!(
                "skipping {id}: no oracle dump in {DUMP_DIR} (docs/experiments/clef-oracle.md)"
            );
            continue;
        };
        let oracle: Value = serde_json::from_str(&text).unwrap();
        let npz = read_npz(&dump.join(format!("{id}.npz")));
        let t0 = std::time::Instant::now();
        let got = model.answer(&fixture(id), 64000).unwrap();
        eprintln!(
            "{id}: {} tokens in {:.1}s",
            got.tokens,
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(
            got.response["usage"], oracle["usage"],
            "{id}: usage (token count)"
        );
        let mut worst = 0f64;
        for (qid, probs) in &got.probabilities {
            let (_, _, bytes) = &npz[&format!("probs_{qid}")];
            let want = f32s(bytes);
            let q = oracle["questions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|q| q["id"] == qid.as_str())
                .unwrap();
            for (opt, w) in q["option_ids"].as_array().unwrap().iter().zip(&want) {
                let p = probs[opt.as_str().unwrap()];
                worst = worst.max((p - f64::from(*w)).abs());
            }
        }
        eprintln!("{id}: max |dp| = {worst:.3e}");
        assert!(worst <= 1e-4, "{id}: max |dp| {worst:.3e}");
        assert!(
            eq(&got.response["answers"], &oracle["answers"]),
            "{id}: answers differ\n got {}\nwant {}",
            got.response["answers"],
            oracle["answers"]
        );
        assert_eq!(got.response["model"], oracle["model"]);
    }
}

const PACK_DIR: &str = "/home/royalcat/models/clef-pack-g128";

/// End to end through the int8 pack (CPU, packed experts dequantized to f32) against the
/// f32 BF16-source oracle dumps. The int8 quantization moves probabilities (the
/// AutoRound card measured max 0.044, mean 0.006 on its own records); every question's
/// top option must agree. Prints per-question probabilities, max and mean |dp|.
#[test]
#[ignore = "needs the full clef pack (CLEF_PACK_DIR) and oracle dumps; ~1 min per record"]
fn clef_pack_answers_match_the_oracle_top1() {
    let dir = PathBuf::from(std::env::var("CLEF_PACK_DIR").unwrap_or_else(|_| PACK_DIR.into()));
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no clef pack at {}", dir.display());
        return;
    }
    let ids = std::env::var("CLEF_IDS")
        .unwrap_or_else(|_| "checkout-errors,invoice,long-incident".into());
    let model = ClefModel::load(&dir).unwrap();
    let (mut worst, mut sum, mut n) = (0f64, 0f64, 0usize);
    for id in ids.split(',') {
        let dump = PathBuf::from(DUMP_DIR);
        let Ok(text) = std::fs::read_to_string(dump.join(format!("{id}.json"))) else {
            eprintln!(
                "skipping {id}: no oracle dump in {DUMP_DIR} (docs/experiments/clef-oracle.md)"
            );
            continue;
        };
        let oracle: Value = serde_json::from_str(&text).unwrap();
        let npz = read_npz(&dump.join(format!("{id}.npz")));
        let t0 = std::time::Instant::now();
        let got = model.answer(&fixture(id), 64000).unwrap();
        eprintln!(
            "{id}: {} tokens in {:.1}s",
            got.tokens,
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(got.response["usage"], oracle["usage"], "{id}: token count");
        let (mut rec_worst, mut rec_sum, mut rec_n) = (0f64, 0f64, 0usize);
        for (qid, probs) in &got.probabilities {
            let want = f32s(&npz[&format!("probs_{qid}")].2);
            let q = oracle["questions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|q| q["id"] == qid.as_str())
                .unwrap();
            let opts: Vec<&str> = q["option_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o.as_str().unwrap())
                .collect();
            let mut top = (0usize, f64::MIN);
            let mut top_want = (0usize, f32::MIN);
            for (i, (opt, w)) in opts.iter().zip(&want).enumerate() {
                let p = probs[*opt];
                rec_worst = rec_worst.max((p - f64::from(*w)).abs());
                rec_sum += (p - f64::from(*w)).abs();
                rec_n += 1;
                if p > top.1 {
                    top = (i, p);
                }
                if *w > top_want.1 {
                    top_want = (i, *w);
                }
                eprintln!("  {id}/{qid}/{opt}: pack {p:.4} oracle {w:.4}");
            }
            assert_eq!(
                top.0, top_want.0,
                "{id}/{qid}: top-1 differs from the oracle"
            );
        }
        eprintln!(
            "{id}: max |dp| = {rec_worst:.4}, mean |dp| = {:.4} over {rec_n} options",
            rec_sum / rec_n as f64
        );
        worst = worst.max(rec_worst);
        sum += rec_sum;
        n += rec_n;
    }
    eprintln!(
        "all records: max |dp| = {worst:.4}, mean |dp| = {:.4} over {n} options",
        sum / n.max(1) as f64
    );
    assert!(n > 0, "no record ran");
    // The card's own measurement was 0.044 max / 0.006 mean on other records; far above
    // it would point at an unpacking or dequantization bug.
    assert!(worst < 0.1, "max |dp| {worst:.4}");
}

/// The W8A8 host emulation of the NPU path (`EngineKind::Emu`: int8 activations for the
/// attention projections and the experts, streamed from the pack) against the f32
/// BF16-source oracle AND against the CPU-dequant pack path, per question, plus the
/// per-layer hidden cosine vs the oracle dump (where it has the layer).
#[test]
#[ignore = "needs the full clef pack (CLEF_PACK_DIR) and oracle dumps; ~1-2 min per record"]
fn clef_emu_answers_match_the_oracle_and_the_cpu_pack_path() {
    let dir = PathBuf::from(std::env::var("CLEF_PACK_DIR").unwrap_or_else(|_| PACK_DIR.into()));
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no clef pack at {}", dir.display());
        return;
    }
    let ids = std::env::var("CLEF_IDS")
        .unwrap_or_else(|_| "checkout-errors,invoice,long-incident".into());
    let threads = default_threads();
    let cpu = ClefModel::load_kind(&dir, threads, EngineKind::Cpu).unwrap();
    let emu = ClefModel::load_kind(&dir, threads, EngineKind::Emu).unwrap();
    let layer_log = std::env::var_os("CLEF_LAYER_LOG").is_some();
    let (mut worst_o, mut sum_o, mut worst_c, mut sum_c, mut n) = (0f64, 0f64, 0f64, 0f64, 0usize);
    for id in ids.split(',') {
        let dump = PathBuf::from(DUMP_DIR);
        let Ok(text) = std::fs::read_to_string(dump.join(format!("{id}.json"))) else {
            eprintln!(
                "skipping {id}: no oracle dump in {DUMP_DIR} (docs/experiments/clef-oracle.md)"
            );
            continue;
        };
        let oracle: Value = serde_json::from_str(&text).unwrap();
        let npz = read_npz(&dump.join(format!("{id}.npz")));
        let req = fixture(id);
        let t0 = std::time::Instant::now();
        let got_cpu = cpu.answer(&req, 64000).unwrap();
        let t_cpu = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let got_emu = emu.answer(&req, 64000).unwrap();
        eprintln!(
            "{id}: {} tokens, cpu-pack {t_cpu:.1}s, emu {:.1}s",
            got_emu.tokens,
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(
            got_emu.response["usage"], oracle["usage"],
            "{id}: token count"
        );
        let (mut rw_o, mut rs_o, mut rw_c, mut rs_c, mut rn) = (0f64, 0f64, 0f64, 0f64, 0usize);
        for (qid, p_emu) in &got_emu.probabilities {
            let p_cpu = &got_cpu
                .probabilities
                .iter()
                .find(|(q, _)| q == qid)
                .unwrap()
                .1;
            let want = f32s(&npz[&format!("probs_{qid}")].2);
            let q = oracle["questions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|q| q["id"] == qid.as_str())
                .unwrap();
            let opts: Vec<&str> = q["option_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o.as_str().unwrap())
                .collect();
            let argmax = |f: &dyn Fn(usize) -> f64| {
                (0..opts.len())
                    .max_by(|&a, &b| f(a).total_cmp(&f(b)))
                    .unwrap()
            };
            for (i, (opt, w)) in opts.iter().zip(&want).enumerate() {
                let (pe, pc) = (p_emu[*opt], p_cpu[*opt]);
                rw_o = rw_o.max((pe - f64::from(*w)).abs());
                rs_o += (pe - f64::from(*w)).abs();
                rw_c = rw_c.max((pe - pc).abs());
                rs_c += (pe - pc).abs();
                rn += 1;
                eprintln!("  {id}/{qid}/{opt}: emu {pe:.4} cpu-pack {pc:.4} oracle {w:.4}");
                let _ = i;
            }
            let t_emu = argmax(&|i| p_emu[opts[i]]);
            let t_orc = argmax(&|i| f64::from(want[i]));
            assert_eq!(
                t_emu, t_orc,
                "{id}/{qid}: emu top-1 differs from the oracle"
            );
        }
        eprintln!(
            "{id}: emu vs oracle max {rw_o:.4} mean {:.4}; emu vs cpu-pack max {rw_c:.4} mean {:.4}",
            rs_o / rn as f64,
            rs_c / rn as f64
        );
        worst_o = worst_o.max(rw_o);
        sum_o += rs_o;
        worst_c = worst_c.max(rw_c);
        sum_c += rs_c;
        n += rn;

        // per-layer hidden cosine vs the oracle, both paths (whole layers the dump has)
        if layer_log || id == "checkout-errors" {
            let enc = encode_record(&emu.tokenizer, &req, 64000, None).unwrap();
            let mut rows: Vec<(usize, f32, f32)> = Vec::new();
            for (name, model, col) in [("emu", &emu, 0usize), ("cpu-pack", &cpu, 1)] {
                let _ = name;
                model
                    .backbone
                    .forward_tapped(&enc.input_ids, &model.opts, &mut |t| {
                        if let Some(w) = npz.get(&format!("hidden_L{}", t.layer)) {
                            let c = crate::clef::math::cosine(t.hidden, &f32s(&w.2));
                            if col == 0 {
                                rows.push((t.layer, c, 0.0));
                            } else if let Some(r) = rows.iter_mut().find(|r| r.0 == t.layer) {
                                r.2 = c;
                            }
                        }
                    })
                    .unwrap();
            }
            for (l, ce, cc) in &rows {
                if layer_log || *l % 8 == 7 || *l == 0 {
                    eprintln!(
                        "  {id} layer {l:2}: cosine vs oracle  emu {ce:.6}  cpu-pack {cc:.6}"
                    );
                }
            }
            if let Some(w) = rows.iter().min_by(|a, b| a.1.total_cmp(&b.1)) {
                eprintln!(
                    "  {id}: lowest emu layer cosine {:.6} at layer {}",
                    w.1, w.0
                );
            }
        }
    }
    eprintln!(
        "all records: emu vs oracle max |dp| {worst_o:.4} mean {:.4}; emu vs cpu-pack max {worst_c:.4} mean {:.4} over {n} options",
        sum_o / n.max(1) as f64,
        sum_c / n.max(1) as f64
    );
    assert!(n > 0, "no record ran");
    assert!(worst_o < 0.05, "emu vs oracle max |dp| {worst_o:.4}");
}

/// Phase 7 gate: records answered as ONE batch through the host emulation of the NPU
/// engine equal the same records answered alone, bit for bit (probabilities and answers),
/// a refused record fails alone, and the batch reads each expert once per layer. Prints
/// the wall times and the expert-read counts (the quantity that matters on the board).
#[test]
#[ignore = "needs the full clef pack (CLEF_PACK_DIR); ~1-2 min on the dev host"]
fn clef_batch_equals_single_records_on_the_emulated_engine() {
    let dir = PathBuf::from(std::env::var("CLEF_PACK_DIR").unwrap_or_else(|_| PACK_DIR.into()));
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no clef pack at {}", dir.display());
        return;
    }
    let ids = std::env::var("CLEF_IDS")
        .unwrap_or_else(|_| "checkout-errors,invoice,long-incident,noul-only".into());
    let model = ClefModel::load_kind(&dir, default_threads(), EngineKind::Emu).unwrap();
    let mut reqs: Vec<ClefRequest> = ids.split(',').map(fixture).collect();
    // a media request: refused up front, must not take the others down
    reqs.push(
        ClefRequest::from_value(json!({
            "model": "clef-omni", "state": "x", "images": ["data:image/png;base64,AAAA"],
            "questions": {"q": {"type": "noul"}},
        }))
        .unwrap(),
    );
    let refs: Vec<&ClefRequest> = reqs.iter().collect();

    let (mut single_wall, mut single_reads) = (0f64, 0usize);
    let mut singles = Vec::new();
    for r in &refs[..refs.len() - 1] {
        let t0 = std::time::Instant::now();
        singles.push(model.answer(r, 64000).unwrap());
        single_wall += t0.elapsed().as_secs_f64();
        let reads = model.backbone.last_phases().map_or(0, |p| p.experts);
        single_reads += reads;
        eprintln!(
            "single: {} tokens, {reads} expert reads, {:.1}s",
            singles.last().unwrap().tokens,
            t0.elapsed().as_secs_f64()
        );
    }
    let t0 = std::time::Instant::now();
    let batch = model.answer_batch(&refs, 64000, 4096);
    let batch_wall = t0.elapsed().as_secs_f64();
    assert_eq!(batch.chunks.len(), 1, "one shared pass");
    assert!(
        batch
            .results
            .last()
            .unwrap()
            .as_ref()
            .is_err_and(|e| e.downcast_ref::<Invalid>().is_some()),
        "the media request is refused alone"
    );
    for (i, want) in singles.iter().enumerate() {
        let got = batch.results[i].as_ref().unwrap();
        assert_eq!(got.tokens, want.tokens, "{i}: tokens");
        assert_eq!(
            got.probabilities, want.probabilities,
            "{i}: probabilities (bit-exact)"
        );
        assert!(eq(&got.response, &want.response), "{i}: answers differ");
    }
    let batch_reads = batch.experts_read();
    eprintln!(
        "batch of {} ({} rows): {} expert reads vs {single_reads} for the singles; \
         wall {batch_wall:.1}s vs {single_wall:.1}s",
        singles.len(),
        batch.chunks[0].tokens,
        batch_reads
    );
    assert!(batch_reads < single_reads);
    // the same through the row cap: 2 passes, still identical
    let capped = model.answer_batch(&refs, 64000, 300);
    assert!(capped.chunks.len() >= 2);
    for (i, want) in singles.iter().enumerate() {
        assert_eq!(
            capped.results[i].as_ref().unwrap().probabilities,
            want.probabilities,
            "{i}: capped batch"
        );
    }
}

// ---- Phase 8f: mixed media batches (server + decide share `answer_encoded`)

/// A media fixture with its `data/...` paths made absolute against the app dir (cargo runs
/// tests with the package root as CWD, but the server may not).
fn media_fixture(id: &str) -> ClefRequest {
    let all: Vec<Value> =
        serde_json::from_str(include_str!("../../../tools/clef_fixtures_media.json")).unwrap();
    let mut v = all
        .into_iter()
        .find(|c| c["id"] == id)
        .expect("media fixture");
    for key in ["images", "videos", "audio"] {
        if let Some(items) = v[key].as_array_mut() {
            for item in items {
                let rel = item.as_str().unwrap().trim_start_matches("./");
                *item = Value::String(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join(rel)
                        .to_string_lossy()
                        .into(),
                );
            }
        }
    }
    v.as_object_mut().unwrap().remove("id");
    ClefRequest::from_value(v).unwrap()
}

/// The exact mixed batch of the 8f gate: one text record, an image, an audio clip, a silent
/// video and an audio-in-video record. Plus one undecodable image for the isolation check.
fn mixed_media_ids() -> Vec<&'static str> {
    vec![
        "checkout-errors", // text (from the text fixtures)
        "media-image-small",
        "media-audio-5s",
        "media-video-silent",
        "media-video-audio-small",
    ]
}

fn mixed_requests() -> Vec<ClefRequest> {
    let mut out = vec![fixture("checkout-errors")];
    out.extend(mixed_media_ids().into_iter().skip(1).map(media_fixture));
    out
}

/// Compare a mixed batch with the same records answered alone. `emu` selects the host
/// emulation of the int8 engine (the `ROCKET_CLEF_EMU=1` AutoRound path); otherwise the
/// plain bf16 CPU path is used. `tol` is the probability tolerance: `0` pins exact f64
/// equality (EMU, where the batch is byte-identical to the singles), while the `gemm`
/// CPU path uses a small envelope — the batch changes the expert bucket's row count, so
/// its f32 summation order shifts (hidden-state relative-to-peak stays < 1e-4, pinned by
/// `batch_tests::the_cpu_f32_path_batches_to_within_float_rounding`; the head amplifies
/// that into ~2e-3 of probability).
fn mixed_media_batch_gate(dir: &Path, kind: EngineKind, tol: f64) {
    if !dir.join("model.safetensors.index.json").exists() {
        eprintln!("skipping: no checkpoint at {}", dir.display());
        return;
    }
    let model = ClefModel::load_kind(dir, default_threads(), kind).unwrap();
    let requests = mixed_requests();
    let refs: Vec<&ClefRequest> = requests.iter().collect();

    let (mut single_wall, mut single_reads) = (0f64, 0usize);
    let mut singles = Vec::new();
    for r in &refs {
        let t0 = std::time::Instant::now();
        singles.push(model.answer(r, 64000).unwrap());
        single_wall += t0.elapsed().as_secs_f64();
        single_reads += model.backbone.last_phases().map_or(0, |p| p.experts);
        let done = singles.last().unwrap();
        eprintln!(
            "single: {} tokens, {} questions, {:.1}s",
            done.tokens,
            done.questions,
            t0.elapsed().as_secs_f64()
        );
    }

    let t0 = std::time::Instant::now();
    let batch = model.answer_batch(&refs, 64000, 0); // one shared pass
    let batch_wall = t0.elapsed().as_secs_f64();
    assert_eq!(batch.chunks.len(), 1, "one shared thinker pass");
    for (i, (want, got)) in singles.iter().zip(&batch.results).enumerate() {
        let got = got.as_ref().unwrap_or_else(|e| panic!("record {i}: {e:#}"));
        assert_eq!(got.tokens, want.tokens, "record {i}: tokens");
        assert_eq!(got.questions, want.questions, "record {i}: questions");
        if tol == 0.0 {
            assert!(eq(&got.response, &want.response), "record {i}: answers");
            assert_eq!(
                got.probabilities, want.probabilities,
                "record {i}: probabilities (bit-exact)"
            );
        } else {
            // The plain f32 CPU path runs expert GEMMs on the row count of the bucket,
            // so a batch changes the f32 summation order. The hidden-state difference is
            // < 1e-4 relative to peak; the head amplifies it into a small probability
            // drift, and a discrete answer may flip only on such a near-tie (logged).
            assert_eq!(got.probabilities.len(), want.probabilities.len());
            let mut worst = 0.0f64;
            for ((qid, got_p), (want_qid, want_p)) in
                got.probabilities.iter().zip(&want.probabilities)
            {
                assert_eq!(qid, want_qid, "record {i}: question order");
                for (opt, p) in want_p {
                    let q = got_p.get(opt).unwrap();
                    worst = worst.max((p - q).abs());
                    assert!(
                        (p - q).abs() <= tol,
                        "record {i} question {qid} option {opt}: |{p} - {q}| > {tol}"
                    );
                }
            }
            if !eq(&got.response, &want.response) {
                eprintln!(
                    "record {i}: rounded confidences differ (max |dp| {worst:.2e}); \
                     single vs batch: {} / {}",
                    want.response["answers"], got.response["answers"]
                );
            }
        }
    }
    let batch_reads = batch.experts_read();
    eprintln!(
        "batch of {} mixed records ({} rows): {} expert reads vs {single_reads} singles; \
         wall {batch_wall:.1}s vs {single_wall:.1}s",
        refs.len(),
        batch.chunks[0].tokens,
        batch_reads
    );

    // Failure isolation: one undecodable media fails alone; the valid records still answer
    // bit-identical to their single runs (the bad record is never part of the shared pass).
    let mut broken = requests.clone();
    broken.push(
        ClefRequest::from_value(json!({
            "model": "clef-omni", "state": "x",
            "images": ["data:image/png;base64,AAAA"],
            "questions": {"q": {"type": "noul"}},
        }))
        .unwrap(),
    );
    let broken_refs: Vec<&ClefRequest> = broken.iter().collect();
    let isolated = model.answer_batch(&broken_refs, 64000, 0);
    assert!(
        isolated.results.last().unwrap().is_err(),
        "the undecodable image must fail alone"
    );
    for (i, want) in singles.iter().enumerate() {
        let got = isolated.results[i]
            .as_ref()
            .unwrap_or_else(|e| panic!("isolation record {i}: {e:#}"));
        if tol == 0.0 {
            assert_eq!(
                got.probabilities, want.probabilities,
                "isolation record {i}"
            );
        } else {
            for ((_, got_p), (_, want_p)) in got.probabilities.iter().zip(&want.probabilities) {
                for (opt, p) in want_p {
                    let q = got_p.get(opt).unwrap();
                    assert!(
                        (p - q).abs() <= tol,
                        "isolation record {i} option {opt}: |{p} - {q}| > {tol}"
                    );
                }
            }
        }
    }
}

/// Gate: `ROCKET_CLEF_EMU=1` on the AutoRound shards, mixed text/image/audio/video batch
/// answers byte-identical to the same records alone (and a bad media fails only itself).
#[test]
#[ignore = "needs the AutoRound shards; ~2-4 min on the dev host (CLEF_AUTOROUND_DIR)"]
fn clef_mixed_media_batch_equals_singles_on_the_emulated_engine() {
    mixed_media_batch_gate(&PathBuf::from(autoround_dir()), EngineKind::Emu, 0.0);
}

/// Gate: the BF16 CPU thinker's mixed media batch stays inside the `gemm` path's
/// float-rounding envelope (hidden state < 1e-4 relative-to-peak; the head amplifies that
/// to ~2e-3 of probability). CLEF_MODEL_DIR.
#[test]
#[ignore = "needs the BF16 checkpoint; ~2-4 min on the dev host (CLEF_MODEL_DIR)"]
fn clef_mixed_media_batch_within_the_cpu_rounding_envelope() {
    mixed_media_batch_gate(&PathBuf::from(model_dir()), EngineKind::Cpu, 5e-3);
}
