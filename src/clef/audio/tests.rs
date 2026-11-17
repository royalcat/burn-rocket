use super::*;
use crate::clef::encode::encode_request;
use crate::clef::media::{AUDIO_PAD, Positions, rope_index};
use crate::clef::pipeline::EngineKind;
use crate::clef::request::ClefRequest;
use crate::clef::testutil::{autoround_dir, f32s, model_dir};
use crate::clef::vision::ProbedVideo;
use crate::clef::vision::tests::{
    answers_gate, app_path, compare, dir_from_env, dump_of, fixture, have_ffmpeg, i64s, is_invalid,
    oracle, tokenizer,
};
use serde_json::Value;
use std::path::PathBuf;

/// Records of the oracle with audio (explicit clips and/or soundtracks).
const AUDIO_IDS: [&str; 4] = [
    "media-audio-5s",
    "media-video-audio-small",
    "media-video-test",
    "media-mixed-av",
];

// ---- config / structure units ----------------------------------------------------------

#[test]
fn the_shipped_config_is_valid_and_matches_the_geometry_of_the_docs() {
    let c = AudioConfig::shipped();
    c.validate().unwrap();
    assert_eq!(
        (
            c.head_dim(),
            c.mel_out(),
            c.steps(),
            c.conv_flat(),
            c.window(13)
        ),
        (64, 16, 13, 7680, 104)
    );
    if let Some(dir) = dir_from_env("CLEF_MODEL_DIR", model_dir()) {
        assert_eq!(AudioConfig::from_dir(&dir).unwrap(), c);
    }
}

#[test]
fn chunks_cut_every_clip_into_100_frame_pieces() {
    let c = chunks_of(&[250, 100, 7]);
    let want = [
        (0, 0, 100),
        (0, 100, 100),
        (0, 200, 50),
        (1, 0, 100),
        (2, 0, 7),
    ];
    assert_eq!(c.len(), want.len());
    for (ch, (clip, start, len)) in c.iter().zip(want) {
        assert_eq!((ch.clip, ch.start, ch.len), (clip, start, len));
    }
    // the kept steps: 13 per full chunk, the strided tail otherwise
    let kept: Vec<usize> = c.iter().map(|ch| audio_out_len(ch.len)).collect();
    assert_eq!(kept, [13, 13, 7, 13, 1]);
}

#[test]
fn windows_restart_at_every_clip_and_end_short() {
    // 208 tokens = two full 104-token windows; 130 = one full + 26; windows never cross clips
    assert_eq!(
        window_bounds(&[208, 130], 104),
        [(0, 104), (104, 208), (208, 312), (312, 338)]
    );
    // a batch of short chunks only: the window is `max_valid * 8` (a 7-step chunk -> 56)
    assert_eq!(window_bounds(&[7, 7], 56), [(0, 7), (7, 14)]);
    assert_eq!(window_bounds(&[120], 104), [(0, 104), (104, 120)]);
}

#[test]
fn the_sinusoid_table_restarts_per_chunk_and_matches_the_formula() {
    let t = sinusoid_table(13, 1280);
    assert_eq!(t.len(), 13 * 1280);
    // position 0: sin = 0, cos = 1
    assert!(t[..640].iter().all(|&v| v == 0.0));
    assert!(t[640..1280].iter().all(|&v| v == 1.0));
    // channel j of position p: sin(p * exp(-ln(1e4)/639 * j))
    for (p, j) in [(1usize, 1usize), (5, 100), (12, 639)] {
        let w = (-(10000f64.ln() / 639.0) * j as f64).exp();
        let want = (p as f64 * w).sin();
        assert!(
            (f64::from(t[p * 1280 + j]) - want).abs() < 2e-6,
            "p{p} j{j}"
        );
        let want = (p as f64 * w).cos();
        assert!(
            (f64::from(t[p * 1280 + 640 + j]) - want).abs() < 2e-6,
            "p{p} j{j}"
        );
    }
}

fn lcg_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Naive `Conv2d(k3, s2, p1)` on a channel-major `[b, c, h, w]` activation, weight `[oc, c, 3, 3]`.
fn direct_conv(
    src: &[f32],
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    wt: &[f32],
    oc: usize,
) -> Vec<f32> {
    let (ho, wo) = (sub2(h), sub2(w));
    let mut out = vec![0f32; b * oc * ho * wo];
    for bi in 0..b {
        for o in 0..oc {
            for f in 0..ho {
                for t in 0..wo {
                    let mut acc = 0f32;
                    for ic in 0..c {
                        for kh in 0..3 {
                            for kw in 0..3 {
                                let (y, x) = ((2 * f + kh) as isize - 1, (2 * t + kw) as isize - 1);
                                if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
                                    continue;
                                }
                                acc += wt[((o * c + ic) * 3 + kh) * 3 + kw]
                                    * src[((bi * c + ic) * h + y as usize) * w + x as usize];
                            }
                        }
                    }
                    out[((bi * oc + o) * ho + f) * wo + t] = acc;
                }
            }
        }
    }
    out
}

#[test]
fn im2col_plus_a_matmul_is_the_stride_2_padded_convolution() {
    let (b, c, h, w, oc) = (2usize, 3usize, 10usize, 9usize, 5usize);
    let src = lcg_vec(1, b * c * h * w, 1.0);
    let wt = lcg_vec(2, oc * c * 9, 0.5);
    let want = direct_conv(&src, b, c, h, w, &wt, oc);
    let (cols, ho, wo) = im2col(&src, b, c, h, w);
    assert_eq!((ho, wo), (5, 5));
    let y = super::super::math::matmul_nt(&cols, b * ho * wo, c * 9, &wt, oc, 1);
    for bi in 0..b {
        for o in 0..oc {
            for p in 0..ho * wo {
                let (g, e) = (
                    y[(bi * ho * wo + p) * oc + o],
                    want[(bi * oc + o) * ho * wo + p],
                );
                assert!((g - e).abs() < 1e-4, "b{bi} o{o} p{p}: {g} vs {e}");
            }
        }
    }
}

#[test]
fn the_first_conv_equals_the_naive_convolution_with_gelu() {
    let (mels, frames, oc) = (128usize, 100usize, 4usize);
    let x = lcg_vec(3, mels * frames, 1.0);
    let w = lcg_vec(4, oc * 9, 0.4);
    let bias = lcg_vec(5, oc, 0.2);
    let got = conv1(&x, 1, &w, &bias, mels, frames);
    let want = direct_conv(&x, 1, 1, mels, frames, &w, oc);
    let (ho, wo) = (64, 50);
    for o in 0..oc {
        for p in 0..ho * wo {
            let e = crate::clef::math::gelu(want[o * ho * wo + p] + bias[o]);
            assert!((got[o * ho * wo + p] - e).abs() < 1e-5, "o{o} p{p}");
        }
    }
}

// ---- a synthetic checkpoint: open / index / run / window semantics ------------------------

/// A tiny tower (2 layers, d 32, 2 heads of 16, FFN 48, downsample 8, out 24) with random BF16
/// weights.
fn synthetic_tower_dir(tag: &str) -> PathBuf {
    use crate::clef::pack::build::write_safetensors;
    use crate::clef::st::Dtype;
    let d = std::env::temp_dir().join(format!("clef-audio-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let (dm, ffn, ds, out, layers) = (32usize, 48usize, 8usize, 24usize, 2usize);
    let ac = serde_json::json!({
        "d_model": dm, "encoder_layers": layers, "encoder_attention_heads": 2,
        "encoder_ffn_dim": ffn, "num_mel_bins": 128, "downsample_hidden_size": ds,
        "output_dim": out, "n_window": 50, "n_window_infer": 800,
    });
    std::fs::write(
        d.join("config.json"),
        serde_json::to_vec(&serde_json::json!({"thinker_config": {"audio_config": ac}})).unwrap(),
    )
    .unwrap();
    let mut t: Vec<(String, Vec<usize>, f32, bool)> = Vec::new();
    let mut push = |n: &str, s: &[usize], sc: f32, unit: bool| {
        t.push((format!("{PREFIX}{n}"), s.to_vec(), sc, unit))
    };
    push("conv2d1.weight", &[ds, 1, 3, 3], 0.4, false);
    push("conv2d1.bias", &[ds], 0.1, false);
    for c in [2, 3] {
        push(&format!("conv2d{c}.weight"), &[ds, ds, 3, 3], 0.15, false);
        push(&format!("conv2d{c}.bias"), &[ds], 0.1, false);
    }
    push("conv_out.weight", &[dm, ds * 16], 0.1, false);
    for l in 0..layers {
        let p = format!("layers.{l}.");
        for (n, s, sc, unit) in [
            ("self_attn_layer_norm.weight", vec![dm], 0.1, true),
            ("self_attn_layer_norm.bias", vec![dm], 0.1, false),
            ("self_attn.q_proj.weight", vec![dm, dm], 0.3, false),
            ("self_attn.q_proj.bias", vec![dm], 0.1, false),
            ("self_attn.k_proj.weight", vec![dm, dm], 0.3, false),
            ("self_attn.k_proj.bias", vec![dm], 0.1, false),
            ("self_attn.v_proj.weight", vec![dm, dm], 0.3, false),
            ("self_attn.v_proj.bias", vec![dm], 0.1, false),
            ("self_attn.out_proj.weight", vec![dm, dm], 0.2, false),
            ("self_attn.out_proj.bias", vec![dm], 0.1, false),
            ("final_layer_norm.weight", vec![dm], 0.1, true),
            ("final_layer_norm.bias", vec![dm], 0.1, false),
            ("fc1.weight", vec![ffn, dm], 0.2, false),
            ("fc1.bias", vec![ffn], 0.1, false),
            ("fc2.weight", vec![dm, ffn], 0.15, false),
            ("fc2.bias", vec![dm], 0.1, false),
        ] {
            push(&format!("{p}{n}"), &s, sc, unit);
        }
    }
    push("ln_post.weight", &[dm], 0.1, true);
    push("ln_post.bias", &[dm], 0.1, false);
    push("proj1.weight", &[dm, dm], 0.2, false);
    push("proj1.bias", &[dm], 0.1, false);
    push("proj2.weight", &[out, dm], 0.2, false);
    push("proj2.bias", &[out], 0.1, false);
    let mut seed = 7u64;
    let mut wm = serde_json::Map::new();
    let mut shard: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, shape, sc, unit) in &t {
        let n: usize = shape.iter().product();
        seed += 1;
        let bytes: Vec<u8> = lcg_vec(seed, n, *sc)
            .into_iter()
            .flat_map(|v| {
                let v = v + if *unit { 1.0 } else { 0.0 };
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        shard.push((name.clone(), Dtype::Bf16, shape.clone(), bytes));
        wm.insert(
            name.clone(),
            Value::String("model-00001-of-00001.safetensors".into()),
        );
    }
    write_safetensors(&d.join("model-00001-of-00001.safetensors"), &shard).unwrap();
    std::fs::write(
        d.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": wm})).unwrap(),
    )
    .unwrap();
    d
}

/// Mel features for two synthetic clips of `a_frames` / `b_frames` valid frames.
fn synthetic_mel(a_frames: usize, b_frames: usize, seed_b: u64) -> MelBatch {
    let frames = a_frames.max(b_frames);
    let mk = |valid: usize, seed: u64| {
        let mut v = lcg_vec(seed, 128 * frames, 0.6);
        // frames past the valid length are padding (any value: they are never read)
        for m in 0..128 {
            for f in valid..frames {
                v[m * frames + f] = 0.0;
            }
        }
        v
    };
    MelBatch {
        frames,
        valid: vec![a_frames, b_frames],
        feats: vec![mk(a_frames, 11), mk(b_frames, seed_b)],
    }
}

#[test]
fn a_synthetic_checkpoint_opens_and_runs_and_windows_isolate_tokens() {
    let dir = synthetic_tower_dir("run");
    assert!(AudioTower::available(&dir));
    let cpu = AudioTower::open(&dir, TowerDevice::Cpu, 1).unwrap();
    assert_eq!(cpu.config().d_model, 32);
    // clip A: 250 frames = 13 + 13 + 7 = 33 tokens; clip B: 100 frames = 13 tokens
    let mel = synthetic_mel(250, 100, 12);
    let taps: Vec<String> = ["conv_out", "layer_0", "layer_1", "ln_post"]
        .map(String::from)
        .into();
    let out = cpu.forward(&mel, 1, &taps).unwrap();
    assert_eq!(out.lens, [33, 13]);
    assert_eq!(out.features.len(), 46 * 24);
    assert_eq!(out.taps["conv_out"].len(), 46 * 32);
    assert!(out.features.iter().all(|v| v.is_finite()));
    // repeat: deterministic
    let again = cpu.forward(&mel, 1, &[]).unwrap();
    assert_eq!(again.features, out.features);
    // the second clip changing leaves the first clip's tokens untouched (windows never cross
    // clips; the features of a clip are independent of its batch mates at the tower level)
    let mel2 = synthetic_mel(250, 100, 99);
    let other = cpu.forward(&mel2, 1, &[]).unwrap();
    assert_eq!(&other.features[..33 * 24], &out.features[..33 * 24]);
    assert_ne!(&other.features[33 * 24..], &out.features[33 * 24..]);
    // a clip alone gives the same tokens as inside the batch (same max window: 13 steps)
    let solo = MelBatch {
        frames: 250,
        valid: vec![250],
        feats: vec![mel.feats[0].clone()],
    };
    let alone = cpu.forward(&solo, 1, &[]).unwrap();
    assert_eq!(alone.features, &out.features[..33 * 24]);
    // within a clip the window is 104 tokens: 250 frames is 33 tokens, one window, so the
    // first chunk's tokens DO see the last ones; a long clip (1500 frames = 195 tokens, two
    // windows) isolates its first window from the tail
    let long = |tail_seed: u64| {
        let frames = 1500;
        let mut feats = lcg_vec(21, 128 * frames, 0.6);
        let tail = lcg_vec(tail_seed, 128 * frames, 0.6);
        // frames >= 1100 come from `tail` (the token index of frame 1100 is 143 > 104)
        for m in 0..128 {
            for f in 1100..frames {
                feats[m * frames + f] = tail[m * frames + f];
            }
        }
        MelBatch {
            frames,
            valid: vec![frames],
            feats: vec![feats],
        }
    };
    let (la, lb) = (
        cpu.forward(&long(5), 1, &[]).unwrap(),
        cpu.forward(&long(6), 1, &[]).unwrap(),
    );
    assert_eq!(la.lens, [195]);
    // tokens 0..104 form the first window and depend on chunks 0..8 only (frames < 800)
    assert_eq!(&la.features[..104 * 24], &lb.features[..104 * 24]);
    assert_ne!(&la.features[150 * 24..], &lb.features[150 * 24..]);
    // the fp16 emulation follows the f32 tower closely
    let emu = AudioTower::open(&dir, TowerDevice::Emu16, 1).unwrap();
    let e = emu.forward(&mel, 1, &[]).unwrap();
    let (cos, _, rel) = compare(&e.features, &out.features);
    assert!(cos > 0.999 && rel < 0.05, "emu16: cos {cos} rel {rel}");
    // the placement check reports the agreement
    let report = emu.check_against_cpu(&mel, 1).unwrap();
    assert!(report.contains("cos 0.99"), "{report}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_tower_dir_without_audio_tensors_is_not_available() {
    let d = std::env::temp_dir().join(format!("clef-audio-none-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    assert!(!AudioTower::available(&d));
    std::fs::write(
        d.join("model.safetensors.index.json"),
        br#"{"weight_map": {"thinker.model.norm.weight": "a.safetensors"}}"#,
    )
    .unwrap();
    assert!(!AudioTower::available(&d));
    std::fs::remove_dir_all(&d).ok();
}

// ---- requests / encoding -----------------------------------------------------------------

fn audio_request(audio: Vec<String>) -> ClefRequest {
    ClefRequest::from_value(serde_json::json!({
        "model": "clef-omni", "state": "a clip", "audio": audio,
        "questions": {"q": {"type": "noul"}},
    }))
    .unwrap()
}

#[test]
fn audio_requests_encode_decode_failures_and_caps_are_client_errors() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let Some(tok) = tokenizer() else { return };
    let wav = app_path("data/speech5s.wav");
    if !wav.exists() {
        return;
    }
    let s = |p: &PathBuf| p.to_string_lossy().to_string();
    let enc = encode_request(&tok, &audio_request(vec![s(&wav)]), 64000).unwrap();
    let media = enc.media.as_ref().unwrap();
    assert_eq!(media.layout.audio_frames, [500]);
    assert_eq!(media.layout.explicit_audio, 1);
    assert!(!media.layout.use_audio_in_video);
    assert_eq!(
        enc.input_ids.iter().filter(|&&t| t == AUDIO_PAD).count(),
        65
    );
    // audio-only records get plain text positions (HF's else branch)
    let pos = rope_index(&enc.input_ids, &media.layout).unwrap();
    assert_eq!(pos, Positions::text(enc.input_ids.len()));
    // a data: URL of the same file gives the same record
    let bytes = std::fs::read(&wav).unwrap();
    let url = format!(
        "data:audio/wav;base64,{}",
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes)
    );
    let enc2 = encode_request(&tok, &audio_request(vec![url]), 64000).unwrap();
    assert_eq!(enc2.input_ids, enc.input_ids);
    assert_eq!(enc2.media.unwrap().audio, media.audio);
    // two clips: ids hold two audio blocks, one mel batch
    let enc3 = encode_request(&tok, &audio_request(vec![s(&wav), s(&wav)]), 64000).unwrap();
    assert_eq!(enc3.media.as_ref().unwrap().layout.audio_frames, [500, 500]);
    // the token cap counts the audio block (65 pads + markers)
    let err = encode_request(&tok, &audio_request(vec![s(&wav)]), 100).unwrap_err();
    assert!(is_invalid(&err), "{err:#}");
    // undecodable / missing / no audio stream / too short: 422 naming the entry
    for (value, what) in [
        (s(&app_path("Cargo.toml")), "audio[0]"),
        ("/nonexistent.wav".to_string(), "audio[0]"),
        (s(&app_path("data/clef-tiny.png")), "audio[0]"),
    ] {
        let err = encode_request(&tok, &audio_request(vec![value.clone()]), 64000).unwrap_err();
        assert!(
            is_invalid(&err) && format!("{err:#}").contains(what),
            "{value}: {err:#}"
        );
    }
    let short = std::env::temp_dir().join(format!("clef-short-{}.wav", std::process::id()));
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=0.01",
        ])
        .args(["-ar", "16000", "-ac", "1"])
        .arg(&short)
        .status()
        .is_ok_and(|s| s.success());
    if ok {
        let err = encode_request(&tok, &audio_request(vec![s(&short)]), 64000).unwrap_err();
        assert!(
            is_invalid(&err) && format!("{err:#}").contains("too short"),
            "{err:#}"
        );
    }
    std::fs::remove_file(&short).ok();
    // the length cap (this port's guard): a 3 s clip against ROCKET_CLEF_AUDIO_MAX_SECONDS=1
    let long = std::env::temp_dir().join(format!("clef-long-{}.wav", std::process::id()));
    let ok = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3"])
        .args(["-ar", "16000", "-ac", "1"])
        .arg(&long)
        .status()
        .is_ok_and(|s| s.success());
    if ok {
        // decoding stops one second past the cap (here 1 s): a 3 s clip arrives as <= 2 s of
        // samples, which is over the cap, so a request with that cap is refused
        let value = s(&long);
        let samples = decode::decode_first_stream(std::path::Path::new(&value), 16000).unwrap();
        assert!(
            samples.len() > 16000 && samples.len() <= 2 * 16000 + 16,
            "{}",
            samples.len()
        );
        let err = decode::check_length_with(samples.len(), 1, "audio[0]").unwrap_err();
        assert!(
            is_invalid(&err) && format!("{err:#}").contains("1 s"),
            "{err:#}"
        );
        // and at the default cap the same clip is fine
        assert!(decode::check_length(3 * 16000, "audio[0]").is_ok());
    }
    std::fs::remove_file(&long).ok();
}

// ---- oracle gates ----------------------------------------------------------------------------

/// The oracle's mel batch of a record: `[n, 128, F]` features and the valid lengths.
fn dump_mel(id: &str) -> Option<MelBatch> {
    let path = dump_of(id)?;
    let d = oracle(
        &path,
        &["media_input_features", "media_feature_attention_mask"],
    );
    let (_, shape, bytes) = &d["media_input_features"];
    let (n, frames) = (shape[0], shape[2]);
    let all = f32s(bytes);
    let (_, mshape, mbytes) = &d["media_feature_attention_mask"];
    let mask: Vec<i32> = mbytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    let mf = mshape[1];
    let valid = (0..n)
        .map(|i| {
            mask[i * mf..(i + 1) * mf]
                .iter()
                .filter(|&&v| v != 0)
                .count()
        })
        .collect();
    let feats = (0..n)
        .map(|i| all[i * 128 * frames..(i + 1) * 128 * frames].to_vec())
        .collect();
    Some(MelBatch {
        frames,
        valid,
        feats,
    })
}

/// The audio clips of a fixture request, decoded the way `encode_request` does (explicit
/// clips, then the soundtracks of every video when all have one).
fn decoded_clips(req: &ClefRequest) -> Vec<Vec<f32>> {
    let mut clips: Vec<Vec<f32>> = req
        .audio
        .iter()
        .enumerate()
        .map(|(i, v)| load_audio(v, i).unwrap())
        .collect();
    let probed: Vec<ProbedVideo> = req
        .videos
        .iter()
        .enumerate()
        .map(|(i, v)| ProbedVideo::open(v, i).unwrap())
        .collect();
    if !probed.is_empty() && probed.iter().all(|v| v.probe.has_audio_stream) {
        let tracks: Vec<Option<Vec<f32>>> =
            probed.iter().map(|v| v.soundtrack().unwrap()).collect();
        if tracks.iter().all(Option::is_some) {
            clips.extend(tracks.into_iter().flatten());
        }
    }
    clips
}

#[test]
fn decoded_samples_and_log_mel_match_the_oracle() {
    if !have_ffmpeg() {
        return;
    }
    for id in AUDIO_IDS {
        let (Some(path), Some(want_mel)) = (dump_of(id), dump_mel(id)) else {
            continue;
        };
        let req = ClefRequest::from_value(fixture(id)).unwrap();
        let clips = decoded_clips(&req);
        assert_eq!(clips.len(), want_mel.clips(), "{id}: clip count");
        for (i, clip) in clips.iter().enumerate() {
            let key = format!("media_audio_samples_{i}");
            let want = f32s(&oracle(&path, &[&key])[&key].2);
            assert_eq!(clip.len(), want.len(), "{id} clip {i}: sample count");
            let differ = clip
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "{id} clip {i}: {} samples, {differ} differ from PyAV",
                clip.len()
            );
            assert_eq!(
                differ, 0,
                "{id} clip {i}: decoded samples differ from PyAV's"
            );
        }
        let refs: Vec<&[f32]> = clips.iter().map(Vec::as_slice).collect();
        let got = log_mel_batch(&refs);
        assert_eq!(got.frames, want_mel.frames, "{id}");
        assert_eq!(got.valid, want_mel.valid, "{id}: valid lengths");
        let (mut worst, mut at) = (0f32, (0usize, 0usize, 0usize));
        let (mut n_big, mut total) = (0usize, 0usize);
        for (ci, (g, w)) in got.feats.iter().zip(&want_mel.feats).enumerate() {
            for (i, (a, b)) in g.iter().zip(w).enumerate() {
                let d = (a - b).abs();
                total += 1;
                n_big += usize::from(d > 1e-5);
                if d > worst {
                    worst = d;
                    at = (ci, i / got.frames, i % got.frames);
                }
            }
        }
        eprintln!(
            "{id}: log-mel max |d| {worst:.3e} at (clip, mel, frame) {at:?} (value {:.4} vs {:.4}); \
             {n_big} of {total} values differ by more than 1e-5",
            got.feats[at.0][at.1 * got.frames + at.2],
            want_mel.feats[at.0][at.1 * got.frames + at.2]
        );
        assert!(worst < 2e-4, "{id}: log-mel max |d| {worst}");
    }
}

#[test]
fn audio_records_get_the_oracle_ids_positions_and_counts() {
    if !have_ffmpeg() {
        return;
    }
    let Some(tok) = tokenizer() else { return };
    for id in AUDIO_IDS {
        let Some(path) = dump_of(id) else { continue };
        let req = ClefRequest::from_value(fixture(id)).unwrap();
        let enc = encode_request(&tok, &req, 64000).unwrap();
        let want = oracle(
            &path,
            &["input_ids", "position_ids", "media_use_audio_in_video"],
        );
        let want_ids: Vec<u32> = i64s(&want["input_ids"].2)
            .iter()
            .map(|&v| v as u32)
            .collect();
        assert_eq!(enc.input_ids, want_ids, "{id}: ids differ from the oracle");
        let media = enc.media.as_ref().unwrap();
        let aiv = want["media_use_audio_in_video"].2[0] != 0;
        assert_eq!(media.layout.use_audio_in_video, aiv, "{id}");
        let pos = rope_index(&enc.input_ids, &media.layout).unwrap();
        assert_eq!(
            pos,
            Positions::from_flat(&f32s(&want["position_ids"].2)),
            "{id}: positions"
        );
        let mel = media.audio.as_ref().unwrap();
        let want_mel = dump_mel(id).unwrap();
        assert_eq!(mel.valid, want_mel.valid, "{id}");
        let worst = mel
            .feats
            .iter()
            .zip(&want_mel.feats)
            .flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (a - b).abs()))
            .fold(0f32, f32::max);
        eprintln!("{id}: ids/positions equal, mel max |d| {worst:.3e}");
        assert!(worst < 2e-4);
    }
}

/// The tower on the oracle's own features against the dumped taps.
fn tower_gate(dir: PathBuf, device: TowerDevice, ids: &[&str], bounds: (f64, f64)) {
    let tower = AudioTower::open(&dir, device, 3).unwrap();
    let taps: Vec<String> = ["conv_out", "layer_0", "layer_15", "layer_31", "ln_post"]
        .map(String::from)
        .into();
    for id in ids {
        let (Some(path), Some(mel)) = (dump_of(id), dump_mel(id)) else {
            continue;
        };
        let t0 = std::time::Instant::now();
        let out = tower.forward(&mel, 8, &taps).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let want = oracle(
            &path,
            &[
                "audio_conv_out",
                "audio_layer_0",
                "audio_layer_15",
                "audio_layer_31",
                "audio_ln_post",
                "audio_features",
            ],
        );
        for (tap, key) in [
            ("conv_out", "audio_conv_out"),
            ("layer_0", "audio_layer_0"),
            ("layer_15", "audio_layer_15"),
            ("layer_31", "audio_layer_31"),
            ("ln_post", "audio_ln_post"),
        ] {
            let w = f32s(&want[key].2);
            let (cos, md, rel) = compare(&out.taps[tap], &w);
            eprintln!("{id} {tap}: cos {cos:.9} max|d| {md:.3e} rel-to-peak {rel:.3e}");
            assert!(cos >= bounds.0, "{id} {tap}: cosine {cos}");
        }
        let w = f32s(&want["audio_features"].2);
        let (cos, md, rel) = compare(&out.features, &w);
        eprintln!(
            "{id} features: cos {cos:.9} max|d| {md:.3e} rel-to-peak {rel:.3e} ({} tokens, {secs:.1}s)",
            out.features.len() / 2048
        );
        assert!(cos >= bounds.0, "{id}: features cosine {cos}");
        assert!(rel <= bounds.1, "{id}: features rel-to-peak {rel}");
    }
}

#[test]
#[ignore = "runs the 32-layer f32 tower in release; needs the BF16 checkpoint and the dumps"]
fn the_f32_tower_matches_the_oracle() {
    let Some(dir) = dir_from_env("CLEF_MODEL_DIR", model_dir()) else {
        return;
    };
    tower_gate(dir, TowerDevice::Cpu, &AUDIO_IDS, (0.999_999, 1e-4));
}

#[test]
#[ignore = "runs the 32-layer tower through the fp16 emulation; needs the BF16 checkpoint and the dumps"]
fn the_fp16_emulated_tower_follows_the_oracle() {
    let Some(dir) = dir_from_env("CLEF_MODEL_DIR", model_dir()) else {
        return;
    };
    tower_gate(dir, TowerDevice::Emu16, &AUDIO_IDS, (0.9999, 0.1));
}

#[test]
#[ignore = "runs the tower from the AutoRound repo's shards; needs the checkpoint and the dumps"]
fn the_tower_opens_on_the_autoround_shards_too() {
    let Some(dir) = dir_from_env("CLEF_AUTOROUND_DIR", autoround_dir()) else {
        return;
    };
    tower_gate(
        dir,
        TowerDevice::Cpu,
        &["media-audio-5s"],
        (0.999_999, 1e-4),
    );
}

fn audio_gate_ids() -> Vec<String> {
    std::env::var("CLEF_IDS")
        .unwrap_or_else(|_| AUDIO_IDS.join(","))
        .split(',')
        .map(String::from)
        .collect()
}

#[test]
#[ignore = "streams the BF16 checkpoint through the f32 CPU thinker (~30 s/record); needs the dumps"]
fn audio_answers_match_the_oracle_on_the_bf16_cpu_path() {
    let Some(dir) = dir_from_env("CLEF_MODEL_DIR", model_dir()) else {
        return;
    };
    let ids = audio_gate_ids();
    let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
    let bound = if crate::clef::autoround_store::ShardStore::is_autoround(&dir) {
        0.03
    } else {
        5e-3
    };
    answers_gate(dir, EngineKind::Cpu, &ids, bound);
}

#[test]
#[ignore = "streams the AutoRound shards through the emulated int8 thinker; needs the dumps"]
fn audio_answers_stay_close_to_the_oracle_on_the_emulated_int8_path() {
    let Some(dir) = dir_from_env("CLEF_AUTOROUND_DIR", autoround_dir()) else {
        return;
    };
    let ids = audio_gate_ids();
    let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
    answers_gate(dir, EngineKind::Emu, &ids, 0.12);
}
