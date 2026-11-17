use super::compute::{Emu16Exec, full_attention, pad_heads, pad_heads_transposed, unpack_padded};
use super::*;
use crate::clef::encode::{encode_request, newline_token};
use crate::clef::media::MediaFeatures;
use crate::clef::pipeline::{ClefModel, EngineKind};
use crate::clef::request::ClefRequest;
use crate::clef::st::Dtype;
use crate::clef::testutil::{autoround_dir, f32s, model_dir, read_npz_keys};
use half::f16;
use image::RgbImage;
use serde_json::Value;
use std::path::PathBuf;

const DUMP_DIR: &str = "/tmp/opencode/clef/dump";

pub(crate) fn app_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

pub(crate) fn dir_from_env(var: &str, default: &str) -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.into()));
    if d.join("model.safetensors.index.json").exists() {
        Some(d)
    } else {
        eprintln!("skipping: no checkpoint at {} ({var})", d.display());
        None
    }
}

pub(crate) fn dump_of(id: &str) -> Option<PathBuf> {
    let p = PathBuf::from(DUMP_DIR).join(format!("{id}.npz"));
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping: no oracle dump {}", p.display());
        None
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

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// `(cosine, max |d|, max |d| / peak of want)`.
pub(crate) fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut dot, mut ng, mut nw, mut md, mut peak) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        dot += g * w;
        ng += g * g;
        nw += w * w;
        md = md.max((g - w).abs());
        peak = peak.max(w.abs());
    }
    (
        dot / (ng.sqrt() * nw.sqrt()).max(1e-300),
        md,
        md / peak.max(1e-30),
    )
}

fn rgb_pattern(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            (x * 5 + y * 3) as u8,
            (x * 2 + y * 11 + 7) as u8,
            (x * 13 + y) as u8,
        ])
    })
}

// ---- config / preprocessing units ----------------------------------------------------

#[test]
fn the_shipped_config_is_valid_and_matches_the_geometry_of_the_docs() {
    let c = VisionConfig::shipped();
    c.validate().unwrap();
    assert_eq!(
        (
            c.head_dim(),
            c.patch_dim(),
            c.factor(),
            c.merge_unit(),
            c.pos_side()
        ),
        (72, 1536, 32, 4, 48)
    );
    // the real config.json agrees, when a checkpoint is around
    if let Some(dir) = dir_from_env("CLEF_MODEL_DIR", model_dir()) {
        assert_eq!(VisionConfig::from_dir(&dir).unwrap(), c);
    }
}

#[test]
fn patches_are_merge_block_major_with_channels_then_time_then_pixels() {
    let cfg = VisionConfig::shipped();
    let img = rgb_pattern(64, 64); // 4x4 patch grid = 2x2 merge blocks
    let (px, grid) = patchify_pairs(&[(&img, &img)], &cfg).unwrap();
    assert_eq!(grid, [1, 4, 4]);
    assert_eq!(px.len(), 16 * 1536);
    let at = |patch: usize, c: usize, t: usize, py: usize, pxx: usize| {
        px[patch * 1536 + c * 512 + t * 256 + py * 16 + pxx]
    };
    // patch order: block (0,0): (0,0) (0,1) (1,0) (1,1); then block (0,1): (0,2) (0,3) (1,2) (1,3)
    for (patch, (row, col)) in [
        (0, (0, 0)),
        (2, (1, 0)),
        (3, (1, 1)),
        (4, (0, 2)),
        (11, (3, 1)),
        (13, (2, 3)),
    ] {
        for (c, t, py, pxx) in [(0, 0, 0, 0), (1, 1, 2, 3), (2, 0, 15, 15)] {
            let (x, y) = (col * 16 + pxx, row * 16 + py);
            let want = normalize_channel(img.get_pixel(x as u32, y as u32).0[c]);
            assert_eq!(at(patch, c, t, py, pxx), want, "patch {patch} c{c} t{t}");
        }
    }
    // a real pair: the second frame fills the second temporal slot
    let other = RgbImage::from_fn(64, 64, |x, y| image::Rgb([(x + y) as u8; 3]));
    let (pair, g) = patchify_pairs(&[(&img, &other)], &cfg).unwrap();
    assert_eq!(g, [1, 4, 4]);
    let want = normalize_channel(other.get_pixel(5, 1).0[0]);
    assert_eq!(pair[256 + 16 + 5], want);
    assert_eq!(pair[1], px[1]);
    // two groups come one after the other
    let (two, g2) = patchify_pairs(&[(&img, &img), (&other, &other)], &cfg).unwrap();
    assert_eq!(g2, [2, 4, 4]);
    assert_eq!(&two[..16 * 1536], &px[..]);
    // sizes off the 32-grid and mismatched frames are errors
    assert!(patchify_pairs(&[(&rgb_pattern(48, 64), &rgb_pattern(48, 64))], &cfg).is_err());
    assert!(patchify_pairs(&[(&img, &rgb_pattern(32, 64))], &cfg).is_err());
}

#[test]
fn images_resize_to_the_documented_grids() {
    let cfg = VisionConfig::shipped();
    for ((h, w), grid) in [
        ((64, 96), [1, 14, 20]),
        ((686, 960), [1, 42, 60]),
        ((300, 300), [1, 18, 18]),
    ] {
        let p = prepare_image(&rgb_pattern(w, h), &cfg).unwrap();
        assert_eq!(p.grid, grid, "{h}x{w}");
        assert_eq!(p.pixels.len(), p.patches() * 1536);
        assert_eq!(p.tokens(), grid[1] * grid[2] / 4);
    }
    // a 32-aligned size inside the budget is not resized at all (the 256x256 = 65536 floor)
    let p = prepare_image(&rgb_pattern(256, 256), &cfg).unwrap();
    assert_eq!(p.resized, (256, 256));
    assert_eq!(
        p.pixels[0],
        normalize_channel(rgb_pattern(256, 256).get_pixel(0, 0).0[0])
    );
    // refusals are client errors
    let err = prepare_image(&rgb_pattern(3000, 10), &cfg).unwrap_err();
    assert!(
        err.downcast_ref::<crate::clef::request::Invalid>()
            .is_some(),
        "{err:#}"
    );
}

pub(crate) fn tokenizer() -> Option<tokenizers::Tokenizer> {
    let dir = dir_from_env("CLEF_MODEL_DIR", model_dir())?;
    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).ok()
}

#[test]
fn encode_request_builds_the_image_block_and_counts_it_against_the_cap() {
    let Some(tok) = tokenizer() else { return };
    let png = app_path("data/clef-tiny.png");
    if !png.exists() {
        eprintln!("skipping: no {}", png.display());
        return;
    }
    let req = |images: Value| {
        ClefRequest::from_value(serde_json::json!({
            "model": "clef-omni", "state": "see image", "images": images,
            "questions": {"q": {"type": "noul"}},
        }))
        .unwrap()
    };
    let invalid = |e: &anyhow::Error| e.downcast_ref::<crate::clef::request::Invalid>().is_some();
    let r = req(serde_json::json!([png.to_str().unwrap()]));
    let enc = encode_request(&tok, &r, 8192).unwrap();
    let media = enc.media.as_ref().expect("image record");
    assert_eq!(media.layout.image_grids, [[1, 14, 20]]);
    let n_pad = enc
        .input_ids
        .iter()
        .filter(|&&t| t == crate::clef::media::IMAGE_PAD)
        .count();
    assert_eq!(n_pad, 70);
    // the block follows the 35-token prefix: <vs> pads <ve> \n
    let nl = newline_token(&tok).unwrap();
    assert_eq!(enc.input_ids[35], crate::clef::media::VISION_START);
    assert_eq!(enc.input_ids[35 + 71], crate::clef::media::VISION_END);
    assert_eq!(enc.input_ids[35 + 72], nl);
    // text-only requests carry no media
    let plain = encode_request(&tok, &req(serde_json::json!([])), 8192).unwrap();
    assert!(plain.media.is_none());
    // 70 media tokens + schema do not fit a 100-token cap: a client error
    let err = encode_request(&tok, &r, 100).unwrap_err();
    assert!(invalid(&err), "{err:#}");
    // an unreadable image is a client error naming the entry
    let err =
        encode_request(&tok, &req(serde_json::json!(["/nonexistent/x.png"])), 8192).unwrap_err();
    assert!(format!("{err:#}").contains("images[0]"), "{err:#}");
    assert!(invalid(&err));
    // audio/video stay refused
    let r = ClefRequest::from_value(serde_json::json!({
        "model": "m", "state": "s", "audio": ["a.wav"], "questions": {"q": {"type": "noul"}},
    }))
    .unwrap();
    assert!(invalid(&encode_request(&tok, &r, 8192).unwrap_err()));
}

// ---- math units ----------------------------------------------------------------------

#[test]
fn gelu_tanh_matches_torch_values() {
    for (x, want) in [
        (0.0f32, 0.0f32),
        (1.0, 0.841_192),
        (-1.0, -0.158_808),
        (2.0, 1.954_597_7),
    ] {
        assert!(
            (gelu_tanh(x) - want).abs() < 2e-6,
            "gelu_tanh({x}) = {}",
            gelu_tanh(x)
        );
    }
}

#[test]
fn position_embeddings_are_the_bilinear_resample_in_block_order() {
    let mut cfg = VisionConfig::shipped();
    cfg.hidden = 4;
    cfg.heads = 1;
    let mut rng = Lcg(7);
    let table = rng.vec(2304 * 4);
    for grid in [[1, 4, 6], [2, 6, 4]] {
        let (t, gh, gw) = (grid[0], grid[1], grid[2]);
        let got = position_embeddings(&table, grid, &cfg);
        assert_eq!(got.len(), t * gh * gw * 4);
        // naive F.interpolate(align_corners=True), then raster -> block-major
        let src = |i: usize, n: usize| i as f64 * 47.0 / (n - 1).max(1) as f64;
        let naive = |r: usize, c: usize, ch: usize| -> f64 {
            let (sr, sc) = (src(r, gh), src(c, gw));
            let (r0, c0) = (sr.floor() as usize, sc.floor() as usize);
            let (r1, c1) = ((r0 + 1).min(47), (c0 + 1).min(47));
            let (fr, fc) = (sr - r0 as f64, sc - c0 as f64);
            let v = |r: usize, c: usize| f64::from(table[(r * 48 + c) * 4 + ch]);
            v(r0, c0) * (1.0 - fr) * (1.0 - fc)
                + v(r0, c1) * (1.0 - fr) * fc
                + v(r1, c0) * fr * (1.0 - fc)
                + v(r1, c1) * fr * fc
        };
        let fp = gh * gw;
        for frame in 0..t {
            let mut idx = 0;
            for br in 0..gh / 2 {
                for bc in 0..gw / 2 {
                    for (ir, ic) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        for ch in 0..4 {
                            let want = naive(br * 2 + ir, bc * 2 + ic, ch);
                            let g = f64::from(got[(frame * fp + idx) * 4 + ch]);
                            assert!(
                                (g - want).abs() < 1e-5,
                                "{grid:?} patch {idx} ch {ch}: {g} vs {want}"
                            );
                        }
                        idx += 1;
                    }
                }
            }
        }
    }
}

#[test]
fn rope_is_axial_rotate_half_over_the_72_wide_head() {
    let cfg = VisionConfig::shipped();
    let (gh, gw) = (4, 6);
    let (cos, sin) = rope_tables(gh, gw, &cfg);
    let (heads, hd, fp) = (2, 72, gh * gw);
    let mut rng = Lcg(3);
    let x0 = rng.vec(2 * fp * heads * hd); // two temporal groups
    let mut x = x0.clone();
    apply_rope(&mut x, heads, hd, &cos, &sin, fp);
    let pos = patch_positions(gh, gw, 2);
    let inv: Vec<f64> = (0..18)
        .map(|i| 1.0 / 10000f64.powf(2.0 * i as f64 / 36.0))
        .collect();
    for i in 0..2 * fp {
        let (row, col) = pos[i % fp];
        // full 72-wide cos/sin: [row*inv (18) | col*inv (18)] twice
        let ang: Vec<f64> = (0..36)
            .map(|j| {
                if j < 18 {
                    row as f64 * inv[j]
                } else {
                    col as f64 * inv[j - 18]
                }
            })
            .collect();
        for h in 0..heads {
            let base = (i * heads + h) * hd;
            for j in 0..hd {
                let a = ang[j % 36];
                let rot = if j < 36 {
                    -x0[base + j + 36]
                } else {
                    x0[base + j - 36]
                };
                let want = f64::from(x0[base + j]) * a.cos() + f64::from(rot) * a.sin();
                assert!(
                    (f64::from(x[base + j]) - want).abs() < 2e-6,
                    "patch {i} head {h} dim {j}"
                );
            }
        }
    }
}

fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    segs: &[(usize, usize)],
    heads: usize,
    hd: usize,
) -> Vec<f64> {
    let width = heads * hd;
    let mut out = vec![0f64; q.len()];
    for &(s, e) in segs {
        for h in 0..heads {
            for i in s..e {
                let sc: Vec<f64> = (s..e)
                    .map(|j| {
                        (0..hd)
                            .map(|d| {
                                f64::from(q[i * width + h * hd + d])
                                    * f64::from(k[j * width + h * hd + d])
                            })
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let ex: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = ex.iter().sum();
                for d in 0..hd {
                    out[i * width + h * hd + d] = (s..e)
                        .map(|j| ex[j - s] / z * f64::from(v[j * width + h * hd + d]))
                        .sum();
                }
            }
        }
    }
    out
}

fn max_abs(got: &[f32], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0f64, f64::max)
}

#[test]
fn full_attention_stays_inside_its_segments() {
    let (heads, hd) = (3, 72);
    let segs = [(0usize, 70usize), (70, 100), (100, 229)]; // ragged, > one 128-row block
    let p = 229;
    let mut rng = Lcg(11);
    let (q, k, v) = (
        rng.vec(p * heads * hd),
        rng.vec(p * heads * hd),
        rng.vec(p * heads * hd),
    );
    let got = full_attention(&q, &k, &v, &segs, heads, hd);
    let want = naive_attention(&q, &k, &v, &segs, heads, hd);
    let md = max_abs(&got, &want);
    assert!(md < 2e-6, "max abs diff {md}");
    // changing another frame's keys must not move this frame's output
    let mut k2 = k.clone();
    k2[100 * heads * hd..].iter_mut().for_each(|x| *x += 3.0);
    let again = full_attention(&q, &k2, &v, &segs, heads, hd);
    assert_eq!(&got[..100 * heads * hd], &again[..100 * heads * hd]);
}

/// The driver's layout contract (`[h][n][hdp]` q/k, `[h][dvp][n]` v, `[h][n][dvp]` out) executed by
/// a naive f64 model: zero padding the 72-wide head to 96/80 must not change the attention.
#[test]
fn padding_the_head_for_the_driver_changes_nothing() {
    let (heads, hd, n) = (4, 72, 50);
    let (hdp, dvp) = (96, 80);
    let mut rng = Lcg(5);
    let width = heads * hd;
    // fp16-rounded inputs, as the device sees them
    let r = |v: Vec<f32>| -> Vec<f32> { v.iter().map(|&x| f16::from_f32(x).to_f32()).collect() };
    let (q, k, v) = (
        r(rng.vec(n * width)),
        r(rng.vec(n * width)),
        r(rng.vec(n * width)),
    );
    let (q16, k16) = (
        pad_heads(&q, n, heads, hd, hdp),
        pad_heads(&k, n, heads, hd, hdp),
    );
    let v16 = pad_heads_transposed(&v, n, heads, hd, dvp);
    assert_eq!((q16.len(), v16.len()), (heads * n * hdp, heads * dvp * n));
    // padding columns/rows are exactly zero
    assert!(
        q16.chunks(hdp)
            .all(|r| r[hd..].iter().all(|x| x.to_f32() == 0.0))
    );
    assert!(
        v16.chunks(dvp * n)
            .all(|h| h[hd * n..].iter().all(|x| x.to_f32() == 0.0))
    );
    // the device model: scores over hdp, softmax, AV over the transposed v
    let scale = 1.0 / (hd as f64).sqrt();
    let mut o16 = vec![f16::ZERO; heads * n * dvp];
    for h in 0..heads {
        for i in 0..n {
            let sc: Vec<f64> = (0..n)
                .map(|j| {
                    (0..hdp)
                        .map(|d| {
                            f64::from(q16[(h * n + i) * hdp + d].to_f32())
                                * f64::from(k16[(h * n + j) * hdp + d].to_f32())
                        })
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
            let z: f64 = ex.iter().sum();
            for d in 0..dvp {
                let acc: f64 = (0..n)
                    .map(|j| ex[j] / z * f64::from(v16[(h * dvp + d) * n + j].to_f32()))
                    .sum();
                o16[(h * n + i) * dvp + d] = f16::from_f64(acc);
            }
        }
    }
    let mut got = vec![0f32; n * width];
    unpack_padded(&o16, n, heads, hd, dvp, &mut got);
    let want = naive_attention(&q, &k, &v, &[(0, n)], heads, hd);
    let md = max_abs(&got, &want);
    assert!(
        md < 2e-3,
        "padded layout vs plain attention: max abs diff {md}"
    ); // fp16 output
}

#[test]
fn fp16_emulation_flags_values_outside_the_range() {
    let big = vec![70000.0f32; 8];
    let mut q = vec![0.5f32; 8];
    q[0] = 70000.0;
    let err = Emu16Exec
        .attention(&q, &big, &big, &[(0, 2)], 1, 4, 1)
        .unwrap_err();
    assert!(
        err.downcast_ref::<compute::Fp16Range>().is_some(),
        "{err:#}"
    );
    // in range is fine
    let ok = vec![1.0f32; 8];
    assert!(
        Emu16Exec
            .attention(&ok, &ok, &ok, &[(0, 2)], 1, 4, 1)
            .is_ok()
    );
}

// ---- oracle gates (need the dumps; the checkpoint where a tower runs) -----------------

pub(crate) fn oracle(path: &std::path::Path, keys: &[&str]) -> crate::clef::testutil::Npz {
    let keys: Vec<String> = keys.iter().map(|s| s.to_string()).collect();
    read_npz_keys(path, |k| keys.iter().any(|w| w == k))
}

pub(crate) fn i64s(b: &[u8]) -> Vec<i64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

pub(crate) fn fixture(id: &str) -> Value {
    let all: Vec<Value> =
        serde_json::from_slice(&std::fs::read(app_path("tools/clef_fixtures_media.json")).unwrap())
            .unwrap();
    let mut v = all.into_iter().find(|c| c["id"] == id).expect("fixture");
    // fixtures say `data/...`: make them absolute against the app dir
    for key in ["images", "videos", "audio"] {
        if let Some(items) = v[key].as_array_mut() {
            for i in items {
                let p = i.as_str().unwrap().to_string();
                *i = Value::String(app_path(&p).to_string_lossy().into());
            }
        }
    }
    v.as_object_mut().unwrap().remove("id");
    v
}

#[test]
fn decode_and_preprocessing_match_the_oracle_bit_for_bit() {
    for id in ["media-image-small", "media-image-cat", "media-images-two"] {
        let Some(path) = dump_of(id) else { return };
        let fx = fixture(id);
        let n_images = fx["images"].as_array().unwrap().len();
        let want = oracle(
            &path,
            &[
                "media_image_rgb_0",
                "media_image_rgb_1",
                "media_pixel_values",
                "media_image_grid_thw",
            ],
        );
        let cfg = VisionConfig::shipped();
        let grids = i64s(&want["media_image_grid_thw"].2);
        let mut got_pixels: Vec<f32> = Vec::new();
        let mut got_grids: Vec<i64> = Vec::new();
        for i in 0..n_images {
            let value = fx["images"][i].as_str().unwrap();
            let rgb = load_rgb(value, i).unwrap();
            let oracle_rgb = &want[&format!("media_image_rgb_{i}")];
            assert_eq!(
                (rgb.height() as usize, rgb.width() as usize, 3),
                (oracle_rgb.1[0], oracle_rgb.1[1], oracle_rgb.1[2]),
                "{id}[{i}] size"
            );
            let diff = rgb
                .as_raw()
                .iter()
                .zip(&oracle_rgb.2)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(diff, 0, "{id}[{i}]: {diff} decoded bytes differ from PIL");
            let p = prepare_image(&rgb, &cfg).unwrap();
            got_grids.extend(p.grid.iter().map(|&g| g as i64));
            got_pixels.extend_from_slice(&p.pixels);
        }
        assert_eq!(got_grids, grids, "{id}: grids");
        let want_px = f32s(&want["media_pixel_values"].2);
        assert_eq!(got_pixels.len(), want_px.len(), "{id}");
        let bad = got_pixels
            .iter()
            .zip(&want_px)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "{id}: {} pixel values, {bad} differ from the processor",
            want_px.len()
        );
        assert_eq!(
            bad, 0,
            "{id}: pixel values differ (resize/normalize/patch order)"
        );
    }
}

/// Run the tower on `id`'s image and compare every tapped block, the merger output and the
/// DeepStack taps with the oracle's f32 dump.
fn tower_gate(id: &str, device: TowerDevice, min_cos: f64, max_rel_taps: f64, max_rel_out: f64) {
    tower_gate_of("image", id, device, min_cos, max_rel_taps, max_rel_out)
}

/// `kind` is `"image"` or `"video"` (the dump's key prefix and pixel/grid keys).
fn tower_gate_of(
    kind: &str,
    id: &str,
    device: TowerDevice,
    min_cos: f64,
    max_rel_taps: f64,
    max_rel_out: f64,
) {
    let Some(dir) = dir_from_env("CLEF_MODEL_DIR", model_dir()) else {
        return;
    };
    let Some(path) = dump_of(id) else { return };
    let (px_key, grid_key) = if kind == "video" {
        ("media_pixel_values_videos", "media_video_grid_thw")
    } else {
        ("media_pixel_values", "media_image_grid_thw")
    };
    let mut keys: Vec<String> = [0, 8, 16, 24, 26]
        .iter()
        .map(|k| format!("{kind}_block_{k}"))
        .collect();
    keys.extend(
        [
            "tower_last",
            "merged",
            "deepstack_0",
            "deepstack_1",
            "deepstack_2",
        ]
        .iter()
        .map(|s| format!("{kind}_{s}")),
    );
    keys.extend([px_key.to_string(), grid_key.to_string()]);
    let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
    let want = oracle(&path, &keys);
    let px = f32s(&want[px_key].2);
    let g: Vec<usize> = i64s(&want[grid_key].2)
        .iter()
        .map(|&v| v as usize)
        .collect();
    assert_eq!(g.len(), 3, "this gate covers one {kind}");
    let tower = VisionTower::open(&dir, device, 3).unwrap();
    let t0 = std::time::Instant::now();
    let out = tower
        .forward(&px, [g[0], g[1], g[2]], 8, &[0, 8, 16, 24, 26])
        .unwrap();
    eprintln!(
        "{id}: {} on {} patches ({:?}) in {:.1}s",
        tower.label(),
        g[0] * g[1] * g[2],
        g,
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(
        tower.fallbacks(),
        0,
        "an fp16 overflow fell back to the CPU"
    );
    let mut report = Vec::new();
    let mut check = |name: &str, got: &[f32], key: &str, limit: f64| {
        let (cos, md, rel) = compare(got, &f32s(&want[key].2));
        eprintln!("  {name:>12}: cos {cos:.9} max|d| {md:.3e} rel-to-peak {rel:.3e}");
        report.push((name.to_string(), cos, rel, limit));
    };
    for k in [0usize, 8, 16, 24, 26] {
        check(
            &format!("block {k}"),
            &out.taps[&k],
            &format!("{kind}_block_{k}"),
            max_rel_taps,
        );
    }
    check(
        "tower last",
        &out.taps[&26],
        &format!("{kind}_tower_last"),
        max_rel_taps,
    );
    check(
        "merged",
        &out.merged,
        &format!("{kind}_merged"),
        max_rel_out,
    );
    for j in 0..3 {
        check(
            &format!("deepstack {j}"),
            &out.deepstack[j],
            &format!("{kind}_deepstack_{j}"),
            max_rel_out,
        );
    }
    for (name, cos, rel, limit) in report {
        assert!(cos >= min_cos, "{id} {name}: cosine {cos}");
        assert!(rel <= limit, "{id} {name}: rel-to-peak {rel} > {limit}");
    }
}

#[test]
fn the_f32_tower_matches_the_oracle_on_the_small_image() {
    // f32 vs f32: the oracle's own HF-vs-plain-torch selfcheck is 2e-5 relative to the peak
    tower_gate("media-image-small", TowerDevice::Cpu, 0.999999, 5e-4, 1e-4);
}

#[test]
fn the_fp16_emulation_tracks_the_oracle_on_the_small_image() {
    // what the NPU numerics (fp16 operands + results, f32 accumulation) should reach
    tower_gate("media-image-small", TowerDevice::Emu16, 0.9999, 0.1, 0.05);
}

#[test]
#[ignore = "2520 patches: ~20 s on the dev host; needs the BF16 checkpoint and the cat dump"]
fn the_f32_tower_matches_the_oracle_on_the_cat() {
    tower_gate("media-image-cat", TowerDevice::Cpu, 0.999999, 5e-4, 1e-4);
}

#[test]
#[ignore = "2520 patches through the fp16 emulation; needs the BF16 checkpoint and the cat dump"]
fn the_fp16_emulation_tracks_the_oracle_on_the_cat() {
    tower_gate("media-image-cat", TowerDevice::Emu16, 0.9999, 0.1, 0.05);
}

#[test]
fn the_tower_opens_on_the_autoround_shards_too() {
    // the tower tensors are BF16 in both checkpoints (shards 6/7 there)
    let Some(dir) = dir_from_env("CLEF_AUTOROUND_DIR", autoround_dir()) else {
        return;
    };
    let Some(path) = dump_of("media-image-small") else {
        return;
    };
    let want = oracle(&path, &["image_merged", "media_pixel_values"]);
    let px = f32s(&want["media_pixel_values"].2);
    let tower = VisionTower::open(&dir, TowerDevice::Cpu, 3).unwrap();
    let out = tower.forward(&px, [1, 14, 20], 8, &[]).unwrap();
    let (cos, _, rel) = compare(&out.merged, &f32s(&want["image_merged"].2));
    eprintln!("autoround tower: cos {cos:.9} rel {rel:.3e}");
    assert!(cos >= 0.999999 && rel <= 1e-4);
}

#[test]
fn pack_directories_have_no_tower() {
    let d = std::env::temp_dir().join(format!("clef-no-tower-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    assert!(!VisionTower::available(&d));
    std::fs::write(
        d.join("model.safetensors.index.json"),
        br#"{"weight_map": {"thinker.model.norm.weight": "a.safetensors"}}"#,
    )
    .unwrap();
    assert!(!VisionTower::available(&d));
    std::fs::write(
        d.join("model.safetensors.index.json"),
        br#"{"weight_map": {"thinker.visual.patch_embed.proj.weight": "a.safetensors"}}"#,
    )
    .unwrap();
    assert!(VisionTower::available(&d));
    std::fs::remove_dir_all(&d).ok();
}

/// Image requests end to end through `ClefModel` (decode -> tower -> splice/DeepStack ->
/// thinker -> head) against the oracle's per-question probabilities.
pub(crate) fn answers_gate(dir: PathBuf, kind: EngineKind, ids: &[&str], max_dp: f64) {
    let model = ClefModel::load_kind(&dir, 8, kind).unwrap();
    assert!(model.images_supported());
    for id in ids {
        let Some(path) = dump_of(id) else { continue };
        let req = ClefRequest::from_value(fixture(id)).unwrap();
        let enc = model.encode(&req, 64000).unwrap();
        let t0 = std::time::Instant::now();
        let done = model.answer(&req, 64000).unwrap();
        let (mut worst, mut same_top1) = (0f64, true);
        for (qid, probs) in &done.probabilities {
            let key = format!("probs_{qid}");
            let want: Vec<f64> = f32s(&oracle(&path, &[&key])[&key].2)
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            // the dump's option order is the question's option order
            let order = &enc
                .questions
                .iter()
                .find(|q| &q.id == qid)
                .unwrap()
                .option_ids;
            let got: Vec<f64> = order.iter().map(|o| probs[o]).collect();
            assert_eq!(got.len(), want.len(), "{id}/{qid}");
            let argmax = |v: &[f64]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            same_top1 &= argmax(&got) == argmax(&want);
            for (g, w) in got.iter().zip(&want) {
                worst = worst.max((g - w).abs());
            }
        }
        eprintln!(
            "{id}: {} tokens, max |dp| {worst:.3e}, top-1 equal {same_top1}, {:.1}s",
            done.tokens,
            t0.elapsed().as_secs_f64()
        );
        assert!(same_top1, "{id}: top-1 differs");
        assert!(worst <= max_dp, "{id}: max |dp| {worst} > {max_dp}");
    }
}

pub(crate) fn gate_ids() -> Vec<String> {
    std::env::var("CLEF_IDS")
        .unwrap_or_else(|_| "media-image-small,media-image-cat".into())
        .split(',')
        .map(String::from)
        .collect()
}

#[test]
#[ignore = "streams the 66 GiB BF16 checkpoint through the f32 CPU thinker (~30 s/record); needs the dumps"]
fn image_answers_match_the_oracle_on_the_bf16_cpu_path() {
    let Some(dir) = dir_from_env("CLEF_MODEL_DIR", model_dir()) else {
        return;
    };
    let ids = gate_ids();
    let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
    // BF16 weights: tight. The AutoRound shards on the f32 CPU path carry the int8 weight
    // quantization of the text path (0.0136 max |dp| on the text fixtures): bounded, not tight.
    let bound = if crate::clef::autoround_store::ShardStore::is_autoround(&dir) {
        0.03
    } else {
        5e-3
    };
    answers_gate(dir, EngineKind::Cpu, &ids, bound);
}

#[test]
#[ignore = "streams the AutoRound shards through the emulated int8 thinker; needs the dumps"]
fn image_answers_stay_close_to_the_oracle_on_the_emulated_int8_path() {
    let Some(dir) = dir_from_env("CLEF_AUTOROUND_DIR", autoround_dir()) else {
        return;
    };
    let ids = gate_ids();
    let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
    answers_gate(dir, EngineKind::Emu, &ids, 0.05);
}

#[test]
fn media_features_assemble_from_tower_outputs() {
    // the tower hands 8b's assemble() exactly P/4 rows per image and per DeepStack tap
    let mut out = TowerOutputs::default();
    let ids = {
        let mut v = vec![1u32, crate::clef::media::VISION_START];
        v.extend([crate::clef::media::IMAGE_PAD; 6]);
        v.extend([crate::clef::media::VISION_END, 198]);
        v
    };
    let h = 4;
    out.image = vec![1.0; 6 * h];
    for d in out.image_deepstack.iter_mut() {
        *d = vec![0.5; 6 * h];
    }
    let f = MediaFeatures::assemble(&ids, None, &out, h).unwrap();
    assert_eq!((f.splice_idx.len(), f.deepstack_idx.len()), (6, 6));
    out.image.truncate(5 * h);
    assert!(MediaFeatures::assemble(&ids, None, &out, h).is_err());
}

// ---- a synthetic checkpoint: the open/index/widen path and the per-frame semantics ----

/// A tiny tower (3 blocks, hidden 32, 2 heads of 16, MLP 48, out 24, 8x8 position table) with
/// random BF16 weights split over two shards and a standard HF index.
fn synthetic_tower_dir(tag: &str) -> PathBuf {
    use crate::clef::pack::build::write_safetensors;
    let d = std::env::temp_dir().join(format!("clef-tower-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let (h, inter, out, mu) = (32usize, 48usize, 24usize, 128usize);
    let vc = serde_json::json!({
        "depth": 3, "hidden_size": h, "num_heads": 2, "intermediate_size": inter,
        "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
        "out_hidden_size": out, "in_channels": 3, "num_position_embeddings": 64,
        "deepstack_visual_indexes": [0, 1, 2],
    });
    std::fs::write(
        d.join("config.json"),
        serde_json::to_vec(&serde_json::json!({"thinker_config": {"vision_config": vc}})).unwrap(),
    )
    .unwrap();
    let mut rng = Lcg(99);
    let mut tensors: Vec<(String, Vec<usize>, f32)> = Vec::new(); // name, shape, scale
    let mut push =
        |n: &str, s: &[usize], sc: f32| tensors.push((format!("{PREFIX}{n}"), s.to_vec(), sc));
    push("patch_embed.proj.weight", &[h, 3, 2, 16, 16], 0.03);
    push("patch_embed.proj.bias", &[h], 0.1);
    push("pos_embed.weight", &[64, h], 0.5);
    for b in 0..3 {
        let p = format!("blocks.{b}.");
        for (n, s, sc) in [
            ("norm1.weight", vec![h], 0.1),
            ("norm1.bias", vec![h], 0.1),
            ("attn.qkv.weight", vec![3 * h, h], 0.25),
            ("attn.qkv.bias", vec![3 * h], 0.1),
            ("attn.proj.weight", vec![h, h], 0.2),
            ("attn.proj.bias", vec![h], 0.1),
            ("norm2.weight", vec![h], 0.1),
            ("norm2.bias", vec![h], 0.1),
            ("mlp.linear_fc1.weight", vec![inter, h], 0.2),
            ("mlp.linear_fc1.bias", vec![inter], 0.1),
            ("mlp.linear_fc2.weight", vec![h, inter], 0.15),
            ("mlp.linear_fc2.bias", vec![h], 0.1),
        ] {
            push(&format!("{p}{n}"), &s, sc);
        }
    }
    let merger = |pre: &str, ln: usize, push: &mut dyn FnMut(&str, &[usize], f32)| {
        push(&format!("{pre}.ln_q.weight"), &[ln], 0.1);
        push(&format!("{pre}.ln_q.bias"), &[ln], 0.1);
        push(&format!("{pre}.mlp.0.weight"), &[mu, mu], 0.08);
        push(&format!("{pre}.mlp.0.bias"), &[mu], 0.1);
        push(&format!("{pre}.mlp.2.weight"), &[out, mu], 0.08);
        push(&format!("{pre}.mlp.2.bias"), &[out], 0.1);
    };
    merger("merger", h, &mut push);
    for j in 0..3 {
        merger(&format!("merger_list.{j}"), mu, &mut push);
    }
    // norm weights ~1: the random scale above is a perturbation around it
    type ShardTensors = Vec<(String, Dtype, Vec<usize>, Vec<u8>)>;
    let mut shards: [ShardTensors; 2] = [Vec::new(), Vec::new()];
    let mut wm = serde_json::Map::new();
    for (i, (name, shape, sc)) in tensors.iter().enumerate() {
        let n: usize = shape.iter().product();
        let unit = name.contains("norm") || name.contains("ln_q.weight");
        let bytes: Vec<u8> = (0..n)
            .flat_map(|_| {
                let v = rng.next() * sc
                    + if unit && name.ends_with("weight") {
                        1.0
                    } else {
                        0.0
                    };
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let which = i % 2;
        shards[which].push((name.clone(), Dtype::Bf16, shape.clone(), bytes));
        wm.insert(
            name.clone(),
            Value::String(format!("model-0000{}-of-00002.safetensors", which + 1)),
        );
    }
    // a text tensor next to them, which the tower must ignore
    wm.insert(
        "thinker.model.norm.weight".into(),
        Value::String("model-00001-of-00002.safetensors".into()),
    );
    shards[0].push((
        "thinker.model.norm.weight".into(),
        Dtype::Bf16,
        vec![2],
        vec![0; 4],
    ));
    for (i, s) in shards.iter().enumerate() {
        write_safetensors(
            &d.join(format!("model-0000{}-of-00002.safetensors", i + 1)),
            s,
        )
        .unwrap();
    }
    std::fs::write(
        d.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": wm})).unwrap(),
    )
    .unwrap();
    d
}

#[test]
fn a_synthetic_checkpoint_opens_and_runs_with_per_frame_attention() {
    let dir = synthetic_tower_dir("run");
    assert!(VisionTower::available(&dir));
    let cpu = VisionTower::open(&dir, TowerDevice::Cpu, 1).unwrap();
    assert_eq!(cpu.config().depth, 3);
    let mut rng = Lcg(5);
    let (gh, gw) = (4, 6);
    let one = |seed: u64| Lcg(seed).vec(gh * gw * 1536);
    let (f0, f1) = (one(1), one(2));
    let r0 = cpu.forward(&f0, [1, gh, gw], 2, &[2]).unwrap();
    let r1 = cpu.forward(&f1, [1, gh, gw], 2, &[2]).unwrap();
    assert_eq!(r0.merged.len(), gh * gw / 4 * 24);
    assert!(r0.deepstack.iter().all(|d| d.len() == r0.merged.len()));
    assert!(r0.merged.iter().all(|v| v.is_finite()) && r0.merged.iter().any(|v| *v != 0.0));
    assert_eq!(r0.taps[&2].len(), gh * gw * 32);
    // a two-frame grid is two independent runs stacked (own rotary rows, no attention across)
    let both: Vec<f32> = f0.iter().chain(&f1).copied().collect();
    let r = cpu.forward(&both, [2, gh, gw], 2, &[2]).unwrap();
    let cat = |a: &[f32], b: &[f32]| a.iter().chain(b).copied().collect::<Vec<f32>>();
    // (gemm rounds differently for different row counts: equal to fp32 noise, not bit-equal)
    let close = |got: &[f32], want: Vec<f32>| {
        let (_, md, rel) = compare(got, &want);
        assert!(
            rel < 1e-5,
            "stacked vs separate runs: max|d| {md} rel {rel}"
        );
    };
    close(&r.merged, cat(&r0.merged, &r1.merged));
    for j in 0..3 {
        close(&r.deepstack[j], cat(&r0.deepstack[j], &r1.deepstack[j]));
    }
    // the fp16 emulation follows the f32 run at fp16 noise
    let emu = VisionTower::open(&dir, TowerDevice::Emu16, 1).unwrap();
    let e = emu.forward(&f0, [1, gh, gw], 2, &[]).unwrap();
    let (cos, _, rel) = compare(&e.merged, &r0.merged);
    assert!(
        cos > 0.999 && rel < 0.05,
        "emu16 vs f32: cos {cos} rel {rel}"
    );
    // the on-board numerics check reports the placement against the f32 reference
    let img = PreparedImage {
        pixels: f0.clone(),
        grid: [1, gh, gw],
        source: (gh * 16, gw * 16),
        resized: (gh * 16, gw * 16),
    };
    let report = emu.check_against_cpu(&img, 2).unwrap();
    assert!(
        report.contains("merged cos 0.99") && report.contains("deepstack2"),
        "{report}"
    );
    // bad grids and pixel counts are errors, not panics
    assert!(cpu.forward(&f0, [1, 3, 8], 2, &[]).is_err());
    assert!(cpu.forward(&f0[..100], [1, gh, gw], 2, &[]).is_err());
    // the placement a build has no hardware for is a clear error
    #[cfg(not(all(feature = "npu", target_arch = "aarch64")))]
    assert!(
        format!(
            "{:#}",
            VisionTower::open(&dir, TowerDevice::Npu, 1).unwrap_err()
        )
        .contains("NPU")
    );
    let _ = rng.next();
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn encode_images_concatenates_per_image_rows_for_the_splice() {
    let dir = synthetic_tower_dir("enc");
    let t = VisionTower::open(&dir, TowerDevice::Cpu, 1).unwrap();
    let mk = |seed: u64, gh: usize, gw: usize| PreparedImage {
        pixels: Lcg(seed).vec(gh * gw * 1536),
        grid: [1, gh, gw],
        source: (gh * 16, gw * 16),
        resized: (gh * 16, gw * 16),
    };
    let (a, b) = (mk(1, 4, 4), mk(2, 2, 6));
    let out = t.encode_images(&[a.clone(), b.clone()], 2).unwrap();
    assert_eq!(out.image.len(), (a.tokens() + b.tokens()) * 24);
    for d in &out.image_deepstack {
        assert_eq!(d.len(), out.image.len());
    }
    let ra = t.forward(&a.pixels, a.grid, 2, &[]).unwrap();
    let (_, _, rel) = compare(&out.image[..a.tokens() * 24], &ra.merged);
    assert!(rel < 1e-5, "{rel}");
    assert!(out.video.is_empty() && out.audio.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

// ---- videos (phase 8d) ---------------------------------------------------------------

pub(crate) fn have_ffmpeg() -> bool {
    ["ffmpeg", "ffprobe"].iter().all(|b| {
        std::process::Command::new(b)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

pub(crate) fn is_invalid(e: &anyhow::Error) -> bool {
    e.downcast_ref::<crate::clef::request::Invalid>().is_some()
}

fn video_request(videos: Vec<String>) -> ClefRequest {
    ClefRequest::from_value(serde_json::json!({
        "model": "clef-omni", "state": "a clip", "videos": videos,
        "questions": {"q": {"type": "noul"}},
    }))
    .unwrap()
}

#[test]
fn video_decode_and_preprocessing_match_the_oracle() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    for id in ["media-video-silent", "media-video-test"] {
        let Some(path) = dump_of(id) else { return };
        let fx = fixture(id);
        let value = fx["videos"][0].as_str().unwrap();
        let want = oracle(
            &path,
            &[
                "media_video_frames_0",
                "media_pixel_values_videos",
                "media_video_grid_thw",
            ],
        );
        let pv = ProbedVideo::open(value, 0).unwrap();
        let (frames, sampled) = pv.decode_even().unwrap();
        let oracle_frames = &want["media_video_frames_0"];
        assert_eq!(
            (
                frames.len(),
                frames[0].height() as usize,
                frames[0].width() as usize
            ),
            (oracle_frames.1[0], oracle_frames.1[1], oracle_frames.1[2]),
            "{id}: frame count / size ({sampled} sampled)"
        );
        let (mut differ, mut max_d) = (0usize, 0u8);
        let fb = oracle_frames.1[1] * oracle_frames.1[2] * 3;
        for (i, f) in frames.iter().enumerate() {
            for (a, b) in f
                .as_raw()
                .iter()
                .zip(&oracle_frames.2[i * fb..(i + 1) * fb])
            {
                if a != b {
                    differ += 1;
                    max_d = max_d.max(a.abs_diff(*b));
                }
            }
        }
        let p = pv.prepare(&VisionConfig::shipped(), 100_000).unwrap();
        let want_grid: Vec<usize> = i64s(&want["media_video_grid_thw"].2)
            .iter()
            .map(|&v| v as usize)
            .collect();
        assert_eq!(p.grid.to_vec(), want_grid, "{id}: grid");
        let want_px = f32s(&want["media_pixel_values_videos"].2);
        assert_eq!(p.pixels.len(), want_px.len(), "{id}");
        let (mut bad, mut max_px) = (0usize, 0f32);
        for (a, b) in p.pixels.iter().zip(&want_px) {
            if a.to_bits() != b.to_bits() {
                bad += 1;
                max_px = max_px.max((a - b).abs());
            }
        }
        eprintln!(
            "{id}: {} frames ({sampled} sampled) {}x{} -> {:?}: {differ} of {} decoded bytes differ \
             (max {max_d}), {bad} of {} pixel values differ (max {max_px:.3e})",
            frames.len(),
            frames[0].width(),
            frames[0].height(),
            p.grid,
            frames.len() * fb,
            want_px.len()
        );
        assert_eq!(differ, 0, "{id}: ffmpeg frames differ from PyAV's");
        assert_eq!(bad, 0, "{id}: pixel values differ");
    }
}

#[test]
fn a_video_record_gets_the_oracle_ids_positions_and_counts() {
    if !have_ffmpeg() {
        return;
    }
    let (Some(tok), Some(path)) = (tokenizer(), dump_of("media-video-silent")) else {
        return;
    };
    let fx = fixture("media-video-silent");
    let req = ClefRequest::from_value(fx).unwrap();
    let enc = encode_request(&tok, &req, 64000).unwrap();
    let want = oracle(&path, &["input_ids", "position_ids"]);
    let want_ids: Vec<u32> = i64s(&want["input_ids"].2)
        .iter()
        .map(|&v| v as u32)
        .collect();
    assert_eq!(enc.input_ids, want_ids, "ids differ from the oracle");
    let media = enc.media.as_ref().unwrap();
    assert!(media.images.is_empty() && media.videos.len() == 1);
    assert_eq!(media.layout.video_grids, [[4, 20, 28]]);
    assert!(!media.layout.use_audio_in_video);
    let pos = crate::clef::media::rope_index(&enc.input_ids, &media.layout).unwrap();
    assert_eq!(
        pos,
        crate::clef::media::Positions::from_flat(&f32s(&want["position_ids"].2))
    );
    let n_pad = |t: u32| enc.input_ids.iter().filter(|&&x| x == t).count();
    assert_eq!(n_pad(crate::clef::media::VIDEO_PAD), 560);
    assert_eq!(n_pad(crate::clef::media::IMAGE_PAD), 0);
    // the block counts against the cap
    let err = encode_request(&tok, &req, 300).unwrap_err();
    assert!(is_invalid(&err), "{err:#}");
    assert!(format!("{err:#}").contains("videos[0]"), "{err:#}");
}

#[test]
fn soundtracks_are_used_only_when_every_video_has_one() {
    if !have_ffmpeg() {
        return;
    }
    let Some(tok) = tokenizer() else { return };
    let (silent, sound) = (
        app_path("data/clef-video-silent.mp4"),
        app_path("data/test.mp4"),
    );
    if !silent.exists() || !sound.exists() {
        return;
    }
    let s = |p: &PathBuf| p.to_string_lossy().to_string();
    // every video has a track: use_audio_in_video, the track joins the mel batch (400 frames)
    let enc = encode_request(&tok, &video_request(vec![s(&sound)]), 64000).unwrap();
    let media = enc.media.as_ref().unwrap();
    assert!(media.layout.use_audio_in_video);
    assert_eq!(media.layout.audio_frames, [400]);
    assert_eq!(media.audio.as_ref().unwrap().valid, [400]);
    let n = |t: u32| enc.input_ids.iter().filter(|&&x| x == t).count();
    assert_eq!(
        (
            n(crate::clef::media::VIDEO_PAD),
            n(crate::clef::media::AUDIO_PAD)
        ),
        (660, 52)
    );
    let enc2 = encode_request(&tok, &video_request(vec![s(&sound), s(&sound)]), 64000).unwrap();
    assert_eq!(enc2.media.as_ref().unwrap().layout.audio_frames, [400, 400]);
    // one silent video makes the release ignore EVERY track: served, two video blocks
    let enc = encode_request(&tok, &video_request(vec![s(&sound), s(&silent)]), 64000).unwrap();
    let media = enc.media.as_ref().unwrap();
    assert_eq!(media.layout.video_grids, [[4, 22, 30], [4, 20, 28]]);
    assert!(!media.layout.use_audio_in_video);
    let n_pad = enc
        .input_ids
        .iter()
        .filter(|&&x| x == crate::clef::media::VIDEO_PAD)
        .count();
    assert_eq!(n_pad, 660 + 560);
    // data: URLs go through the same probe
    let bytes = std::fs::read(&sound).unwrap();
    let url = format!(
        "data:video/mp4;base64,{}",
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes)
    );
    let enc = encode_request(&tok, &video_request(vec![url]), 64000).unwrap();
    assert!(enc.media.as_ref().unwrap().layout.use_audio_in_video);
    // a file that is no video is a client error naming the entry
    let png = app_path("data/clef-tiny.png");
    let not_media = app_path("Cargo.toml");
    let err = encode_request(&tok, &video_request(vec![s(&not_media)]), 64000).unwrap_err();
    assert!(
        is_invalid(&err) && format!("{err:#}").contains("videos[0]"),
        "{err:#}"
    );
    let err =
        encode_request(&tok, &video_request(vec!["/nonexistent.mp4".into()]), 64000).unwrap_err();
    assert!(
        is_invalid(&err) && format!("{err:#}").contains("videos[0]"),
        "{err:#}"
    );
    // images and videos in one record: image block first whatever the JSON order
    let png_s = s(&png);
    let req = ClefRequest::from_value(serde_json::json!({
        "model": "m", "state": "x", "videos": [s(&silent)], "images": [png_s],
        "questions": {"q": {"type": "noul"}},
    }))
    .unwrap();
    let enc = encode_request(&tok, &req, 64000).unwrap();
    let ids = &enc.input_ids;
    let first = |t: u32| ids.iter().position(|&x| x == t).unwrap();
    assert!(first(crate::clef::media::IMAGE_PAD) < first(crate::clef::media::VIDEO_PAD));
}

#[test]
fn a_long_clip_is_refused_by_the_frame_cap_before_any_decode() {
    if !have_ffmpeg() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("clef-long-video-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let clip = dir.join("long.mp4");
    // 40 s at 10 fps -> 80 frames sampled at 2 fps (> the default cap of 64); silent
    let ok = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg("testsrc=size=64x64:rate=10:duration=40")
        .args(["-pix_fmt", "yuv420p"])
        .arg(&clip)
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipping: ffmpeg cannot synthesize the clip");
        return;
    }
    let err = ProbedVideo::open(clip.to_str().unwrap(), 0).err().unwrap();
    assert!(is_invalid(&err), "{err:#}");
    assert!(format!("{err:#}").contains("frames sampled"), "{err:#}");
    // 10 s -> 20 sampled frames (even), 64x64 is upscaled to 131072+ pixels: a 10 group video
    let short = dir.join("short.mp4");
    assert!(
        std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc=size=64x48:rate=10:duration=9.7")
            .args(["-pix_fmt", "yuv420p"])
            .arg(&short)
            .status()
            .unwrap()
            .success()
    );
    let v = ProbedVideo::open(short.to_str().unwrap(), 0).unwrap();
    assert_eq!(v.keep.len(), 20, "{:?}", v.keep);
    assert!(!v.probe.has_audio_stream);
    let p = v.prepare(&VisionConfig::shipped(), 100_000).unwrap();
    // 64x48 -> smart-resize up to the 131072-pixel floor, 10 temporal groups
    assert_eq!(p.grid[0], 10);
    assert!(p.resized.0 * p.resized.1 >= VIDEO_MIN_PIXELS);
    assert_eq!(p.pixels.len(), p.patches() * 1536);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn encode_media_puts_video_rows_in_the_video_slots_with_per_group_attention() {
    let dir = synthetic_tower_dir("vid");
    let t = VisionTower::open(&dir, TowerDevice::Cpu, 1).unwrap();
    let (gh, gw) = (4usize, 4usize);
    let mk = |seed: u64, groups: usize| PreparedVideo {
        pixels: Lcg(seed).vec(groups * gh * gw * 1536),
        grid: [groups, gh, gw],
        source: (gh * 16, gw * 16),
        capped: (gh * 16, gw * 16),
        resized: (gh * 16, gw * 16),
        sampled: groups * 2,
    };
    let img = PreparedImage {
        pixels: Lcg(9).vec(gh * gw * 1536),
        grid: [1, gh, gw],
        source: (gh * 16, gw * 16),
        resized: (gh * 16, gw * 16),
    };
    let v = mk(3, 3);
    let out = t
        .encode_media(std::slice::from_ref(&img), std::slice::from_ref(&v), 2)
        .unwrap();
    assert_eq!(out.image.len(), img.tokens() * 24);
    assert_eq!(out.video.len(), v.tokens() * 24);
    for d in &out.video_deepstack {
        assert_eq!(d.len(), out.video.len());
    }
    // attention never crosses temporal groups: the groups of a video are independent given
    // the same patches (a [3,h,w] run equals three [1,h,w] runs, group by group)
    let per = gh * gw * 1536;
    let mut stacked = Vec::new();
    for g in 0..3 {
        let one = t
            .forward(&v.pixels[g * per..(g + 1) * per], [1, gh, gw], 2, &[])
            .unwrap();
        stacked.extend(one.merged);
    }
    let (cos, _, rel) = compare(&out.video, &stacked);
    assert!(cos > 0.999_999 && rel < 1e-5, "cos {cos} rel {rel}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "2240 patches through the f32 tower in release (~10 s); needs the BF16 checkpoint and dumps"]
fn the_f32_tower_matches_the_oracle_on_the_silent_video() {
    tower_gate_of(
        "video",
        "media-video-silent",
        TowerDevice::Cpu,
        0.999999,
        5e-4,
        1e-4,
    );
}

#[test]
#[ignore = "2640 patches through the f32 tower in release; needs the BF16 checkpoint and dumps"]
fn the_f32_tower_matches_the_oracle_on_the_video_with_a_track() {
    tower_gate_of(
        "video",
        "media-video-test",
        TowerDevice::Cpu,
        0.999999,
        5e-4,
        1e-4,
    );
}

#[test]
#[ignore = "2240 patches through the fp16 emulation; needs the BF16 checkpoint and dumps"]
fn the_fp16_emulation_tracks_the_oracle_on_the_silent_video() {
    tower_gate_of(
        "video",
        "media-video-silent",
        TowerDevice::Emu16,
        0.9999,
        0.1,
        0.05,
    );
}

#[test]
#[ignore = "2640 patches through the fp16 emulation; needs the BF16 checkpoint and dumps"]
fn the_fp16_emulation_tracks_the_oracle_on_the_video_with_a_track() {
    tower_gate_of(
        "video",
        "media-video-test",
        TowerDevice::Emu16,
        0.9999,
        0.1,
        0.05,
    );
}

#[test]
#[ignore = "streams the BF16 checkpoint through the f32 CPU thinker (~30 s); needs the dumps"]
fn video_answers_match_the_oracle_on_the_bf16_cpu_path() {
    let Some(dir) = dir_from_env("CLEF_MODEL_DIR", model_dir()) else {
        return;
    };
    answers_gate(dir, EngineKind::Cpu, &["media-video-silent"], 5e-3);
}

#[test]
#[ignore = "streams the AutoRound shards through the emulated int8 thinker; needs the dumps"]
fn video_answers_stay_close_to_the_oracle_on_the_emulated_int8_path() {
    let Some(dir) = dir_from_env("CLEF_AUTOROUND_DIR", autoround_dir()) else {
        return;
    };
    // The emulated int8 thinker is far from the f32 oracle on this 796-token record even when
    // fed the ORACLE's own tower outputs (8b gate with CLEF_IDS=media-video-silent: `motion`
    // max |dp| 9.7e-2, final-norm cosine 0.945, 17.5k router set mismatches); the answer sits at
    // p = 0.38, where the softmax is steep. This arm gates the video path (decode, tower,
    // splice, DeepStack, positions) to within that band (measured 8.0e-2) with top-1 equal;
    // the tight check is the BF16 CPU arm above (1.2e-3).
    answers_gate(dir, EngineKind::Emu, &["media-video-silent"], 0.12);
}

/// Frames above the per-frame cap are converted by swscale to the capped size: compare the
/// ffmpeg CLI path with the release's PyAV `load_video` on clips made for that
/// (`/tmp/opencode/clef/vid8d/<name>.{mp4,u8,json}`, written by the PyAV dump script in the
/// phase 8d notes in `docs/experiments/clef-omni.md`). Skips when the dump is absent.
#[test]
fn scaled_frames_match_pyav_on_clips_above_the_frame_cap() {
    if !have_ffmpeg() {
        return;
    }
    let dir = PathBuf::from("/tmp/opencode/clef/vid8d");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        eprintln!("skipping: no {}", dir.display());
        return;
    };
    let mut names: Vec<String> = rd
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.file_name()
                .to_str()?
                .strip_suffix(".json")
                .map(String::from)
        })
        .collect();
    names.sort();
    for name in names {
        let meta: Value =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{name}.json"))).unwrap())
                .unwrap();
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let want = std::fs::read(dir.join(format!("{name}.u8"))).unwrap();
        let pv = ProbedVideo::open(dir.join(format!("{name}.mp4")).to_str().unwrap(), 0).unwrap();
        let (frames, _) = pv.decode_even().unwrap();
        assert_eq!(
            (
                frames.len(),
                frames[0].height() as usize,
                frames[0].width() as usize
            ),
            (shape[0], shape[1], shape[2]),
            "{name}: frames / size"
        );
        let fb = shape[1] * shape[2] * 3;
        let (mut differ, mut max_d) = (0usize, 0u8);
        for (i, f) in frames.iter().enumerate() {
            for (a, b) in f.as_raw().iter().zip(&want[i * fb..(i + 1) * fb]) {
                if a != b {
                    differ += 1;
                    max_d = max_d.max(a.abs_diff(*b));
                }
            }
        }
        eprintln!(
            "{name}: source {}x{} -> {}x{}, {} frames: {differ} of {} bytes differ from PyAV (max {max_d})",
            pv.probe.width,
            pv.probe.height,
            shape[2],
            shape[1],
            shape[0],
            frames.len() * fb
        );
        assert_eq!(differ, 0, "{name}: scaled frames differ from PyAV");
    }
}
