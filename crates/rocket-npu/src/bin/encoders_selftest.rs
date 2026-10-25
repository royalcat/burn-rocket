//! On-device self-test for the M9 encoders (SAM ViT-Det + SigLIP).
//!
//! ```text
//! encoders_selftest --write-blob DIR                 # the synthetic fixture blob + pixels
//! encoders_selftest sam    [--blob P] [--pixels P] [--ref] [--npu] [--ctx N]
//!                          [--reps R] [--out P] [--hidden P]
//! encoders_selftest siglip [--blob P] [--pixels P] [--ref] [--npu] [--ctx N]
//!                          [--reps R] [--out P] [--hidden P]
//! encoders_selftest --io sam_ref|sam_npu|sam_ctx|siglip_ref|siglip_npu|siglip_ctx BLOB PIXELS OUT
//! ```
//!
//! The `sam` / `siglip` modes load a blob (the fixture written by `--write-blob` when the
//! defaults are used), then run every requested arm and print the **per-layer cosine**
//! table of each device arm against the host reference, plus the neck / final output:
//!
//! - `--ref`  the exact host reference (the C `fd < 0` datapath; device-free),
//! - `--npu`  the one-shot NPC path (the C `fd >= 0` per-call encode),
//! - `--ctx N` the resident path with `N` workers (the C `rocket_*_encode_ctx`).
//!
//! With none of the three, every arm that can run does. The gate is `cosine >= 0.9999`
//! per layer and on the neck/final output, against the host reference.
//!
//! `--write-blob` also prints the blob and pixel sizes; the same files drive the C
//! library through `tools/c-oracle/ab_board.c` (`sam` / `sam_ref` family), so the two
//! host references can be compared bit for bit.
//!
//! `--io` is the raw-file exchange of `tools/c-oracle/ab_board.c`: `OUT` receives the
//! neck / final output and `OUT.hidden` the `(n_layers+1)` intermediate token blocks.
//!
//! Exit: 0 pass, 1 numeric failure, 2 no device (only when a device arm was asked for).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use half::f16;
use rocket_npu::Device;
use rocket_npu::ops::sam::{
    SamContext, SamGeometry, SamModel, sam_encode, sam_encode_ctx, sam_encode_ref,
};
use rocket_npu::ops::siglip::{
    SiglipContext, SiglipGeometry, SiglipModel, siglip_encode, siglip_encode_ctx, siglip_encode_ref,
};

/// The per-layer / output cosine gate of the M9 brief.
const COS_GATE: f64 = 0.9999;

// ── Raw-file helpers ─────────────────────────────────────────────────────────

fn read_f16(path: &Path) -> Result<Vec<f16>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if raw.len() % 2 != 0 {
        return Err(format!("{}: odd byte length {}", path.display(), raw.len()));
    }
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
        .collect())
}

fn write_f16(path: &Path, v: &[f16]) -> Result<(), String> {
    let mut raw = Vec::with_capacity(v.len() * 2);
    for x in v {
        raw.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    std::fs::write(path, raw).map_err(|e| format!("write {}: {e}", path.display()))
}

/// The synthetic pixels of one image: a deterministic LCG in `[-1, 1)`.
fn synthetic_pixels(ic: usize, side: usize, seed: u32) -> Vec<f16> {
    let mut st = 0x9e37_79b9u32 ^ seed;
    (0..ic * side * side)
        .map(|_| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = ((st >> 8) & 0xffff) as f32 / 65535.0;
            f16::from_f32(-1.0 + 2.0 * u)
        })
        .collect()
}

// ── Metrics ──────────────────────────────────────────────────────────────────

/// The cosine of two fp16 vectors (f64 accumulation, as the C gates do).
fn cosine(a: &[f16], b: &[f16]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (x.to_f64(), y.to_f64());
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

/// `(exact, max_abs)` of two fp16 vectors.
fn delta(a: &[f16], b: &[f16]) -> (usize, f64) {
    let mut exact = 0;
    let mut max = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        if x == y {
            exact += 1;
        }
        max = max.max((x.to_f64() - y.to_f64()).abs());
    }
    (exact, max)
}

/// One scored arm: the per-layer cosines, the neck/final cosine, and the timing.
struct Score {
    layers: Vec<f64>,
    out: f64,
    wall_ms: f64,
    exact: usize,
    max_abs: f64,
}

impl Score {
    fn worst(&self) -> f64 {
        self.layers.iter().cloned().fold(self.out, f64::min)
    }

    fn pass(&self) -> bool {
        self.worst() >= COS_GATE
    }
}

// ── The SAM run ──────────────────────────────────────────────────────────────

fn sam_hidden_len(m: &SamModel) -> usize {
    (m.geom.n_layers + 1) * m.geom.tokens() * m.geom.d
}

fn sam_neck_len(m: &SamModel) -> usize {
    m.geom.neck_out * m.geom.tokens()
}

/// Score one SAM arm against the reference's `(hidden, neck)`.
fn sam_score(
    hidden: &[f16],
    out: &[f16],
    rhidden: &[f16],
    rout: &[f16],
    m: &SamModel,
    wall_ms: f64,
) -> Score {
    let gd = m.geom.tokens() * m.geom.d;
    let layers = (0..=m.geom.n_layers)
        .map(|k| {
            cosine(
                &hidden[k * gd..(k + 1) * gd],
                &rhidden[k * gd..(k + 1) * gd],
            )
        })
        .collect();
    let (exact, max_abs) = delta(out, rout);
    Score {
        layers,
        out: cosine(out, rout),
        wall_ms,
        exact,
        max_abs,
    }
}

fn print_sam_arm(tag: &str, s: &Score, model: &SamModel) {
    println!("  [{tag}] layer  0 (embeddings)  cos={:.9}", s.layers[0]);
    for k in 1..=model.geom.n_layers {
        println!(
            "  [{tag}] layer {k:2} {:<7}     cos={:.9}",
            if model.layers[k - 1].windowed {
                "(win)"
            } else {
                "(GLOBAL)"
            },
            s.layers[k]
        );
    }
    println!(
        "  [{tag}] neck output          cos={:.9}  (exact {}/{}, max_abs {:.3e})",
        s.out,
        s.exact,
        sam_neck_len(model),
        s.max_abs
    );
    println!(
        "  [{tag}] worst cos {:.9}  {:.1} ms  -> {}",
        s.worst(),
        s.wall_ms,
        if s.pass() { "PASS" } else { "FAIL" }
    );
}

// ── The SigLIP run ───────────────────────────────────────────────────────────

fn siglip_hidden_len(m: &SiglipModel) -> usize {
    (m.geom.n_layers + 1) * m.ntok() * m.geom.d
}

fn siglip_score(
    hidden: &[f16],
    out: &[f16],
    rhidden: &[f16],
    rout: &[f16],
    m: &SiglipModel,
    wall_ms: f64,
) -> Score {
    let td = m.ntok() * m.geom.d;
    let layers = (0..=m.geom.n_layers)
        .map(|k| {
            cosine(
                &hidden[k * td..(k + 1) * td],
                &rhidden[k * td..(k + 1) * td],
            )
        })
        .collect();
    let (exact, max_abs) = delta(out, rout);
    Score {
        layers,
        out: cosine(out, rout),
        wall_ms,
        exact,
        max_abs,
    }
}

fn print_siglip_arm(tag: &str, s: &Score, model: &SiglipModel) {
    println!("  [{tag}] layer  0 (embeddings)  cos={:.9}", s.layers[0]);
    for k in 1..=model.geom.n_layers {
        println!("  [{tag}] layer {k:2}              cos={:.9}", s.layers[k]);
    }
    println!(
        "  [{tag}] encoder output       cos={:.9}  (exact {}/{}, max_abs {:.3e})",
        s.out,
        s.exact,
        model.ntok() * model.geom.d,
        s.max_abs
    );
    println!(
        "  [{tag}] worst cos {:.9}  {:.1} ms  -> {}",
        s.worst(),
        s.wall_ms,
        if s.pass() { "PASS" } else { "FAIL" }
    );
}

// ── Driver ───────────────────────────────────────────────────────────────────

struct Args {
    dev: Option<Device>,
    blob: PathBuf,
    pixels: PathBuf,
    out: Option<PathBuf>,
    hidden: Option<PathBuf>,
    host: bool,
    npu: bool,
    ctx: Option<usize>,
    reps: usize,
    /// Load the host reference from this path (+ `.hidden`) instead of computing it.
    ref_in: Option<PathBuf>,
    /// Write the host reference to this path (+ `.hidden`).
    ref_out: Option<PathBuf>,
}

fn parse_args(dir: &Path, default_blob: &str, default_pixels: &str, args: &[String]) -> Args {
    let mut a = Args {
        dev: Device::open().ok(),
        blob: dir.join(default_blob),
        pixels: dir.join(default_pixels),
        out: None,
        hidden: None,
        host: false,
        npu: false,
        ctx: None,
        reps: 1,
        ref_in: None,
        ref_out: None,
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let v = || args.get(i + 1).cloned().unwrap_or_default();
        match flag {
            "--blob" => {
                a.blob = PathBuf::from(v());
                i += 1;
            }
            "--pixels" => {
                a.pixels = PathBuf::from(v());
                i += 1;
            }
            "--out" => {
                a.out = Some(PathBuf::from(v()));
                i += 1;
            }
            "--hidden" => {
                a.hidden = Some(PathBuf::from(v()));
                i += 1;
            }
            "--ref-in" => {
                a.ref_in = Some(PathBuf::from(v()));
                i += 1;
            }
            "--ref-out" => {
                a.ref_out = Some(PathBuf::from(v()));
                i += 1;
            }
            "--ctx" => {
                a.ctx = Some(v().parse().expect("--ctx N"));
                i += 1;
            }
            "--reps" => {
                a.reps = v().parse().expect("--reps R");
                i += 1;
            }
            "--ref" => a.host = true,
            "--npu" => a.npu = true,
            other => panic!("unknown argument '{other}'"),
        }
        i += 1;
    }
    if !a.host && !a.npu && a.ctx.is_none() {
        a.host = true;
        a.npu = true;
        a.ctx = Some(3);
    }
    a
}

fn run_sam(a: &Args) -> Result<bool, String> {
    let model = SamModel::load(&a.blob).map_err(|e| format!("load {}: {e}", a.blob.display()))?;
    let g = &model.geom;
    println!(
        "sam: d={} layers={} heads={} dhead={} d_ff={} grid={} win={} neck_out={} eps={}",
        g.d, g.n_layers, g.n_head, g.dhead, g.d_ff, g.grid, g.win, g.neck_out, g.eps
    );
    let pixels = read_f16(&a.pixels)?;
    if pixels.len() < g.ic * g.image_size * g.image_size {
        return Err(format!(
            "{}: {} pixels, need {}",
            a.pixels.display(),
            pixels.len(),
            g.ic * g.image_size * g.image_size
        ));
    }
    let (hn, nn) = (sam_hidden_len(&model), sam_neck_len(&model));
    let mut ref_hidden = vec![f16::ZERO; hn];
    let mut ref_out = vec![f16::ZERO; nn];
    let mut ok = true;

    // The host reference is the baseline every other arm is scored against. With
    // `--ref-in` it is read from a file instead (the naive host matmul is far too slow
    // to repeat at a real 1024px geometry, so a board run loads the dev host's copy —
    // which the A/B has already shown to be bit-identical to the C's).
    if let Some(p) = &a.ref_in {
        ref_out = read_f16(p)?;
        ref_hidden = read_f16(&PathBuf::from(format!("{}.hidden", p.display())))?;
        if ref_out.len() < nn || ref_hidden.len() < hn {
            return Err(format!("{}: reference is too small", p.display()));
        }
        println!(
            "=== host reference loaded from {} ({} + {} values) ===",
            p.display(),
            ref_hidden.len(),
            ref_out.len()
        );
    } else {
        let t = Instant::now();
        sam_encode_ref(&model, &pixels, &mut ref_out, Some(&mut ref_hidden))
            .map_err(|e| format!("sam_encode_ref: {e}"))?;
        let ref_ms = t.elapsed().as_secs_f64() * 1e3;
        if a.host || a.ref_out.is_some() {
            println!("=== host reference (fd<0) ===");
            println!(
                "  [ref] {} hidden blocks, {} neck values, {:.1} ms",
                g.n_layers + 1,
                nn,
                ref_ms
            );
        }
        // Determinism: the reference has no state, so a second run must match bit for bit.
        let mut h2 = vec![f16::ZERO; hn];
        let mut o2 = vec![f16::ZERO; nn];
        sam_encode_ref(&model, &pixels, &mut o2, Some(&mut h2)).map_err(|e| e.to_string())?;
        if h2 != ref_hidden || o2 != ref_out {
            println!("  [ref] NOT deterministic -> FAIL");
            ok = false;
        }
    }
    if let Some(p) = &a.ref_out {
        write_f16(p, &ref_out)?;
        write_f16(
            &PathBuf::from(format!("{}.hidden", p.display())),
            &ref_hidden,
        )?;
    }

    let mut hidden = vec![f16::ZERO; hn];
    let mut out = vec![f16::ZERO; nn];

    if a.npu {
        match &a.dev {
            None => println!("note: no NPU — the one-shot arm is skipped"),
            Some(dev) => {
                println!("=== NPU one-shot (fd>=0) ===");
                let t = Instant::now();
                sam_encode(dev, &model, &pixels, &mut out, Some(&mut hidden))
                    .map_err(|e| format!("sam_encode: {e}"))?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let s = sam_score(&hidden, &out, &ref_hidden, &ref_out, &model, ms);
                print_sam_arm("npu", &s, &model);
                ok &= s.pass();
                if let Some(p) = &a.out {
                    write_f16(p, &out)?;
                }
                if let Some(p) = &a.hidden {
                    write_f16(p, &hidden)?;
                }
            }
        }
    }

    if let Some(nt) = a.ctx {
        match &a.dev {
            None => println!("note: no NPU — the resident arm is skipped"),
            Some(_) => {
                println!("=== resident ctx (nthreads={nt}) ===");
                let t = Instant::now();
                let mut c = SamContext::create(&model, nt).map_err(|e| format!("ctx: {e}"))?;
                let create_ms = t.elapsed().as_secs_f64() * 1e3;
                let mut best = f64::INFINITY;
                let mut s = None;
                for rep in 0..a.reps.max(1) {
                    let t = Instant::now();
                    sam_encode_ctx(&mut c, &pixels, &mut out, Some(&mut hidden))
                        .map_err(|e| format!("sam_encode_ctx: {e}"))?;
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    best = best.min(ms);
                    if rep == 0 {
                        s = Some(sam_score(&hidden, &out, &ref_hidden, &ref_out, &model, ms));
                    }
                }
                let mut s = s.expect("at least one rep");
                s.wall_ms = best;
                println!(
                    "  ctx created in {:.1} ms (nthreads={})",
                    create_ms,
                    c.nthreads()
                );
                print_sam_arm("res", &s, &model);
                ok &= s.pass();
                if let Some(p) = &a.out {
                    write_f16(p, &out)?;
                }
                if let Some(p) = &a.hidden {
                    write_f16(p, &hidden)?;
                }
            }
        }
    }
    Ok(ok)
}

fn run_siglip(a: &Args) -> Result<bool, String> {
    let model =
        SiglipModel::load(&a.blob).map_err(|e| format!("load {}: {e}", a.blob.display()))?;
    let g = &model.geom;
    println!(
        "siglip: d={} layers={} heads={} dh={} d_ff={} L={} patch_dim={} image={} eps={}",
        g.d,
        g.n_layers,
        g.n_head,
        g.d / g.n_head,
        g.d_ff,
        g.l,
        g.patch_dim,
        g.image_size,
        g.eps
    );
    let pixels = read_f16(&a.pixels)?;
    if pixels.len() < g.ic * g.image_size * g.image_size {
        return Err(format!(
            "{}: {} pixels, need {}",
            a.pixels.display(),
            pixels.len(),
            g.ic * g.image_size * g.image_size
        ));
    }
    let td = model.ntok() * g.d;
    let hn = siglip_hidden_len(&model);
    let mut ref_hidden = vec![f16::ZERO; hn];
    let mut ref_out = vec![f16::ZERO; td];
    let mut ok = true;

    if let Some(p) = &a.ref_in {
        ref_out = read_f16(p)?;
        ref_hidden = read_f16(&PathBuf::from(format!("{}.hidden", p.display())))?;
        if ref_out.len() < td || ref_hidden.len() < hn {
            return Err(format!("{}: reference is too small", p.display()));
        }
        println!(
            "=== host reference loaded from {} ({} + {} values) ===",
            p.display(),
            ref_hidden.len(),
            ref_out.len()
        );
    } else {
        let t = Instant::now();
        siglip_encode_ref(&model, &pixels, &mut ref_out, Some(&mut ref_hidden))
            .map_err(|e| format!("siglip_encode_ref: {e}"))?;
        let ref_ms = t.elapsed().as_secs_f64() * 1e3;
        if a.host || a.ref_out.is_some() {
            println!("=== host reference (fd<0) ===");
            println!(
                "  [ref] {} hidden blocks, {td} output values, {:.1} ms",
                g.n_layers + 1,
                ref_ms
            );
        }
        let mut h2 = vec![f16::ZERO; hn];
        let mut o2 = vec![f16::ZERO; td];
        siglip_encode_ref(&model, &pixels, &mut o2, Some(&mut h2)).map_err(|e| e.to_string())?;
        if h2 != ref_hidden || o2 != ref_out {
            println!("  [ref] NOT deterministic -> FAIL");
            ok = false;
        }
    }
    if let Some(p) = &a.ref_out {
        write_f16(p, &ref_out)?;
        write_f16(
            &PathBuf::from(format!("{}.hidden", p.display())),
            &ref_hidden,
        )?;
    }

    let mut hidden = vec![f16::ZERO; hn];
    let mut out = vec![f16::ZERO; td];

    if a.npu {
        match &a.dev {
            None => println!("note: no NPU — the one-shot arm is skipped"),
            Some(dev) => {
                println!("=== NPU one-shot (fd>=0) ===");
                let t = Instant::now();
                siglip_encode(dev, &model, &pixels, &mut out, Some(&mut hidden))
                    .map_err(|e| format!("siglip_encode: {e}"))?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let s = siglip_score(&hidden, &out, &ref_hidden, &ref_out, &model, ms);
                print_siglip_arm("npu", &s, &model);
                ok &= s.pass();
                if let Some(p) = &a.out {
                    write_f16(p, &out)?;
                }
                if let Some(p) = &a.hidden {
                    write_f16(p, &hidden)?;
                }
            }
        }
    }

    if let Some(nt) = a.ctx {
        match &a.dev {
            None => println!("note: no NPU — the resident arm is skipped"),
            Some(_) => {
                println!("=== resident ctx (nthreads={nt}) ===");
                let t = Instant::now();
                let mut c = SiglipContext::create(&model, nt).map_err(|e| format!("ctx: {e}"))?;
                let create_ms = t.elapsed().as_secs_f64() * 1e3;
                let mut best = f64::INFINITY;
                let mut s = None;
                for rep in 0..a.reps.max(1) {
                    let t = Instant::now();
                    siglip_encode_ctx(&mut c, &pixels, &mut out, Some(&mut hidden))
                        .map_err(|e| format!("siglip_encode_ctx: {e}"))?;
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    best = best.min(ms);
                    if rep == 0 {
                        s = Some(siglip_score(
                            &hidden,
                            &out,
                            &ref_hidden,
                            &ref_out,
                            &model,
                            ms,
                        ));
                    }
                }
                let mut s = s.expect("at least one rep");
                s.wall_ms = best;
                println!(
                    "  ctx created in {:.1} ms (nthreads={}, fa={})",
                    create_ms,
                    c.nthreads(),
                    c.uses_fa()
                );
                print_siglip_arm("res", &s, &model);
                ok &= s.pass();
                if let Some(p) = &a.out {
                    write_f16(p, &out)?;
                }
                if let Some(p) = &a.hidden {
                    write_f16(p, &hidden)?;
                }
            }
        }
    }
    Ok(ok)
}

// ── The fixtures ─────────────────────────────────────────────────────────────

/// Write the synthetic blobs and pixels the self-test and the C A/B both consume.
fn write_blob(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let sam = SamModel::synthetic(SamGeometry::fixture(), 0);
    let siglip = SiglipModel::synthetic(SiglipGeometry::fixture(), 0);
    let sam_px = synthetic_pixels(sam.geom.ic, sam.geom.image_size, 1);
    let sig_px = synthetic_pixels(siglip.geom.ic, siglip.geom.image_size, 2);
    std::fs::write(dir.join("sam.f16"), sam.to_blob()).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("siglip.f16"), siglip.to_blob()).map_err(|e| e.to_string())?;
    write_f16(&dir.join("sam_pixels.raw"), &sam_px)?;
    write_f16(&dir.join("siglip_pixels.raw"), &sig_px)?;
    println!(
        "wrote {} sam.f16 ({} bytes), siglip.f16 ({} bytes), {} + {} pixel values",
        dir.display(),
        sam.to_blob().len(),
        siglip.to_blob().len(),
        sam_px.len(),
        sig_px.len()
    );
    Ok(())
}

// ── The raw-file A/B mode ────────────────────────────────────────────────────

fn io_mode(dev: &Device, args: &[String]) -> Result<(), String> {
    let family = args.first().ok_or("missing --io family")?.as_str();
    let blob = args.get(1).ok_or("missing BLOB")?;
    let pixels_path = args.get(2).ok_or("missing PIXELS")?;
    let out_path = PathBuf::from(args.get(3).ok_or("missing OUT")?);
    let hid_path = PathBuf::from(format!("{}.hidden", out_path.display()));
    let pixels = read_f16(Path::new(pixels_path))?;

    match family {
        "sam_ref" | "sam_npu" | "sam_ctx" => {
            let m = SamModel::load(blob).map_err(|e| e.to_string())?;
            let hn = sam_hidden_len(&m);
            let nn = sam_neck_len(&m);
            let mut out = vec![f16::ZERO; nn];
            let mut hid = vec![f16::ZERO; hn];
            match family {
                "sam_ref" => sam_encode_ref(&m, &pixels, &mut out, Some(&mut hid)),
                "sam_npu" => sam_encode(dev, &m, &pixels, &mut out, Some(&mut hid)),
                _ => {
                    let mut c = SamContext::create(&m, 3).map_err(|e| e.to_string())?;
                    sam_encode_ctx(&mut c, &pixels, &mut out, Some(&mut hid))
                }
            }
            .map_err(|e| e.to_string())?;
            write_f16(&out_path, &out)?;
            write_f16(&hid_path, &hid)
        }
        "siglip_ref" | "siglip_npu" | "siglip_ctx" => {
            let m = SiglipModel::load(blob).map_err(|e| e.to_string())?;
            let hn = siglip_hidden_len(&m);
            let td = m.ntok() * m.geom.d;
            let mut out = vec![f16::ZERO; td];
            let mut hid = vec![f16::ZERO; hn];
            match family {
                "siglip_ref" => siglip_encode_ref(&m, &pixels, &mut out, Some(&mut hid)),
                "siglip_npu" => siglip_encode(dev, &m, &pixels, &mut out, Some(&mut hid)),
                _ => {
                    let mut c = SiglipContext::create(&m, 3).map_err(|e| e.to_string())?;
                    siglip_encode_ctx(&mut c, &pixels, &mut out, Some(&mut hid))
                }
            }
            .map_err(|e| e.to_string())?;
            write_f16(&out_path, &out)?;
            write_f16(&hid_path, &hid)
        }
        other => Err(format!("unknown --io family '{other}'")),
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let usage = "\x20 encoders_selftest --write-blob DIR\n\
                 \x20 encoders_selftest sam    [--blob P] [--pixels P] [--ref] [--npu] [--ctx N] [--reps R] [--out P] [--hidden P] [--ref-in P] [--ref-out P]\n\
                 \x20 encoders_selftest siglip [--blob P] [--pixels P] [--ref] [--npu] [--ctx N] [--reps R] [--out P] [--hidden P]\n\
                 \x20 encoders_selftest --io sam_ref|sam_npu|sam_ctx|siglip_ref|siglip_npu|siglip_ctx BLOB PIXELS OUT";
    if argv.is_empty() || argv[0] == "--help" || argv[0] == "-h" {
        println!("{usage}");
        return ExitCode::SUCCESS;
    }
    // `--io` needs a device for the device families; `*_ref` does not.
    if argv[0] == "--io" {
        let rest = &argv[1..];
        let host_ref = rest.first().is_some_and(|f| f.ends_with("_ref"));
        let dev = match Device::open() {
            Ok(d) => d,
            Err(e) if host_ref => {
                // The reference arm never touches the device; a placeholder keeps the
                // signature honest without opening one.
                let _ = e;
                eprintln!("note: no NPU; running the host reference arm");
                match io_mode_ref(rest) {
                    Ok(()) => return ExitCode::SUCCESS,
                    Err(e) => {
                        eprintln!("encoders_selftest --io: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            Err(e) => {
                eprintln!("encoders_selftest: cannot open the NPU: {e}");
                return ExitCode::from(2);
            }
        };
        return match io_mode(&dev, rest) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("encoders_selftest --io: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if argv[0] == "--write-blob" {
        let dir = argv
            .get(1)
            .map_or_else(|| PathBuf::from("."), PathBuf::from);
        return match write_blob(&dir) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("encoders_selftest --write-blob: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let dir = PathBuf::from(".");
    let res = match argv[0].as_str() {
        "sam" => run_sam(&parse_args(&dir, "sam.f16", "sam_pixels.raw", &argv[1..])),
        "siglip" => run_siglip(&parse_args(
            &dir,
            "siglip.f16",
            "siglip_pixels.raw",
            &argv[1..],
        )),
        other => {
            eprintln!("unknown mode '{other}'\n{usage}");
            return ExitCode::from(2);
        }
    };
    match res {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("encoders_selftest {e}");
            ExitCode::FAILURE
        }
    }
}

/// The `--io` families that never touch the device (`*_ref`), reachable when `Device::open`
/// fails (the x86 host build of this crate has no NPU).
fn io_mode_ref(args: &[String]) -> Result<(), String> {
    let family = args.first().ok_or("missing --io family")?.as_str();
    let blob = args.get(1).ok_or("missing BLOB")?;
    let pixels_path = args.get(2).ok_or("missing PIXELS")?;
    let out_path = PathBuf::from(args.get(3).ok_or("missing OUT")?);
    let hid_path = PathBuf::from(format!("{}.hidden", out_path.display()));
    let pixels = read_f16(Path::new(pixels_path))?;
    match family {
        "sam_ref" => {
            let m = SamModel::load(blob).map_err(|e| e.to_string())?;
            let mut out = vec![f16::ZERO; sam_neck_len(&m)];
            let mut hid = vec![f16::ZERO; sam_hidden_len(&m)];
            sam_encode_ref(&m, &pixels, &mut out, Some(&mut hid)).map_err(|e| e.to_string())?;
            write_f16(&out_path, &out)?;
            write_f16(&hid_path, &hid)
        }
        "siglip_ref" => {
            let m = SiglipModel::load(blob).map_err(|e| e.to_string())?;
            let mut out = vec![f16::ZERO; m.ntok() * m.geom.d];
            let mut hid = vec![f16::ZERO; siglip_hidden_len(&m)];
            siglip_encode_ref(&m, &pixels, &mut out, Some(&mut hid)).map_err(|e| e.to_string())?;
            write_f16(&out_path, &out)?;
            write_f16(&hid_path, &hid)
        }
        other => Err(format!("'{other}' needs the NPU")),
    }
}
