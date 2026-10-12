//! Hugging Face Hub model resolution for the llama.cpp-style `-hf` flag:
//! `-hf <owner>/<name>:<quant>` downloads a GGUF (plus a matching `mmproj`) into
//! a local directory the family loaders then open.
//!
//! Quant is required (no implicit default), matching llama.cpp. The mmproj
//! choice is `mmproj-<quant>.gguf`, then `mmproj.gguf`, then none; `--mmproj`
//! overrides it with an explicit quant or filename.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hf_hub::HFClientSync;

/// A parsed `-hf` spec.
#[derive(Debug, Clone)]
pub struct HubModel {
    pub repo: String,
    pub quant: String,
    pub revision: Option<String>,
    /// Explicit mmproj quant or filename (`--mmproj`).
    pub mmproj: Option<String>,
}

impl HubModel {
    /// Parse `owner/name:quant`.
    pub fn parse(spec: &str) -> Result<Self> {
        let (repo, quant) = spec.rsplit_once(':').ok_or_else(|| {
            anyhow::anyhow!(
                "-hf expects <owner>/<name>:<quant> (e.g. unsloth/Qwen3.5-0.8B-GGUF:Q8_0)"
            )
        })?;
        if repo.is_empty() || quant.is_empty() {
            bail!("-hf expects <owner>/<name>:<quant> (e.g. unsloth/Qwen3.5-0.8B-GGUF:Q8_0)");
        }
        if !repo.contains('/') {
            bail!("-hf repo must be <owner>/<name>, got '{repo}'");
        }
        Ok(Self {
            repo: repo.to_string(),
            quant: quant.to_string(),
            revision: None,
            mmproj: None,
        })
    }
}

/// Download `hub` into `dest` and return `dest`.
pub fn resolve(hub: &HubModel, dest: &Path) -> Result<PathBuf> {
    let (owner, name) = hub
        .repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("repo must be <owner>/<name>"))?;
    std::fs::create_dir_all(dest)
        .with_context(|| format!("create model dir {}", dest.display()))?;
    let client = HFClientSync::new().context("create Hugging Face client")?;
    let repo = client.model(owner, name);

    // Main weights: `<name>-<quant>.gguf`, trying the repo basename without a
    // trailing `-GGUF` (the usual convention: repo `X-GGUF`, file `X-Q8_0.gguf`).
    let base = name
        .strip_suffix("-GGUF")
        .or_else(|| name.strip_suffix("-gguf"))
        .unwrap_or(name);
    let mut main_candidates = vec![format!("{base}-{}.gguf", hub.quant)];
    if base != name {
        main_candidates.push(format!("{name}-{}.gguf", hub.quant));
    }
    main_candidates.push(format!("{name}.{}.gguf", hub.quant));
    let mut main = None;
    let mut tried = Vec::new();
    for cand in &main_candidates {
        tried.push(cand.clone());
        match repo
            .download_file()
            .filename(cand.clone())
            .maybe_revision(hub.revision.clone())
            .local_dir(dest.to_path_buf())
            .send()
        {
            Ok(p) => {
                main = Some(p);
                break;
            }
            Err(e) => eprintln!("hf: {cand} not available ({e})"),
        }
    }
    if let Some(p) = main {
        println!("hf: {} -> {}", hub.repo, p.display());
    } else {
        bail!(
            "no GGUF matching quant '{}' in {} (tried: {tried:?})",
            hub.quant,
            hub.repo
        );
    }

    // mmproj (multimodal families): explicit override, else `mmproj-<quant>`,
    // else plain `mmproj`.
    let mmproj_candidates: Vec<String> = match &hub.mmproj {
        Some(m) if m.ends_with(".gguf") => vec![m.clone()],
        Some(q) => vec![format!("mmproj-{q}.gguf"), format!("mmproj.{q}.gguf")],
        None => vec![
            format!("mmproj-{}.gguf", hub.quant),
            "mmproj.gguf".to_string(),
        ],
    };
    for cand in &mmproj_candidates {
        if let Ok(p) = repo
            .download_file()
            .filename(cand.clone())
            .maybe_revision(hub.revision.clone())
            .local_dir(dest.to_path_buf())
            .send()
        {
            println!("hf: mmproj -> {}", p.display());
            return Ok(dest.to_path_buf());
        }
    }
    Ok(dest.to_path_buf())
}

/// Download `config.json` from a model repo (cached) and parse it.
pub fn fetch_config_json(owner: &str, name: &str) -> Result<serde_json::Value> {
    let client = HFClientSync::new().context("create Hugging Face client")?;
    let path = client
        .model(owner, name)
        .download_file()
        .filename("config.json")
        .send()
        .with_context(|| format!("fetch {owner}/{name} config.json"))?;
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// Scan an argument list for `-hf`/`--hf` and resolve it to a model directory,
/// appending `--model-dir <dir>` to the remaining arguments.
///
/// Returns the (possibly extended) argument list. If `-hf` is absent the input
/// is returned unchanged.
pub fn resolve_args(args: impl Iterator<Item = String>) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let mut hf: Option<String> = None;
    let mut mmproj: Option<String> = None;
    let mut revision: Option<String> = None;
    let mut it = args.peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-hf" | "--hf" => {
                hf = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for {a}"))?,
                );
            }
            "--mmproj" => {
                mmproj = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --mmproj"))?,
                )
            }
            "--revision" => {
                revision = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --revision"))?,
                )
            }
            _ => out.push(a),
        }
    }
    let Some(spec) = hf else {
        // No -hf: but a stray --mmproj/--revision is still consumed above; put it back.
        if let Some(m) = mmproj {
            out.push("--mmproj".into());
            out.push(m);
        }
        if let Some(r) = revision {
            out.push("--revision".into());
            out.push(r);
        }
        return Ok(out);
    };
    let mut hub = HubModel::parse(&spec)?;
    hub.mmproj = mmproj;
    hub.revision = revision;
    let dest = hub_cache_dir(&hub);
    resolve(&hub, &dest)?;
    out.push("--model-dir".into());
    out.push(dest.to_string_lossy().into_owned());
    Ok(out)
}

/// The local directory a `-hf` model is downloaded into (under the HF cache root
/// when available, else a temp dir).
fn hub_cache_dir(hub: &HubModel) -> PathBuf {
    let root = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .map(|p| p.join("models"))
        .or_else(|| std::env::var_os("HF_HUB_CACHE").map(PathBuf::from))
        .unwrap_or_else(|| std::env::temp_dir().join("rocket-hf"));
    let slug = hub.repo.replace('/', "--");
    root.join(format!("{slug}-{}", hub.quant))
}
