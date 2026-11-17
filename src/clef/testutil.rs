//! Shared test helpers (oracle `.npz` reader, checkpoint directory resolution).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Newest snapshot directory of `repo` (`owner/name`) in the Hugging Face cache
/// (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`).
pub(crate) fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let hub = if let Ok(c) = std::env::var("HF_HUB_CACHE") {
        PathBuf::from(c)
    } else if let Ok(h) = std::env::var("HF_HOME") {
        PathBuf::from(h).join("hub")
    } else {
        PathBuf::from(std::env::var("HOME").ok()?).join(".cache/huggingface/hub")
    };
    let snaps = hub
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

fn resolve(var: &str, repo: &str, cell: &'static OnceLock<String>) -> &'static str {
    cell.get_or_init(|| {
        std::env::var(var)
            .ok()
            .or_else(|| hf_snapshot(repo).map(|p| p.display().to_string()))
            .unwrap_or_else(|| format!("<no snapshot of {repo}; set {var} or HF_HOME>"))
    })
}

/// BF16 `Cloudflare/clef-omni`: `$CLEF_MODEL_DIR`, else the newest HF cache snapshot.
pub(crate) fn model_dir() -> &'static str {
    static C: OnceLock<String> = OnceLock::new();
    resolve("CLEF_MODEL_DIR", "Cloudflare/clef-omni", &C)
}

/// `moeshawky/clef-omni-int8-autoround`: `$CLEF_AUTOROUND_DIR`, else the newest snapshot.
pub(crate) fn autoround_dir() -> &'static str {
    static C: OnceLock<String> = OnceLock::new();
    resolve(
        "CLEF_AUTOROUND_DIR",
        "moeshawky/clef-omni-int8-autoround",
        &C,
    )
}

/// Arrays of an uncompressed `np.savez` archive (zip64-aware local headers).
pub(crate) type Npz = BTreeMap<String, (String, Vec<usize>, Vec<u8>)>;

pub(crate) fn read_npz(path: &Path) -> Npz {
    read_npz_keys(path, |_| true)
}

/// Like [`read_npz`] but copies only the arrays `want(name)` accepts (the media dumps
/// are up to 0.55 GB).
pub(crate) fn read_npz_keys(path: &Path, want: impl Fn(&str) -> bool) -> Npz {
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mapping of a test fixture.
    let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let u16le = |o: usize| u16::from_le_bytes(m[o..o + 2].try_into().unwrap()) as usize;
    let u32le = |o: usize| u32::from_le_bytes(m[o..o + 4].try_into().unwrap()) as usize;
    let u64le = |o: usize| u64::from_le_bytes(m[o..o + 8].try_into().unwrap()) as usize;
    let mut out = BTreeMap::new();
    let mut p = 0;
    while p + 30 <= m.len() && u32le(p) == 0x0403_4b50 {
        assert_eq!(u16le(p + 8), 0, "compressed npz entries are not supported");
        let (mut csize, mut usize_) = (u32le(p + 18), u32le(p + 22));
        let (nlen, elen) = (u16le(p + 26), u16le(p + 28));
        let name = String::from_utf8(m[p + 30..p + 30 + nlen].to_vec()).unwrap();
        let mut e = p + 30 + nlen;
        let eend = e + elen;
        while e + 4 <= eend {
            let (id, sz) = (u16le(e), u16le(e + 2));
            if id == 1 {
                // zip64: uncompressed then compressed sizes, when the header says 0xFFFFFFFF
                let mut q = e + 4;
                if usize_ == 0xFFFF_FFFF {
                    usize_ = u64le(q);
                    q += 8;
                }
                if csize == 0xFFFF_FFFF {
                    csize = u64le(q);
                }
            }
            e += 4 + sz;
        }
        let key = name.trim_end_matches(".npy").to_string();
        if !want(&key) {
            p = eend + csize;
            continue;
        }
        let data = &m[eend..eend + csize];
        let _ = usize_;
        // .npy: magic(6) ver(2) hlen(2 or 4) header
        assert_eq!(&data[..6], b"\x93NUMPY");
        let (hlen, hoff) = if data[6] == 1 {
            (u16::from_le_bytes([data[8], data[9]]) as usize, 10)
        } else {
            (
                u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize,
                12,
            )
        };
        let header = std::str::from_utf8(&data[hoff..hoff + hlen]).unwrap();
        let descr = header
            .split("'descr': '")
            .nth(1)
            .unwrap()
            .split('\'')
            .next()
            .unwrap();
        let shape_txt = header
            .split("'shape': (")
            .nth(1)
            .unwrap()
            .split(')')
            .next()
            .unwrap();
        let shape: Vec<usize> = shape_txt
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let mut bytes = data[hoff + hlen..].to_vec();
        // numpy writes a Fortran-ordered array (e.g. `expand(...).astype`) column-major:
        // bring 2-D ones to C order.
        if header.contains("'fortran_order': True") && shape.len() == 2 {
            let (r, c) = (shape[0], shape[1]);
            let es = bytes.len() / (r * c);
            let mut t = vec![0u8; bytes.len()];
            for i in 0..r {
                for j in 0..c {
                    let (src, dst) = ((j * r + i) * es, (i * c + j) * es);
                    t[dst..dst + es].copy_from_slice(&bytes[src..src + es]);
                }
            }
            bytes = t;
        }
        out.insert(
            name.trim_end_matches(".npy").to_string(),
            (descr.to_string(), shape, bytes),
        );
        p = eend + csize;
    }
    out
}

pub(crate) fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}
