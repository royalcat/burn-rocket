//! Reading the `moeshawky/clef-omni-int8-autoround` checkpoint (AutoRound 0.16.1,
//! `packing_format auto_round:auto_gptq`, 8 bit, group 128, symmetric) and unpacking its
//! GPTQ-style tensors into `[N, K]` signed int8 codes + `[N, K/group]` f16 scales.
//!
//! Measured layout of a quantized linear `<module>` (e.g. `...mlp.experts.0.gate_proj`),
//! for a weight of logical shape `[N, K]` (out, in):
//!
//! | tensor | dtype | shape | meaning |
//! |---|---|---|---|
//! | `<module>.qweight` | I32 | `[K/4, N]` | word `[kb, n]` holds `k = 4*kb + j` for `j = 0..4` in byte `j` (little end first), unsigned `q in 0..=255` |
//! | `<module>.qzeros`  | I32 | `[K/group, N/4]` | four unsigned zero points per word (along N), AutoGPTQ v1 convention: **stored = zero - 1**, always `0x7f7f7f7f` here (zero = 128) |
//! | `<module>.scales`  | F16 | `[K/group, N]` | per (group, out-channel) scale, may be negative |
//!
//! `w[n, k] = (q - zero) * scale[k / group, n]`; with zero = 128 the signed code
//! `q - 128` is in `[-128, 127]`. There is no `g_idx`. Everything else in
//! `thinker.model.layers` (norms, the router `mlp.gate.weight`, `q_norm`/`k_norm`) is BF16.

use crate::clef::st::{Dtype, SafeFile, TensorInfo};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

/// The stored zero-point word: `zero - 1 = 127` in every byte.
pub const STORED_ZERO_WORD: u32 = 0x7f7f_7f7f;
/// The unsigned zero point the signed codes are re-centred on.
pub const ZERO_POINT: i32 = 128;

/// A GPTQ-packed quantized linear read from a checkpoint.
pub struct QLinear<'a> {
    pub qweight: &'a [u8],
    pub qzeros: &'a [u8],
    /// f16 bits, `[K/group, N]` row-major.
    pub scales: &'a [u8],
    pub n: usize,
    pub k: usize,
    pub group: usize,
}

/// Unpack `q` into signed codes `[N, K]` row-major (`q - 128`) and transpose `scales`
/// `[K/g, N] -> [N, K/g]` (f16 bits as little-endian bytes). Verifies every stored zero
/// point.
pub fn unpack(l: &QLinear<'_>, codes: &mut [i8], scales: &mut [u8]) -> Result<()> {
    let (n, k, g) = (l.n, l.k, l.group);
    if !k.is_multiple_of(4) || !k.is_multiple_of(g) || !n.is_multiple_of(4) {
        bail!("unsupported quantized shape N={n} K={k} group={g}");
    }
    let groups = k / g;
    if l.qweight.len() != k / 4 * n * 4
        || l.qzeros.len() != groups * (n / 4) * 4
        || l.scales.len() != groups * n * 2
        || codes.len() != n * k
        || scales.len() != n * groups * 2
    {
        bail!("quantized tensor sizes disagree with N={n} K={k} group={g}");
    }
    // zero points: all must be the symmetric 128
    let bad = l
        .qzeros
        .as_chunks::<4>()
        .0
        .par_iter()
        .any(|w| u32::from_le_bytes(*w) != STORED_ZERO_WORD);
    if bad {
        bail!("non-symmetric zero point found (expected stored 0x7f7f7f7f everywhere)");
    }
    let qw = l.qweight;
    // Row blocks of BLK output channels: one cache line of qweight words per k-block
    // feeds BLK rows, instead of a strided read per element.
    const BLK: usize = 16;
    codes
        .par_chunks_mut(BLK * k)
        .enumerate()
        .for_each(|(bi, rows)| {
            let n0 = bi * BLK;
            let nb = rows.len() / k;
            let kq = k / 4;
            // 4 k-blocks at a time: four words per row are 16 contiguous output bytes
            // (one store instead of four per row).
            let mut kb = 0;
            while kb + 4 <= kq {
                let base = |j: usize| ((kb + j) * n + n0) * 4;
                for r in 0..nb {
                    let mut v = [0i8; 16];
                    for j in 0..4 {
                        let o = base(j) + r * 4;
                        let w = u32::from_le_bytes(qw[o..o + 4].try_into().unwrap());
                        for (d, b) in v[j * 4..j * 4 + 4]
                            .iter_mut()
                            .zip((w ^ 0x8080_8080).to_le_bytes())
                        {
                            *d = b as i8;
                        }
                    }
                    let o = r * k + kb * 4;
                    rows[o..o + 16].copy_from_slice(&v);
                }
                kb += 4;
            }
            for kb in kb..kq {
                let base = (kb * n + n0) * 4;
                for r in 0..nb {
                    let w =
                        u32::from_le_bytes(qw[base + r * 4..base + r * 4 + 4].try_into().unwrap());
                    let o = r * k + kb * 4;
                    // (q ^ 0x80) as i8 == q - 128 for q in 0..=255
                    let b = (w ^ 0x8080_8080).to_le_bytes();
                    for j in 0..4 {
                        rows[o + j] = b[j] as i8;
                    }
                }
            }
        });
    let sc: &[[u8; 2]] = l.scales.as_chunks::<2>().0;
    scales
        .par_chunks_mut(groups * 2)
        .enumerate()
        .for_each(|(row, out)| {
            for (gi, o) in out.chunks_mut(2).enumerate() {
                o.copy_from_slice(&sc[gi * n + row]);
            }
        });
    Ok(())
}

/// Reference dequantization straight from the packed form (slow, per element; the unit
/// tests and the BF16 comparison use it to cross-check [`unpack`]).
pub fn dequant_reference(l: &QLinear<'_>) -> Vec<f32> {
    let (n, k, g) = (l.n, l.k, l.group);
    let mut w = vec![0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            let word = u32::from_le_bytes(
                l.qweight[((col / 4) * n + row) * 4..][..4]
                    .try_into()
                    .unwrap(),
            );
            let q = ((word >> (8 * (col % 4))) & 0xff) as i32;
            let zw = u32::from_le_bytes(
                l.qzeros[((col / g) * (n / 4) + row / 4) * 4..][..4]
                    .try_into()
                    .unwrap(),
            );
            let zero = ((zw >> (8 * (row % 4))) & 0xff) as i32 + 1;
            let s = crate::clef::st::f16_to_f32(u16::from_le_bytes(
                l.scales[((col / g) * n + row) * 2..][..2]
                    .try_into()
                    .unwrap(),
            ));
            w[row * k + col] = (q - zero) as f32 * s;
        }
    }
    w
}

/// Index + mmaps of the AutoRound shards.
pub struct Source {
    pub dir: std::path::PathBuf,
    files: Vec<SafeFile>,
    /// Shard file names, parallel to `files`.
    names: Vec<String>,
    map: BTreeMap<String, usize>,
    pub index_sha256: String,
}

/// Where a tensor lives in its shard file (absolute byte offsets).
#[derive(Debug, Clone, Copy)]
pub struct Loc {
    pub shard: usize,
    pub start: u64,
    pub end: u64,
}

impl Source {
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_filtered(dir, |_| true)
    }

    /// Like [`Source::open`], but only the tensors `keep` accepts are indexed and only
    /// the shards holding at least one of them are opened (the talker / code2wav shards
    /// are never touched by a text-only run).
    pub fn open_filtered(dir: &Path, keep: impl Fn(&str) -> bool) -> Result<Self> {
        let p = dir.join("model.safetensors.index.json");
        let raw = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        let idx: serde_json::Value = serde_json::from_slice(&raw)?;
        let wm = idx["weight_map"]
            .as_object()
            .context("index has no weight_map")?;
        let mut names: Vec<String> = Vec::new();
        let mut map = BTreeMap::new();
        for (k, v) in wm {
            if !keep(k) {
                continue;
            }
            let f = v.as_str().context("shard name")?.to_string();
            let i = match names.iter().position(|n| *n == f) {
                Some(i) => i,
                None => {
                    names.push(f);
                    names.len() - 1
                }
            };
            map.insert(k.clone(), i);
        }
        let files = names
            .iter()
            .map(|n| SafeFile::open(&dir.join(n)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dir: dir.to_path_buf(),
            files,
            names,
            map,
            index_sha256: super::sha256_hex(&raw),
        })
    }

    pub fn shard_names(&self) -> &[String] {
        &self.names
    }

    pub fn shard_files(&self) -> &[SafeFile] {
        &self.files
    }

    /// Absolute file offsets of `name` inside its shard.
    pub fn locate(&self, name: &str) -> Result<Loc> {
        let (f, t) = self.get(name)?;
        let shard = self.map[name];
        Ok(Loc {
            shard,
            start: (f.data_start() + t.start) as u64,
            end: (f.data_start() + t.end) as u64,
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    pub fn get(&self, name: &str) -> Result<(&SafeFile, &TensorInfo)> {
        let i = *self
            .map
            .get(name)
            .with_context(|| format!("{name}: not in the shard index"))?;
        let f = &self.files[i];
        let t = f
            .tensors
            .get(name)
            .with_context(|| format!("{name}: missing from its shard"))?;
        Ok((f, t))
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        Ok(self.get(name)?.1)
    }

    pub fn bytes(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&[u8]> {
        let (f, t) = self.get(name)?;
        if t.dtype != dtype || t.shape != shape {
            bail!(
                "{name}: {:?} {:?}, expected {dtype:?} {shape:?}",
                t.dtype,
                t.shape
            );
        }
        Ok(f.bytes(t))
    }

    /// A quantized linear `<module>` of logical shape `[n, k]`.
    pub fn qlinear(&self, module: &str, n: usize, k: usize, group: usize) -> Result<QLinear<'_>> {
        let g = k / group;
        Ok(QLinear {
            qweight: self.bytes(&format!("{module}.qweight"), Dtype::I32, &[k / 4, n])?,
            qzeros: self.bytes(&format!("{module}.qzeros"), Dtype::I32, &[g, n / 4])?,
            scales: self.bytes(&format!("{module}.scales"), Dtype::F16, &[g, n])?,
            n,
            k,
            group,
        })
    }

    /// Drop every shard's resident mapped pages from this process.
    pub fn release(&self) {
        self.files.iter().for_each(SafeFile::release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Tiny = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<i32>, Vec<f32>);

    /// Hand-build a packed `[N=8, K=8]` group-4 layer... group must divide K and
    /// `unpack` wants K % 4 == 0 and N % 4 == 0.
    fn tiny() -> Tiny {
        let (n, k, g) = (8usize, 8usize, 4usize);
        // q[row][col] unsigned, scales per (group, row)
        let q: Vec<i32> = (0..n * k).map(|i| ((i * 37 + 11) % 256) as i32).collect();
        let sc: Vec<f32> = (0..(k / g) * n).map(|i| 0.5 * (i as f32 - 6.0)).collect(); // incl 0, negatives
        let mut qw = vec![0u8; k / 4 * n * 4];
        for kb in 0..k / 4 {
            for row in 0..n {
                let mut w = 0u32;
                for j in 0..4 {
                    w |= (q[row * k + kb * 4 + j] as u32) << (8 * j);
                }
                qw[(kb * n + row) * 4..][..4].copy_from_slice(&w.to_le_bytes());
            }
        }
        let qz: Vec<u8> = (0..(k / g) * (n / 4))
            .flat_map(|_| STORED_ZERO_WORD.to_le_bytes())
            .collect();
        let scales: Vec<u8> = sc
            .iter()
            .flat_map(|&s| half_bits(s).to_le_bytes())
            .collect();
        (qw, qz, scales, q, sc)
    }

    fn half_bits(v: f32) -> u16 {
        // exact for the small multiples of 0.5 used here (0 is special-cased)
        super::super::f32_to_f16_bits(v)
    }

    #[test]
    fn unpack_matches_the_hand_built_weights_and_the_zero_point_convention() {
        let (qw, qz, scales, q, sc) = tiny();
        let l = QLinear {
            qweight: &qw,
            qzeros: &qz,
            scales: &scales,
            n: 8,
            k: 8,
            group: 4,
        };
        let mut codes = vec![0i8; 64];
        let mut s = vec![0u8; 32];
        unpack(&l, &mut codes, &mut s).unwrap();
        for i in 0..64 {
            assert_eq!(i32::from(codes[i]), q[i] - 128, "code {i}");
        }
        // scales transposed [K/g, N] -> [N, K/g]
        for row in 0..8 {
            for gi in 0..2 {
                assert_eq!(
                    u16::from_le_bytes([s[(row * 2 + gi) * 2], s[(row * 2 + gi) * 2 + 1]]),
                    half_bits(sc[gi * 8 + row])
                );
            }
        }
        // dequant of the unpacked form == the per-element reference
        let want = dequant_reference(&l);
        for row in 0..8 {
            for col in 0..8 {
                let o = (row * 2 + col / 4) * 2;
                let sv = crate::clef::st::f16_to_f32(u16::from_le_bytes([s[o], s[o + 1]]));
                let got = f32::from(codes[row * 8 + col]) * sv;
                assert_eq!(got, want[row * 8 + col], "w[{row},{col}]");
                assert_eq!(
                    got,
                    (q[row * 8 + col] - 128) as f32 * sc[(col / 4) * 8 + row]
                );
            }
        }
        // the extremes: q=0 -> -128, q=255 -> 127
        assert_eq!(0x80u32, 128);
    }

    #[test]
    fn a_non_symmetric_zero_point_is_refused() {
        let (qw, mut qz, scales, _, _) = tiny();
        qz[5] = 0x70;
        let l = QLinear {
            qweight: &qw,
            qzeros: &qz,
            scales: &scales,
            n: 8,
            k: 8,
            group: 4,
        };
        assert!(unpack(&l, &mut [0i8; 64], &mut [0u8; 32]).is_err());
    }

    #[test]
    fn extreme_bytes_map_to_the_signed_range() {
        // one word 0x00 0x7f 0x80 0xff -> -128, -1, 0, 127
        let mut qw = vec![0u8; 4 * 4];
        qw[..4].copy_from_slice(&0xff80_7f00u32.to_le_bytes());
        let qz: Vec<u8> = STORED_ZERO_WORD.to_le_bytes().to_vec();
        let scales = vec![0u8; 4 * 2];
        let l = QLinear {
            qweight: &qw,
            qzeros: &qz,
            scales: &scales,
            n: 4,
            k: 4,
            group: 4,
        };
        let mut codes = vec![0i8; 16];
        unpack(&l, &mut codes, &mut [0u8; 8]).unwrap();
        assert_eq!(&codes[..4], &[-128, -1, 0, 127]);
    }
}
