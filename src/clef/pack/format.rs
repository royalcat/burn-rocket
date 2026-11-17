//! Clef pack format v1: the on-disk layout shared by the packer and the reader.
//!
//! See `docs/experiments/clef-pack.md` for the prose description. Everything is little-endian.
//!
//! ```text
//! layer_NN.experts
//!   [0, 4096)                 header (below), zero padded
//!   [4096 + e * stride, ...)  expert e's record, `stride = round_up(record_bytes, 4096)`
//! record (offsets are relative to the record start and recorded in the header)
//!   gu_codes     i8  [2*inter, hidden]     gate rows then up rows, N-major (`[N, K]`)
//!   down_codes   i8  [hidden, inter]
//!   gu_scales    f16 [2*inter, hidden/group]
//!   down_scales  f16 [hidden, inter/group]
//! ```
//!
//! The codes are the checkpoint's symmetric signed 8-bit values (`q - zero_point`), so
//! `w[n, k] = code[n, k] * scale[n, k / group]` with no further offset, and they are the
//! `[N, K]` row-major int8 layout `I8Context::pack_weight_gw` takes (scales
//! `[N, K/group]` like `matmul_gw`'s `b_scale`).

use anyhow::{Result, bail};

pub const MAGIC: [u8; 8] = *b"CLEFXPK1";
pub const FORMAT_VERSION: u32 = 1;
/// File and expert alignment (a page, and the O_DIRECT block size).
pub const ALIGN: usize = 4096;
/// The header block of a layer file.
pub const HEADER_BYTES: usize = ALIGN;

pub const MANIFEST: &str = "manifest.json";
pub const RESIDENT: &str = "resident.safetensors";
pub const LM_HEAD: &str = "lm_head.safetensors";
pub const LM_HEAD_TENSOR: &str = "thinker.lm_head.weight";

pub fn layer_file(layer: usize) -> String {
    format!("layer_{layer:02}.experts")
}

pub const fn round_up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

/// Geometry of one expert record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    pub hidden: usize,
    pub inter: usize,
    pub group: usize,
}

/// Where each section of a record starts (bytes from the record start) and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordLayout {
    pub gu_codes: usize,
    pub down_codes: usize,
    pub gu_scales: usize,
    pub down_scales: usize,
    /// Bytes used by the four sections.
    pub record_bytes: usize,
    /// `round_up(record_bytes, ALIGN)`: the distance between two experts.
    pub stride: usize,
}

impl Dims {
    pub fn validate(&self) -> Result<()> {
        if self.group == 0
            || !self.hidden.is_multiple_of(self.group)
            || !self.inter.is_multiple_of(self.group)
        {
            bail!(
                "group {} must divide hidden {} and inter {}",
                self.group,
                self.hidden,
                self.inter
            );
        }
        Ok(())
    }

    pub fn gu_codes_len(&self) -> usize {
        2 * self.inter * self.hidden
    }
    pub fn down_codes_len(&self) -> usize {
        self.hidden * self.inter
    }
    /// f16 elements.
    pub fn gu_scales_len(&self) -> usize {
        2 * self.inter * (self.hidden / self.group)
    }
    /// f16 elements.
    pub fn down_scales_len(&self) -> usize {
        self.hidden * (self.inter / self.group)
    }

    pub fn layout(&self) -> RecordLayout {
        // Sections start on a 64-byte boundary so the f16 scale rows can be read as slices.
        let a = |v: usize| round_up(v, 64);
        let gu_codes = 0;
        let down_codes = a(gu_codes + self.gu_codes_len());
        let gu_scales = a(down_codes + self.down_codes_len());
        let down_scales = a(gu_scales + 2 * self.gu_scales_len());
        let record_bytes = down_scales + 2 * self.down_scales_len();
        RecordLayout {
            gu_codes,
            down_codes,
            gu_scales,
            down_scales,
            record_bytes,
            stride: round_up(record_bytes, ALIGN),
        }
    }
}

/// Parsed `layer_NN.experts` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerHeader {
    pub layer: usize,
    pub experts: usize,
    pub dims: Dims,
    pub layout: RecordLayout,
    /// Byte offset of expert 0's record.
    pub data_start: usize,
}

const NFIELDS: usize = 14;

impl LayerHeader {
    pub fn new(layer: usize, experts: usize, dims: Dims) -> Self {
        Self {
            layer,
            experts,
            dims,
            layout: dims.layout(),
            data_start: HEADER_BYTES,
        }
    }

    /// Offset of expert `e`'s record in the file.
    pub fn expert_offset(&self, e: usize) -> u64 {
        (self.data_start + e * self.layout.stride) as u64
    }

    pub fn file_bytes(&self) -> u64 {
        self.expert_offset(self.experts)
    }

    pub fn encode(&self) -> Vec<u8> {
        let l = &self.layout;
        let fields: [u64; NFIELDS] = [
            u64::from(FORMAT_VERSION),
            self.layer as u64,
            self.experts as u64,
            self.dims.hidden as u64,
            self.dims.inter as u64,
            self.dims.group as u64,
            l.stride as u64,
            self.data_start as u64,
            l.gu_codes as u64,
            l.down_codes as u64,
            l.gu_scales as u64,
            l.down_scales as u64,
            l.record_bytes as u64,
            0,
        ];
        let mut out = vec![0u8; HEADER_BYTES];
        out[..8].copy_from_slice(&MAGIC);
        for (i, f) in fields.iter().enumerate() {
            out[8 + i * 8..16 + i * 8].copy_from_slice(&f.to_le_bytes());
        }
        out
    }

    pub fn decode(b: &[u8]) -> Result<Self> {
        if b.len() < 8 + NFIELDS * 8 || b[..8] != MAGIC {
            bail!("not a clef pack layer file (bad magic)");
        }
        let f =
            |i: usize| u64::from_le_bytes(b[8 + i * 8..16 + i * 8].try_into().unwrap()) as usize;
        if f(0) != FORMAT_VERSION as usize {
            bail!(
                "unsupported clef pack version {} (this build reads {FORMAT_VERSION})",
                f(0)
            );
        }
        let dims = Dims {
            hidden: f(3),
            inter: f(4),
            group: f(5),
        };
        dims.validate()?;
        let layout = RecordLayout {
            gu_codes: f(8),
            down_codes: f(9),
            gu_scales: f(10),
            down_scales: f(11),
            record_bytes: f(12),
            stride: f(6),
        };
        if layout != dims.layout() {
            bail!("layer header layout {layout:?} disagrees with the geometry {dims:?}");
        }
        let h = Self {
            layer: f(1),
            experts: f(2),
            dims,
            layout,
            data_start: f(7),
        };
        if h.data_start != HEADER_BYTES || !h.data_start.is_multiple_of(ALIGN) {
            bail!("unexpected data_start {}", h.data_start);
        }
        Ok(h)
    }
}

/// Borrowed sections of one expert record.
pub struct ExpertRecord<'a> {
    pub dims: Dims,
    /// `[2*inter, hidden]`, gate rows then up rows.
    pub gu_codes: &'a [i8],
    /// `[hidden, inter]`
    pub down_codes: &'a [i8],
    /// f16 bits `[2*inter, hidden/group]`
    pub gu_scales: &'a [u16],
    /// f16 bits `[hidden, inter/group]`
    pub down_scales: &'a [u16],
}

impl<'a> ExpertRecord<'a> {
    /// View the first `layout.record_bytes` of `buf` (must be 2-byte aligned: aligned
    /// read buffers always are).
    pub fn parse(dims: Dims, buf: &'a [u8]) -> Result<Self> {
        let l = dims.layout();
        if buf.len() < l.record_bytes {
            bail!(
                "expert buffer of {} bytes < record {}",
                buf.len(),
                l.record_bytes
            );
        }
        let codes = |off: usize, n: usize| -> &'a [i8] {
            // SAFETY: i8 and u8 have the same layout; range checked above.
            unsafe { std::slice::from_raw_parts(buf[off..off + n].as_ptr().cast::<i8>(), n) }
        };
        let words = |off: usize, n: usize| -> Result<&'a [u16]> {
            let bytes = &buf[off..off + 2 * n];
            // SAFETY: every bit pattern is a valid u16; alignment is checked by `align_to`.
            let (pre, mid, post) = unsafe { bytes.align_to::<u16>() };
            if !pre.is_empty() || !post.is_empty() {
                bail!("expert scales are not 2-byte aligned");
            }
            Ok(mid)
        };
        Ok(Self {
            dims,
            gu_codes: codes(l.gu_codes, dims.gu_codes_len()),
            down_codes: codes(l.down_codes, dims.down_codes_len()),
            gu_scales: words(l.gu_scales, dims.gu_scales_len())?,
            down_scales: words(l.down_scales, dims.down_scales_len())?,
        })
    }
}

/// Mutable sections of one expert record while it is being packed.
pub struct ExpertRecordMut<'a> {
    pub gu_codes: &'a mut [i8],
    pub down_codes: &'a mut [i8],
    pub gu_scales: &'a mut [u8],
    pub down_scales: &'a mut [u8],
}

impl<'a> ExpertRecordMut<'a> {
    /// Split one `stride`-byte record region into its sections.
    pub fn split(dims: Dims, region: &'a mut [u8]) -> Self {
        let l = dims.layout();
        let (a, rest) = region.split_at_mut(l.down_codes);
        let (b, rest) = rest.split_at_mut(l.gu_scales - l.down_codes);
        let (c, rest) = rest.split_at_mut(l.down_scales - l.gu_scales);
        let gu_codes = &mut a[..dims.gu_codes_len()];
        let down_codes = &mut b[..dims.down_codes_len()];
        // SAFETY: i8/u8 share a layout.
        let as_i8 = |s: &'a mut [u8]| -> &'a mut [i8] {
            unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<i8>(), s.len()) }
        };
        Self {
            gu_codes: as_i8(gu_codes),
            down_codes: as_i8(down_codes),
            gu_scales: &mut c[..2 * dims.gu_scales_len()],
            down_scales: &mut rest[..2 * dims.down_scales_len()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: Dims = Dims {
        hidden: 2048,
        inter: 768,
        group: 128,
    };

    #[test]
    fn the_real_record_is_4_57_mib_and_page_exact() {
        let l = REAL.layout();
        assert_eq!(l.gu_codes, 0);
        assert_eq!(l.down_codes, 3_145_728);
        assert_eq!(l.gu_scales, 3_145_728 + 1_572_864);
        assert_eq!(l.down_scales, l.gu_scales + 49_152);
        assert_eq!(l.record_bytes, 4_792_320); // 4.57 MiB
        assert_eq!(l.stride, l.record_bytes, "no padding at the real geometry");
        assert_eq!(l.stride % ALIGN, 0);
    }

    #[test]
    fn a_ragged_geometry_pads_the_stride_to_a_page() {
        let d = Dims {
            hidden: 64,
            inter: 32,
            group: 32,
        };
        let l = d.layout();
        assert_eq!(l.stride % ALIGN, 0);
        assert!(l.stride >= l.record_bytes && l.stride < l.record_bytes + ALIGN);
        // sections start 64-byte aligned and do not overlap
        for off in [l.down_codes, l.gu_scales, l.down_scales] {
            assert_eq!(off % 64, 0);
        }
        assert!(l.down_codes >= d.gu_codes_len());
        assert!(l.gu_scales >= l.down_codes + d.down_codes_len());
        assert!(l.down_scales >= l.gu_scales + 2 * d.gu_scales_len());
    }

    #[test]
    fn the_header_round_trips_and_rejects_corruption() {
        let h = LayerHeader::new(7, 128, REAL);
        let b = h.encode();
        assert_eq!(b.len(), HEADER_BYTES);
        assert_eq!(LayerHeader::decode(&b).unwrap(), h);
        assert_eq!(h.expert_offset(0), 4096);
        assert_eq!(h.expert_offset(3), 4096 + 3 * 4_792_320);
        assert_eq!(h.file_bytes(), 4096 + 128 * 4_792_320);
        let mut bad = b.clone();
        bad[0] = b'X';
        assert!(LayerHeader::decode(&bad).is_err());
        let mut bad = b.clone();
        bad[8] = 9; // version
        assert!(LayerHeader::decode(&bad).is_err());
        let mut bad = b;
        bad[8 + 6 * 8] ^= 1; // stride no longer matches the geometry
        assert!(LayerHeader::decode(&bad).is_err());
    }

    #[test]
    fn mutable_and_borrowed_views_cover_the_same_bytes() {
        let d = Dims {
            hidden: 64,
            inter: 32,
            group: 32,
        };
        let l = d.layout();
        let mut region = vec![0u8; l.stride];
        {
            let r = ExpertRecordMut::split(d, &mut region);
            r.gu_codes[0] = -3;
            *r.gu_codes.last_mut().unwrap() = 5;
            r.down_codes[0] = 7;
            r.gu_scales[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
            let n = r.down_scales.len();
            r.down_scales[n - 2..].copy_from_slice(&0x4000u16.to_le_bytes());
        }
        let v = ExpertRecord::parse(d, &region).unwrap();
        assert_eq!(v.gu_codes.len(), d.gu_codes_len());
        assert_eq!((v.gu_codes[0], *v.gu_codes.last().unwrap()), (-3, 5));
        assert_eq!(v.down_codes[0], 7);
        assert_eq!(v.gu_scales[0], 0x3c00);
        assert_eq!(*v.down_scales.last().unwrap(), 0x4000);
        assert_eq!(v.down_scales.len(), d.down_scales_len());
    }
}
