//! Reading a Clef pack: aligned O_DIRECT expert reads, resident tensors, and the
//! pack-backed weight sources for the CPU backbone.

use super::format::{ALIGN, Dims, ExpertRecord, HEADER_BYTES, LayerHeader, MANIFEST};
use super::{Manifest, TensorEntry};
use crate::clef::head::LmHeadFile;
use crate::clef::model::{ExpertSource, ExpertWeights, LayerWeights, TextConfig, WeightSource};
use crate::clef::st::{Dtype, SafeFile, TensorInfo, f16_to_f32};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::ffi::CString;
use std::path::{Path, PathBuf};

/// A page-aligned heap buffer (O_DIRECT needs aligned address, length and offset).
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: plain owned memory with no thread affinity.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        assert!(len > 0 && len.is_multiple_of(ALIGN));
        let layout = Layout::from_size_align(len, ALIGN).expect("layout");
        // SAFETY: non-zero size, valid layout.
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory");
        Self { ptr, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr/len describe our live allocation.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, ALIGN).expect("layout");
        // SAFETY: allocated with the same layout in `new`.
        unsafe { dealloc(self.ptr, layout) };
    }
}

/// A read-only file opened O_DIRECT when the filesystem allows it, else buffered with
/// `posix_fadvise(DONTNEED)` after every read (so a 28 GiB pack does not evict the cache).
pub struct DiskFile {
    fd: i32,
    direct: bool,
    len: u64,
}

impl DiskFile {
    pub fn open(path: &Path) -> Result<Self> {
        let c = CString::new(path.as_os_str().as_encoded_bytes())?;
        // SAFETY: valid NUL-terminated path; read-only flags.
        let try_open = |flags: i32| unsafe {
            libc::open(c.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC | flags)
        };
        let mut fd = if std::env::var_os("CLEF_PACK_NO_DIRECT").is_some() {
            -1
        } else {
            try_open(libc::O_DIRECT)
        };
        let mut direct = fd >= 0;
        if direct {
            // Some filesystems accept the flag at open and refuse the read.
            let mut probe = AlignedBuf::new(ALIGN);
            // SAFETY: aligned buffer of ALIGN bytes.
            let rc = unsafe { libc::pread(fd, probe.as_mut_slice().as_mut_ptr().cast(), ALIGN, 0) };
            if rc < 0 {
                // SAFETY: fd was just opened by us.
                unsafe { libc::close(fd) };
                fd = -1;
                direct = false;
            }
        }
        if fd < 0 {
            fd = try_open(0);
        }
        if fd < 0 {
            bail!(
                "open {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: zeroed stat filled by fstat.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            bail!(
                "fstat {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
        Ok(Self {
            fd,
            direct,
            len: st.st_size as u64,
        })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }

    /// Advise the kernel to drop this file's page cache (a no-op for O_DIRECT reads, which
    /// never populate it).
    pub fn drop_cache(&self) {
        if !self.direct {
            // SAFETY: advisory call on our fd; len 0 = to the end of the file.
            unsafe { libc::posix_fadvise(self.fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    /// Fill `buf` from `off` (stops short only at EOF); returns the bytes read.
    pub fn read_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
        let mut got = 0usize;
        while got < buf.len() {
            // SAFETY: the destination range is inside `buf`.
            let rc = unsafe {
                libc::pread(
                    self.fd,
                    buf[got..].as_mut_ptr().cast(),
                    buf.len() - got,
                    (off + got as u64) as libc::off_t,
                )
            };
            if rc < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            if rc == 0 {
                break;
            }
            got += rc as usize;
        }
        if !self.direct {
            // SAFETY: advisory call on our fd.
            unsafe {
                libc::posix_fadvise(
                    self.fd,
                    off as libc::off_t,
                    got as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                )
            };
        }
        Ok(got)
    }
}

impl Drop for DiskFile {
    fn drop(&mut self) {
        // SAFETY: closing our own fd.
        unsafe { libc::close(self.fd) };
    }
}

thread_local! {
    static EXPERT_BUF: RefCell<Option<AlignedBuf>> = const { RefCell::new(None) };
}

/// An opened pack directory.
pub struct PackDir {
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub dims: Dims,
    layers: Vec<(DiskFile, LayerHeader)>,
    resident: SafeFile,
}

impl PackDir {
    /// Whether `dir` looks like a pack (has a `manifest.json` with the pack's format name).
    pub fn is_pack(dir: &Path) -> bool {
        std::fs::read(dir.join(MANIFEST))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|v| v["format"] == super::FORMAT_NAME)
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let mp = dir.join(MANIFEST);
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(&mp).with_context(|| format!("read {}", mp.display()))?,
        )
        .with_context(|| format!("parse {}", mp.display()))?;
        let dims = manifest.validate()?;
        let m = &manifest.model;
        let mut layers = Vec::with_capacity(m.layers);
        for l in 0..m.layers {
            let name = super::format::layer_file(l);
            let path = dir.join(&name);
            let f = DiskFile::open(&path)?;
            let want = manifest
                .files
                .iter()
                .find(|x| x.name == name)
                .map(|x| x.bytes);
            if Some(f.len()) != want {
                bail!("{name}: {} bytes on disk, manifest says {want:?}", f.len());
            }
            let mut hb = AlignedBuf::new(HEADER_BYTES);
            if f.read_at(hb.as_mut_slice(), 0)? != HEADER_BYTES {
                bail!("{name}: short header");
            }
            let h = LayerHeader::decode(hb.as_slice()).with_context(|| name.clone())?;
            if h.layer != l || h.experts != m.experts || h.dims != dims {
                bail!("{name}: header disagrees with the manifest");
            }
            layers.push((f, h));
        }
        let resident = SafeFile::open(&dir.join(&manifest.resident.file))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            dims,
            layers,
            resident,
        })
    }

    pub fn layers(&self) -> usize {
        self.manifest.model.layers
    }

    pub fn experts(&self) -> usize {
        self.manifest.model.experts
    }

    /// Bytes of one expert read (record + padding to the stride).
    pub fn stride(&self) -> usize {
        self.dims.layout().stride
    }

    pub fn is_direct(&self) -> bool {
        self.layers.first().is_some_and(|(f, _)| f.is_direct())
    }

    pub fn new_buffer(&self) -> AlignedBuf {
        AlignedBuf::new(self.stride())
    }

    /// One read of expert `e` of `layer` into `buf` (at least `stride` bytes).
    pub fn read_expert(&self, layer: usize, e: usize, buf: &mut AlignedBuf) -> Result<()> {
        let (f, h) = self
            .layers
            .get(layer)
            .with_context(|| format!("layer {layer} out of range"))?;
        if e >= h.experts {
            bail!("expert {e} out of range (layer has {})", h.experts);
        }
        let stride = h.layout.stride;
        if buf.len() < stride {
            bail!("buffer of {} bytes < stride {stride}", buf.len());
        }
        let got = f.read_at(&mut buf.as_mut_slice()[..stride], h.expert_offset(e))?;
        if got != stride {
            bail!("layer {layer} expert {e}: short read ({got} of {stride} bytes)");
        }
        Ok(())
    }

    /// Run `f` on expert `e`'s record through this thread's reusable aligned buffer.
    pub fn with_expert<R>(
        &self,
        layer: usize,
        e: usize,
        f: impl FnOnce(&ExpertRecord<'_>) -> R,
    ) -> Result<R> {
        EXPERT_BUF.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().is_none_or(|b| b.len() < self.stride()) {
                *slot = Some(self.new_buffer());
            }
            let buf = slot.as_mut().unwrap();
            self.read_expert(layer, e, buf)?;
            let rec = ExpertRecord::parse(self.dims, buf.as_slice())?;
            Ok(f(&rec))
        })
    }

    /// Drop the page cache of `layer`'s expert file (buffered fallback only).
    pub fn drop_layer_cache(&self, layer: usize) {
        if let Some((f, _)) = self.layers.get(layer) {
            f.drop_cache();
        }
    }

    pub fn resident(&self) -> &SafeFile {
        &self.resident
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.resident
            .tensors
            .get(name)
            .with_context(|| format!("{name}: not in resident.safetensors"))
    }

    /// The manifest's table entry of a resident tensor.
    pub fn table_entry(&self, name: &str) -> Option<&TensorEntry> {
        self.manifest
            .resident
            .tensors
            .iter()
            .find(|t| t.name == name)
    }

    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        if t.shape != shape {
            bail!("{name}: shape {:?}, expected {shape:?}", t.shape);
        }
        Ok(self.resident.to_f32(t))
    }

    /// Dequantize `<module>.codes` / `<module>.scales` (`[n, k]` int8 + `[n, k/group]`
    /// f16) to f32 `[n, k]`.
    pub fn dequant_linear(&self, module: &str, n: usize, k: usize) -> Result<Vec<f32>> {
        let g = self.dims.group;
        let ct = self.tensor(&format!("{module}.codes"))?;
        let st = self.tensor(&format!("{module}.scales"))?;
        if ct.dtype != Dtype::I8
            || ct.shape != [n, k]
            || st.dtype != Dtype::F16
            || st.shape != [n, k / g]
        {
            bail!(
                "{module}: unexpected codes/scales shapes {:?} / {:?}",
                ct.shape,
                st.shape
            );
        }
        let codes = self.resident.bytes(ct);
        let scales = self.resident.bytes(st);
        let mut w = vec![0f32; n * k];
        w.par_chunks_mut(k).enumerate().for_each(|(row, out)| {
            let c = &codes[row * k..(row + 1) * k];
            let s = &scales[row * (k / g) * 2..(row + 1) * (k / g) * 2];
            dequant_row(c, s, g, out);
        });
        Ok(w)
    }

    /// Drop the resident file's mapped pages from this process.
    pub fn release(&self) {
        self.resident.release();
    }

    pub fn lm_head(&self) -> Result<LmHeadFile> {
        let lm = &self.manifest.lm_head;
        LmHeadFile::from_file(&self.dir.join(&lm.file), &lm.tensor)
    }
}

/// `out[j] = code[j] as f32 * scale[j / group]` for one row (`scales`: f16 LE bytes).
pub fn dequant_row(codes: &[u8], scales: &[u8], group: usize, out: &mut [f32]) {
    for (gi, (c, o)) in codes.chunks(group).zip(out.chunks_mut(group)).enumerate() {
        let s = f16_to_f32(u16::from_le_bytes([scales[gi * 2], scales[gi * 2 + 1]]));
        for (v, &q) in o.iter_mut().zip(c) {
            *v = f32::from(q as i8) * s;
        }
    }
}

fn dequant_rows_f16words(codes: &[i8], scales: &[u16], k: usize, group: usize, out: &mut [f32]) {
    let g = k / group;
    for (row, o) in out.chunks_mut(k).enumerate() {
        for (gi, (c, oo)) in codes[row * k..(row + 1) * k]
            .chunks(group)
            .zip(o.chunks_mut(group))
            .enumerate()
        {
            let s = f16_to_f32(scales[row * g + gi]);
            for (v, &q) in oo.iter_mut().zip(c) {
                *v = f32::from(q) * s;
            }
        }
    }
}

/// Widen an expert record to f32 (`codes * scale`).
pub fn dequant_expert(rec: &ExpertRecord<'_>) -> ExpertWeights {
    let d = rec.dims;
    let gu_n = d.inter * d.hidden;
    let mut gu = vec![0f32; 2 * gu_n];
    dequant_rows_f16words(rec.gu_codes, rec.gu_scales, d.hidden, d.group, &mut gu);
    let up = gu.split_off(gu_n);
    let mut down = vec![0f32; d.hidden * d.inter];
    dequant_rows_f16words(rec.down_codes, rec.down_scales, d.inter, d.group, &mut down);
    ExpertWeights { gate: gu, up, down }
}

/// [`ExpertSource`] over a [`PackDir`]: one O_DIRECT read per expert, widened to f32 on
/// the CPU so the phase-3 backbone runs on the packed weights.
pub struct PackedExperts<'a> {
    pub pack: &'a PackDir,
}

impl ExpertSource for PackedExperts<'_> {
    fn expert(&self, layer: usize, e: usize) -> Result<ExpertWeights> {
        self.pack.with_expert(layer, e, dequant_expert)
    }
}

/// The backbone's whole weight source backed by a pack: attention/router/norm tensors from
/// `resident.safetensors` (int8 projections dequantized per layer) and experts from the
/// layer files.
pub struct PackedWeights {
    pub pack: PackDir,
    cfg: TextConfig,
}

impl PackedWeights {
    pub fn open(dir: &Path, cfg: TextConfig) -> Result<Self> {
        let pack = PackDir::open(dir)?;
        let m = &pack.manifest.model;
        if (m.layers, m.hidden, m.experts, m.moe_intermediate)
            != (cfg.layers, cfg.hidden, cfg.experts, cfg.inter)
        {
            bail!("pack geometry disagrees with config.json");
        }
        Ok(Self { pack, cfg })
    }
}

impl ExpertSource for PackedWeights {
    fn expert(&self, layer: usize, e: usize) -> Result<ExpertWeights> {
        PackedExperts { pack: &self.pack }.expert(layer, e)
    }
}

impl WeightSource for PackedWeights {
    fn layer_weights(&self, l: usize) -> Result<LayerWeights> {
        let (c, p) = (&self.cfg, &self.pack);
        let (h, d) = (c.hidden, c.head_dim);
        let pre = format!("thinker.model.layers.{l}.");
        let f = |n: &str, shape: &[usize]| p.f32(&format!("{pre}{n}"), shape);
        let q =
            |n: &str, o: usize, i: usize| p.dequant_linear(&format!("{pre}self_attn.{n}"), o, i);
        Ok(LayerWeights {
            input_norm: f("input_layernorm.weight", &[h])?,
            post_norm: f("post_attention_layernorm.weight", &[h])?,
            q: q("q_proj", c.heads * d, h)?,
            k: q("k_proj", c.kv_heads * d, h)?,
            v: q("v_proj", c.kv_heads * d, h)?,
            o: q("o_proj", h, c.heads * d)?,
            q_norm: f("self_attn.q_norm.weight", &[d])?,
            k_norm: f("self_attn.k_norm.weight", &[d])?,
            router: f("mlp.gate.weight", &[c.experts, h])?,
        })
    }

    fn experts(&self) -> &dyn ExpertSource {
        self
    }

    fn embed_row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        let t = self.pack.tensor("thinker.model.embed_tokens.weight")?;
        self.pack.resident().row_f32(t, id as usize, out)
    }

    fn final_norm(&self) -> Result<Vec<f32>> {
        self.pack
            .f32("thinker.model.norm.weight", &[self.cfg.hidden])
    }

    fn release(&self) {
        self.pack.release();
    }

    fn layer_small(&self, l: usize) -> Result<crate::clef::model::LayerSmall> {
        let (c, p) = (&self.cfg, &self.pack);
        let (h, d) = (c.hidden, c.head_dim);
        let pre = format!("thinker.model.layers.{l}.");
        let f = |n: &str, shape: &[usize]| p.f32(&format!("{pre}{n}"), shape);
        Ok(crate::clef::model::LayerSmall {
            input_norm: f("input_layernorm.weight", &[h])?,
            post_norm: f("post_attention_layernorm.weight", &[h])?,
            q_norm: f("self_attn.q_norm.weight", &[d])?,
            k_norm: f("self_attn.k_norm.weight", &[d])?,
            router: f("mlp.gate.weight", &[c.experts, h])?,
        })
    }
}
