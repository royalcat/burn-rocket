//! RK3588 NPU offload for the model's projections (`--npu`).
//!
//! Projection weights are packed once into resident NPU buffers (fp16, through
//! `librocketnpu`); the CPU keeps no copies (pack-and-drop). Every projection
//! forward is converted to fp16, run on the NPU and converted back to f32 for
//! the CPU-side ops (attention, norms, RoPE, activations).
//!
//! aarch64-only, behind the `npu` cargo feature.

#![cfg(all(feature = "npu", target_arch = "aarch64"))]

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_rocket::half::f16;
use burn_rocket::{RocketCtx, RocketWeight};
use std::sync::Arc;

/// Which projection of a transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjKind {
    Q = 0,
    K = 1,
    V = 2,
    O = 3,
    Gate = 4,
    Up = 5,
    Down = 6,
}

impl ProjKind {
    pub const ALL: [ProjKind; 7] = [
        Self::Q,
        Self::K,
        Self::V,
        Self::O,
        Self::Gate,
        Self::Up,
        Self::Down,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Safetensors key suffix within `model.layers.{i}`.
    pub fn key(self) -> &'static str {
        match self {
            Self::Q => "self_attn.q_proj.weight",
            Self::K => "self_attn.k_proj.weight",
            Self::V => "self_attn.v_proj.weight",
            Self::O => "self_attn.o_proj.weight",
            Self::Gate => "mlp.gate_proj.weight",
            Self::Up => "mlp.up_proj.weight",
            Self::Down => "mlp.down_proj.weight",
        }
    }
}

struct Slot {
    weight: RocketWeight,
    k: usize,
    n: usize,
}

// SAFETY: `librocketnpu` contexts are not thread-safe, but this model is only used
// from one thread at a time (the CLI main thread, or the server's request handler
// behind a mutex). Burn's `Module` trait requires its containers to be `Send + Sync`.
unsafe impl Send for NpuModel {}
unsafe impl Sync for NpuModel {}

/// One NPU context holding the resident weights of the whole model.
///
/// Not `Send`/`Sync` (`librocketnpu` contexts are single-threaded); the server
/// serializes requests, so one instance per process is enough.
pub struct NpuModel {
    ctx: RocketCtx,
    slots: Vec<Option<Slot>>,
    n_layers: usize,
}

impl NpuModel {
    pub fn new(n_layers: usize, nthreads: usize) -> anyhow::Result<Self> {        let ctx = RocketCtx::new(nthreads)
            .map_err(|e| anyhow::anyhow!("NPU context creation failed: {e}"))?;
        Ok(Self {
            ctx,
            slots: (0..n_layers * ProjKind::ALL.len()).map(|_| None).collect(),
            n_layers,
        })
    }

    /// Pack a `[n, k]` row-major fp16 weight (HF layout `[out, in]`, no transpose)
    /// into a resident NPU buffer.
    pub fn pack(
        &mut self,
        layer: usize,
        kind: ProjKind,
        k: usize,
        n: usize,
        b: &[f16],
    ) -> anyhow::Result<()> {
        assert!(layer < self.n_layers);
        // M is only a warm-up row count for tiling; the handle is reused for any
        // M >= 256 with the same K/N.
        let weight = self
            .ctx
            .pack_weight(512, k, n, b)
            .map_err(|e| anyhow::anyhow!("NPU weight pack failed (layer {layer}, {kind:?}): {e}"))?;
        self.slots[layer * ProjKind::ALL.len() + kind.index()] = Some(Slot { weight, k, n });
        Ok(())
    }

    /// `x` is `[1, m, k]` f32; returns `[1, m, n]` f32.
    pub fn forward(&self, layer: usize, kind: ProjKind, x: Tensor<3>) -> Tensor<3> {
        let [b, m, k] = x.dims();
        assert_eq!(b, 1, "the NPU path supports batch size 1");
        let slot = self.slots[layer * ProjKind::ALL.len() + kind.index()]
            .as_ref()
            .unwrap_or_else(|| panic!("NPU weight not packed (layer {layer}, {kind:?})"));
        assert_eq!(k, slot.k, "activation K mismatch for {kind:?}");
        let n = slot.n;

        let data = x.to_data();
        let v: Vec<f32> = data.try_to_vec().expect("f32 activations");
        let a16 = burn_rocket::f32_to_f16(&v);

        // The resident weight is packed for the M >= 256 tiling (Mt is capped there and
        // the layout is M-independent). Small requests are padded up to 256 rows so they
        // can reuse the same resident weights instead of needing a re-pack; the extra
        // rows are ignored on readback.
        let padded_m = m.div_ceil(4) * 4;
        let padded_m = if padded_m < 256 { 256 } else { padded_m };
        let a16 = if padded_m == m {
            a16
        } else {
            burn_rocket::pad_rows(&a16, m, k, padded_m)
        };

        let mut c16 = vec![f16::ZERO; padded_m * n];
        self.ctx
            .matmul_prepacked(padded_m, k, n, &a16, &mut c16, &slot.weight)
            .unwrap_or_else(|e| panic!("NPU matmul failed (layer {layer}, {kind:?}): {e}"));

        let c32: Vec<f32> = c16[..m * n].iter().map(|x| x.to_f32()).collect();
        Tensor::<3>::from_data(TensorData::new(c32, [1, m, n]), &x.device())
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}

/// A handle to one packed projection, stored in the model's `Proj` fields.
#[derive(Clone)]
pub struct NpuRef {
    model: Arc<NpuModel>,
    layer: usize,
    kind: ProjKind,
}

impl std::fmt::Debug for NpuRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NpuRef(layer={}, {:?})", self.layer, self.kind)
    }
}

impl NpuRef {
    pub fn new(model: Arc<NpuModel>, layer: usize, kind: ProjKind) -> Self {
        Self { model, layer, kind }
    }

    #[inline]
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.model.forward(self.layer, self.kind, x)
    }
}
