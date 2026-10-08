//! Projection-weight plumbing shared by the Qwen-family models: a CPU `Linear`
//! or an RK3588 NPU-resident weight (`--npu`), plus the projection kinds and
//! fused groups the Qwen3-Embedding NPU loader packs.

use burn::module::{
    Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper, ModuleVisitor,
    Param, ParamId,
};
use burn::nn::Linear;
use burn::prelude::*;

/// Which projection of a transformer layer (loader + `Proj` handles).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
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

#[cfg(all(feature = "npu", target_arch = "aarch64"))]
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

/// Projection groups that share one input activation and run as one NPU matmul
/// (their weights are packed concatenated along N).
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedGroup {
    Qkv = 0,
    GateUp = 1,
}

/// A projection weight: a regular CPU `Linear`, or (with `--npu`) a handle to a
/// weight resident on the RK3588 NPU. The NPU variant holds no CPU-side weights.
///
/// `Proj` is a transparent module wrapper: on the CPU path the store loads (and
/// `--quant q8` quantizes) the inner `Linear` under the usual `..._proj.weight`
/// key; in the NPU build the fields are `#[module(skip)]` and the weights are
/// packed straight from the store instead.
#[derive(Debug, Clone)]
pub enum Proj {
    Cpu(Linear),
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    Npu(burn_rocket::WeightId),
}

impl Module for Proj {
    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        match self {
            Proj::Cpu(l) => l.visit(visitor),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(_) => {}
        }
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        match self {
            Proj::Cpu(l) => Proj::Cpu(l.map(mapper)),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(id) => Proj::Npu(id),
        }
    }

    fn to_device(self, device: &Device) -> Self {
        match self {
            Proj::Cpu(l) => Proj::Cpu(l.to_device(device)),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(id) => Proj::Npu(id),
        }
    }

    fn fork(self, device: &Device) -> Self {
        match self {
            Proj::Cpu(l) => Proj::Cpu(l.fork(device)),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(id) => Proj::Npu(id),
        }
    }

    fn collect_devices(&self, devices: Devices) -> Devices {
        match self {
            Proj::Cpu(l) => l.collect_devices(devices),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(_) => devices,
        }
    }

    fn valid(&self) -> Self {
        match self {
            Proj::Cpu(l) => Proj::Cpu(l.valid()),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(id) => Proj::Npu(*id),
        }
    }

    fn train(self) -> Self {
        match self {
            Proj::Cpu(l) => Proj::Cpu(l.train()),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(id) => Proj::Npu(id),
        }
    }

    fn materialize(self) -> Self {
        match self {
            Proj::Cpu(l) => Proj::Cpu(l.materialize()),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(id) => Proj::Npu(id),
        }
    }
}

impl ModuleDisplayDefault for Proj {
    fn content(&self, content: Content) -> Option<Content> {
        match self {
            Proj::Cpu(l) => ModuleDisplayDefault::content(l, content),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(id) => Some(content.add_formatted(&format!("WeightId({})", id.id()))),
        }
    }
}

impl ModuleDisplay for Proj {}

impl Proj {
    /// A CPU linear whose weight is uninitialized until the store loads it (or
    /// the NPU loader replaces the whole `Proj`). In the NPU build the projection
    /// fields are `#[module(skip)]`, so only the NPU loader touches them.
    pub fn stub(input: usize, output: usize, device: &Device) -> Self {
        Proj::Cpu(Linear {
            weight: Param::uninitialized(
                ParamId::new(),
                move |device, _| Tensor::zeros([input, output], device),
                device.clone(),
                false,
                Shape::new([input, output]),
            ),
            bias: None,
        })
    }

    #[inline]
    pub fn forward(&self, x: Tensor<3>, quantized: bool) -> Tensor<3> {
        match self {
            Proj::Cpu(l) => linear_forward(l, x, quantized),
            #[cfg(all(feature = "npu", target_arch = "aarch64"))]
            Proj::Npu(id) => burn_rocket::matmul(x, id),
        }
    }
}

/// Linear forward that also supports Q8-resident weights: the weight is dequantized
/// to f32 on the fly and used through the normal float linear path. Only the current
/// layer's weights are materialized, so the resident model stays ~0.6 GB smaller.
pub fn linear_forward(lin: &Linear, x: Tensor<3>, quantized: bool) -> Tensor<3> {
    if !quantized {
        return lin.forward(x);
    }
    let weight = lin.weight.val().dequantize(); // [in, out] f32
    burn::tensor::module::linear(x, weight, lin.bias.as_ref().map(|b| b.val()))
}
