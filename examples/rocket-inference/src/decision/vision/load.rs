//! Reading the `visual.*` tensors of a d3 checkpoint into a [`VisionTower`].
//!
//! Every d3 checkpoint stores its vision tower beside the text backbone under
//! the `visual.` prefix (`visual.patch_embed.proj.weight`, `visual.blocks.{i}.*`,
//! `visual.merger.*`), BF16 in the shipped files. The app reads them through
//! [`crate::util::checkpoint::AnyCheckpoint`] (the [`TensorSource`] impl in
//! `decision/load.rs`), the standalone probe through its own safetensors reader,
//! so this module names no storage format.
//!
//! Orientation: safetensors stores `[out, in]`; the tower keeps `[in, out]` (so a
//! forward is `x @ W`), and biases are reshaped from `[n]` to `[1, n]`. The patch
//! embedding's Conv3d weight `[out, C, T, p, p]` is flattened to `[out, C*T*p*p]`
//! — the same `[channel, temporal, patch_row, patch_col]` order the processor
//! writes each patch row in, which is what makes the convolution a matmul.

use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::TensorData;

use super::config::VisionConfig;
use super::model::VisionTower;

/// The checkpoint key prefix of the vision tower.
pub const VISION_PREFIX: &str = "visual.";

/// One tensor by its HF key, or `None` when the checkpoint does not have it.
///
/// Implemented for [`crate::util::checkpoint::AnyCheckpoint`] in `decision/load.rs`
/// (the app) and by the standalone probe for its own safetensors reader.
pub trait TensorSource {
    fn vision_tensor(&mut self, key: &str) -> Result<Option<TensorData>>;
}

impl VisionTower {
    /// Build and fill a tower from a checkpoint's `visual.*` tensors.
    ///
    /// Reads exactly `python -c "visual"`'s tensor set: 153 tensors for d3-lite
    /// (12 blocks), `12 + 8` more per extra layer for d3-nano. A missing tensor is
    /// an error; the tower's own shape checks reject a mismatched geometry.
    pub fn load(
        source: &mut impl TensorSource,
        config: &VisionConfig,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let mut tower = VisionTower::new_stub(config.clone(), device);
        let mut applied = 0usize;
        let mut read = |key: String| -> Result<TensorData> {
            source
                .vision_tensor(&key)?
                .ok_or_else(|| anyhow::anyhow!("the checkpoint has no {key}"))
        };

        tower.set(
            "patch_embed.proj.weight",
            patch_weight(
                &read(format!("{VISION_PREFIX}patch_embed.proj.weight"))?,
                config,
                device,
            )?,
        )?;
        applied += 1;
        tower.set(
            "patch_embed.proj.bias",
            vector(
                &read(format!("{VISION_PREFIX}patch_embed.proj.bias"))?,
                device,
            )?,
        )?;
        applied += 1;
        tower.set(
            "pos_embed.weight",
            matrix(&read(format!("{VISION_PREFIX}pos_embed.weight"))?, device)?,
        )?;
        applied += 1;

        for index in 0..config.depth {
            for suffix in BLOCK_SUFFIXES {
                let key = format!("{VISION_PREFIX}blocks.{index}.{suffix}");
                let data = read(key)?;
                let tensor = if suffix.ends_with("weight") && !suffix.starts_with("norm") {
                    transpose(matrix(&data, device)?)
                } else {
                    vector(&data, device)?
                };
                tower.set(&format!("blocks.{index}.{suffix}"), tensor)?;
                applied += 1;
            }
        }

        for suffix in MERGER_SUFFIXES {
            let key = format!("{VISION_PREFIX}merger.{suffix}");
            let data = read(key)?;
            let tensor = if suffix.ends_with("weight") && !suffix.starts_with("norm") {
                transpose(matrix(&data, device)?)
            } else {
                vector(&data, device)?
            };
            tower.set(&format!("merger.{suffix}"), tensor)?;
            applied += 1;
        }

        let expected = tensor_count(config);
        if applied != expected {
            bail!("loaded {applied} vision tensors, expected {expected}");
        }
        Ok(tower)
    }
}

/// The per-block tensors, in checkpoint order.
pub const BLOCK_SUFFIXES: [&str; 12] = [
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias",
    "mlp.linear_fc2.weight",
    "mlp.linear_fc2.bias",
];

/// The merger's tensors, in checkpoint order.
pub const MERGER_SUFFIXES: [&str; 6] = [
    "norm.weight",
    "norm.bias",
    "linear_fc1.weight",
    "linear_fc1.bias",
    "linear_fc2.weight",
    "linear_fc2.bias",
];

/// How many `visual.*` tensors a tower of this geometry has.
pub fn tensor_count(config: &VisionConfig) -> usize {
    // patch weight + patch bias + position table + blocks + merger.
    3 + config.depth * BLOCK_SUFFIXES.len() + MERGER_SUFFIXES.len()
}

/// A `[out, in]` weight as the tower's `[in, out]`.
fn transpose(tensor: Tensor<2>) -> Tensor<2> {
    tensor.swap_dims(0, 1)
}

/// A 2-D tensor in the checkpoint's orientation.
fn matrix(data: &TensorData, device: &Device) -> Result<Tensor<2>> {
    if data.shape().num_dims() != 2 {
        bail!("expected a 2-D tensor, got {:?}", data.shape());
    }
    Ok(Tensor::<2>::from_data(data.clone(), device))
}

/// A bias as `[1, n]`: `[n]` from safetensors, `[n, 1]` from a reversed store.
fn vector(data: &TensorData, device: &Device) -> Result<Tensor<2>> {
    let shape = data.shape();
    Ok(match shape.num_dims() {
        1 => Tensor::<1>::from_data(data.clone(), device).reshape([1, shape.dims::<1>()[0]]),
        2 => {
            let [a, b] = shape.dims::<2>();
            if a != 1 && b != 1 {
                bail!("expected a bias vector, got {a}x{b}");
            }
            Tensor::<2>::from_data(data.clone(), device).reshape([1, a * b])
        }
        n => bail!("expected a 1-D bias, got {n} dimensions"),
    })
}

/// The patch embedding's Conv3d weight flattened to `[patch_dim, hidden]`.
///
/// The checkpoint stores `[hidden, C, T, p, p]`; the pixel rows the processor
/// writes are `[C, T, p, p]`, so the kernel equals a matmul over the flattened
/// patch (the runtime's `linearize_patch_embed` does exactly this).
fn patch_weight(data: &TensorData, config: &VisionConfig, device: &Device) -> Result<Tensor<2>> {
    let shape = data.shape();
    let hidden = config.hidden_size;
    let patch_dim = config.patch_dim();
    match shape.num_dims() {
        5 => {
            let [out, c, t, p, q] = shape.dims::<5>();
            let (c_expected, t_expected, p_expected) = (
                config.in_channels,
                config.temporal_patch_size,
                config.patch_size,
            );
            if (c, t, p, q) != (c_expected, t_expected, p_expected, p_expected) {
                bail!(
                    "patch_embed.proj.weight is [out, C={c}, T={t}, {p}, {q}], expected \
                     [out, {c_expected}, {t_expected}, {p_expected}, {p_expected}]"
                );
            }
            if out != hidden {
                bail!("patch_embed.proj.weight has {out} output channels, expected {hidden}");
            }
            let flat = Tensor::<5>::from_data(data.clone(), device).reshape([hidden, patch_dim]);
            Ok(transpose(flat))
        }
        2 => {
            let [a, b] = shape.dims::<2>();
            let flat = Tensor::<2>::from_data(data.clone(), device);
            if (a, b) == (hidden, patch_dim) {
                Ok(transpose(flat))
            } else if (a, b) == (patch_dim, hidden) {
                Ok(flat)
            } else {
                bail!(
                    "patch_embed.proj.weight is {a}x{b}, expected {hidden}x{patch_dim} \
                     (a flattened Conv3d) or its transpose"
                );
            }
        }
        n => bail!("patch_embed.proj.weight has {n} dimensions, expected 5"),
    }
}

/// Load the tower of a d3 checkpoint (`config.json` `vision_config` + `visual.*`).
pub fn load_tower(
    source: &mut impl TensorSource,
    config_json: &serde_json::Value,
    device: &Device,
) -> Result<VisionTower> {
    let config = VisionConfig::from_value(config_json.clone())
        .context("parse the vision config of the d3 checkpoint")?;
    VisionTower::load(source, &config, device)
}

#[cfg(test)]
mod tests {
    use super::super::processor::PreparedImage;
    use super::*;

    /// An in-memory source: key -> (shape, values).
    #[derive(Default)]
    struct Map {
        tensors: std::collections::HashMap<String, TensorData>,
    }

    impl Map {
        fn zeros(mut self, key: impl Into<String>, shape: &[usize]) -> Self {
            let n: usize = shape.iter().product();
            self.tensors
                .insert(key.into(), TensorData::new(vec![0.0f32; n], shape));
            self
        }
    }

    impl TensorSource for Map {
        fn vision_tensor(&mut self, key: &str) -> Result<Option<TensorData>> {
            Ok(self.tensors.get(key).cloned())
        }
    }

    fn config() -> VisionConfig {
        VisionConfig {
            depth: 2,
            hidden_size: 8,
            num_heads: 2,
            intermediate_size: 16,
            out_hidden_size: 4,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            num_position_embeddings: 9,
            rope_theta: 10_000.0,
            hidden_act: "gelu_pytorch_tanh".to_string(),
        }
    }

    fn lite_source(cfg: &VisionConfig) -> Map {
        let (h, i, merged) = (
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.merge_unit() * cfg.hidden_size,
        );
        let mut map = Map::default()
            .zeros(
                "visual.patch_embed.proj.weight",
                &[
                    h,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ],
            )
            .zeros("visual.patch_embed.proj.bias", &[h])
            .zeros("visual.pos_embed.weight", &[cfg.num_position_embeddings, h]);
        for index in 0..cfg.depth {
            for suffix in BLOCK_SUFFIXES {
                let shape: Vec<usize> = match suffix {
                    "attn.qkv.weight" => vec![3 * h, h],
                    "attn.qkv.bias" => vec![3 * h],
                    "attn.proj.weight" => vec![h, h],
                    "mlp.linear_fc1.weight" => vec![i, h],
                    "mlp.linear_fc1.bias" => vec![i],
                    "mlp.linear_fc2.weight" => vec![h, i],
                    _ => vec![h],
                };
                map = map.zeros(format!("visual.blocks.{index}.{suffix}"), &shape);
            }
        }
        map.zeros("visual.merger.norm.weight", &[h])
            .zeros("visual.merger.norm.bias", &[h])
            .zeros("visual.merger.linear_fc1.weight", &[merged, merged])
            .zeros("visual.merger.linear_fc1.bias", &[merged])
            .zeros(
                "visual.merger.linear_fc2.weight",
                &[cfg.out_hidden_size, merged],
            )
            .zeros("visual.merger.linear_fc2.bias", &[cfg.out_hidden_size])
    }

    #[test]
    fn a_full_tensor_set_loads() {
        let cfg = config();
        let mut source = lite_source(&cfg);
        let device = Device::default();
        let tower = VisionTower::load(&mut source, &cfg, &device).unwrap();
        assert_eq!(tensor_count(&cfg), 3 + 2 * 12 + 6);
        assert_eq!(tower.config, cfg);
        // The tower runs with the loaded (zero) weights: a 2x2 patch grid
        // merges into one token of the output width.
        let image = PreparedImage {
            pixel_values: Tensor::<2>::zeros([4, cfg.patch_dim()], &device),
            grid: [1, 2, 2],
            tokens: 1,
            source: (4, 4),
            resized: (4, 4),
        };
        assert_eq!(
            tower.encode(&image).unwrap().dims(),
            [1, cfg.out_hidden_size]
        );
    }

    #[test]
    fn a_missing_tensor_is_named() {
        let cfg = config();
        let mut source = lite_source(&cfg);
        source.tensors.remove("visual.blocks.1.mlp.linear_fc2.bias");
        let device = Device::default();
        let err = VisionTower::load(&mut source, &cfg, &device).unwrap_err();
        assert!(
            format!("{err:#}").contains("visual.blocks.1.mlp.linear_fc2.bias"),
            "{err:#}"
        );
    }

    #[test]
    fn a_transposed_weight_is_refused() {
        // The patch weight must be `[out, C, T, p, p]`; a `[C, T, p, p, out]`
        // tensor is refused instead of being silently mis-loaded.
        let cfg = config();
        let mut source = lite_source(&cfg);
        source.tensors.insert(
            "visual.patch_embed.proj.weight".to_string(),
            TensorData::new(
                vec![0.0f32; cfg.patch_dim() * cfg.hidden_size],
                [
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                    cfg.hidden_size,
                ],
            ),
        );
        let device = Device::default();
        assert!(
            VisionTower::load(&mut source, &cfg, &device)
                .is_err_and(|e| format!("{e:#}").contains("expected [out,"))
        );
    }

    #[test]
    fn the_tensor_count_follows_the_depth() {
        let mut cfg = config();
        cfg.depth = 24;
        assert_eq!(tensor_count(&cfg), 3 + 24 * 12 + 6);
    }
}
