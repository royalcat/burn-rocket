//! The d3 vision tower's geometry (`config.json` `vision_config`).
//!
//! The checkpoint's image processor and tower are the Qwen3.5-VL pair: the
//! processor is Qwen2-VL's (`patch_size * merge_size = 32` pixel factor), the
//! tower a plain pre-norm ViT with axial 2-D RoPE on the patch grid and a
//! 2x2 patch merger. Everything here is dimension-generic (d3-lite and d3-nano
//! differ only in `hidden_size`/`depth`/`num_heads`/`intermediate_size`).
//!
//! Reference: `transformers/models/qwen3_5/configuration_qwen3_5.py`
//! (`Qwen3_5VisionConfig`) and `modeling_qwen3_5.py` (`Qwen3_5VisionModel`).
//! Note that the tower's LayerNorm eps is *hard-coded* to `1e-6` in the model
//! (`nn.LayerNorm(..., eps=1e-6)`), so it is a constant here, not a config field.

use anyhow::{Result, bail};

/// The LayerNorm eps every vision norm uses (`nn.LayerNorm(..., eps=1e-6)`).
pub const LAYER_NORM_EPS: f64 = 1e-6;

/// The tower's geometry, validated from `vision_config`.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    /// Transformer blocks.
    pub depth: usize,
    /// Patch-embedding / residual width.
    pub hidden_size: usize,
    /// Attention heads (`head_dim = hidden_size / num_heads`).
    pub num_heads: usize,
    /// MLP width.
    pub intermediate_size: usize,
    /// Width of one merged image token (the text backbone's hidden size).
    pub out_hidden_size: usize,
    /// Spatial patch side.
    pub patch_size: usize,
    /// Frames per temporal patch (the still image is duplicated over them).
    pub temporal_patch_size: usize,
    /// Patches per side of one merged token.
    pub spatial_merge_size: usize,
    /// Input channels (RGB).
    pub in_channels: usize,
    /// Rows of the learned position table (a `side x side` square).
    pub num_position_embeddings: usize,
    /// Axial RoPE theta.
    pub rope_theta: f64,
    /// `hidden_act` (only `gelu_pytorch_tanh` is implemented).
    pub hidden_act: String,
}

impl VisionConfig {
    /// `head_dim`: per-head width (`hidden_size / num_heads`).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Values per patch as the processor writes them (`C * T * p * p`).
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Patches per merged image token (`merge_size^2`).
    pub fn merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// The position table's side (`int(sqrt(num_position_embeddings))`).
    pub fn position_grid_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt().round() as usize
    }

    /// Image tokens of one image whose patch grid is `(grid_h, grid_w)`.
    pub fn merged_tokens(&self, grid_h: usize, grid_w: usize) -> usize {
        grid_h * grid_w / self.merge_unit()
    }

    /// The processor's pixel factor: a resized side is a multiple of this.
    pub fn pixel_factor(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    /// Parse and validate the `vision_config` object of a `config.json`.
    pub fn from_value(value: serde_json::Value) -> Result<Self> {
        let object = value
            .get("vision_config")
            .cloned()
            .unwrap_or_else(|| value.clone());
        if !object.is_object() {
            bail!("vision_config is not a JSON object");
        }
        let number = |key: &str| -> Result<Option<f64>> {
            match object.get(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => v
                    .as_f64()
                    .map(Some)
                    .ok_or_else(|| anyhow::anyhow!("vision_config.{key} is not a number")),
            }
        };
        let size = |key: &str| -> Result<Option<usize>> {
            Ok(match number(key)? {
                Some(v) if v >= 0.0 && v.fract() == 0.0 => Some(v as usize),
                Some(v) => bail!("vision_config.{key} is not a non-negative integer ({v})"),
                None => None,
            })
        };
        let required = |key: &str| -> Result<usize> {
            size(key)?.ok_or_else(|| anyhow::anyhow!("vision_config.{key} is missing"))
        };
        let rope = object
            .get("rope_parameters")
            .or_else(|| object.get("rope_scaling"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        if let Some(deepstack) = object
            .get("deepstack_visual_indexes")
            .and_then(serde_json::Value::as_array)
            && !deepstack.is_empty()
        {
            bail!(
                "vision_config.deepstack_visual_indexes has {} entries; the deepstack merger \
                 is not implemented (both d3 checkpoints ship an empty list)",
                deepstack.len()
            );
        }
        let rope_theta = match rope.get("rope_theta") {
            Some(value) => value
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("rope_parameters.rope_theta is not a number"))?,
            None => 10_000.0,
        };
        let rope_type = rope
            .get("rope_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("axial")
            .to_string();
        let config = Self {
            depth: required("depth")?,
            hidden_size: required("hidden_size")?,
            num_heads: required("num_heads")?,
            intermediate_size: required("intermediate_size")?,
            out_hidden_size: required("out_hidden_size")?,
            patch_size: required("patch_size")?,
            temporal_patch_size: required("temporal_patch_size")?,
            spatial_merge_size: required("spatial_merge_size")?,
            in_channels: size("in_channels")?.unwrap_or(3),
            num_position_embeddings: required("num_position_embeddings")?,
            rope_theta,
            hidden_act: object
                .get("hidden_act")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("gelu_pytorch_tanh")
                .to_string(),
        };
        config.validate()?;
        if rope_type != "axial" {
            bail!(
                "the vision tower implements axial RoPE only, the checkpoint asks for '{rope_type}'"
            );
        }
        Ok(config)
    }

    /// Reject a geometry the tower cannot run (before any weight is read).
    pub fn validate(&self) -> Result<()> {
        if self.depth == 0 {
            bail!("vision_config.depth is 0");
        }
        if self.hidden_size == 0 || self.num_heads == 0 || self.intermediate_size == 0 {
            bail!(
                "vision_config has an empty dimension (hidden {}, heads {}, intermediate {})",
                self.hidden_size,
                self.num_heads,
                self.intermediate_size
            );
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            bail!(
                "vision_config.hidden_size {} is not divisible by num_heads {}",
                self.hidden_size,
                self.num_heads
            );
        }
        if !self.head_dim().is_multiple_of(4) {
            bail!(
                "the axial RoPE needs a head dim divisible by 4, got {}",
                self.head_dim()
            );
        }
        if self.patch_size == 0 || self.temporal_patch_size == 0 || self.spatial_merge_size == 0 {
            bail!("vision_config has a zero patch/merge size");
        }
        if self.in_channels != 3 {
            bail!(
                "the image processor writes 3 RGB channels, vision_config.in_channels is {}",
                self.in_channels
            );
        }
        let side = self.position_grid_side();
        if side * side != self.num_position_embeddings {
            bail!(
                "vision_config.num_position_embeddings {} is not a perfect square (the position table is square)",
                self.num_position_embeddings
            );
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            bail!(
                "rope_theta must be finite and positive, got {}",
                self.rope_theta
            );
        }
        if self.hidden_act != "gelu_pytorch_tanh" {
            bail!(
                "the vision tower implements hidden_act 'gelu_pytorch_tanh' only, the checkpoint asks for '{}'",
                self.hidden_act
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lite() -> serde_json::Value {
        serde_json::json!({
            "vision_config": {
                "depth": 12,
                "hidden_size": 768,
                "num_heads": 12,
                "intermediate_size": 3072,
                "out_hidden_size": 1024,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "in_channels": 3,
                "num_position_embeddings": 2304,
                "hidden_act": "gelu_pytorch_tanh",
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
            }
        })
    }

    #[test]
    fn a_lite_config_parses() {
        let cfg = VisionConfig::from_value(lite()).unwrap();
        assert_eq!(cfg.depth, 12);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.patch_dim(), 3 * 2 * 16 * 16);
        assert_eq!(cfg.merge_unit(), 4);
        assert_eq!(cfg.position_grid_side(), 48);
        assert_eq!(cfg.merged_tokens(62, 56), 868);
        assert_eq!(cfg.pixel_factor(), 32);
        assert_eq!(cfg.rope_theta, 10_000.0);
    }

    #[test]
    fn a_nano_geometry_is_dimension_generic() {
        let mut value = lite();
        let vision = value.get_mut("vision_config").unwrap();
        vision["depth"] = serde_json::json!(24);
        vision["hidden_size"] = serde_json::json!(1024);
        vision["num_heads"] = serde_json::json!(16);
        vision["intermediate_size"] = serde_json::json!(4096);
        vision["out_hidden_size"] = serde_json::json!(2048);
        let cfg = VisionConfig::from_value(value).unwrap();
        assert_eq!(cfg.head_dim(), 64, "the rope layout is unchanged");
        assert_eq!(cfg.patch_dim(), 1536);
        assert_eq!(cfg.out_hidden_size, 2048);
    }

    #[test]
    fn a_bare_vision_config_object_parses() {
        let mut value = lite();
        let vision = value.get_mut("vision_config").unwrap().take();
        let cfg = VisionConfig::from_value(vision).unwrap();
        assert_eq!(cfg.depth, 12);
    }

    #[test]
    fn broken_geometries_are_refused() {
        let mut value = lite();
        value["vision_config"]["num_heads"] = serde_json::json!(0);
        assert!(
            VisionConfig::from_value(value)
                .is_err_and(|e| e.to_string().contains("empty dimension"))
        );

        let mut value = lite();
        value["vision_config"]["hidden_size"] = serde_json::json!(770);
        assert!(
            VisionConfig::from_value(value).is_err_and(|e| e.to_string().contains("divisible"))
        );

        let mut value = lite();
        value["vision_config"]["hidden_size"] = serde_json::json!(680);
        value["vision_config"]["num_heads"] = serde_json::json!(8); // head_dim 85
        assert!(
            VisionConfig::from_value(value)
                .is_err_and(|e| e.to_string().contains("divisible by 4"))
        );

        let mut value = lite();
        value["vision_config"]["num_position_embeddings"] = serde_json::json!(2300);
        assert!(
            VisionConfig::from_value(value)
                .is_err_and(|e| e.to_string().contains("perfect square"))
        );

        let mut value = lite();
        value["vision_config"]["hidden_act"] = serde_json::json!("silu");
        assert!(
            VisionConfig::from_value(value)
                .is_err_and(|e| e.to_string().contains("gelu_pytorch_tanh"))
        );

        let mut value = lite();
        value["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([5, 11]);
        assert!(
            VisionConfig::from_value(value).is_err_and(|e| e.to_string().contains("deepstack"))
        );

        let mut value = lite();
        value["vision_config"]["rope_parameters"]["rope_type"] = serde_json::json!("linear");
        assert!(VisionConfig::from_value(value).is_err_and(|e| e.to_string().contains("axial")));

        let mut value = lite();
        value["vision_config"]["in_channels"] = serde_json::json!(4);
        assert!(VisionConfig::from_value(value).is_err_and(|e| e.to_string().contains("3 RGB")));

        let mut value = lite();
        value["vision_config"]["depth"] = serde_json::json!("twelve");
        assert!(
            VisionConfig::from_value(value).is_err_and(|e| e.to_string().contains("not a number"))
        );
    }
}
