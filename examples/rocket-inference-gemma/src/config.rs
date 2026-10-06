//! Deserialized HF configuration for the EmbeddingGemma 2 checkpoint family
//! (`google/embeddinggemma-2`); the text sub-config is the adapted Gemma 4
//! decoder shared with the Gemma 4 checkpoints.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;

/// Vocabulary size shared by EmbeddingGemma 2 and Gemma 4.
pub const DEFAULT_VOCAB: usize = 262_144;

fn default_vocab() -> usize {
    DEFAULT_VOCAB
}

fn default_eps() -> f64 {
    1e-6
}

fn default_act() -> String {
    "gelu_pytorch_tanh".to_string()
}

fn default_sliding_window() -> usize {
    512
}

fn default_patch_size() -> usize {
    16
}

fn default_pooling_kernel() -> usize {
    3
}

/// Top-level `config.json` of the checkpoint.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Emb2Config {
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(default)]
    pub audio_config: Option<AudioConfig>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub video_token_id: Option<u32>,
    #[serde(default)]
    pub audio_token_id: Option<u32>,
}

impl Emb2Config {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }
}

/// Per-layer attention override, keyed by layer index (zero-padded strings in
/// the JSON, e.g. `"05"`).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct LayerOverride {
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
}

/// RoPE parameters per layer type.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct RopeSpec {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub sliding_attention: Option<RopeSpec>,
    #[serde(default)]
    pub full_attention: Option<RopeSpec>,
}

/// The `text_config` (model_type `embedding_gemma2_text`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    #[serde(default = "default_vocab")]
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub embedding_dim: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_act")]
    pub hidden_activation: String,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub per_layer_config: HashMap<String, LayerOverride>,
    #[serde(default)]
    pub rope_parameters: RopeParameters,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl TextConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Resolve the effective override for a layer, tolerating both `"5"` and
    /// `"05"` key spellings.
    pub fn layer_override(&self, layer: usize) -> LayerOverride {
        for key in [layer.to_string(), format!("{layer:02}")] {
            if let Some(v) = self.per_layer_config.get(&key) {
                return v.clone();
            }
        }
        LayerOverride::default()
    }

    pub fn layer_types(&self) -> Vec<String> {
        if self.layer_types.len() == self.num_hidden_layers {
            return self.layer_types.clone();
        }
        // Fallback: 5 sliding, 1 full (the reference default pattern), last full.
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect()
    }
}

/// Gemma 4 vision tower config (subset needed for inference).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_act")]
    pub hidden_activation: String,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_pooling_kernel")]
    pub pooling_kernel_size: usize,
    #[serde(default)]
    pub position_embedding_size: usize,
    #[serde(default)]
    pub default_output_length: usize,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_parameters: Option<RopeSpec>,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// Gemma 4 audio tower config (subset needed for inference).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_audio_act")]
    pub hidden_act: String,
    #[serde(default = "default_subsample_channels")]
    pub subsampling_conv_channels: [usize; 2],
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f64,
    #[serde(default = "default_attention_chunk")]
    pub attention_chunk_size: usize,
    #[serde(default = "default_context_left")]
    pub attention_context_left: usize,
    #[serde(default)]
    pub attention_context_right: usize,
    #[serde(default = "default_logit_cap")]
    pub attention_logit_cap: f64,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub output_proj_dims: usize,
}

fn default_audio_act() -> String {
    "silu".to_string()
}

fn default_subsample_channels() -> [usize; 2] {
    [128, 32]
}

fn default_conv_kernel() -> usize {
    5
}

fn default_residual_weight() -> f64 {
    0.5
}

fn default_attention_chunk() -> usize {
    12
}

fn default_context_left() -> usize {
    13
}

fn default_logit_cap() -> f64 {
    50.0
}
