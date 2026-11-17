//! The d3 vision path: the Qwen3.5 vision tower plus the Qwen2-VL image and
//! Qwen3-VL video preprocessing that feed it.
//!
//! The tower turns one image (or one video's sampled frames) into
//! `[tokens, out_hidden]` merged tokens, i.e. the features the text backbone sees
//! in place of each `<|image_pad|>`/`<|video_pad|>`; [`processor`] produces the
//! `[patches, 1536]` pixel matrix and the `[1, H/16, W/16]` image or
//! `[T, H/16, W/16]` video grid from decoded RGB pixels, and [`video`] is the
//! d3 sampling half (2 fps, 4-32 frames, temporal pairs, timestamps).
//! [`crate::decision::model`] splices those features into the prompt and
//! [`crate::decision::mrope`] supplies the multimodal rotary positions;
//! [`crate::decision::request::MediaMode`] decides how an `images`/`videos` entry
//! is read (local path in the CLI, data URL on the server).
//!
//! Split:
//!
//! | module | contents |
//! |---|---|
//! | [`config`] | `vision_config` geometry and its validation |
//! | [`processor`] | `smart_resize`, the antialiased bicubic uint8 resize, patchify, video pairs |
//! | [`model`] | the tower forward (position table, axial RoPE, attention, merger) |
//! | [`video`] | decode (ffprobe/ffmpeg), 2 fps sampling, timestamps, the prompt group |
//! | [`load`] | the `visual.*` tensor reader (`TensorSource`) |

// This module is shared with the standalone probe `src/bin/decision_vision_probe.rs`
// (included by path), which consumes the whole surface — `prepare_image`,
// `smart_resize`, `patches`, `source`/`resized` and the load helpers. The app
// reaches most of it by full path, so the unused-item lints would fire here for
// things the probe (and the parity tooling) actually use.
#![allow(dead_code, unused_imports)]

pub mod config;
pub mod load;
pub mod model;
pub mod processor;
pub mod video;

pub use config::{LAYER_NORM_EPS, VisionConfig};
pub use load::{TensorSource, VISION_PREFIX, load_tower, tensor_count};
pub use model::VisionTower;
pub use processor::{
    PreparedImage, PreparedVideo, prepare_from_rgb, prepare_image, prepare_video, smart_resize,
};
pub use video::{
    DecodedVideo, decode as decode_video, sample_indices, timestamps, video_replacement,
};
