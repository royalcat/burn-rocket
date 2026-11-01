//! The d3 image path's vision half: the Qwen3.5 vision tower and the Qwen2-VL
//! image preprocessing that feeds it.
//!
//! The tower turns one image into `[tokens, out_hidden]` merged tokens, i.e. the
//! features the text backbone sees in place of each `<|image_pad|>`; [`processor`]
//! produces the `[patches, 1536]` pixel matrix and the `[1, H/16, W/16]` grid from
//! decoded RGB pixels. [`crate::decision::model`] splices those features into the
//! prompt and [`crate::decision::mrope`] supplies the multimodal rotary positions;
//! [`crate::decision::request::MediaMode`] decides how an `images` entry is read
//! (local path in the CLI, data URL on the server).
//!
//! Split:
//!
//! | module | contents |
//! |---|---|
//! | [`config`] | `vision_config` geometry and its validation |
//! | [`processor`] | `smart_resize`, the antialiased bicubic uint8 resize, patchify |
//! | [`model`] | the tower forward (position table, axial RoPE, attention, merger) |
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

pub use config::{LAYER_NORM_EPS, VisionConfig};
pub use load::{TensorSource, VISION_PREFIX, load_tower, tensor_count};
pub use model::VisionTower;
pub use processor::{PreparedImage, prepare_from_rgb, prepare_image, smart_resize};
