//! The d3 image path's vision half: the Qwen3.5 vision tower and the Qwen2-VL
//! image preprocessing that feeds it (Phase 2a).
//!
//! The tower turns one image into `[tokens, out_hidden]` merged tokens, i.e. the
//! features the text backbone sees in place of each `<|image_pad|>`; [`processor`]
//! produces the `[patches, 1536]` pixel matrix and the `[1, H/16, W/16]` grid from
//! a file. Nothing here touches the prompt: the multimodal RoPE positions, the
//! placeholder splice and the request's `images` field are Phase 2b, so
//! `system_one`/`decide`/`/v1/systemone` stay text-only for now.
//!
//! Split:
//!
//! | module | contents |
//! |---|---|
//! | [`config`] | `vision_config` geometry and its validation |
//! | [`processor`] | `smart_resize`, the antialiased bicubic uint8 resize, patchify |
//! | [`model`] | the tower forward (position table, axial RoPE, attention, merger) |
//! | [`load`] | the `visual.*` tensor reader (`TensorSource`) |

#![allow(dead_code, unused_imports)] // Phase 2b wires the tower into the image request;
// until then the main binary only re-exports it and `decision_vision_probe` is the one consumer.

pub mod config;
pub mod load;
pub mod model;
pub mod processor;

pub use config::{LAYER_NORM_EPS, VisionConfig};
pub use load::{TensorSource, VISION_PREFIX, load_tower, tensor_count};
pub use model::VisionTower;
pub use processor::{PreparedImage, prepare_from_rgb, prepare_image, smart_resize};
