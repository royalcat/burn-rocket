//! Clef-Omni (Cloudflare): Qwen3-Omni thinker + `JointSchemaHead`. The device-free CPU
//! port of the head (phase 2) and of the layer-streamed thinker text decoder (phase 3).
//! Nothing is wired into a command, so the items are only used by tests for now.
#![allow(dead_code)]

pub mod answer;
pub mod head;
pub mod math;
pub mod model;
pub mod st;
#[cfg(test)]
mod testutil;

#[allow(unused_imports)]
pub use head::{ClefHead, HeadConfig, LmHeadFile, LmHeadRows, QuestionSpans};
#[allow(unused_imports)]
pub use model::{Backbone, BackboneOut, ExpertSource, ExpertWeights, ForwardOpts, Tap, TextConfig};
