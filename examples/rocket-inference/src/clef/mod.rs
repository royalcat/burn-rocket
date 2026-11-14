//! Clef-Omni (Cloudflare): Qwen3-Omni thinker + `JointSchemaHead`. The device-free CPU
//! port of the head (phase 2) and of the layer-streamed thinker text decoder (phase 3).
//! Phase 4 adds the text-only encoder, request validation and the `ClefModel` pipeline
//! behind `decide` and `serve` (CPU only; media and NPU expert streaming come later).
#![allow(dead_code)]

pub mod answer;
pub mod encode;
pub mod head;
pub mod math;
pub mod model;
pub mod pipeline;
pub mod request;
pub mod st;
#[cfg(test)]
mod testutil;

#[allow(unused_imports)]
pub use head::{ClefHead, HeadConfig, LmHeadFile, LmHeadRows, QuestionSpans};
#[allow(unused_imports)]
pub use model::{Backbone, BackboneOut, ExpertSource, ExpertWeights, ForwardOpts, Tap, TextConfig};

#[allow(unused_imports)]
pub use encode::{EncodedQuestion, EncodedRecord, encode_record};
#[allow(unused_imports)]
pub use pipeline::{Answered, ClefModel};
#[allow(unused_imports)]
pub use request::{ClefRequest, Invalid};
