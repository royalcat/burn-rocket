//! Clef-Omni (Cloudflare): Qwen3-Omni thinker + `JointSchemaHead`. The device-free CPU
//! port of the head (phase 2) and of the layer-streamed thinker text decoder (phase 3).
//! Phase 4 adds the text-only encoder, request validation and the `ClefModel` pipeline
//! behind `decide` and `serve`; phase 6 puts the thinker on the NPU, phase 8b adds the media
//! plumbing, 8c the vision tower (`images` requests) and 8d `videos` (silent ones).
#![allow(dead_code)]

pub mod answer;
pub mod attn;
pub mod autoround_store;
#[cfg(test)]
mod batch_tests;
pub mod encode;
pub mod engine;
#[cfg(all(feature = "npu", target_arch = "aarch64"))]
pub mod engine_npu;
pub mod head;
pub mod int8;
pub mod math;
pub mod media;
pub mod model;
pub mod pack;
pub mod pipeline;
pub mod request;
pub mod st;
#[cfg(test)]
mod testutil;
pub mod vision;

#[allow(unused_imports)]
pub use head::{ClefHead, HeadConfig, LmHeadFile, LmHeadRows, QuestionSpans};
#[allow(unused_imports)]
pub use model::{Backbone, BackboneOut, ExpertSource, ExpertWeights, ForwardOpts, Tap, TextConfig};

#[allow(unused_imports)]
pub use encode::{
    EncodedQuestion, EncodedRecord, RecordMedia, encode_record, encode_record_with_media,
    encode_request,
};
#[allow(unused_imports)]
pub use media::{MediaFeatures, MediaLayout, Positions, TowerOutputs, build_media_ids, rope_index};
#[allow(unused_imports)]
pub use pipeline::{Answered, BatchAnswers, ClefDevice, ClefModel, EngineKind};
#[allow(unused_imports)]
pub use request::{ClefRequest, Invalid};
