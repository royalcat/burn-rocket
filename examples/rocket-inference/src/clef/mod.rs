//! Clef-Omni (Cloudflare): Qwen3-Omni thinker + `JointSchemaHead`. This module is the
//! device-free CPU port of the head (phase 2); the thinker backbone is not here yet.
//! Nothing is wired into a command, so the items are only used by tests for now.
#![allow(dead_code)]

pub mod answer;
pub mod head;
pub mod math;
pub mod st;

#[allow(unused_imports)]
pub use head::{ClefHead, HeadConfig, LmHeadFile, LmHeadRows, QuestionSpans};
