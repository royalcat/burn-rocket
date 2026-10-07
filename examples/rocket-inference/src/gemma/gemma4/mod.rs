//! Gemma 4 E2B-it: text generation (chat template, sampling, prefill/decode),
//! the OpenAI-compatible `/v1/chat/completions` server, and its CLI.

pub mod chat;
pub mod cli;
pub mod loader;
pub mod model;
pub mod server;
