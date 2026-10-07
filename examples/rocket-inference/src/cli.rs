//! Shared CLI plumbing: the family-first usage text and the flag parser used by
//! every family's subcommands.
//!
//! The CLI is `rocket-inference <family> <command> [flags]`; each family owns
//! its flag namespace (families do not share a global option struct).

use anyhow::{Result, bail};

/// Top-level usage text (`rocket-inference help`).
pub const USAGE: &str = "\
rocket-inference - Burn inference on the RK3588 (flex CPU backend, optional NPU offload)

usage: rocket-inference <family|serve> <command> [flags]

families and commands:
  serve   [--model-dir <dir>] [--family auto|qwen3|embeddinggemma|intent|gemma4] [flags]
          one server for one loaded model; the routes follow the model:
          /v1/embeddings (+ /embed) for embedding models, /v1/chat/completions
          and the Ollama API (/api/*) for chat models, /health and /v1/models always
  qwen3   bench | embed | gemm | tokenize
          Qwen3-Embedding-0.6B (OpenAI /v1/embeddings)
  intent  gen
          Qwen3.5-0.8B intent/query-planner (Ollama /api/chat, /api/generate)
  gemma   embed | bench | tokenize
          EmbeddingGemma 2 (text/image/video/audio embeddings)
  gemma   gen
          Gemma 4 E2B-it text generation

Flags are per family and command; see README.md.";

/// Parse one subcommand's `--flag value` / `--flag` arguments.
///
/// `take*` methods consume the flag (last occurrence wins), so a family that
/// only reads its own flags can call [`FlagArgs::finish`] to reject the rest.
pub struct FlagArgs {
    args: Vec<String>,
}

impl FlagArgs {
    pub fn new(it: impl Iterator<Item = String>) -> Self {
        Self {
            args: it.collect(),
        }
    }

    /// Remove and return the last value of `--flag value`.
    pub fn take(&mut self, flag: &str) -> Result<Option<String>> {
        let mut result = None;
        let mut i = 0;
        while i < self.args.len() {
            if self.args[i] == flag {
                if i + 1 >= self.args.len() {
                    bail!("missing value for {flag}");
                }
                let value = self.args.remove(i + 1);
                self.args.remove(i);
                result = Some(value);
            } else {
                i += 1;
            }
        }
        Ok(result)
    }

    /// Remove `--flag` (a boolean) and report whether it was present.
    pub fn take_bool(&mut self, flag: &str) -> bool {
        let before = self.args.len();
        self.args.retain(|a| a != flag);
        self.args.len() != before
    }

    /// Like [`FlagArgs::take_bool`], but distinguishes "absent" from "false".
    /// Useful for validating that a family actually accepts the flag.
    pub fn take_flag(&mut self, flag: &str) -> Option<bool> {
        self.take_bool(flag).then_some(true)
    }

    /// `--flag value` parsed as `T`.
    pub fn take_parsed<T>(&mut self, flag: &str) -> Result<Option<T>>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        match self.take(flag)? {
            Some(v) => v
                .parse::<T>()
                .map(Some)
                .map_err(|e| anyhow::anyhow!("bad value for {flag}: {e}")),
            None => Ok(None),
        }
    }

    /// `--flag a|b|c` restricted to a set of names.
    pub fn take_choice(&mut self, flag: &str, choices: &[&str]) -> Result<Option<String>> {
        let Some(v) = self.take(flag)? else {
            return Ok(None);
        };
        if !choices.contains(&v.as_str()) {
            bail!("{flag} must be {}, got {v}", choices.join("|"));
        }
        Ok(Some(v))
    }

    /// Reject any argument not consumed by the command.
    pub fn finish(self, what: &str) -> Result<()> {
        if let Some(unknown) = self.args.first() {
            bail!("unknown argument '{unknown}' for '{what}' (see README.md for flags)");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> FlagArgs {
        FlagArgs::new(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn take_last_value_wins() {
        let mut f = args(&["--text", "a", "--text", "b", "--out", "x"]);
        assert_eq!(f.take("--text").unwrap().as_deref(), Some("b"));
        assert_eq!(f.take("--out").unwrap().as_deref(), Some("x"));
        f.finish("test").unwrap();
    }

    #[test]
    fn take_reports_missing_value() {
        let mut f = args(&["--text"]);
        assert!(f.take("--text").is_err());
    }

    #[test]
    fn take_bool_and_choice() {
        let mut f = args(&["--npu", "--quant", "q8", "--transb"]);
        assert!(f.take_bool("--npu"));
        assert!(!f.take_bool("--pure-npu"));
        assert_eq!(f.take_choice("--quant", &["none", "q8"]).unwrap().as_deref(), Some("q8"));
        assert!(f.take_bool("--transb"));
        f.finish("test").unwrap();
    }

    #[test]
    fn choice_rejects_unknown_value() {
        let mut f = args(&["--quant", "q4"]);
        assert!(f.take_choice("--quant", &["none", "q8"]).is_err());
    }

    #[test]
    fn finish_rejects_leftovers() {
        let f = args(&["--wat"]);
        assert!(f.finish("qwen3 embed").is_err());
    }

    #[test]
    fn parsed_numbers() {
        let mut f = args(&["--tokens", "3633", "--port", "8383"]);
        assert_eq!(f.take_parsed::<usize>("--tokens").unwrap(), Some(3633));
        assert_eq!(f.take_parsed::<u16>("--port").unwrap(), Some(8383));
        assert!(f.take_parsed::<usize>("--reps").unwrap().is_none());
        // `--tokens have-to-parse`
        let mut bad = args(&["--tokens", "many"]);
        assert!(bad.take_parsed::<usize>("--tokens").is_err());
    }
}
