//! Shared CLI plumbing: the command usage texts, the flag parser and the
//! per-stage CPU/NPU placement flags every command shares.
//!
//! The CLI is family-agnostic (`rocket-inference <command> [flags]`): the model
//! family is detected from the checkpoint and `--family` overrides it. Every
//! model command prints the resolved placement once, before loading.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use burn::tensor::DType;

use crate::family::Family;

/// Top-level usage text (`rocket-inference help`).
pub const USAGE: &str = "\
rocket-inference - Burn inference on the RK3588 (flex CPU backend, per-stage NPU offload)

usage: rocket-inference <command> [flags]

commands:
  serve      HTTP server for the detected checkpoint (embedding and/or chat routes)
  embed      one embedding for text and/or media (embedding models)
  bench      embedding throughput over repeated forwards (embedding models)
  gen        one chat completion (chat models)
  tokenize   tokenize text with the checkpoint's tokenizer
  gemm       flex matmul microbenchmark (no model)
  help       this text; `rocket-inference help <command>` for one command

The model family is detected from --model-dir (config.json `model_type`, or the
GGUF architecture of a local *.gguf); --family overrides the detection.
--model-dir <dir> is required (or `-hf <owner>/<name>:<quant>`).

Every stage runs on the CPU unless it is placed on the NPU:
  --device auto|cpu|npu        base placement for every stage (default auto)
  --text-device ...            text-backbone projections
  --text-attn-device ...       text-backbone attention (qwen3, embeddinggemma)
  --vision-device ...          vision-tower projections (embeddinggemma)
  --vision-attn-device ...     vision-tower attention (embeddinggemma)
  --decode-device ...          single-token decode projections (chat models)
`auto` places a stage on the NPU when the build supports it (aarch64 + the
`npu` feature) and the CPU otherwise; decode defaults to the CPU even when the
NPU is available. `--device cpu` is the CPU path; `--device npu` places every
stage (decode included) on the NPU. A `placement:` line reports the resolved
split at load.

Flags are documented per command (`rocket-inference help <command>`), in
README.md and in docs/recipes/.";

/// Usage text for one command (`rocket-inference help <command>`).
pub fn command_usage(cmd: &str) -> Option<&'static str> {
    Some(match cmd {
        "serve" => SERVE_USAGE,
        "embed" => EMBED_USAGE,
        "bench" => BENCH_USAGE,
        "gen" => GEN_USAGE,
        "tokenize" => TOKENIZE_USAGE,
        "gemm" => GEMM_USAGE,
        _ => return None,
    })
}

const SERVE_USAGE: &str = "\
rocket-inference serve - one HTTP server for the detected checkpoint

usage: rocket-inference serve --model-dir <dir> [flags]

Routes follow the model: /v1/embeddings (+ /embed for multimodal embedding
models), /v1/chat/completions and the Ollama API for chat models; /health and
/v1/models always. `--family auto` detects the loader from the checkpoint.

server:
  --port <n>                   listen port (default 8383)
  --model-name <name>          served model name (default: the model dir name)
  --max-tokens <n>             prompt/context cap (family default)
  --max-new-tokens <n>         default generation cap (family default)
  --temperature <f>            default sampling temperature (default 0 = greedy)
  --delta-chunk <n>            delta-rule chunk width (intent, default 64)
  --video-fps <f>              video sampling rate (embeddinggemma, default 1.0)
  --video-max-frames <n>       video frame cap (embeddinggemma, default 32)

placement:
  --device / --text-device / --text-attn-device / --vision-device /
  --vision-attn-device / --decode-device   auto|cpu|npu
  --npu-threads / --npu-dtype / --int8-group / --keep-cpu-weights

model and memory:
  --model-dir / -hf / --family / --cpu-backend / --dtype / --quant /
  --defer-towers / --embed-f16 / --attn-chunk / --chunk / --key-block

See `rocket-inference help` for the shared placement flags and README.md for
the endpoints.";

const EMBED_USAGE: &str = "\
rocket-inference embed - one embedding for text and/or media

usage: rocket-inference embed --model-dir <dir> [flags]

Valid for embedding models (qwen3, embeddinggemma). Media inputs and the task
prompt are EmbeddingGemma 2 only; qwen3 accepts text/token inputs.

input:
  --text <s>                   inline text
  --text-file <path>           read the text from a file
  --tokens <n>                 truncate the tokenized input to n tokens
  --image <path>               image input (embeddinggemma)
  --video <path>               video input, 1 fps uniform cap (embeddinggemma)
  --audio <path>               audio input, WAV or ffmpeg-decodable (embeddinggemma)
  --max-soft-tokens <n>        image soft-token cap (embeddinggemma)
  --video-soft-tokens <n>      per-frame video soft tokens (embeddinggemma)
  --video-fps <f>              video sampling rate (default 1.0)
  --video-max-frames <n>       video frame cap (default 32)
  --prompt <name>              task prefix from config_sentence_transformers.json
                               (embeddinggemma; `none`/`raw` = no prefix)
  --dim <n>                    MRL truncation before normalization (embeddinggemma)
  --no-normalize               keep the raw vector (embeddinggemma; default L2)

output and debug:
  --out <path>                 write the vector as JSON (default: print it)
  --dump-pixels/--dump-ids/--dump-image/--dump-vision/--dump-vision-layers/
  --dump-audio/--dump-audio-layers <path>
                               raw reference dumps (embeddinggemma)

placement / model / memory:
  --device / --text-device / --text-attn-device / --vision-device /
  --vision-attn-device auto|cpu|npu
  --model-dir / -hf / --family / --cpu-backend / --dtype / --quant /
  --npu-threads / --npu-dtype / --int8-group / --defer-towers /
  --attn-kernel / --attn-chunk / --chunk / --key-block /
  --vision-attn-kernel

See `rocket-inference help` for the shared placement flags and docs/recipes/
for balanced, full-offload and speed command lines.";

const BENCH_USAGE: &str = "\
rocket-inference bench - embedding throughput over repeated forwards

usage: rocket-inference bench --model-dir <dir> [flags]

Valid for embedding models (qwen3, embeddinggemma). Uses the same input and
media flags as `embed` (`--text`, `--text-file`, `--image`, `--video`,
`--audio`, `--tokens`, `--max-soft-tokens`, `--video-*`, `--prompt`), but has
no `--out`; it adds:

  --reps <n>                   measured repetitions (default 3)

Each rep prints the wall time, tokens/s and the model's stage timings (and,
with `--device npu`, the NPU call/convert/device breakdown).

See `rocket-inference help embed` for the input flags.";

const GEN_USAGE: &str = "\
rocket-inference gen - one chat completion

usage: rocket-inference gen --model-dir <dir> [flags]

Valid for chat models (intent, gemma4).

input:
  --text <s>                   user message (or the raw prompt with --raw)
  --text-file <path>           read the user message from a file
  --max-tokens <n>             prompt-token cap (intent, default 30000)
  --raw                        skip the chat template (intent)
  --messages <json>            full message list, [{\"role\",\"content\"}]
                               (gemma4)
  --image <path>               image part (gemma4)
  --audio <path>               audio part (gemma4)
  --max-soft-tokens <n>        image soft-token cap (gemma4)

generation:
  --max-new-tokens <n>         token cap (default 256 intent / 64 gemma4)
  --temperature <f>            sampling temperature (default 0 = greedy)
  --sample                     enable sampling (gemma4; default greedy)
  --top-k <n> --top-p <f>      sampling cutoffs (gemma4)
  --enable-thinking            enable the thinking turn (gemma4)
  --delta-chunk <n>            delta-rule chunk width (intent, default 64)

output and debug:
  --out <path>                 write the generated token ids as JSON
  --dump-hidden/--dump-steps <path>   intent hidden-state / decode-step dumps
  --dump-logits <path>         gemma4 top-8 logits per step

placement / model / memory:
  --device / --text-device / --decode-device auto|cpu|npu
  --model-dir / -hf / --family / --cpu-backend / --dtype / --quant /
  --npu-threads / --keep-cpu-weights / --defer-towers / --attn-chunk /
  --embed-f16

See `rocket-inference help` for the shared placement flags and docs/recipes/
for balanced, full-offload and speed command lines.";

const TOKENIZE_USAGE: &str = "\
rocket-inference tokenize - tokenize text with the checkpoint's tokenizer

usage: rocket-inference tokenize --model-dir <dir> [flags]

  --text <s>                   inline text
  --text-file <path>           read the text from a file
  --prompt <name>              prepend a task prefix (embeddinggemma)
  --tokens <n>                 truncate to n tokens before printing

The model family is detected from --model-dir (or set with --family); see
`rocket-inference help` for those flags.";

const GEMM_USAGE: &str = "\
rocket-inference gemm - flex matmul microbenchmark (no model)

usage: rocket-inference gemm [flags]

  --m <n> --n <n> --k <n>      matrix shape (defaults 3632 3072 1024)
  --transb                     store B as [n, k] and transpose it for the matmul
  --reps <n>                   measured repetitions (default 3)
  --dtype auto|f32|f16|bf16    operand dtype (default f32)
  --quant none|q8              quantize B to Q8 before the matmul
  --cpu-backend flex|cpu       Burn backend (default flex)";

/// Parse one subcommand's `--flag value` / `--flag` arguments.
///
/// `take*` methods consume the flag (last occurrence wins), so a family that
/// only reads its own flags can call [`FlagArgs::finish`] to reject the rest.
pub struct FlagArgs {
    args: Vec<String>,
}

impl FlagArgs {
    pub fn new(it: impl Iterator<Item = String>) -> Self {
        Self { args: it.collect() }
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
            bail!("unknown argument '{unknown}' for '{what}' (see `rocket-inference help {what}`)");
        }
        Ok(())
    }
}

/// One stage's device choice (`--device` and the per-stage flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cpu,
    Npu,
}

impl Device {
    fn parse(value: &str) -> Self {
        match value {
            "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "npu" => Self::Npu,
            _ => unreachable!("validated by take_choice"),
        }
    }
}

/// Resident NPU weight precision (`--npu-dtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuDtype {
    Fp16,
    Int8,
}

impl NpuDtype {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
    }
}

/// Whether the CPU copies of NPU-packed projections are kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepCpu {
    Auto,
    Yes,
    No,
}

/// The CPU attention kernel (`--attn-kernel` / `--vision-attn-kernel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKernel {
    Fused,
    Blocked,
}

impl AttnKernel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fused => "fused",
            Self::Blocked => "blocked",
        }
    }
}

/// Whether this build has the RK3588 NPU backend (aarch64 + the `npu` feature).
pub fn npu_available() -> bool {
    cfg!(all(feature = "npu", target_arch = "aarch64"))
}

/// Flags as given (unset = the family/command default), before resolution.
#[derive(Debug, Default)]
pub struct LoadArgs {
    pub model_dir: Option<PathBuf>,
    pub family: Option<Family>,
    pub cpu_backend: Option<String>,
    pub dtype: Option<DType>,
    pub quant_q8: Option<bool>,
    pub defer_towers: Option<bool>,
    pub keep_cpu: Option<KeepCpu>,
    pub device: Option<Device>,
    pub text_device: Option<Device>,
    pub text_attn_device: Option<Device>,
    pub vision_device: Option<Device>,
    pub vision_attn_device: Option<Device>,
    pub decode_device: Option<Device>,
    pub npu_threads: Option<usize>,
    pub npu_dtype: Option<NpuDtype>,
    pub int8_group: Option<usize>,
    pub attn_kernel: Option<AttnKernel>,
    pub chunk: Option<usize>,
    pub key_block: Option<usize>,
    pub vision_attn_kernel: Option<AttnKernel>,
    pub attn_chunk: Option<usize>,
    pub embed_f16: Option<bool>,
}

impl LoadArgs {
    /// Consume the shared model/placement/memory flags from `f`.
    pub fn parse(f: &mut FlagArgs) -> Result<Self> {
        let mut a = Self::default();
        if let Some(v) = f.take("--model-dir")? {
            a.model_dir = Some(PathBuf::from(v));
        }
        if let Some(v) = f.take("--family")? {
            a.family = match v.as_str() {
                "auto" => None,
                other => Some(Family::parse(other)?),
            };
        }
        if let Some(v) = f.take_choice("--cpu-backend", &["flex", "cpu"])? {
            a.cpu_backend = Some(v);
        }
        if let Some(v) = f.take_choice("--dtype", &["auto", "f32", "f16", "bf16"])? {
            a.dtype = match v.as_str() {
                "auto" => None,
                "f32" => Some(DType::F32),
                "f16" => Some(DType::F16),
                "bf16" => Some(DType::BF16),
                _ => unreachable!(),
            };
        }
        if let Some(v) = f.take_choice("--quant", &["none", "q8"])? {
            a.quant_q8 = Some(v == "q8");
        }
        a.defer_towers = f.take_flag("--defer-towers");
        if let Some(v) = f.take_choice("--keep-cpu-weights", &["auto", "yes", "no"])? {
            a.keep_cpu = Some(match v.as_str() {
                "yes" => KeepCpu::Yes,
                "no" => KeepCpu::No,
                _ => KeepCpu::Auto,
            });
        }
        a.device = device_flag(f, "--device")?;
        a.text_device = device_flag(f, "--text-device")?;
        a.text_attn_device = device_flag(f, "--text-attn-device")?;
        a.vision_device = device_flag(f, "--vision-device")?;
        a.vision_attn_device = device_flag(f, "--vision-attn-device")?;
        a.decode_device = device_flag(f, "--decode-device")?;
        if let Some(v) = f.take_parsed("--npu-threads")? {
            if v == 0 {
                bail!("--npu-threads must be positive");
            }
            a.npu_threads = Some(v);
        }
        if let Some(v) = f.take_choice("--npu-dtype", &["fp16", "int8"])? {
            a.npu_dtype = Some(if v == "int8" {
                NpuDtype::Int8
            } else {
                NpuDtype::Fp16
            });
        }
        if let Some(v) = f.take_parsed("--int8-group")? {
            if v == 0 || v % 32 != 0 {
                bail!("--int8-group must be a positive multiple of 32 (got {v})");
            }
            a.int8_group = Some(v);
        }
        if let Some(v) = f.take_choice("--attn-kernel", &["fused", "blocked"])? {
            a.attn_kernel = Some(if v == "blocked" {
                AttnKernel::Blocked
            } else {
                AttnKernel::Fused
            });
        }
        a.chunk = f.take_parsed("--chunk")?;
        a.key_block = f.take_parsed("--key-block")?;
        if let Some(v) = f.take_choice("--vision-attn-kernel", &["fused", "blocked"])? {
            a.vision_attn_kernel = Some(if v == "blocked" {
                AttnKernel::Blocked
            } else {
                AttnKernel::Fused
            });
        }
        if let Some(v) = f.take_parsed("--attn-chunk")? {
            if v == 0 {
                bail!("--attn-chunk must be positive");
            }
            a.attn_chunk = Some(v);
        }
        a.embed_f16 = f.take_flag("--embed-f16");
        Ok(a)
    }

    /// `--model-dir`, with a message that names the `-hf` alternative.
    pub fn model_dir(&self) -> Result<PathBuf> {
        self.model_dir.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "--model-dir <dir> is required (or `-hf <owner>/<name>:<quant>` for a Hub GGUF)"
            )
        })
    }

    /// The `--family` override, else detection from the checkpoint.
    pub fn family(&self, model_dir: &Path) -> Result<Family> {
        match self.family {
            Some(f) => Ok(f),
            None => crate::family::detect_family(model_dir),
        }
    }

    /// Validate against the detected family and resolve defaults.
    pub fn resolve(self, model_dir: PathBuf, family: Family) -> Result<ResolvedLoad> {
        // ---- flags that only exist for some families ----
        if self.text_attn_device.is_some() && !family.supports_text_attn() {
            bail!(
                "--text-attn-device does not apply to the {} model (its attention is \
                 not a separately placeable stage)",
                family.name()
            );
        }
        if self.vision_device.is_some() && !family.supports_vision() {
            bail!(
                "--vision-device does not apply to the {} model (no vision tower)",
                family.name()
            );
        }
        if self.vision_attn_device.is_some() && !family.supports_vision() {
            bail!(
                "--vision-attn-device does not apply to the {} model (no vision tower)",
                family.name()
            );
        }
        if self.decode_device.is_some() && !family.supports_decode() {
            bail!(
                "--decode-device does not apply to the {} model (no decode stage)",
                family.name()
            );
        }
        if self.dtype.is_some() && !family.supports_dtype() {
            bail!(
                "--dtype does not apply to the {} model (it runs its native f32 body)",
                family.name()
            );
        }
        if self.quant_q8 == Some(true) && !family.supports_quant() {
            bail!("--quant does not apply to the {} model", family.name());
        }
        if self.defer_towers.is_some() && !family.supports_defer_towers() {
            bail!(
                "--defer-towers does not apply to the {} model (no multimodal towers)",
                family.name()
            );
        }
        if self.keep_cpu.is_some() && !family.supports_keep_cpu_weights() {
            bail!(
                "--keep-cpu-weights does not apply to the {} model \
                 (its packed projections never keep CPU copies)",
                family.name()
            );
        }
        if self.attn_kernel.is_some() && !family.supports_attn_kernel() {
            bail!(
                "--attn-kernel does not apply to the {} model",
                family.name()
            );
        }
        if (self.chunk.is_some() || self.key_block.is_some()) && !family.supports_attn_tiles() {
            bail!(
                "--chunk/--key-block only apply to the qwen3 CPU attention path, \
                 not the {} model",
                family.name()
            );
        }
        if self.vision_attn_kernel.is_some() && !family.supports_vision_attn_kernel() {
            bail!(
                "--vision-attn-kernel does not apply to the {} model",
                family.name()
            );
        }
        if self.attn_chunk.is_some() && !family.supports_attn_chunk() {
            bail!("--attn-chunk does not apply to the {} model", family.name());
        }
        if self.embed_f16.is_some() && !family.supports_embed_f16() {
            bail!("--embed-f16 does not apply to the {} model", family.name());
        }
        if self.npu_dtype.is_some() && !family.supports_int8() {
            bail!(
                "--npu-dtype only applies to the embeddinggemma text backbone, \
                 not the {} model",
                family.name()
            );
        }
        if self.int8_group.is_some() && !family.supports_int8() {
            bail!("--int8-group only applies to the embeddinggemma text backbone");
        }

        // ---- placement ----
        let avail = npu_available();
        let base = self.device.unwrap_or(Device::Auto);
        if base == Device::Npu && !avail {
            bail!("{}", npu_build_error("--device"));
        }
        let stage = |explicit: Option<Device>, auto_npu: bool, flag: &str| -> Result<bool> {
            stage_placement(explicit, base, auto_npu, avail)
                .ok_or_else(|| anyhow::anyhow!("{}", npu_build_error(flag)))
        };
        let placement = Placement {
            text: stage(self.text_device, true, "--text-device")?,
            text_attn: if family.supports_text_attn() {
                stage(self.text_attn_device, true, "--text-attn-device")?
            } else {
                false
            },
            vision: if family.supports_vision() {
                stage(self.vision_device, true, "--vision-device")?
            } else {
                false
            },
            vision_attn: if family.supports_vision() {
                stage(self.vision_attn_device, true, "--vision-attn-device")?
            } else {
                false
            },
            decode: if family.supports_decode() {
                stage(self.decode_device, false, "--decode-device")?
            } else {
                false
            },
        };

        // ---- NPU precision ----
        let npu_dtype = self.npu_dtype.unwrap_or(NpuDtype::Fp16);
        if npu_dtype == NpuDtype::Int8 {
            if !placement.text {
                bail!(
                    "--npu-dtype int8 requires the text projections on the NPU \
                     (`--text-device npu`)"
                );
            }
            if self.int8_group.is_some() && !avail {
                bail!("{}", npu_build_error("--int8-group"));
            }
        } else if self.int8_group.is_some() {
            bail!("--int8-group requires `--npu-dtype int8`");
        }
        let int8_group = self.int8_group.unwrap_or(32);

        // ---- CPU copies of NPU-packed projections ----
        let keep_cpu_weights = if family.supports_keep_cpu_weights() {
            if placement.decode && !placement.text {
                bail!(
                    "decode on the NPU requires the text projections packed \
                     (`--text-device npu`)"
                );
            }
            match self.keep_cpu {
                Some(KeepCpu::Yes) => true,
                Some(KeepCpu::No) => {
                    if !placement.decode {
                        bail!(
                            "the decode stage runs on the CPU and needs its CPU weights; \
                             place it on the NPU (`--decode-device npu`) or keep the copies \
                             (`--keep-cpu-weights yes`)"
                        );
                    }
                    if !placement.text {
                        bail!("--keep-cpu-weights no requires packed text projections");
                    }
                    false
                }
                Some(KeepCpu::Auto) | None => !placement.decode,
            }
        } else {
            true
        };

        // ---- dtype / backend / attention defaults ----
        let dtype_explicit = self.dtype;
        let dtype = dtype_explicit
            .unwrap_or_else(|| crate::util::checkpoint::default_dtype_in_dir(&model_dir));
        let cpu_backend = self.cpu_backend.clone().unwrap_or_else(default_backend);
        let attn_kernel = self.attn_kernel.unwrap_or(AttnKernel::Fused);
        let vision_attn_kernel = self.vision_attn_kernel.unwrap_or(AttnKernel::Fused);

        Ok(ResolvedLoad {
            model_dir,
            family,
            cpu_backend,
            dtype,
            dtype_explicit,
            quant_q8: self.quant_q8.unwrap_or(false),
            defer_towers: self.defer_towers.unwrap_or(false),
            keep_cpu_weights,
            placement,
            npu_threads: self.npu_threads.unwrap_or(5),
            npu_dtype,
            int8_group,
            attn_kernel,
            chunk: self.chunk.unwrap_or(256),
            key_block: self.key_block.unwrap_or(256),
            vision_attn_kernel,
            attn_chunk: self.attn_chunk.unwrap_or(1024),
            embed_f16: self.embed_f16.unwrap_or(false),
        })
    }
}

/// Resolve one stage: the explicit flag, else the base device, else the build
/// default. `None` = the stage asked for the NPU on a build without it.
fn stage_placement(
    explicit: Option<Device>,
    base: Device,
    auto_npu: bool,
    available: bool,
) -> Option<bool> {
    match explicit.unwrap_or(base) {
        Device::Auto => Some(available && auto_npu),
        Device::Cpu => Some(false),
        Device::Npu if available => Some(true),
        Device::Npu => None,
    }
}

fn device_flag(f: &mut FlagArgs, flag: &str) -> Result<Option<Device>> {
    Ok(f.take_choice(flag, &["auto", "cpu", "npu"])?
        .map(|v| Device::parse(&v)))
}

fn npu_build_error(flag: &str) -> String {
    format!(
        "{flag} npu requires an aarch64 build with the `npu` feature \
         (this build has no NPU support; use `cpu` or `auto`)"
    )
}

fn default_backend() -> String {
    if cfg!(feature = "cpu") {
        "cpu".to_string()
    } else {
        "flex".to_string()
    }
}

/// The resolved CPU/NPU split. `true` = the stage runs on the NPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Placement {
    pub text: bool,
    pub text_attn: bool,
    pub vision: bool,
    pub vision_attn: bool,
    pub decode: bool,
}

impl Placement {
    pub fn any_npu(self) -> bool {
        self.text || self.text_attn || self.vision || self.vision_attn || self.decode
    }

    /// The `--vision-attn` mode for the EmbeddingGemma 2 loader: the NPU op
    /// when the vision attention stage is placed there, else the CPU kernel.
    pub fn vision_attn(self, kernel: AttnKernel) -> crate::gemma::vision::VisionAttn {
        if self.vision_attn {
            crate::gemma::vision::VisionAttn::Npu
        } else {
            match kernel {
                AttnKernel::Fused => crate::gemma::vision::VisionAttn::Fused,
                AttnKernel::Blocked => crate::gemma::vision::VisionAttn::Blocked,
            }
        }
    }
}

/// Fully resolved load settings for one command.
#[derive(Debug)]
pub struct ResolvedLoad {
    pub model_dir: PathBuf,
    pub family: Family,
    pub cpu_backend: String,
    /// The resolved dtype (the checkpoint native when `--dtype` was absent).
    pub dtype: DType,
    /// `--dtype` as given; `None` = the checkpoint native.
    pub dtype_explicit: Option<DType>,
    pub quant_q8: bool,
    pub defer_towers: bool,
    pub keep_cpu_weights: bool,
    pub placement: Placement,
    pub npu_threads: usize,
    pub npu_dtype: NpuDtype,
    pub int8_group: usize,
    pub attn_kernel: AttnKernel,
    pub chunk: usize,
    pub key_block: usize,
    pub vision_attn_kernel: AttnKernel,
    pub attn_chunk: usize,
    pub embed_f16: bool,
}

impl ResolvedLoad {
    /// The one-line placement report printed before loading.
    pub fn placement_line(&self) -> String {
        let text_npu = self.placement.text;
        let text = if text_npu {
            format!("NPU({})", self.npu_dtype.name())
        } else {
            format!("CPU({})", dtype_name(self.dtype))
        };
        let attn = |on: bool, kernel: AttnKernel| {
            if on {
                "NPU".to_string()
            } else {
                format!("CPU({})", kernel.name())
            }
        };
        match self.family {
            Family::Qwen3Embed => format!(
                "placement: text={text} text-attn={}",
                attn(self.placement.text_attn, self.attn_kernel)
            ),
            Family::Emb2 => {
                let lazy = if self.defer_towers { "(lazy)" } else { "" };
                format!(
                    "placement: text={text} text-attn={} vision={}{lazy} vision-attn={} audio=CPU",
                    attn(self.placement.text_attn, self.attn_kernel),
                    if self.placement.vision {
                        "NPU".to_string()
                    } else {
                        "CPU".to_string()
                    },
                    attn(self.placement.vision_attn, self.vision_attn_kernel)
                )
            }
            Family::Intent | Family::Gemma4 => format!(
                "placement: text={}{} decode={}",
                text,
                if text_npu { " (prefill)" } else { "" },
                if self.placement.decode {
                    "NPU".to_string()
                } else {
                    "CPU".to_string()
                }
            ),
        }
    }
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> FlagArgs {
        FlagArgs::new(list.iter().map(|s| s.to_string()))
    }

    fn resolve(list: &[&str], family: Family) -> Result<ResolvedLoad> {
        let mut f = args(list);
        let load = LoadArgs::parse(&mut f)?;
        f.finish("test")?;
        load.resolve(PathBuf::from("/models/test"), family)
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
        assert!(!f.take_bool("--not-a-flag"));
        assert_eq!(
            f.take_choice("--quant", &["none", "q8"])
                .unwrap()
                .as_deref(),
            Some("q8")
        );
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

    // ---- placement resolution ----
    //
    // The host test build has no `npu` feature, so `auto` resolves to the CPU
    // and explicit `npu` is rejected. The aarch64 `npu` build resolves the same
    // cases the other way; the logic itself is shared.

    #[test]
    fn host_build_resolves_auto_to_cpu() {
        let r = resolve(&[], Family::Emb2).unwrap();
        assert_eq!(r.placement, Placement::default());
        assert!(r.placement_line().contains("text=CPU("));
        assert!(r.placement_line().contains("vision=CPU"));
        assert!(r.placement_line().contains("audio=CPU"));
    }

    #[test]
    fn an_npu_build_places_stages_by_default() {
        // `auto` on an NPU build: weights/attention/towers off, decode CPU.
        for stage in [true, true, true, true] {
            assert_eq!(stage_placement(None, Device::Auto, stage, true), Some(true));
        }
        assert_eq!(
            stage_placement(None, Device::Auto, false, true),
            Some(false)
        );
        // `--device npu` includes decode.
        assert_eq!(stage_placement(None, Device::Npu, false, true), Some(true));
        // `--device cpu` is the explicit CPU path.
        assert_eq!(stage_placement(None, Device::Cpu, true, true), Some(false));
        // A stage flag overrides the base; explicit `auto` is the build default.
        assert_eq!(
            stage_placement(Some(Device::Cpu), Device::Npu, true, true),
            Some(false)
        );
        assert_eq!(
            stage_placement(Some(Device::Auto), Device::Npu, false, true),
            Some(false)
        );
        // An explicit NPU stage on a CPU build is refused.
        assert_eq!(stage_placement(None, Device::Npu, true, false), None);
    }

    #[test]
    fn explicit_npu_is_rejected_without_the_feature() {
        let err = resolve(&["--device", "npu"], Family::Emb2).unwrap_err();
        assert!(format!("{err:#}").contains("aarch64 build"));
        let err = resolve(&["--vision-device", "npu"], Family::Emb2).unwrap_err();
        assert!(format!("{err:#}").contains("--vision-device npu"));
    }

    #[test]
    fn device_base_and_stage_overrides_resolve() {
        let r = resolve(&["--device", "cpu"], Family::Emb2).unwrap();
        assert_eq!(r.placement, Placement::default());
        // Explicit `auto` is the build default, not the base.
        let r = resolve(
            &["--device", "cpu", "--text-device", "cpu"],
            Family::Qwen3Embed,
        )
        .unwrap();
        assert!(!r.placement.text);
        assert!(!r.placement.text_attn);
    }

    #[test]
    fn stage_flags_are_family_scoped() {
        assert!(resolve(&["--vision-device", "cpu"], Family::Qwen3Embed).is_err());
        assert!(resolve(&["--text-attn-device", "cpu"], Family::Intent).is_err());
        assert!(resolve(&["--decode-device", "cpu"], Family::Emb2).is_err());
        assert!(resolve(&["--npu-dtype", "int8"], Family::Qwen3Embed).is_err());
        assert!(resolve(&["--keep-cpu-weights", "yes"], Family::Emb2).is_err());
        assert!(resolve(&["--dtype", "f32"], Family::Intent).is_err());
        assert!(resolve(&["--chunk", "128"], Family::Emb2).is_err());
        assert!(resolve(&["--embed-f16"], Family::Gemma4).is_err());
    }

    #[test]
    fn int8_requires_text_on_the_npu() {
        let err = resolve(&["--npu-dtype", "int8"], Family::Emb2).unwrap_err();
        assert!(format!("{err:#}").contains("--text-device npu"));
        let err = resolve(&["--int8-group", "64"], Family::Emb2).unwrap_err();
        assert!(format!("{err:#}").contains("--npu-dtype int8"));
    }

    #[test]
    fn keep_cpu_weights_tracks_the_decode_stage() {
        // The host build has no NPU, so decode cannot be placed there; keeping
        // the copies is the only valid choice.
        let r = resolve(&[], Family::Intent).unwrap();
        assert!(r.keep_cpu_weights);
        let err = resolve(&["--keep-cpu-weights", "no"], Family::Intent).unwrap_err();
        assert!(format!("{err:#}").contains("decode stage runs on the CPU"));
    }
}
