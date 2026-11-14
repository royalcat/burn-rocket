//! `encode_record` of the Clef-Omni release (`tools/clef_joint_schema_model.py`): the
//! text-only prompt layout and the question/option token spans the head reads.
//!
//! Layout (token ids are the concatenation of separately tokenized pieces, exactly like
//! the Python `_tokens(tokenizer, text)` = `tokenizer(text, add_special_tokens=False)`):
//!
//! ```text
//! prefix  "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
//! state   render(state)  (cut to max_state_tokens, then to what max_length leaves)
//! schema  "\n\nSCHEMA FIELDS:\n" then per question
//!           "\nFIELD n\nID: id\nTYPE: t\nINSTRUCTION: " | instructions | "\nALLOWED OPTIONS:\n"
//!           per option "OPTION k: " | {"description":..,"option_id":..} | "\n"
//!           "END FIELD\n"
//! suffix  "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
//! ```
//!
//! Special tokens inside the text pieces (`<|im_start|>` ...) tokenize to their special
//! ids, as in the Python tokenizer (which is the same Rust `tokenizers` core).

use super::media::{MediaLayout, build_media_ids};
use super::request::{ClefRequest, Invalid, question_type_index};
use super::vision::{PreparedImage, VisionConfig, prepare_value};
use anyhow::{Context, Result};
use serde_json::{Number, Value};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
Each answer must be exactly one of that field's allowed options.";
/// The release's `DEFAULT_MAX_LENGTH`.
pub const DEFAULT_MAX_LENGTH: usize = 64000;

/// Python `float.__repr__` (shortest round-trip digits, exponent form for
/// `decpt > 16 || decpt < -3`).
fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let sign = if f.is_sign_negative() { "-" } else { "" };
    if f == 0.0 {
        return format!("{sign}0.0");
    }
    // Shortest round-trip digits: Rust's `{:e}` ("1.2345e-7", "1e22").
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;
    let body = if !(-3..=16).contains(&decpt) {
        let mut m = String::from(&digits[..1]);
        if digits.len() > 1 {
            m.push('.');
            m.push_str(&digits[1..]);
        }
        let sign_e = if exp < 0 { '-' } else { '+' };
        format!("{m}e{sign_e}{:02}", exp.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(decpt as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    };
    format!("{sign}{body}")
}

/// A JSON number as Python's `json.loads` + `json.dumps` would print it: integer
/// literals stay integers (arbitrary size), everything else goes through a float.
fn render_number(n: &Number) -> String {
    let s = n.to_string();
    if s.bytes().all(|b| b.is_ascii_digit() || b == b'-') {
        return if s == "-0" { "0".into() } else { s };
    }
    match s.parse::<f64>() {
        Ok(f) => py_float_repr(f),
        Err(_) => s,
    }
}

fn dump(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&render_number(n)),
        // `ensure_ascii=False`: serde_json escapes the same set as Python (`"`, `\`,
        // `\b\f\n\r\t`, other C0 controls as `\u00XX`) and leaves everything else.
        Value::String(s) => out.push_str(&serde_json::to_string(s).expect("string")),
        Value::Array(a) => {
            out.push('[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                dump(v, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            // `sort_keys=True`: code-point order == UTF-8 byte order.
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).expect("key"));
                out.push(':');
                dump(&o[k], out);
            }
            out.push('}');
        }
    }
}

/// `render()`: a string verbatim, anything else compact sorted-key JSON.
pub fn render(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => {
            let mut out = String::new();
            dump(other, &mut out);
            out
        }
    }
}

/// `question_options`: `(option_id, description)` in the order the model scores them;
/// a JSON `null` description is Python's `None` (omitted from the option's JSON).
pub fn question_options(question: &Value) -> Result<Vec<(String, Option<Value>)>> {
    let ty = question["type"].as_str().context("question has no type")?;
    let some = |v: &Value| (!v.is_null()).then(|| v.clone());
    match ty {
        "noul" => {
            let user = question["criteria"].as_object();
            let describe = |key: &str, default: &str| {
                let v = user.and_then(|u| u.get(key));
                match v {
                    Some(v) => some(v),
                    None => Some(Value::String(default.into())),
                }
            };
            Ok(vec![
                (
                    "true".into(),
                    describe("true", "The proposition is true or the answer is yes."),
                ),
                (
                    "false".into(),
                    describe("false", "The proposition is false or the answer is no."),
                ),
            ])
        }
        "choice" => {
            let crit = question["criteria"]
                .as_object()
                .context("choice criteria must be an object")?;
            let mut items: Vec<(String, Option<Value>)> =
                crit.iter().map(|(k, v)| (k.clone(), some(v))).collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(items)
        }
        "score" => match &question["criteria"] {
            Value::Array(a) => Ok(a
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), some(v)))
                .collect()),
            // `enumerate(dict)` walks the keys.
            Value::Object(o) => Ok(o
                .keys()
                .enumerate()
                .map(|(i, k)| (i.to_string(), Some(Value::String(k.clone()))))
                .collect()),
            _ => anyhow::bail!("score criteria must be a list"),
        },
        other => anyhow::bail!("type must be noul, choice, or score (got {other})"),
    }
}

/// Token layout of one question inside the encoded record (half-open ranges).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedQuestion {
    pub id: String,
    pub question_type: usize,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}

impl EncodedQuestion {
    pub fn spans(&self) -> super::head::QuestionSpans {
        super::head::QuestionSpans {
            question_type: self.question_type,
            question_span: self.question_span,
            option_spans: self.option_spans.clone(),
        }
    }
}

/// The prepared media of an encoded record (images only for now): the layout the
/// placeholder block and the mRoPE positions derive from, and the tower inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordMedia {
    pub layout: MediaLayout,
    pub images: Vec<PreparedImage>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedRecord {
    pub input_ids: Vec<u32>,
    pub questions: Vec<EncodedQuestion>,
    /// Decoded + preprocessed media (`None` for a text-only record). Shared so a record
    /// clones cheaply.
    pub media: Option<Arc<RecordMedia>>,
}

fn tokens(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

/// The record prefix tokens (system turn + `STATE:\n`), before any media block.
pub fn prefix_tokens(tok: &Tokenizer) -> Result<Vec<u32>> {
    tokens(
        tok,
        &format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"),
    )
}

/// Token id of `"\n"` (the separator after every media placeholder).
pub fn newline_token(tok: &Tokenizer) -> Result<u32> {
    let ids = tokens(tok, "\n")?;
    anyhow::ensure!(ids.len() == 1, "`\\n` is not a single token");
    Ok(ids[0])
}

/// `encode_record` for a text-only request. `max_length` caps the whole record (the
/// state is cut to what the schema leaves; a schema that does not fit is [`Invalid`]);
/// `max_state_tokens` additionally caps the state before that.
pub fn encode_record(
    tok: &Tokenizer,
    request: &ClefRequest,
    max_length: usize,
    max_state_tokens: Option<usize>,
) -> Result<EncodedRecord> {
    encode_record_with_media(tok, request, max_length, max_state_tokens, &[])
}

/// [`encode_record`] with the record's media block (`media_ids`, see
/// [`super::media::build_media_ids`]) appended to the prefix, as the release does
/// (`prefix_ids + media_ids`): it counts toward the fixed length, only the state is cut.
/// The media block occupies `prefix_len .. prefix_len + media_ids.len()` of the ids
/// (35 tokens for the shipped tokenizer, see [`prefix_tokens`]).
pub fn encode_record_with_media(
    tok: &Tokenizer,
    request: &ClefRequest,
    max_length: usize,
    max_state_tokens: Option<usize>,
    media_ids: &[u32],
) -> Result<EncodedRecord> {
    let mut schema_ids = tokens(tok, "\n\nSCHEMA FIELDS:\n")?;
    let mut questions: Vec<EncodedQuestion> = Vec::new();
    for (index, (id, question)) in request.questions.iter().enumerate() {
        let ty = question["type"].as_str().context("question has no type")?;
        schema_ids.extend(tokens(
            tok,
            &format!("\nFIELD {}\nID: {id}\nTYPE: {ty}\nINSTRUCTION: ", index + 1),
        )?);
        let question_start = schema_ids.len();
        let instructions = match question.get("instructions") {
            None | Some(Value::Null) => Value::String(id.clone()),
            Some(Value::String(s)) if s.is_empty() => Value::String(id.clone()),
            Some(v) => v.clone(),
        };
        schema_ids.extend(tokens(tok, &render(&instructions))?);
        let question_end = schema_ids.len();
        schema_ids.extend(tokens(tok, "\nALLOWED OPTIONS:\n")?);

        let mut option_spans = Vec::new();
        let mut option_ids = Vec::new();
        for (option_index, (option_id, description)) in
            question_options(question)?.into_iter().enumerate()
        {
            schema_ids.extend(tokens(tok, &format!("OPTION {}: ", option_index + 1))?);
            let option_start = schema_ids.len();
            let mut semantics = serde_json::Map::new();
            semantics.insert("option_id".into(), Value::String(option_id.clone()));
            if let Some(d) = description {
                semantics.insert("description".into(), d);
            }
            schema_ids.extend(tokens(tok, &render(&Value::Object(semantics)))?);
            option_spans.push((option_start, schema_ids.len()));
            option_ids.push(option_id);
            schema_ids.extend(tokens(tok, "\n")?);
        }
        schema_ids.extend(tokens(tok, "END FIELD\n")?);
        questions.push(EncodedQuestion {
            id: id.clone(),
            question_type: question_type_index(ty).context("question type")?,
            question_span: (question_start, question_end),
            option_spans,
            option_ids,
        });
    }

    let mut prefix_ids = prefix_tokens(tok)?;
    prefix_ids.extend_from_slice(media_ids);
    let suffix_ids = tokens(
        tok,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let mut state_ids = tokens(tok, &render(&request.state))?;
    if let Some(cap) = max_state_tokens {
        state_ids.truncate(cap);
    }
    let fixed = prefix_ids.len() + schema_ids.len() + suffix_ids.len();
    if fixed > max_length {
        return Err(Invalid(format!(
            "schema requires {fixed} tokens before state; maximum is {max_length}"
        ))
        .into());
    }
    state_ids.truncate(max_length - fixed);
    let offset = prefix_ids.len() + state_ids.len();
    for q in &mut questions {
        q.question_span = (q.question_span.0 + offset, q.question_span.1 + offset);
        for s in &mut q.option_spans {
            *s = (s.0 + offset, s.1 + offset);
        }
    }
    let mut input_ids = prefix_ids;
    input_ids.extend(state_ids);
    input_ids.extend(schema_ids);
    input_ids.extend(suffix_ids);
    if questions.is_empty() {
        return Err(Invalid("record produced no model input or questions".into()).into());
    }
    Ok(EncodedRecord {
        input_ids,
        questions,
        media: None,
    })
}

/// [`encode_record`] for any request this build serves: text-only, or with `images`.
///
/// Images are fetched/decoded/preprocessed here (client errors are [`Invalid`]), their grids
/// give the placeholder block (`<|vision_start|> <|image_pad|> x n <|vision_end|>` + `\n` per
/// image, in request order), and the block counts toward `max_length` like the release's
/// `fixed_length`. The prepared pixel rows travel with the record to the thinker pass.
/// Audio/video requests are refused ([`ClefRequest::media_refusal`]).
pub fn encode_request(
    tok: &Tokenizer,
    request: &ClefRequest,
    max_length: usize,
) -> Result<EncodedRecord> {
    if let Some(refusal) = request.media_refusal() {
        return Err(refusal.into());
    }
    if request.images.is_empty() {
        return encode_record(tok, request, max_length, None);
    }
    let cfg = VisionConfig::shipped();
    let mut images = Vec::with_capacity(request.images.len());
    let mut media_tokens = 0usize;
    for (i, value) in request.images.iter().enumerate() {
        let img = prepare_value(value, i, &cfg)?;
        // + <|vision_start|>, <|vision_end|> and the "\n" after the placeholder
        media_tokens += img.tokens() + 3;
        if media_tokens > max_length {
            return Err(Invalid(format!(
                "the images need over {max_length} tokens before the state (images[{i}] is the \
                 one that crosses the limit)"
            ))
            .into());
        }
        images.push(img);
    }
    let layout = MediaLayout {
        image_grids: images.iter().map(|i| i.grid).collect(),
        ..Default::default()
    };
    let media_ids = build_media_ids(&layout, newline_token(tok)?)?;
    let mut enc = encode_record_with_media(tok, request, max_length, None, &media_ids)?;
    enc.media = Some(Arc::new(RecordMedia { layout, images }));
    Ok(enc)
}

#[cfg(test)]
mod tests;
