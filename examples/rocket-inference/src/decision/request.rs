//! The `/v1/systemone` request envelope and the CLI's case inputs.
//!
//! ```json
//! {"state": <any JSON>, "questions": {"<id>": {"type": "...", ...}, ...},
//!  "images": [...], "videos": [...]}
//! ```
//!
//! The envelopes are parsed from a [`serde_json::Value`] rather than with a
//! `Deserialize` derive: the reference HTTP server checks the fields itself, so
//! the client-facing messages (`unknown field`, `images takes a list of base64
//! data URLs`, ...) and the accepted field set are part of the contract. A
//! question that cannot be answered is reported inside the response's `answers`
//! (see [`crate::decision::model`]); only envelope problems fail the request.
//!
//! Media: `images` is a list of strings shared by every question of the request.
//! How those strings are read depends on the transport ([`MediaMode`]): the
//! server takes `data:image/{png,jpeg,webp};base64,...` URLs only (the reference
//! `d3_server.py` contract, see [`decode_data_url`]), the CLI takes local paths so
//! the parity fixtures can point at the checkpoint's own assets. `videos` is
//! phase 3: a non-empty list is refused up front.

use anyhow::Result;
use image::RgbImage;
use serde_json::{Map, Value};

use crate::gemma::media::load_image;

/// The fields a request may carry (the reference's `ACCEPTED_FIELDS`).
pub const ACCEPTED_FIELDS: [&str; 5] = ["state", "questions", "images", "videos", "model"];

/// `d3_runtime.MAX_IMAGE_BYTES`: the decoded size limit of a served image.
pub const MAX_IMAGE_BYTES: usize = 8_000_000;
/// `d3_runtime.MAX_IMAGE_SOURCE_PIXELS`: the source-pixel limit of a served image.
pub const MAX_IMAGE_SOURCE_PIXELS: u64 = 16_000_000;
/// `d3_runtime.IMAGE_FORMATS`: what a served image may be.
pub const IMAGE_FORMATS: [&str; 3] = ["PNG", "JPEG", "WEBP"];
/// `d3_runtime.DATA_URL_IMAGE_TYPES`: the data-URL subtypes the server accepts.
pub const DATA_URL_IMAGE_TYPES: [&str; 4] = ["png", "jpeg", "jpg", "webp"];

/// How a request's `images` entries are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaMode {
    /// Local filesystem paths (the CLI, i.e. the parity fixtures).
    Paths,
    /// `data:image/<type>;base64,<payload>` URLs only (the HTTP server).
    DataUrls,
}

impl MediaMode {
    /// Decode one `images` entry to RGB8, or the message a client gets.
    pub fn decode(self, value: &str, index: usize) -> Result<RgbImage, String> {
        let decoded = match self {
            Self::Paths => load_image(std::path::Path::new(value))
                .map_err(|e| format!("cannot read the image {value} ({e:#})")),
            Self::DataUrls => decode_data_url(value),
        };
        decoded.map_err(|message| format!("images[{index}]: {message}"))
    }
}

/// One System One request.
#[derive(Debug, Clone)]
pub struct DecisionRequest {
    /// The state the questions are about (`null` when absent).
    pub state: Value,
    /// Question id -> question object, request order.
    pub questions: Map<String, Value>,
    /// Image inputs shared by every question, as given (paths or data URLs,
    /// depending on the transport).
    pub images: Vec<String>,
    /// Video inputs: stored for phase 3, refused by [`DecisionRequest::parse`].
    pub videos: Vec<String>,
}

/// One case of a `--requests` array: a request plus the name reported back.
#[derive(Debug, Clone)]
pub struct CaseInput {
    pub name: Option<String>,
    pub request: DecisionRequest,
}

impl DecisionRequest {
    /// Parse one request object (`POST /v1/systemone`, `decide --request`).
    ///
    /// Unknown fields are refused, exactly like the reference server; a JSON
    /// value that is not an object is refused too.
    pub fn from_value(value: Value) -> Result<Self, String> {
        Self::parse(value, true)
    }

    fn parse(value: Value, strict: bool) -> Result<Self, String> {
        let Value::Object(mut object) = value else {
            return Err(
                "expected a JSON object such as {\"questions\": {...}, \"state\": ...}".to_string(),
            );
        };
        if strict {
            let mut unknown: Vec<String> = object
                .keys()
                .filter(|key| !ACCEPTED_FIELDS.contains(&key.as_str()))
                .cloned()
                .collect();
            if !unknown.is_empty() {
                unknown.sort();
                let named = unknown
                    .iter()
                    .map(|key| format!("'{key}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "{named} not understood; a request may carry {}",
                    ACCEPTED_FIELDS.join(", ")
                ));
            }
        }
        let state = object.remove("state").unwrap_or(Value::Null);
        let questions = object.remove("questions").unwrap_or(Value::Null);
        let images = object.remove("images").unwrap_or(Value::Null);
        let videos = object.remove("videos").unwrap_or(Value::Null);
        // Accepted and ignored: the answer always names the served model.
        let _model = object.remove("model");

        let Value::Object(questions) = questions else {
            return Err(
                "questions is required: an object of one or more question IDs, each to a question"
                    .to_string(),
            );
        };
        if questions.is_empty() {
            return Err(
                "questions is required: an object of one or more question IDs, each to a question"
                    .to_string(),
            );
        }
        if questions.keys().any(|key| key.is_empty()) {
            return Err("question IDs must be nonempty strings".to_string());
        }
        let images = media_list("images", &images)?;
        let videos = media_list("videos", &videos)?;
        if !videos.is_empty() {
            return Err("videos are not supported yet".to_string());
        }
        Ok(Self {
            state,
            questions,
            images,
            videos,
        })
    }

    /// Decode this request's `images` for the transport in `mode`, in list order.
    ///
    /// The message is the reference's (`images[0]: <reason>`), which the server
    /// returns as a 422 `detail` string.
    pub fn decode_images(&self, mode: MediaMode) -> Result<Vec<RgbImage>, String> {
        self.images
            .iter()
            .enumerate()
            .map(|(index, value)| mode.decode(value, index))
            .collect()
    }
}

/// `images`/`videos` must be absent, `null` or a list of nonempty strings.
fn media_list(name: &str, value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| match item {
                Value::String(text) if !text.is_empty() => Ok(text.clone()),
                Value::String(_) => Err(format!("{name}[{index}] must be a nonempty string")),
                _ => Err(format!("{name} takes a list of base64 data URLs")),
            })
            .collect(),
        _ => Err(format!("{name} takes a list of base64 data URLs")),
    }
}

/// `d3_runtime._split_data_url`: `data:<media>/<rest>;base64,<text>` ->
/// `(<rest>, <text>)`; the head is matched lower-cased and stripped.
fn split_data_url<'a>(value: &'a str, media: &str) -> Result<(String, &'a str), String> {
    let (head, text) = value
        .split_once(',')
        .ok_or_else(|| format!("expected a base64 data URL: data:{media}/<type>;base64,<data>"))?;
    let head = head.trim().to_lowercase();
    let opening = format!("data:{media}/");
    let closing = ";base64";
    if !head.starts_with(&opening) || !head.ends_with(closing) {
        return Err(format!(
            "expected a base64 data URL: data:{media}/<type>;base64,<data>"
        ));
    }
    Ok((
        head[opening.len()..head.len() - closing.len()].to_string(),
        text,
    ))
}

/// One served image: the strict, server-side reading of an `images` entry
/// (`d3_runtime.load_image(..., strict=True)`), i.e. data URLs only, PNG/JPEG/
/// WebP, at most [`MAX_IMAGE_BYTES`] decoded bytes and [`MAX_IMAGE_SOURCE_PIXELS`]
/// pixels, and a payload that decodes.
///
/// The pixels come from the same decoder the file path uses
/// ([`crate::gemma::media::load_image`]: mozjpeg/libjpeg-turbo for JPEGs, PIL's
/// 16-bit PNG rules), through a temporary file named after the data URL's subtype.
pub fn decode_data_url(value: &str) -> Result<RgbImage, String> {
    if !value.starts_with("data:") {
        return Err(
            "strict mode: images must be given as a data: URL string, not a file path, \
             web address or PIL image"
                .to_string(),
        );
    }
    let (subtype, text) = split_data_url(value, "image")?;
    if !DATA_URL_IMAGE_TYPES.contains(&subtype.as_str()) {
        return Err(format!(
            "image data URLs take the types {}, got {subtype:?}",
            DATA_URL_IMAGE_TYPES.join(", ")
        ));
    }
    let bytes = decode_base64(text, MAX_IMAGE_BYTES)?;
    let (format, width, height) = probe_image(&bytes)?;
    if !IMAGE_FORMATS.contains(&format.as_str()) {
        return Err(format!("the image is {format}; send PNG, JPEG or WebP"));
    }
    let pixels = width as u64 * height as u64;
    if pixels > MAX_IMAGE_SOURCE_PIXELS {
        return Err(format!(
            "the image has {} pixels; at most {} are accepted",
            thousands(pixels),
            thousands(MAX_IMAGE_SOURCE_PIXELS)
        ));
    }
    // The decoders pick their backend by extension, so the temp file carries the
    // data URL's own subtype (`jpg` normalizes to the same JPEG path).
    let path =
        write_temp(&bytes, &subtype).map_err(|e| format!("the image could not be read ({e})"))?;
    let image = load_image(&path).map_err(|e| format!("the image could not be decoded ({e:#})"));
    let _ = std::fs::remove_file(&path);
    image
}

/// `d3_runtime._decode_base64`: validated base64 with the reference's limits.
fn decode_base64(text: &str, limit: usize) -> Result<Vec<u8>, String> {
    use base64::Engine;
    // 4 bytes of base64 encode 3 bytes; the reference refuses an over-long text
    // before reading it.
    let allowed = 4 * limit.div_ceil(3);
    if text.len() > allowed {
        return Err(format!(
            "the encoded data exceeds the {} byte limit",
            thousands(limit as u64)
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(text)
        .map_err(|_| "the data URL does not hold valid base64".to_string())?;
    if bytes.len() > limit {
        return Err(format!(
            "the decoded data has {} bytes, over the {} byte limit",
            thousands(bytes.len() as u64),
            thousands(limit as u64)
        ));
    }
    Ok(bytes)
}

/// `{value:,}` — the reference's digit grouping in its size messages.
fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// What `PIL.Image.open(...).format`/`.size` report for an encoded image: the
/// container format and the pixel dimensions, from the header alone.
fn probe_image(bytes: &[u8]) -> Result<(String, u32, u32), String> {
    use std::io::Cursor;
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("the image could not be identified ({e})"))?;
    let name = match reader.format() {
        Some(image::ImageFormat::Png) => "PNG".to_string(),
        Some(image::ImageFormat::Jpeg) => "JPEG".to_string(),
        Some(image::ImageFormat::WebP) => "WEBP".to_string(),
        // Anything else is refused by name before its size is read (the
        // reference checks the format first too).
        Some(other) => return Ok((format!("{other:?}").to_uppercase(), 0, 0)),
        None => return Err("the image could not be identified (unknown format)".to_string()),
    };
    let (width, height) = reader
        .into_dimensions()
        .map_err(|e| format!("the image could not be identified ({e})"))?;
    Ok((name, width, height))
}

/// Write upload bytes to the app's temp directory (the decoder reads a path).
fn write_temp(bytes: &[u8], ext: &str) -> Result<std::path::PathBuf, String> {
    let dir = std::env::temp_dir().join("rocket-inference");
    std::fs::create_dir_all(&dir).map_err(|e| format!("temp dir: {e}"))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("decision-{}-{nanos}.{ext}", std::process::id()));
    std::fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

impl CaseInput {
    /// Parse one `--requests` array entry: `name` is read (and stripped) here,
    /// the rest is the request; extra keys are ignored.
    pub fn from_value(value: Value) -> Result<Self, String> {
        let Value::Object(mut object) = value else {
            return Err("a case must be a JSON object".to_string());
        };
        let name = match object.remove("name") {
            None | Some(Value::Null) => None,
            Some(Value::String(name)) => Some(name),
            Some(_) => return Err("a case name must be a string".to_string()),
        };
        let request = DecisionRequest::parse(Value::Object(object), false)?;
        Ok(Self { name, request })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: serde_json::Value) -> Result<DecisionRequest, String> {
        DecisionRequest::from_value(value)
    }

    #[test]
    fn a_request_keeps_question_order_and_defaults_the_state() {
        let req = parse(json!({
            "questions": {"z": {"type": "noul"}, "a": {"type": "noul"}},
        }))
        .unwrap();
        assert!(req.state.is_null());
        let ids: Vec<&String> = req.questions.keys().collect();
        assert_eq!(ids, ["z", "a"], "serde_json keeps insertion order");
    }

    #[test]
    fn unknown_fields_are_refused_like_the_reference_server() {
        let err =
            parse(json!({"questions": {"q": {"type": "noul"}}, "temperature": 1})).unwrap_err();
        assert_eq!(
            err,
            "'temperature' not understood; a request may carry state, questions, images, videos, model"
        );
        let err = parse(json!({"questions": {"q": {}}, "b": 1, "a": 2})).unwrap_err();
        assert_eq!(
            err,
            "'a', 'b' not understood; a request may carry state, questions, images, videos, model"
        );
        // `model` is accepted and ignored.
        let req = parse(json!({"questions": {"q": {"type": "noul"}}, "model": "ignored"})).unwrap();
        assert_eq!(req.questions.len(), 1);
    }

    #[test]
    fn media_is_parsed_and_videos_are_refused() {
        let req = parse(json!({
            "questions": {"q": {"type": "noul"}},
            "images": ["a.png", "data:image/png;base64,AA=="],
        }))
        .unwrap();
        assert_eq!(req.images, ["a.png", "data:image/png;base64,AA=="]);
        assert!(req.videos.is_empty());

        let err = parse(json!({
            "questions": {"q": {"type": "noul"}},
            "videos": ["data:video/mp4;base64,AA=="],
        }))
        .unwrap_err();
        assert_eq!(err, "videos are not supported yet");

        // A media list of the wrong shape or with a non-string entry.
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "images": 5})).unwrap_err();
        assert_eq!(err, "images takes a list of base64 data URLs");
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "videos": {}})).unwrap_err();
        assert_eq!(err, "videos takes a list of base64 data URLs");
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "images": [7]})).unwrap_err();
        assert_eq!(err, "images takes a list of base64 data URLs");
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "images": [""]})).unwrap_err();
        assert_eq!(err, "images[0] must be a nonempty string");
        // An empty media list is the same as none.
        let req = parse(json!({"questions": {"q": {"type": "noul"}}, "images": [], "videos": []}))
            .unwrap();
        assert!(req.images.is_empty() && req.videos.is_empty());
    }

    /// The strict server reading of an `images` entry: data URLs only, with the
    /// reference's messages.
    #[test]
    fn data_urls_are_checked_like_the_reference_server() {
        assert_eq!(
            decode_data_url("data/cat.jpeg").unwrap_err(),
            "strict mode: images must be given as a data: URL string, not a file path, \
             web address or PIL image"
        );
        assert_eq!(
            decode_data_url("https://example.com/cat.jpeg").unwrap_err(),
            "strict mode: images must be given as a data: URL string, not a file path, \
             web address or PIL image"
        );
        // A well-formed URL of an unsupported subtype is refused by name.
        let err = decode_data_url("data:image/gif;base64,QUJD").unwrap_err();
        assert_eq!(
            err,
            "image data URLs take the types png, jpeg, jpg, webp, got \"gif\""
        );
        // Not a base64 data URL at all.
        assert_eq!(
            decode_data_url("data:image/png,QUJD").unwrap_err(),
            "expected a base64 data URL: data:image/<type>;base64,<data>"
        );
        // Invalid base64.
        let err = decode_data_url("data:image/png;base64,!!").unwrap_err();
        assert_eq!(err, "the data URL does not hold valid base64");
        // An over-long payload is refused before it is decoded.
        let huge = "A".repeat(4 * MAX_IMAGE_BYTES.div_ceil(3) + 4);
        let err = decode_data_url(&format!("data:image/png;base64,{huge}")).unwrap_err();
        assert_eq!(err, "the encoded data exceeds the 8,000,000 byte limit");
        // Valid base64 that is not an image.
        let err = decode_data_url("data:image/png;base64,QUJD").unwrap_err();
        assert!(
            err.starts_with("the image could not be identified"),
            "{err}"
        );
    }

    /// A real data URL round trip: the repo's cat JPEG re-encodes to the same
    /// pixels as the file.
    #[test]
    fn a_data_url_decodes_to_the_same_pixels_as_the_path() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cat.jpeg");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipping: {} is not available", path.display());
            return;
        };
        use base64::Engine;
        let url = format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        );
        let decoded = decode_data_url(&url).unwrap();
        let expected = crate::gemma::media::load_image(&path).unwrap();
        assert_eq!(decoded.dimensions(), expected.dimensions());
        assert_eq!(decoded.as_raw(), expected.as_raw(), "same RGB8 pixels");
        // ... and the same image through the path mode.
        let via_path = MediaMode::Paths.decode(path.to_str().unwrap(), 0).unwrap();
        assert_eq!(via_path.as_raw(), expected.as_raw());
        // The error carries the reference's `images[i]: ` prefix.
        let err = MediaMode::Paths
            .decode("/nonexistent/none.png", 2)
            .unwrap_err();
        assert!(err.starts_with("images[2]: "), "{err}");
    }

    #[test]
    fn bad_envelopes_are_refused() {
        const MISSING: &str =
            "questions is required: an object of one or more question IDs, each to a question";
        assert_eq!(
            parse(json!([])).unwrap_err(),
            "expected a JSON object such as {\"questions\": {...}, \"state\": ...}"
        );
        for bad in [
            json!({}),
            json!({"questions": {}}),
            json!({"questions": []}),
            json!({"questions": "q"}),
            json!({"questions": null}),
        ] {
            assert_eq!(
                parse(bad.clone()).unwrap_err(),
                MISSING,
                "message for {bad}"
            );
        }
        assert_eq!(
            parse(json!({"questions": {"": {"type": "noul"}}})).unwrap_err(),
            "question IDs must be nonempty strings"
        );
    }

    #[test]
    fn a_case_carries_its_name_and_request() {
        let case = CaseInput::from_value(json!({"name": "one", "state": "s", "questions": {}}));
        assert!(case.is_err(), "an empty questions object is refused");
        let case =
            CaseInput::from_value(json!({"name": "one", "state": "s", "questions": {"q": {}}}))
                .unwrap();
        assert_eq!(case.name.as_deref(), Some("one"));
        assert_eq!(case.request.state, json!("s"));
        // Extra case-level keys are ignored (the CLI's own format).
        let case = CaseInput::from_value(
            json!({"name": null, "state": 1, "questions": {"q": {}}, "elapsed_s": 1.0}),
        )
        .unwrap();
        assert!(case.name.is_none());
        assert!(CaseInput::from_value(json!(["not", "an", "object"])).is_err());
    }

    #[test]
    fn size_messages_group_the_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(8_000_000), "8,000,000");
        assert_eq!(thousands(16_000_000), "16,000,000");
    }
}
