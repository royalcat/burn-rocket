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
//! Media: `images` and `videos` are lists shared by every question of the
//! request. How those entries are read depends on the transport ([`MediaMode`]):
//! the server takes `data:image/{png,jpeg,webp};base64,...` and
//! `data:video/{mp4,webm,quicktime,x-matroska};base64,...` URLs only (the
//! reference `d3_server.py` contract, see [`decode_data_url`] and
//! [`decode_video_data_url`]), the CLI takes local paths so the parity fixtures
//! can point at the checkpoint's own assets. Image bytes decode by content (magic
//! number), like `PIL.Image.open`, so a JPEG declared as `image/png` decodes as a
//! JPEG; a video is written to a temporary file and decoded by `ffmpeg`, exactly
//! like the same clip given by path.

use anyhow::Result;
use image::RgbImage;
use serde_json::{Map, Value};

use crate::decision::vision::video::{self, DecodedVideo};
use crate::gemma::media::{decode_image, load_image};

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
/// `d3_runtime.load_image(..., strict=True)`'s refusal of anything that is not a
/// `data:` URL string (the server never reads files or fetches URLs for a client).
pub const STRICT_IMAGE_MESSAGE: &str = "strict mode: images must be given as a data: URL string, not a file path, web address or PIL image";
/// The in-process loader's list of accepted shapes (a non-string or empty entry).
pub const IMAGE_SHAPES_MESSAGE: &str =
    "an image is a PIL image, a local path, an http(s) URL or a data:image/...;base64 URL";
/// `d3_runtime.load_video`'s list of accepted shapes (a non-string or empty
/// entry).
pub const VIDEO_SHAPES_MESSAGE: &str =
    "a video is a path, an http(s) URL, a data:video/...;base64 URL or a frame array";
/// `d3_runtime.load_video(..., strict=True)`'s refusal of anything that is not a
/// `data:` URL string: the server never reads files or fetches URLs for a client.
pub const STRICT_VIDEO_MESSAGE: &str =
    "cannot load this video with strict=True; pass the clip inline as a data:video/... URL";

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
    ///
    /// An entry that is not a nonempty string gets the mode's own wording, which
    /// is the reference's: the strict server loader refuses anything that is not
    /// a `data:` URL string, the in-process path loader names the four accepted
    /// shapes.
    pub fn decode(self, value: &Value, index: usize) -> Result<RgbImage, String> {
        let decoded = match value.as_str().filter(|text| !text.is_empty()) {
            None => Err(match self {
                Self::DataUrls => STRICT_IMAGE_MESSAGE.to_string(),
                Self::Paths => IMAGE_SHAPES_MESSAGE.to_string(),
            }),
            Some(text) => match self {
                Self::Paths => load_image(std::path::Path::new(text))
                    .map_err(|e| format!("cannot read the image {text} ({e:#})")),
                Self::DataUrls => decode_data_url(text),
            },
        };
        decoded.map_err(|message| format!("images[{index}]: {message}"))
    }

    /// Decode one `videos` entry, or the message a client gets
    /// (`d3_runtime.load_video`, prefixed `videos[i]: `).
    ///
    /// `Paths` reads a local file (`no video file at <path>` when it is absent);
    /// `DataUrls` takes the reference's strict contract (data URLs of MP4, WebM,
    /// QuickTime or Matroska at most 32,000,000 decoded bytes, 8,294,400 pixels per
    /// frame and 300 seconds) and never touches the filesystem beyond a temporary
    /// file the decoder reads and removes.
    pub fn decode_video(self, value: &Value, index: usize) -> Result<DecodedVideo, String> {
        let decoded = match value.as_str().filter(|text| !text.is_empty()) {
            None => Err(match self {
                Self::DataUrls => STRICT_VIDEO_MESSAGE.to_string(),
                Self::Paths => VIDEO_SHAPES_MESSAGE.to_string(),
            }),
            Some(text) => match self {
                Self::Paths => {
                    let path = std::path::Path::new(text);
                    if !path.is_file() {
                        Err(format!("no video file at {text}"))
                    } else {
                        video::decode(path, false)
                    }
                }
                Self::DataUrls => decode_video_data_url(text),
            },
        };
        decoded.map_err(|message| format!("videos[{index}]: {message}"))
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
    /// depending on the transport; kept as JSON so a non-string entry gets the
    /// loader's own message rather than an envelope error).
    pub images: Vec<Value>,
    /// Video inputs shared by every question, as given (paths or data URLs, like
    /// [`DecisionRequest::images`]).
    pub videos: Vec<Value>,
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

    /// Decode this request's `videos` for the transport in `mode`, in list order.
    ///
    /// The message is the reference's (`videos[0]: <reason>`), which the server
    /// returns as a 422 `detail` string.
    pub fn decode_videos(&self, mode: MediaMode) -> Result<Vec<DecodedVideo>, String> {
        self.videos
            .iter()
            .enumerate()
            .map(|(index, value)| mode.decode_video(value, index))
            .collect()
    }
}

/// `images`/`videos` must be absent, `null` or a list; the entries themselves are
/// checked by the loader, which knows the transport
/// (`d3_server.py` checks the list shape here and the items in `load_images`).
fn media_list(name: &str, value: &Value) -> Result<Vec<Value>, String> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => Ok(items.clone()),
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
/// ([`crate::gemma::media::decode_image`]: mozjpeg/libjpeg-turbo for JPEGs, PIL's
/// 16-bit PNG rules), which picks the decoder by the payload's magic number — the
/// declared subtype is only a whitelist, exactly like the reference, so a JPEG
/// declared `image/png` still decodes.
pub fn decode_data_url(value: &str) -> Result<RgbImage, String> {
    if !value.starts_with("data:") {
        return Err(STRICT_IMAGE_MESSAGE.to_string());
    }
    let (subtype, text) = split_data_url(value, "image")?;
    if !DATA_URL_IMAGE_TYPES.contains(&subtype.as_str()) {
        // `{subtype!r}` in the reference, i.e. a Python repr with single quotes.
        return Err(format!(
            "image data URLs take the types {}, got '{subtype}'",
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
    decode_image(&bytes).map_err(|e| format!("the image could not be decoded ({e:#})"))
}

/// One served video: the strict, server-side reading of a `videos` entry
/// (`d3_runtime.load_video(..., strict=True)`), i.e. a `data:video/<type>;base64,`
/// URL of MP4, WebM, QuickTime or Matroska, at most
/// [`video::MAX_BYTES`](crate::decision::vision::video::MAX_BYTES) decoded bytes,
/// whose frames are at most 8,294,400 pixels and whose clip is at most 300
/// seconds.
///
/// The reference decodes encoded bytes from a temporary file so they decode
/// exactly like the same clip given by path; this writes one too and removes it
/// after the frames are read.
pub fn decode_video_data_url(value: &str) -> Result<DecodedVideo, String> {
    if !value.starts_with("data:") {
        return Err(STRICT_VIDEO_MESSAGE.to_string());
    }
    let (rest, text) = split_data_url(value, "video")?;
    let subtype = rest.split(';').next().unwrap_or("");
    if !video::FORMATS.contains(&subtype) {
        return Err(format!(
            "video data URLs take the types {}, got '{subtype}'",
            video::FORMATS.join(", ")
        ));
    }
    let bytes = decode_base64(text, video::MAX_BYTES)?;
    video::decode_bytes(&bytes, video::extension_for(subtype), true)
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
    fn media_is_parsed_for_both_fields() {
        let req = parse(json!({
            "questions": {"q": {"type": "noul"}},
            "images": ["a.png", "data:image/png;base64,AA=="],
            "videos": ["clip.mp4", "data:video/mp4;base64,AA=="],
        }))
        .unwrap();
        assert_eq!(
            req.images,
            [json!("a.png"), json!("data:image/png;base64,AA==")],
            "entries are kept as given; the transport decodes them"
        );
        assert_eq!(
            req.videos,
            [json!("clip.mp4"), json!("data:video/mp4;base64,AA==")]
        );

        // Only the list shape is checked here; the entries are the loader's job
        // (so the message can name the transport's accepted shapes).
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "images": 5})).unwrap_err();
        assert_eq!(err, "images takes a list of base64 data URLs");
        let err = parse(json!({"questions": {"q": {"type": "noul"}}, "videos": {}})).unwrap_err();
        assert_eq!(err, "videos takes a list of base64 data URLs");
        assert!(parse(json!({"questions": {"q": {"type": "noul"}}, "images": [7]})).is_ok());
        // An empty media list is the same as none.
        let req = parse(json!({"questions": {"q": {"type": "noul"}}, "images": [], "videos": []}))
            .unwrap();
        assert!(req.images.is_empty() && req.videos.is_empty());
    }

    /// The strict server reading of a `videos` entry: data URLs of the four
    /// whitelisted types with the reference's messages.
    #[test]
    fn video_data_urls_are_checked_like_the_reference_server() {
        assert_eq!(
            decode_video_data_url("clip.mp4").unwrap_err(),
            "cannot load this video with strict=True; pass the clip inline as a \
             data:video/... URL"
        );
        let err = decode_video_data_url("data:video/gif;base64,QUJD").unwrap_err();
        assert_eq!(
            err,
            "video data URLs take the types mp4, webm, quicktime, x-matroska, got 'gif'"
        );
        assert_eq!(
            decode_video_data_url("data:video/mp4,QUJD").unwrap_err(),
            "expected a base64 data URL: data:video/<type>;base64,<data>"
        );
        assert_eq!(
            decode_video_data_url("data:video/mp4;base64,!!").unwrap_err(),
            "the data URL does not hold valid base64"
        );
        // The encoded-text guard comes before any decode.
        let huge = "A".repeat(4 * video::MAX_BYTES.div_ceil(3) + 4);
        let err = decode_video_data_url(&format!("data:video/mp4;base64,{huge}")).unwrap_err();
        assert_eq!(err, "the encoded data exceeds the 32,000,000 byte limit");
        // Valid base64 that is not a video.
        let err = decode_video_data_url("data:video/mp4;base64,QUJD").unwrap_err();
        assert!(err.contains("the video could not be opened"), "{err}");

        // A non-string or empty entry takes the transport's wording.
        let req = parse(json!({"questions": {"q": {"type": "noul"}}, "videos": [""]})).unwrap();
        assert_eq!(
            req.decode_videos(MediaMode::DataUrls).unwrap_err(),
            format!("videos[0]: {STRICT_VIDEO_MESSAGE}")
        );
        assert_eq!(
            req.decode_videos(MediaMode::Paths).unwrap_err(),
            format!("videos[0]: {VIDEO_SHAPES_MESSAGE}")
        );
    }

    /// A real data URL round trip through ffmpeg: the repo's test clip encodes to
    /// the same sampled frames as the file. The temporary file the decoder writes
    /// is gone afterwards.
    #[test]
    fn a_video_data_url_decodes_to_the_same_frames_as_the_path() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/test.mp4");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipping: {} is not available", path.display());
            return;
        };
        use base64::Engine;
        let url = format!(
            "data:video/mp4;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        );
        let decoded = decode_video_data_url(&url).unwrap();
        let expected = video::decode(&path, true).unwrap();
        assert_eq!(decoded.indices, expected.indices);
        assert_eq!(decoded.fps, expected.fps);
        assert_eq!(decoded.total, expected.total);
        assert_eq!(decoded.frame_count(), expected.frame_count());
        for (a, b) in decoded.frames.iter().zip(&expected.frames) {
            assert_eq!(a.as_raw(), b.as_raw(), "same RGB8 frames");
        }
        // The path mode agrees too.
        let via_path = MediaMode::Paths
            .decode_video(&json!(path.to_str().unwrap()), 0)
            .unwrap();
        assert_eq!(via_path.frames.len(), expected.frames.len());
        // A missing file names itself.
        assert_eq!(
            MediaMode::Paths
                .decode_video(&json!("/nonexistent/clip.mp4"), 2)
                .unwrap_err(),
            "videos[2]: no video file at /nonexistent/clip.mp4"
        );
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
            "image data URLs take the types png, jpeg, jpg, webp, got 'gif'"
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
        let via_path = MediaMode::Paths
            .decode(&json!(path.to_str().unwrap()), 0)
            .unwrap();
        assert_eq!(via_path.as_raw(), expected.as_raw());
        // The error carries the reference's `images[i]: ` prefix.
        let err = MediaMode::Paths
            .decode(&json!("/nonexistent/none.png"), 2)
            .unwrap_err();
        assert!(err.starts_with("images[2]: "), "{err}");
    }

    /// The reference's per-entry messages: the strict server loader for anything
    /// that is not a nonempty `data:` URL string, the in-process loader's shape
    /// list otherwise.
    #[test]
    fn an_unusable_entry_gets_the_transports_wording() {
        let empty = json!("");
        assert_eq!(
            MediaMode::DataUrls.decode(&empty, 0).unwrap_err(),
            "images[0]: strict mode: images must be given as a data: URL string, not a file \
             path, web address or PIL image"
        );
        assert_eq!(
            MediaMode::Paths.decode(&empty, 0).unwrap_err(),
            "images[0]: an image is a PIL image, a local path, an http(s) URL or a \
             data:image/...;base64 URL"
        );
        // A non-string entry takes the same wording as an empty one.
        let number = json!(7);
        assert!(
            MediaMode::DataUrls
                .decode(&number, 2)
                .unwrap_err()
                .starts_with("images[2]: strict mode:")
        );
        assert!(
            MediaMode::Paths
                .decode(&number, 2)
                .unwrap_err()
                .starts_with("images[2]: an image is a PIL image")
        );
    }

    /// The decoder is chosen by the payload, not by the declared subtype: the
    /// reference's strict loader only *whitelists* the subtype, PIL then sniffs
    /// the content. A JPEG sent as `image/png` (and a PNG sent as `image/jpeg`)
    /// decodes to the same pixels as the file.
    #[test]
    fn the_declared_subtype_does_not_select_the_decoder() {
        use base64::Engine;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let Ok(jpeg) = std::fs::read(dir.join("cat.jpeg")) else {
            eprintln!("skipping: cat.jpeg is not available");
            return;
        };
        let encoded = base64::engine::general_purpose::STANDARD.encode(&jpeg);
        let expected = crate::gemma::media::load_image(&dir.join("cat.jpeg")).unwrap();
        for subtype in ["png", "webp", "jpeg"] {
            let url = format!("data:image/{subtype};base64,{encoded}");
            let decoded = decode_data_url(&url).unwrap_or_else(|e| panic!("{subtype}: {e}"));
            assert_eq!(
                decoded.as_raw(),
                expected.as_raw(),
                "a JPEG declared {subtype} must decode as a JPEG"
            );
        }
        // A real PNG declared as a JPEG decodes too.
        let png = {
            let img = image::RgbImage::from_raw(2, 1, vec![9, 8, 7, 6, 5, 4]).unwrap();
            let mut cursor = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(img)
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
            cursor.into_inner()
        };
        let url = format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png)
        );
        let decoded = decode_data_url(&url).unwrap();
        assert_eq!(decoded.dimensions(), (2, 1));
        assert_eq!(decoded.as_raw(), &[9, 8, 7, 6, 5, 4]);
    }

    /// A truncated JPEG is refused by the data-URL path as well: the structural
    /// check runs before any decoder (the reference's PIL raises
    /// `image file is truncated`), and no temporary path leaks into the message.
    #[test]
    fn a_truncated_jpeg_data_url_is_refused() {
        use base64::Engine;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cat.jpeg");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipping: {} is not available", path.display());
            return;
        };
        let cut = &bytes[..bytes.len() - 36];
        let url = format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(cut)
        );
        let message = decode_data_url(&url).unwrap_err();
        assert!(
            message.starts_with("the image could not be decoded ("),
            "{message}"
        );
        assert!(message.contains("truncated JPEG"), "{message}");
        assert!(
            !message.contains("rocket-inference"),
            "no temp path in the message: {message}"
        );
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
