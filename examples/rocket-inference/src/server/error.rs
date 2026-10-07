//! HTTP error plumbing shared by every endpoint.
//!
//! Compute errors carry only a status and a message; the API flavor (OpenAI vs
//! Ollama JSON shapes) is applied by the handler via [`OpenAiError`] /
//! [`OllamaError`]. Panics inside a forward are contained and mapped to errors
//! (hard NPU device failures still log + exit for a clean supervisor restart).

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use crate::util::http::panic_message;

/// A status + message, independent of the API flavor.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
}

/// OpenAI-style error body: `{"error": {"message", "type"}}`.
pub struct OpenAiError(pub ApiError);

impl From<ApiError> for OpenAiError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}

#[derive(Serialize)]
struct OpenAiErrorBody {
    error: OpenAiErrorDetail,
}

#[derive(Serialize)]
struct OpenAiErrorDetail {
    message: String,
    r#type: &'static str,
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        let ApiError { status, message } = self.0;
        let body = OpenAiErrorBody {
            error: OpenAiErrorDetail {
                message,
                r#type: if status.is_server_error() {
                    "server_error"
                } else {
                    "invalid_request_error"
                },
            },
        };
        (status, Json(body)).into_response()
    }
}

/// Ollama-style error body: `{"error": "..."}`.
pub struct OllamaError(pub ApiError);

impl From<ApiError> for OllamaError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}

#[derive(Serialize)]
struct OllamaErrorBody {
    error: String,
}

impl IntoResponse for OllamaError {
    fn into_response(self) -> Response {
        let ApiError { status, message } = self.0;
        (status, Json(OllamaErrorBody { error: message })).into_response()
    }
}

/// Run a blocking compute closure on a worker thread, converting panics and
/// join failures into HTTP errors. Every forward goes through this so a panic
/// (e.g. an NPU allocation failure) becomes a response instead of poisoning the
/// model lock and killing every later request.
pub async fn blocking<T, F>(f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ApiError> + Send + 'static,
{
    match tokio::task::spawn_blocking(move || catch_compute(f)).await {
        Ok(result) => result,
        Err(e) => Err(ApiError::internal(format!("worker: {e}"))),
    }
}

pub fn catch_compute<T>(f: impl FnOnce() -> Result<T, ApiError>) -> Result<T, ApiError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(panic_to_api(payload)),
    }
}

fn panic_to_api(payload: Box<dyn Any + Send>) -> ApiError {
    #[cfg(all(feature = "npu", target_arch = "aarch64"))]
    if let Some(failure) = payload.downcast_ref::<burn_rocket::OpFailure>() {
        let rc = failure.error.rc;
        let detail = format!(
            "{} failed (rc={rc}, m={} k={} n={})",
            failure.error.op, failure.m, failure.k, failure.n
        );
        if rc == burn_rocket::ffi::ROCKET_E_NOMEM {
            return ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("NPU out of memory: {detail}; retry the request"),
            );
        }
        if rc == burn_rocket::ffi::ROCKET_E_SHAPE || rc == burn_rocket::ffi::ROCKET_E_TILING {
            return ApiError::internal(format!("NPU error: {detail}"));
        }
        eprintln!("fatal NPU failure: {detail}; exiting for a clean restart");
        std::process::exit(1);
    }
    ApiError::internal(format!("internal error: {}", panic_message(&*payload)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_compute_turns_panics_into_server_errors() {
        let err = catch_compute::<()>(|| panic!("boom")).unwrap_err();
        assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            err.message.contains("boom"),
            "message was {:?}",
            err.message
        );

        let ok = catch_compute(|| Ok::<_, ApiError>(3)).unwrap();
        assert_eq!(ok, 3);
    }

    #[test]
    fn catch_compute_keeps_handler_errors() {
        let err = catch_compute::<()>(|| Err(ApiError::bad_request("bad input"))).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "bad input");
    }
}
