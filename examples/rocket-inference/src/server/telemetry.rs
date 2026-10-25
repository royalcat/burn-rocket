//! Opt-in OpenTelemetry trace export for `serve`.
//!
//! `OTEL_SDK_ENABLED` is the master switch (off unless true). Everything else
//! comes from the standard OpenTelemetry environment variables:
//!
//! - resource: `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`
//! - exporter: `OTEL_EXPORTER_OTLP_[TRACES_]ENDPOINT`, `..._HEADERS`,
//!   `..._TIMEOUT`, `..._COMPRESSION`, `OTEL_EXPORTER_OTLP_PROTOCOL`
//! - sampler and processor: `OTEL_TRACES_SAMPLER[_ARG]`, `OTEL_BSP_*`,
//!   `OTEL_SPAN_*`
//! - propagation: `OTEL_PROPAGATORS` (`tracecontext`, `baggage`; W3C default)
//!
//! Traces go out over OTLP/HTTP with protobuf through a blocking client: the
//! SDK's batch processor exports on its own thread, so no Tokio reactor is
//! involved. gRPC is not compiled in. Metrics and logs are not exported.
//!
//! The switch is off by default: no exporter thread, no network, no span, and
//! the log output is unchanged. A configuration or startup error is reported
//! and serving continues without telemetry.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use axum::{
    extract::{MatchedPath, Request, State},
    http::{HeaderMap, header::HOST},
    middleware::Next,
    response::Response,
};
use opentelemetry::{
    KeyValue, global,
    propagation::{TextMapCompositePropagator, TextMapPropagator},
    trace::{Status, TracerProvider as _},
};
use opentelemetry_http::HeaderExtractor;
use opentelemetry_sdk::{
    Resource,
    propagation::{BaggagePropagator, TraceContextPropagator},
    trace::SdkTracerProvider,
};
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{
    EnvFilter, Layer, filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt,
};

use super::AppState;

const SERVICE_NAME: &str = "rocket-inference";

/// Whether the OpenTelemetry layer is installed. The middleware and the
/// `inference` span are skipped entirely when it is off, so the disabled path
/// keeps the pre-telemetry output and cost.
static ENABLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Keeps the tracer provider alive for the process lifetime and flushes the
/// queued spans when the server stops.
pub(crate) struct Telemetry {
    provider: Option<SdkTracerProvider>,
}

impl Telemetry {
    /// Install the log subscriber and, when enabled, the OTLP trace pipeline.
    /// Never fails: telemetry problems are logged and serving continues.
    pub(crate) fn init() -> Self {
        use std::io::IsTerminal;

        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(false)
            .with_ansi(std::io::stdout().is_terminal())
            .with_filter(filter);

        let Some(settings) = Settings::from_env() else {
            let _ = tracing_subscriber::registry().with(fmt_layer).try_init();
            return Self { provider: None };
        };

        match start(&settings) {
            Ok(provider) => {
                let tracer = provider.tracer(SERVICE_NAME);
                let otel_layer = tracing_opentelemetry::layer()
                    .with_tracer(tracer)
                    .with_tracked_inactivity(false)
                    // The log layer keeps its `RUST_LOG` filter; spans and span
                    // events are exported at INFO regardless of stdout verbosity.
                    .with_filter(LevelFilter::INFO);
                match tracing_subscriber::registry()
                    .with(fmt_layer)
                    .with(otel_layer)
                    .try_init()
                {
                    Ok(()) => {
                        ENABLED.store(true, Ordering::Relaxed);
                        tracing::info!("OpenTelemetry trace export enabled");
                        Self {
                            provider: Some(provider),
                        }
                    }
                    Err(error) => {
                        let _ = provider.shutdown();
                        eprintln!("rocket-inference: OpenTelemetry disabled: {error}");
                        Self { provider: None }
                    }
                }
            }
            Err(error) => {
                let _ = tracing_subscriber::registry().with(fmt_layer).try_init();
                eprintln!("rocket-inference: OpenTelemetry disabled: {error:#}");
                Self { provider: None }
            }
        }
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take()
            && let Err(error) = provider.shutdown()
        {
            eprintln!("rocket-inference: OpenTelemetry shutdown: {error}");
        }
    }
}

/// The part of the configuration this crate has to interpret itself. The rest
/// is read by the SDK and the OTLP exporter from the environment.
struct Settings {
    propagators: Propagators,
}

impl Settings {
    fn from_env() -> Option<Self> {
        if !master_enabled() || sdk_disabled() {
            return None;
        }
        match parse_exporter(std::env::var("OTEL_TRACES_EXPORTER").ok().as_deref()) {
            ExporterChoice::Otlp => {}
            ExporterChoice::None => return None,
            ExporterChoice::Unsupported(value) => {
                eprintln!(
                    "rocket-inference: OpenTelemetry disabled: unsupported \
                     OTEL_TRACES_EXPORTER={value:?} (supported: otlp, none)"
                );
                return None;
            }
        }
        if let Some(protocol) = incompatible_protocol() {
            eprintln!(
                "rocket-inference: OpenTelemetry disabled: \
                 OTEL_EXPORTER_OTLP_PROTOCOL={protocol:?} is not compiled in; \
                 this build exports OTLP/HTTP with protobuf \
                 (set http/protobuf or unset the variable)"
            );
            return None;
        }
        Some(Self {
            propagators: parse_propagators(std::env::var("OTEL_PROPAGATORS").ok().as_deref()),
        })
    }
}

fn master_enabled() -> bool {
    match std::env::var("OTEL_SDK_ENABLED") {
        Ok(value) => match parse_bool(&value) {
            Some(enabled) => enabled,
            None => {
                eprintln!(
                    "rocket-inference: ignoring invalid OTEL_SDK_ENABLED={value:?} \
                     (expected true or false)"
                );
                false
            }
        },
        Err(_) => false,
    }
}

fn sdk_disabled() -> bool {
    std::env::var("OTEL_SDK_DISABLED")
        .ok()
        .and_then(|value| parse_bool(&value))
        .unwrap_or(false)
}

/// Boolean env values, case-insensitive; `None` is an unrecognized value.
fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" | "" => Some(false),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExporterChoice {
    Otlp,
    None,
    Unsupported(String),
}

fn parse_exporter(value: Option<&str>) -> ExporterChoice {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return ExporterChoice::Otlp;
    };
    let mut choice = None;
    for part in value.split(',') {
        match part.trim().to_ascii_lowercase().as_str() {
            "none" => return ExporterChoice::None,
            "otlp" => choice = Some(ExporterChoice::Otlp),
            "" => {}
            other => {
                if choice.is_none() {
                    choice = Some(ExporterChoice::Unsupported(other.to_owned()));
                }
            }
        }
    }
    choice.unwrap_or(ExporterChoice::Otlp)
}

/// The only protocol compiled in is `http/protobuf`. The exporter silently
/// falls back to it for unknown values, but `grpc` is a common
/// misconfiguration (default port 4317), so that one is refused explicitly.
fn incompatible_protocol() -> Option<String> {
    let protocol = [
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
    ]
    .into_iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
    })?;
    if protocol == "grpc" {
        Some(protocol)
    } else if protocol == "http/json" {
        eprintln!(
            "rocket-inference: OTEL_EXPORTER_OTLP_PROTOCOL=http/json is not \
             compiled in; exporting http/protobuf instead"
        );
        None
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Propagators {
    trace_context: bool,
    baggage: bool,
}

impl Propagators {
    const DEFAULT: Self = Self {
        trace_context: true,
        baggage: true,
    };
    const NONE: Self = Self {
        trace_context: false,
        baggage: false,
    };
}

fn parse_propagators(value: Option<&str>) -> Propagators {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Propagators::DEFAULT;
    };
    let mut propagators = Propagators::NONE;
    for part in value.split(',') {
        match part.trim().to_ascii_lowercase().as_str() {
            "none" => return Propagators::NONE,
            "tracecontext" => propagators.trace_context = true,
            "baggage" => propagators.baggage = true,
            "" => {}
            other => {
                eprintln!("rocket-inference: ignoring unknown OTEL_PROPAGATORS entry {other:?}")
            }
        }
    }
    propagators
}

fn install_propagators(propagators: Propagators) {
    let mut list: Vec<Box<dyn TextMapPropagator + Send + Sync>> = Vec::new();
    if propagators.trace_context {
        list.push(Box::new(TraceContextPropagator::new()));
    }
    if propagators.baggage {
        list.push(Box::new(BaggagePropagator::new()));
    }
    global::set_text_map_propagator(TextMapCompositePropagator::new(list));
}

fn start(settings: &Settings) -> Result<SdkTracerProvider> {
    install_propagators(settings.propagators);
    // The exporter reads the endpoint/headers/timeout/compression/protocol
    // environment itself; the blocking client feature gives it a reqwest client
    // that works on the batch processor's thread.
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
        .context("build the OTLP/HTTP span exporter")?;
    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource())
        .build())
}

/// The SDK defaults `service.name` to `unknown_service:<exe>`; use the binary
/// name instead, unless the operator configured one through either standard
/// variable. `service.version` is added for deployment identification.
fn resource() -> Resource {
    let mut builder = Resource::builder();
    if !env_nonempty("OTEL_SERVICE_NAME") && !resource_attribute_present_in_env("service.name") {
        builder = builder.with_service_name(SERVICE_NAME);
    }
    if !resource_attribute_present_in_env("service.version") {
        builder =
            builder.with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")));
    }
    builder.build()
}

fn env_nonempty(name: &str) -> bool {
    std::env::var(name)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

fn resource_attribute_present_in_env(key: &str) -> bool {
    std::env::var("OTEL_RESOURCE_ATTRIBUTES")
        .map(|value| resource_attribute_present(&value, key))
        .unwrap_or(false)
}

fn resource_attribute_present(value: &str, key: &str) -> bool {
    value.split(',').any(|pair| {
        pair.split_once('=')
            .is_some_and(|(name, _)| name.trim() == key)
    })
}

/// One HTTP server span per request, skipping `/health`. An incoming W3C
/// `traceparent`/`tracestate` (and baggage when enabled) becomes the parent.
pub(crate) async fn trace_http(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    if !enabled() || req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|matched| matched.as_str().to_owned());
    let server = host_parts(req.headers());
    let parent = global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(req.headers()))
    });
    let name = match &route {
        Some(route) => format!("{method} {route}"),
        None => format!("{method} {path}"),
    };
    let span = tracing::info_span!(
        "http.request",
        otel.kind = "server",
        otel.name = %name,
    );
    // `set_parent` only works before the span is first entered, which is here.
    let _ = span.set_parent(parent);
    span.set_attribute("http.request.method", method.as_str().to_owned());
    span.set_attribute("url.path", path);
    if let Some(route) = route {
        span.set_attribute("http.route", route);
    }
    span.set_attribute("rocket.model.name", state.settings.model_name.clone());
    if let Some((address, port)) = server {
        span.set_attribute("server.address", address);
        if let Some(port) = port {
            span.set_attribute("server.port", i64::from(port));
        }
    }

    let response = next.run(req).instrument(span.clone()).await;
    let status = response.status();
    span.set_attribute("http.response.status_code", i64::from(status.as_u16()));
    if status.is_server_error() {
        span.set_status(Status::error(String::new()));
    }
    response
}

/// `Host` header -> `(server.address, server.port)`, handling `host:port`,
/// `[::1]:port` and bare names.
fn host_parts(headers: &HeaderMap) -> Option<(String, Option<u16>)> {
    let host = headers.get(HOST)?.to_str().ok()?.trim();
    if host.is_empty() {
        return None;
    }
    if let Some(rest) = host.strip_prefix('[') {
        let (address, after) = rest.split_once(']')?;
        let port = after.strip_prefix(':').and_then(|port| port.parse().ok());
        return Some((address.to_owned(), port));
    }
    match host.rsplit_once(':') {
        Some((address, port)) if !address.is_empty() => {
            Some((address.to_owned(), port.parse().ok()))
        }
        _ => Some((host.to_owned(), None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_values_are_case_insensitive() {
        for value in ["1", "true", "TRUE", "yes", "on"] {
            assert_eq!(parse_bool(value), Some(true), "{value:?}");
        }
        for value in ["0", "false", "FALSE", "no", "off", ""] {
            assert_eq!(parse_bool(value), Some(false), "{value:?}");
        }
        assert_eq!(parse_bool("maybe"), None);
    }

    #[test]
    fn exporters_select_otlp_by_default_and_none_disables() {
        assert_eq!(parse_exporter(None), ExporterChoice::Otlp);
        assert_eq!(parse_exporter(Some("")), ExporterChoice::Otlp);
        assert_eq!(parse_exporter(Some("otlp")), ExporterChoice::Otlp);
        assert_eq!(parse_exporter(Some("OTLP")), ExporterChoice::Otlp);
        assert_eq!(parse_exporter(Some("none")), ExporterChoice::None);
        assert_eq!(
            parse_exporter(Some("zipkin")),
            ExporterChoice::Unsupported("zipkin".to_owned())
        );
        assert_eq!(parse_exporter(Some("otlp,zipkin")), ExporterChoice::Otlp);
    }

    #[test]
    fn propagators_default_to_w3c_and_none_clears_them() {
        assert_eq!(parse_propagators(None), Propagators::DEFAULT);
        assert_eq!(
            parse_propagators(Some("tracecontext,baggage")),
            Propagators::DEFAULT
        );
        assert_eq!(
            parse_propagators(Some("baggage")),
            Propagators {
                trace_context: false,
                baggage: true
            }
        );
        assert_eq!(parse_propagators(Some("none")), Propagators::NONE);
        assert_eq!(
            parse_propagators(Some("tracecontext,none")),
            Propagators::NONE
        );
    }

    #[test]
    fn resource_attributes_are_matched_by_key() {
        assert!(resource_attribute_present(
            "service.name=embeddings,deployment.environment=prod",
            "service.name"
        ));
        assert!(resource_attribute_present(
            "deployment.environment=prod, service.name = embeddings ",
            "service.name"
        ));
        assert!(!resource_attribute_present(
            "deployment.environment=prod",
            "service.name"
        ));
        assert!(!resource_attribute_present("", "service.name"));
    }

    #[test]
    fn host_headers_split_ports_and_ipv6() {
        let host = |value: &str| {
            let mut headers = HeaderMap::new();
            headers.insert(HOST, value.parse().unwrap());
            host_parts(&headers)
        };
        assert_eq!(
            host("example.com:8383"),
            Some(("example.com".to_owned(), Some(8383)))
        );
        assert_eq!(host("example.com"), Some(("example.com".to_owned(), None)));
        assert_eq!(host("[::1]:8383"), Some(("::1".to_owned(), Some(8383))));
        assert_eq!(host("[::1]"), Some(("::1".to_owned(), None)));
        assert_eq!(
            host("example.com:notaport"),
            Some(("example.com".to_owned(), None))
        );
    }
}
