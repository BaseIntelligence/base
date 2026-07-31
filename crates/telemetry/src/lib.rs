//! Observability primitives for base services: JSON tracing, Prometheus metrics,
//! and a reusable axum router exposing `/healthz`, `/readyz`, and `/metrics`.
//!
//! Readiness probes (DB, chain, …) are injectable callbacks so this crate stays free
//! of sqlx / chain client dependencies.

#![forbid(unsafe_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Once, OnceLock};

use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde::Serialize;
use thiserror::Error;
use tracing_subscriber::EnvFilter;

/// Failures while installing tracing or the metrics recorder.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TelemetryError {
    /// `tracing-subscriber` refused to install (often already set).
    #[error("tracing init failed: {0}")]
    Tracing(String),
    /// Prometheus recorder could not be installed.
    #[error("metrics init failed: {0}")]
    Metrics(String),
}

static TRACING_INIT: Once = Once::new();
static TRACING_ERROR: OnceLock<String> = OnceLock::new();

/// Install a process-wide JSON `tracing` subscriber.
///
/// Filter comes from `RUST_LOG`, defaulting to `info` when unset or invalid.
/// Safe to call more than once after the first successful install.
///
/// # Errors
///
/// Returns [`TelemetryError::Tracing`] when the global subscriber cannot be set
/// on the first attempt.
pub fn init_tracing() -> Result<(), TelemetryError> {
    TRACING_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let result = tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_current_span(true)
            .with_span_list(true)
            .try_init();
        if let Err(err) = result {
            let _ = TRACING_ERROR.set(err.to_string());
        }
    });

    match TRACING_ERROR.get() {
        None => Ok(()),
        Some(msg) => Err(TelemetryError::Tracing(msg.clone())),
    }
}

static METRICS_INIT: Once = Once::new();
static METRICS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
static METRICS_ERROR: OnceLock<String> = OnceLock::new();

/// Install the global Prometheus metrics recorder and return a scrape handle.
///
/// Idempotent. Registers `gbase_up` gauge = 1 so `/metrics` is non-empty after init.
///
/// # Errors
///
/// Returns [`TelemetryError::Metrics`] if the recorder cannot be installed.
pub fn init_metrics() -> Result<PrometheusHandle, TelemetryError> {
    METRICS_INIT.call_once(|| match PrometheusBuilder::new().install_recorder() {
        Ok(handle) => {
            metrics::describe_gauge!("gbase_up", "1 when telemetry metrics are live");
            metrics::gauge!("gbase_up").set(1.0);
            let _ = METRICS_HANDLE.set(handle);
        }
        Err(err) => {
            let _ = METRICS_ERROR.set(err.to_string());
        }
    });

    if let Some(handle) = METRICS_HANDLE.get() {
        return Ok(handle.clone());
    }

    let msg = METRICS_ERROR
        .get()
        .cloned()
        .unwrap_or_else(|| "metrics recorder was not installed".to_owned());
    Err(TelemetryError::Metrics(msg))
}

/// Borrow the process-wide metrics handle after [`init_metrics`].
///
/// # Errors
///
/// Returns [`TelemetryError::Metrics`] if metrics were never successfully installed.
pub fn metrics_handle() -> Result<PrometheusHandle, TelemetryError> {
    METRICS_HANDLE.get().cloned().ok_or_else(|| {
        TelemetryError::Metrics("call init_metrics() before metrics_handle()".to_owned())
    })
}

/// Outcome of a single readiness probe.
pub type ReadyOutcome = Result<(), String>;

/// Boxed async readiness check used by `/readyz`.
pub type ReadyCheckFuture = Pin<Box<dyn Future<Output = ReadyOutcome> + Send>>;

/// Factory for a readiness probe (DB ping, chain head, …).
pub type ReadyCheckFn = Arc<dyn Fn() -> ReadyCheckFuture + Send + Sync>;

/// Named readiness probe registered on the health router.
#[derive(Clone)]
pub struct ReadyCheck {
    /// Short identifier shown in `/readyz` JSON (e.g. `"db"`, `"chain"`).
    pub name: String,
    /// Async callback returning `Ok(())` when ready.
    pub check: ReadyCheckFn,
}

impl ReadyCheck {
    /// Build a named check from an async closure.
    pub fn new<N, F, Fut>(name: N, check: F) -> Self
    where
        N: Into<String>,
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ReadyOutcome> + Send + 'static,
    {
        let check = Arc::new(move || -> ReadyCheckFuture { Box::pin(check()) });
        Self {
            name: name.into(),
            check,
        }
    }

    /// Always-ready probe (default for unit tests).
    #[must_use]
    pub fn always_ok(name: impl Into<String>) -> Self {
        Self::new(name, || async { Ok(()) })
    }

    /// Always-failing probe (edge-case tests).
    #[must_use]
    pub fn always_err(name: impl Into<String>, message: impl Into<String>) -> Self {
        let message = message.into();
        Self::new(name, move || {
            let message = message.clone();
            async move { Err(message) }
        })
    }
}

/// Shared state for the health / metrics router.
#[derive(Clone)]
pub struct HealthState {
    checks: Arc<Vec<ReadyCheck>>,
    metrics: PrometheusHandle,
}

/// Builder for the reusable observability router.
#[derive(Clone, Default)]
pub struct HealthRouterBuilder {
    checks: Vec<ReadyCheck>,
    metrics: Option<PrometheusHandle>,
}

impl HealthRouterBuilder {
    /// Empty builder (no readiness checks).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a Prometheus scrape handle (from [`init_metrics`]).
    #[must_use]
    pub fn metrics_handle(mut self, handle: PrometheusHandle) -> Self {
        self.metrics = Some(handle);
        self
    }

    /// Register one readiness probe (e.g. DB or chain).
    #[must_use]
    pub fn readiness_check(mut self, check: ReadyCheck) -> Self {
        self.checks.push(check);
        self
    }

    /// Register many readiness probes.
    #[must_use]
    pub fn readiness_checks<I>(mut self, checks: I) -> Self
    where
        I: IntoIterator<Item = ReadyCheck>,
    {
        self.checks.extend(checks);
        self
    }

    /// Build the axum `Router` with `/healthz`, `/readyz`, and `/metrics`.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::Metrics`] when no metrics handle was provided and
    /// [`init_metrics`] has not been called successfully.
    pub fn build(self) -> Result<Router, TelemetryError> {
        let metrics = match self.metrics {
            Some(handle) => handle,
            None => init_metrics()?,
        };
        let state = HealthState {
            checks: Arc::new(self.checks),
            metrics,
        };
        Ok(Router::new()
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .route("/metrics", get(metrics_handler))
            .with_state(state))
    }
}

/// Convenience: router with the given metrics handle and zero readiness checks.
///
/// Empty checks mean `/readyz` always returns 200 (suitable for unit tests).
///
/// # Errors
///
/// Propagates builder failures (should not occur when a handle is supplied).
pub fn health_router(metrics: PrometheusHandle) -> Result<Router, TelemetryError> {
    HealthRouterBuilder::new().metrics_handle(metrics).build()
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

#[derive(Debug, Serialize)]
struct ReadyBody {
    status: &'static str,
    checks: Vec<CheckStatus>,
}

#[derive(Debug, Serialize)]
struct CheckStatus {
    name: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn readyz(State(state): State<HealthState>) -> Response {
    let mut checks = Vec::with_capacity(state.checks.len());
    let mut all_ok = true;

    for probe in state.checks.iter() {
        match (probe.check)().await {
            Ok(()) => checks.push(CheckStatus {
                name: probe.name.clone(),
                ok: true,
                error: None,
            }),
            Err(err) => {
                all_ok = false;
                checks.push(CheckStatus {
                    name: probe.name.clone(),
                    ok: false,
                    error: Some(err),
                });
            }
        }
    }

    let status = if all_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let body = ReadyBody {
        status: if all_ok { "ok" } else { "not_ready" },
        checks,
    };
    (status, Json(body)).into_response()
}

async fn metrics_handler(State(state): State<HealthState>) -> Response {
    let body = state.metrics.render();
    let mut response = Response::new(body.into());
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn body_text(response: Response) -> String {
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        String::from_utf8(bytes.to_vec()).expect("utf8 body")
    }

    fn test_router(checks: Vec<ReadyCheck>) -> Router {
        let handle = init_metrics().expect("metrics");
        HealthRouterBuilder::new()
            .metrics_handle(handle)
            .readiness_checks(checks)
            .build()
            .expect("router")
    }

    /// S1: GET /healthz → 200 and body contains ok.
    #[tokio::test]
    async fn healthz_returns_200_ok() {
        let app = test_router(vec![]);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/healthz")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("oneshot");

        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        assert!(
            text.contains("ok"),
            "healthz body should contain ok, got {text:?}"
        );
    }

    /// S2: GET /metrics → 200 and a well-formed Prometheus exposition body.
    #[tokio::test]
    async fn metrics_returns_well_formed_prometheus_body() {
        let app = test_router(vec![]);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/metrics")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("oneshot");

        assert_eq!(response.status(), StatusCode::OK);

        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            content_type.contains("text/plain"),
            "expected prometheus text/plain content-type, got {content_type:?}"
        );

        let text = body_text(response).await;
        assert!(
            !text.is_empty(),
            "prometheus body must not be empty after init_metrics"
        );
        let well_formed = text.contains("# HELP")
            || text.contains("# TYPE")
            || text.lines().any(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with('#') && trimmed.contains(' ')
            });
        assert!(
            well_formed,
            "body is not well-formed prometheus exposition:\n{text}"
        );
        assert!(
            text.contains("gbase_up"),
            "expected gbase_up metric in body:\n{text}"
        );
    }

    /// S3 edge: empty checks → /readyz 200; failing check → 503.
    #[tokio::test]
    async fn readyz_default_ok_and_failing_check_503() {
        let ok_app = test_router(vec![]);
        let ok_response = ok_app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/readyz")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("oneshot");
        assert_eq!(ok_response.status(), StatusCode::OK);
        let ok_text = body_text(ok_response).await;
        assert!(ok_text.contains("ok"), "readyz default body: {ok_text}");

        let fail_app = test_router(vec![
            ReadyCheck::always_ok("db"),
            ReadyCheck::always_err("chain", "rpc timeout"),
        ]);
        let fail_response = fail_app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/readyz")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("oneshot");
        assert_eq!(fail_response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let fail_text = body_text(fail_response).await;
        assert!(
            fail_text.contains("not_ready") || fail_text.contains("chain"),
            "readyz failure body: {fail_text}"
        );
        assert!(
            fail_text.contains("rpc timeout"),
            "readyz should surface check error: {fail_text}"
        );
    }

    /// S3 happy path with injectable always-ok checks (DB + chain hooks).
    #[tokio::test]
    async fn readyz_with_ok_checks_returns_200() {
        let app = test_router(vec![
            ReadyCheck::always_ok("db"),
            ReadyCheck::always_ok("chain"),
        ]);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/readyz")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("oneshot");
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        assert!(text.contains("db"), "{text}");
        assert!(text.contains("chain"), "{text}");
    }

    /// S4: `init_tracing` is callable; second call does not panic.
    #[test]
    fn init_tracing_is_safe_to_call_twice() {
        let first = init_tracing();
        let second = init_tracing();
        assert_eq!(
            first.is_ok(),
            second.is_ok(),
            "first={first:?} second={second:?}"
        );
    }

    #[test]
    fn init_metrics_is_idempotent() {
        let a = init_metrics().expect("a");
        let b = init_metrics().expect("b");
        let _ = a.render();
        let _ = b.render();
    }
}
