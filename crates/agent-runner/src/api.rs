//! Axum handlers and typed JSON errors for the agent-runner surface.

use agent_dispatch::TaskDescriptorV1;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::auth::{unix_now_ms, DispatchAuthError, SignedDispatchRequest};
use crate::store::{CapacityExhausted, RunnerState, TaskLifecycle};

/// `GET /v1/capacity` body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityResponse {
    /// Effective max concurrency after clamp to `1..=5`.
    pub max_concurrency: u32,
    /// Occupied concurrency slots (accepted tasks not yet finished).
    pub current_load: u32,
}

/// `POST /v1/task` 202 body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskAccepted {
    /// Runner-assigned task id (UUID).
    pub task_id: String,
}

/// `GET /v1/task/{id}` body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskView {
    /// Task id path echo.
    pub task_id: String,
    /// Lifecycle status.
    pub status: TaskLifecycle,
    /// Present when lifecycle is `completed` or `failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<agent_dispatch::TaskResultV1>,
}

/// Typed API failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "task_not_found",
            message: message.into(),
        }
    }

    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }

    fn unauthorized(err: DispatchAuthError) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: err.code(),
            message: err.to_string(),
        }
    }

    /// Over-capacity: HTTP 503 + `capacity_exhausted` (retryable).
    ///
    /// Chosen over 429 because AGENT_CHALLENGE §4.4 maps "503 exhausted" to
    /// `Timeout` (retryable hop outcome). This is slot exhaustion, not rate limiting.
    fn capacity_exhausted() -> Self {
        let _ = CapacityExhausted;
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "capacity_exhausted",
            message: "runner at max concurrency; retry when a slot frees".into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": self.message,
            "code": self.code,
        });
        (self.status, Json(body)).into_response()
    }
}

/// Mount all runner routes on a fresh router.
#[must_use]
pub fn router(state: RunnerState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/v1/capacity", get(capacity))
        .route("/v1/task", post(post_task))
        .route("/v1/task/{id}", get(get_task))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn readyz() -> &'static str {
    "ready"
}

async fn capacity(State(st): State<RunnerState>) -> Json<CapacityResponse> {
    Json(st.capacity_async().await)
}

/// Accept a task descriptor, optionally under signed single-use dispatch auth.
async fn post_task(
    State(st): State<RunnerState>,
    body: Result<Json<serde_json::Value>, axum::extract::rejection::JsonRejection>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    let Json(raw) = body.map_err(|e| {
        ApiError::bad_request("invalid_json", format!("invalid task descriptor JSON: {e}"))
    })?;

    let descriptor = if st.auth_enabled() {
        let envelope: SignedDispatchRequest = serde_json::from_value(raw).map_err(|e| {
            // Missing auth fields → 401 (not 400) so unsigned probes get unauthorized.
            tracing::info!(
                event = "runner_dispatch_auth_reject",
                reason = "malformed_or_unsigned",
                "dispatch auth rejected"
            );
            let _ = e;
            ApiError::unauthorized(DispatchAuthError::Unauthorized)
        })?;
        st.verify_dispatch_auth(&envelope, unix_now_ms(), Instant::now())
            .await
            .map_err(|err| {
                tracing::info!(
                    event = "runner_dispatch_auth_reject",
                    reason = err.code(),
                    "dispatch auth rejected"
                );
                ApiError::unauthorized(err)
            })?;
        envelope.descriptor
    } else {
        serde_json::from_value::<TaskDescriptorV1>(raw).map_err(|e| {
            ApiError::bad_request("invalid_json", format!("invalid task descriptor JSON: {e}"))
        })?
    };

    if descriptor.protocol != agent_dispatch::DISPATCH_PROTOCOL {
        return Err(ApiError::bad_request(
            "invalid_protocol",
            format!(
                "protocol must be {}",
                agent_dispatch::DISPATCH_PROTOCOL
            ),
        ));
    }
    let task_id = st.accept_task(descriptor).await.map_err(|_: CapacityExhausted| {
        tracing::info!(
            event = "runner_capacity_exhausted",
            "dispatch refused: concurrency slots full"
        );
        ApiError::capacity_exhausted()
    })?;
    tracing::info!(event = "runner_task_accepted", %task_id, "task accepted");
    Ok((
        StatusCode::ACCEPTED,
        Json(TaskAccepted { task_id }),
    ))
}

async fn get_task(
    State(st): State<RunnerState>,
    Path(id): Path<String>,
) -> Result<Json<TaskView>, ApiError> {
    let Some((lifecycle, result)) = st.get_task(&id).await else {
        return Err(ApiError::not_found(format!("task not found: {id}")));
    };
    Ok(Json(TaskView {
        task_id: id,
        status: lifecycle,
        result,
    }))
}
