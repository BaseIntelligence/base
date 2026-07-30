//! Axum handlers and typed JSON errors for the agent-runner surface.

use agent_dispatch::TaskDescriptorV1;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::store::{RunnerState, TaskLifecycle};

/// `GET /v1/capacity` body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityResponse {
    /// Effective configured max concurrency (advertisement only until todo 19).
    pub max_concurrency: u32,
    /// Tasks currently in `running` lifecycle.
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

/// Accept a task descriptor.
///
/// **Auth stub (todo 18):** no signature / nonce check. Any syntactically valid
/// [`TaskDescriptorV1`] is accepted. Do not expose this route on a public
/// ingress until authenticated dispatch ships.
async fn post_task(
    State(st): State<RunnerState>,
    body: Result<Json<TaskDescriptorV1>, axum::extract::rejection::JsonRejection>,
) -> Result<(StatusCode, Json<TaskAccepted>), ApiError> {
    let Json(descriptor) = body.map_err(|e| {
        ApiError::bad_request("invalid_json", format!("invalid task descriptor JSON: {e}"))
    })?;
    if descriptor.protocol != agent_dispatch::DISPATCH_PROTOCOL {
        return Err(ApiError::bad_request(
            "invalid_protocol",
            format!(
                "protocol must be {}",
                agent_dispatch::DISPATCH_PROTOCOL
            ),
        ));
    }
    let task_id = st.accept_task(descriptor).await;
    tracing::info!(
        event = "runner_task_accepted",
        %task_id,
        "task accepted (auth not enforced — todo 18)"
    );
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
