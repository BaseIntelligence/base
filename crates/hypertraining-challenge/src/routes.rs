//! Miner HTTP routes: `GET /health`, `POST /v1/submissions` (brief §7).
//!
//! Mount via [`submission_router`]. Binary (todo 13) serves this on port 8091.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::submission::{SubmissionError, SubmissionRequest, SubmissionService};

/// Shared HTTP state.
pub type AppState = Arc<SubmissionService>;

/// Router with health + miner submit (no auth beyond JSON schema for now).
pub fn submission_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/submissions", post(post_submission))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn post_submission(
    State(svc): State<AppState>,
    body: Result<Json<SubmissionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => {
            return json_err(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                &rej.body_text(),
            );
        }
    };
    match svc.accept_response(req) {
        Ok(accepted) => (StatusCode::ACCEPTED, Json(accepted)).into_response(),
        Err(e) => map_submission_err(&e),
    }
}

fn map_submission_err(err: &SubmissionError) -> Response {
    let (code, status) = match &err {
        SubmissionError::EmptyField(_)
        | SubmissionError::InvalidTopology
        | SubmissionError::UnknownFormat(_)
        | SubmissionError::UnknownAccumulateDtype(_)
        | SubmissionError::UnknownScalingRecipe(_) => ("invalid_request", StatusCode::BAD_REQUEST),
        SubmissionError::Attestation(_) => ("attestation_rejected", StatusCode::BAD_REQUEST),
        SubmissionError::Admission(_) => ("admission_rejected", StatusCode::BAD_REQUEST),
    };
    json_err(status, code, &err.to_string())
}


fn json_err(status: StatusCode, code: &str, message: &str) -> Response {
    let body: Value = json!({
        "error": message,
        "code": code,
    });
    (status, Json(body)).into_response()
}


#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::submission::example_valid_request;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::String(
                String::from_utf8_lossy(&bytes).into_owned(),
            ))
        };
        (status, v)
    }

    #[tokio::test]
    async fn health_ok() {
        let app = submission_router(Arc::new(SubmissionService::default()));
        let (st, _) = call(
            app,
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
    }

    #[tokio::test]
    async fn post_valid_accepted() {
        let app = submission_router(Arc::new(SubmissionService::default()));
        let body = serde_json::to_vec(&example_valid_request()).unwrap();
        let (st, v) = call(
            app,
            Request::builder()
                .method("POST")
                .uri("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(st, StatusCode::ACCEPTED);
        assert_eq!(v["status"], "accepted");
        assert_eq!(v["challenge_id"], "hypertraining");
        assert!(v["submission_id"].as_str().unwrap().starts_with("ht-sub-"));
    }
}
