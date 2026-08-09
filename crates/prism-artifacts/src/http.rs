//! Fail-closed admin routes for artifact receipt / operator re-stage.
//!
//! Mounted under the same `PRISM_ADMIN_TOKENS_FILE` bearer gate as retry /
//! playground. Empty tokens → 503 (never open).

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, MethodRouter};
use axum::Json;
use serde_json::json;

use crate::receive::{receive_bytes, verify_parked, ReceiveSource, MAX_CHECKPOINT_BYTES};
use crate::{artifact_dir_for, validate_submission_id};

/// `POST /v1/admin/artifacts/{submission_id}/receive` — raw body upload.
pub fn artifact_receive_route<S>() -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    post(post_receive).layer(DefaultBodyLimit::max(MAX_CHECKPOINT_BYTES))
}

/// `GET /v1/admin/artifacts/{submission_id}` — receipt JSON after verify.
pub fn artifact_get_route<S>() -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    get(get_receipt)
}

fn json_err(code: StatusCode, kind: &str, msg: &str) -> Response {
    (code, Json(json!({"error": msg, "code": kind}))).into_response()
}

async fn post_receive(
    Path(submission_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(e) = validate_submission_id(&submission_id) {
        return json_err(
            StatusCode::BAD_REQUEST,
            "invalid_submission_id",
            &e.to_string(),
        );
    }
    let sha = headers
        .get("x-prism-sha256")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let Some(sha) = sha else {
        return json_err(
            StatusCode::BAD_REQUEST,
            "missing_sha256",
            "X-Prism-Sha256 header required",
        );
    };
    let filename = headers
        .get("x-prism-filename")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("checkpoint.pt");
    let dest = artifact_dir_for(&submission_id);
    let result = tokio::task::spawn_blocking({
        let sid = submission_id.clone();
        let sha = sha.to_owned();
        let filename = filename.to_owned();
        let body = body.to_vec();
        move || {
            receive_bytes(
                &sid,
                &dest,
                &filename,
                &body,
                Some(&sha),
                ReceiveSource::AdminUpload,
            )
        }
    })
    .await;
    match result {
        Ok(Ok((_path, receipt))) => (StatusCode::OK, Json(json!(receipt))).into_response(),
        Ok(Err(e)) => {
            let msg = e.to_string();
            let integrity = msg.starts_with("integrity");
            json_err(
                if integrity {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                },
                if integrity {
                    "integrity"
                } else {
                    "receive_failed"
                },
                &msg,
            )
        }
        Err(e) => json_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "receive_failed",
            &format!("join: {e}"),
        ),
    }
}

async fn get_receipt(Path(submission_id): Path<String>) -> Response {
    if let Err(e) = validate_submission_id(&submission_id) {
        return json_err(
            StatusCode::BAD_REQUEST,
            "invalid_submission_id",
            &e.to_string(),
        );
    }
    match verify_parked(&submission_id) {
        Ok(receipt) => (StatusCode::OK, Json(json!(receipt))).into_response(),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("missing") {
                json_err(StatusCode::NOT_FOUND, "not_found", &msg)
            } else {
                json_err(StatusCode::BAD_REQUEST, "integrity", &msg)
            }
        }
    }
}
