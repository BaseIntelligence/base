//! Fail-closed admin routes for artifact receipt / operator re-stage.
//!
//! Mounted under the same `PRISM_ADMIN_TOKENS_FILE` bearer gate as retry /
//! playground. Empty tokens → 503 (never open).
//!
//! Size budget: FP32 × 1.5 = `n_params * 6` bytes ([`checkpoint_byte_budget`]).
//! `n_params` is resolved from the submission store, else `X-Prism-N-Params`;
//! unknown → reject (fail-closed). HTTP body ceiling stays at recipe max
//! ([`MAX_CHECKPOINT_BYTES`]); the per-receive check is tighter.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, MethodRouter};
use axum::Json;
use prism_store::PrismStore;
use serde_json::json;

use crate::receive::{
    checkpoint_byte_budget, receive_bytes, verify_parked, ReceiveSource, MAX_CHECKPOINT_BYTES,
};
use crate::{artifact_dir_for, validate_submission_id};

/// Narrow state for admin artifact receive (needs store for `n_params`).
#[derive(Clone)]
pub struct ArtifactReceiveState {
    /// Submission store (metrics → `n_params`).
    pub store: Arc<dyn PrismStore>,
}

/// `POST /v1/admin/artifacts/{submission_id}/receive` — raw body upload.
pub fn artifact_receive_route<S>(st: ArtifactReceiveState) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    post(post_receive)
        .layer(DefaultBodyLimit::max(MAX_CHECKPOINT_BYTES))
        .with_state(st)
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

#[allow(clippy::result_large_err)] // axum Response is the natural error carrier here
fn parse_n_params_header(headers: &HeaderMap) -> Result<Option<u64>, Response> {
    let Some(raw) = headers
        .get("x-prism-n-params")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Ok(None);
    };
    raw.parse::<u64>().map(Some).map_err(|_| {
        json_err(
            StatusCode::BAD_REQUEST,
            "invalid_n_params",
            "X-Prism-N-Params must be a positive integer",
        )
    })
}

#[allow(clippy::result_large_err)] // axum Response is the natural error carrier here
async fn resolve_n_params(
    store: &Arc<dyn PrismStore>,
    submission_id: &str,
    header: Option<u64>,
) -> Result<u64, Response> {
    if header == Some(0) {
        return Err(json_err(
            StatusCode::BAD_REQUEST,
            "invalid_n_params",
            "X-Prism-N-Params must be > 0",
        ));
    }
    // Prefer measured metrics on the submission; header only when unknown.
    match store.get(submission_id).await {
        Ok(Some(row)) => {
            if let Some(n) = row.n_params().filter(|n| *n > 0) {
                return Ok(n);
            }
            if let Some(n) = header.filter(|n| *n > 0) {
                return Ok(n);
            }
            Err(json_err(
                StatusCode::BAD_REQUEST,
                "n_params_unknown",
                "submission has no measured n_params; pass X-Prism-N-Params",
            ))
        }
        Ok(None) => {
            // No row — allow explicit header for operator re-stage of a park id.
            if let Some(n) = header.filter(|n| *n > 0) {
                return Ok(n);
            }
            Err(json_err(
                StatusCode::BAD_REQUEST,
                "n_params_unknown",
                "submission not found; pass X-Prism-N-Params",
            ))
        }
        Err(e) => Err(json_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "store_error",
            &format!("store: {e}"),
        )),
    }
}

async fn post_receive(
    State(st): State<ArtifactReceiveState>,
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
    let header_n = match parse_n_params_header(&headers) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let n_params = match resolve_n_params(&st.store, &submission_id, header_n).await {
        Ok(n) => n,
        Err(r) => return r,
    };
    let max_bytes = match checkpoint_byte_budget(n_params) {
        Ok(b) => b,
        Err(e) => {
            return json_err(StatusCode::BAD_REQUEST, "invalid_n_params", &e.to_string());
        }
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
                max_bytes,
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
