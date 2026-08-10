//! Miner + admin + public HTTP routes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Multipart, Path as AxumPath, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bounty_challenge_task::{
    score_epoch, EpochScoreInput, BUG_ID_DOMAIN, CHALLENGE_ID, SCORE_MAX, SCORING_VERSION,
    TARGET_BUGS,
};
use bounty_store::{BountyStore, BugPatch, BugRow, BugStatus, StageEvent};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use submission_gating::{MetagraphCache, METAGRAPH_CACHE_TTL_SECS};
use tokio_util::io::ReaderStream;

/// Shared HTTP state.
pub struct AppState {
    /// Persistence.
    pub store: Arc<dyn BountyStore>,
    /// Backend label (`memory` / `postgres` / …).
    pub backend_mode: String,
    /// Artifacts root for video files.
    pub artifacts_root: PathBuf,
    /// Admin bearer token sha256 hex digests.
    pub admin_token_hashes: Vec<String>,
    /// Cached metagraph for intake membership. `None` disables the check.
    pub metagraph: Option<Arc<MetagraphCache>>,
    /// Current chain epoch cache.
    pub epoch: Arc<AtomicU64>,
    /// Max raw upload bytes (default ~100 MiB).
    pub max_upload_bytes: usize,
}

/// Build router.
pub fn bounty_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/status", get(get_status))
        .route("/v1/bugs", post(post_bug).get(list_bugs))
        .route("/v1/bugs/{id}", get(get_bug))
        .route("/v1/bugs/{id}/video", get(get_bug_video))
        .route("/v1/admin/bugs", get(admin_list_bugs))
        .route("/v1/admin/bugs/{id}", get(admin_get_bug))
        .route("/v1/admin/bugs/{id}/approve", post(admin_approve))
        .route("/v1/admin/bugs/{id}/reject", post(admin_reject))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({"status":"ok","challenge_id":CHALLENGE_ID})),
    )
}

async fn get_status(State(st): State<Arc<AppState>>) -> Response {
    let epoch = st.epoch.load(Ordering::Relaxed);
    let pending = st
        .store
        .list_bugs(Some("pending_admin"), None, 500)
        .await
        .map_or(0, |v| v.len());
    let approved_rows = st
        .store
        .approved_points_for_epoch(epoch)
        .await
        .unwrap_or_default();
    let total_approved: u64 = approved_rows.values().map(|p| u64::from(*p)).sum();
    let preview = score_epoch(&EpochScoreInput {
        approved_points: approved_rows,
    });
    Json(json!({
        "challenge_id": CHALLENGE_ID,
        "scoring_version": SCORING_VERSION,
        "target_bugs": TARGET_BUGS,
        "backend": st.backend_mode,
        "epoch": epoch,
        "pending_admin": pending,
        "approved_this_epoch": total_approved,
        "burn_preview": {
            "miner_pool": preview.miner_pool,
            "burn_units": preview.burn_units,
            "capped": preview.capped,
            "score_max": SCORE_MAX,
        },
    }))
    .into_response()
}

fn json_err(code: StatusCode, kind: &str, msg: &str) -> Response {
    (code, Json(json!({"error": kind, "message": msg}))).into_response()
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn token_hash(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    hex::encode(h.finalize())
}

fn bearer_token(headers: &HeaderMap) -> Result<&str, Response> {
    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let token = auth
        .strip_prefix("Bearer ")
        .or_else(|| auth.strip_prefix("bearer "))
        .unwrap_or("")
        .trim();
    if token.is_empty() {
        return Err(json_err(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing bearer",
        ));
    }
    Ok(token)
}

fn check_admin(st: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    if st.admin_token_hashes.is_empty() {
        return Err(json_err(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "admin tokens not configured",
        ));
    }
    let token = bearer_token(headers)?;
    let th = token_hash(token);
    if !st.admin_token_hashes.iter().any(|h| h == &th) {
        return Err(json_err(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "bad token",
        ));
    }
    Ok(())
}

fn miner_hotkey(headers: &HeaderMap) -> Result<String, Response> {
    let hk = headers
        .get("x-miner-hotkey")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            json_err(
                StatusCode::BAD_REQUEST,
                "missing_hotkey",
                "X-Miner-Hotkey required",
            )
        })?;
    let hk = hk.to_ascii_lowercase();
    if hk.len() != 64 || !hk.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(json_err(
            StatusCode::BAD_REQUEST,
            "bad_hotkey",
            "X-Miner-Hotkey must be 64 hex chars",
        ));
    }
    Ok(hk)
}

fn metagraph_check(st: &AppState, hotkey: &str) -> Result<Option<String>, Response> {
    let Some(cache) = &st.metagraph else {
        return Ok(None);
    };
    match cache.snapshot_fresh(METAGRAPH_CACHE_TTL_SECS) {
        Some(view) => match view.uid_of_hex(hotkey) {
            Some(_) => Ok(view.coldkey_hex_of(hotkey)),
            None => Err(json_err(
                StatusCode::FORBIDDEN,
                "hotkey_not_in_metagraph",
                "miner hotkey is not registered on this subnet",
            )),
        },
        None => Err(json_err(
            StatusCode::SERVICE_UNAVAILABLE,
            "metagraph_unavailable",
            "metagraph snapshot not ready; retry shortly",
        )),
    }
}

fn bug_id(hotkey: &str, video: &[u8], title: &str, description: &str, at_ms: u64) -> String {
    let mut h = Sha256::new();
    h.update(BUG_ID_DOMAIN);
    h.update(hotkey.as_bytes());
    h.update(video);
    h.update(title.as_bytes());
    h.update(description.as_bytes());
    h.update(at_ms.to_le_bytes());
    hex::encode(h.finalize())
}

fn ext_for_mime(mime: &str, filename: Option<&str>) -> &'static str {
    let lower = mime.to_ascii_lowercase();
    if lower.contains("webm") {
        return "webm";
    }
    if lower.contains("quicktime") || lower.contains("mov") {
        return "mov";
    }
    if let Some(name) = filename {
        let n = name.to_ascii_lowercase();
        if n.ends_with(".webm") {
            return "webm";
        }
        if n.ends_with(".mov") {
            return "mov";
        }
    }
    "mp4"
}

fn validate_video_mime(mime: &str, filename: Option<&str>) -> bool {
    let m = mime.to_ascii_lowercase();
    if m.contains("mp4") || m.contains("webm") || m.contains("quicktime") {
        return true;
    }
    if let Some(name) = filename {
        let n = name.to_ascii_lowercase();
        return n.ends_with(".mp4") || n.ends_with(".webm") || n.ends_with(".mov");
    }
    false
}

fn bug_public_json(b: &BugRow, admin: bool) -> serde_json::Value {
    let mut v = json!({
        "id": b.id,
        "miner_hotkey": b.miner_hotkey,
        "app_id": b.app_id,
        "title": b.title,
        "description": b.description,
        "steps": b.steps,
        "status": b.status.as_str(),
        "nearest_id": b.nearest_id,
        "video_sha256": b.video_sha256,
        "video_bytes": b.video_bytes,
        "reject_reason": b.reject_reason,
        "epoch": b.epoch,
        "created_at_ms": b.created_at_ms,
        "updated_at_ms": b.updated_at_ms,
        "has_video": b.video_path.is_some(),
    });
    if admin {
        v["agentic_verdict"] = b.agentic_verdict.clone().unwrap_or(json!(null));
        v["miner_coldkey"] = json!(b.miner_coldkey);
        v["video_path"] = json!(b.video_path);
    }
    v
}

async fn post_bug(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let hotkey = match miner_hotkey(&headers) {
        Ok(h) => h,
        Err(r) => return r,
    };
    let coldkey = match metagraph_check(&st, &hotkey) {
        Ok(c) => c,
        Err(r) => return r,
    };

    let mut title = String::new();
    let mut description = String::new();
    let mut app_id = String::new();
    let mut steps: Option<String> = None;
    let mut video: Option<(Vec<u8>, String, Option<String>)> = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return json_err(
                    StatusCode::BAD_REQUEST,
                    "multipart",
                    &format!("invalid multipart: {e}"),
                );
            }
        };
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "title" => title = field.text().await.unwrap_or_default(),
            "description" => description = field.text().await.unwrap_or_default(),
            "app_id" => app_id = field.text().await.unwrap_or_default(),
            "steps" => {
                let t = field.text().await.unwrap_or_default();
                if !t.trim().is_empty() {
                    steps = Some(t);
                }
            }
            "video" => {
                let mime = field.content_type().map(str::to_owned).unwrap_or_default();
                let filename = field.file_name().map(str::to_owned);
                let bytes = match field.bytes().await {
                    Ok(b) => b.to_vec(),
                    Err(e) => {
                        return json_err(
                            StatusCode::BAD_REQUEST,
                            "video_read",
                            &format!("failed to read video: {e}"),
                        );
                    }
                };
                if bytes.len() > st.max_upload_bytes {
                    return json_err(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "video_too_large",
                        &format!("max {} bytes", st.max_upload_bytes),
                    );
                }
                if bytes.is_empty() {
                    return json_err(StatusCode::BAD_REQUEST, "empty_video", "video is empty");
                }
                if !validate_video_mime(&mime, filename.as_deref()) {
                    return json_err(
                        StatusCode::BAD_REQUEST,
                        "bad_video_type",
                        "video must be mp4/webm/mov",
                    );
                }
                video = Some((bytes, mime, filename));
            }
            _ => {}
        }
    }

    let title = title.trim().to_owned();
    let description = description.trim().to_owned();
    let app_id = app_id.trim().to_owned();
    if title.is_empty() || title.len() > 256 {
        return json_err(
            StatusCode::BAD_REQUEST,
            "bad_title",
            "title required (≤256)",
        );
    }
    if description.is_empty() || description.len() > 65_536 {
        return json_err(
            StatusCode::BAD_REQUEST,
            "bad_description",
            "description required (≤64KiB)",
        );
    }
    if app_id.is_empty() || app_id.len() > 128 {
        return json_err(
            StatusCode::BAD_REQUEST,
            "bad_app_id",
            "app_id required (≤128)",
        );
    }
    let Some((video_bytes, mime, filename)) = video else {
        return json_err(
            StatusCode::BAD_REQUEST,
            "missing_video",
            "video field required",
        );
    };

    let at_ms = now_ms();
    let id = bug_id(&hotkey, &video_bytes, &title, &description, at_ms);
    let ext = ext_for_mime(&mime, filename.as_deref());
    let dir = st.artifacts_root.join(&id);
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        return json_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifacts",
            &e.to_string(),
        );
    }
    let raw_path = dir.join(format!("raw.{ext}"));
    if let Err(e) = tokio::fs::write(&raw_path, &video_bytes).await {
        return json_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifacts",
            &e.to_string(),
        );
    }

    let row = BugRow {
        id: id.clone(),
        miner_hotkey: hotkey,
        miner_coldkey: coldkey,
        app_id,
        title,
        description,
        steps,
        status: BugStatus::Uploaded,
        agentic_verdict: None,
        nearest_id: None,
        video_sha256: None,
        video_bytes: None,
        video_path: Some(raw_path.display().to_string()),
        reject_reason: None,
        epoch: st.epoch.load(Ordering::Relaxed),
        created_at_ms: at_ms,
        updated_at_ms: at_ms,
    };
    match st.store.insert_bug(&row).await {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(json!({
                "id": id,
                "status": "uploaded",
                "epoch": row.epoch,
            })),
        )
            .into_response(),
        Err(bounty_store::StoreError::Duplicate) => {
            json_err(StatusCode::CONFLICT, "duplicate", "bug id already exists")
        }
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    status: Option<String>,
    mine: Option<String>,
    limit: Option<u32>,
}

async fn list_bugs(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<ListQuery>,
) -> Response {
    let limit = q.limit.unwrap_or(50).min(200);
    let miner = if matches!(q.mine.as_deref(), Some("1") | Some("true") | Some("yes")) {
        match miner_hotkey(&headers) {
            Ok(h) => Some(h),
            Err(r) => return r,
        }
    } else {
        None
    };
    let status = q.status.as_deref();
    match st.store.list_bugs(status, miner.as_deref(), limit).await {
        Ok(rows) => Json(json!({
            "bugs": rows.iter().map(|b| bug_public_json(b, false)).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn get_bug(State(st): State<Arc<AppState>>, AxumPath(id): AxumPath<String>) -> Response {
    match st.store.get_bug(&id).await {
        Ok(Some(b)) => Json(bug_public_json(&b, false)).into_response(),
        Ok(None) => json_err(StatusCode::NOT_FOUND, "not_found", "bug not found"),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn get_bug_video(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let bug = match st.store.get_bug(&id).await {
        Ok(Some(b)) => b,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "not_found", "bug not found"),
        Err(e) => {
            return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string());
        }
    };
    let Some(path) = bug.video_path.as_deref() else {
        return json_err(StatusCode::NOT_FOUND, "no_video", "video not ready");
    };
    // Prefer compressed artifact when present.
    let compressed = st.artifacts_root.join(&id).join("video.mp4");
    let serve = if compressed.is_file() {
        compressed
    } else {
        PathBuf::from(path)
    };
    serve_video_file(&serve).await
}

async fn serve_video_file(path: &Path) -> Response {
    let file = match tokio::fs::File::open(path).await {
        Ok(f) => f,
        Err(_) => return json_err(StatusCode::NOT_FOUND, "no_video", "video file missing"),
    };
    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);
    (StatusCode::OK, [(header::CONTENT_TYPE, "video/mp4")], body).into_response()
}

async fn admin_list_bugs(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<ListQuery>,
) -> Response {
    if let Err(r) = check_admin(&st, &headers) {
        return r;
    }
    let limit = q.limit.unwrap_or(50).min(200);
    let status = q.status.as_deref().unwrap_or("pending_admin");
    match st.store.list_bugs(Some(status), None, limit).await {
        Ok(rows) => Json(json!({
            "bugs": rows.iter().map(|b| bug_public_json(b, true)).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn admin_get_bug(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Err(r) = check_admin(&st, &headers) {
        return r;
    }
    match st.store.get_bug(&id).await {
        Ok(Some(b)) => {
            let events = st.store.events(&id).await.unwrap_or_default();
            let mut v = bug_public_json(&b, true);
            v["events"] = json!(events);
            Json(v).into_response()
        }
        Ok(None) => json_err(StatusCode::NOT_FOUND, "not_found", "bug not found"),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn admin_approve(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Err(r) = check_admin(&st, &headers) {
        return r;
    }
    let bug = match st.store.get_bug(&id).await {
        Ok(Some(b)) => b,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "not_found", "bug not found"),
        Err(e) => {
            return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string());
        }
    };
    if bug.status != BugStatus::PendingAdmin {
        return json_err(
            StatusCode::CONFLICT,
            "not_pending",
            &format!("status={}", bug.status.as_str()),
        );
    }
    match st
        .store
        .apply(
            &id,
            &BugPatch {
                status: Some(BugStatus::Approved),
                ..BugPatch::default()
            },
            Some(&StageEvent {
                stage: "approved".into(),
                detail: Some(json!({"by": "admin"})),
                at_ms: 0,
            }),
        )
        .await
    {
        Ok(b) => Json(bug_public_json(&b, true)).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct RejectBody {
    reason: Option<String>,
}

async fn admin_reject(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
    body: Option<Json<RejectBody>>,
) -> Response {
    if let Err(r) = check_admin(&st, &headers) {
        return r;
    }
    let bug = match st.store.get_bug(&id).await {
        Ok(Some(b)) => b,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "not_found", "bug not found"),
        Err(e) => {
            return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string());
        }
    };
    if bug.status != BugStatus::PendingAdmin {
        return json_err(
            StatusCode::CONFLICT,
            "not_pending",
            &format!("status={}", bug.status.as_str()),
        );
    }
    let reason = body
        .and_then(|b| b.reason.clone())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "admin_reject".into());
    let reason: String = reason.chars().take(500).collect();
    match st
        .store
        .apply(
            &id,
            &BugPatch {
                status: Some(BugStatus::Rejected),
                reject_reason: Some(reason.clone()),
                ..BugPatch::default()
            },
            Some(&StageEvent {
                stage: "rejected".into(),
                detail: Some(json!({"by": "admin", "reason": reason})),
                at_ms: 0,
            }),
        )
        .await
    {
        Ok(b) => Json(bug_public_json(&b, true)).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use bounty_store::MemoryBountyStore;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn state_with_admin(token: &str) -> Arc<AppState> {
        let dir = tempfile::tempdir().unwrap();
        // Leak tempdir for test process lifetime.
        let artifacts = dir.path().to_path_buf();
        std::mem::forget(dir);
        Arc::new(AppState {
            store: Arc::new(MemoryBountyStore::new()),
            backend_mode: "memory".into(),
            artifacts_root: artifacts,
            admin_token_hashes: vec![token_hash(token)],
            metagraph: None,
            epoch: Arc::new(AtomicU64::new(42)),
            max_upload_bytes: 1_000_000,
        })
    }

    #[tokio::test]
    async fn health_ok() {
        let st = state_with_admin("secret");
        let app = bounty_router(st);
        let res = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn multipart_submit_and_admin_approve() {
        let token = "test-admin-token";
        let st = state_with_admin(token);
        let app = bounty_router(Arc::clone(&st));
        let hk = "ab".repeat(32);
        let boundary = "----BOUND";
        let mut body = Vec::new();
        for (name, val) in [
            ("title", "crash"),
            ("description", "null deref"),
            ("app_id", "demo"),
        ] {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(val.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"video\"; filename=\"a.mp4\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: video/mp4\r\n\r\n");
        body.extend_from_slice(b"fake-mp4-bytes");
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let res = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/bugs")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .header("x-miner-hotkey", &hk)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::ACCEPTED);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let id = v["id"].as_str().unwrap().to_owned();

        // Force pending_admin for admin path (pipeline covered elsewhere).
        st.store
            .apply(
                &id,
                &BugPatch {
                    status: Some(BugStatus::PendingAdmin),
                    ..BugPatch::default()
                },
                None,
            )
            .await
            .unwrap();

        let app = bounty_router(Arc::clone(&st));
        let res = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(format!("/v1/admin/bugs/{id}/approve"))
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bug = st.store.get_bug(&id).await.unwrap().unwrap();
        assert_eq!(bug.status, BugStatus::Approved);
    }
}
