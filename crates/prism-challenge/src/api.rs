//! Public PRISM HTTP API (miners + operators).
//!
//! | Route | Purpose |
//! |-------|---------|
//! | `GET  /health` | liveness |
//! | `POST /v1/submissions` | accept a two-script recipe |
//! | `GET  /v1/submissions` | list (`status` / `miner` filter, limit) |
//! | `GET  /v1/submissions/{id}` | full detail + event timeline |
//! | `GET  /v1/submissions/{id}/events` | journal only |
//! | `GET  /v1/status` | queue sizes + backend + recipe pin |
//! | `GET  /v1/jobs` | orchestrator jobs view (active/last per pod) |
//! | `GET  /v1/recipe` | recipe descriptor (full data contract) |
//! | `GET  /v1/recipe/baseline` | baseline sources pairs |
//!
//! The API never blocks on the chain: acceptance timestamps the chain epoch
//! read at boot loop; if that read fails the epoch stays at the last known
//! value (still `>= 0`), never guessed.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use submission_gating::{GatingState, GatingStore, MetagraphCache};

use prism_recipe::{BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY};

use crate::CHALLENGE_ID;
use prism_pipeline::{SubmissionError, SubmissionRequest};
use prism_store::{FinalScore, PrismStore, Stage, StoreError, SubmissionState};

/// Shared HTTP app state.
#[derive(Debug)]
pub struct AppState {
    /// Store.
    pub store: Arc<dyn PrismStore>,
    /// Current chain epoch cache (advanced by the worker loop).
    pub epoch: std::sync::atomic::AtomicU64,
    /// Netuid.
    pub netuid: u16,
    /// Eval backend label (`lium` / `sim`) for the status view.
    pub backend_mode: &'static str,
    /// Max orchestrator attempts per submission (retry guard).
    pub retry_max: u32,
    /// Submission gating (1-max). `None` disables intake gating (tests/dev).
    pub gating: Option<Arc<dyn GatingStore>>,
    /// Cached metagraph snapshot for intake membership. `None` disables the
    /// membership check (tests/dev).
    pub metagraph: Option<Arc<MetagraphCache>>,
}

/// Router over the full API surface.
pub fn submission_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/submissions", post(post_submission))
        .route("/v1/submissions", get(list_submissions))
        .route("/v1/submissions/{id}", get(get_submission))
        .route("/v1/submissions/{id}/events", get(get_events))
        .route("/v1/submissions/{id}/retry", post(post_retry))
        .route("/v1/status", get(get_status))
        .route("/v1/jobs", get(get_jobs))
        .route("/v1/recipe", get(get_recipe))
        .route("/v1/recipe/baseline", get(get_recipe_baseline))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({"status":"ok","challenge_id":CHALLENGE_ID})),
    )
}

fn parse_submission_body(
    headers: &axum::http::HeaderMap,
    body: &[u8],
) -> Result<SubmissionRequest, String> {
    let ct = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if ct.contains("application/zip") || ct.contains("application/x-zip-compressed") {
        let hk = headers
            .get("x-miner-hotkey")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| "X-Miner-Hotkey required for application/zip".to_owned())?;
        let (architecture_py, training_py) =
            prism_recipe::sources_from_zip(body).map_err(|e| e.to_string())?;
        return Ok(SubmissionRequest {
            miner_hotkey: hk.to_owned(),
            architecture_py,
            training_py,
            zip_base64: None,
            label: None,
        });
    }
    serde_json::from_slice(body).map_err(|e| format!("invalid_json: {e}"))
}

/// Intake gates for a fresh submission: metagraph membership (fail closed
/// when the cache has no snapshot) + one accepted submission per
/// `(challenge, hotkey)`. Returns the metagraph uid on pass.
async fn intake_gates(st: &AppState, hotkey: &str) -> Result<Option<u32>, Response> {
    let mut uid = None;
    if let Some(cache) = &st.metagraph {
        match cache.snapshot() {
            Some(view) => match view.uid_of_hex(hotkey) {
                Some(u) => uid = Some(u),
                None => {
                    return Err(json_err(
                        StatusCode::FORBIDDEN,
                        "hotkey_not_in_metagraph",
                        "miner hotkey is not registered on this subnet",
                    ));
                }
            },
            None => {
                return Err(json_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "metagraph_unavailable",
                    "metagraph snapshot not ready; retry shortly",
                ));
            }
        }
    }
    gate_one_max(st, hotkey).await?;
    Ok(uid)
}

/// The 1-max gating check alone (metagraph not consulted when no cache is
/// configured, e.g. unit tests).
async fn gate_one_max(st: &AppState, hotkey: &str) -> Result<Option<u32>, Response> {
    if let Some(g) = &st.gating {
        match g.get(CHALLENGE_ID, hotkey).await {
            Ok(Some(row)) if row.state != GatingState::Open => {
                return Err(json_err(
                    StatusCode::CONFLICT,
                    "submission_gated",
                    &format!(
                        "hotkey is '{}' for this challenge; one accepted submission max",
                        row.state.as_str()
                    ),
                ));
            }
            Ok(_) => {}
            Err(e) => {
                return Err(json_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "gating",
                    &e.to_string(),
                ));
            }
        }
    }
    Ok(None)
}

/// POST body: JSON sources, JSON+`zip_base64`, or raw `application/zip`
/// with `X-Miner-Hotkey`.
async fn post_submission(
    State(st): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: bytes::Bytes,
) -> Response {
    let mut req = match parse_submission_body(&headers, body.as_ref()) {
        Ok(r) => r,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "invalid_submission", &e),
    };
    if let Err(e) = prism_pipeline::expand_zip_fields(&mut req) {
        return json_err(StatusCode::BAD_REQUEST, "zip", &e);
    }
    if let Err(e) = prism_pipeline::validate(&req) {
        return map_submission_err(&e);
    }
    if let Err(e) = prism_recipe::check_contract(&req.architecture_py, &req.training_py) {
        return json_err(StatusCode::BAD_REQUEST, "contract", &e.to_string());
    }
    // Normalize hotkey case so gating + ids are case-stable.
    req.miner_hotkey = req.miner_hotkey.trim().to_ascii_lowercase();
    let id = prism_pipeline::submission_id(&req);

    // Idempotent duplicate: identical contract bytes never conflict gating.
    let exists = match st.store.get(&id).await {
        Ok(r) => r.is_some(),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let uid = if exists {
        None
    } else {
        match intake_gates(&st, &req.miner_hotkey).await {
            Ok(u) => u,
            Err(resp) => return resp,
        }
    };
    let epoch = st.epoch.load(std::sync::atomic::Ordering::Relaxed);
    let row = SubmissionState {
        id: id.clone(),
        miner_hotkey: req.miner_hotkey.trim().to_owned(),
        epoch,
        netuid: st.netuid,
        status: Stage::Queued,
        architecture_py: req.architecture_py.clone(),
        training_py: req.training_py.clone(),
        label: req.label.clone(),
        pod_id: None,
        pod_provider: None,
        receipt: None,
        metrics_json: None,
        bpb: None,
        review: None,
        similarity: None,
        final_score: None,
        retry_count: 0,
        error_detail: None,
        created_at_ms: now_ms(),
        updated_at_ms: now_ms(),
    };
    // Idempotent no-op duplicate accepted (same id → 200 OK {status:"already-queued"}).
    match st.store.insert_queued(&row).await {
        Ok(()) => {
            // Registration finalizes only after the row is queued so intake
            // failures never consume the miner's single slot.
            if !exists {
                if let Some(g) = &st.gating {
                    if let Err(e) = g
                        .mark_registered(CHALLENGE_ID, &req.miner_hotkey, uid)
                        .await
                    {
                        return json_err(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "gating",
                            &e.to_string(),
                        );
                    }
                }
            }
            (
                StatusCode::ACCEPTED,
                Json(prism_pipeline::SubmissionAccepted {
                    submission_id: id,
                    status: "accepted".into(),
                }),
            )
                .into_response()
        }
        Err(StoreError::Backend(e)) if e.contains("duplicate") || e.contains("unique") => (
            StatusCode::OK,
            Json(json!({"submission_id": id, "status": "already-queued"})),
        )
            .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    status: Option<String>,
    /// Miner hotkey filter (case-insensitive).
    miner: Option<String>,
    limit: Option<u32>,
}

async fn list_submissions(State(st): State<Arc<AppState>>, Query(q): Query<ListQuery>) -> Response {
    let limit = q.limit.unwrap_or(50).min(500);
    let miner = q.miner.as_deref().map(str::trim).filter(|s| !s.is_empty());
    match st.store.list(q.status.as_deref(), miner, limit).await {
        Ok(rows) => Json(json!({
            "submissions": rows.iter().map(list_view).collect::<Vec<_>>()
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn get_submission(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.store.get(&id).await {
        Ok(Some(row)) => {
            let events = st.store.events(&id).await.unwrap_or_default();
            Json(json!({
                "submission": detail_view(&row),
                "events": events,
            }))
            .into_response()
        }
        Ok(None) => json_err(StatusCode::NOT_FOUND, "not_found", "submission not found"),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn get_events(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.store.events(&id).await {
        Ok(evs) => Json(json!({ "events": evs })).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

/// `POST /v1/submissions/{id}/retry` — requeue a failed row (guard: max attempts).
async fn post_retry(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let row = match st.store.get(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "unknown_submission", &id),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    if row.status != Stage::Failed {
        return json_err(
            StatusCode::CONFLICT,
            "not_failed",
            &format!("status={}", row.status.as_str()),
        );
    }
    if row.retry_count >= st.retry_max {
        return json_err(
            StatusCode::CONFLICT,
            "retry_exhausted",
            &format!("retry_count={} max={}", row.retry_count, st.retry_max),
        );
    }
    match st.store.reset_for_retry(&id).await {
        Ok(_row) => (
            StatusCode::ACCEPTED,
            Json(json!({"submission_id": id, "status": "queued"})),
        )
            .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn get_status(State(st): State<Arc<AppState>>) -> Response {
    let queued = st
        .store
        .list(Some("queued"), None, 1000)
        .await
        .map_or(0, |v| v.len());
    let active = st
        .store
        .list(Some("running"), None, 1000)
        .await
        .map_or(0, |v| v.len())
        + st.store
            .list(Some("provisioning"), None, 1000)
            .await
            .map_or(0, |v| v.len());
    let done_24h = st
        .store
        .list(None, None, 200)
        .await
        .map_or(0, |v| v.iter().filter(|r| r.status.is_terminal()).count());
    Json(json!({
        "challenge_id": CHALLENGE_ID,
        "backend": st.backend_mode,
        "netuid": st.netuid,
        "epoch": st.epoch.load(std::sync::atomic::Ordering::Relaxed),
        "queues": {"queued": queued, "active": active},
        "recent_terminal": done_24h,
        "recipe_pin": prism_recipe::recipe_pin_hex(),
    }))
    .into_response()
}

/// Orchestrator job view: one row per active/recent pod (for ops).
async fn get_jobs(State(st): State<Arc<AppState>>) -> Response {
    let rows = match st.store.list(None, None, 200).await {
        Ok(v) => v,
        Err(e) => {
            return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string());
        }
    };
    let actionable: Vec<Value> = rows
        .iter()
        .filter(|r| !r.status.is_terminal() || r.pod_id.is_some())
        .take(200)
        .map(|r| {
            json!({
                "submission_id": r.id,
                "miner_hotkey": r.miner_hotkey,
                "status": r.status.as_str(),
                "pod_id": r.pod_id,
                "bpb": r.bpb,
                "retry_count": r.retry_count,
            })
        })
        .collect();
    Json(json!({"jobs": actionable})).into_response()
}

async fn get_recipe() -> impl IntoResponse {
    Json(prism_recipe::descriptor())
}

async fn get_recipe_baseline() -> impl IntoResponse {
    Json(json!({
        "architecture_py": BASELINE_ARCHITECTURE_PY,
        "training_py": BASELINE_TRAINING_PY,
    }))
}

/// Feed cache: call this from the worker loop every tick.
pub fn record_epoch(st: &AppState, epoch: u64) {
    st.epoch.store(epoch, std::sync::atomic::Ordering::Relaxed);
}

fn list_view(r: &SubmissionState) -> Value {
    json!({
        "id": r.id,
        "miner_hotkey": r.miner_hotkey,
        "epoch": r.epoch,
        "status": r.status.as_str(),
        "label": r.label,
        "bpb": r.bpb,
        "n_params": r.n_params(),
        "score": r.final_score.as_ref().map(|f| match f {
            FinalScore::Score(v) => json!({"kind":"score","value":v}),
            FinalScore::NoScore(c) => json!({"kind":"no_score","reason":c}),
        }),
        "created_at_ms": r.created_at_ms,
        "updated_at_ms": r.updated_at_ms,
    })
}

fn detail_view(r: &SubmissionState) -> Value {
    let mut v = list_view(r);
    if let Value::Object(m) = &mut v {
        m.insert(
            "architecture_sha256".into(),
            json!(sha256_hex(&r.architecture_py)),
        );
        m.insert("training_sha256".into(), json!(sha256_hex(&r.training_py)));
        m.insert("pod_id".into(), json!(r.pod_id));
        m.insert("pod_provider".into(), json!(r.pod_provider));
        m.insert(
            "receipt".into(),
            r.receipt.as_ref().map_or(Value::Null, |x| json!(x)),
        );
        m.insert("metrics".into(), json!(r.metrics_json));
        m.insert(
            "review".into(),
            r.review.as_ref().map_or(Value::Null, |x| {
                json!({
                    "quality_score": x.quality_score,
                    "issues": x.issues,
                    "prompt_version": x.prompt_version,
                })
            }),
        );
        m.insert(
            "similarity".into(),
            r.similarity.as_ref().map_or(Value::Null, |x| {
                json!({
                    "kind": format!("{:?}", x.kind),
                    "score": x.score,
                    "closest": x.closest,
                    "evidence": x.evidence,
                    "prompt_version": x.prompt_version,
                })
            }),
        );
        m.insert("error_detail".into(), json!(r.error_detail));
    }
    v
}

fn sha256_hex(s: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(s.as_bytes()))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn map_submission_err(err: &SubmissionError) -> Response {
    json_err(StatusCode::BAD_REQUEST, "invalid_request", &err.to_string())
}

fn json_err(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": message,
            "code": code,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use prism_store::{MemoryPrismStore, StatePatch};
    use tower::ServiceExt;

    fn state() -> Arc<AppState> {
        Arc::new(AppState {
            store: Arc::new(MemoryPrismStore::new()),
            epoch: std::sync::atomic::AtomicU64::new(7),
            netuid: 541,
            backend_mode: "sim",
            retry_max: 2,
            gating: None,
            metagraph: None,
        })
    }

    fn gated_state(
        metagraph_hotkeys: &[[u8; 32]],
    ) -> (Arc<AppState>, Arc<submission_gating::MemoryGatingStore>) {
        let gating = Arc::new(submission_gating::MemoryGatingStore::new());
        let cache = Arc::new(MetagraphCache::new());
        cache.update(
            541,
            &metagraph_hotkeys
                .iter()
                .map(|h| h.to_vec())
                .collect::<Vec<_>>(),
        );
        (
            Arc::new(AppState {
                store: Arc::new(MemoryPrismStore::new()),
                epoch: std::sync::atomic::AtomicU64::new(7),
                netuid: 541,
                backend_mode: "sim",
                retry_max: 2,
                gating: Some(Arc::clone(&gating) as Arc<dyn GatingStore>),
                metagraph: Some(cache),
            }),
            gating,
        )
    }

    #[tokio::test]
    async fn retry_requeues_failed_then_guard_blocks() {
        let st = state();
        let app = submission_router(Arc::clone(&st));
        let id = prism_pipeline::submission_id(&crate::example_valid_request());
        // Seed via POST.
        let body = serde_json::to_vec(&crate::example_valid_request()).unwrap();
        let (_s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(v["submission_id"], id);
        // Force failed.
        st.store
            .apply(
                &id,
                &StatePatch {
                    status: Some(Stage::Failed),
                    ..StatePatch::default()
                },
                None,
            )
            .await
            .unwrap();
        let (s, v) = call(
            app.clone(),
            Request::post(format!("/v1/submissions/{id}/retry"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        // Re-fail and retry again → retry_max=2 blocks the third.
        st.store
            .apply(
                &id,
                &StatePatch {
                    status: Some(Stage::Failed),
                    ..StatePatch::default()
                },
                None,
            )
            .await
            .unwrap();
        let (s, _v) = call(
            app.clone(),
            Request::post(format!("/v1/submissions/{id}/retry"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED);
        st.store
            .apply(
                &id,
                &StatePatch {
                    status: Some(Stage::Failed),
                    ..StatePatch::default()
                },
                None,
            )
            .await
            .unwrap();
        let (s, _v) = call(
            app,
            Request::post(format!("/v1/submissions/{id}/retry"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT);
    }

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        (status, v)
    }

    #[tokio::test]
    async fn list_filters_by_miner() {
        let st = state();
        let app = submission_router(Arc::clone(&st));
        let mut req_a = crate::example_valid_request();
        req_a.miner_hotkey = "aa".repeat(32);
        let mut req_b = crate::example_valid_request();
        // Distinct sources so submission ids differ.
        req_b.architecture_py = format!("{}\n#b", req_b.architecture_py);
        req_b.miner_hotkey = "bb".repeat(32);
        for req in [&req_a, &req_b] {
            let body = serde_json::to_vec(req).unwrap();
            let (s, _) = call(
                app.clone(),
                Request::post("/v1/submissions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await;
            assert_eq!(s, StatusCode::ACCEPTED);
        }
        let miner = "aa".repeat(32);
        let (s, v) = call(
            app,
            Request::get(format!("/v1/submissions?miner={miner}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let rows = v["submissions"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["miner_hotkey"], miner);
    }

    #[tokio::test]
    async fn post_then_get_roundtrip() {
        let st = state();
        let app = submission_router(st);
        let body = serde_json::to_vec(&crate::example_valid_request()).unwrap();
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let id = v["submission_id"].as_str().unwrap().to_owned();

        let (s2, v2) = call(
            app.clone(),
            Request::get(format!("/v1/submissions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s2, StatusCode::OK, "{v2}");
        assert_eq!(v2["submission"]["status"], "queued");
        assert_eq!(v2["submission"]["epoch"], 7);

        let (s3, v3) = call(
            app,
            Request::get("/v1/submissions").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s3, StatusCode::OK);
        assert_eq!(v3["submissions"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn status_and_recipe_recipe_baseline() {
        let st = state();
        let app = submission_router(st);
        let (s, v) = call(
            app.clone(),
            Request::get("/v1/status").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["backend"], "sim");
        let (s, v) = call(
            app.clone(),
            Request::get("/v1/recipe").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["train_hours_cap"], 6.0);
        let (s, v) = call(
            app,
            Request::get("/v1/recipe/baseline")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(v["architecture_py"]
            .as_str()
            .unwrap()
            .contains("build_model"));
        assert!(v["training_py"].as_str().unwrap().contains("def train("));
    }

    #[tokio::test]
    async fn get_unknown_404() {
        let st = state();
        let app = submission_router(st);
        let (s, _) = call(
            app,
            Request::get("/v1/submissions/nope")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn gated_intake_requires_metagraph_membership() {
        // Metagraph has 0x22 but the submission hotkey is 0x11 → 403.
        let (st, _g) = gated_state(&[[0x22; 32]]);
        let app = submission_router(st);
        let body = serde_json::to_vec(&crate::example_valid_request()).unwrap();
        let (s, v) = call(
            app,
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{v}");
        assert_eq!(v["code"], "hotkey_not_in_metagraph");
    }

    #[tokio::test]
    async fn gated_intake_one_max_then_conflict() {
        let (st, gating) = gated_state(&[[0x11; 32]]);
        let app = submission_router(Arc::clone(&st));
        let body = serde_json::to_vec(&crate::example_valid_request()).unwrap();
        let (s, _v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED);
        // Gating row is registered with the metagraph uid.
        let row = gating
            .get("prism", &"11".repeat(32))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.state, submission_gating::GatingState::Registered);
        assert_eq!(row.uid, Some(0));

        // A *different* submission from the same hotkey → 409.
        let mut other = crate::example_valid_request();
        other.architecture_py.push_str("\n# v2\n");
        let body = serde_json::to_vec(&other).unwrap();
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["code"], "submission_gated");

        // The identical re-POST stays idempotent (200 already-queued).
        let body = serde_json::to_vec(&crate::example_valid_request()).unwrap();
        let (s, v) = call(
            app,
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["status"], "already-queued");
    }

    #[tokio::test]
    async fn gated_intake_503_until_first_snapshot() {
        // Cache configured but never refreshed → fail closed with 503.
        let gating = Arc::new(submission_gating::MemoryGatingStore::new());
        let st = Arc::new(AppState {
            store: Arc::new(MemoryPrismStore::new()),
            epoch: std::sync::atomic::AtomicU64::new(7),
            netuid: 541,
            backend_mode: "sim",
            retry_max: 2,
            gating: Some(gating as Arc<dyn GatingStore>),
            metagraph: Some(Arc::new(MetagraphCache::new())),
        });
        let app = submission_router(st);
        let body = serde_json::to_vec(&crate::example_valid_request()).unwrap();
        let (s, v) = call(
            app,
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["code"], "metagraph_unavailable");
    }
}
