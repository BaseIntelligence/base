//! Public PRISM HTTP API (miners + operators).
//!
//! | Route | Purpose |
//! |-------|---------|
//! | `GET  /health` | liveness |
//! | `POST /v1/submissions` | accept a two-script recipe |
//! | `POST /v1/submissions/precheck` | advisory copy-gate (quota 3/coldkey/UTC day) |
//! | `GET  /v1/submissions` | list (`status` / `miner` filter, limit) |
//! | `GET  /v1/submissions/{id}` | full detail + event timeline + `eval` |
//! | `GET  /v1/submissions/{id}/events` | journal only |
//! | `POST /v1/submissions/{id}/attribution` | 2×2 attribution run plans (JSON) |
//! | `POST /v1/submissions/{id}/zone-b` | Zone B self-report intake (mounted by the service bin from `prism-attribution`) |
//! | `GET  /v1/submissions/{id}/metrics?zone=a\|b` | Zone A rows / Zone B chain |
//! | `GET  /v1/anchors` | anchor-set registry with status |
//! | `GET  /v1/preregistration` | anchor pre-registration hash-commits |
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
use prism_eval_store::{anchors_view, eval_detail, prereg_view, zone_a_view, zone_b_view};
use prism_pipeline::{
    ephemeral_candidate, evaluate_copy_precheck, precheck_json, precheck_quota_exceeded_json,
    precheck_skipped, quota_identity, quota_view, utc_day, SubmissionError, SubmissionRequest,
    PRECHECK_DAILY_LIMIT,
};
use prism_store::eval::EvalStore;
use prism_store::{FinalScore, PrismStore, Stage, StoreError, SubmissionState};

/// Shared HTTP app state.
#[derive(Debug)]
pub struct AppState {
    /// Store.
    pub store: Arc<dyn PrismStore>,
    /// Eval store (v3 composite runs, anchor registry, Zone B reports).
    pub eval_store: Arc<dyn EvalStore>,
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
        .route("/v1/submissions/precheck", post(post_precheck))
        .route("/v1/submissions", get(list_submissions))
        .route("/v1/submissions/{id}", get(get_submission))
        .route("/v1/submissions/{id}/events", get(get_events))
        .route("/v1/submissions/{id}/metrics", get(get_submission_metrics))
        // Attribution planner (2×2 matrix run plans): split into the
        // `prism-attribution` crate for the per-crate LOC cap; the route
        // carries its own state (the submission store).
        .route(
            "/v1/submissions/{id}/attribution",
            prism_attribution::attribution_route(Arc::clone(&state.store)),
        )
        .route("/v1/submissions/{id}/retry", post(post_retry))
        .route("/v1/anchors", get(get_anchors))
        .route("/v1/preregistration", get(get_preregistration))
        .route("/v1/status", get(get_status))
        .route("/v1/jobs", get(get_jobs))
        .route("/v1/recipe", get(get_recipe))
        .route("/v1/recipe/baseline", get(get_recipe_baseline))
        .route("/v1/architectures", get(get_architectures))
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
        let arch_id = headers
            .get("x-prism-arch-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let (architecture_py, training_py) = if arch_id.is_some() {
            (
                String::new(),
                prism_recipe::training_from_zip(body).map_err(|e| e.to_string())?,
            )
        } else {
            prism_recipe::sources_from_zip(body).map_err(|e| e.to_string())?
        };
        return Ok(SubmissionRequest {
            miner_hotkey: hk.to_owned(),
            architecture_py,
            training_py,
            zip_base64: None,
            arch_id,
            label: None,
        });
    }
    serde_json::from_slice(body).map_err(|e| format!("invalid_json: {e}"))
}

/// Metagraph membership only (fail closed when configured but empty).
#[allow(clippy::result_large_err)] // mirrors other intake helpers returning `Response`
fn metagraph_uid(st: &AppState, hotkey: &str) -> Result<Option<u32>, Response> {
    let Some(cache) = &st.metagraph else {
        return Ok(None);
    };
    match cache.snapshot() {
        Some(view) => match view.uid_of_hex(hotkey) {
            Some(u) => Ok(Some(u)),
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

/// Intake gates: metagraph membership + one accepted submission per
/// `(challenge, hotkey)`. Returns the metagraph uid on pass.
async fn intake_gates(
    st: &AppState,
    hotkey: &str,
    challenge: &str,
) -> Result<Option<u32>, Response> {
    let uid = metagraph_uid(st, hotkey)?;
    gate_one_max(st, hotkey, challenge).await?;
    Ok(uid)
}

/// The 1-max gating check alone (metagraph not consulted when no cache is
/// configured, e.g. unit tests).
async fn gate_one_max(
    st: &AppState,
    hotkey: &str,
    challenge: &str,
) -> Result<Option<u32>, Response> {
    if let Some(g) = &st.gating {
        match g.get(challenge, hotkey).await {
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

/// Materialize a training-only request's architecture from the registry
/// (`Ok(())` for architecture submissions — nothing to pull).
async fn materialize_arch(st: &AppState, req: &mut SubmissionRequest) -> Result<(), Response> {
    let Some(arch_id) = req
        .arch_id
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
    else {
        return Ok(());
    };
    match st.store.get_arch(&arch_id).await {
        Ok(Some(rec)) => {
            req.architecture_py = rec.architecture_py;
            req.arch_id = Some(arch_id);
            Ok(())
        }
        Ok(None) => Err(json_err(
            StatusCode::NOT_FOUND,
            "unknown_arch",
            "arch_id is not in the published architecture registry",
        )),
        Err(e) => Err(json_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "store",
            &e.to_string(),
        )),
    }
}

/// `POST /v1/submissions/precheck` — advisory copy-gate (same logic as
/// intake), no submission row, no 1-max gate, no Lium. Quota: 3/coldkey/UTC day.
async fn post_precheck(
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
    if let Err(resp) = materialize_arch(&st, &mut req).await {
        return resp;
    }
    if let Err(e) = prism_recipe::check_contract(&req.architecture_py, &req.training_py) {
        return json_err(StatusCode::BAD_REQUEST, "contract", &e.to_string());
    }
    req.miner_hotkey = req.miner_hotkey.trim().to_ascii_lowercase();
    if let Err(resp) = metagraph_uid(&st, &req.miner_hotkey) {
        return resp;
    }
    let miner_coldkey = st
        .metagraph
        .as_ref()
        .and_then(|c| c.snapshot())
        .and_then(|v| v.coldkey_hex_of(&req.miner_hotkey));
    let (identity, identity_kind) = quota_identity(&req.miner_hotkey, miner_coldkey.as_deref());
    let day = utc_day(now_ms() / 1000);
    let used = match st
        .store
        .precheck_quota_try_consume(&identity, &day, PRECHECK_DAILY_LIMIT)
        .await
    {
        Ok(Some(n)) => n,
        Ok(None) => {
            let used = st
                .store
                .precheck_quota_get(&identity, &day)
                .await
                .unwrap_or(PRECHECK_DAILY_LIMIT);
            let q = quota_view(day, used, identity_kind);
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(precheck_quota_exceeded_json(&q)),
            )
                .into_response();
        }
        Err(e) => {
            return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string());
        }
    };
    let quota = quota_view(day, used, identity_kind);
    if req.arch_id.is_some() {
        return Json(precheck_json(&precheck_skipped(quota))).into_response();
    }
    let candidate = ephemeral_candidate(
        &req.miner_hotkey,
        miner_coldkey,
        &req.architecture_py,
        now_ms(),
    );
    let recent = st.store.list_champions(64).await.unwrap_or_default();
    let result = evaluate_copy_precheck(&candidate, &recent, quota);
    Json(precheck_json(&result)).into_response()
}

/// Queued row for an accepted submission. The coldkey comes from the live
/// metagraph snapshot (None off-chain), so quota / corpus dedup can key on the
/// owner rather than a rotatable hotkey.
fn queued_row(
    st: &AppState,
    req: &SubmissionRequest,
    id: String,
    tree_blob: Option<Vec<u8>>,
) -> SubmissionState {
    let miner_hotkey = req.miner_hotkey.trim().to_owned();
    let miner_coldkey = st
        .metagraph
        .as_ref()
        .and_then(|c| c.snapshot())
        .and_then(|v| v.coldkey_hex_of(&miner_hotkey));
    SubmissionState {
        id,
        miner_hotkey,
        miner_coldkey,
        epoch: st.epoch.load(std::sync::atomic::Ordering::Relaxed),
        netuid: st.netuid,
        status: Stage::Queued,
        architecture_py: req.architecture_py.clone(),
        training_py: req.training_py.clone(),
        tree_blob,
        label: req.label.clone(),
        pod_id: None,
        pod_provider: None,
        receipt: None,
        metrics_json: None,
        bpb: None,
        arch_id: req.arch_id.clone(),
        review: None,
        similarity: None,
        final_score: None,
        retry_count: 0,
        error_detail: None,
        created_at_ms: now_ms(),
        updated_at_ms: now_ms(),
    }
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
    // Training-only: pull the architecture from the registry (miner-sent
    // source is rejected by validate above, so the registry is the only
    // source of truth — this is what makes the pre-LLM copy gate safe to
    // skip on these rows).
    if let Err(resp) = materialize_arch(&st, &mut req).await {
        return resp;
    }
    if let Err(e) = prism_recipe::check_contract(&req.architecture_py, &req.training_py) {
        return json_err(StatusCode::BAD_REQUEST, "contract", &e.to_string());
    }
    // Normalize hotkey case so gating + ids are case-stable.
    req.miner_hotkey = req.miner_hotkey.trim().to_ascii_lowercase();
    let id = prism_pipeline::submission_id(&req);
    let gate_challenge = prism_pipeline::gating_key(req.arch_id.as_deref());

    // Idempotent duplicate: identical contract bytes never conflict gating.
    let exists = match st.store.get(&id).await {
        Ok(r) => r.is_some(),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let uid = if exists {
        None
    } else {
        match intake_gates(&st, &req.miner_hotkey, &gate_challenge).await {
            Ok(u) => u,
            Err(resp) => return resp,
        }
    };
    let tree_blob = match prism_pipeline::tree_blob_for(&req) {
        Ok(b) => b,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "tree", &e),
    };
    let row = queued_row(&st, &req, id.clone(), tree_blob);
    // Idempotent no-op duplicate accepted (same id → 200 OK {status:"already-queued"}).
    match st.store.insert_queued(&row).await {
        Ok(()) => {
            // Registration finalizes only after the row is queued so intake
            // failures never consume the miner's single slot.
            if !exists {
                if let Some(g) = &st.gating {
                    if let Err(e) = g
                        .mark_registered(&gate_challenge, &req.miner_hotkey, uid)
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
            let mut detail = detail_view(&row);
            if let Value::Object(m) = &mut detail {
                m.insert("eval".into(), eval_json(&st, &id).await);
            }
            Json(json!({
                "submission": detail,
                "events": events,
            }))
            .into_response()
        }
        Ok(None) => json_err(StatusCode::NOT_FOUND, "not_found", "submission not found"),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

/// The `eval` field for the detail view: the stored composite outcome +
/// run provenance, `null` when the run has no finalized eval (v2 path).
async fn eval_json(st: &AppState, id: &str) -> Value {
    let Ok(Some(run)) = st.eval_store.eval_run(id).await else {
        return Value::Null;
    };
    let groups = st
        .eval_store
        .eval_groups(&run.run_id)
        .await
        .unwrap_or_default();
    eval_detail(&run, &groups)
}

#[derive(Debug, Deserialize)]
struct MetricsQuery {
    /// `a` (organizer-measured `org.*` rows) | `b` (participant-reported
    /// chain); default `a`.
    zone: Option<String>,
}

/// `GET /v1/submissions/{id}/metrics?zone=a|b` — raw metric rows. Zone B is
/// always labelled participant-reported and never feeds scoring.
async fn get_submission_metrics(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<MetricsQuery>,
) -> Response {
    match st.store.get(&id).await {
        Ok(Some(_)) => {}
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "not_found", "submission not found"),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
    match q.zone.as_deref().unwrap_or("a") {
        "a" => {
            let run = st.eval_store.eval_run(&id).await;
            let (metrics, mirrors) = match run {
                Ok(Some(r)) => (
                    st.eval_store
                        .eval_metrics(&r.run_id)
                        .await
                        .unwrap_or_default(),
                    st.eval_store
                        .eval_mirrors(&r.run_id)
                        .await
                        .unwrap_or_default(),
                ),
                _ => (Vec::new(), Vec::new()),
            };
            Json(zone_a_view(&metrics, &mirrors)).into_response()
        }
        "b" => match st.eval_store.metric_reports(&id).await {
            Ok(reports) => Json(zone_b_view(&reports)).into_response(),
            Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
        },
        other => json_err(
            StatusCode::BAD_REQUEST,
            "invalid_zone",
            &format!("zone must be 'a' or 'b', got '{other}'"),
        ),
    }
}

/// `GET /v1/anchors` — every known anchor set with status.
async fn get_anchors(State(st): State<Arc<AppState>>) -> Response {
    match st.eval_store.anchor_sets().await {
        Ok(sets) => Json(anchors_view(&sets)).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

/// `GET /v1/preregistration` — anchor pre-registration hash-commits.
async fn get_preregistration(State(st): State<Arc<AppState>>) -> Response {
    match st.eval_store.preregistrations().await {
        Ok(entries) => Json(prereg_view(&entries)).into_response(),
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

/// `GET /v1/architectures` — published architecture registry (leaderboard
/// source: per-arch best bpb across all trainers).
async fn get_architectures(State(st): State<Arc<AppState>>) -> Response {
    match st.store.list_archs(200).await {
        Ok(rows) => Json(json!({
            "architectures": rows.iter().map(|a| json!({
                "arch_id": a.arch_id,
                "owner_hotkey": a.owner_hotkey,
                "arch_digest": a.arch_digest,
                "source_submission": a.source_submission,
                "best_bpb": a.best_bpb,
                "created_at_ms": a.created_at_ms,
            })).collect::<Vec<_>>()
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
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
        "arch_id": r.arch_id,
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
            eval_store: Arc::new(crate::MemoryEvalStore::new()),
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
        let keys: Vec<Vec<u8>> = metagraph_hotkeys.iter().map(|h| h.to_vec()).collect();
        // Tests that do not care about shared coldkeys: each hotkey owns itself.
        cache.update(541, &keys, &keys);
        (
            Arc::new(AppState {
                store: Arc::new(MemoryPrismStore::new()),
                eval_store: Arc::new(crate::MemoryEvalStore::new()),
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

    fn arch_fixture() -> (String, String) {
        let arch_src =
            "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(16, 16)\n".to_owned();
        let arch = prism_pipeline::arch_id_for(&arch_src);
        (arch, arch_src)
    }

    async fn seed_arch(st: &AppState, arch: &str, arch_src: &str) {
        st.store
            .publish_arch(&prism_store::ArchitectureRecord {
                arch_id: arch.to_owned(),
                owner_hotkey: "aa".repeat(32),
                arch_digest: prism_pipeline::arch_digest(arch_src),
                architecture_py: arch_src.to_owned(),
                source_submission: "src".into(),
                best_bpb: Some(2.0),
                created_at_ms: 1,
            })
            .await
            .unwrap();
    }

    fn training_body(hotkey: &str, train: &str, arch: &str) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "miner_hotkey": hotkey,
            "training_py": train,
            "arch_id": arch,
        }))
        .unwrap()
    }

    const TRAIN_HOOKED: &str = concat!(
        "import prism_telemetry\n",
        "def train(model, ctx):\n",
        "    prism_telemetry.report(loss=1.0, step=1)\n",
        "    prism_telemetry.finish_evaluation()\n",
        "    return {'loss': 1.0}\n",
    );

    #[tokio::test]
    async fn training_only_intake_materializes_arch_and_gates_per_arch() {
        let hk = [0x11; 32];
        let (st, gating) = gated_state(&[hk]);
        let (arch, arch_src) = arch_fixture();
        seed_arch(&st, &arch, &arch_src).await;
        let app = submission_router(Arc::clone(&st));
        let hotkey_hex = hex::encode(hk);

        // Accept: 202, materialized source, composite gating key registered.
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(training_body(&hotkey_hex, TRAIN_HOOKED, &arch)))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let id = v["submission_id"].as_str().unwrap().to_owned();
        let row = st.store.get(&id).await.unwrap().expect("row");
        assert_eq!(row.architecture_py, arch_src, "arch pulled from registry");
        assert_eq!(row.arch_id.as_deref(), Some(arch.as_str()));
        let gate_key = prism_pipeline::gating_key(Some(&arch));
        let g = gating
            .get(&gate_key, &hotkey_hex)
            .await
            .unwrap()
            .expect("gating row");
        assert_eq!(g.state, GatingState::Registered);
        // The plain 1-max architecture gate is untouched.
        assert!(gating
            .get(CHALLENGE_ID, &hotkey_hex)
            .await
            .unwrap()
            .is_none());

        // Idempotent: identical bytes → already-queued, no gate conflict.
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(training_body(&hotkey_hex, TRAIN_HOOKED, &arch)))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["status"], "already-queued");

        // Different training on the same (hotkey, arch) → gated 409.
        let other_train = TRAIN_HOOKED.replace("loss=1.0", "loss=0.9");
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(training_body(&hotkey_hex, &other_train, &arch)))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["code"], "submission_gated");

        // A different published arch has an independent gate slot.
        let second_src =
            "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n".to_owned();
        let second = prism_pipeline::arch_id_for(&second_src);
        seed_arch(&st, &second, &second_src).await;
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(training_body(
                    &hotkey_hex,
                    &other_train,
                    &second,
                )))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    }

    #[tokio::test]
    async fn training_only_unknown_arch_404_and_source_rejected() {
        let hk = [0x22; 32];
        let (st, _g) = gated_state(&[hk]);
        let (arch, _src) = arch_fixture();
        let app = submission_router(Arc::clone(&st));
        let hotkey_hex = hex::encode(hk);

        // Unknown registry id → 404.
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(training_body(&hotkey_hex, TRAIN_HOOKED, &arch)))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
        assert_eq!(v["code"], "unknown_arch");

        // arch_id + inline source → 400 (registry is the only arch source).
        let body = serde_json::to_vec(&json!({
            "miner_hotkey": hotkey_hex,
            "architecture_py": "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n",
            "training_py": TRAIN_HOOKED,
            "arch_id": arch,
        }))
        .unwrap();
        let (s, v) = call(
            app,
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
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
    async fn precheck_detects_copy_without_queuing() {
        let st = state();
        let victim_hk = "aa".repeat(32);
        let mut victim = crate::example_valid_request();
        victim.miner_hotkey = victim_hk;
        let vid = prism_pipeline::submission_id(&victim);
        let mut row = SubmissionState {
            id: vid,
            miner_hotkey: victim.miner_hotkey.clone(),
            miner_coldkey: Some("11".repeat(32)),
            epoch: 1,
            netuid: 541,
            status: Stage::Terminated,
            architecture_py: victim.architecture_py.clone(),
            training_py: victim.training_py.clone(),
            tree_blob: None,
            label: None,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: None,
            bpb: Some(1.0),
            arch_id: None,
            review: None,
            similarity: None,
            final_score: Some(FinalScore::Score(1)),
            retry_count: 0,
            error_detail: None,
            created_at_ms: 1_000,
            updated_at_ms: 1_000,
        };
        // Ensure created_at is in the past relative to precheck `now_ms`.
        row.created_at_ms = 1;
        st.store.insert_queued(&row).await.unwrap();
        // Force terminated without going through claim (memory insert is queued).
        st.store
            .apply(
                &row.id,
                &StatePatch {
                    status: Some(Stage::Terminated),
                    final_score: Some(FinalScore::Score(1)),
                    ..StatePatch::default()
                },
                None,
            )
            .await
            .unwrap();

        let app = submission_router(Arc::clone(&st));
        let mut copy = crate::example_valid_request();
        copy.miner_hotkey = "bb".repeat(32);
        copy.architecture_py = victim.architecture_py;
        let body = serde_json::to_vec(&copy).unwrap();
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions/precheck")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["similar"], true);
        assert_eq!(v["verdict"], "copied");
        assert!(v["matched_against"].as_str().unwrap().starts_with("subm:"));
        assert_eq!(v["quota"]["used"], 1);
        assert_eq!(v["quota"]["remaining"], 2);
        // No new submission row for the copier.
        let listed = st.store.list(None, None, 50).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].miner_hotkey, "aa".repeat(32));
    }

    #[tokio::test]
    async fn precheck_quota_is_three_then_429() {
        let st = state();
        let app = submission_router(Arc::clone(&st));
        let body = serde_json::to_vec(&crate::example_valid_request()).unwrap();
        for i in 1..=3 {
            let (s, v) = call(
                app.clone(),
                Request::post("/v1/submissions/precheck")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "attempt {i}: {v}");
            assert_eq!(v["similar"], false);
            assert_eq!(v["verdict"], "clean");
            assert_eq!(v["quota"]["used"], i);
        }
        let (s, v) = call(
            app,
            Request::post("/v1/submissions/precheck")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{v}");
        assert_eq!(v["code"], "precheck_quota_exceeded");
        assert_eq!(v["quota"]["remaining"], 0);
        assert_eq!(v["quota"]["used"], 3);
    }

    #[tokio::test]
    async fn gated_intake_503_until_first_snapshot() {
        // Cache configured but never refreshed → fail closed with 503.
        let gating = Arc::new(submission_gating::MemoryGatingStore::new());
        let st = Arc::new(AppState {
            store: Arc::new(MemoryPrismStore::new()),
            eval_store: Arc::new(crate::MemoryEvalStore::new()),
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

    async fn post_one(app: &Router) -> String {
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
        v["submission_id"].as_str().unwrap().to_owned()
    }

    fn eval_blob() -> Value {
        json!({
            "bpb": 1.0,
            "tokens_seen": 1_000_000_000_u64,
            "tokens_seen_source": "train_stream",
            "wall_clock_seconds": 3600.0,
            "gpu_type": "H100",
            "n_params": 125_000_000_u64,
            "recipe": "1.3.0",
            "train_metrics": { "miner.train.loss": 0.7 },
            "battery": { "metrics": { "org.g1.bpb_code": 1.0 } },
        })
    }

    #[tokio::test]
    async fn detail_eval_null_then_populated() {
        let st = state();
        let app = submission_router(Arc::clone(&st));
        let id = post_one(&app).await;

        let (s, v) = call(
            app.clone(),
            Request::get(format!("/v1/submissions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["submission"]["eval"], Value::Null, "no eval pre-finalize");

        prism_eval_store::finalize_composite(
            &st.eval_store,
            &id,
            &eval_blob(),
            &prism_eval_store::AnchorInput::v0_placeholder(),
        )
        .await
        .unwrap();

        let (s, v) = call(
            app.clone(),
            Request::get(format!("/v1/submissions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let eval = &v["submission"]["eval"];
        assert_eq!(eval["scoring_mode"], "shadow");
        assert_eq!(
            eval["status"], "ineligible",
            "partial battery is ineligible"
        );
        assert_eq!(eval["groups"].as_array().unwrap().len(), 8);
        assert_eq!(eval["anchor_version"], 0);
        assert_eq!(eval["prereg_hash"].as_str().unwrap().len(), 64);
    }

    #[tokio::test]
    async fn metrics_zone_a_and_b_and_bad_zone() {
        let st = state();
        let app = submission_router(Arc::clone(&st));
        let id = post_one(&app).await;
        prism_eval_store::finalize_composite(
            &st.eval_store,
            &id,
            &eval_blob(),
            &prism_eval_store::AnchorInput::v0_placeholder(),
        )
        .await
        .unwrap();

        let (s, v) = call(
            app.clone(),
            Request::get(format!("/v1/submissions/{id}/metrics?zone=a"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["provenance"], "organizer-measured");
        assert_eq!(v["metrics"][0]["key"], "org.g1.bpb_code");

        let (s, v) = call(
            app.clone(),
            Request::get(format!("/v1/submissions/{id}/metrics?zone=b"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["provenance"], "participant-reported");
        assert_eq!(v["reports"][0]["verdict"], "ok");
        assert_eq!(v["reports"][0]["seq"], 0);

        let (s, v) = call(
            app.clone(),
            Request::get(format!("/v1/submissions/{id}/metrics?zone=c"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(v["code"], "invalid_zone");

        let (s, _) = call(
            app,
            Request::get("/v1/submissions/nope/metrics?zone=a")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn anchors_and_preregistration_views() {
        let st = state();
        let app = submission_router(Arc::clone(&st));
        let id = post_one(&app).await;
        prism_eval_store::finalize_composite(
            &st.eval_store,
            &id,
            &eval_blob(),
            &prism_eval_store::AnchorInput::v0_placeholder(),
        )
        .await
        .unwrap();

        let (s, v) = call(
            app.clone(),
            Request::get("/v1/anchors").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let anchors = v["anchors"].as_array().unwrap();
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0]["version"], 0);
        assert_eq!(anchors[0]["status"], "placeholder");

        let (s, v) = call(
            app,
            Request::get("/v1/preregistration")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let regs = v["preregistrations"].as_array().unwrap();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0]["hash"], anchors[0]["prereg_hash"]);
    }
}
