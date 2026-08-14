//! Public PRISM HTTP API (miners + operators).
//!
//! Routes: `/health`, `/v1/submissions` (+ precheck/retry/events/logs/diff/
//! metrics/inference/attribution/zone-b), `/v1/anchors`, `/v1/preregistration`,
//! `/v1/status`, `/v1/jobs`, `/v1/recipe` (+ baseline), admin playground/artifacts.
//! Acceptance uses the cached chain epoch (never blocks on RPC).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::middleware::from_fn_with_state;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use submission_gating::{infra_resubmit_allowed, GatingState, GatingStore, MetagraphCache};

use prism_recipe::{BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY};

use crate::CHALLENGE_ID;
use prism_eval_store::{detail_view, eval_json, list_view};
use prism_intake::{
    coldkey_of, json_err, map_submission_err, map_zip_err, materialize_arch, metagraph_uid, now_ms,
    parse_submission_body, require_bearer, verify_bearer, BearerGate,
};
use prism_pipeline::{is_automodel_request, SubmissionRequest};
use prism_store::eval::EvalStore;
use prism_store::{FinalScore, PrismStore, Stage, StoreError, SubmissionState};

/// Shared HTTP app state.
#[derive(Debug)]
pub struct AppState {
    pub store: Arc<dyn PrismStore>,
    /// Eval store (v3 composite runs, anchor registry, Zone B reports).
    pub eval_store: Arc<dyn EvalStore>,
    /// Current chain epoch cache (advanced by the worker loop).
    pub epoch: std::sync::atomic::AtomicU64,
    pub netuid: u16,
    pub backend_mode: &'static str,
    pub retry_max: u32,
    pub gating: Option<Arc<dyn GatingStore>>,
    pub metagraph: Option<Arc<MetagraphCache>>,
    /// Operator bearer hashes. Empty → retry/admin/playground answer 503.
    pub admin_token_hashes: Vec<String>,
    /// Miner Lium API keys (BYOK). `None` → Sim / operator-only billing.
    pub payer_vault: Option<std::sync::Arc<prism_lium_payer::PayerKeyVault>>,
    /// Live harness log tails (shared with orchestrator).
    pub logs: std::sync::Arc<prism_orphan::LogBuffer>,
}

pub fn submission_router(state: Arc<AppState>) -> Router {
    let admin: BearerGate = (state.admin_token_hashes.clone().into(), "admin");
    if admin.0.is_empty() {
        tracing::warn!(
            event = "prism_admin_auth_unconfigured",
            "no admin bearer tokens: manual retry + playground + artifacts answer 503"
        );
    }
    let operator = Router::new()
        .route(
            "/v1/admin/playground/complete",
            prism_playground::playground_route(prism_playground::PlaygroundState {
                store: Arc::clone(&state.store),
                infer_script: prism_playground::default_infer_script(),
            }),
        )
        .route(
            "/v1/admin/artifacts/{submission_id}/receive",
            prism_artifacts::artifact_receive_route(prism_artifacts::ArtifactReceiveState {
                store: Arc::clone(&state.store),
            }),
        )
        .route(
            "/v1/admin/artifacts/{submission_id}",
            prism_artifacts::artifact_get_route(),
        )
        .route("/v1/admin/gating/{hotkey}/reset", post(admin_reset_gating))
        .route_layer(from_fn_with_state(admin, require_bearer));
    Router::new()
        .route("/health", get(health))
        .route("/v1/submissions", post(post_submission))
        // Advisory copy-gate: shares the intake front-end but writes no row,
        // so it lives in `prism-intake` with its own narrow state.
        .route(
            "/v1/submissions/precheck",
            prism_intake::precheck_route(Arc::clone(&state.store), state.metagraph.clone()),
        )
        .route("/v1/submissions", get(list_submissions))
        .route("/v1/submissions/{id}", get(get_submission))
        .route(
            "/v1/submissions/{id}/diff",
            prism_attribution::diff_route(Arc::clone(&state.store)),
        )
        .route("/v1/submissions/{id}/events", get(get_events))
        .route("/v1/submissions/{id}/logs", get(get_logs))
        .route("/v1/submissions/{id}/retry", post(post_retry))
        // Attribution planner (2×2 matrix run plans) and the v3 read-only
        // eval surface: split into the `prism-attribution` crate for the
        // per-crate LOC cap; each route carries its own narrow state.
        .route(
            "/v1/submissions/{id}/metrics",
            prism_attribution::metrics_route(
                Arc::clone(&state.store),
                Arc::clone(&state.eval_store),
            ),
        )
        .route(
            "/v1/submissions/{id}/inference",
            prism_attribution::inference_route(Arc::clone(&state.store)),
        )
        .route(
            "/v1/submissions/{id}/attribution",
            prism_attribution::attribution_route(Arc::clone(&state.store)),
        )
        .route(
            "/v1/anchors",
            prism_attribution::anchors_route(Arc::clone(&state.eval_store)),
        )
        .route(
            "/v1/preregistration",
            prism_attribution::prereg_route(Arc::clone(&state.eval_store)),
        )
        .route("/v1/status", get(get_status))
        .route("/v1/jobs", get(get_jobs))
        .route("/v1/recipe", get(get_recipe))
        .route("/v1/recipe/baseline", get(get_recipe_baseline))
        .route("/v1/architectures", get(get_architectures))
        .merge(operator)
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({"status":"ok","challenge_id":CHALLENGE_ID})),
    )
}

/// Intake gates: metagraph membership + one accepted submission per
/// `(challenge, hotkey)`. Returns the metagraph uid on pass.
async fn intake_gates(
    st: &AppState,
    hotkey: &str,
    challenge: &str,
) -> Result<Option<u32>, Response> {
    let uid = metagraph_uid(st.metagraph.as_deref(), hotkey)?;
    gate_one_max(st, hotkey, challenge).await?;
    Ok(uid)
}

async fn gate_one_max(
    st: &AppState,
    hotkey: &str,
    challenge: &str,
) -> Result<Option<u32>, Response> {
    if let Some(g) = &st.gating {
        match g.get(challenge, hotkey).await {
            Ok(Some(row))
                if row.state != GatingState::Open && !infra_resubmit_allowed(&row, now_ms()) =>
            {
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

/// [`prism_pipeline::queued_row`] with the coldkey resolved from the live
/// metagraph snapshot (`None` off-chain).
fn queued_row(
    st: &AppState,
    req: &SubmissionRequest,
    id: String,
    tree_blob: Option<Vec<u8>>,
) -> SubmissionState {
    let coldkey = coldkey_of(st.metagraph.as_deref(), req.miner_hotkey.trim());
    prism_pipeline::queued_row(
        req,
        id,
        coldkey,
        tree_blob,
        st.epoch.load(std::sync::atomic::Ordering::Relaxed),
        st.netuid,
        now_ms(),
    )
}

/// POST body: JSON sources, JSON+`zip_base64`, or raw `application/zip`
/// with `X-Miner-Hotkey`.
async fn post_submission(
    State(st): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: bytes::Bytes,
) -> Response {
    let miner_lium_key = headers
        .get(prism_lium_payer::LIUM_API_KEY_HEADER)
        .or_else(|| headers.get("X-Lium-Api-Key"))
        .and_then(|v| v.to_str().ok())
        .and_then(prism_lium_payer::normalize_lium_api_key);
    let mut req = match parse_submission_body(&headers, body.as_ref()) {
        Ok(r) => r,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "invalid_submission", &e),
    };
    if let Err(e) = prism_pipeline::expand_zip_fields(&mut req) {
        return map_zip_err(&e);
    }
    if let Err(e) = prism_pipeline::validate(&req) {
        return map_submission_err(&e);
    }
    // Training-only (legacy): pull architecture from the registry. AutoModel
    // rows skip the 1.x contract check — review uses GET …/diff.
    if !is_automodel_request(&req) {
        if let Err(resp) = materialize_arch(st.store.as_ref(), &mut req).await {
            return resp;
        }
        if let Err(e) = prism_recipe::check_contract(&req.architecture_py, &req.training_py) {
            return json_err(StatusCode::BAD_REQUEST, "contract", &e.to_string());
        }
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
    // Live Lium: miners fund their own pod via X-Lium-Api-Key (not persisted).
    if prism_lium_payer::require_miner_lium(st.backend_mode) && miner_lium_key.is_none() && !exists
    {
        return json_err(
            StatusCode::BAD_REQUEST,
            "missing_lium_api_key",
            "live Prism eval requires header X-Lium-Api-Key (miner-funded Lium account)",
        );
    }
    let row = queued_row(&st, &req, id.clone(), tree_blob);
    // Idempotent no-op duplicate accepted (same id → 200 OK {status:"already-queued"}).
    match st.store.insert_queued(&row).await {
        Ok(()) => {
            if let (Some(vault), Some(key)) = (&st.payer_vault, miner_lium_key.as_ref()) {
                vault.insert(id.clone(), key.clone());
            }
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
        Err(StoreError::Backend(e)) if e.contains("duplicate") || e.contains("unique") => {
            if let (Some(vault), Some(key)) = (&st.payer_vault, miner_lium_key.as_ref()) {
                vault.insert(id.clone(), key.clone());
            }
            (
                StatusCode::OK,
                Json(json!({"submission_id": id, "status": "already-queued"})),
            )
                .into_response()
        }
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
                m.insert("eval".into(), eval_json(st.eval_store.as_ref(), &id).await);
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

async fn get_events(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.store.events(&id).await {
        Ok(evs) => Json(json!({ "events": evs })).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct LogsQuery {
    since: Option<u64>,
}

async fn get_logs(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<LogsQuery>,
) -> Response {
    if st.store.get(&id).await.ok().flatten().is_none() {
        return json_err(StatusCode::NOT_FOUND, "not_found", "submission not found");
    }
    let body = prism_orphan::logs_json(
        Some(st.logs.as_ref()),
        st.store.as_ref(),
        &id,
        q.since.unwrap_or(0),
    )
    .await;
    Json(body).into_response()
}

/// `POST /v1/admin/gating/{hotkey}/reset` — clear 1-max open slot (operator).
async fn admin_reset_gating(
    State(st): State<Arc<AppState>>,
    Path(hotkey): Path<String>,
) -> Response {
    let Some(gating) = &st.gating else {
        return json_err(
            StatusCode::SERVICE_UNAVAILABLE,
            "gating_disabled",
            "submission gating is not enabled",
        );
    };
    match gating.reset_open(CHALLENGE_ID, &hotkey).await {
        Ok(()) => Json(json!({"ok": true, "hotkey": hotkey})).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "gating", &e.to_string()),
    }
}

/// `POST /v1/submissions/{id}/retry` — requeue a failed row (guard: max attempts).
async fn post_retry(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: axum::http::HeaderMap,
) -> Response {
    let row = match st.store.get(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "unknown_submission", &id),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    if row.status != Stage::Failed {
        return json_err(
            StatusCode::CONFLICT,
            "not_failed",
            &format!(
                "status={} — /retry only accepts failed rows; for miner infra retry send X-Lium-Api-Key (and the usual X-Miner-Hotkey / body hotkey). Admin Bearer is for operator retries of non-infra failures",
                row.status.as_str()
            ),
        );
    }
    let gate_key = prism_pipeline::gating_key(row.arch_id.as_deref());
    let mut infra = matches!(row.final_score, Some(FinalScore::NoScore(6)));
    if infra {
        if let Some(g) = &st.gating {
            infra = g
                .get(&gate_key, &row.miner_hotkey)
                .await
                .ok()
                .flatten()
                .is_some_and(|gr| infra_resubmit_allowed(&gr, now_ms()));
        }
    }
    if !infra {
        if let Err(resp) = verify_bearer(&st.admin_token_hashes, "admin", &headers) {
            return resp;
        }
    }
    // Prefer the request header, else reuse the sealed BYOK vault entry for
    // this submission_id (auto-/admin-retry must not drop miner Lium keys).
    let header_lium_key = headers
        .get(prism_lium_payer::LIUM_API_KEY_HEADER)
        .or_else(|| headers.get("X-Lium-Api-Key"))
        .and_then(|v| v.to_str().ok())
        .and_then(prism_lium_payer::normalize_lium_api_key);
    let miner_lium_key = header_lium_key
        .clone()
        .or_else(|| st.payer_vault.as_ref().and_then(|v| v.get(&id)));
    if infra
        && row.metrics_json.is_none()
        && prism_lium_payer::require_miner_lium(st.backend_mode)
        && miner_lium_key.is_none()
    {
        return json_err(
            StatusCode::BAD_REQUEST,
            "missing_lium_api_key",
            "live Prism retry requires X-Lium-Api-Key (or a sealed payer vault entry) when another GPU run is needed",
        );
    }
    if row.retry_count >= st.retry_max && !infra {
        return json_err(
            StatusCode::CONFLICT,
            "retry_exhausted",
            &format!("retry_count={} max={}", row.retry_count, st.retry_max),
        );
    }
    if infra {
        if let Some(g) = &st.gating {
            let _ = g.reset_open(&gate_key, &row.miner_hotkey).await;
            let _ = g.mark_registered(&gate_key, &row.miner_hotkey, None).await;
        }
    }
    match st.store.reset_for_retry(&id, true).await {
        Ok(_) => {
            if let (Some(vault), Some(key)) = (&st.payer_vault, miner_lium_key) {
                // Re-seal so TTL covers the new attempt even when the header
                // was omitted and we reused the vault entry.
                vault.insert(id.clone(), key);
            }
            (
                StatusCode::ACCEPTED,
                Json(json!({"submission_id": id, "status": "queued"})),
            )
                .into_response()
        }
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

async fn get_jobs(State(st): State<Arc<AppState>>) -> Response {
    let rows = match st.store.list(None, None, 200).await {
        Ok(v) => v,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let jobs: Vec<Value> = rows
        .iter()
        .filter(|r| !r.status.is_terminal() || r.pod_id.is_some())
        .take(200)
        .map(|r| {
            json!({"submission_id": r.id, "miner_hotkey": r.miner_hotkey, "status": r.status.as_str(),
                "pod_id": r.pod_id, "bpb": r.bpb, "retry_count": r.retry_count})
        })
        .collect();
    Json(json!({"jobs": jobs})).into_response()
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

async fn get_architectures(State(st): State<Arc<AppState>>) -> Response {
    match st.store.list_archs(200).await {
        Ok(rows) => Json(json!({
            "architectures": rows.iter().map(|a| json!({
                "arch_id": a.arch_id, "owner_hotkey": a.owner_hotkey, "arch_digest": a.arch_digest,
                "source_submission": a.source_submission, "best_bpb": a.best_bpb,
                "created_at_ms": a.created_at_ms,
            })).collect::<Vec<_>>()
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

pub fn record_epoch(st: &AppState, epoch: u64) {
    st.epoch.store(epoch, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::NoScoreReasonCode;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use prism_store::{FinalScore, MemoryPrismStore, StatePatch};
    use tower::ServiceExt;

    fn state() -> Arc<AppState> {
        // Transitional: training-only `arch_id` unit tests still exercise 1.x.
        std::env::set_var(prism_automodel::ALLOW_LEGACY_ENV, "1");
        Arc::new(AppState {
            store: Arc::new(MemoryPrismStore::new()),
            eval_store: Arc::new(crate::MemoryEvalStore::new()),
            epoch: std::sync::atomic::AtomicU64::new(7),
            netuid: 541,
            backend_mode: "sim",
            retry_max: 2,
            gating: None,
            metagraph: None,
            admin_token_hashes: vec![],
            payer_vault: None,
            logs: std::sync::Arc::new(prism_orphan::LogBuffer::new()),
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
                admin_token_hashes: vec![],
                payer_vault: None,
                logs: std::sync::Arc::new(prism_orphan::LogBuffer::new()),
            }),
            gating,
        )
    }

    #[tokio::test]
    async fn retry_requeues_failed_then_guard_blocks() {
        const ADMIN: &str = "test-admin-token";
        let mut st = state();
        // Rebuild with a configured admin bearer (fail-closed otherwise).
        let inner = Arc::new(AppState {
            store: Arc::clone(&st.store),
            eval_store: Arc::clone(&st.eval_store),
            epoch: std::sync::atomic::AtomicU64::new(7),
            netuid: 541,
            backend_mode: "sim",
            retry_max: 2,
            gating: None,
            metagraph: None,
            admin_token_hashes: vec![prism_intake::token_hash(ADMIN)],
            payer_vault: None,
            logs: std::sync::Arc::new(prism_orphan::LogBuffer::new()),
        });
        st = inner;
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
                .header("authorization", format!("Bearer {ADMIN}"))
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
                .header("authorization", format!("Bearer {ADMIN}"))
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
                .header("authorization", format!("Bearer {ADMIN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT);
    }

    /// Infra retry on live Lium must reuse the sealed BYOK vault when the
    /// miner does not resend `X-Lium-Api-Key` (regression: dropped key →
    /// `missing_lium_api_key` instead of replaying the real infra error).
    #[tokio::test]
    async fn retry_reuses_payer_vault_lium_key() {
        let vault = Arc::new(prism_lium_payer::PayerKeyVault::new());
        let st = Arc::new(AppState {
            store: Arc::new(MemoryPrismStore::new()),
            eval_store: Arc::new(crate::MemoryEvalStore::new()),
            epoch: std::sync::atomic::AtomicU64::new(7),
            netuid: 541,
            backend_mode: "lium",
            retry_max: 2,
            gating: None,
            metagraph: None,
            admin_token_hashes: vec![],
            payer_vault: Some(Arc::clone(&vault)),
            logs: std::sync::Arc::new(prism_orphan::LogBuffer::new()),
        });
        let app = submission_router(Arc::clone(&st));
        let req = crate::example_valid_request();
        let id = prism_pipeline::submission_id(&req);
        let body = serde_json::to_vec(&req).unwrap();
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .header("x-lium-api-key", "sk_test_miner_lium")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert_eq!(vault.get(&id).as_deref(), Some("sk_test_miner_lium"));
        st.store
            .apply(
                &id,
                &StatePatch {
                    status: Some(Stage::Failed),
                    final_score: Some(FinalScore::NoScore(
                        NoScoreReasonCode::ChallengeInternal as u8,
                    )),
                    ..StatePatch::default()
                },
                None,
            )
            .await
            .unwrap();
        // No Lium header — vault must satisfy the live retry gate.
        let (s, v) = call(
            app.clone(),
            Request::post(format!("/v1/submissions/{id}/retry"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert_eq!(v["status"], "queued");
        assert_eq!(vault.get(&id).as_deref(), Some("sk_test_miner_lium"));

        // Without vault entry, live infra retry still demands the header.
        vault.remove(&id);
        st.store
            .apply(
                &id,
                &StatePatch {
                    status: Some(Stage::Failed),
                    final_score: Some(FinalScore::NoScore(
                        NoScoreReasonCode::ChallengeInternal as u8,
                    )),
                    ..StatePatch::default()
                },
                None,
            )
            .await
            .unwrap();
        let (s, v) = call(
            app,
            Request::post(format!("/v1/submissions/{id}/retry"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(v["code"], "missing_lium_api_key");
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
        assert_eq!(v["version"], "2.0.0");
        assert_eq!(v["train_hours_cap"], 6.0);
        assert_eq!(v["automodel_pin_id"], prism_automodel::AUTOMODEL_PIN_ID);
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
    async fn infra_blocked_allows_resubmit_within_window() {
        let (st, gating) = gated_state(&[[0x11; 32]]);
        let hk = "11".repeat(32);
        gating.mark_registered("prism", &hk, Some(0)).await.unwrap();
        gating
            .set_terminal(
                "prism",
                &hk,
                submission_gating::GatingState::Blocked,
                Some("install"),
            )
            .await
            .unwrap();
        let app = submission_router(Arc::clone(&st));
        let mut req = crate::example_valid_request();
        req.architecture_py.push_str("\n# infra-retry\n");
        let body = serde_json::to_vec(&req).unwrap();
        let (s, v) = call(
            app,
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    }

    #[tokio::test]
    async fn infra_blocked_rejects_after_window() {
        let (st, gating) = gated_state(&[[0x11; 32]]);
        let hk = "11".repeat(32);
        gating.mark_registered("prism", &hk, Some(0)).await.unwrap();
        gating
            .set_terminal(
                "prism",
                &hk,
                submission_gating::GatingState::Blocked,
                Some("install"),
            )
            .await
            .unwrap();
        assert!(gating.set_updated_at_ms(
            "prism",
            &hk,
            now_ms().saturating_sub(submission_gating::INFRA_RESUBMIT_WINDOW_MS + 1),
        ));
        let app = submission_router(st);
        let mut req = crate::example_valid_request();
        req.architecture_py.push_str("\n# too-late\n");
        let body = serde_json::to_vec(&req).unwrap();
        let (s, v) = call(
            app,
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["code"], "submission_gated");
    }

    #[tokio::test]
    async fn challenge_internal_retry_bypasses_retry_max_in_window() {
        let (st, gating) = gated_state(&[[0x11; 32]]);
        let app = submission_router(Arc::clone(&st));
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
        let hk = "11".repeat(32);
        // Burn retry_count past retry_max with a ChallengeInternal terminal.
        st.store
            .apply(
                &id,
                &StatePatch {
                    status: Some(Stage::Failed),
                    final_score: Some(FinalScore::NoScore(
                        NoScoreReasonCode::ChallengeInternal as u8,
                    )),
                    retry_bump: st.retry_max.saturating_add(1),
                    ..StatePatch::default()
                },
                None,
            )
            .await
            .unwrap();
        gating
            .set_terminal(
                "prism",
                &hk,
                submission_gating::GatingState::Blocked,
                Some("install"),
            )
            .await
            .unwrap();
        let (s, v) = call(
            app,
            Request::post(format!("/v1/submissions/{id}/retry"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
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
            admin_token_hashes: vec![],
            payer_vault: None,
            logs: std::sync::Arc::new(prism_orphan::LogBuffer::new()),
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
            "battery": { "metrics": { "org.g1.bits_per_byte_code": 1.0 } },
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
        assert_eq!(v["metrics"][0]["key"], "org.g1.bits_per_byte_code");

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
    async fn inference_traces_paginated_from_metrics_json() {
        let st = state();
        let app = submission_router(Arc::clone(&st));
        let id = post_one(&app).await;
        let traces = json!({
            "version": 1,
            "truncated": false,
            "n_items": 2,
            "bytes_approx": 120,
            "caps": { "global": 2500 },
            "items": [
                {
                    "kind": "mc",
                    "group": "g2",
                    "task": "hellaswag",
                    "cluster": "g2/hellaswag",
                    "prompt": "The dog",
                    "choices": [" barked", " flew"],
                    "gold": 0,
                    "selected": 0,
                    "value": 1.0,
                    "choice_logprobs": [{"i": 0, "sum_lp": -0.1, "n_tok": 1, "norm_lp": -0.01}]
                },
                {
                    "kind": "mc",
                    "group": "g3",
                    "cluster": "mqar/n4",
                    "prompt": "k1 v1",
                    "choices": [" v1", " v9"],
                    "gold": 0,
                    "selected": 1,
                    "value": 0.0
                }
            ]
        });
        st.store
            .apply(
                &id,
                &StatePatch {
                    metrics_json: Some(json!({
                        "bpb": 1.5,
                        "inference_traces": traces,
                    })),
                    ..StatePatch::default()
                },
                None,
            )
            .await
            .unwrap();

        let (s, v) = call(
            app.clone(),
            Request::get(format!("/v1/submissions/{id}/inference?group=g2&limit=10"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["provenance"], "organizer-measured");
        assert_eq!(v["battery"]["total"], 1);
        assert_eq!(v["battery"]["items"][0]["prompt"], "The dog");
        assert_eq!(v["battery"]["items"][0]["choices"][0], " barked");
        assert_eq!(v["battery"]["items"][0]["selected"], 0);

        let (s, _) = call(
            app,
            Request::get("/v1/submissions/nope/inference")
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

    #[tokio::test]
    async fn automodel_fixture_intake_and_diff_route() {
        let st = state();
        let app = submission_router(Arc::clone(&st));
        let req = crate::example_automodel_request();
        let id = prism_pipeline::submission_id(&req);
        let body = serde_json::to_vec(&req).unwrap();
        // Wire JSON omits packed_tree (serde skip) — re-expand via zip_base64.
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert_eq!(v["submission_id"], id);
        assert_eq!(v["status"], "accepted");

        let (s, v) = call(
            app,
            Request::get(format!("/v1/submissions/{id}/diff"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["pin_id"], prism_automodel::FIXTURE_PIN_ID);
        assert!(v["patch"].as_str().unwrap().contains("ToyModel"));
        assert!(!v["diffstat"]["files"].as_array().unwrap().is_empty());
        let classes: Vec<&str> = v["diffstat"]["files"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["class"].as_str())
            .collect();
        assert!(classes.contains(&"arch"), "{classes:?}");
    }

    #[tokio::test]
    async fn legacy_zip_rejected_unsupported_layout() {
        use base64::Engine as _;
        let st = state();
        let app = submission_router(st);
        let arch = "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n";
        let train = "def train(model, ctx):\n    return {}\n";
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            use std::io::Write;
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default();
            w.start_file("architecture.py", opts).unwrap();
            w.write_all(arch.as_bytes()).unwrap();
            w.start_file("training.py", opts).unwrap();
            w.write_all(train.as_bytes()).unwrap();
            w.finish().unwrap();
        }
        let zip_b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        let body = serde_json::json!({
            "miner_hotkey": "11".repeat(32),
            "zip_base64": zip_b64,
        });
        let (s, v) = call(
            app,
            Request::post("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(v["code"], "unsupported_layout");
    }
}
