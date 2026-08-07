//! HTTP API: miner / viewer / admin winners / annotate (deprecated) / ops.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use bytes::Bytes;
use design_harness::{
    encode_env_into_extras, harness_from_zip, harness_id, validate_bundle, HarnessBundle,
};
use design_prompts::{load_prompt_set, prompt_set_digest, select_prompts_for_round};
use design_store::{
    DesignStore, HarnessRow, RoundAward, RunOrigin, RunStage, RunState, StageEvent, StoreError,
    StorePatch,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use submission_gating::{GatingState, GatingStore, MetagraphCache};

use design_challenge_task::{
    manual_daily_run_quota, round_id_at, round_secs, scheduled_daily_run_cap, CHALLENGE_ID,
    MIN_ANNOTATIONS_PER_PAIR, RUN_ID_DOMAIN,
};

/// Hook invoked after admin persists winners (score + leaf emit).
#[async_trait]
pub trait AdminAwardHook: Send + Sync + std::fmt::Debug {
    /// Award round and emit exact-E leaves.
    async fn on_winners(&self, round_id: u64, harness_ids: &[String]) -> Result<(), String>;
}

/// Shared app state.
pub struct AppState {
    /// Store.
    pub store: Arc<dyn DesignStore>,
    /// Chain epoch cache.
    pub epoch: std::sync::atomic::AtomicU64,
    /// Netuid.
    pub netuid: u16,
    /// Backend label.
    pub backend_mode: &'static str,
    /// Annotator / admin token hashes (sha256 hex of bearer tokens).
    pub annotator_token_hashes: Vec<String>,
    /// Admin token hashes (falls back to annotator hashes when empty).
    pub admin_token_hashes: Vec<String>,
    /// CSP frame-ancestors.
    pub frame_ancestors: String,
    /// Max retries.
    pub retry_max: u32,
    /// Optional award hook (orchestrator).
    pub award_hook: Option<Arc<dyn AdminAwardHook>>,
    /// Submission gating (1-max). `None` disables intake gating (tests/dev).
    pub gating: Option<Arc<dyn GatingStore>>,
    /// Cached metagraph snapshot for intake membership. `None` disables the
    /// membership check (tests/dev).
    pub metagraph: Option<Arc<MetagraphCache>>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("netuid", &self.netuid)
            .field("backend_mode", &self.backend_mode)
            .field("admin_tokens", &self.admin_token_hashes.len())
            .field("has_award_hook", &self.award_hook.is_some())
            .finish_non_exhaustive()
    }
}

/// Build router.
pub fn design_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/status", get(get_status))
        .route("/v1/stats", get(crate::stats::get_stats))
        .route("/v1/dashboard", get(crate::stats::get_dashboard))
        .route("/v1/jobs", get(get_jobs))
        .route("/v1/harness", post(post_harness))
        .route("/v1/harness/{id}", get(get_harness))
        .route("/v1/harness", get(list_harness))
        .route("/v1/quota/{hotkey}", get(get_quota))
        .route("/v1/miners/{hotkey}", get(crate::stats::get_miner))
        .route("/v1/prompts", get(get_prompts))
        .route("/v1/rounds", get(get_rounds))
        .route("/v1/runs/{id}", get(get_run))
        .route("/v1/runs/{id}/events", get(get_run_events))
        .route("/v1/runs/{id}/logs", get(get_run_logs))
        .route("/v1/runs/{id}/retry", post(post_retry))
        .route("/v1/runs/{id}/pages", get(get_pages))
        .route("/v1/runs/{id}/bundle.json", get(get_bundle_json))
        .route("/v1/view/{id}/{page}", get(view_page))
        .route("/v1/annotate/next", get(annotate_next))
        .route("/v1/annotate", post(post_annotate))
        .route("/v1/admin/rounds/{id}/candidates", get(admin_candidates))
        .route("/v1/admin/rounds/{id}/winners", post(admin_winners))
        .route(
            "/v1/admin/rounds/current/requeue",
            post(admin_requeue_current),
        )
        .route("/v1/rounds/{id}/leaderboard", get(leaderboard))
        .with_state(state)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn utc_day(secs: u64) -> String {
    // YYYY-MM-DD from unix days (UTC).
    let days = secs / 86_400;
    // Civil from days since 1970-01-01 (Howard Hinnant algorithm, simplified).
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

fn json_err(code: StatusCode, kind: &str, msg: &str) -> Response {
    (code, Json(json!({"error": kind, "message": msg}))).into_response()
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

fn check_token_hashes(hashes: &[String], headers: &HeaderMap) -> Result<String, Response> {
    let token = bearer_token(headers)?;
    let th = token_hash(token);
    if !hashes.iter().any(|h| h == &th) {
        return Err(json_err(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "bad token",
        ));
    }
    Ok(th)
}

fn check_annotator(st: &AppState, headers: &HeaderMap) -> Result<String, Response> {
    check_token_hashes(&st.annotator_token_hashes, headers)
}

fn check_admin(st: &AppState, headers: &HeaderMap) -> Result<String, Response> {
    let hashes = if st.admin_token_hashes.is_empty() {
        &st.annotator_token_hashes
    } else {
        &st.admin_token_hashes
    };
    check_token_hashes(hashes, headers)
}

async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({"status":"ok","challenge_id":CHALLENGE_ID})),
    )
}

async fn get_status(State(st): State<Arc<AppState>>) -> Response {
    let queued = st
        .store
        .list_runs(Some("queued"), 500)
        .await
        .map(|v| v.len())
        .unwrap_or(0);
    Json(json!({
        "challenge_id": CHALLENGE_ID,
        "backend": st.backend_mode,
        "epoch": st.epoch.load(std::sync::atomic::Ordering::Relaxed),
        "netuid": st.netuid,
        "round_id": round_id_at(now_secs()),
        "queued_runs": queued,
        "prompt_set_digest": prompt_set_digest(),
    }))
    .into_response()
}

async fn get_jobs(State(st): State<Arc<AppState>>) -> Response {
    match st.store.list_runs(None, 50).await {
        Ok(rows) => Json(json!({
            "jobs": rows.iter().map(|r| json!({
                "id": r.id,
                "status": r.status.as_str(),
                "round_id": r.round_id,
                "prompt_id": r.prompt_id,
            })).collect::<Vec<_>>()
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct HarnessSubmitJson {
    miner_hotkey: String,
    #[serde(default)]
    agent_py: Option<String>,
    #[serde(default)]
    pyproject_toml: Option<String>,
    #[serde(default)]
    extra_files: BTreeMap<String, String>,
    /// Optional base64-encoded ZIP (`agent.py` + `pyproject.toml`).
    #[serde(default)]
    zip_base64: Option<String>,
    #[serde(default)]
    env_vars: BTreeMap<String, String>,
}

fn parse_harness_body(headers: &HeaderMap, body: &[u8]) -> Result<HarnessBundle, String> {
    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if ct.contains("application/zip") || ct.contains("application/x-zip-compressed") {
        let hk = headers
            .get("x-miner-hotkey")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| "X-Miner-Hotkey required for application/zip".to_owned())?;
        let env_vars = match headers.get("x-env-json").and_then(|v| v.to_str().ok()) {
            Some(raw) => {
                serde_json::from_str(raw).map_err(|e| format!("X-Env-Json invalid: {e}"))?
            }
            None => BTreeMap::new(),
        };
        return harness_from_zip(hk, body, env_vars).map_err(|e| e.to_string());
    }
    let submit: HarnessSubmitJson =
        serde_json::from_slice(body).map_err(|e| format!("invalid_json: {e}"))?;
    if let Some(b64) = submit.zip_base64.as_deref().filter(|s| !s.is_empty()) {
        let zip = base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .map_err(|e| format!("zip_base64: {e}"))?;
        return harness_from_zip(&submit.miner_hotkey, &zip, submit.env_vars)
            .map_err(|e| e.to_string());
    }
    let bundle = HarnessBundle {
        miner_hotkey: submit.miner_hotkey,
        agent_py: submit
            .agent_py
            .ok_or_else(|| "agent_py required (or zip_base64 / application/zip)".to_owned())?,
        pyproject_toml: submit.pyproject_toml.ok_or_else(|| {
            "pyproject_toml required (or zip_base64 / application/zip)".to_owned()
        })?,
        extra_files: submit.extra_files,
        env_vars: submit.env_vars,
    };
    validate_bundle(&bundle).map_err(|e| e.to_string())?;
    Ok(bundle)
}

async fn post_harness(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req = match parse_harness_body(&headers, body.as_ref()) {
        Ok(b) => b,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "invalid_harness", &e),
    };
    if let Err(e) = validate_bundle(&req) {
        return json_err(StatusCode::BAD_REQUEST, "invalid_harness", &e.to_string());
    }
    let id = harness_id(&req);
    let hotkey = req.miner_hotkey.trim().to_ascii_lowercase();

    // Idempotent duplicate: identical digest re-POST never conflicts gating.
    let exists = match st.store.get_harness(&id).await {
        Ok(h) => h.is_some(),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };

    let mut uid = None;
    if !exists {
        // Metagraph membership (fail closed when a cache is configured but has
        // no snapshot yet).
        if let Some(cache) = &st.metagraph {
            match cache.snapshot() {
                Some(view) => match view.uid_of_hex(&hotkey) {
                    Some(u) => uid = Some(u),
                    None => {
                        return json_err(
                            StatusCode::FORBIDDEN,
                            "hotkey_not_in_metagraph",
                            "miner hotkey is not registered on this subnet",
                        );
                    }
                },
                None => {
                    return json_err(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "metagraph_unavailable",
                        "metagraph snapshot not ready; retry shortly",
                    );
                }
            }
        }
        // One accepted submission per (challenge, hotkey).
        if let Some(g) = &st.gating {
            match g.get(CHALLENGE_ID, &hotkey).await {
                Ok(Some(row)) if row.state != GatingState::Open => {
                    return json_err(
                        StatusCode::CONFLICT,
                        "submission_gated",
                        &format!(
                            "hotkey is '{}' for this challenge; one accepted submission max",
                            row.state.as_str()
                        ),
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    return json_err(StatusCode::INTERNAL_SERVER_ERROR, "gating", &e.to_string());
                }
            }
        }
    }

    let mut extras = req.extra_files;
    encode_env_into_extras(&mut extras, &req.env_vars);
    let row = HarnessRow {
        id: id.clone(),
        miner_hotkey: hotkey.clone(),
        agent_py: req.agent_py,
        pyproject_toml: req.pyproject_toml,
        extra_files: extras,
        active: true,
        eliminated_until_round: 0,
        created_at_ms: 0,
    };
    let fresh = match st.store.insert_harness(&row).await {
        Ok(()) => true,
        Err(StoreError::Duplicate) => false,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let harness = match st.store.get_harness(&id).await {
        Ok(Some(h)) => h,
        Ok(None) => row,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    if fresh {
        let _ = st
            .store
            .deactivate_other_harnesses(&hotkey, &harness.id)
            .await;
    }
    // Always schedule into the next round when that round has no runs yet —
    // including identical re-POST (idempotent digest) so rolling rounds keep
    // receiving work without requiring a new harness digest.
    let rid = round_id_at(now_secs()).saturating_add(1);
    let epoch = st.epoch.load(std::sync::atomic::Ordering::Relaxed);
    let run_ids = match schedule_harness_for_round(
        st.store.as_ref(),
        &harness,
        rid,
        st.netuid,
        epoch,
        RunOrigin::Manual,
    )
    .await
    {
        Ok(ids) => ids,
        Err(e) => return json_err(StatusCode::CONFLICT, "schedule", &e),
    };
    // Registration finalizes only after runs are scheduled so scheduling
    // failures (quota) never consume the miner's single slot.
    if fresh {
        if let Some(g) = &st.gating {
            if let Err(e) = g.mark_registered(CHALLENGE_ID, &hotkey, uid).await {
                return json_err(StatusCode::INTERNAL_SERVER_ERROR, "gating", &e.to_string());
            }
        }
    }
    // Digest already known → idempotent OK; runs may still be created for a
    // future round that had none yet (rolling auto-schedule path).
    let status = if fresh { "accepted" } else { "already-queued" };
    let code = if fresh {
        StatusCode::ACCEPTED
    } else {
        StatusCode::OK
    };
    (
        code,
        Json(json!({
            "harness_id": id,
            "status": status,
            "round_id": rid,
            "round_opens_at_secs": rid.saturating_mul(round_secs()),
            "run_ids": run_ids,
            "poll": {
                "run": "/v1/runs/{id}",
                "events": "/v1/runs/{id}/events",
                "logs": "/v1/runs/{id}/logs?since=0",
                "hint_ms": 1000,
            },
        })),
    )
        .into_response()
}

/// Schedule an active harness into `rid` (create missing queued runs).
///
/// Idempotent: if runs for `(harness, round)` already exist, returns those ids.
/// Honours the elimination cooldown and the daily ceiling **for `origin`**.
/// Used by submit ([`RunOrigin::Manual`], next round) and by the orchestrator
/// ([`RunOrigin::Scheduled`], every open round).
///
/// The two origins never share a budget: the organizer dispatches
/// `rounds/day × prompts/round` runs to every registered harness, so charging
/// that volume to the miner's anti-spam quota would lock an honest harness out
/// of most of its own day.
pub async fn schedule_harness_for_round(
    store: &dyn DesignStore,
    harness: &HarnessRow,
    rid: u64,
    netuid: u16,
    epoch: u64,
    origin: RunOrigin,
) -> Result<Vec<String>, String> {
    if !harness.active {
        return Ok(vec![]);
    }
    if harness.eliminated_until_round > rid {
        return Err(format!(
            "eliminated until round {}",
            harness.eliminated_until_round
        ));
    }
    let existing = store.runs_for_round(rid).await.map_err(|e| e.to_string())?;
    let run_ids: Vec<String> = existing
        .iter()
        .filter(|r| r.harness_id == harness.id)
        .map(|r| r.id.clone())
        .collect();
    if !run_ids.is_empty() {
        return Ok(run_ids);
    }
    let secs = now_secs();
    let day = utc_day(secs);
    let used = store
        .quota_get(&harness.miner_hotkey, &day)
        .await
        .map_err(|e| e.to_string())?
        .used(origin);
    if store
        .get_round(rid)
        .await
        .map_err(|e| e.to_string())?
        .is_none()
    {
        let opens = rid * round_secs();
        store
            .insert_round(&design_store::RoundRow {
                round_id: rid,
                epoch,
                netuid,
                prompt_set_digest: prompt_set_digest(),
                status: "open".into(),
                opens_at_secs: opens,
                closes_at_secs: opens + round_secs(),
            })
            .await
            .map_err(|e| e.to_string())?;
    }
    let prompts = select_prompts_for_round(rid).map_err(|e| e.to_string())?;
    let mut run_ids: Vec<String> = Vec::new();
    let cap = origin_daily_cap(origin);
    // All-or-nothing: a round that can only afford part of its prompt set would
    // leave the harness permanently short for that round (re-scheduling is a
    // no-op once any run exists), so refuse instead of degrading it.
    let needed = u32::try_from(prompts.len()).unwrap_or(u32::MAX);
    if used.saturating_add(needed) > cap {
        return Err(format!(
            "daily {} run quota exceeded ({used}+{needed}/{cap})",
            origin.as_str()
        ));
    }
    for p in prompts {
        let run_id = make_run_id(rid, &harness.id, &p.id);
        if store
            .get_run(&run_id)
            .await
            .map_err(|e| e.to_string())?
            .is_some()
        {
            run_ids.push(run_id);
            continue;
        }
        let row = RunState {
            id: run_id.clone(),
            round_id: rid,
            harness_id: harness.id.clone(),
            prompt_id: p.id.clone(),
            status: RunStage::Queued,
            artifact_digest: None,
            sanitize_report: None,
            agentic_verdict: None,
            error_detail: None,
            final_score: None,
            retry_count: 0,
            created_at_ms: now_ms(),
            updated_at_ms: now_ms(),
        };
        store.insert_run(&row).await.map_err(|e| e.to_string())?;
        let _ = store
            .apply_run(
                &run_id,
                &StorePatch::default(),
                Some(&StageEvent {
                    stage: "queued".into(),
                    detail: Some(json!({"prompt_id": p.id, "round_id": rid})),
                    at_ms: now_ms(),
                }),
            )
            .await;
        let _ = store
            .quota_bump(&harness.miner_hotkey, &day, 1, origin)
            .await;
        run_ids.push(run_id);
    }
    Ok(run_ids)
}

/// Daily ceiling that applies to `origin`.
fn origin_daily_cap(origin: RunOrigin) -> u32 {
    match origin {
        RunOrigin::Manual => manual_daily_run_quota(),
        RunOrigin::Scheduled => scheduled_daily_run_cap(),
    }
}

fn make_run_id(round_id: u64, harness_id: &str, prompt_id: &str) -> String {
    let mut h = Sha256::new();
    h.update(RUN_ID_DOMAIN);
    h.update(round_id.to_be_bytes());
    h.update(harness_id.as_bytes());
    h.update(prompt_id.as_bytes());
    hex::encode(h.finalize())
}

async fn get_harness(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.store.get_harness(&id).await {
        Ok(Some(h)) => Json(json!({
            "id": h.id,
            "miner_hotkey": h.miner_hotkey,
            "active": h.active,
            "eliminated_until_round": h.eliminated_until_round,
        }))
        .into_response(),
        Ok(None) => json_err(StatusCode::NOT_FOUND, "not_found", "harness"),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct MinerQuery {
    miner: Option<String>,
}

async fn list_harness(State(st): State<Arc<AppState>>, Query(q): Query<MinerQuery>) -> Response {
    let Some(miner) = q.miner else {
        return json_err(StatusCode::BAD_REQUEST, "missing_miner", "miner=");
    };
    match st.store.list_harnesses(&miner).await {
        Ok(rows) => Json(json!({
            "harnesses": rows.iter().map(|h| json!({
                "id": h.id,
                "active": h.active,
                "eliminated_until_round": h.eliminated_until_round,
            })).collect::<Vec<_>>()
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn get_quota(State(st): State<Arc<AppState>>, Path(hotkey): Path<String>) -> Response {
    let day = utc_day(now_secs());
    match st.store.quota_get(&hotkey, &day).await {
        Ok(used) => Json(quota_json(&hotkey, &day, used)).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

/// Per-origin quota view shared by `/v1/quota/{hotkey}` and `/v1/miners`.
///
/// `runs_used` / `limit` stay the whole-day totals for existing clients;
/// enforcement is the per-origin pair below them.
pub(crate) fn quota_json(hotkey: &str, day: &str, used: design_store::QuotaUsage) -> Value {
    let manual_limit = manual_daily_run_quota();
    let scheduled_limit = scheduled_daily_run_cap();
    json!({
        "miner_hotkey": hotkey,
        "day": day,
        "runs_used": used.total,
        "limit": manual_limit.saturating_add(scheduled_limit),
        "manual": {
            "runs_used": used.manual,
            "limit": manual_limit,
            "remaining": manual_limit.saturating_sub(used.manual),
        },
        "scheduled": {
            "runs_used": used.scheduled(),
            "limit": scheduled_limit,
            "remaining": scheduled_limit.saturating_sub(used.scheduled()),
        },
    })
}

async fn get_prompts() -> Response {
    match load_prompt_set() {
        Ok(set) => Json(json!({
            "digest": prompt_set_digest(),
            "prompts": set,
            "current_round": round_id_at(now_secs()),
            "selected": select_prompts_for_round(round_id_at(now_secs())).unwrap_or_default(),
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "prompts", &e.to_string()),
    }
}

async fn get_rounds(State(st): State<Arc<AppState>>) -> Response {
    match st.store.list_rounds(50).await {
        Ok(rows) => Json(json!({"rounds": rows})).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn get_run(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match crate::stats::run_status_json(st.store.as_ref(), &id).await {
        Ok(v) => Json(v).into_response(),
        Err(_) => json_err(StatusCode::NOT_FOUND, "not_found", "run"),
    }
}

async fn get_run_events(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.store.run_events(&id).await {
        Ok(evs) => Json(json!({
            "run_id": id,
            "events": evs,
            "count": evs.len(),
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct LogsQuery {
    /// Cursor: return log events with seq >= since (default 0).
    since: Option<u64>,
    /// If set, return only the last N log chunks.
    tail: Option<u32>,
}

async fn get_run_logs(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<LogsQuery>,
) -> Response {
    if st.store.get_run(&id).await.ok().flatten().is_none() {
        return json_err(StatusCode::NOT_FOUND, "not_found", "run");
    }
    let evs = match st.store.run_events(&id).await {
        Ok(e) => e,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let since = q.since.unwrap_or(0);
    let mut logs: Vec<Value> = Vec::new();
    let mut next_since = since;
    for e in evs {
        if e.stage != "log" {
            continue;
        }
        let seq = e
            .detail
            .as_ref()
            .and_then(|d| d.get("seq"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if seq < since {
            continue;
        }
        logs.push(json!({
            "seq": seq,
            "phase": e.detail.as_ref().and_then(|d| d.get("phase")).cloned().unwrap_or(Value::Null),
            "stream": e.detail.as_ref().and_then(|d| d.get("stream")).cloned().unwrap_or(Value::Null),
            "text": e.detail.as_ref().and_then(|d| d.get("text")).cloned().unwrap_or(Value::Null),
            "bytes": e.detail.as_ref().and_then(|d| d.get("bytes")).cloned().unwrap_or(Value::Null),
            "truncated": e.detail.as_ref().and_then(|d| d.get("truncated")).cloned().unwrap_or(json!(false)),
            "at_ms": e.at_ms,
        }));
        next_since = next_since.max(seq.saturating_add(1));
    }
    if let Some(tail) = q.tail {
        let n = tail as usize;
        if logs.len() > n {
            logs = logs.split_off(logs.len() - n);
        }
    }
    Json(json!({
        "run_id": id,
        "logs": logs,
        "next_since": next_since,
        "poll_hint_ms": 1000,
    }))
    .into_response()
}

async fn post_retry(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let row = match st.store.get_run(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "not_found", "run"),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    if row.status != RunStage::Failed {
        return json_err(StatusCode::CONFLICT, "not_failed", row.status.as_str());
    }
    if row.retry_count >= st.retry_max {
        return json_err(StatusCode::CONFLICT, "retry_exhausted", "max retries");
    }
    match st.store.reset_run(&id).await {
        Ok(_) => (
            StatusCode::ACCEPTED,
            Json(json!({"run_id": id, "status": "queued"})),
        )
            .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn get_pages(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.store.list_pages(&id).await {
        Ok(pages) => Json(json!({
            "pages": pages.iter().map(|p| json!({
                "path": p.path,
                "bytes": p.bytes,
                "raw_sha256": p.raw_sha256,
            })).collect::<Vec<_>>()
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

/// `bundle.json` used to embed every page's produced HTML. The viewer is
/// screenshots-only now, so the route is retired with a pointer instead of
/// serving a silently hollow bundle.
async fn get_bundle_json() -> Response {
    json_err(
        StatusCode::GONE,
        "gone",
        "bundle.json no longer embeds produced HTML; use /v1/runs/{id}/pages for page metadata and /v1/view/{id}/index.png for the screenshot",
    )
}

/// Screenshots-only viewer: produced HTML is never served. Only captured PNG
/// artifacts (`index.png`) are public; any non-PNG page request is 410 Gone.
async fn view_page(
    State(st): State<Arc<AppState>>,
    Path((id, page)): Path<(String, String)>,
) -> Response {
    if !page.ends_with(".png") {
        return json_err(
            StatusCode::GONE,
            "gone",
            "produced HTML is never served; fetch the index.png screenshot instead",
        );
    }
    match st.store.get_page(&id, &page).await {
        Ok(Some(body)) => {
            let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(body.trim()) else {
                return json_err(StatusCode::INTERNAL_SERVER_ERROR, "artifact", "bad png b64");
            };
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("image/png"),
            );
            headers.insert(
                header::CACHE_CONTROL,
                header::HeaderValue::from_static("private, no-store"),
            );
            headers.insert(
                header::HeaderName::from_static("x-content-type-options"),
                header::HeaderValue::from_static("nosniff"),
            );
            (StatusCode::OK, headers, bytes).into_response()
        }
        Ok(None) => json_err(StatusCode::NOT_FOUND, "not_found", "page"),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct AnnotateNextQuery {
    annotator: String,
}

async fn annotate_next(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<AnnotateNextQuery>,
) -> Response {
    if let Err(r) = check_annotator(&st, &headers) {
        return r;
    }
    let rid = round_id_at(now_secs());
    match st
        .store
        .next_pair(rid, &q.annotator, MIN_ANNOTATIONS_PER_PAIR)
        .await
    {
        Ok(Some(p)) => Json(json!({
            "pair_id": p.id,
            "round_id": p.round_id,
            "prompt_id": p.prompt_id,
            "run_a_id": p.run_a_id,
            "run_b_id": p.run_b_id,
        }))
        .into_response(),
        Ok(None) => json_err(StatusCode::NOT_FOUND, "no_pair", "none available"),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct AnnotateBody {
    pair_id: String,
    annotator_id: String,
    winner_run_id: String,
}

async fn post_annotate(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<AnnotateBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Err(r) = check_annotator(&st, &headers) {
        return r;
    }
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "invalid_json", &e.body_text()),
    };
    match st
        .store
        .insert_annotation(&req.pair_id, &req.annotator_id, &req.winner_run_id)
        .await
    {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({"status": "ok"}))).into_response(),
        Err(StoreError::Duplicate) => {
            json_err(StatusCode::CONFLICT, "duplicate", "already annotated")
        }
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn leaderboard(State(st): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    match st.store.ratings_for_round(id).await {
        Ok(rows) => Json(
            json!({"round_id": id, "ratings": rows.iter().map(|r| json!({
            "miner_hotkey": r.miner_hotkey,
            "rating": r.rating,
            "wins": r.wins,
            "losses": r.losses,
            "final_score": match &r.final_score {
                Some(design_store::FinalScore::Score(v)) => json!({"score": v}),
                Some(design_store::FinalScore::NoScore(c)) => json!({"no_score": c}),
                None => Value::Null,
            },
        })).collect::<Vec<_>>()}),
        )
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

/// Advance epoch cache (worker).
pub fn record_epoch(st: &AppState, epoch: u64) {
    st.epoch.store(epoch, std::sync::atomic::Ordering::Relaxed);
}

/// Mark run awaiting annotation after sanitize (deprecated leaf path).
pub async fn mark_awaiting(
    store: &dyn DesignStore,
    run_id: &str,
    digest: &str,
    report: Value,
) -> Result<(), StoreError> {
    store
        .apply_run(
            run_id,
            &StorePatch {
                status: Some(RunStage::AwaitingAnnotation),
                artifact_digest: Some(digest.to_owned()),
                sanitize_report: Some(report),
                ..StorePatch::default()
            },
            Some(&StageEvent {
                stage: "awaiting_annotation".into(),
                detail: None,
                at_ms: now_ms(),
            }),
        )
        .await?;
    Ok(())
}

/// Mark run awaiting admin after clean agentic review.
pub async fn mark_awaiting_admin(
    store: &dyn DesignStore,
    run_id: &str,
    digest: &str,
    report: Value,
    verdict: Value,
) -> Result<(), StoreError> {
    store
        .apply_run(
            run_id,
            &StorePatch {
                status: Some(RunStage::AwaitingAdmin),
                artifact_digest: Some(digest.to_owned()),
                sanitize_report: Some(report),
                agentic_verdict: Some(verdict),
                ..StorePatch::default()
            },
            Some(&StageEvent {
                stage: "awaiting_admin".into(),
                detail: None,
                at_ms: now_ms(),
            }),
        )
        .await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct WinnersBody {
    harness_ids: Vec<String>,
}

/// Manually schedule every active harness into the CURRENT open round.
///
/// Operator escape hatch when a round opened with no/few runs (e.g. challenge
/// restart). Idempotent for the current round: `schedule_harness_for_round`
/// returns the existing run ids for a `(harness, round)` pair that already has
/// runs, so a repeated call creates nothing and consumes no quota. This is
/// organizer work, so it draws on the scheduled cap, never on the miner's
/// submission quota. Harnesses that fail scheduling are reported under
/// `skipped`; one bad harness never blocks the rest.
async fn admin_requeue_current(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(r) = check_admin(&st, &headers) {
        return r;
    }
    let rid = round_id_at(now_secs());
    let harnesses = match st.store.list_active_harnesses(rid).await {
        Ok(h) => h,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let epoch = st.epoch.load(std::sync::atomic::Ordering::Relaxed);
    let mut scheduled = Vec::new();
    let mut skipped = Vec::new();
    for harness in &harnesses {
        match schedule_harness_for_round(
            st.store.as_ref(),
            harness,
            rid,
            st.netuid,
            epoch,
            RunOrigin::Scheduled,
        )
        .await
        {
            Ok(run_ids) => scheduled.push(json!({
                "harness_id": harness.id,
                "miner_hotkey": harness.miner_hotkey,
                "run_ids": run_ids,
            })),
            Err(e) => skipped.push(json!({
                "harness_id": harness.id,
                "miner_hotkey": harness.miner_hotkey,
                "reason": e,
            })),
        }
    }
    Json(json!({
        "round_id": rid,
        "scheduled": scheduled,
        "skipped": skipped,
    }))
    .into_response()
}

async fn admin_candidates(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<u64>,
) -> Response {
    if let Err(r) = check_admin(&st, &headers) {
        return r;
    }
    let runs = match st.store.runs_for_round(id).await {
        Ok(r) => r,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let mut out = Vec::new();
    for r in runs
        .into_iter()
        .filter(|r| r.status == RunStage::AwaitingAdmin)
    {
        let pages = st.store.list_pages(&r.id).await.unwrap_or_default();
        let harness = st.store.get_harness(&r.harness_id).await.ok().flatten();
        out.push(json!({
            "run_id": r.id,
            "harness_id": r.harness_id,
            "miner_hotkey": harness.map(|h| h.miner_hotkey),
            "prompt_id": r.prompt_id,
            "artifact_digest": r.artifact_digest,
            "agentic_verdict": r.agentic_verdict,
            "pages": pages.iter().map(|p| p.path.clone()).collect::<Vec<_>>(),
        }));
    }
    Json(json!({"round_id": id, "candidates": out})).into_response()
}

async fn admin_winners(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<u64>,
    body: Result<Json<WinnersBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let token_hash = match check_admin(&st, &headers) {
        Ok(h) => h,
        Err(r) => return r,
    };
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "invalid_json", &e.body_text()),
    };
    let mut ids = req.harness_ids;
    ids.sort();
    ids.dedup();
    if ids.is_empty() || ids.len() > 2 {
        return json_err(
            StatusCode::BAD_REQUEST,
            "invalid_winners",
            "harness_ids length must be 1 or 2",
        );
    }
    let runs = match st.store.runs_for_round(id).await {
        Ok(r) => r,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let eligible: std::collections::HashSet<_> = runs
        .iter()
        .filter(|r| r.status == RunStage::AwaitingAdmin)
        .map(|r| r.harness_id.clone())
        .collect();
    for hid in &ids {
        if !eligible.contains(hid) {
            return json_err(
                StatusCode::BAD_REQUEST,
                "not_candidate",
                &format!("harness {hid} is not a clean awaiting_admin candidate"),
            );
        }
    }
    let award = RoundAward {
        round_id: id,
        winner_harness_ids: ids.clone(),
        awarded_at_ms: now_ms(),
        admin_token_hash: Some(token_hash),
    };
    match st.store.set_round_award(&award).await {
        Ok(()) => {}
        Err(StoreError::Duplicate) => {
            return json_err(StatusCode::CONFLICT, "duplicate", "winners already set");
        }
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
    let _ = st.store.set_round_status(id, "scoring").await;
    if let Some(hook) = &st.award_hook {
        if let Err(e) = hook.on_winners(id, &ids).await {
            return json_err(StatusCode::INTERNAL_SERVER_ERROR, "award", &e);
        }
    }
    (
        StatusCode::ACCEPTED,
        Json(json!({
            "status": "ok",
            "round_id": id,
            "harness_ids": ids,
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
    use design_store::MemoryDesignStore;
    use http_body_util::BodyExt;
    use submission_gating::MemoryGatingStore;
    use tower::ServiceExt;

    fn hk(byte: u8) -> String {
        hex::encode([byte; 32])
    }

    fn app_state(
        metagraph_hotkeys: Option<Vec<[u8; 32]>>,
    ) -> (Arc<AppState>, Arc<MemoryGatingStore>) {
        let gating = Arc::new(MemoryGatingStore::new());
        let metagraph = metagraph_hotkeys.map(|keys| {
            let cache = Arc::new(MetagraphCache::new());
            cache.update(541, &keys.iter().map(|h| h.to_vec()).collect::<Vec<_>>());
            cache
        });
        (
            Arc::new(AppState {
                store: Arc::new(MemoryDesignStore::new()),
                epoch: std::sync::atomic::AtomicU64::new(0),
                netuid: 541,
                backend_mode: "memory",
                annotator_token_hashes: vec![],
                admin_token_hashes: vec![],
                frame_ancestors: "'none'".into(),
                retry_max: 2,
                award_hook: None,
                gating: Some(Arc::clone(&gating) as Arc<dyn GatingStore>),
                metagraph,
            }),
            gating,
        )
    }

    fn submit_body(hotkey: &str, marker: &str) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "miner_hotkey": hotkey,
            "agent_py": format!("def run(task, llm, out):\n    out.write_page('index.html', '<html>{marker}</html>')\n"),
            "pyproject_toml": "[project]\nname='x'\nversion='0.1.0'\n",
        }))
        .unwrap()
    }

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    async fn post(app: Router, body: Vec<u8>) -> (StatusCode, Value) {
        call(
            app,
            Request::post("/v1/harness")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
    }

    async fn schedule(
        store: &dyn DesignStore,
        harness: &HarnessRow,
        rid: u64,
        origin: RunOrigin,
    ) -> Result<Vec<String>, String> {
        schedule_harness_for_round(store, harness, rid, 100, 0, origin).await
    }

    fn harness_row(hotkey: &str, marker: &str) -> HarnessRow {
        HarnessRow {
            id: format!("harness-{marker}"),
            miner_hotkey: hotkey.to_owned(),
            agent_py: format!("def run(task, llm, out):\n    pass  # {marker}\n"),
            pyproject_toml: "[project]\nname='x'\nversion='0.1.0'\n".into(),
            extra_files: BTreeMap::new(),
            active: true,
            eliminated_until_round: 0,
            created_at_ms: 0,
        }
    }

    /// A harness that participates in every round of a UTC day must never be
    /// quota-blocked: the organizer dispatches `rounds/day × prompts/round`
    /// runs, which used to overrun a shared 10-run/day ceiling after ~3 rounds.
    #[tokio::test]
    async fn full_day_of_scheduled_rounds_is_never_quota_blocked() {
        let (st, _g) = app_state(None);
        let harness = harness_row(&hk(0xAA), "day");
        st.store.insert_harness(&harness).await.unwrap();

        let per_round = design_challenge_task::prompts_per_round();
        let rounds = design_challenge_task::rounds_per_day_effective();
        let base = round_id_at(now_secs());
        let mut total = 0usize;
        for i in 0..rounds {
            let ids = schedule(st.store.as_ref(), &harness, base + i, RunOrigin::Scheduled)
                .await
                .unwrap_or_else(|e| panic!("round {i} of {rounds} refused: {e}"));
            assert_eq!(ids.len(), per_round, "round {i} scheduled a partial set");
            total += ids.len();
        }
        assert_eq!(
            total,
            usize::try_from(rounds).unwrap() * per_round,
            "every round of the day must run all its prompts"
        );

        let used = st
            .store
            .quota_get(&hk(0xAA), &utc_day(now_secs()))
            .await
            .unwrap();
        assert_eq!(usize::try_from(used.total).unwrap(), total);
        assert_eq!(used.manual, 0, "organizer work is not miner spend");
        assert!(used.scheduled() <= scheduled_daily_run_cap());
    }

    /// The miner-facing intake keeps a hard anti-spam ceiling, and burning it
    /// leaves organizer-scheduled rounds untouched.
    #[tokio::test]
    async fn manual_submissions_stay_rate_limited() {
        let (st, _g) = app_state(None);
        let per_round = u32::try_from(design_challenge_task::prompts_per_round()).unwrap();
        let cap = manual_daily_run_quota();
        let base = round_id_at(now_secs());

        let mut manual_runs = 0u32;
        let mut blocked = None;
        for i in 0..(cap / per_round + 2) {
            // A fresh digest per attempt: the worst case for intake spam.
            let harness = harness_row(&hk(0xAA), &format!("spam{i}"));
            st.store.insert_harness(&harness).await.unwrap();
            match schedule(
                st.store.as_ref(),
                &harness,
                base + u64::from(i),
                RunOrigin::Manual,
            )
            .await
            {
                Ok(ids) => manual_runs += u32::try_from(ids.len()).unwrap(),
                Err(e) => {
                    blocked = Some(e);
                    break;
                }
            }
        }
        let err = blocked.expect("manual submissions must hit the anti-spam ceiling");
        assert!(err.contains("manual"), "{err}");
        assert!(
            manual_runs <= cap,
            "{manual_runs} manual runs exceeded {cap}"
        );

        // The spent manual budget must not touch the organizer's schedule.
        let harness = harness_row(&hk(0xAA), "day");
        st.store.insert_harness(&harness).await.unwrap();
        let ids = schedule(
            st.store.as_ref(),
            &harness,
            base + 100,
            RunOrigin::Scheduled,
        )
        .await
        .expect("scheduled rounds must survive an exhausted manual quota");
        assert_eq!(ids.len(), design_challenge_task::prompts_per_round());
    }

    #[tokio::test]
    async fn accepted_runs_target_next_round() {
        let (st, _g) = app_state(None);
        let app = design_router(Arc::clone(&st));
        let (s, v) = post(app, submit_body(&hk(0xAA), "a")).await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert_eq!(v["status"], "accepted");
        let expected = round_id_at(now_secs()) + 1;
        assert_eq!(v["round_id"], expected);
        let run_ids = v["run_ids"].as_array().unwrap();
        assert_eq!(run_ids.len(), design_challenge_task::prompts_per_round());
        // Runs are queued for the *next* round, not the current one.
        let run = st
            .store
            .get_run(run_ids[0].as_str().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(run.round_id, expected);
        assert_eq!(run.status, RunStage::Queued);
        assert!(v["round_opens_at_secs"].as_u64().unwrap() > now_secs());
    }

    #[tokio::test]
    async fn gating_metagraph_and_one_max() {
        let (st, gating) = app_state(Some(vec![[0xAA; 32]]));
        let app = design_router(Arc::clone(&st));

        // Hotkey not in metagraph → 403.
        let (s, v) = post(app.clone(), submit_body(&hk(0xBB), "b")).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{v}");
        assert_eq!(v["error"], "hotkey_not_in_metagraph");

        // Member hotkey → accepted; gating row registered with uid.
        let (s, v) = post(app.clone(), submit_body(&hk(0xAA), "a")).await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let row = gating.get(CHALLENGE_ID, &hk(0xAA)).await.unwrap().unwrap();
        assert_eq!(row.state, GatingState::Registered);
        assert_eq!(row.uid, Some(0));

        // A different harness from the same hotkey → 409.
        let (s, v) = post(app.clone(), submit_body(&hk(0xAA), "v2")).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["error"], "submission_gated");

        // Identical re-POST stays idempotent.
        let (s, v) = post(app, submit_body(&hk(0xAA), "a")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["status"], "already-queued");
    }

    #[tokio::test]
    async fn same_digest_resubmit_schedules_missing_next_round() {
        let (st, _g) = app_state(None);
        let app = design_router(Arc::clone(&st));
        let body = submit_body(&hk(0xAA), "agent-v1");
        let (s, v) = post(app.clone(), body.clone()).await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let rid = v["round_id"].as_u64().unwrap();
        let harness_id = v["harness_id"].as_str().unwrap().to_owned();

        // Simulate round rollover: clear next-round runs so a re-POST must
        // recreate them (previous bug returned empty when create=false).
        let runs = st.store.runs_for_round(rid).await.unwrap();
        assert!(!runs.is_empty());
        // Memory store has no delete — schedule a *later* round via the public helper.
        let later = rid + 1;
        let harness = st.store.get_harness(&harness_id).await.unwrap().unwrap();
        let ids = schedule(st.store.as_ref(), &harness, later, RunOrigin::Scheduled)
            .await
            .unwrap();
        assert_eq!(ids.len(), design_challenge_task::prompts_per_round());
        let again = schedule(st.store.as_ref(), &harness, later, RunOrigin::Scheduled)
            .await
            .unwrap();
        assert_eq!(again, ids, "idempotent for the same round");
    }

    #[tokio::test]
    async fn view_page_serves_screenshots_only() {
        let (st, _g) = app_state(None);
        let run_id = "a".repeat(64);
        // index.html exists in the store, but produced HTML is never served.
        let png_bytes = [0x89, 0x50, 0x4E, 0x47];
        st.store
            .put_artifacts(
                &run_id,
                &[
                    (
                        "index.html".to_owned(),
                        "<html><script>alert(1)</script>miner</html>".to_owned(),
                        "raw".to_owned(),
                        "00".repeat(32),
                        42_u32,
                    ),
                    (
                        "index.png".to_owned(),
                        base64::engine::general_purpose::STANDARD.encode(png_bytes),
                        "raw".to_owned(),
                        "11".repeat(32),
                        4_u32,
                    ),
                ],
            )
            .await
            .unwrap();
        let app = design_router(Arc::clone(&st));
        for url in [
            format!("/v1/view/{run_id}/index.html"),
            format!("/v1/view/{run_id}/index"),
        ] {
            let res = app
                .clone()
                .oneshot(Request::get(&url).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::GONE, "{url}");
            assert!(res.headers().get(header::SET_COOKIE).is_none());
            let bytes = res.into_body().collect().await.unwrap().to_bytes();
            let v: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(v["error"], "gone", "{url}");
            // The stored HTML must not leak into the 410 body.
            assert!(!String::from_utf8_lossy(&bytes).contains("miner"), "{url}");
        }
        // The PNG screenshot is served as image/png.
        let res = app
            .clone()
            .oneshot(
                Request::get(format!("/v1/view/{run_id}/index.png"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            res.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("image/png")
        );
        assert_eq!(
            res.headers()
                .get(header::X_CONTENT_TYPE_OPTIONS)
                .and_then(|v| v.to_str().ok()),
            Some("nosniff")
        );
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(bytes.as_ref(), &png_bytes);
        // Unknown png → 404.
        let (s, v) = call(
            app,
            Request::get(format!("/v1/view/{run_id}/missing.png"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
    }

    #[tokio::test]
    async fn bundle_json_is_gone() {
        let (st, _g) = app_state(None);
        let run_id = "b".repeat(64);
        st.store
            .put_artifacts(
                &run_id,
                &[(
                    "index.html".to_owned(),
                    "<html>miner</html>".to_owned(),
                    "raw".to_owned(),
                    "00".repeat(32),
                    7_u32,
                )],
            )
            .await
            .unwrap();
        let app = design_router(Arc::clone(&st));
        let (s, v) = call(
            app,
            Request::get(format!("/v1/runs/{run_id}/bundle.json"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::GONE, "{v}");
        assert_eq!(v["error"], "gone");
        // The stored HTML must not leak into the response body.
        assert!(!v.to_string().contains("miner"));
    }

    #[tokio::test]
    async fn admin_requeue_schedules_current_round_once() {
        let admin_token = "test-admin-token";
        let gating = Arc::new(MemoryGatingStore::new());
        let st = Arc::new(AppState {
            store: Arc::new(MemoryDesignStore::new()),
            epoch: std::sync::atomic::AtomicU64::new(0),
            netuid: 541,
            backend_mode: "memory",
            annotator_token_hashes: vec![],
            admin_token_hashes: vec![token_hash(admin_token)],
            frame_ancestors: "'none'".into(),
            retry_max: 2,
            award_hook: None,
            gating: Some(Arc::clone(&gating) as Arc<dyn GatingStore>),
            metagraph: None,
        });
        let app = design_router(Arc::clone(&st));

        // Operator-protected like the other admin routes.
        let (s, v) = call(
            app.clone(),
            Request::post("/v1/admin/rounds/current/requeue")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{v}");

        // Two active harnesses, each auto-scheduled into the NEXT round.
        let (s, v) = post(app.clone(), submit_body(&hk(0xAA), "a")).await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let (s, v) = post(app.clone(), submit_body(&hk(0xBB), "b")).await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let current = round_id_at(now_secs());
        assert!(st.store.runs_for_round(current).await.unwrap().is_empty());

        // First requeue schedules both harnesses into the current round.
        let requeue = || {
            Request::post("/v1/admin/rounds/current/requeue")
                .header(header::AUTHORIZATION, format!("Bearer {admin_token}"))
                .body(Body::empty())
                .unwrap()
        };
        let (s, v) = call(app.clone(), requeue()).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["round_id"], current);
        assert_eq!(v["scheduled"].as_array().unwrap().len(), 2, "{v}");
        assert!(v["skipped"].as_array().unwrap().is_empty(), "{v}");
        let runs = st.store.runs_for_round(current).await.unwrap();
        assert_eq!(runs.len(), 2 * design_challenge_task::prompts_per_round());
        assert!(runs.iter().all(|r| r.status == RunStage::Queued));

        // Second call is a no-op: same run ids, no new runs, quota untouched.
        let (s, v2) = call(app.clone(), requeue()).await;
        assert_eq!(s, StatusCode::OK, "{v2}");
        assert_eq!(
            v["scheduled"].as_array().unwrap(),
            v2["scheduled"].as_array().unwrap(),
            "idempotent requeue returns the same run ids"
        );
        assert_eq!(
            st.store.runs_for_round(current).await.unwrap().len(),
            runs.len()
        );
        let day = utc_day(now_secs());
        let used = st.store.quota_get(&hk(0xAA), &day).await.unwrap();
        assert_eq!(
            usize::try_from(used.total).unwrap(),
            2 * design_challenge_task::prompts_per_round(),
            "next-round + current-round schedule only"
        );
        assert_eq!(
            usize::try_from(used.manual).unwrap(),
            design_challenge_task::prompts_per_round(),
            "only the miner's own submission is charged to the manual quota"
        );
    }

    #[tokio::test]
    async fn gating_503_until_snapshot_ready() {
        let gating = Arc::new(MemoryGatingStore::new());
        let st = Arc::new(AppState {
            store: Arc::new(MemoryDesignStore::new()),
            epoch: std::sync::atomic::AtomicU64::new(0),
            netuid: 541,
            backend_mode: "memory",
            annotator_token_hashes: vec![],
            admin_token_hashes: vec![],
            frame_ancestors: "'none'".into(),
            retry_max: 2,
            award_hook: None,
            gating: Some(gating as Arc<dyn GatingStore>),
            metagraph: Some(Arc::new(MetagraphCache::new())),
        });
        let app = design_router(st);
        let (s, v) = post(app, submit_body(&hk(0xAA), "a")).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["error"], "metagraph_unavailable");
    }
}
