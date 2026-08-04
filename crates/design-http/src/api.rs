//! HTTP API: miner / viewer / admin winners / annotate (deprecated) / ops.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use design_harness::{harness_id, validate_bundle, HarnessBundle};
use design_prompts::{load_prompt_set, prompt_set_digest, select_prompts_for_round};
use design_sanitize::viewer_headers;
use design_store::{
    DesignStore, HarnessRow, RoundAward, RunStage, RunState, StageEvent, StoreError, StorePatch,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use design_challenge_task::{
    round_id_at, CHALLENGE_ID, DAILY_RUN_QUOTA, MIN_ANNOTATIONS_PER_PAIR, ROUND_SECS, RUN_ID_DOMAIN,
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

async fn post_harness(
    State(st): State<Arc<AppState>>,
    body: Result<Json<HarnessBundle>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "invalid_json", &e.body_text()),
    };
    if let Err(e) = validate_bundle(&req) {
        return json_err(StatusCode::BAD_REQUEST, "invalid_harness", &e.to_string());
    }
    let id = harness_id(&req);
    let row = HarnessRow {
        id: id.clone(),
        miner_hotkey: req.miner_hotkey.trim().to_ascii_lowercase(),
        agent_py: req.agent_py,
        pyproject_toml: req.pyproject_toml,
        extra_files: req.extra_files,
        active: true,
        eliminated_until_round: 0,
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
    let run_ids = match schedule_runs(&st, &harness).await {
        Ok(ids) => ids,
        Err(e) => return json_err(StatusCode::CONFLICT, "schedule", &e),
    };
    let rid = round_id_at(now_secs());
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

async fn schedule_runs(st: &AppState, harness: &HarnessRow) -> Result<Vec<String>, String> {
    let secs = now_secs();
    let rid = round_id_at(secs);
    if harness.eliminated_until_round > rid {
        return Err(format!(
            "eliminated until round {}",
            harness.eliminated_until_round
        ));
    }
    let day = utc_day(secs);
    let used = st
        .store
        .quota_get(&harness.miner_hotkey, &day)
        .await
        .map_err(|e| e.to_string())?;
    // Ensure round row exists.
    if st
        .store
        .get_round(rid)
        .await
        .map_err(|e| e.to_string())?
        .is_none()
    {
        let opens = rid * ROUND_SECS;
        let epoch = st.epoch.load(std::sync::atomic::Ordering::Relaxed);
        st.store
            .insert_round(&design_store::RoundRow {
                round_id: rid,
                epoch,
                netuid: st.netuid,
                prompt_set_digest: prompt_set_digest(),
                status: "open".into(),
                opens_at_secs: opens,
                closes_at_secs: opens + ROUND_SECS,
            })
            .await
            .map_err(|e| e.to_string())?;
    }
    let prompts = select_prompts_for_round(rid).map_err(|e| e.to_string())?;
    let existing = st
        .store
        .runs_for_round(rid)
        .await
        .map_err(|e| e.to_string())?;
    let mut run_ids: Vec<String> = existing
        .iter()
        .filter(|r| r.harness_id == harness.id)
        .map(|r| r.id.clone())
        .collect();
    if !run_ids.is_empty() {
        return Ok(run_ids);
    }
    if used >= DAILY_RUN_QUOTA {
        return Err("daily quota exceeded".into());
    }
    let remaining = DAILY_RUN_QUOTA.saturating_sub(used);
    let n = (prompts.len() as u32).min(remaining);
    for p in prompts.into_iter().take(n as usize) {
        let run_id = make_run_id(rid, &harness.id, &p.id);
        if st
            .store
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
        st.store.insert_run(&row).await.map_err(|e| e.to_string())?;
        let _ = st
            .store
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
        let _ = st.store.quota_bump(&harness.miner_hotkey, &day, 1).await;
        run_ids.push(run_id);
    }
    Ok(run_ids)
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
        Ok(used) => Json(json!({
            "miner_hotkey": hotkey,
            "day": day,
            "runs_used": used,
            "limit": DAILY_RUN_QUOTA,
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
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

async fn get_bundle_json(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.store.list_pages(&id).await {
        Ok(pages) => {
            let mut map = BTreeMap::new();
            for p in pages {
                map.insert(p.path, p.sanitized_html);
            }
            Json(json!({"run_id": id, "pages": map})).into_response()
        }
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

async fn view_page(
    State(st): State<Arc<AppState>>,
    Path((id, page)): Path<(String, String)>,
) -> Response {
    let path = if page.ends_with(".html") {
        page
    } else {
        format!("{page}.html")
    };
    match st.store.get_page(&id, &path).await {
        Ok(Some(html)) => {
            let mut headers = HeaderMap::new();
            for (k, v) in viewer_headers(&st.frame_ancestors) {
                if let (Ok(name), Ok(val)) = (
                    header::HeaderName::try_from(k),
                    header::HeaderValue::try_from(v),
                ) {
                    headers.insert(name, val);
                }
            }
            headers.insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("text/html; charset=utf-8"),
            );
            (StatusCode::OK, headers, html).into_response()
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
