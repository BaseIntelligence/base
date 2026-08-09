//! Subnet-facing aggregate stats / dashboard JSON.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use design_challenge_task::{
    daily_run_quota, manual_daily_run_quota, round_id_at, round_secs, scheduled_daily_run_cap,
    CHALLENGE_ID,
};
use design_prompts::prompt_set_digest;
use design_store::{DesignStore, RunStage};
use serde_json::{json, Value};

use crate::api::AppState;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn utc_day(secs: u64) -> String {
    let days = secs / 86_400;
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

fn count_by_status(runs: &[design_store::RunState]) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for r in runs {
        *m.entry(r.status.as_str()).or_insert(0) += 1;
    }
    m
}

/// `GET /v1/stats` — compact aggregate for polling UIs.
pub async fn get_stats(State(st): State<Arc<AppState>>) -> Response {
    let secs = now_secs();
    let rid = round_id_at(secs);
    let runs = st.store.list_runs(None, 500).await.unwrap_or_default();
    let by_status = count_by_status(&runs);
    let round = st.store.get_round(rid).await.ok().flatten();
    let ratings = st.store.ratings_for_round(rid).await.unwrap_or_default();
    let elim = ratings
        .iter()
        .filter(|r| {
            // Eliminated miners still appear via harness flag; count Score(0) with rating 0
            // is noisy — use harness elimination when listing later. Here: bottom-rated count placeholder.
            r.rating == 0 && r.wins + r.losses > 0
        })
        .count();
    let opens = rid * round_secs();
    Json(json!({
        "challenge_id": CHALLENGE_ID,
        "backend": st.backend_mode,
        "epoch": st.epoch.load(std::sync::atomic::Ordering::Relaxed),
        "netuid": st.netuid,
        "round": {
            "round_id": rid,
            "status": round.as_ref().map(|r| r.status.clone()).unwrap_or_else(|| "open".into()),
            "opens_at_secs": opens,
            "closes_at_secs": opens + round_secs(),
            "seconds_remaining": opens.saturating_add(round_secs()).saturating_sub(secs),
            "prompt_set_digest": prompt_set_digest(),
        },
        "queue": {
            "queued": by_status.get("queued").copied().unwrap_or(0),
            "installing": by_status.get("installing").copied().unwrap_or(0),
            "running": by_status.get("running").copied().unwrap_or(0),
            "sanitizing": by_status.get("sanitizing").copied().unwrap_or(0),
            "agentic_review": by_status.get("agentic_review").copied().unwrap_or(0),
            "awaiting_admin": by_status.get("awaiting_admin").copied().unwrap_or(0),
            "awaiting_annotation": by_status.get("awaiting_annotation").copied().unwrap_or(0),
            "scored": by_status.get("scored").copied().unwrap_or(0),
            "failed": by_status.get("failed").copied().unwrap_or(0),
            "total_recent": runs.len(),
        },
        "ratings_count": ratings.len(),
        "elimination_signal_count": elim,
        "agents": st.store.count_harness_miners().await.unwrap_or(0),
        "daily_run_quota": daily_run_quota(),
        "manual_daily_run_quota": manual_daily_run_quota(),
        "scheduled_daily_run_cap": scheduled_daily_run_cap(),
    }))
    .into_response()
}

/// Cap for dashboard `recent_runs` (site gallery + activity feed).
const RECENT_RUNS_LIMIT: usize = 40;
/// Wider scan so a queue flood of brand-new rows cannot hide screenshot runs.
const RECENT_RUNS_SCAN: u32 = 500;

/// Stages that normally carry a captured `index.png` after sanitize.
fn likely_has_screenshot(status: RunStage) -> bool {
    matches!(
        status,
        RunStage::AgenticReview
            | RunStage::AwaitingAdmin
            | RunStage::AwaitingAnnotation
            | RunStage::Scored
    )
}

/// Prefer post-sanitize / scored runs (gallery candidates), then fill with the
/// newest remaining rows so the feed still reflects live queue activity.
fn select_recent_runs(runs: &[design_store::RunState], limit: usize) -> Vec<Value> {
    let mut viewable = Vec::new();
    let mut other = Vec::new();
    for r in runs {
        if likely_has_screenshot(r.status) {
            viewable.push(r);
        } else {
            other.push(r);
        }
    }
    viewable
        .into_iter()
        .chain(other)
        .take(limit)
        .map(|r| {
            let (prompt_title, _) = prompt_fields(&r.prompt_id);
            json!({
                "id": r.id,
                "status": r.status.as_str(),
                "round_id": r.round_id,
                "harness_id": r.harness_id,
                "prompt_id": r.prompt_id,
                "prompt_title": prompt_title,
                "error_detail": r.error_detail,
                "created_at_ms": r.created_at_ms,
                "updated_at_ms": r.updated_at_ms,
            })
        })
        .collect()
}

/// `GET /v1/dashboard` — one-shot UI payload (status + leaderboard + recent jobs).
pub async fn get_dashboard(State(st): State<Arc<AppState>>) -> Response {
    let secs = now_secs();
    let rid = round_id_at(secs);
    let runs = st
        .store
        .list_runs(None, RECENT_RUNS_SCAN)
        .await
        .unwrap_or_default();
    let by_status = count_by_status(&runs);
    let round = st.store.get_round(rid).await.ok().flatten();
    let ratings = st.store.ratings_for_round(rid).await.unwrap_or_default();
    let agents = st.store.count_harness_miners().await.unwrap_or(0);
    let prev = if rid > 0 {
        st.store
            .ratings_for_round(rid - 1)
            .await
            .unwrap_or_default()
    } else {
        vec![]
    };
    let opens = rid * round_secs();
    let jobs = select_recent_runs(&runs, RECENT_RUNS_LIMIT);
    let lb = |rows: &[design_store::RatingRow]| -> Vec<Value> {
        rows.iter()
            .map(|r| {
                json!({
                    "miner_hotkey": r.miner_hotkey,
                    "rating": r.rating,
                    "wins": r.wins,
                    "losses": r.losses,
                    "final_score": match &r.final_score {
                        Some(design_store::FinalScore::Score(v)) => json!({"score": v}),
                        Some(design_store::FinalScore::NoScore(c)) => json!({"no_score": c}),
                        None => Value::Null,
                    },
                })
            })
            .collect()
    };
    Json(json!({
        "challenge_id": CHALLENGE_ID,
        "backend": st.backend_mode,
        "epoch": st.epoch.load(std::sync::atomic::Ordering::Relaxed),
        "netuid": st.netuid,
        "agents": agents,
        "round": {
            "round_id": rid,
            "status": round.as_ref().map(|r| r.status.clone()).unwrap_or_else(|| "open".into()),
            "opens_at_secs": opens,
            "closes_at_secs": opens + round_secs(),
            "seconds_remaining": opens.saturating_add(round_secs()).saturating_sub(secs),
            "prompt_set_digest": prompt_set_digest(),
        },
        "queue": by_status,
        "leaderboard": {
            "current_round": rid,
            "ratings": lb(&ratings),
            "previous_round": rid.saturating_sub(1),
            "previous_ratings": lb(&prev),
        },
        "recent_runs": jobs,
        "daily_run_quota": daily_run_quota(),
        "manual_daily_run_quota": manual_daily_run_quota(),
        "scheduled_daily_run_cap": scheduled_daily_run_cap(),
        "poll_hint_ms": 1000,
    }))
    .into_response()
}

/// `GET /v1/miners/{hotkey}` — harnesses, runs, quota, ratings for one miner.
pub async fn get_miner(State(st): State<Arc<AppState>>, Path(hotkey): Path<String>) -> Response {
    let hk = hotkey.trim().to_ascii_lowercase();
    let secs = now_secs();
    let rid = round_id_at(secs);
    let day = utc_day(secs);
    let harnesses = match st.store.list_harnesses(&hk).await {
        Ok(h) => h,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let used = st.store.quota_get(&hk, &day).await.unwrap_or_default();
    // All recent generations for this miner (not only the current round) so
    // public agent history can list every design run the harness produced.
    let hid: std::collections::HashSet<_> = harnesses.iter().map(|h| h.id.clone()).collect();
    let recent = st.store.list_runs(None, 200).await.unwrap_or_default();
    let runs: Vec<Value> = recent
        .into_iter()
        .filter(|r| hid.contains(&r.harness_id))
        .map(|r| {
            let (prompt_title, _) = prompt_fields(&r.prompt_id);
            json!({
                "id": r.id,
                "status": r.status.as_str(),
                "prompt_id": r.prompt_id,
                "prompt_title": prompt_title,
                "round_id": r.round_id,
                "harness_id": r.harness_id,
                "terminal": r.status.is_terminal()
                    || r.status == RunStage::AwaitingAdmin
                    || r.status == RunStage::AwaitingAnnotation,
                "error_detail": r.error_detail,
                "artifact_digest": r.artifact_digest,
                "created_at_ms": r.created_at_ms,
                "updated_at_ms": r.updated_at_ms,
            })
        })
        .collect();
    let rating = st
        .store
        .ratings_for_round(rid)
        .await
        .unwrap_or_default()
        .into_iter()
        .find(|r| r.miner_hotkey == hk);
    Json(json!({
        "miner_hotkey": hk,
        "round_id": rid,
        "quota": crate::api::quota_json(&hk, &day, used),
        "harnesses": harnesses.iter().map(|h| json!({
            "id": h.id,
            "active": h.active,
            "eliminated_until_round": h.eliminated_until_round,
        })).collect::<Vec<_>>(),
        "runs": runs,
        "rating": rating.map(|r| json!({
            "rating": r.rating,
            "wins": r.wins,
            "losses": r.losses,
            "final_score": match r.final_score {
                Some(design_store::FinalScore::Score(v)) => json!({"score": v}),
                Some(design_store::FinalScore::NoScore(c)) => json!({"no_score": c}),
                None => Value::Null,
            },
        })),
    }))
    .into_response()
}

/// Pinned-bank prompt for `prompt_id` as `(title, full brief)`; both `None`
/// when the id is not in the bank (rotated bank / synthetic id).
fn prompt_fields(prompt_id: &str) -> (Option<String>, Option<String>) {
    design_prompts::load_bank()
        .ok()
        .and_then(|bank| bank.into_iter().find(|p| p.id == prompt_id))
        .map_or((None, None), |p| (Some(p.title), Some(p.prompt)))
}

/// Helper used by unit tests / callers that have a store handle.
pub async fn run_status_json(store: &dyn DesignStore, id: &str) -> Result<Value, String> {
    let r = store
        .get_run(id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "not found".to_owned())?;
    let pages = store.list_pages(id).await.unwrap_or_default();
    let events = store.run_events(id).await.unwrap_or_default();
    let (prompt_title, prompt) = prompt_fields(&r.prompt_id);
    let miner_hotkey = store
        .get_harness(&r.harness_id)
        .await
        .ok()
        .flatten()
        .map(|h| h.miner_hotkey);
    Ok(json!({
        "id": r.id,
        "round_id": r.round_id,
        "harness_id": r.harness_id,
        "miner_hotkey": miner_hotkey,
        "prompt_id": r.prompt_id,
        "prompt_title": prompt_title,
        "prompt": prompt,
        "status": r.status.as_str(),
        "stage": r.status.as_str(),
        "terminal": r.status.is_terminal(),
        "artifact_digest": r.artifact_digest,
        "sanitize_report": r.sanitize_report,
        "agentic_verdict": r.agentic_verdict,
        "error_detail": r.error_detail,
        "reject_reason": r.reject_reason,
        "attempt_epoch": r.attempt_epoch,
        "awaiting_admin_epoch": r.awaiting_admin_epoch,
        "retry_count": r.retry_count,
        "created_at_ms": r.created_at_ms,
        "updated_at_ms": r.updated_at_ms,
        "final_score": match &r.final_score {
            Some(design_store::FinalScore::Score(v)) => json!({"score": v}),
            Some(design_store::FinalScore::NoScore(c)) => json!({"no_score": c}),
            None => Value::Null,
        },
        "pages": pages.iter().map(|p| json!({
            "path": p.path,
            "bytes": p.bytes,
            "raw_sha256": p.raw_sha256,
        })).collect::<Vec<_>>(),
        "screenshot_url": pages
            .iter()
            .any(|p| p.path == "index.png")
            .then(|| format!("/challenge/design/v1/view/{id}/index.png")),
        "event_count": events.len(),
        "admin_ready": r.status == RunStage::AwaitingAdmin,
        "annotation_ready": r.status == RunStage::AwaitingAnnotation,
    }))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use design_store::{RunStage, RunState};

    use super::select_recent_runs;

    fn run(id: &str, status: RunStage, created_at_ms: u64) -> RunState {
        RunState {
            id: id.into(),
            round_id: 1,
            harness_id: "h".into(),
            prompt_id: "p01".into(),
            status,
            artifact_digest: None,
            sanitize_report: None,
            agentic_verdict: None,
            error_detail: None,
            reject_reason: None,
            attempt_epoch: None,
            awaiting_admin_epoch: None,
            final_score: None,
            retry_count: 0,
            created_at_ms,
            updated_at_ms: created_at_ms,
        }
    }

    #[test]
    fn select_recent_runs_prefers_screenshot_stages() {
        // Newest-first scan: a flood of queued rows would otherwise fill the
        // site gallery window and hide the awaiting_admin candidate.
        let runs = vec![
            run("q1", RunStage::Queued, 100),
            run("q2", RunStage::Queued, 99),
            run("q3", RunStage::Queued, 98),
            run("a1", RunStage::AwaitingAdmin, 97),
            run("s1", RunStage::Scored, 96),
        ];
        let jobs = select_recent_runs(&runs, 3);
        let ids: Vec<&str> = jobs
            .iter()
            .filter_map(|j| j.get("id").and_then(|v| v.as_str()))
            .collect();
        assert_eq!(ids, vec!["a1", "s1", "q1"]);
    }
}
