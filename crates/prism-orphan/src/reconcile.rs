//! Resume-first mid-flight reconcile; fail only when unreattachable.

use std::sync::Arc;

use bundle::NoScoreReasonCode;
use prism_lium::EvalJobBackend;
use prism_lium_payer::PayerBackendFactory;
use prism_pipeline::{mid_pod_resume, resume_measurement};
use prism_store::{FinalScore, PrismStore, Stage, StageEvent, StatePatch, SubmissionState};
use submission_gating::{GatingState, GatingStore};
use tracing::{info, warn};

/// Event / gating class for control-plane detach (infra — miner may resubmit).
pub const CLASS_ORPHAN: &str = "install";
/// Boot path reason token (surfaced in `error_detail` + FE banner).
pub const REASON_CONTROL_PLANE_RESTART: &str = "control_plane_restart";
/// Detach path when the worker died without a process restart.
pub const REASON_HARNESS_DETACHED: &str = "harness_detached";

/// Counts from one reconcile pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReconcileReport {
    /// Provisioning/running rows failed as orphans.
    pub failed: u32,
    /// Post-measure review rows requeued.
    pub requeued: u32,
    /// Mid-pod rows requeued for resume (pod left running).
    pub resumed: u32,
    /// Pods where terminate was attempted.
    pub terminate_attempts: u32,
}

/// All non-terminal mid-flight rows (provisioning → scoring).
pub async fn midflight_rows(store: &dyn PrismStore) -> Result<Vec<SubmissionState>, String> {
    let mut out = Vec::new();
    for st in [
        "provisioning",
        "running",
        "llm_review",
        "similarity",
        "scoring",
    ] {
        let batch = store
            .list(Some(st), None, 1000)
            .await
            .map_err(|e| e.to_string())?;
        out.extend(batch);
    }
    Ok(out)
}

/// Reconcile orphans once.
///
/// Resume-first for `provisioning`/`running` with a live pod + reattachable
/// payer/SSH: requeue to `queued` **keeping** `pod_id` (no terminate).
/// Fail-closed only when the pod is dead or there is no way to reattach.
pub async fn reconcile_once(
    store: &dyn PrismStore,
    active: &crate::ActiveJobs,
    payer: Option<&PayerBackendFactory>,
    operator: Arc<dyn EvalJobBackend>,
    grace_secs: u64,
    boot: bool,
    gating: Option<&Arc<dyn GatingStore>>,
) -> Result<ReconcileReport, String> {
    let reason = if boot {
        REASON_CONTROL_PLANE_RESTART
    } else {
        REASON_HARNESS_DETACHED
    };
    let mut report = ReconcileReport::default();
    for row in midflight_rows(store).await? {
        if should_skip(&row, active, grace_secs, boot) {
            continue;
        }
        if resume_measurement(&row).is_some()
            && matches!(
                row.status,
                Stage::LlmReview | Stage::Similarity | Stage::Scoring
            )
        {
            if store.reset_for_retry(&row.id, false).await.is_ok() {
                let _ = store
                    .apply(
                        &row.id,
                        &StatePatch::default(),
                        Some(&StageEvent {
                            stage: Stage::Queued,
                            detail: Some(serde_json::json!({
                                "orphan_requeue": true,
                                "reason": reason,
                            })),
                            at_ms: 0,
                        }),
                    )
                    .await;
                report.requeued = report.requeued.saturating_add(1);
                info!(submission_id = %row.id, reason, "orphan requeue (post-measure)");
            }
            continue;
        }
        if mid_pod_resume(&row) {
            match try_resume(store, payer, Arc::clone(&operator), &row, reason).await {
                Ok(true) => {
                    report.resumed = report.resumed.saturating_add(1);
                    continue;
                }
                Ok(false) => {}
                Err(e) => {
                    warn!(submission_id = %row.id, error = %e, "pod resume probe failed");
                }
            }
        }
        // Pre-pod (screens / claim / not yet rented): never fail-orphan.
        // A restart here has no Lium instance; fail-closed with `pod (none)`
        // burned the miner slot and asked them to stop a pod that does not exist.
        if row.pod_id.is_none() {
            if requeue_pre_pod(store, &row, reason).await {
                report.requeued = report.requeued.saturating_add(1);
            }
            continue;
        }
        // Unreattachable mid-pod: fail + best-effort stop.
        let mut terminated = false;
        let key_present = payer.is_some_and(|p| p.vault.get(&row.id).is_some());
        if let Some(pod) = row.pod_id.as_deref() {
            let be = match payer {
                Some(p) => p
                    .resolve(&row.id, Arc::clone(&operator))
                    .unwrap_or_else(|e| {
                        warn!(submission_id = %row.id, error = %e, "orphan: payer resolve");
                        Arc::clone(&operator)
                    }),
                None => Arc::clone(&operator),
            };
            report.terminate_attempts = report.terminate_attempts.saturating_add(1);
            if be.terminate(pod).await.is_ok() {
                let _ = be.verify_terminated(pod).await;
                terminated = true;
            }
        }
        if let Some(p) = payer {
            p.vault.remove(&row.id);
        }
        let msg = orphan_message(reason, row.pod_id.as_deref(), key_present, terminated);
        fail_orphan(store, gating, &row, &msg).await;
        report.failed = report.failed.saturating_add(1);
        info!(submission_id = %row.id, reason, %terminated, "orphan marked failed");
    }
    Ok(report)
}

async fn requeue_pre_pod(store: &dyn PrismStore, row: &SubmissionState, reason: &str) -> bool {
    let ok = store
        .apply(
            &row.id,
            &StatePatch {
                status: Some(Stage::Queued),
                error_detail: None,
                ..StatePatch::default()
            },
            Some(&StageEvent {
                stage: Stage::Queued,
                detail: Some(serde_json::json!({
                    "pre_pod_requeue": true,
                    "reason": reason,
                })),
                at_ms: 0,
            }),
        )
        .await
        .is_ok();
    if ok {
        info!(submission_id = %row.id, reason, "pre-pod mid-flight requeued");
    }
    ok
}

/// Probe pod + vault; on success requeue to `queued` keeping `pod_id`.
async fn try_resume(
    store: &dyn PrismStore,
    payer: Option<&PayerBackendFactory>,
    operator: Arc<dyn EvalJobBackend>,
    row: &SubmissionState,
    reason: &str,
) -> Result<bool, String> {
    let Some(pod) = row.pod_id.as_deref() else {
        return Ok(false);
    };
    if let Some(p) = payer {
        if p.vault.get(&row.id).is_none() && !p.allow_operator_fallback {
            return Ok(false);
        }
    }
    let be = match payer {
        Some(p) => p.resolve(&row.id, Arc::clone(&operator))?,
        None => operator,
    };
    let alive = be.instance_running(pod).await.map_err(|e| e.to_string())?;
    if !alive {
        return Ok(false);
    }
    // Keep pod_id / screens; workers claim Queued and call resume_eval.
    store
        .apply(
            &row.id,
            &StatePatch {
                status: Some(Stage::Queued),
                error_detail: None,
                ..StatePatch::default()
            },
            Some(&StageEvent {
                stage: Stage::Queued,
                detail: Some(serde_json::json!({
                    "pod_resume": true,
                    "reason": reason,
                    "pod_id": pod,
                })),
                at_ms: 0,
            }),
        )
        .await
        .map_err(|e| e.to_string())?;
    info!(submission_id = %row.id, %pod, reason, "mid-pod resume queued (pod left running)");
    Ok(true)
}

fn should_skip(
    row: &SubmissionState,
    active: &crate::ActiveJobs,
    grace_secs: u64,
    boot: bool,
) -> bool {
    // Live worker hold: skip unless heartbeat is extremely stale (wedged task).
    if active.contains(&row.id) {
        const WEDGE_SECS: u64 = 30 * 60;
        return active.age_secs(&row.id).is_none_or(|age| age < WEDGE_SECS);
    }
    // Process restart: every inactive mid-flight row is considered immediately.
    if boot {
        return false;
    }
    // Worker died without restart: wait for grace since last DB update/heartbeat.
    row_age_ok(row, grace_secs)
}

fn row_age_ok(row: &SubmissionState, grace_secs: u64) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    now.saturating_sub(row.updated_at_ms) < grace_secs.saturating_mul(1000)
}

fn orphan_message(
    reason: &str,
    pod_id: Option<&str>,
    key_present: bool,
    terminated: bool,
) -> String {
    let pod = pod_id.unwrap_or("(none)");
    if terminated {
        format!(
            "{reason}: harness detached after control-plane loss; Lium pod {pod} stop requested. Resubmit with X-Lium-Api-Key."
        )
    } else if key_present {
        format!(
            "{reason}: harness detached; could not confirm stop for pod {pod}. Stop the pod on Lium if it is still running, then resubmit with X-Lium-Api-Key."
        )
    } else {
        format!(
            "{reason}: harness detached and miner Lium key was not in vault (restart). Stop your Lium pod {pod} manually if it is still billing, then resubmit with X-Lium-Api-Key."
        )
    }
}

async fn fail_orphan(
    store: &dyn PrismStore,
    gating: Option<&Arc<dyn GatingStore>>,
    row: &SubmissionState,
    msg: &str,
) {
    let _ = store
        .apply(
            &row.id,
            &StatePatch {
                status: Some(Stage::Failed),
                error_detail: Some(msg.to_owned()),
                final_score: Some(FinalScore::NoScore(
                    NoScoreReasonCode::ChallengeInternal as u8,
                )),
                ..StatePatch::default()
            },
            Some(&StageEvent {
                stage: Stage::Failed,
                detail: Some(serde_json::json!({
                    "class": CLASS_ORPHAN,
                    "error": msg,
                    "orphan": true,
                })),
                at_ms: 0,
            }),
        )
        .await;
    if let Some(g) = gating {
        let key = prism_pipeline::gating_key(row.arch_id.as_deref());
        let _ = g
            .set_terminal(
                &key,
                &row.miner_hotkey,
                GatingState::Blocked,
                Some(CLASS_ORPHAN),
            )
            .await;
    }
}
