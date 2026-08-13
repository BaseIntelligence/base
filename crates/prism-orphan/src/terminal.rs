//! Shared terminal / measure teardown helpers for orchestrator + reconcile.

use std::sync::Arc;
use std::time::Duration;

use bundle::NoScoreReasonCode;
use prism_lium::{EvalJobBackend, EvalReceipt, RemoteExecResult};
use prism_lium_payer::PayerBackendFactory;
use prism_pipeline::gating_key;
use prism_review::SimilarityVerdict;
use prism_store::{FinalScore, PrismStore, Stage, StageEvent, StatePatch, SubmissionState};
use submission_gating::{GatingState, GatingStore};
use tracing::{info, warn};

/// Terminal `failed` + gating `blocked`.
pub async fn fail_terminal(
    store: &dyn PrismStore,
    gating: Option<&Arc<dyn GatingStore>>,
    row: &SubmissionState,
    class: &str,
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
                detail: Some(serde_json::json!({"class": class, "error": msg})),
                at_ms: 0,
            }),
        )
        .await;
    if let Some(g) = gating {
        let _ = g
            .set_terminal(
                &gating_key(row.arch_id.as_deref()),
                &row.miner_hotkey,
                GatingState::Blocked,
                Some(class),
            )
            .await;
    }
}

/// Gating `rejected` for cheat-class terminals.
pub async fn reject_gating(gating: Option<&Arc<dyn GatingStore>>, row: &SubmissionState) {
    if let Some(g) = gating {
        let _ = g
            .set_terminal(
                &gating_key(row.arch_id.as_deref()),
                &row.miner_hotkey,
                GatingState::Rejected,
                None,
            )
            .await;
    }
}

/// Terminal Score(0) reject before any Lium rent.
pub async fn reject_pre_pod(
    store: &dyn PrismStore,
    gating: Option<&Arc<dyn GatingStore>>,
    row: &SubmissionState,
    similarity: Option<SimilarityVerdict>,
    detail: Option<serde_json::Value>,
    error_detail: String,
) {
    let _ = store
        .apply(
            &row.id,
            &StatePatch {
                status: Some(Stage::Rejected),
                final_score: Some(FinalScore::Score(0)),
                similarity,
                error_detail: Some(error_detail),
                ..StatePatch::default()
            },
            Some(&StageEvent {
                stage: Stage::Rejected,
                detail,
                at_ms: 0,
            }),
        )
        .await;
    reject_gating(gating, row).await;
}

/// Harvest artifacts, terminate pod, drop vault key, build receipt.
pub async fn finish_measure(
    backend: &Arc<dyn EvalJobBackend>,
    payer: Option<&PayerBackendFactory>,
    id: &str,
    row: &SubmissionState,
    pod_id: &str,
    provider: String,
    image_digest: String,
    metrics: Result<RemoteExecResult, prism_lium::LiumError>,
) -> Result<(RemoteExecResult, EvalReceipt), String> {
    if let Ok(ref m) = metrics {
        let dest = prism_lium::artifact_dir_for(id);
        match backend
            .harvest_artifacts(pod_id, &dest, id.as_bytes(), m.n_params)
            .await
        {
            Ok(path) => {
                info!(
                    submission_id = %id,
                    path = %path.display(),
                    n_params = ?m.n_params,
                    "checkpoint secure-received"
                );
            }
            Err(e) => {
                warn!(submission_id = %id, error = %e, "checkpoint secure receive failed");
            }
        }
    }
    if let Err(e) = backend.terminate(pod_id).await {
        warn!(error = %e, %pod_id, "terminate failed");
    }
    let mut termination_verified = backend.verify_terminated(pod_id).await.unwrap_or(false);
    if !termination_verified {
        tokio::time::sleep(Duration::from_secs(5)).await;
        termination_verified = backend.verify_terminated(pod_id).await.unwrap_or(false);
    }
    if let Some(p) = payer {
        p.vault.remove(id);
    }
    let receipt = EvalReceipt {
        provider,
        pod_id: pod_id.to_owned(),
        image_digest,
        submission_hash: EvalReceipt::hash_submission(&row.architecture_py, &row.training_py),
        metrics_hash: metrics.as_ref().ok().map_or_else(
            || "none".into(),
            |m| EvalReceipt::hash_metrics_bytes(&serde_json::to_vec(m).unwrap_or_default()),
        ),
        termination_verified,
    };
    let metrics = metrics.map_err(|e| format!("exec: {e}"))?;
    Ok((metrics, receipt))
}
