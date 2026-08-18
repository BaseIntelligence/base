//! Master eval pipeline: provision → exec → receipt → score → terminate.

use std::sync::Arc;

use bundle::ScoreOrAbsence;
use prism_lium::{EvalJobBackend, EvalReceipt, InstanceSpec, NoScoreGate, RemoteExecResult};
use thiserror::Error;
use tracing::{info, warn};

use crate::config::PrismConfig;
use crate::score::{score_from_pipeline, PipelineOutcome};
use crate::submission::SubmissionRequest;

/// Pipeline errors.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Lium(#[from] prism_lium::LiumError),
    #[error("hotkey decode: {0}")]
    Hotkey(String),
}

/// Input for one miner eval.
#[derive(Debug, Clone)]
pub struct PipelineInput {
    pub request: SubmissionRequest,
}

/// Result of one pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub score: ScoreOrAbsence,
    pub receipt: EvalReceipt,
    pub bpb: Option<f64>,
    pub pod_id: String,
}

/// Run master-centralized eval (Sim or Real backend).
///
/// Always terminates the pod in a `finally`-equivalent path and records
/// `termination_verified` on the receipt.
///
/// # Errors
/// Backend / integrity failures (caller maps to NoScore as needed).
pub async fn run_sim_pipeline(
    backend: Arc<dyn EvalJobBackend>,
    cfg: &PrismConfig,
    input: PipelineInput,
) -> Result<PipelineResult, PipelineError> {
    run_eval_pipeline(backend, cfg, input).await
}

/// Alias for production + sim (same control flow; backend selects Real vs Sim).
pub async fn run_eval_pipeline(
    backend: Arc<dyn EvalJobBackend>,
    cfg: &PrismConfig,
    input: PipelineInput,
) -> Result<PipelineResult, PipelineError> {
    let name = format!(
        "prism-{}",
        &EvalReceipt::hash_submission(&input.request.architecture_py, &input.request.training_py)
            [..12]
    );
    let spec = InstanceSpec {
        name: name.clone(),
        max_lifetime_hours: cfg.max_lifetime_hours,
        max_price_per_hour: cfg.max_price_per_hour,
        gpu_count: prism_lium::pod_gpu_count_from_env(),
        image_digest: cfg.default_image_digest.clone(),
        ssh_public_keys: cfg.ssh_public_keys.clone(),
        ssh_key_name: Some("prism-mission-worker".into()),
        preferred_offer_id: cfg.preferred_offer_id.clone(),
        template_id: cfg.template_id.clone(),
        template_name: cfg.template_name.clone(),
    };

    let inst = backend.provision(&spec).await?;
    let pod_id = inst.id.clone();
    info!(%pod_id, "prism eval provisioned");

    let exec_result: Result<RemoteExecResult, _> = backend
        .exec_eval(
            &pod_id,
            &input.request.architecture_py,
            &input.request.training_py,
            None,
        )
        .await;

    // Always terminate
    if let Err(e) = backend.terminate(&pod_id).await {
        warn!(error = %e, %pod_id, "terminate failed");
    }
    let mut termination_verified = backend.verify_terminated(&pod_id).await.unwrap_or(false);
    if !termination_verified {
        // Second confirmation attempt — billing leak is critical.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        termination_verified = backend.verify_terminated(&pod_id).await.unwrap_or(false);
    }

    let submission_hash =
        EvalReceipt::hash_submission(&input.request.architecture_py, &input.request.training_py);

    match exec_result {
        Ok(metrics) => {
            let metrics_bytes = serde_json::to_vec(&metrics).unwrap_or_default();
            let image_digest = cfg.default_image_digest.clone().unwrap_or_default();
            let receipt = EvalReceipt {
                provider: inst.provider.clone(),
                pod_id: pod_id.clone(),
                image_digest,
                submission_hash,
                metrics_hash: EvalReceipt::hash_metrics_bytes(&metrics_bytes),
                termination_verified,
            };
            if let Err(e) = NoScoreGate::check(&receipt, cfg.require_image_digest) {
                warn!(error = %e, "integrity gate failed");
                return Ok(PipelineResult {
                    score: score_from_pipeline(&PipelineOutcome::ChallengeInternal),
                    receipt,
                    bpb: Some(metrics.bpb),
                    pod_id,
                });
            }
            let score = score_from_pipeline(&PipelineOutcome::Measured { bpb: metrics.bpb });
            Ok(PipelineResult {
                score,
                receipt,
                bpb: Some(metrics.bpb),
                pod_id,
            })
        }
        Err(e) => {
            let receipt = EvalReceipt {
                provider: inst.provider.clone(),
                pod_id: pod_id.clone(),
                image_digest: cfg.default_image_digest.clone().unwrap_or_default(),
                submission_hash,
                metrics_hash: "none".into(),
                termination_verified,
            };
            // If terminate failed, surface integrity NoScore
            if !termination_verified {
                return Ok(PipelineResult {
                    score: score_from_pipeline(&PipelineOutcome::ChallengeInternal),
                    receipt,
                    bpb: None,
                    pod_id,
                });
            }
            Err(e.into())
        }
    }
}

/// Post-run retry resume: a row still carrying its completed measurement
/// (receipt + metrics survive `reset_for_retry`) decodes back into the
/// measure-phase products — an infra retry of a review/similarity/agentic
/// stage must never re-provision a pod for an already-measured run. Decode
/// drift falls back to `None` (caller re-measures).
#[must_use]
pub fn resume_measurement(
    row: &prism_store::SubmissionState,
) -> Option<(RemoteExecResult, EvalReceipt)> {
    let receipt = row.receipt.clone()?;
    let metrics = serde_json::from_value::<RemoteExecResult>(row.metrics_json.clone()?).ok()?;
    Some((metrics, receipt))
}

/// True when a mid-pod row should reattach (has `pod_id`, no completed measure).
#[must_use]
pub fn mid_pod_resume(row: &prism_store::SubmissionState) -> bool {
    row.pod_id.is_some() && resume_measurement(row).is_none()
}

/// Store patch persisting a completed measurement (receipt + metrics + bpb);
/// the metrics blob also lands the per-step telemetry series master-side.
#[must_use]
pub fn measurement_patch(m: &RemoteExecResult, r: &EvalReceipt) -> prism_store::StatePatch {
    prism_store::StatePatch {
        receipt: Some(r.clone()),
        metrics_json: serde_json::to_value(m).ok(),
        bpb: Some(m.bpb),
        ..prism_store::StatePatch::default()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::submission::example_valid_request;
    use prism_lium::SimLiumBackend;

    #[tokio::test]
    async fn sim_pipeline_scores_and_terminates() {
        let backend: Arc<dyn EvalJobBackend> = Arc::new(SimLiumBackend::new());
        let cfg = PrismConfig::sim();
        let out = run_sim_pipeline(
            backend.clone(),
            &cfg,
            PipelineInput {
                request: example_valid_request(),
            },
        )
        .await
        .unwrap();
        assert!(out.receipt.termination_verified);
        assert!(matches!(out.score, ScoreOrAbsence::Score { value } if value > 0));
        assert!(backend.verify_terminated(&out.pod_id).await.unwrap());
    }
}
