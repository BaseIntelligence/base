//! Post-score registry hooks, run by the orchestrator after a submission
//! finalizes with a real measured score (all in one place so the
//! orchestrator stays within its LOC cap):
//!
//! 1. **Architecture publish** — architecture submissions (no `arch_id` yet)
//!    register their `architecture.py` in the published registry on first
//!    score; the row is back-linked. Training-only rows already reference
//!    their arch. Idempotent on digest (simultaneous duplicates share the
//!    first registration).
//! 2. **Arch best bpb** — lower-wins update feeding owner credit + the
//!    public leaderboard (any trainer's result counts).
//! 3. **Top-model publish** — when the row's bpb is a new global best
//!    (≤ best scored bpb ever AND < last published bpb), publish to GitHub
//!    and journal the publication. No-op without a configured publisher.

use std::sync::Arc;

use prism_store::{
    ArchitectureRecord, FinalScore, PrismStore, PublishArchOutcome, StatePatch, SubmissionState,
    TopModelPublication,
};
use tracing::{info, warn};

use crate::publish::{TopModelPublisher, TopModelRequest, TOPMODEL_REPO_PATH};

/// Run registry + top-model bookkeeping for one finalized row.
#[allow(clippy::too_many_lines)]
pub async fn post_score_hooks(
    store: &Arc<dyn PrismStore>,
    publisher: Option<&TopModelPublisher>,
    row: &SubmissionState,
) {
    let (Some(bpb), Some(FinalScore::Score(v))) = (row.bpb, &row.final_score) else {
        return;
    };
    if *v == 0 {
        return; // cheat / copy-gate zero never publishes nor sets arch best
    }

    // (1) Architecture publish (architecture submissions only).
    let mut arch_id = row.arch_id.clone();
    if arch_id.is_none() {
        let rec = ArchitectureRecord {
            arch_id: prism_pipeline::arch_id_for(&row.architecture_py),
            owner_hotkey: row.miner_hotkey.clone(),
            arch_digest: prism_pipeline::arch_digest(&row.architecture_py),
            architecture_py: row.architecture_py.clone(),
            source_submission: row.id.clone(),
            best_bpb: Some(bpb),
            created_at_ms: row.created_at_ms,
        };
        match store.publish_arch(&rec).await {
            Ok(PublishArchOutcome::Published) => {
                info!(submission_id = %row.id, arch_id = %rec.arch_id, "architecture published");
                arch_id = Some(rec.arch_id.clone());
            }
            Ok(PublishArchOutcome::Duplicate(existing)) => {
                info!(submission_id = %row.id, arch_id = %existing, "architecture digest already registered");
                arch_id = Some(existing);
            }
            Err(e) => {
                warn!(submission_id = %row.id, error = %e, "arch publish failed (row still scored)");
            }
        }
        if let Some(a) = &arch_id {
            let _ = store
                .apply(
                    &row.id,
                    &StatePatch {
                        arch_id: Some(a.clone()),
                        ..StatePatch::default()
                    },
                    None,
                )
                .await;
        }
    }

    // (2) Arch best bpb (lower wins; any trainer's result counts).
    if let Some(a) = &arch_id {
        if let Err(e) = store.note_arch_best_bpb(a, bpb).await {
            warn!(submission_id = %row.id, arch_id = %a, error = %e, "arch best_bpb update failed");
        }
    }

    // (3) Top-model publish on a new global best — GitHub (optional) + HF
    // (optional). Both require a verified secure-receive receipt when
    // `PRISM_TOPMODEL_REQUIRE_WEIGHTS=1` (default); source-only is opt-in.
    // Recipe 2.0 / AutoModel only — legacy 1.x never becomes the published
    // top-model champion (historical FE rows stay in Postgres).
    if !row.weight_eligible() {
        info!(
            submission_id = %row.id,
            "top-model: skip legacy / non-AutoModel row (weight-ineligible)"
        );
        return;
    }
    let last = store.last_publication_bpb().await.unwrap_or(None);
    let global = store.best_scored_bpb().await.unwrap_or(None);
    let is_global_best = global.is_some_and(|g| bpb <= g);
    let beats_published = last.is_none_or(|l| bpb < l);
    if !(is_global_best && beats_published) {
        return;
    }
    let ckpt = match prism_artifacts::verify_parked(&row.id) {
        Ok(receipt) => {
            let p = prism_artifacts::artifact_dir_for(&row.id).join(&receipt.path);
            info!(
                submission_id = %row.id,
                sha = %receipt.sha256,
                source = %receipt.source,
                "top-model: verified artifact receipt"
            );
            Some(p)
        }
        Err(e) => {
            warn!(
                submission_id = %row.id,
                error = %e,
                "top-model: secure receive missing/invalid (refusing weight publish)"
            );
            None
        }
    };
    let req = TopModelRequest {
        submission_id: row.id.clone(),
        arch_id: arch_id.clone(),
        owner_hotkey: row.miner_hotkey.clone(),
        bpb,
        architecture_py: row.architecture_py.clone(),
        training_py: row.training_py.clone(),
        metrics_json: row.metrics_json.clone(),
        checkpoint_path: ckpt,
        // Novel AutoModel / tree modules so HF cards reload without a stock
        // transformers arch already on the Hub.
        extra_files: crate::hf::extra_files_from_tree_blob(row.tree_blob.as_deref()),
    };
    let mut journaled = false;
    if let Some(publisher) = publisher {
        match publisher.publish(&req).await {
            Ok(sha) => {
                info!(submission_id = %row.id, bpb, commit = %sha, "top model published to GitHub");
                let rec = TopModelPublication {
                    submission_id: row.id.clone(),
                    arch_id: arch_id.clone(),
                    owner_hotkey: row.miner_hotkey.clone(),
                    bpb,
                    repo_path: TOPMODEL_REPO_PATH.to_owned(),
                    commit_sha: Some(sha),
                };
                if let Err(e) = store.record_publication(&rec).await {
                    warn!(submission_id = %row.id, error = %e, "publication journal failed");
                } else {
                    journaled = true;
                }
            }
            Err(e) => {
                warn!(submission_id = %row.id, error = %e, "top-model publish failed (will retry on next best)");
            }
        }
    }
    if let Some(hf) = crate::hf::HfTopModelPublisher::from_env() {
        match hf.publish(&req).await {
            Ok(oid) => {
                if !journaled {
                    let rec = TopModelPublication {
                        submission_id: row.id.clone(),
                        arch_id,
                        owner_hotkey: row.miner_hotkey.clone(),
                        bpb,
                        repo_path: format!("hf:{}", hf.repo_id()),
                        commit_sha: Some(oid),
                    };
                    if let Err(e) = store.record_publication(&rec).await {
                        warn!(submission_id = %row.id, error = %e, "hf publication journal failed");
                    }
                }
            }
            Err(e) => {
                warn!(
                    submission_id = %row.id,
                    error = %e,
                    "top-model HuggingFace publish failed (will retry on next best)"
                );
            }
        }
    }
}
