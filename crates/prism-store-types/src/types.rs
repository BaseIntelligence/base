//! The persistence data contract: submission row, stage lattice, patch,
//! error taxonomy, and the registry / epoch / top-model records.

use prism_lium_types::EvalReceipt;
use prism_review::{ReviewVerdict, SimilarityVerdict};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable hex id for a submission.
pub type SubmissionId = String;

/// Score columns mirrored from the leaf (chain-facing truth).
#[derive(Debug, Clone, PartialEq)]
pub enum FinalScore {
    /// Real lattice score.
    Score(u64),
    /// Explicit absence (code mirrors `NoScoreReasonCode`).
    NoScore(u8),
}

/// One submission's state (single DB row).
#[derive(Debug, Clone)]
pub struct SubmissionState {
    /// Contract-bytes sha256.
    pub id: SubmissionId,
    /// Miner hotkey hex (64).
    pub miner_hotkey: String,
    /// Owning coldkey (lowercase 64 hex), when known at intake.
    pub miner_coldkey: Option<String>,
    /// Chain epoch at acceptance.
    pub epoch: u64,
    /// Netuid.
    pub netuid: u16,
    /// Lifecycle.
    pub status: Stage,
    /// architecture.py bytes (review/similarity/replay input).
    pub architecture_py: String,
    /// training.py bytes.
    pub training_py: String,
    /// Packed v3 source tree (`prism_tree::StagedTree::pack`), when the
    /// submission was a multi-file ZIP. `None` for legacy two-script rows.
    pub tree_blob: Option<Vec<u8>>,
    /// Optional human label.
    pub label: Option<String>,
    /// Lium pod id.
    pub pod_id: Option<String>,
    /// Pod provider (`lium` / `sim`).
    pub pod_provider: Option<String>,
    /// Receipt set once the pod phase completes.
    pub receipt: Option<EvalReceipt>,
    /// Full harness metrics blob (`RemoteExecResult` JSON, incl. telemetry).
    pub metrics_json: Option<serde_json::Value>,
    /// Measured bpb.
    pub bpb: Option<f64>,
    /// Registry architecture this row trains (`None` = architecture
    /// submission pre-publish / legacy). Set at intake for training-only
    /// entries; back-linked on publish for architecture submissions.
    pub arch_id: Option<String>,
    /// LLM review verdict.
    pub review: Option<ReviewVerdict>,
    /// Similarity verdict.
    pub similarity: Option<SimilarityVerdict>,
    /// Final chain-facing score.
    pub final_score: Option<FinalScore>,
    /// Attempts.
    pub retry_count: u32,
    /// Error detail on `failed`.
    pub error_detail: Option<String>,
    /// Accepted at (unix ms).
    pub created_at_ms: u64,
    /// Updated at (unix ms).
    pub updated_at_ms: u64,
}

impl SubmissionState {
    /// Model parameter count from the harness metrics blob, when measured.
    #[must_use]
    pub fn n_params(&self) -> Option<u64> {
        self.metrics_json.as_ref()?.get("n_params")?.as_u64()
    }
}

/// Lifecycle (the DB CHECK mirrors this list — keep in sync).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Queued.
    Queued,
    /// Lium rent in flight.
    Provisioning,
    /// Harness running on pod.
    Running,
    /// LLM review.
    LlmReview,
    /// Similarity review.
    Similarity,
    /// Scoring + leaf emit.
    Scoring,
    /// Terminated + verified.
    Terminated,
    /// Failure path.
    Failed,
    /// Pre-LLM copy gate: byte/AST copy of a strictly-earlier architecture
    /// (`created_at` ordered) — terminal, LLM review skipped, Score(0).
    Rejected,
}

impl Stage {
    /// DB string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Provisioning => "provisioning",
            Self::Running => "running",
            Self::LlmReview => "llm_review",
            Self::Similarity => "similarity",
            Self::Scoring => "scoring",
            Self::Terminated => "terminated",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
        }
    }

    /// Parse DB string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "queued" => Some(Self::Queued),
            "provisioning" => Some(Self::Provisioning),
            "running" => Some(Self::Running),
            "llm_review" => Some(Self::LlmReview),
            "similarity" => Some(Self::Similarity),
            "scoring" => Some(Self::Scoring),
            "terminated" => Some(Self::Terminated),
            "failed" => Some(Self::Failed),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }

    /// Terminal (replay writes a new row).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Terminated | Self::Failed | Self::Rejected)
    }
}

/// One journal entry (`prism_stage_event`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageEvent {
    /// Stage entered/annotated.
    pub stage: Stage,
    /// Structured detail (bounded).
    pub detail: Option<serde_json::Value>,
    /// Event unix ms (0 = server-side timestamp).
    pub at_ms: u64,
}

/// Persistence errors.
#[derive(Debug, Error)]
pub enum StoreError {
    /// Backend fault.
    #[error("backend: {0}")]
    Backend(String),
    /// Row vanished mid-flight.
    #[error("not found")]
    NotFound,
}

/// Partial update bundle.
#[derive(Debug, Clone, Default)]
pub struct StatePatch {
    /// New status.
    pub status: Option<Stage>,
    /// Pod id.
    pub pod_id: Option<String>,
    /// Pod provider.
    pub pod_provider: Option<String>,
    /// Receipt.
    pub receipt: Option<EvalReceipt>,
    /// Full harness metrics blob (carries telemetry for the per-step table).
    pub metrics_json: Option<serde_json::Value>,
    /// bpb.
    pub bpb: Option<f64>,
    /// Registry architecture link (set-once at intake / publish back-link).
    pub arch_id: Option<String>,
    /// Review verdict.
    pub review: Option<ReviewVerdict>,
    /// Similarity verdict.
    pub similarity: Option<SimilarityVerdict>,
    /// Final score.
    pub final_score: Option<FinalScore>,
    /// Error detail.
    pub error_detail: Option<String>,
    /// Bump retry counter.
    pub retry_bump: u32,
}

/// Published architecture (migration 0010 `prism_architecture`).
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureRecord {
    /// `arch_<hex16>`.
    pub arch_id: String,
    /// Owner miner hotkey (64 hex).
    pub owner_hotkey: String,
    /// Full sha256 hex of `architecture_py` (unique).
    pub arch_digest: String,
    /// Architecture source (contract-validated at publish).
    pub architecture_py: String,
    /// Originating `prism_submission.id`.
    pub source_submission: String,
    /// Best measured bpb on this arch across all trainers (lower is better).
    pub best_bpb: Option<f64>,
    /// Creation unix ms.
    pub created_at_ms: u64,
}

/// Outcome of an architecture publish attempt (idempotent on digest).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublishArchOutcome {
    /// New row inserted.
    Published,
    /// Same digest already registered (simultaneous duplicate) — existing id.
    Duplicate(String),
}

/// Live Prism contest id. Pre-v2.1 harvests are a **different competition**.
pub const PRISM_COMPETITION_ID: &str = "prism-v2.1";
/// Scoring generation for [`PRISM_COMPETITION_ID`] (recipe `2.1.x` lattice).
pub const PRISM_SCORING_GENERATION: u16 = 21;

/// Stamp live contest identity onto a newly harvested **2.1** metrics blob.
///
/// No-ops on recipe 1.x / 2.0 so a rematerialized old harvest cannot gain
/// generation 21. Does not overwrite an existing id/generation.
pub fn stamp_v21_competition(metrics: &mut serde_json::Value) {
    if !recipe_is_v21(metrics) {
        return;
    }
    let Some(obj) = metrics.as_object_mut() else {
        return;
    };
    obj.entry("competition_id")
        .or_insert_with(|| serde_json::Value::String(PRISM_COMPETITION_ID.into()));
    obj.entry("scoring_generation")
        .or_insert_with(|| serde_json::json!(PRISM_SCORING_GENERATION));
}

/// Fail-closed emission eligibility: **recipe 2.1 / generation 21 only**.
///
/// Recipe `2.0.0`, `AutoModel` pin alone, and `.prism/automodel.patch` trees
/// are the previous contest — they must not win WTA or carry. Unknown /
/// missing signals are **not** eligible.
#[must_use]
pub fn submission_weight_eligible(
    metrics_json: Option<&serde_json::Value>,
    _tree_blob: Option<&[u8]>,
) -> bool {
    metrics_json.is_some_and(metrics_are_v21_competition)
}

fn metrics_are_v21_competition(m: &serde_json::Value) -> bool {
    competition_id_is_live(m) || scoring_generation_is_live(m) || recipe_is_v21(m)
}

fn metric_str<'a>(m: &'a serde_json::Value, path: &str) -> Option<&'a str> {
    if path.starts_with('/') {
        m.pointer(path).and_then(serde_json::Value::as_str)
    } else {
        m.get(path).and_then(serde_json::Value::as_str)
    }
}

fn recipe_is_v21(m: &serde_json::Value) -> bool {
    for path in ["recipe", "/pod_manifest/recipe"] {
        if let Some(s) = metric_str(m, path) {
            let mut parts = s.trim().split('.');
            if parts.next() == Some("2") && parts.next() == Some("1") {
                return true;
            }
        }
    }
    false
}

fn competition_id_is_live(m: &serde_json::Value) -> bool {
    for path in ["competition_id", "/pod_manifest/competition_id"] {
        if metric_str(m, path).is_some_and(|s| s.trim() == PRISM_COMPETITION_ID) {
            return true;
        }
    }
    false
}

fn scoring_generation_is_live(m: &serde_json::Value) -> bool {
    for path in ["scoring_generation", "/pod_manifest/scoring_generation"] {
        let v = if path.starts_with('/') {
            m.pointer(path)
        } else {
            m.get(path)
        };
        let Some(v) = v else { continue };
        if v.as_u64() == Some(u64::from(PRISM_SCORING_GENERATION)) {
            return true;
        }
        if v.as_str()
            .and_then(|s| s.trim().parse::<u16>().ok())
            .is_some_and(|n| n == PRISM_SCORING_GENERATION)
        {
            return true;
        }
    }
    false
}

impl SubmissionState {
    /// Whether this row may receive on-chain Prism weight (v2.1 contest).
    #[must_use]
    pub fn weight_eligible(&self) -> bool {
        submission_weight_eligible(self.metrics_json.as_ref(), self.tree_blob.as_deref())
    }
}

/// One scored row inside an emission batch (competition scoring input).
/// Batches are epoch-close assignments (see [`PrismStore::assign_emit_batch`]),
/// not acceptance-epoch lookups: a row's acceptance `epoch` is intake metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochScoreRow {
    /// Miner hotkey.
    pub miner_hotkey: String,
    /// Registry arch this row trained (when linked).
    pub arch_id: Option<String>,
    /// Final lattice score (or absence).
    pub final_score: FinalScore,
    /// Recipe 2.1 / generation 21 only — older contests must not win WTA.
    pub weight_eligible: bool,
}

#[cfg(test)]
mod weight_eligible_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn recipe_21_metrics_eligible() {
        assert!(submission_weight_eligible(
            Some(&json!({"recipe": "2.1.0"})),
            None
        ));
        assert!(submission_weight_eligible(
            Some(&json!({"pod_manifest": {"recipe": "2.1.3"}})),
            None
        ));
    }

    #[test]
    fn competition_id_and_generation_eligible() {
        assert!(submission_weight_eligible(
            Some(&json!({"competition_id": PRISM_COMPETITION_ID})),
            None
        ));
        assert!(submission_weight_eligible(
            Some(&json!({"scoring_generation": PRISM_SCORING_GENERATION})),
            None
        ));
        assert!(submission_weight_eligible(
            Some(&json!({"scoring_generation": "21"})),
            None
        ));
    }

    #[test]
    fn old_gen_and_automodel_pin_are_ineligible() {
        assert!(!submission_weight_eligible(
            Some(&json!({"recipe": "2.0.0"})),
            None
        ));
        assert!(!submission_weight_eligible(
            Some(&json!({"pod_manifest": {"automodel_base": "automodel@v0.5.0"}})),
            None
        ));
        let mut blob = b"ustar....".to_vec();
        blob.extend_from_slice(b".prism/automodel.patch");
        assert!(!submission_weight_eligible(None, Some(&blob)));
    }

    #[test]
    fn stamp_skips_old_recipe_and_fills_new() {
        let mut old = json!({"recipe": "2.0.0"});
        stamp_v21_competition(&mut old);
        assert!(old.get("competition_id").is_none());
        assert!(old.get("scoring_generation").is_none());
        assert!(!submission_weight_eligible(Some(&old), None));
        let mut new = json!({"recipe": "2.1.0"});
        stamp_v21_competition(&mut new);
        assert_eq!(new["competition_id"], PRISM_COMPETITION_ID);
        assert_eq!(new["scoring_generation"], PRISM_SCORING_GENERATION);
        assert!(submission_weight_eligible(Some(&new), None));
    }

    #[test]
    fn legacy_and_unknown_fail_closed() {
        assert!(!submission_weight_eligible(
            Some(&json!({"recipe": "1.4.0"})),
            None
        ));
        assert!(!submission_weight_eligible(None, None));
        assert!(!submission_weight_eligible(
            Some(&json!({"bpb": 4.2})),
            Some(b"no-patch-here")
        ));
    }
}

/// Top-model publication journal row (migration 0010).
#[derive(Debug, Clone, PartialEq)]
pub struct TopModelPublication {
    /// Submission that set the global best.
    pub submission_id: String,
    /// Registry arch (when linked).
    pub arch_id: Option<String>,
    /// Miner hotkey that set the best.
    pub owner_hotkey: String,
    /// Global-best bpb at publish time.
    pub bpb: f64,
    /// Repo path published (`top-model/`).
    pub repo_path: String,
    /// GitHub commit sha of the publish (`None` = dry-run).
    pub commit_sha: Option<String>,
}
