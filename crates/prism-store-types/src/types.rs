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
