//! Orchestrator persistence: single truth behind the public status API.
//!
//! Production impl: [`crate::dbprism::DbPrismStore`]. This async trait +
//! memory impl keep the challenge testable offline.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use prism_lium::{EvalReceipt, TelemetryPoint};
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
    /// (created_at ordered) — terminal, LLM review skipped, Score(0).
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

/// Async persistence contract (production + sim).
#[async_trait]
pub trait PrismStore: Send + Sync + std::fmt::Debug {
    /// Insert the acceptance row (status=queued).
    async fn insert_queued(&self, row: &SubmissionState) -> Result<(), StoreError>;

    /// Fetch one row.
    async fn get(&self, id: &str) -> Result<Option<SubmissionState>, StoreError>;

    /// Atomically claim the next queued row for work (`SKIP LOCKED`, moves
    /// to provisioning).
    async fn claim_next(&self) -> Result<Option<SubmissionState>, StoreError>;

    /// Apply a partial update + journal the event (one logical step).
    async fn apply(
        &self,
        id: &str,
        update: &StatePatch,
        event: Option<&StageEvent>,
    ) -> Result<SubmissionState, StoreError>;

    /// Retry reset: clears exec/score fields and re-queues a failed row.
    /// Implementations MUST actually null the pod/receipt/score columns
    /// (SQL) or reset the in-memory row mirror-equivalently.
    async fn reset_for_retry(&self, id: &str) -> Result<SubmissionState, StoreError>;

    /// Newsfeed listing for the API (`status` / `miner` optional filters).
    async fn list(
        &self,
        status: Option<&str>,
        miner: Option<&str>,
        limit: u32,
    ) -> Result<Vec<SubmissionState>, StoreError>;

    /// Ascending journal.
    async fn events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError>;

    /// Per-step telemetry series (migration 0009 `prism_telemetry`), ordered
    /// by step. Empty when the harness reported nothing (pre-1.1.0 recipes).
    async fn telemetry(&self, id: &str) -> Result<Vec<TelemetryPoint>, StoreError>;

    /// Latest final score per miner (for D24 leaf emission), epoch-filtered.
    async fn scores_for_epoch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<(String, FinalScore)>, StoreError>;

    /// Non-terminal rows beyond grace — for the stuck sweep.
    async fn list_stuck(&self, grace_secs: u64) -> Result<Vec<SubmissionState>, StoreError>;
}

/// In-memory store (CI / sim).
#[derive(Debug, Default)]
pub struct MemoryPrismStore {
    rows: Mutex<VecDeque<SubmissionState>>,
    events: Mutex<Vec<(String, StageEvent)>>,
    telemetry: Mutex<std::collections::HashMap<String, Vec<TelemetryPoint>>>,
}

impl MemoryPrismStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Extract the miner-reported loss series from a `RemoteExecResult` JSON blob
/// (`telemetry.loss_series`). Empty when absent or malformed — the harness
/// controls the shape, so a parse miss means a pre-telemetry recipe.
pub(crate) fn telemetry_from_metrics(metrics_json: &serde_json::Value) -> Vec<TelemetryPoint> {
    metrics_json
        .get("telemetry")
        .and_then(|t| t.get("loss_series"))
        .and_then(|s| serde_json::from_value::<Vec<TelemetryPoint>>(s.clone()).ok())
        .unwrap_or_default()
}

#[async_trait]
impl PrismStore for MemoryPrismStore {
    async fn insert_queued(&self, row: &SubmissionState) -> Result<(), StoreError> {
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        // Match Postgres unique(id): reject duplicates so a second POST cannot
        // enqueue another billable Lium claim for the same submission_id.
        if rows.iter().any(|r| r.id == row.id) {
            return Err(StoreError::Backend("duplicate submission_id".into()));
        }
        rows.push_back(row.clone());
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<SubmissionState>, StoreError> {
        Ok(self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .find(|r| r.id == id)
            .cloned())
    }

    async fn claim_next(&self) -> Result<Option<SubmissionState>, StoreError> {
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let pos = rows.iter().position(|r| r.status == Stage::Queued);
        let Some(i) = pos else { return Ok(None) };
        let mut row = rows.remove(i).ok_or(StoreError::Backend("pop".into()))?;
        row.status = Stage::Provisioning;
        rows.push_back(row.clone());
        Ok(Some(row))
    }

    async fn apply(
        &self,
        id: &str,
        update: &StatePatch,
        event: Option<&StageEvent>,
    ) -> Result<SubmissionState, StoreError> {
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let row = rows
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(StoreError::NotFound)?;
        if let Some(s) = update.status {
            row.status = s;
        }
        if let Some(v) = &update.pod_id {
            row.pod_id = Some(v.clone());
        }
        if let Some(v) = &update.pod_provider {
            row.pod_provider = Some(v.clone());
        }
        if let Some(v) = &update.receipt {
            row.receipt = Some(v.clone());
        }
        if let Some(v) = &update.metrics_json {
            row.metrics_json = Some(v.clone());
        }
        if let Some(v) = update.bpb {
            row.bpb = Some(v);
        }
        if let Some(v) = &update.review {
            row.review = Some(v.clone());
        }
        if let Some(v) = &update.similarity {
            row.similarity = Some(v.clone());
        }
        if let Some(v) = &update.final_score {
            row.final_score = Some(v.clone());
        }
        if let Some(v) = &update.error_detail {
            row.error_detail = Some(v.clone());
        }
        row.retry_count = row.retry_count.saturating_add(update.retry_bump);
        row.updated_at_ms = now_ms();
        let out = row.clone();
        drop(rows);
        if let Some(m) = &update.metrics_json {
            let series = telemetry_from_metrics(m);
            if !series.is_empty() {
                self.telemetry
                    .lock()
                    .map_err(|_| StoreError::Backend("poison".into()))?
                    .insert(id.to_owned(), series);
            }
        }
        if let Some(e) = event {
            self.events
                .lock()
                .map_err(|_| StoreError::Backend("poison".into()))?
                .push((id.to_owned(), e.clone()));
        }
        Ok(out)
    }

    async fn reset_for_retry(&self, id: &str) -> Result<SubmissionState, StoreError> {
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let row = rows
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(StoreError::NotFound)?;
        row.status = Stage::Queued;
        row.pod_id = None;
        row.pod_provider = None;
        row.receipt = None;
        row.metrics_json = None;
        row.bpb = None;
        row.review = None;
        row.similarity = None;
        row.final_score = None;
        row.error_detail = None;
        row.retry_count = row.retry_count.saturating_add(1);
        row.updated_at_ms = now_ms();
        let out = row.clone();
        drop(rows);
        self.telemetry
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .remove(id);
        Ok(out)
    }

    async fn list(
        &self,
        status: Option<&str>,
        miner: Option<&str>,
        limit: u32,
    ) -> Result<Vec<SubmissionState>, StoreError> {
        let st = status.and_then(Stage::parse);
        let miner_norm = miner.map(|m| m.trim().to_ascii_lowercase());
        let mut v: Vec<_> = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|r| st.is_none() || Some(r.status) == st)
            .filter(|r| {
                miner_norm
                    .as_ref()
                    .is_none_or(|m| r.miner_hotkey.to_ascii_lowercase() == *m)
            })
            .cloned()
            .collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.created_at_ms));
        v.truncate(limit as usize);
        Ok(v)
    }

    async fn events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError> {
        Ok(self
            .events
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|(k, _)| k == id)
            .map(|(_, e)| e.clone())
            .collect())
    }

    async fn telemetry(&self, id: &str) -> Result<Vec<TelemetryPoint>, StoreError> {
        Ok(self
            .telemetry
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(id)
            .cloned()
            .unwrap_or_default())
    }

    async fn scores_for_epoch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<(String, FinalScore)>, StoreError> {
        let mut by_miner: std::collections::BTreeMap<String, FinalScore> =
            std::collections::BTreeMap::new();
        for r in self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
        {
            if r.netuid == netuid && r.epoch == epoch {
                if let Some(fs) = &r.final_score {
                    by_miner.insert(r.miner_hotkey.clone(), fs.clone());
                }
            }
        }
        Ok(by_miner.into_iter().collect())
    }

    async fn list_stuck(&self, grace_secs: u64) -> Result<Vec<SubmissionState>, StoreError> {
        let cutoff = now_ms().saturating_sub(grace_secs.saturating_mul(1000));
        Ok(self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|r| !r.status.is_terminal() && r.updated_at_ms < cutoff)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn row(id: &str, hotkey: &str) -> SubmissionState {
        SubmissionState {
            id: id.into(),
            miner_hotkey: hotkey.into(),
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            architecture_py: "a".into(),
            training_py: "t".into(),
            label: None,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: None,
            bpb: None,
            review: None,
            similarity: None,
            final_score: None,
            retry_count: 0,
            error_detail: None,
            created_at_ms: 1,
            updated_at_ms: 1,
        }
    }

    #[tokio::test]
    async fn claim_is_fifo_and_moves_stage() {
        let s = MemoryPrismStore::new();
        s.insert_queued(&row("a", "11")).await.unwrap();
        s.insert_queued(&row("b", "22")).await.unwrap();
        let first = s.claim_next().await.unwrap().unwrap();
        assert_eq!(first.id, "a");
        assert_eq!(first.status, Stage::Provisioning);
        let row = s.get("a").await.unwrap().unwrap();
        assert_eq!(row.status, Stage::Provisioning);
    }

    #[tokio::test]
    async fn insert_queued_rejects_duplicate_id() {
        let s = MemoryPrismStore::new();
        s.insert_queued(&row("a", "11")).await.unwrap();
        let err = s.insert_queued(&row("a", "11")).await.unwrap_err();
        assert!(
            err.to_string().contains("duplicate"),
            "expected duplicate error, got {err}"
        );
        assert_eq!(s.list(None, None, 10).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn list_filters_by_miner_case_insensitive() {
        let s = MemoryPrismStore::new();
        s.insert_queued(&row("a", "AABBCC")).await.unwrap();
        s.insert_queued(&row("b", "ddeeff")).await.unwrap();
        let only_a = s.list(None, Some("aabbcc"), 10).await.unwrap();
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].id, "a");
    }

    #[tokio::test]
    async fn apply_patches_and_journals() {
        let s = MemoryPrismStore::new();
        s.insert_queued(&row("a", "11")).await.unwrap();
        let ev = StageEvent {
            stage: Stage::Provisioning,
            detail: Some(serde_json::json!({"pod": "p1"})),
            at_ms: 5,
        };
        s.apply(
            "a",
            &StatePatch {
                status: Some(Stage::Running),
                ..StatePatch::default()
            },
            Some(&ev),
        )
        .await
        .unwrap();
        assert_eq!(s.events("a").await.unwrap().len(), 1);
        let row = s.get("a").await.unwrap().unwrap();
        assert_eq!(row.status, Stage::Running);
    }

    #[tokio::test]
    async fn metrics_json_populates_telemetry_and_n_params() {
        let s = MemoryPrismStore::new();
        s.insert_queued(&row("a", "11")).await.unwrap();
        let metrics = serde_json::json!({
            "bpb": 1.5,
            "tokens_seen": 2048,
            "wall_clock_seconds": 12.0,
            "gpu_type": "SIM",
            "notes": "sim-eval",
            "n_params": 12_000_000_u64,
            "telemetry": {
                "finish_reason": "finish_evaluation",
                "report_count": 2,
                "loss_series": [
                    {"step": 1, "loss": 4.0, "grad_norm": 1.0, "at_secs": 0.5},
                    {"step": 2, "loss": 3.0}
                ]
            }
        });
        s.apply(
            "a",
            &StatePatch {
                metrics_json: Some(metrics),
                ..StatePatch::default()
            },
            None,
        )
        .await
        .unwrap();
        let row = s.get("a").await.unwrap().unwrap();
        assert_eq!(row.n_params(), Some(12_000_000));
        let series = s.telemetry("a").await.unwrap();
        assert_eq!(series.len(), 2);
        assert_eq!(series[0].step, 1);
        assert!((series[1].loss - 3.0).abs() < f64::EPSILON);
        // Retry clears both the blob and the derived series.
        s.apply(
            "a",
            &StatePatch {
                status: Some(Stage::Failed),
                ..StatePatch::default()
            },
            None,
        )
        .await
        .unwrap();
        s.reset_for_retry("a").await.unwrap();
        assert!(s.telemetry("a").await.unwrap().is_empty());
        assert!(s.get("a").await.unwrap().unwrap().metrics_json.is_none());
    }

    #[tokio::test]
    async fn metrics_without_telemetry_leaves_series_empty() {
        let s = MemoryPrismStore::new();
        s.insert_queued(&row("a", "11")).await.unwrap();
        s.apply(
            "a",
            &StatePatch {
                metrics_json: Some(serde_json::json!({"bpb": 2.0})),
                ..StatePatch::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(s.telemetry("a").await.unwrap().is_empty());
        assert!(s.get("a").await.unwrap().unwrap().n_params().is_none());
    }
}
