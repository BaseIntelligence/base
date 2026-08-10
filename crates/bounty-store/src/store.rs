//! Async persistence contract + in-memory impl.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Chain-facing score columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalScore {
    /// Lattice score.
    Score(u64),
    /// Explicit absence (`NoScoreReasonCode` as u8).
    NoScore(u8),
}

/// Bug lifecycle (DB CHECK mirrors this list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BugStatus {
    /// Raw video accepted; awaiting compress worker.
    Uploaded,
    /// ffmpeg / fingerprint in flight.
    Processing,
    /// Agentic similar-24h review.
    AgenticReview,
    /// Novel; waiting for admin approve/reject.
    PendingAdmin,
    /// Admin-approved (+1 epoch point).
    Approved,
    /// Terminal reject (duplicate / admin / policy).
    Rejected,
    /// Infra failure (retryable or terminal).
    Failed,
}

impl BugStatus {
    /// DB string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Uploaded => "uploaded",
            Self::Processing => "processing",
            Self::AgenticReview => "agentic_review",
            Self::PendingAdmin => "pending_admin",
            Self::Approved => "approved",
            Self::Rejected => "rejected",
            Self::Failed => "failed",
        }
    }

    /// Parse DB string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "uploaded" => Some(Self::Uploaded),
            "processing" => Some(Self::Processing),
            "agentic_review" => Some(Self::AgenticReview),
            "pending_admin" => Some(Self::PendingAdmin),
            "approved" => Some(Self::Approved),
            "rejected" => Some(Self::Rejected),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }

    /// Terminal.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Approved | Self::Rejected | Self::Failed)
    }
}

/// One bug report row.
#[derive(Debug, Clone)]
pub struct BugRow {
    /// Stable id (digest hex).
    pub id: String,
    /// Miner hotkey (lowercase 64 hex).
    pub miner_hotkey: String,
    /// Owning coldkey when known at intake.
    pub miner_coldkey: Option<String>,
    /// Target app slug.
    pub app_id: String,
    /// Short title.
    pub title: String,
    /// Description body.
    pub description: String,
    /// Optional repro steps.
    pub steps: Option<String>,
    /// Lifecycle.
    pub status: BugStatus,
    /// Agentic similarity verdict JSON.
    pub agentic_verdict: Option<Value>,
    /// Nearest duplicate bug id (when flagged).
    pub nearest_id: Option<String>,
    /// Compressed video sha256 hex.
    pub video_sha256: Option<String>,
    /// Compressed video byte length.
    pub video_bytes: Option<u64>,
    /// Artifact path under bounty-artifacts volume.
    pub video_path: Option<String>,
    /// Miner-visible reject reason.
    pub reject_reason: Option<String>,
    /// Chain epoch at acceptance.
    pub epoch: u64,
    /// Created unix ms.
    pub created_at_ms: u64,
    /// Updated unix ms.
    pub updated_at_ms: u64,
}

/// Append-only stage journal entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageEvent {
    /// Stage entered/annotated.
    pub stage: String,
    /// Structured detail (bounded).
    pub detail: Option<Value>,
    /// Event unix ms (0 = server-side timestamp).
    pub at_ms: u64,
}

/// Per-(epoch, miner) score / points projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochScoreRow {
    /// Chain epoch.
    pub epoch: u64,
    /// Miner hotkey.
    pub miner_hotkey: String,
    /// Approved bug points this epoch.
    pub points: u32,
    /// Final chain-facing score.
    pub final_score: Option<FinalScore>,
}

/// Partial bug update.
#[derive(Debug, Clone, Default)]
pub struct BugPatch {
    /// New status.
    pub status: Option<BugStatus>,
    /// Agentic verdict JSON.
    pub agentic_verdict: Option<Value>,
    /// Nearest duplicate id.
    pub nearest_id: Option<String>,
    /// Video sha256.
    pub video_sha256: Option<String>,
    /// Video bytes.
    pub video_bytes: Option<u64>,
    /// Video path.
    pub video_path: Option<String>,
    /// Reject reason.
    pub reject_reason: Option<String>,
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
    /// Duplicate key.
    #[error("duplicate")]
    Duplicate,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Async persistence contract (production + sim).
#[async_trait]
pub trait BountyStore: Send + Sync + std::fmt::Debug {
    /// Insert a freshly uploaded bug (`status=uploaded`).
    async fn insert_bug(&self, row: &BugRow) -> Result<(), StoreError>;

    /// Fetch one bug.
    async fn get_bug(&self, id: &str) -> Result<Option<BugRow>, StoreError>;

    /// Apply a partial update + optional journal event.
    async fn apply(
        &self,
        id: &str,
        patch: &BugPatch,
        event: Option<&StageEvent>,
    ) -> Result<BugRow, StoreError>;

    /// Claim next `uploaded` row → `processing` (`SKIP LOCKED` in PG).
    async fn claim_next(&self) -> Result<Option<BugRow>, StoreError>;

    /// List bugs (`status` / `miner` optional filters), newest first.
    async fn list_bugs(
        &self,
        status: Option<&str>,
        miner: Option<&str>,
        limit: u32,
    ) -> Result<Vec<BugRow>, StoreError>;

    /// Similarity corpus: bugs in `approved|pending_admin|rejected` with
    /// `created_at_ms >= since_ms`, excluding same hotkey/coldkey.
    async fn list_similarity_corpus(
        &self,
        since_ms: u64,
        exclude_hotkey: &str,
        exclude_coldkey: Option<&str>,
        limit: u32,
    ) -> Result<Vec<BugRow>, StoreError>;

    /// Ascending stage journal for a bug.
    async fn events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError>;

    /// Approved point counts for an epoch (hotkey → count).
    async fn approved_points_for_epoch(
        &self,
        epoch: u64,
    ) -> Result<BTreeMap<String, u32>, StoreError>;

    /// Upsert epoch score row.
    async fn upsert_epoch_score(&self, row: &EpochScoreRow) -> Result<(), StoreError>;

    /// List epoch score rows.
    async fn list_epoch_scores(&self, epoch: u64) -> Result<Vec<EpochScoreRow>, StoreError>;
}

/// In-memory store for tests / local sim.
#[derive(Debug, Default)]
pub struct MemoryBountyStore {
    bugs: Mutex<HashMap<String, BugRow>>,
    events: Mutex<HashMap<String, VecDeque<StageEvent>>>,
    epoch_scores: Mutex<HashMap<(u64, String), EpochScoreRow>>,
    /// FIFO of uploaded ids awaiting claim.
    queue: Mutex<VecDeque<String>>,
}

impl MemoryBountyStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl BountyStore for MemoryBountyStore {
    async fn insert_bug(&self, row: &BugRow) -> Result<(), StoreError> {
        let mut bugs = self
            .bugs
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        if bugs.contains_key(&row.id) {
            return Err(StoreError::Duplicate);
        }
        bugs.insert(row.id.clone(), row.clone());
        if row.status == BugStatus::Uploaded {
            let mut q = self
                .queue
                .lock()
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            q.push_back(row.id.clone());
        }
        let mut ev = self
            .events
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        ev.entry(row.id.clone()).or_default().push_back(StageEvent {
            stage: row.status.as_str().into(),
            detail: None,
            at_ms: row.created_at_ms,
        });
        Ok(())
    }

    async fn get_bug(&self, id: &str) -> Result<Option<BugRow>, StoreError> {
        let bugs = self
            .bugs
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(bugs.get(id).cloned())
    }

    async fn apply(
        &self,
        id: &str,
        patch: &BugPatch,
        event: Option<&StageEvent>,
    ) -> Result<BugRow, StoreError> {
        let mut bugs = self
            .bugs
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let row = bugs.get_mut(id).ok_or(StoreError::NotFound)?;
        if let Some(s) = patch.status {
            row.status = s;
        }
        if let Some(v) = &patch.agentic_verdict {
            row.agentic_verdict = Some(v.clone());
        }
        if let Some(n) = &patch.nearest_id {
            row.nearest_id = Some(n.clone());
        }
        if let Some(h) = &patch.video_sha256 {
            row.video_sha256 = Some(h.clone());
        }
        if let Some(b) = patch.video_bytes {
            row.video_bytes = Some(b);
        }
        if let Some(p) = &patch.video_path {
            row.video_path = Some(p.clone());
        }
        if let Some(r) = &patch.reject_reason {
            row.reject_reason = Some(r.clone());
        }
        row.updated_at_ms = now_ms();
        let out = row.clone();
        drop(bugs);
        if let Some(ev) = event {
            let mut events = self
                .events
                .lock()
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let mut e = ev.clone();
            if e.at_ms == 0 {
                e.at_ms = now_ms();
            }
            events.entry(id.to_owned()).or_default().push_back(e);
        }
        Ok(out)
    }

    async fn claim_next(&self) -> Result<Option<BugRow>, StoreError> {
        let mut q = self
            .queue
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let mut bugs = self
            .bugs
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        while let Some(id) = q.pop_front() {
            if let Some(row) = bugs.get_mut(&id) {
                if row.status == BugStatus::Uploaded {
                    row.status = BugStatus::Processing;
                    row.updated_at_ms = now_ms();
                    return Ok(Some(row.clone()));
                }
            }
        }
        Ok(None)
    }

    async fn list_bugs(
        &self,
        status: Option<&str>,
        miner: Option<&str>,
        limit: u32,
    ) -> Result<Vec<BugRow>, StoreError> {
        let bugs = self
            .bugs
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let mut rows: Vec<_> = bugs
            .values()
            .filter(|r| status.is_none_or(|s| r.status.as_str() == s))
            .filter(|r| miner.is_none_or(|m| r.miner_hotkey == m))
            .cloned()
            .collect();
        rows.sort_by_key(|b| std::cmp::Reverse(b.created_at_ms));
        rows.truncate(limit as usize);
        Ok(rows)
    }

    async fn list_similarity_corpus(
        &self,
        since_ms: u64,
        exclude_hotkey: &str,
        exclude_coldkey: Option<&str>,
        limit: u32,
    ) -> Result<Vec<BugRow>, StoreError> {
        let bugs = self
            .bugs
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let mut rows: Vec<_> = bugs
            .values()
            .filter(|r| {
                matches!(
                    r.status,
                    BugStatus::Approved | BugStatus::PendingAdmin | BugStatus::Rejected
                )
            })
            .filter(|r| r.created_at_ms >= since_ms)
            .filter(|r| r.miner_hotkey != exclude_hotkey)
            .filter(|r| exclude_coldkey.is_none_or(|ck| r.miner_coldkey.as_deref() != Some(ck)))
            .cloned()
            .collect();
        rows.sort_by_key(|b| std::cmp::Reverse(b.created_at_ms));
        rows.truncate(limit as usize);
        Ok(rows)
    }

    async fn events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError> {
        let events = self
            .events
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(events
            .get(id)
            .map(|d| d.iter().cloned().collect())
            .unwrap_or_default())
    }

    async fn approved_points_for_epoch(
        &self,
        epoch: u64,
    ) -> Result<BTreeMap<String, u32>, StoreError> {
        let bugs = self
            .bugs
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let mut out = BTreeMap::new();
        for r in bugs.values() {
            if r.epoch == epoch && r.status == BugStatus::Approved {
                *out.entry(r.miner_hotkey.clone()).or_insert(0u32) += 1;
            }
        }
        Ok(out)
    }

    async fn upsert_epoch_score(&self, row: &EpochScoreRow) -> Result<(), StoreError> {
        let mut scores = self
            .epoch_scores
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        scores.insert((row.epoch, row.miner_hotkey.clone()), row.clone());
        Ok(())
    }

    async fn list_epoch_scores(&self, epoch: u64) -> Result<Vec<EpochScoreRow>, StoreError> {
        let scores = self
            .epoch_scores
            .lock()
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let mut rows: Vec<_> = scores
            .values()
            .filter(|r| r.epoch == epoch)
            .cloned()
            .collect();
        rows.sort_by_key(|r| r.miner_hotkey.clone());
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn sample(id: &str, status: BugStatus) -> BugRow {
        BugRow {
            id: id.into(),
            miner_hotkey: "aa".repeat(32),
            miner_coldkey: None,
            app_id: "demo".into(),
            title: "t".into(),
            description: "d".into(),
            steps: None,
            status,
            agentic_verdict: None,
            nearest_id: None,
            video_sha256: None,
            video_bytes: None,
            video_path: None,
            reject_reason: None,
            epoch: 1,
            created_at_ms: 1_000,
            updated_at_ms: 1_000,
        }
    }

    #[tokio::test]
    async fn memory_insert_claim_approve_points() {
        let s = MemoryBountyStore::new();
        s.insert_bug(&sample("b1", BugStatus::Uploaded))
            .await
            .unwrap();
        let claimed = s.claim_next().await.unwrap().unwrap();
        assert_eq!(claimed.status, BugStatus::Processing);
        s.apply(
            "b1",
            &BugPatch {
                status: Some(BugStatus::Approved),
                ..BugPatch::default()
            },
            Some(&StageEvent {
                stage: "approved".into(),
                detail: None,
                at_ms: 0,
            }),
        )
        .await
        .unwrap();
        let pts = s.approved_points_for_epoch(1).await.unwrap();
        assert_eq!(pts.get(&"aa".repeat(32)), Some(&1));
        assert_eq!(s.events("b1").await.unwrap().len(), 2);
    }
}
