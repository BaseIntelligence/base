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

/// Run lifecycle (DB CHECK mirrors this list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStage {
    /// Queued for sandbox.
    Queued,
    /// pip install phase.
    Installing,
    /// Harness run phase.
    Running,
    /// HTML sanitize.
    Sanitizing,
    /// Agentic anti-cheat review.
    AgenticReview,
    /// Waiting for admin winner selection (clean only).
    AwaitingAdmin,
    /// Deprecated: pairwise annotation (kept for legacy rows).
    AwaitingAnnotation,
    /// Terminal scored.
    Scored,
    /// Terminal failed.
    Failed,
    /// Terminal rejected (pre-LLM copy gate; no LLM review ran).
    Rejected,
}

impl RunStage {
    /// DB string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Installing => "installing",
            Self::Running => "running",
            Self::Sanitizing => "sanitizing",
            Self::AgenticReview => "agentic_review",
            Self::AwaitingAdmin => "awaiting_admin",
            Self::AwaitingAnnotation => "awaiting_annotation",
            Self::Scored => "scored",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
        }
    }

    /// Parse DB string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "queued" => Some(Self::Queued),
            "installing" => Some(Self::Installing),
            "running" => Some(Self::Running),
            "sanitizing" => Some(Self::Sanitizing),
            "agentic_review" => Some(Self::AgenticReview),
            "awaiting_admin" => Some(Self::AwaitingAdmin),
            "awaiting_annotation" => Some(Self::AwaitingAnnotation),
            "scored" => Some(Self::Scored),
            "failed" => Some(Self::Failed),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }

    /// Terminal.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Scored | Self::Failed | Self::Rejected)
    }
}

/// Harness row.
#[derive(Debug, Clone)]
pub struct HarnessRow {
    /// Digest id.
    pub id: String,
    /// Miner hotkey.
    pub miner_hotkey: String,
    /// agent.py.
    pub agent_py: String,
    /// pyproject.toml.
    pub pyproject_toml: String,
    /// Extra files.
    pub extra_files: BTreeMap<String, String>,
    /// Active flag.
    pub active: bool,
    /// Eliminated until round.
    pub eliminated_until_round: u64,
    /// Store-assigned creation unix ms (ordering input for the copy gate).
    pub created_at_ms: u64,
}

/// Round row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundRow {
    /// Round id.
    pub round_id: u64,
    /// Epoch.
    pub epoch: u64,
    /// Netuid.
    pub netuid: u16,
    /// Prompt set digest.
    pub prompt_set_digest: String,
    /// Status: open|closed|scoring|emitted.
    pub status: String,
    /// Opens unix secs.
    pub opens_at_secs: u64,
    /// Closes unix secs.
    pub closes_at_secs: u64,
}

/// Run row.
#[derive(Debug, Clone)]
pub struct RunState {
    /// Id.
    pub id: String,
    /// Round.
    pub round_id: u64,
    /// Harness.
    pub harness_id: String,
    /// Prompt.
    pub prompt_id: String,
    /// Status.
    pub status: RunStage,
    /// Artifact digest.
    pub artifact_digest: Option<String>,
    /// Sanitize report.
    pub sanitize_report: Option<Value>,
    /// Agentic anti-cheat verdict JSON (audit).
    pub agentic_verdict: Option<Value>,
    /// Error.
    pub error_detail: Option<String>,
    /// Final score.
    pub final_score: Option<FinalScore>,
    /// Retries.
    pub retry_count: u32,
    /// Created ms.
    pub created_at_ms: u64,
    /// Updated ms.
    pub updated_at_ms: u64,
}

/// Persisted admin award for a round (1 or 2 winner harness ids).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundAward {
    /// Round.
    pub round_id: u64,
    /// Winner harness digests (len 1 or 2).
    pub winner_harness_ids: Vec<String>,
    /// Awarded unix ms.
    pub awarded_at_ms: u64,
    /// Admin token hash (audit).
    pub admin_token_hash: Option<String>,
}

/// Pairwise comparison.
#[derive(Debug, Clone)]
pub struct PairRow {
    /// Id.
    pub id: String,
    /// Round.
    pub round_id: u64,
    /// Prompt.
    pub prompt_id: String,
    /// Run A.
    pub run_a_id: String,
    /// Run B.
    pub run_b_id: String,
}

/// Rating row.
#[derive(Debug, Clone)]
pub struct RatingRow {
    /// Round.
    pub round_id: u64,
    /// Miner.
    pub miner_hotkey: String,
    /// Elo.
    pub rating: i32,
    /// Wins.
    pub wins: u32,
    /// Losses.
    pub losses: u32,
    /// Final.
    pub final_score: Option<FinalScore>,
}

/// Sanitized page (raw never exposed via this type for viewer paths).
#[derive(Debug, Clone)]
pub struct ArtifactPage {
    /// Path.
    pub path: String,
    /// Sanitized HTML.
    pub sanitized_html: String,
    /// Raw sha256.
    pub raw_sha256: String,
    /// Raw bytes length.
    pub bytes: u32,
}

/// Stage journal entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageEvent {
    /// Stage.
    pub stage: String,
    /// Detail.
    pub detail: Option<Value>,
    /// At ms.
    pub at_ms: u64,
}

/// Partial run update.
#[derive(Debug, Clone, Default)]
pub struct StorePatch {
    /// Status.
    pub status: Option<RunStage>,
    /// Digest.
    pub artifact_digest: Option<String>,
    /// Report.
    pub sanitize_report: Option<Value>,
    /// Agentic verdict JSON.
    pub agentic_verdict: Option<Value>,
    /// Error.
    pub error_detail: Option<String>,
    /// Score.
    pub final_score: Option<FinalScore>,
    /// Retry bump.
    pub retry_bump: u32,
}

/// Store errors.
#[derive(Debug, Error)]
pub enum StoreError {
    /// Backend.
    #[error("backend: {0}")]
    Backend(String),
    /// Missing.
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

/// Persistence contract.
#[async_trait]
pub trait DesignStore: Send + Sync + std::fmt::Debug {
    /// Insert harness (duplicate → [`StoreError::Duplicate`]).
    async fn insert_harness(&self, row: &HarnessRow) -> Result<(), StoreError>;
    /// Get harness.
    async fn get_harness(&self, id: &str) -> Result<Option<HarnessRow>, StoreError>;
    /// List by miner.
    async fn list_harnesses(&self, miner: &str) -> Result<Vec<HarnessRow>, StoreError>;
    /// Set elimination.
    async fn set_eliminated(&self, id: &str, until_round: u64) -> Result<(), StoreError>;
    /// Recent harnesses for agentic corpus (newest first).
    async fn list_recent_harnesses(&self, limit: u32) -> Result<Vec<HarnessRow>, StoreError>;

    /// Insert round.
    async fn insert_round(&self, row: &RoundRow) -> Result<(), StoreError>;
    /// Get round.
    async fn get_round(&self, round_id: u64) -> Result<Option<RoundRow>, StoreError>;
    /// Update round status.
    async fn set_round_status(&self, round_id: u64, status: &str) -> Result<(), StoreError>;
    /// List rounds.
    async fn list_rounds(&self, limit: u32) -> Result<Vec<RoundRow>, StoreError>;

    /// Insert queued run.
    async fn insert_run(&self, row: &RunState) -> Result<(), StoreError>;
    /// Get run.
    async fn get_run(&self, id: &str) -> Result<Option<RunState>, StoreError>;
    /// Claim next queued → installing. Runs registered for a future round
    /// (`round_id > max_round`) are not claimable yet.
    async fn claim_next_run(&self, max_round: u64) -> Result<Option<RunState>, StoreError>;
    /// Apply patch + optional event.
    async fn apply_run(
        &self,
        id: &str,
        patch: &StorePatch,
        event: Option<&StageEvent>,
    ) -> Result<RunState, StoreError>;
    /// Reset failed → queued.
    async fn reset_run(&self, id: &str) -> Result<RunState, StoreError>;
    /// List runs.
    async fn list_runs(
        &self,
        status: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RunState>, StoreError>;
    /// Runs for round.
    async fn runs_for_round(&self, round_id: u64) -> Result<Vec<RunState>, StoreError>;
    /// Stuck runs.
    async fn list_stuck_runs(&self, grace_secs: u64) -> Result<Vec<RunState>, StoreError>;
    /// Events.
    async fn run_events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError>;

    /// Store sanitized pages (+ raw kept only in DB adapter internals).
    async fn put_artifacts(
        &self,
        run_id: &str,
        pages: &[(String, String, String, String, u32)],
    ) -> Result<(), StoreError>;
    /// Viewer pages (sanitized only).
    async fn list_pages(&self, run_id: &str) -> Result<Vec<ArtifactPage>, StoreError>;
    /// One sanitized page.
    async fn get_page(&self, run_id: &str, path: &str) -> Result<Option<String>, StoreError>;

    /// Insert pair.
    async fn insert_pair(&self, pair: &PairRow) -> Result<(), StoreError>;
    /// Next pair for annotator.
    async fn next_pair(
        &self,
        round_id: u64,
        annotator_id: &str,
        min_annotations: u32,
    ) -> Result<Option<PairRow>, StoreError>;
    /// Insert annotation.
    async fn insert_annotation(
        &self,
        pair_id: &str,
        annotator_id: &str,
        winner_run_id: &str,
    ) -> Result<(), StoreError>;
    /// Annotations as (pair_id, annotator, winner, created_ms) ordered.
    async fn annotations_for_round(
        &self,
        round_id: u64,
    ) -> Result<Vec<(String, String, String, u64)>, StoreError>;
    /// Pairs for round.
    async fn pairs_for_round(&self, round_id: u64) -> Result<Vec<PairRow>, StoreError>;

    /// Upsert rating.
    async fn upsert_rating(&self, row: &RatingRow) -> Result<(), StoreError>;
    /// Ratings for round.
    async fn ratings_for_round(&self, round_id: u64) -> Result<Vec<RatingRow>, StoreError>;
    /// Scores for epoch leaf emit.
    async fn scores_for_epoch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<(String, FinalScore)>, StoreError>;

    /// Persist admin winners (1 or 2 harness ids). Duplicate → [`StoreError::Duplicate`].
    async fn set_round_award(&self, award: &RoundAward) -> Result<(), StoreError>;
    /// Load award if present.
    async fn get_round_award(&self, round_id: u64) -> Result<Option<RoundAward>, StoreError>;
    /// Awards with `round_id` in inclusive `[from_round, to_round]`.
    async fn list_round_awards(
        &self,
        from_round: u64,
        to_round: u64,
    ) -> Result<Vec<RoundAward>, StoreError>;

    /// Quota get.
    async fn quota_get(&self, miner: &str, day: &str) -> Result<u32, StoreError>;
    /// Quota bump; returns new total.
    async fn quota_bump(&self, miner: &str, day: &str, bump: u32) -> Result<u32, StoreError>;
}

/// In-memory store.
#[derive(Debug, Default)]
pub struct MemoryDesignStore {
    harnesses: Mutex<HashMap<String, HarnessRow>>,
    rounds: Mutex<HashMap<u64, RoundRow>>,
    runs: Mutex<VecDeque<RunState>>,
    events: Mutex<Vec<(String, StageEvent)>>,
    pages: Mutex<HashMap<String, Vec<ArtifactPage>>>,
    pairs: Mutex<Vec<PairRow>>,
    annotations: Mutex<Vec<(String, String, String, u64)>>,
    ratings: Mutex<HashMap<(u64, String), RatingRow>>,
    awards: Mutex<HashMap<u64, RoundAward>>,
    quota: Mutex<HashMap<(String, String), u32>>,
}

impl MemoryDesignStore {
    /// Empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl DesignStore for MemoryDesignStore {
    async fn insert_harness(&self, row: &HarnessRow) -> Result<(), StoreError> {
        let mut m = self
            .harnesses
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        if m.contains_key(&row.id) {
            return Err(StoreError::Duplicate);
        }
        let mut row = row.clone();
        if row.created_at_ms == 0 {
            row.created_at_ms = now_ms();
        }
        m.insert(row.id.clone(), row);
        Ok(())
    }

    async fn get_harness(&self, id: &str) -> Result<Option<HarnessRow>, StoreError> {
        Ok(self
            .harnesses
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(id)
            .cloned())
    }

    async fn list_harnesses(&self, miner: &str) -> Result<Vec<HarnessRow>, StoreError> {
        Ok(self
            .harnesses
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .values()
            .filter(|h| h.miner_hotkey == miner)
            .cloned()
            .collect())
    }

    async fn set_eliminated(&self, id: &str, until_round: u64) -> Result<(), StoreError> {
        let mut m = self
            .harnesses
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let h = m.get_mut(id).ok_or(StoreError::NotFound)?;
        h.eliminated_until_round = until_round;
        Ok(())
    }

    async fn list_recent_harnesses(&self, limit: u32) -> Result<Vec<HarnessRow>, StoreError> {
        let mut v: Vec<_> = self
            .harnesses
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .values()
            .cloned()
            .collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v.truncate(limit as usize);
        Ok(v)
    }

    async fn insert_round(&self, row: &RoundRow) -> Result<(), StoreError> {
        let mut m = self
            .rounds
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        if m.contains_key(&row.round_id) {
            return Err(StoreError::Duplicate);
        }
        m.insert(row.round_id, row.clone());
        Ok(())
    }

    async fn get_round(&self, round_id: u64) -> Result<Option<RoundRow>, StoreError> {
        Ok(self
            .rounds
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(&round_id)
            .cloned())
    }

    async fn set_round_status(&self, round_id: u64, status: &str) -> Result<(), StoreError> {
        let mut m = self
            .rounds
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let r = m.get_mut(&round_id).ok_or(StoreError::NotFound)?;
        r.status = status.to_owned();
        Ok(())
    }

    async fn list_rounds(&self, limit: u32) -> Result<Vec<RoundRow>, StoreError> {
        let mut v: Vec<_> = self
            .rounds
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .values()
            .cloned()
            .collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.round_id));
        v.truncate(limit as usize);
        Ok(v)
    }

    async fn insert_run(&self, row: &RunState) -> Result<(), StoreError> {
        self.runs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .push_back(row.clone());
        Ok(())
    }

    async fn get_run(&self, id: &str) -> Result<Option<RunState>, StoreError> {
        Ok(self
            .runs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .find(|r| r.id == id)
            .cloned())
    }

    async fn claim_next_run(&self, max_round: u64) -> Result<Option<RunState>, StoreError> {
        let mut rows = self
            .runs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let pos = rows
            .iter()
            .position(|r| r.status == RunStage::Queued && r.round_id <= max_round);
        let Some(i) = pos else { return Ok(None) };
        let mut row = rows.remove(i).ok_or(StoreError::Backend("pop".into()))?;
        row.status = RunStage::Installing;
        row.updated_at_ms = now_ms();
        rows.push_back(row.clone());
        Ok(Some(row))
    }

    async fn apply_run(
        &self,
        id: &str,
        patch: &StorePatch,
        event: Option<&StageEvent>,
    ) -> Result<RunState, StoreError> {
        let mut rows = self
            .runs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let row = rows
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(StoreError::NotFound)?;
        if let Some(s) = patch.status {
            row.status = s;
        }
        if let Some(v) = &patch.artifact_digest {
            row.artifact_digest = Some(v.clone());
        }
        if let Some(v) = &patch.sanitize_report {
            row.sanitize_report = Some(v.clone());
        }
        if let Some(v) = &patch.agentic_verdict {
            row.agentic_verdict = Some(v.clone());
        }
        if let Some(v) = &patch.error_detail {
            row.error_detail = Some(v.clone());
        }
        if let Some(v) = &patch.final_score {
            row.final_score = Some(v.clone());
        }
        row.retry_count = row.retry_count.saturating_add(patch.retry_bump);
        row.updated_at_ms = now_ms();
        let out = row.clone();
        drop(rows);
        if let Some(e) = event {
            self.events
                .lock()
                .map_err(|_| StoreError::Backend("poison".into()))?
                .push((id.to_owned(), e.clone()));
        }
        Ok(out)
    }

    async fn reset_run(&self, id: &str) -> Result<RunState, StoreError> {
        let mut rows = self
            .runs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let row = rows
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(StoreError::NotFound)?;
        row.status = RunStage::Queued;
        row.artifact_digest = None;
        row.sanitize_report = None;
        row.agentic_verdict = None;
        row.error_detail = None;
        row.final_score = None;
        row.retry_count = row.retry_count.saturating_add(1);
        row.updated_at_ms = now_ms();
        Ok(row.clone())
    }

    async fn list_runs(
        &self,
        status: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RunState>, StoreError> {
        let st = status.and_then(RunStage::parse);
        let mut v: Vec<_> = self
            .runs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|r| st.is_none() || Some(r.status) == st)
            .cloned()
            .collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.created_at_ms));
        v.truncate(limit as usize);
        Ok(v)
    }

    async fn runs_for_round(&self, round_id: u64) -> Result<Vec<RunState>, StoreError> {
        Ok(self
            .runs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|r| r.round_id == round_id)
            .cloned()
            .collect())
    }

    async fn list_stuck_runs(&self, grace_secs: u64) -> Result<Vec<RunState>, StoreError> {
        let cutoff = now_ms().saturating_sub(grace_secs.saturating_mul(1000));
        Ok(self
            .runs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|r| {
                matches!(
                    r.status,
                    RunStage::Installing
                        | RunStage::Running
                        | RunStage::Sanitizing
                        | RunStage::AgenticReview
                ) && r.updated_at_ms < cutoff
            })
            .cloned()
            .collect())
    }

    async fn run_events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError> {
        Ok(self
            .events
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|(k, _)| k == id)
            .map(|(_, e)| e.clone())
            .collect())
    }

    async fn put_artifacts(
        &self,
        run_id: &str,
        pages: &[(String, String, String, String, u32)],
    ) -> Result<(), StoreError> {
        let list: Vec<ArtifactPage> = pages
            .iter()
            .map(|(path, sanitized, _raw, sha, bytes)| ArtifactPage {
                path: path.clone(),
                sanitized_html: sanitized.clone(),
                raw_sha256: sha.clone(),
                bytes: *bytes,
            })
            .collect();
        self.pages
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .insert(run_id.to_owned(), list);
        Ok(())
    }

    async fn list_pages(&self, run_id: &str) -> Result<Vec<ArtifactPage>, StoreError> {
        Ok(self
            .pages
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(run_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn get_page(&self, run_id: &str, path: &str) -> Result<Option<String>, StoreError> {
        Ok(self
            .pages
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(run_id)
            .and_then(|pages| {
                pages
                    .iter()
                    .find(|p| p.path == path)
                    .map(|p| p.sanitized_html.clone())
            }))
    }

    async fn insert_pair(&self, pair: &PairRow) -> Result<(), StoreError> {
        self.pairs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .push(pair.clone());
        Ok(())
    }

    async fn next_pair(
        &self,
        round_id: u64,
        annotator_id: &str,
        min_annotations: u32,
    ) -> Result<Option<PairRow>, StoreError> {
        let pairs = self
            .pairs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .clone();
        let anns = self
            .annotations
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .clone();
        for p in pairs.into_iter().filter(|p| p.round_id == round_id) {
            let count = anns.iter().filter(|(pid, _, _, _)| pid == &p.id).count() as u32;
            let already = anns
                .iter()
                .any(|(pid, a, _, _)| pid == &p.id && a == annotator_id);
            if !already && count < min_annotations {
                return Ok(Some(p));
            }
        }
        Ok(None)
    }

    async fn insert_annotation(
        &self,
        pair_id: &str,
        annotator_id: &str,
        winner_run_id: &str,
    ) -> Result<(), StoreError> {
        self.annotations
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .push((
                pair_id.to_owned(),
                annotator_id.to_owned(),
                winner_run_id.to_owned(),
                now_ms(),
            ));
        Ok(())
    }

    async fn annotations_for_round(
        &self,
        round_id: u64,
    ) -> Result<Vec<(String, String, String, u64)>, StoreError> {
        let pair_ids: std::collections::HashSet<_> = self
            .pairs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|p| p.round_id == round_id)
            .map(|p| p.id.clone())
            .collect();
        let mut v: Vec<_> = self
            .annotations
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|(pid, _, _, _)| pair_ids.contains(pid))
            .cloned()
            .collect();
        v.sort_by_key(|(_, _, _, t)| *t);
        Ok(v)
    }

    async fn pairs_for_round(&self, round_id: u64) -> Result<Vec<PairRow>, StoreError> {
        Ok(self
            .pairs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|p| p.round_id == round_id)
            .cloned()
            .collect())
    }

    async fn upsert_rating(&self, row: &RatingRow) -> Result<(), StoreError> {
        self.ratings
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .insert((row.round_id, row.miner_hotkey.clone()), row.clone());
        Ok(())
    }

    async fn ratings_for_round(&self, round_id: u64) -> Result<Vec<RatingRow>, StoreError> {
        let mut v: Vec<_> = self
            .ratings
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .values()
            .filter(|r| r.round_id == round_id)
            .cloned()
            .collect();
        v.sort_by(|a, b| {
            b.rating
                .cmp(&a.rating)
                .then(a.miner_hotkey.cmp(&b.miner_hotkey))
        });
        Ok(v)
    }

    async fn scores_for_epoch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<(String, FinalScore)>, StoreError> {
        let rounds: Vec<u64> = self
            .rounds
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .values()
            .filter(|r| r.netuid == netuid && r.epoch == epoch)
            .map(|r| r.round_id)
            .collect();
        let mut by: BTreeMap<String, FinalScore> = BTreeMap::new();
        let ratings = self
            .ratings
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        for ((rid, _), row) in ratings.iter() {
            if rounds.contains(rid) {
                if let Some(fs) = &row.final_score {
                    by.insert(row.miner_hotkey.clone(), fs.clone());
                }
            }
        }
        Ok(by.into_iter().collect())
    }

    async fn set_round_award(&self, award: &RoundAward) -> Result<(), StoreError> {
        let mut m = self
            .awards
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        if m.contains_key(&award.round_id) {
            return Err(StoreError::Duplicate);
        }
        m.insert(award.round_id, award.clone());
        Ok(())
    }

    async fn get_round_award(&self, round_id: u64) -> Result<Option<RoundAward>, StoreError> {
        Ok(self
            .awards
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(&round_id)
            .cloned())
    }

    async fn list_round_awards(
        &self,
        from_round: u64,
        to_round: u64,
    ) -> Result<Vec<RoundAward>, StoreError> {
        let mut out: Vec<_> = self
            .awards
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .values()
            .filter(|a| a.round_id >= from_round && a.round_id <= to_round)
            .cloned()
            .collect();
        out.sort_by_key(|a| a.round_id);
        Ok(out)
    }

    async fn quota_get(&self, miner: &str, day: &str) -> Result<u32, StoreError> {
        Ok(*self
            .quota
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(&(miner.to_owned(), day.to_owned()))
            .unwrap_or(&0))
    }

    async fn quota_bump(&self, miner: &str, day: &str, bump: u32) -> Result<u32, StoreError> {
        let mut m = self
            .quota
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let e = m.entry((miner.to_owned(), day.to_owned())).or_insert(0);
        *e = e.saturating_add(bump);
        Ok(*e)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[tokio::test]
    async fn claim_and_quota() {
        let s = MemoryDesignStore::new();
        let run = |id: &str, round_id: u64| RunState {
            id: id.into(),
            round_id,
            harness_id: "h".into(),
            prompt_id: "p01".into(),
            status: RunStage::Queued,
            artifact_digest: None,
            sanitize_report: None,
            agentic_verdict: None,
            error_detail: None,
            final_score: None,
            retry_count: 0,
            created_at_ms: 1,
            updated_at_ms: 1,
        };
        s.insert_run(&run("r1", 1)).await.unwrap();
        // Future-round run is not claimable until its round opens.
        s.insert_run(&run("r2", 5)).await.unwrap();
        let claimed = s.claim_next_run(1).await.unwrap().unwrap();
        assert_eq!(claimed.id, "r1");
        assert_eq!(claimed.status, RunStage::Installing);
        assert!(s.claim_next_run(1).await.unwrap().is_none());
        let future = s.claim_next_run(5).await.unwrap().unwrap();
        assert_eq!(future.id, "r2");
        assert_eq!(s.quota_bump("aa", "2026-08-04", 1).await.unwrap(), 1);
    }
}
