//! Orchestrator persistence: single truth behind the public status API.
//!
//! Production impl: [`crate::dbprism::DbPrismStore`]. This async trait +
//! memory impl keep the challenge testable offline.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use prism_lium_types::TelemetryPoint;
use prism_store_types::{
    ArchitectureRecord, EpochScoreRow, FinalScore, PublishArchOutcome, Stage, StageEvent,
    StatePatch, StoreError, SubmissionId, SubmissionState, TopModelPublication,
};

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

    /// Retry reset: clears review/score fields and the emission watermark,
    /// and re-queues a failed row. A completed measurement (pod id/provider,
    /// receipt, metrics, bpb, telemetry series) is RETAINED so a post-run
    /// infra retry resumes without re-measure; a measure-phase failure never
    /// persisted one, so install retries still re-provision from scratch.
    async fn reset_for_retry(&self, id: &str) -> Result<SubmissionState, StoreError>;

    /// Newsfeed listing for the API (`status` / `miner` optional filters).
    async fn list(
        &self,
        status: Option<&str>,
        miner: Option<&str>,
        limit: u32,
    ) -> Result<Vec<SubmissionState>, StoreError>;

    /// Champion corpus for copy/similarity/agentic gates: submissions with
    /// `Score(v)` where `v > 0` (current top + historical WTA ex-tops).
    /// Newest first. Does **not** include baseline (callers add that).
    async fn list_champions(&self, limit: u32) -> Result<Vec<SubmissionState>, StoreError>;

    /// Ascending journal.
    async fn events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError>;

    /// Per-step telemetry series (migration 0009 `prism_telemetry`), ordered
    /// by step. Empty when the harness reported nothing (pre-1.1.0 recipes).
    async fn telemetry(&self, id: &str) -> Result<Vec<TelemetryPoint>, StoreError>;

    /// Emission outbox step 1: atomically assign every scored row not yet
    /// emitted (`final_score` set, no `emitted_epoch`) to leaf `epoch` and
    /// return the batch — one entry per submission (NOT collapsed per miner:
    /// competition scoring aggregates credits).
    async fn assign_emit_batch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<EpochScoreRow>, StoreError>;

    /// Sticky re-read of the batch assigned to `epoch` (recovery resubmit
    /// after a crash between submit and cursor advance).
    async fn emit_batch(&self, netuid: u16, epoch: u64) -> Result<Vec<EpochScoreRow>, StoreError>;

    /// Positive lattice scores (`Score(v)` with `v > 0`) for competition
    /// carry-forward across epochs. Outbox assignment stays exactly-once;
    /// these rows are re-read every tick so a prior winner is not burned
    /// when a later epoch's fresh batch is empty or reject-only.
    async fn active_score_rows(&self, netuid: u16) -> Result<Vec<EpochScoreRow>, StoreError>;

    /// Highest leaf epoch whose set fully landed (`None` = never emitted).
    async fn emit_cursor(&self, netuid: u16) -> Result<Option<u64>, StoreError>;

    /// Advance the cursor (monotonic max) after a fully-submitted set.
    async fn set_emit_cursor(&self, netuid: u16, epoch: u64) -> Result<(), StoreError>;

    /// Assigned-but-not-completed leaf epochs (above the cursor), ascending.
    async fn pending_emit_epochs(&self, netuid: u16) -> Result<Vec<u64>, StoreError>;

    /// Publish an architecture into the registry (idempotent on digest).
    async fn publish_arch(
        &self,
        rec: &ArchitectureRecord,
    ) -> Result<PublishArchOutcome, StoreError>;

    /// Fetch one registered architecture.
    async fn get_arch(&self, arch_id: &str) -> Result<Option<ArchitectureRecord>, StoreError>;

    /// Newest-first registry listing (leaderboard source).
    async fn list_archs(&self, limit: u32) -> Result<Vec<ArchitectureRecord>, StoreError>;

    /// Lower-wins update of the arch's best bpb. Returns `true` when the
    /// record improved.
    async fn note_arch_best_bpb(&self, arch_id: &str, bpb: f64) -> Result<bool, StoreError>;

    /// `(arch_id, owner_hotkey)` for every registered architecture
    /// (competition scoring ownership map).
    async fn arch_owners(&self) -> Result<Vec<(String, String)>, StoreError>;

    /// Journal one top-model publication.
    async fn record_publication(&self, p: &TopModelPublication) -> Result<(), StoreError>;

    /// bpb of the most recent publication (idempotency guard; `None` = never
    /// published).
    async fn last_publication_bpb(&self) -> Result<Option<f64>, StoreError>;

    /// Most recent top-model publication row (`None` = never published).
    async fn last_publication(&self) -> Result<Option<TopModelPublication>, StoreError>;

    /// Best (lowest) bpb across all scored submissions ever (global top-model
    /// trigger baseline).
    async fn best_scored_bpb(&self) -> Result<Option<f64>, StoreError>;

    /// Non-terminal rows beyond grace — for the stuck sweep.
    async fn list_stuck(&self, grace_secs: u64) -> Result<Vec<SubmissionState>, StoreError>;

    /// Precheck attempts used for `(coldkey_or_hotkey, UTC day)` (0 if none).
    async fn precheck_quota_get(&self, identity: &str, day: &str) -> Result<u32, StoreError>;

    /// Consume one precheck attempt when under `limit`. `Some(used)` or `None` if full.
    async fn precheck_quota_try_consume(
        &self,
        identity: &str,
        day: &str,
        limit: u32,
    ) -> Result<Option<u32>, StoreError>;
}

/// In-memory store (CI / sim).
#[derive(Debug, Default)]
pub struct MemoryPrismStore {
    rows: Mutex<VecDeque<SubmissionState>>,
    events: Mutex<Vec<(String, StageEvent)>>,
    telemetry: Mutex<std::collections::HashMap<String, Vec<TelemetryPoint>>>,
    archs: Mutex<std::collections::HashMap<String, ArchitectureRecord>>,
    publications: Mutex<Vec<TopModelPublication>>,
    /// Emission outbox watermark: submission id → assigned leaf epoch.
    emitted: Mutex<std::collections::BTreeMap<SubmissionId, u64>>,
    /// Emit cursor per netuid (highest fully-submitted leaf epoch).
    cursors: Mutex<std::collections::BTreeMap<u16, u64>>,
    /// `(identity, UTC day)` → checks used for similarity precheck.
    precheck_quota: Mutex<std::collections::HashMap<(String, String), u32>>,
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
        if let Some(v) = &update.arch_id {
            row.arch_id = Some(v.clone());
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
        // A completed measurement (pod id/provider, receipt, metrics, bpb and
        // the derived telemetry series) is RETAINED: post-run infra retries
        // resume from it instead of re-measuring. A measure-phase failure
        // never persisted one, so install retries still re-provision.
        row.status = Stage::Queued;
        row.review = None;
        row.similarity = None;
        row.final_score = None;
        row.error_detail = None;
        row.retry_count = row.retry_count.saturating_add(1);
        row.updated_at_ms = now_ms();
        let out = row.clone();
        drop(rows);
        // A re-scored row must re-enter the emission outbox.
        self.emitted
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

    async fn list_champions(&self, limit: u32) -> Result<Vec<SubmissionState>, StoreError> {
        let mut v: Vec<_> = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|r| matches!(r.final_score, Some(FinalScore::Score(s)) if s > 0))
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

    async fn assign_emit_batch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<EpochScoreRow>, StoreError> {
        let rows = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let mut emitted = self
            .emitted
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let mut out = Vec::new();
        for r in rows.iter() {
            if r.netuid == netuid && r.final_score.is_some() && !emitted.contains_key(&r.id) {
                emitted.insert(r.id.clone(), epoch);
                out.push(EpochScoreRow {
                    miner_hotkey: r.miner_hotkey.clone(),
                    arch_id: r.arch_id.clone(),
                    final_score: r.final_score.clone().unwrap_or(FinalScore::Score(0)),
                });
            }
        }
        Ok(out)
    }

    async fn emit_batch(&self, netuid: u16, epoch: u64) -> Result<Vec<EpochScoreRow>, StoreError> {
        let rows = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let emitted = self
            .emitted
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        Ok(rows
            .iter()
            .filter(|r| {
                r.netuid == netuid && r.final_score.is_some() && emitted.get(&r.id) == Some(&epoch)
            })
            .map(|r| EpochScoreRow {
                miner_hotkey: r.miner_hotkey.clone(),
                arch_id: r.arch_id.clone(),
                final_score: r.final_score.clone().unwrap_or(FinalScore::Score(0)),
            })
            .collect())
    }

    async fn active_score_rows(&self, netuid: u16) -> Result<Vec<EpochScoreRow>, StoreError> {
        let rows = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        Ok(rows
            .iter()
            .filter(|r| r.netuid == netuid)
            .filter_map(|r| match &r.final_score {
                Some(FinalScore::Score(v)) if *v > 0 => Some(EpochScoreRow {
                    miner_hotkey: r.miner_hotkey.clone(),
                    arch_id: r.arch_id.clone(),
                    final_score: FinalScore::Score(*v),
                }),
                _ => None,
            })
            .collect())
    }

    async fn emit_cursor(&self, netuid: u16) -> Result<Option<u64>, StoreError> {
        Ok(self
            .cursors
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(&netuid)
            .copied())
    }

    async fn set_emit_cursor(&self, netuid: u16, epoch: u64) -> Result<(), StoreError> {
        let mut cursors = self
            .cursors
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let e = cursors.entry(netuid).or_insert(0);
        *e = (*e).max(epoch);
        Ok(())
    }

    async fn pending_emit_epochs(&self, netuid: u16) -> Result<Vec<u64>, StoreError> {
        // Epochs strictly above the cursor; no cursor yet → everything assigned.
        let min = match self.emit_cursor(netuid).await? {
            Some(c) => c.saturating_add(1),
            None => 0,
        };
        let rows = self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let emitted = self
            .emitted
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        Ok(rows
            .iter()
            .filter(|r| r.netuid == netuid)
            .filter_map(|r| emitted.get(&r.id).copied())
            .filter(|e| *e >= min)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect())
    }

    async fn publish_arch(
        &self,
        rec: &ArchitectureRecord,
    ) -> Result<PublishArchOutcome, StoreError> {
        let mut archs = self
            .archs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        if let Some(existing) = archs.values().find(|a| a.arch_digest == rec.arch_digest) {
            return Ok(PublishArchOutcome::Duplicate(existing.arch_id.clone()));
        }
        archs.insert(rec.arch_id.clone(), rec.clone());
        Ok(PublishArchOutcome::Published)
    }

    async fn get_arch(&self, arch_id: &str) -> Result<Option<ArchitectureRecord>, StoreError> {
        Ok(self
            .archs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .get(arch_id)
            .cloned())
    }

    async fn list_archs(&self, limit: u32) -> Result<Vec<ArchitectureRecord>, StoreError> {
        let mut v: Vec<_> = self
            .archs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .values()
            .cloned()
            .collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.created_at_ms));
        v.truncate(limit as usize);
        Ok(v)
    }

    async fn note_arch_best_bpb(&self, arch_id: &str, bpb: f64) -> Result<bool, StoreError> {
        let mut archs = self
            .archs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let Some(rec) = archs.get_mut(arch_id) else {
            return Ok(false);
        };
        if rec.best_bpb.is_some_and(|b| b <= bpb) {
            return Ok(false);
        }
        rec.best_bpb = Some(bpb);
        Ok(true)
    }

    async fn arch_owners(&self) -> Result<Vec<(String, String)>, StoreError> {
        Ok(self
            .archs
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .values()
            .map(|a| (a.arch_id.clone(), a.owner_hotkey.clone()))
            .collect())
    }

    async fn record_publication(&self, p: &TopModelPublication) -> Result<(), StoreError> {
        self.publications
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .push(p.clone());
        Ok(())
    }

    async fn last_publication_bpb(&self) -> Result<Option<f64>, StoreError> {
        Ok(self
            .publications
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .last()
            .map(|p| p.bpb))
    }

    async fn last_publication(&self) -> Result<Option<TopModelPublication>, StoreError> {
        Ok(self
            .publications
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .last()
            .cloned())
    }

    async fn best_scored_bpb(&self) -> Result<Option<f64>, StoreError> {
        Ok(self
            .rows
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?
            .iter()
            .filter(|r| matches!(r.final_score, Some(FinalScore::Score(v)) if v > 0))
            .filter_map(|r| r.bpb)
            .min_by(f64::total_cmp))
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

    async fn precheck_quota_get(&self, identity: &str, day: &str) -> Result<u32, StoreError> {
        let map = self
            .precheck_quota
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        Ok(map
            .get(&(identity.to_owned(), day.to_owned()))
            .copied()
            .unwrap_or(0))
    }

    async fn precheck_quota_try_consume(
        &self,
        identity: &str,
        day: &str,
        limit: u32,
    ) -> Result<Option<u32>, StoreError> {
        let mut map = self
            .precheck_quota
            .lock()
            .map_err(|_| StoreError::Backend("poison".into()))?;
        let key = (identity.to_owned(), day.to_owned());
        let used = map.get(&key).copied().unwrap_or(0);
        if used >= limit {
            return Ok(None);
        }
        let next = used + 1;
        map.insert(key, next);
        Ok(Some(next))
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
            miner_coldkey: None,
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            architecture_py: "a".into(),
            training_py: "t".into(),
            tree_blob: None,
            label: None,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: None,
            bpb: None,
            arch_id: None,
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
    async fn list_champions_score_positive_only() {
        let s = MemoryPrismStore::new();
        let mut winner = row("w", "11");
        winner.status = Stage::Terminated;
        winner.final_score = Some(FinalScore::Score(42));
        let mut zero = row("z", "22");
        zero.status = Stage::Terminated;
        zero.final_score = Some(FinalScore::Score(0));
        s.insert_queued(&winner).await.unwrap();
        s.insert_queued(&zero).await.unwrap();
        s.insert_queued(&row("q", "33")).await.unwrap();
        let champs = s.list_champions(10).await.unwrap();
        assert_eq!(champs.len(), 1);
        assert_eq!(champs[0].id, "w");
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
        // Post-run retry retains the completed measurement + derived series
        // (only review/score fields reset), so retries never re-measure.
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
        let retried = s.reset_for_retry("a").await.unwrap();
        assert_eq!(retried.status, Stage::Queued);
        assert_eq!(retried.retry_count, 1);
        assert_eq!(s.telemetry("a").await.unwrap().len(), 2);
        assert!(s.get("a").await.unwrap().unwrap().metrics_json.is_some());
    }

    #[tokio::test]
    async fn emit_outbox_assign_once_cursor_monotonic() {
        let s = MemoryPrismStore::new();
        let mut r = row("a", "11");
        r.final_score = Some(FinalScore::Score(42));
        s.insert_queued(&r).await.unwrap();

        let batch = s.assign_emit_batch(541, 9).await.unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].final_score, FinalScore::Score(42));
        // Sticky: re-assign is empty; the assigned epoch re-reads.
        assert!(s.assign_emit_batch(541, 10).await.unwrap().is_empty());
        assert_eq!(s.emit_batch(541, 9).await.unwrap().len(), 1);
        assert_eq!(s.pending_emit_epochs(541).await.unwrap(), vec![9]);
        s.set_emit_cursor(541, 9).await.unwrap();
        assert_eq!(s.emit_cursor(541).await.unwrap(), Some(9));
        // Positive scores remain active for carry after outbox assignment.
        assert_eq!(s.active_score_rows(541).await.unwrap().len(), 1);
        // Monotonic: a stale cursor write never regresses.
        s.set_emit_cursor(541, 4).await.unwrap();
        assert_eq!(s.emit_cursor(541).await.unwrap(), Some(9));
        assert!(s.pending_emit_epochs(541).await.unwrap().is_empty());
        // Unscored rows never enter a batch.
        s.insert_queued(&row("b", "22")).await.unwrap();
        assert!(s.assign_emit_batch(541, 10).await.unwrap().is_empty());
        // Score(0) rejects are not active carry rows.
        let mut zero = row("c", "33");
        zero.final_score = Some(FinalScore::Score(0));
        s.insert_queued(&zero).await.unwrap();
        assert_eq!(s.active_score_rows(541).await.unwrap().len(), 1);
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
