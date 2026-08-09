//! `db::prism_store` (SQL) onto the challenge-side [`PrismStore`] trait.

use async_trait::async_trait;
use db::prism_store as dbs;
use db::PgPool;
use prism_lium_types::EvalReceipt;
use prism_review::{ReviewVerdict, SimilarityKind, SimilarityVerdict};

use crate::arch;
use crate::store::PrismStore;
use prism_store_types::{
    ArchitectureRecord, EpochScoreRow, FinalScore, PublishArchOutcome, Stage, StageEvent,
    StatePatch, StoreError, SubmissionState, TopModelPublication,
};

/// SQL-backed production store.
#[derive(Debug, Clone)]
pub struct DbPrismStore {
    pool: PgPool,
}

impl DbPrismStore {
    /// New from an existing pool.
    #[must_use]
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Pool getter (diagnostics).
    #[must_use]
    pub const fn pool(&self) -> &PgPool {
        &self.pool
    }
}

fn row_to_state(r: dbs::PrismSubmissionRow) -> SubmissionState {
    let status = Stage::parse(&r.status).unwrap_or(Stage::Failed);
    let receipt = r
        .receipt_json
        .as_ref()
        .and_then(|v| serde_json::from_value::<EvalReceipt>(v.clone()).ok());
    let review = r
        .review_json
        .as_ref()
        .and_then(|v| serde_json::from_value::<DbReviewVerdict>(v.clone()).ok())
        .map(ReviewVerdict::from);
    let similarity = r
        .similarity_json
        .as_ref()
        .and_then(|v| serde_json::from_value::<DbSimilarityVerdict>(v.clone()).ok())
        .map(SimilarityVerdict::from);
    let final_score = r.kind.clone().and_then(|k| match k.as_str() {
        "score" => r.score.map(|s| FinalScore::Score(s.cast_unsigned())),
        "no_score" => r
            .absence_reason
            .map(|a| FinalScore::NoScore(u8::try_from(a).unwrap_or(0))),
        _ => None,
    });
    SubmissionState {
        id: r.id,
        miner_hotkey: r.miner_hotkey,
        miner_coldkey: r.miner_coldkey,
        epoch: r.epoch.cast_unsigned(),
        netuid: u16::try_from(r.netuid).unwrap_or(0),
        status,
        architecture_py: r.architecture_py,
        training_py: r.training_py,
        tree_blob: r.tree_blob,
        label: r.label,
        pod_id: r.pod_id,
        pod_provider: r.pod_provider,
        receipt,
        metrics_json: r.metrics_json,
        bpb: r.bpb,
        arch_id: None, // filled by arch::fill_arch_meta at the read sites
        review,
        similarity,
        final_score,
        retry_count: u32::try_from(r.retry_count.max(0)).unwrap_or(u32::MAX),
        error_detail: r.error_detail,
        created_at_ms: 0, // filled by arch::fill_arch_meta
        updated_at_ms: 0,
    }
}

/// Map + fill migration-0010 columns (`arch_id`, `created_at`) on reads.
async fn states_filled(
    pool: &PgPool,
    rows: Vec<dbs::PrismSubmissionRow>,
) -> Result<Vec<SubmissionState>, StoreError> {
    let mut out: Vec<SubmissionState> = rows.into_iter().map(row_to_state).collect();
    arch::fill_arch_meta(pool, &mut out).await?;
    Ok(out)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DbReviewVerdict {
    quality_score: u16,
    issues: Vec<String>,
    prompt_version: String,
}
impl From<&ReviewVerdict> for DbReviewVerdict {
    fn from(v: &ReviewVerdict) -> Self {
        Self {
            quality_score: v.quality_score,
            issues: v.issues.clone(),
            prompt_version: v.prompt_version.to_owned(),
        }
    }
}
impl From<DbReviewVerdict> for ReviewVerdict {
    fn from(v: DbReviewVerdict) -> Self {
        Self {
            quality_score: v.quality_score,
            issues: v.issues,
            prompt_version: prism_review::REVIEW_PROMPT_VERSION,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DbSimilarityVerdict {
    kind: String,
    score: f64,
    closest: Option<String>,
    evidence: Vec<String>,
    prompt_version: String,
}
impl From<&SimilarityVerdict> for DbSimilarityVerdict {
    fn from(v: &SimilarityVerdict) -> Self {
        Self {
            kind: match v.kind {
                SimilarityKind::Original => "original".into(),
                SimilarityKind::Suspicious => "suspicious".into(),
                SimilarityKind::Copied => "copied".into(),
            },
            score: v.score,
            closest: v.closest.clone(),
            evidence: v.evidence.clone(),
            prompt_version: v.prompt_version.to_owned(),
        }
    }
}
impl From<DbSimilarityVerdict> for SimilarityVerdict {
    fn from(v: DbSimilarityVerdict) -> Self {
        Self {
            kind: match v.kind.as_str() {
                "copied" => SimilarityKind::Copied,
                "suspicious" => SimilarityKind::Suspicious,
                _ => SimilarityKind::Original,
            },
            score: v.score,
            closest: v.closest,
            evidence: v.evidence,
            prompt_version: prism_review::SIMILARITY_PROMPT_VERSION,
        }
    }
}

fn kind_score_absence(f: Option<&FinalScore>) -> (Option<&'static str>, Option<i64>, Option<i16>) {
    match f {
        Some(FinalScore::Score(v)) => (
            Some("score"),
            Some(i64::try_from(*v).unwrap_or(i64::MAX)),
            None,
        ),
        Some(FinalScore::NoScore(r)) => (Some("no_score"), None, Some(i16::from(*r))),
        None => (None, None, None),
    }
}

#[async_trait]
impl PrismStore for DbPrismStore {
    async fn insert_queued(&self, row: &SubmissionState) -> Result<(), StoreError> {
        dbs::insert_prism_submission(
            &self.pool,
            &dbs::NewPrismSubmission {
                id: &row.id,
                miner_hotkey: &row.miner_hotkey,
                miner_coldkey: row.miner_coldkey.as_deref(),
                epoch: i64::try_from(row.epoch).unwrap_or(i64::MAX),
                netuid: i32::from(row.netuid),
                label: row.label.as_deref(),
                architecture_py: &row.architecture_py,
                training_py: &row.training_py,
                tree_blob: row.tree_blob.as_deref(),
            },
        )
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        if let Some(a) = &row.arch_id {
            arch::set_row_arch_id(&self.pool, &row.id, a).await?;
        }
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<SubmissionState>, StoreError> {
        let row = dbs::prism_submission(&self.pool, id)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        match row {
            Some(r) => Ok(states_filled(&self.pool, vec![r]).await?.into_iter().next()),
            None => Ok(None),
        }
    }

    async fn claim_next(&self) -> Result<Option<SubmissionState>, StoreError> {
        let row = dbs::claim_prism_submission(&self.pool)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        match row {
            Some(r) => Ok(states_filled(&self.pool, vec![r]).await?.into_iter().next()),
            None => Ok(None),
        }
    }

    async fn apply(
        &self,
        id: &str,
        update: &StatePatch,
        event: Option<&StageEvent>,
    ) -> Result<SubmissionState, StoreError> {
        let (kind, score, absence) = kind_score_absence(update.final_score.as_ref());
        let row = dbs::update_prism_submission(
            &self.pool,
            id,
            update.status.map(Stage::as_str),
            update.pod_id.as_deref(),
            update.pod_provider.as_deref(),
            update
                .receipt
                .as_ref()
                .and_then(|r| serde_json::to_value(r).ok()),
            update.metrics_json.clone(),
            update.bpb,
            update
                .review
                .as_ref()
                .map(DbReviewVerdict::from)
                .and_then(|v| serde_json::to_value(v).ok()),
            update
                .similarity
                .as_ref()
                .map(DbSimilarityVerdict::from)
                .and_then(|v| serde_json::to_value(v).ok()),
            kind,
            score,
            absence,
            update.retry_bump.cast_signed(),
            update.error_detail.as_deref(),
        )
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        if let Some(a) = &update.arch_id {
            arch::set_row_arch_id(&self.pool, id, a).await?;
        }
        if let Some(e) = event {
            dbs::insert_prism_stage_event(
                &self.pool,
                &dbs::NewPrismStageEvent {
                    submission_id: id,
                    stage: e.stage.as_str(),
                    detail: e.detail.clone(),
                },
            )
            .await
            .map_err(|e2| StoreError::Backend(e2.to_string()))?;
        }
        // Fan the miner-reported series out to `prism_telemetry`. Best-effort:
        // the row's `metrics_json` (written above) is the authoritative copy,
        // so a telemetry-table hiccup must not fail the scoring apply.
        if let Some(m) = &update.metrics_json {
            let series = crate::store::telemetry_from_metrics(m);
            if !series.is_empty() {
                if let Err(e) = crate::telemetry::replace_telemetry(&self.pool, id, &series).await {
                    tracing::warn!(submission_id = %id, error = %e, "prism telemetry write failed");
                }
            }
        }
        Ok(states_filled(&self.pool, vec![row])
            .await?
            .into_iter()
            .next()
            .ok_or(StoreError::NotFound)?)
    }

    async fn reset_for_retry(&self, id: &str) -> Result<SubmissionState, StoreError> {
        let row = dbs::reset_prism_submission_for_retry(&self.pool, id)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        dbs::insert_prism_stage_event(
            &self.pool,
            &dbs::NewPrismStageEvent {
                submission_id: id,
                stage: "queued",
                detail: Some(serde_json::json!({"op": "retry"})),
            },
        )
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(states_filled(&self.pool, vec![row])
            .await?
            .into_iter()
            .next()
            .ok_or(StoreError::NotFound)?)
    }

    async fn list(
        &self,
        status: Option<&str>,
        miner: Option<&str>,
        limit: u32,
    ) -> Result<Vec<SubmissionState>, StoreError> {
        let rows = dbs::list_prism_submissions(&self.pool, status, miner, i64::from(limit))
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        states_filled(&self.pool, rows).await
    }

    async fn list_champions(&self, limit: u32) -> Result<Vec<SubmissionState>, StoreError> {
        let rows = dbs::list_prism_champions(&self.pool, i64::from(limit))
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        states_filled(&self.pool, rows).await
    }

    async fn events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError> {
        dbs::prism_stage_events(&self.pool, id)
            .await
            .map(|v| {
                v.into_iter()
                    .map(|(stage, detail, at)| StageEvent {
                        stage: Stage::parse(&stage).unwrap_or(Stage::Failed),
                        detail,
                        at_ms: at.max(0).cast_unsigned() * 1000,
                    })
                    .collect()
            })
            .map_err(|e| StoreError::Backend(e.to_string()))
    }

    async fn telemetry(
        &self,
        id: &str,
    ) -> Result<Vec<prism_lium_types::TelemetryPoint>, StoreError> {
        crate::telemetry::telemetry_for(&self.pool, id).await
    }

    async fn assign_emit_batch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<EpochScoreRow>, StoreError> {
        crate::emit::assign_emit_batch(
            &self.pool,
            i32::from(netuid),
            i64::try_from(epoch).unwrap_or(i64::MAX),
        )
        .await
    }

    async fn emit_batch(&self, netuid: u16, epoch: u64) -> Result<Vec<EpochScoreRow>, StoreError> {
        crate::emit::emit_batch(
            &self.pool,
            i32::from(netuid),
            i64::try_from(epoch).unwrap_or(i64::MAX),
        )
        .await
    }

    async fn active_score_rows(&self, netuid: u16) -> Result<Vec<EpochScoreRow>, StoreError> {
        crate::emit::active_score_rows(&self.pool, i32::from(netuid)).await
    }

    async fn emit_cursor(&self, netuid: u16) -> Result<Option<u64>, StoreError> {
        crate::emit::emit_cursor(&self.pool, i32::from(netuid)).await
    }

    async fn set_emit_cursor(&self, netuid: u16, epoch: u64) -> Result<(), StoreError> {
        crate::emit::set_emit_cursor(
            &self.pool,
            i32::from(netuid),
            i64::try_from(epoch).unwrap_or(i64::MAX),
        )
        .await
    }

    async fn pending_emit_epochs(&self, netuid: u16) -> Result<Vec<u64>, StoreError> {
        crate::emit::pending_emit_epochs(&self.pool, i32::from(netuid)).await
    }

    async fn publish_arch(
        &self,
        rec: &ArchitectureRecord,
    ) -> Result<PublishArchOutcome, StoreError> {
        arch::insert_arch(&self.pool, rec).await
    }

    async fn get_arch(&self, arch_id: &str) -> Result<Option<ArchitectureRecord>, StoreError> {
        arch::get_arch(&self.pool, arch_id).await
    }

    async fn list_archs(&self, limit: u32) -> Result<Vec<ArchitectureRecord>, StoreError> {
        arch::list_archs(&self.pool, i64::from(limit)).await
    }

    async fn note_arch_best_bpb(&self, arch_id: &str, bpb: f64) -> Result<bool, StoreError> {
        arch::note_arch_best_bpb(&self.pool, arch_id, bpb).await
    }

    async fn arch_owners(&self) -> Result<Vec<(String, String)>, StoreError> {
        arch::arch_owners(&self.pool).await
    }

    async fn record_publication(&self, p: &TopModelPublication) -> Result<(), StoreError> {
        arch::record_publication(&self.pool, p).await
    }

    async fn last_publication_bpb(&self) -> Result<Option<f64>, StoreError> {
        arch::last_publication_bpb(&self.pool).await
    }

    async fn last_publication(&self) -> Result<Option<TopModelPublication>, StoreError> {
        arch::last_publication(&self.pool).await
    }

    async fn best_scored_bpb(&self) -> Result<Option<f64>, StoreError> {
        arch::best_scored_bpb(&self.pool).await
    }

    async fn list_stuck(&self, grace_secs: u64) -> Result<Vec<SubmissionState>, StoreError> {
        let rows = dbs::stuck_prism_before_grace(
            &self.pool,
            i64::try_from(grace_secs).unwrap_or(i64::MAX),
        )
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        let mut out = Vec::new();
        for (id, _st, _pod) in rows {
            if let Some(r) = dbs::prism_submission(&self.pool, &id)
                .await
                .map_err(|e| StoreError::Backend(e.to_string()))?
            {
                out.push(row_to_state(r));
            }
        }
        arch::fill_arch_meta(&self.pool, &mut out).await?;
        Ok(out)
    }

    async fn precheck_quota_get(&self, identity: &str, day: &str) -> Result<u32, StoreError> {
        let n = dbs::prism_precheck_quota_get(&self.pool, identity, day)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(u32::try_from(n).unwrap_or(u32::MAX))
    }

    async fn precheck_quota_try_consume(
        &self,
        identity: &str,
        day: &str,
        limit: u32,
    ) -> Result<Option<u32>, StoreError> {
        let limit_i = i32::try_from(limit).unwrap_or(i32::MAX);
        let n = dbs::prism_precheck_quota_try_consume(&self.pool, identity, day, limit_i)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(n.map(|v| u32::try_from(v).unwrap_or(u32::MAX)))
    }
}
