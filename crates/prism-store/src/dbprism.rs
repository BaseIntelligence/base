//! `db::prism_store` (SQL) onto the challenge-side [`PrismStore`] trait.

use async_trait::async_trait;
use db::prism_store as dbs;
use db::PgPool;
use prism_lium::EvalReceipt;
use prism_review::{ReviewVerdict, SimilarityKind, SimilarityVerdict};

use crate::store::{
    FinalScore, PrismStore, Stage, StageEvent, StatePatch, StoreError, SubmissionState,
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
        epoch: r.epoch.cast_unsigned(),
        netuid: u16::try_from(r.netuid).unwrap_or(0),
        status,
        architecture_py: r.architecture_py,
        training_py: r.training_py,
        label: r.label,
        pod_id: r.pod_id,
        pod_provider: r.pod_provider,
        receipt,
        bpb: r.bpb,
        review,
        similarity,
        final_score,
        retry_count: u32::try_from(r.retry_count.max(0)).unwrap_or(u32::MAX),
        error_detail: r.error_detail,
        created_at_ms: 0,
        updated_at_ms: 0,
    }
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
                epoch: i64::try_from(row.epoch).unwrap_or(i64::MAX),
                netuid: i32::from(row.netuid),
                label: row.label.as_deref(),
                architecture_py: &row.architecture_py,
                training_py: &row.training_py,
            },
        )
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))
    }

    async fn get(&self, id: &str) -> Result<Option<SubmissionState>, StoreError> {
        dbs::prism_submission(&self.pool, id)
            .await
            .map(|o| o.map(row_to_state))
            .map_err(|e| StoreError::Backend(e.to_string()))
    }

    async fn claim_next(&self) -> Result<Option<SubmissionState>, StoreError> {
        dbs::claim_prism_submission(&self.pool)
            .await
            .map(|o| o.map(row_to_state))
            .map_err(|e| StoreError::Backend(e.to_string()))
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
            None,
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
        Ok(row_to_state(row))
    }

    async fn list(
        &self,
        status: Option<&str>,
        limit: u32,
    ) -> Result<Vec<SubmissionState>, StoreError> {
        dbs::list_prism_submissions(&self.pool, status, i64::from(limit))
            .await
            .map(|v| v.into_iter().map(row_to_state).collect())
            .map_err(|e| StoreError::Backend(e.to_string()))
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

    async fn scores_for_epoch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<(String, FinalScore)>, StoreError> {
        dbs::prism_scores_for_epoch(
            &self.pool,
            i32::from(netuid),
            i64::try_from(epoch).unwrap_or(i64::MAX),
        )
        .await
        .map(|v| {
            v.into_iter()
                .filter_map(|(hk, kind, score, absence)| match kind.as_str() {
                    "score" => score.map(|s| (hk, FinalScore::Score(s.cast_unsigned()))),
                    "no_score" => {
                        absence.map(|a| (hk, FinalScore::NoScore(u8::try_from(a).unwrap_or(0))))
                    }
                    _ => None,
                })
                .collect()
        })
        .map_err(|e| StoreError::Backend(e.to_string()))
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
        Ok(out)
    }
}
