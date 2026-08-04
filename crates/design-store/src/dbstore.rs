//! `db::design_store` → [`DesignStore`] adapter.

use std::collections::BTreeMap;

use async_trait::async_trait;
use db::design_store as dbs;
use db::PgPool;

use crate::store::{
    ArtifactPage, DesignStore, FinalScore, HarnessRow, PairRow, RatingRow, RoundAward, RoundRow,
    RunStage, RunState, StageEvent, StoreError, StorePatch,
};

/// SQL-backed production store.
#[derive(Debug, Clone)]
pub struct DbDesignStore {
    pool: PgPool,
}

impl DbDesignStore {
    /// From pool.
    #[must_use]
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

fn map_db(e: db::DbError) -> StoreError {
    let s = e.to_string();
    if s.contains("duplicate") || s.contains("unique") {
        StoreError::Duplicate
    } else {
        StoreError::Backend(s)
    }
}

fn extras_from_value(v: &serde_json::Value) -> BTreeMap<String, String> {
    v.as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_owned())))
                .collect()
        })
        .unwrap_or_default()
}

fn harness_from(r: dbs::DesignHarnessRow) -> HarnessRow {
    HarnessRow {
        id: r.id,
        miner_hotkey: r.miner_hotkey,
        agent_py: r.agent_py,
        pyproject_toml: r.pyproject_toml,
        extra_files: extras_from_value(&r.extra_files),
        active: r.active,
        eliminated_until_round: r.eliminated_until_round.cast_unsigned(),
    }
}

fn final_from(kind: Option<&str>, score: Option<i64>, absence: Option<i16>) -> Option<FinalScore> {
    match kind {
        Some("score") => score.map(|s| FinalScore::Score(s.cast_unsigned())),
        Some("no_score") => absence.map(|a| FinalScore::NoScore(u8::try_from(a).unwrap_or(0))),
        _ => None,
    }
}

fn kind_score(f: Option<&FinalScore>) -> (Option<&'static str>, Option<i64>, Option<i16>) {
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

fn run_from(r: dbs::DesignRunRow) -> RunState {
    RunState {
        id: r.id,
        round_id: r.round_id.cast_unsigned(),
        harness_id: r.harness_id,
        prompt_id: r.prompt_id,
        status: RunStage::parse(&r.status).unwrap_or(RunStage::Failed),
        artifact_digest: r.artifact_digest,
        sanitize_report: r.sanitize_report,
        agentic_verdict: r.agentic_verdict,
        error_detail: r.error_detail,
        final_score: final_from(r.kind.as_deref(), r.score, r.absence_reason),
        retry_count: u32::try_from(r.retry_count.max(0)).unwrap_or(u32::MAX),
        created_at_ms: 0,
        updated_at_ms: 0,
    }
}

fn award_from(r: dbs::DesignRoundAwardRow) -> RoundAward {
    let winners = r
        .winner_harness_ids
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    RoundAward {
        round_id: r.round_id.cast_unsigned(),
        winner_harness_ids: winners,
        awarded_at_ms: r.awarded_at_secs.cast_unsigned().saturating_mul(1000),
        admin_token_hash: r.admin_token_hash,
    }
}

fn round_from(r: dbs::DesignRoundRow) -> RoundRow {
    RoundRow {
        round_id: r.round_id.cast_unsigned(),
        epoch: r.epoch.cast_unsigned(),
        netuid: u16::try_from(r.netuid).unwrap_or(0),
        prompt_set_digest: r.prompt_set_digest,
        status: r.status,
        opens_at_secs: 0,
        closes_at_secs: 0,
    }
}

#[async_trait]
impl DesignStore for DbDesignStore {
    async fn insert_harness(&self, row: &HarnessRow) -> Result<(), StoreError> {
        let extras = serde_json::to_value(&row.extra_files).unwrap_or(serde_json::json!({}));
        dbs::insert_design_harness(
            &self.pool,
            &dbs::NewDesignHarness {
                id: &row.id,
                miner_hotkey: &row.miner_hotkey,
                agent_py: &row.agent_py,
                pyproject_toml: &row.pyproject_toml,
                extra_files: extras,
            },
        )
        .await
        .map_err(map_db)
    }

    async fn get_harness(&self, id: &str) -> Result<Option<HarnessRow>, StoreError> {
        Ok(dbs::design_harness(&self.pool, id)
            .await
            .map_err(map_db)?
            .map(harness_from))
    }

    async fn list_harnesses(&self, miner: &str) -> Result<Vec<HarnessRow>, StoreError> {
        Ok(dbs::list_design_harnesses(&self.pool, miner)
            .await
            .map_err(map_db)?
            .into_iter()
            .map(harness_from)
            .collect())
    }

    async fn set_eliminated(&self, id: &str, until_round: u64) -> Result<(), StoreError> {
        dbs::set_design_harness_eliminated(
            &self.pool,
            id,
            i64::try_from(until_round).unwrap_or(i64::MAX),
        )
        .await
        .map_err(map_db)
    }

    async fn list_recent_harnesses(&self, limit: u32) -> Result<Vec<HarnessRow>, StoreError> {
        Ok(
            dbs::list_recent_design_harnesses(&self.pool, i64::from(limit))
                .await
                .map_err(map_db)?
                .into_iter()
                .map(harness_from)
                .collect(),
        )
    }

    async fn insert_round(&self, row: &RoundRow) -> Result<(), StoreError> {
        dbs::insert_design_round(
            &self.pool,
            &dbs::NewDesignRound {
                round_id: i64::try_from(row.round_id).unwrap_or(i64::MAX),
                epoch: i64::try_from(row.epoch).unwrap_or(i64::MAX),
                netuid: i32::from(row.netuid),
                opens_at_secs: i64::try_from(row.opens_at_secs).unwrap_or(0),
                closes_at_secs: i64::try_from(row.closes_at_secs).unwrap_or(0),
                prompt_set_digest: &row.prompt_set_digest,
                status: &row.status,
            },
        )
        .await
        .map_err(map_db)
    }

    async fn get_round(&self, round_id: u64) -> Result<Option<RoundRow>, StoreError> {
        Ok(
            dbs::design_round(&self.pool, i64::try_from(round_id).unwrap_or(i64::MAX))
                .await
                .map_err(map_db)?
                .map(round_from),
        )
    }

    async fn set_round_status(&self, round_id: u64, status: &str) -> Result<(), StoreError> {
        dbs::update_design_round_status(
            &self.pool,
            i64::try_from(round_id).unwrap_or(i64::MAX),
            status,
        )
        .await
        .map_err(map_db)
    }

    async fn list_rounds(&self, limit: u32) -> Result<Vec<RoundRow>, StoreError> {
        Ok(dbs::list_design_rounds(&self.pool, i64::from(limit))
            .await
            .map_err(map_db)?
            .into_iter()
            .map(round_from)
            .collect())
    }

    async fn insert_run(&self, row: &RunState) -> Result<(), StoreError> {
        dbs::insert_design_run(
            &self.pool,
            &dbs::NewDesignRun {
                id: &row.id,
                round_id: i64::try_from(row.round_id).unwrap_or(i64::MAX),
                harness_id: &row.harness_id,
                prompt_id: &row.prompt_id,
            },
        )
        .await
        .map_err(map_db)
    }

    async fn get_run(&self, id: &str) -> Result<Option<RunState>, StoreError> {
        Ok(dbs::design_run(&self.pool, id)
            .await
            .map_err(map_db)?
            .map(run_from))
    }

    async fn claim_next_run(&self) -> Result<Option<RunState>, StoreError> {
        Ok(dbs::claim_design_run(&self.pool)
            .await
            .map_err(map_db)?
            .map(run_from))
    }

    async fn apply_run(
        &self,
        id: &str,
        patch: &StorePatch,
        event: Option<&StageEvent>,
    ) -> Result<RunState, StoreError> {
        let (kind, score, absence) = kind_score(patch.final_score.as_ref());
        let row = dbs::update_design_run(
            &self.pool,
            id,
            patch.status.map(RunStage::as_str),
            patch.artifact_digest.as_deref(),
            patch.sanitize_report.clone(),
            patch.agentic_verdict.clone(),
            patch.error_detail.as_deref(),
            kind,
            score,
            absence,
            i32::try_from(patch.retry_bump).unwrap_or(0),
        )
        .await
        .map_err(map_db)?;
        if let Some(e) = event {
            dbs::insert_design_stage_event(
                &self.pool,
                &dbs::NewDesignStageEvent {
                    run_id: id,
                    stage: &e.stage,
                    detail: e.detail.clone(),
                },
            )
            .await
            .map_err(map_db)?;
        }
        Ok(run_from(row))
    }

    async fn reset_run(&self, id: &str) -> Result<RunState, StoreError> {
        Ok(run_from(
            dbs::reset_design_run_for_retry(&self.pool, id)
                .await
                .map_err(map_db)?,
        ))
    }

    async fn list_runs(
        &self,
        status: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RunState>, StoreError> {
        Ok(dbs::list_design_runs(&self.pool, status, i64::from(limit))
            .await
            .map_err(map_db)?
            .into_iter()
            .map(run_from)
            .collect())
    }

    async fn runs_for_round(&self, round_id: u64) -> Result<Vec<RunState>, StoreError> {
        Ok(
            dbs::design_runs_for_round(&self.pool, i64::try_from(round_id).unwrap_or(i64::MAX))
                .await
                .map_err(map_db)?
                .into_iter()
                .map(run_from)
                .collect(),
        )
    }

    async fn list_stuck_runs(&self, grace_secs: u64) -> Result<Vec<RunState>, StoreError> {
        Ok(
            dbs::stuck_design_runs(&self.pool, i64::try_from(grace_secs).unwrap_or(i64::MAX))
                .await
                .map_err(map_db)?
                .into_iter()
                .map(run_from)
                .collect(),
        )
    }

    async fn run_events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError> {
        Ok(dbs::design_stage_events(&self.pool, id)
            .await
            .map_err(map_db)?
            .into_iter()
            .map(|(stage, detail, secs)| StageEvent {
                stage,
                detail,
                at_ms: (secs.cast_unsigned()).saturating_mul(1000),
            })
            .collect())
    }

    async fn put_artifacts(
        &self,
        run_id: &str,
        pages: &[(String, String, String, String, u32)],
    ) -> Result<(), StoreError> {
        for (path, sanitized, raw, sha, bytes) in pages {
            dbs::insert_design_artifact(
                &self.pool,
                &dbs::NewDesignArtifact {
                    run_id,
                    path,
                    sanitized_html: sanitized,
                    raw_html: raw,
                    raw_sha256: sha,
                    bytes: i32::try_from(*bytes).unwrap_or(i32::MAX),
                },
            )
            .await
            .map_err(map_db)?;
        }
        Ok(())
    }

    async fn list_pages(&self, run_id: &str) -> Result<Vec<ArtifactPage>, StoreError> {
        Ok(dbs::design_artifacts(&self.pool, run_id)
            .await
            .map_err(map_db)?
            .into_iter()
            .map(|a| ArtifactPage {
                path: a.path,
                sanitized_html: a.sanitized_html,
                raw_sha256: a.raw_sha256,
                bytes: u32::try_from(a.bytes.max(0)).unwrap_or(u32::MAX),
            })
            .collect())
    }

    async fn get_page(&self, run_id: &str, path: &str) -> Result<Option<String>, StoreError> {
        Ok(dbs::design_artifact_page(&self.pool, run_id, path)
            .await
            .map_err(map_db)?
            .map(|(_, html)| html))
    }

    async fn insert_pair(&self, pair: &PairRow) -> Result<(), StoreError> {
        dbs::insert_design_pair(
            &self.pool,
            &dbs::NewDesignPair {
                id: &pair.id,
                round_id: i64::try_from(pair.round_id).unwrap_or(i64::MAX),
                prompt_id: &pair.prompt_id,
                run_a_id: &pair.run_a_id,
                run_b_id: &pair.run_b_id,
            },
        )
        .await
        .map_err(map_db)
    }

    async fn next_pair(
        &self,
        round_id: u64,
        annotator_id: &str,
        min_annotations: u32,
    ) -> Result<Option<PairRow>, StoreError> {
        Ok(dbs::next_design_pair_for_annotator(
            &self.pool,
            i64::try_from(round_id).unwrap_or(i64::MAX),
            annotator_id,
            i64::from(min_annotations),
        )
        .await
        .map_err(map_db)?
        .map(|p| PairRow {
            id: p.id,
            round_id: p.round_id.cast_unsigned(),
            prompt_id: p.prompt_id,
            run_a_id: p.run_a_id,
            run_b_id: p.run_b_id,
        }))
    }

    async fn insert_annotation(
        &self,
        pair_id: &str,
        annotator_id: &str,
        winner_run_id: &str,
    ) -> Result<(), StoreError> {
        dbs::insert_design_annotation(
            &self.pool,
            &dbs::NewDesignAnnotation {
                pair_id,
                annotator_id,
                winner_run_id,
            },
        )
        .await
        .map_err(map_db)
    }

    async fn annotations_for_round(
        &self,
        round_id: u64,
    ) -> Result<Vec<(String, String, String, u64)>, StoreError> {
        Ok(dbs::design_annotations_for_round(
            &self.pool,
            i64::try_from(round_id).unwrap_or(i64::MAX),
        )
        .await
        .map_err(map_db)?
        .into_iter()
        .map(|a| {
            (
                a.pair_id,
                a.annotator_id,
                a.winner_run_id,
                a.created_at_secs.cast_unsigned().saturating_mul(1000),
            )
        })
        .collect())
    }

    async fn pairs_for_round(&self, round_id: u64) -> Result<Vec<PairRow>, StoreError> {
        Ok(
            dbs::design_pairs_for_round(&self.pool, i64::try_from(round_id).unwrap_or(i64::MAX))
                .await
                .map_err(map_db)?
                .into_iter()
                .map(|p| PairRow {
                    id: p.id,
                    round_id: p.round_id.cast_unsigned(),
                    prompt_id: p.prompt_id,
                    run_a_id: p.run_a_id,
                    run_b_id: p.run_b_id,
                })
                .collect(),
        )
    }

    async fn upsert_rating(&self, row: &RatingRow) -> Result<(), StoreError> {
        let (kind, score, absence) = kind_score(row.final_score.as_ref());
        dbs::upsert_design_rating(
            &self.pool,
            i64::try_from(row.round_id).unwrap_or(i64::MAX),
            &row.miner_hotkey,
            row.rating,
            i32::try_from(row.wins).unwrap_or(0),
            i32::try_from(row.losses).unwrap_or(0),
            kind,
            score,
            absence,
        )
        .await
        .map_err(map_db)
    }

    async fn ratings_for_round(&self, round_id: u64) -> Result<Vec<RatingRow>, StoreError> {
        Ok(
            dbs::design_ratings_for_round(&self.pool, i64::try_from(round_id).unwrap_or(i64::MAX))
                .await
                .map_err(map_db)?
                .into_iter()
                .map(|r| RatingRow {
                    round_id: r.round_id.cast_unsigned(),
                    miner_hotkey: r.miner_hotkey,
                    rating: r.rating,
                    wins: u32::try_from(r.wins.max(0)).unwrap_or(0),
                    losses: u32::try_from(r.losses.max(0)).unwrap_or(0),
                    final_score: final_from(r.kind.as_deref(), r.score, r.absence_reason),
                })
                .collect(),
        )
    }

    async fn scores_for_epoch(
        &self,
        netuid: u16,
        epoch: u64,
    ) -> Result<Vec<(String, FinalScore)>, StoreError> {
        let rows = dbs::design_scores_for_epoch(
            &self.pool,
            i32::from(netuid),
            i64::try_from(epoch).unwrap_or(i64::MAX),
        )
        .await
        .map_err(map_db)?;
        Ok(rows
            .into_iter()
            .filter_map(|(hk, kind, score, absence)| {
                final_from(Some(&kind), score, absence).map(|fs| (hk, fs))
            })
            .collect())
    }

    async fn set_round_award(&self, award: &RoundAward) -> Result<(), StoreError> {
        let winners = serde_json::to_value(&award.winner_harness_ids)
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        dbs::insert_design_round_award(
            &self.pool,
            i64::try_from(award.round_id).unwrap_or(i64::MAX),
            &winners,
            award.admin_token_hash.as_deref(),
        )
        .await
        .map_err(map_db)
    }

    async fn list_round_awards(
        &self,
        from_round: u64,
        to_round: u64,
    ) -> Result<Vec<RoundAward>, StoreError> {
        Ok(dbs::list_design_round_awards(
            &self.pool,
            i64::try_from(from_round).unwrap_or(0),
            i64::try_from(to_round).unwrap_or(i64::MAX),
        )
        .await
        .map_err(map_db)?
        .into_iter()
        .map(award_from)
        .collect())
    }

    async fn get_round_award(&self, round_id: u64) -> Result<Option<RoundAward>, StoreError> {
        Ok(
            dbs::design_round_award(&self.pool, i64::try_from(round_id).unwrap_or(i64::MAX))
                .await
                .map_err(map_db)?
                .map(award_from),
        )
    }

    async fn quota_get(&self, miner: &str, day: &str) -> Result<u32, StoreError> {
        Ok(u32::try_from(
            dbs::design_quota_get(&self.pool, miner, day)
                .await
                .map_err(map_db)?,
        )
        .unwrap_or(0))
    }

    async fn quota_bump(&self, miner: &str, day: &str, bump: u32) -> Result<u32, StoreError> {
        Ok(u32::try_from(
            dbs::design_quota_bump(&self.pool, miner, day, i32::try_from(bump).unwrap_or(0))
                .await
                .map_err(map_db)?,
        )
        .unwrap_or(0))
    }
}
