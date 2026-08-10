//! Postgres [`BountyStore`] adapter.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_raw_string_hashes)]

use std::collections::BTreeMap;

use async_trait::async_trait;
use bounty_store::{
    BountyStore, BugPatch, BugRow, BugStatus, EpochScoreRow, FinalScore, StageEvent, StoreError,
};
use db::PgPool;
use serde_json::Value;
use sqlx::Row;

/// SQL-backed production store.
#[derive(Debug, Clone)]
pub struct DbBountyStore {
    pool: PgPool,
}

impl DbBountyStore {
    /// From pool.
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

fn map_sqlx(e: sqlx::Error) -> StoreError {
    let s = e.to_string();
    if s.contains("duplicate") || s.contains("unique") {
        StoreError::Duplicate
    } else {
        StoreError::Backend(s)
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

fn row_from(
    id: String,
    miner_hotkey: String,
    miner_coldkey: Option<String>,
    app_id: String,
    title: String,
    description: String,
    steps: Option<String>,
    status: String,
    agentic_verdict: Option<Value>,
    nearest_id: Option<String>,
    video_sha256: Option<String>,
    video_bytes: Option<i64>,
    video_path: Option<String>,
    reject_reason: Option<String>,
    epoch: i64,
    created_at_ms: i64,
    updated_at_ms: i64,
) -> BugRow {
    BugRow {
        id,
        miner_hotkey,
        miner_coldkey,
        app_id,
        title,
        description,
        steps,
        status: BugStatus::parse(&status).unwrap_or(BugStatus::Failed),
        agentic_verdict,
        nearest_id,
        video_sha256,
        video_bytes: video_bytes.map(|b| b.max(0).cast_unsigned()),
        video_path,
        reject_reason,
        epoch: epoch.max(0).cast_unsigned(),
        created_at_ms: created_at_ms.max(0).cast_unsigned(),
        updated_at_ms: updated_at_ms.max(0).cast_unsigned(),
    }
}

macro_rules! bug_select {
    () => {
        r#"
        SELECT id, miner_hotkey, miner_coldkey, app_id, title, description, steps,
               status, agentic_verdict, nearest_id, video_sha256, video_bytes, video_path,
               reject_reason, epoch,
               (EXTRACT(EPOCH FROM created_at) * 1000)::BIGINT AS created_at_ms,
               (EXTRACT(EPOCH FROM updated_at) * 1000)::BIGINT AS updated_at_ms
        FROM bounty_bug
        "#
    };
}

fn map_bug_row(r: sqlx::postgres::PgRow) -> Result<BugRow, StoreError> {
    Ok(row_from(
        r.try_get("id").map_err(map_sqlx)?,
        r.try_get("miner_hotkey").map_err(map_sqlx)?,
        r.try_get("miner_coldkey").map_err(map_sqlx)?,
        r.try_get("app_id").map_err(map_sqlx)?,
        r.try_get("title").map_err(map_sqlx)?,
        r.try_get("description").map_err(map_sqlx)?,
        r.try_get("steps").map_err(map_sqlx)?,
        r.try_get("status").map_err(map_sqlx)?,
        r.try_get("agentic_verdict").map_err(map_sqlx)?,
        r.try_get("nearest_id").map_err(map_sqlx)?,
        r.try_get("video_sha256").map_err(map_sqlx)?,
        r.try_get("video_bytes").map_err(map_sqlx)?,
        r.try_get("video_path").map_err(map_sqlx)?,
        r.try_get("reject_reason").map_err(map_sqlx)?,
        r.try_get("epoch").map_err(map_sqlx)?,
        r.try_get("created_at_ms").map_err(map_sqlx)?,
        r.try_get("updated_at_ms").map_err(map_sqlx)?,
    ))
}

#[async_trait]
impl BountyStore for DbBountyStore {
    async fn insert_bug(&self, row: &BugRow) -> Result<(), StoreError> {
        let epoch = i64::try_from(row.epoch).unwrap_or(i64::MAX);
        let res = sqlx::query(
            r#"
            INSERT INTO bounty_bug (
                id, miner_hotkey, miner_coldkey, app_id, title, description, steps,
                status, agentic_verdict, nearest_id, video_sha256, video_bytes, video_path,
                reject_reason, epoch
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
            "#,
        )
        .bind(&row.id)
        .bind(&row.miner_hotkey)
        .bind(row.miner_coldkey.as_deref())
        .bind(&row.app_id)
        .bind(&row.title)
        .bind(&row.description)
        .bind(row.steps.as_deref())
        .bind(row.status.as_str())
        .bind(&row.agentic_verdict)
        .bind(row.nearest_id.as_deref())
        .bind(row.video_sha256.as_deref())
        .bind(
            row.video_bytes
                .map(|b| i64::try_from(b).unwrap_or(i64::MAX)),
        )
        .bind(row.video_path.as_deref())
        .bind(row.reject_reason.as_deref())
        .bind(epoch)
        .execute(&self.pool)
        .await
        .map_err(map_sqlx)?;
        if res.rows_affected() == 0 {
            return Err(StoreError::Backend("insert_bug: no rows".into()));
        }
        sqlx::query(
            r#"
            INSERT INTO bounty_stage_event (bug_id, stage, detail)
            VALUES ($1, $2, NULL)
            "#,
        )
        .bind(&row.id)
        .bind(row.status.as_str())
        .execute(&self.pool)
        .await
        .map_err(map_sqlx)?;
        Ok(())
    }

    async fn get_bug(&self, id: &str) -> Result<Option<BugRow>, StoreError> {
        let q = concat!(bug_select!(), " WHERE id = $1");
        let row = sqlx::query(q)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(map_sqlx)?;
        row.map(map_bug_row).transpose()
    }

    async fn apply(
        &self,
        id: &str,
        patch: &BugPatch,
        event: Option<&StageEvent>,
    ) -> Result<BugRow, StoreError> {
        let mut tx = self.pool.begin().await.map_err(map_sqlx)?;
        let existing = sqlx::query(concat!(bug_select!(), " WHERE id = $1 FOR UPDATE"))
            .bind(id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(map_sqlx)?
            .ok_or(StoreError::NotFound)?;
        let mut row = map_bug_row(existing)?;
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
        sqlx::query(
            r#"
            UPDATE bounty_bug SET
                status = $2,
                agentic_verdict = $3,
                nearest_id = $4,
                video_sha256 = $5,
                video_bytes = $6,
                video_path = $7,
                reject_reason = $8,
                updated_at = now()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(row.status.as_str())
        .bind(&row.agentic_verdict)
        .bind(row.nearest_id.as_deref())
        .bind(row.video_sha256.as_deref())
        .bind(
            row.video_bytes
                .map(|b| i64::try_from(b).unwrap_or(i64::MAX)),
        )
        .bind(row.video_path.as_deref())
        .bind(row.reject_reason.as_deref())
        .execute(&mut *tx)
        .await
        .map_err(map_sqlx)?;
        if let Some(ev) = event {
            sqlx::query(
                r#"
                INSERT INTO bounty_stage_event (bug_id, stage, detail)
                VALUES ($1, $2, $3)
                "#,
            )
            .bind(id)
            .bind(&ev.stage)
            .bind(&ev.detail)
            .execute(&mut *tx)
            .await
            .map_err(map_sqlx)?;
        }
        tx.commit().await.map_err(map_sqlx)?;
        self.get_bug(id).await?.ok_or(StoreError::NotFound)
    }

    async fn claim_next(&self) -> Result<Option<BugRow>, StoreError> {
        let q = concat!(
            "WITH cte AS (",
            "  SELECT id FROM bounty_bug",
            "  WHERE status = 'uploaded'",
            "  ORDER BY created_at ASC",
            "  FOR UPDATE SKIP LOCKED",
            "  LIMIT 1",
            ")",
            "UPDATE bounty_bug b SET status = 'processing', updated_at = now()",
            "FROM cte WHERE b.id = cte.id",
            "RETURNING b.id, b.miner_hotkey, b.miner_coldkey, b.app_id, b.title, b.description,",
            "  b.steps, b.status, b.agentic_verdict, b.nearest_id, b.video_sha256, b.video_bytes,",
            "  b.video_path, b.reject_reason, b.epoch,",
            "  (EXTRACT(EPOCH FROM b.created_at) * 1000)::BIGINT AS created_at_ms,",
            "  (EXTRACT(EPOCH FROM b.updated_at) * 1000)::BIGINT AS updated_at_ms"
        );
        let row = sqlx::query(q)
            .fetch_optional(&self.pool)
            .await
            .map_err(map_sqlx)?;
        row.map(map_bug_row).transpose()
    }

    async fn list_bugs(
        &self,
        status: Option<&str>,
        miner: Option<&str>,
        limit: u32,
    ) -> Result<Vec<BugRow>, StoreError> {
        let lim = i64::from(limit);
        let q = concat!(
            bug_select!(),
            " WHERE ($1::text IS NULL OR status = $1)",
            "   AND ($2::text IS NULL OR miner_hotkey = $2)",
            " ORDER BY created_at DESC",
            " LIMIT $3"
        );
        let rows = sqlx::query(q)
            .bind(status)
            .bind(miner)
            .bind(lim)
            .fetch_all(&self.pool)
            .await
            .map_err(map_sqlx)?;
        rows.into_iter().map(map_bug_row).collect()
    }

    async fn list_similarity_corpus(
        &self,
        since_ms: u64,
        exclude_hotkey: &str,
        exclude_coldkey: Option<&str>,
        limit: u32,
    ) -> Result<Vec<BugRow>, StoreError> {
        let lim = i64::from(limit);
        let since = i64::try_from(since_ms).unwrap_or(i64::MAX);
        let q = concat!(
            bug_select!(),
            " WHERE status IN ('approved','pending_admin','rejected')",
            "   AND (EXTRACT(EPOCH FROM created_at) * 1000)::BIGINT >= $1",
            "   AND miner_hotkey <> $2",
            "   AND ($3::text IS NULL OR miner_coldkey IS DISTINCT FROM $3)",
            " ORDER BY created_at DESC",
            " LIMIT $4"
        );
        let rows = sqlx::query(q)
            .bind(since)
            .bind(exclude_hotkey)
            .bind(exclude_coldkey)
            .bind(lim)
            .fetch_all(&self.pool)
            .await
            .map_err(map_sqlx)?;
        rows.into_iter().map(map_bug_row).collect()
    }

    async fn events(&self, id: &str) -> Result<Vec<StageEvent>, StoreError> {
        let rows = sqlx::query(
            r#"
            SELECT stage, detail,
                   (EXTRACT(EPOCH FROM created_at) * 1000)::BIGINT AS at_ms
            FROM bounty_stage_event
            WHERE bug_id = $1
            ORDER BY created_at ASC
            "#,
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await
        .map_err(map_sqlx)?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let at_ms: i64 = r.try_get("at_ms").map_err(map_sqlx)?;
            out.push(StageEvent {
                stage: r.try_get("stage").map_err(map_sqlx)?,
                detail: r.try_get("detail").map_err(map_sqlx)?,
                at_ms: at_ms.max(0).cast_unsigned(),
            });
        }
        Ok(out)
    }

    async fn approved_points_for_epoch(
        &self,
        epoch: u64,
    ) -> Result<BTreeMap<String, u32>, StoreError> {
        let ep = i64::try_from(epoch).unwrap_or(i64::MAX);
        let rows = sqlx::query(
            r#"
            SELECT miner_hotkey, COUNT(*)::INT AS points
            FROM bounty_bug
            WHERE epoch = $1 AND status = 'approved'
            GROUP BY miner_hotkey
            "#,
        )
        .bind(ep)
        .fetch_all(&self.pool)
        .await
        .map_err(map_sqlx)?;
        let mut out = BTreeMap::new();
        for r in rows {
            let hk: String = r.try_get("miner_hotkey").map_err(map_sqlx)?;
            let pts: i32 = r.try_get("points").map_err(map_sqlx)?;
            out.insert(hk, pts.max(0).cast_unsigned());
        }
        Ok(out)
    }

    async fn upsert_epoch_score(&self, row: &EpochScoreRow) -> Result<(), StoreError> {
        let (kind, score, absence) = kind_score(row.final_score.as_ref());
        let ep = i64::try_from(row.epoch).unwrap_or(i64::MAX);
        let pts = i32::try_from(row.points).unwrap_or(i32::MAX);
        sqlx::query(
            r#"
            INSERT INTO bounty_epoch_score (epoch, miner_hotkey, points, kind, score, absence_reason)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (epoch, miner_hotkey) DO UPDATE SET
                points = EXCLUDED.points,
                kind = EXCLUDED.kind,
                score = EXCLUDED.score,
                absence_reason = EXCLUDED.absence_reason,
                updated_at = now()
            "#,
        )
        .bind(ep)
        .bind(&row.miner_hotkey)
        .bind(pts)
        .bind(kind)
        .bind(score)
        .bind(absence)
        .execute(&self.pool)
        .await
        .map_err(map_sqlx)?;
        Ok(())
    }

    async fn list_epoch_scores(&self, epoch: u64) -> Result<Vec<EpochScoreRow>, StoreError> {
        let ep = i64::try_from(epoch).unwrap_or(i64::MAX);
        let rows = sqlx::query(
            r#"
            SELECT epoch, miner_hotkey, points, kind, score, absence_reason
            FROM bounty_epoch_score
            WHERE epoch = $1
            ORDER BY miner_hotkey
            "#,
        )
        .bind(ep)
        .fetch_all(&self.pool)
        .await
        .map_err(map_sqlx)?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let epoch_i: i64 = r.try_get("epoch").map_err(map_sqlx)?;
            let points: i32 = r.try_get("points").map_err(map_sqlx)?;
            let kind: Option<String> = r.try_get("kind").map_err(map_sqlx)?;
            let score: Option<i64> = r.try_get("score").map_err(map_sqlx)?;
            let absence: Option<i16> = r.try_get("absence_reason").map_err(map_sqlx)?;
            out.push(EpochScoreRow {
                epoch: epoch_i.max(0).cast_unsigned(),
                miner_hotkey: r.try_get("miner_hotkey").map_err(map_sqlx)?,
                points: points.max(0).cast_unsigned(),
                final_score: final_from(kind.as_deref(), score, absence),
            });
        }
        Ok(out)
    }
}
