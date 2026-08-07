//! Postgres store for PRISM submissions (production path).
//!
//! Implements the challenge-side trait in `prism_challenge::store` without
//! creating a dependency cycle: `db` does not know `prism_challenge`; the
//! challenge crate provides thin adapters around these plain rows.

use serde_json::Value;
use sqlx::PgPool;

use crate::DbError;

/// One `prism_submission` row, plain SQL shape.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct PrismSubmissionRow {
    /// Contract-bytes hash.
    pub id: String,
    /// Miner hotkey (lowercase 64 hex).
    pub miner_hotkey: String,
    /// Owning coldkey (lowercase 64 hex), when known at intake.
    pub miner_coldkey: Option<String>,
    /// Epoch at acceptance.
    pub epoch: i64,
    /// Netuid.
    pub netuid: i32,
    /// Lifecycle string (see table CHECK).
    pub status: String,
    /// Label.
    pub label: Option<String>,
    /// architecture.py.
    pub architecture_py: String,
    /// training.py.
    pub training_py: String,
    /// Pod id.
    pub pod_id: Option<String>,
    /// Pod provider.
    pub pod_provider: Option<String>,
    /// Receipt JSON.
    pub receipt_json: Option<Value>,
    /// Metrics JSON.
    pub metrics_json: Option<Value>,
    /// bpb.
    pub bpb: Option<f64>,
    /// Review verdict JSON.
    pub review_json: Option<Value>,
    /// Similarity verdict JSON.
    pub similarity_json: Option<Value>,
    /// `score` | `no_score"`.
    pub kind: Option<String>,
    /// Score lattice.
    pub score: Option<i64>,
    /// Absence reason code.
    pub absence_reason: Option<i16>,
    /// Attempt count.
    pub retry_count: i32,
    /// Failure detail.
    pub error_detail: Option<String>,
}

/// New queued row.
#[derive(Debug)]
pub struct NewPrismSubmission<'a> {
    /// id sha256 hex (64).
    pub id: &'a str,
    /// miner hotkey.
    pub miner_hotkey: &'a str,
    /// owning coldkey (optional).
    pub miner_coldkey: Option<&'a str>,
    /// epoch.
    pub epoch: i64,
    /// netuid.
    pub netuid: i32,
    /// label.
    pub label: Option<&'a str>,
    /// architecture.py.
    pub architecture_py: &'a str,
    /// training.py.
    pub training_py: &'a str,
}

/// Stage event entry.
#[derive(Debug)]
pub struct NewPrismStageEvent<'a> {
    /// FK row.
    pub submission_id: &'a str,
    /// stage string.
    pub stage: &'a str,
    /// detail.
    pub detail: Option<Value>,
}

/// Column list shared by all row reads.
const COLS: &str = "id, miner_hotkey, miner_coldkey, epoch, netuid, status, label, \
    architecture_py, training_py, pod_id, pod_provider, receipt_json, metrics_json, bpb, \
    review_json, similarity_json, kind, score, absence_reason, retry_count, error_detail";

/// Insert the queued row.
///
/// # Errors
/// SQL error.
pub async fn insert_prism_submission(
    pool: &PgPool,
    n: &NewPrismSubmission<'_>,
) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO prism_submission \
         (id, miner_hotkey, miner_coldkey, epoch, netuid, status, label, architecture_py, training_py) \
         VALUES ($1, $2, $3, $4, $5, 'queued', $6, $7, $8)",
    )
    .bind(n.id)
    .bind(n.miner_hotkey)
    .bind(n.miner_coldkey)
    .bind(n.epoch)
    .bind(n.netuid)
    .bind(n.label)
    .bind(n.architecture_py)
    .bind(n.training_py)
    .execute(pool)
    .await?;
    Ok(())
}

/// Fetch one row.
///
/// `query_as` (not the macro) because the workspace denies `sqlx prepare` on
/// flaky networks: this file uses the runtime-checked lane like `miner_endpoint`.
///
/// # Errors
/// SQL error.
pub async fn prism_submission(
    pool: &PgPool,
    id: &str,
) -> Result<Option<PrismSubmissionRow>, DbError> {
    let q = format!("SELECT {COLS} FROM prism_submission WHERE id = $1");
    let row = sqlx::query_as::<_, PrismSubmissionRow>(&q)
        .bind(id)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

/// Atomically claim the oldest queued row (`SKIP LOCKED`), moving it to
/// `provisioning`. Returns None when empty.
///
/// # Errors
/// SQL error.
pub async fn claim_prism_submission(pool: &PgPool) -> Result<Option<PrismSubmissionRow>, DbError> {
    let q = format!(
        "UPDATE prism_submission SET status = 'provisioning', updated_at = now() \
         WHERE id = ( \
           SELECT id FROM prism_submission WHERE status = 'queued' \
           ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED \
         ) RETURNING {COLS}"
    );
    let row = sqlx::query_as::<_, PrismSubmissionRow>(&q)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

/// Apply a field update (COALESCE keeps unchanged fields) + stamp `updated_at`.
///
/// Soft-typed callers pre-serialize each field into JSON-friendly strings;
/// this function is intentionally low-level per field so the typed layer
/// holds the invariants.
///
/// # Errors
/// SQL error / 0 rows (missing row).
#[allow(clippy::too_many_arguments)]
pub async fn update_prism_submission(
    pool: &PgPool,
    id: &str,
    status: Option<&str>,
    pod_id: Option<&str>,
    pod_provider: Option<&str>,
    receipt_json: Option<Value>,
    metrics_json: Option<Value>,
    bpb: Option<f64>,
    review_json: Option<Value>,
    similarity_json: Option<Value>,
    kind: Option<&str>,
    score: Option<i64>,
    absence_reason: Option<i16>,
    retry_bump: i32,
    error_detail: Option<&str>,
) -> Result<PrismSubmissionRow, DbError> {
    let q = format!(
        "UPDATE prism_submission SET \
           status = COALESCE($2, status), \
           pod_id = COALESCE($3, pod_id), \
           pod_provider = COALESCE($4, pod_provider), \
           receipt_json = COALESCE($5, receipt_json), \
           metrics_json = COALESCE($6, metrics_json), \
           bpb = COALESCE($7, bpb), \
           review_json = COALESCE($8, review_json), \
           similarity_json = COALESCE($9, similarity_json), \
           kind = COALESCE($10, kind), \
           score = COALESCE($11, score), \
           absence_reason = COALESCE($12, absence_reason), \
           retry_count = retry_count + $13, \
           error_detail = COALESCE($14, error_detail), \
           updated_at = now() \
         WHERE id = $1 RETURNING {COLS}"
    );
    let row = sqlx::query_as::<_, PrismSubmissionRow>(&q)
        .bind(id)
        .bind(status)
        .bind(pod_id)
        .bind(pod_provider)
        .bind(receipt_json)
        .bind(metrics_json)
        .bind(bpb)
        .bind(review_json)
        .bind(similarity_json)
        .bind(kind)
        .bind(score)
        .bind(absence_reason)
        .bind(retry_bump)
        .bind(error_detail)
        .fetch_one(pool)
        .await?;
    Ok(row)
}

/// Reset a failed row for a retry: clears all execution/score fields and
/// re-queues it. `retry_count` is bumped (policy enforced by the caller).
///
/// # Errors
/// SQL error / 0 rows for id.
pub async fn reset_prism_submission_for_retry(
    pool: &PgPool,
    id: &str,
) -> Result<PrismSubmissionRow, DbError> {
    let q = format!(
        "UPDATE prism_submission SET            status = 'queued', pod_id = NULL, pod_provider = NULL,            receipt_json = NULL, metrics_json = NULL, bpb = NULL,            review_json = NULL, similarity_json = NULL,            kind = NULL, score = NULL, absence_reason = NULL, emitted_epoch = NULL,            error_detail = NULL, retry_count = retry_count + 1, updated_at = now()          WHERE id = $1 RETURNING {COLS}"
    );
    let row = sqlx::query_as::<_, PrismSubmissionRow>(&q)
        .bind(id)
        .fetch_one(pool)
        .await?;
    Ok(row)
}

/// Append a stage event.
///
/// # Errors
/// SQL error.
pub async fn insert_prism_stage_event(
    pool: &PgPool,
    e: &NewPrismStageEvent<'_>,
) -> Result<(), DbError> {
    sqlx::query("INSERT INTO prism_stage_event (submission_id, stage, detail) VALUES ($1, $2, $3)")
        .bind(e.submission_id)
        .bind(e.stage)
        .bind(&e.detail)
        .execute(pool)
        .await?;
    Ok(())
}

/// List recent rows (newest first), optionally by status and/or miner hotkey.
///
/// # Errors
/// SQL error.
pub async fn list_prism_submissions(
    pool: &PgPool,
    status: Option<&str>,
    miner: Option<&str>,
    limit: i64,
) -> Result<Vec<PrismSubmissionRow>, DbError> {
    let miner = miner
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase);
    let q = format!(
        "SELECT {COLS} FROM prism_submission \
         WHERE ($1::TEXT IS NULL OR status = $1) \
           AND ($2::TEXT IS NULL OR lower(miner_hotkey) = $2) \
         ORDER BY created_at DESC LIMIT $3"
    );
    let rows = sqlx::query_as::<_, PrismSubmissionRow>(&q)
        .bind(status)
        .bind(miner)
        .bind(limit)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

/// Ascending event journal for one row.
///
/// # Errors
/// SQL error.
pub async fn prism_stage_events(
    pool: &PgPool,
    submission_id: &str,
) -> Result<Vec<(String, Option<Value>, i64)>, DbError> {
    let rows: Vec<(String, Option<Value>, i64)> = sqlx::query_as(
        "SELECT stage, detail, (EXTRACT(EPOCH FROM created_at)::BIGINT) \
         FROM prism_stage_event WHERE submission_id = $1 ORDER BY created_at ASC",
    )
    .bind(submission_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Stage statuses stuck non-terminal beyond the grace window (restart sweep).
///
/// # Errors
/// SQL error.
pub async fn stuck_prism_before_grace(
    pool: &PgPool,
    older_than_secs: i64,
) -> Result<Vec<(String, String, Option<String>)>, DbError> {
    let rows: Vec<(String, String, Option<String>)> = sqlx::query_as(
        "SELECT id, status, pod_id FROM prism_submission \
         WHERE status IN ('provisioning','running','llm_review','similarity','scoring') \
           AND updated_at < now() - make_interval(secs => $1)",
    )
    .bind(older_than_secs)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}
