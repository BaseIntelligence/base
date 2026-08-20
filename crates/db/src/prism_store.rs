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
    /// Packed source-tree blob (migration 0018); null for two-script rows.
    pub tree_blob: Option<Vec<u8>>,
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
    /// Packed source-tree blob.
    pub tree_blob: Option<&'a [u8]>,
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

/// Column list shared by single-row reads (`get` / `claim` / `update`).
const COLS: &str = "id, miner_hotkey, miner_coldkey, epoch, netuid, status, label, \
    architecture_py, training_py, tree_blob, pod_id, pod_provider, receipt_json, metrics_json, \
    bpb, review_json, similarity_json, kind, score, absence_reason, retry_count, error_detail";

/// Listing projection: same `PrismSubmissionRow` shape without source trees,
/// telemetry series, or review blobs. Site/reconcile list of hundreds of rows
/// loading `tree_blob` (up to 17 MiB) + scripts was out-of-memory killing the
/// 16 GiB master host and bouncing `prism-challenge` (`control_plane_restart`
/// storms).
///
/// Presence of `receipt_json` / `metrics_json` is preserved as slim stubs so
/// `resume_measurement` / `mid_pod_resume` still classify post-measure vs
/// mid-pod. `n_params` is kept for the submissions list view.
const LIST_COLS: &str = "id, miner_hotkey, miner_coldkey, epoch, netuid, status, label, \
    ''::text AS architecture_py, ''::text AS training_py, NULL::bytea AS tree_blob, \
    pod_id, pod_provider, \
    CASE WHEN receipt_json IS NULL THEN NULL ELSE jsonb_build_object(\
      'provider', COALESCE(receipt_json->>'provider', ''), \
      'pod_id', COALESCE(receipt_json->>'pod_id', ''), \
      'image_digest', '', 'submission_hash', '', 'metrics_hash', '', \
      'termination_verified', COALESCE((receipt_json->>'termination_verified')::boolean, false)\
    ) END AS receipt_json, \
    CASE WHEN metrics_json IS NULL THEN NULL ELSE jsonb_strip_nulls(jsonb_build_object(\
      'bpb', COALESCE(metrics_json->'bpb', '0'::jsonb), \
      'tokens_seen', COALESCE(metrics_json->'tokens_seen', '0'::jsonb), \
      'wall_clock_seconds', COALESCE(metrics_json->'wall_clock_seconds', '0'::jsonb), \
      'notes', '', \
      'n_params', metrics_json->'n_params'\
    )) END AS metrics_json, \
    bpb, NULL::jsonb AS review_json, NULL::jsonb AS similarity_json, \
    kind, score, absence_reason, retry_count, error_detail";

/// Champion corpus for copy/similarity: keep `architecture_py`, drop trees
/// and training scripts (gates compare architecture only).
const CHAMPION_COLS: &str = "id, miner_hotkey, miner_coldkey, epoch, netuid, status, label, \
    architecture_py, ''::text AS training_py, NULL::bytea AS tree_blob, \
    pod_id, pod_provider, receipt_json, \
    CASE WHEN metrics_json IS NULL THEN NULL ELSE jsonb_strip_nulls(jsonb_build_object(\
      'bpb', COALESCE(metrics_json->'bpb', '0'::jsonb), \
      'tokens_seen', COALESCE(metrics_json->'tokens_seen', '0'::jsonb), \
      'wall_clock_seconds', COALESCE(metrics_json->'wall_clock_seconds', '0'::jsonb), \
      'notes', '', \
      'n_params', metrics_json->'n_params'\
    )) END AS metrics_json, \
    bpb, review_json, similarity_json, kind, score, absence_reason, retry_count, error_detail";

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
         (id, miner_hotkey, miner_coldkey, epoch, netuid, status, label, architecture_py, \
          training_py, tree_blob) \
         VALUES ($1, $2, $3, $4, $5, 'queued', $6, $7, $8, $9)",
    )
    .bind(n.id)
    .bind(n.miner_hotkey)
    .bind(n.miner_coldkey)
    .bind(n.epoch)
    .bind(n.netuid)
    .bind(n.label)
    .bind(n.architecture_py)
    .bind(n.training_py)
    .bind(n.tree_blob)
    .execute(pool)
    .await?;
    Ok(())
}

/// Fetch one row (`query_as` runtime-checked lane; no sqlx prepare).
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

/// Atomically claim the next queued row (`SKIP LOCKED`), moving it to
/// `provisioning`. Prefer rows that already have a `pod_id` (control-plane
/// resume) so they are not pushed behind a long FIFO of new submits — those
/// would take the concurrent slots and trigger a second `lium rent`.
/// Among the same class, oldest `created_at` first.
///
/// # Errors
/// SQL error.
pub async fn claim_prism_submission(pool: &PgPool) -> Result<Option<PrismSubmissionRow>, DbError> {
    let q = format!(
        "UPDATE prism_submission SET status = 'provisioning', updated_at = now() \
         WHERE id = ( \
           SELECT id FROM prism_submission WHERE status = 'queued' \
           ORDER BY (pod_id IS NOT NULL) DESC, created_at ASC \
           LIMIT 1 FOR UPDATE SKIP LOCKED \
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

/// Reset a row for retry: clears score fields and re-queues.
/// Completed measurements are retained for post-run review retries; rows
/// without metrics drop stale pod/exec fields and re-provision cleanly.
/// When `bump_retry`, increments `retry_count` and clears screens
/// (manual / llm_infra / ast_infra). When false (Lium 429 / `no_capacity`),
/// keeps attempts **and** similarity/review so sold-out rent ticks do not
/// re-bill LLM screens.
///
/// # Errors
/// SQL error / 0 rows for id.
pub async fn reset_prism_submission_for_retry(
    pool: &PgPool,
    id: &str,
    bump_retry: bool,
) -> Result<PrismSubmissionRow, DbError> {
    let q = format!(
        "UPDATE prism_submission SET \
            status = 'queued', \
            pod_id = CASE WHEN metrics_json IS NULL THEN NULL ELSE pod_id END, \
            pod_provider = CASE WHEN metrics_json IS NULL THEN NULL ELSE pod_provider END, \
            receipt_json = CASE WHEN metrics_json IS NULL THEN NULL ELSE receipt_json END, \
            bpb = CASE WHEN metrics_json IS NULL THEN NULL ELSE bpb END, \
            review_json = CASE WHEN $2 THEN NULL ELSE review_json END, \
            similarity_json = CASE WHEN $2 THEN NULL ELSE similarity_json END, \
            kind = NULL, score = NULL, absence_reason = NULL, emitted_epoch = NULL, \
            error_detail = NULL, \
            retry_count = CASE WHEN $2 THEN retry_count + 1 ELSE retry_count END, \
            updated_at = now() \
          WHERE id = $1 RETURNING {COLS}"
    );
    let row = sqlx::query_as::<_, PrismSubmissionRow>(&q)
        .bind(id)
        .bind(bump_retry)
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
        "SELECT {LIST_COLS} FROM prism_submission \
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

/// Champion corpus: historical Score>0 WTA/leaf winners (current top + ex-tops).
///
/// # Errors
/// SQL error.
pub async fn list_prism_champions(
    pool: &PgPool,
    limit: i64,
) -> Result<Vec<PrismSubmissionRow>, DbError> {
    let q = format!(
        "SELECT {CHAMPION_COLS} FROM prism_submission \
         WHERE kind = 'score' AND score > 0 \
         ORDER BY created_at DESC LIMIT $1"
    );
    let rows = sqlx::query_as::<_, PrismSubmissionRow>(&q)
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

/// Read precheck attempts used for `(coldkey, UTC day)` (0 when absent).
///
/// # Errors
/// SQL error.
pub async fn prism_precheck_quota_get(
    pool: &PgPool,
    miner_coldkey: &str,
    day: &str,
) -> Result<i32, DbError> {
    let row: Option<(i32,)> = sqlx::query_as(
        "SELECT checks_used FROM prism_precheck_quota \
         WHERE miner_coldkey = $1 AND day = $2::date",
    )
    .bind(miner_coldkey)
    .bind(day)
    .fetch_optional(pool)
    .await?;
    Ok(row.map_or(0, |r| r.0))
}

/// Atomically consume one precheck attempt when `checks_used < limit`.
/// Returns `Some(checks_used)` after bump, or `None` when already at limit.
///
/// # Errors
/// SQL error.
pub async fn prism_precheck_quota_try_consume(
    pool: &PgPool,
    miner_coldkey: &str,
    day: &str,
    limit: i32,
) -> Result<Option<i32>, DbError> {
    let row: Option<(i32,)> = sqlx::query_as(
        "INSERT INTO prism_precheck_quota (miner_coldkey, day, checks_used) \
         VALUES ($1, $2::date, 1) \
         ON CONFLICT (miner_coldkey, day) DO UPDATE SET \
           checks_used = prism_precheck_quota.checks_used + 1 \
         WHERE prism_precheck_quota.checks_used < $3 \
         RETURNING checks_used",
    )
    .bind(miner_coldkey)
    .bind(day)
    .bind(limit)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| r.0))
}
