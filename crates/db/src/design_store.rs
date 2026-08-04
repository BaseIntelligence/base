//! Postgres store for design-challenge tables (production path).
//!
//! Plain SQL shapes only — the typed [`design_store`] crate adapts these rows.

use serde_json::Value;
use sqlx::PgPool;

use crate::DbError;

/// `design_harness` row.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DesignHarnessRow {
    /// Content digest id.
    pub id: String,
    /// Miner hotkey (lowercase 64 hex).
    pub miner_hotkey: String,
    /// agent.py source.
    pub agent_py: String,
    /// pyproject.toml.
    pub pyproject_toml: String,
    /// Extra files map.
    pub extra_files: Value,
    /// Whether the harness is active.
    pub active: bool,
    /// Cooldown: eliminated until this round id.
    pub eliminated_until_round: i64,
}

/// `design_round` row.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DesignRoundRow {
    /// Round id (`floor(unix/21600)`).
    pub round_id: i64,
    /// Chain epoch at open.
    pub epoch: i64,
    /// Netuid.
    pub netuid: i32,
    /// Prompt-set digest (hex).
    pub prompt_set_digest: String,
    /// Lifecycle status.
    pub status: String,
}

/// `design_run` row.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DesignRunRow {
    /// Run id.
    pub id: String,
    /// Round.
    pub round_id: i64,
    /// Harness.
    pub harness_id: String,
    /// Prompt id.
    pub prompt_id: String,
    /// Lifecycle status.
    pub status: String,
    /// Artifact digest.
    pub artifact_digest: Option<String>,
    /// Sanitize report JSON.
    pub sanitize_report: Option<Value>,
    /// Agentic anti-cheat verdict JSON.
    pub agentic_verdict: Option<Value>,
    /// Error detail.
    pub error_detail: Option<String>,
    /// `score` | `no_score`.
    pub kind: Option<String>,
    /// Lattice score.
    pub score: Option<i64>,
    /// Absence code.
    pub absence_reason: Option<i16>,
    /// Retry count.
    pub retry_count: i32,
}

/// `design_round_award` row.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DesignRoundAwardRow {
    /// Round.
    pub round_id: i64,
    /// Winner harness ids JSON array.
    pub winner_harness_ids: Value,
    /// Awarded unix secs.
    pub awarded_at_secs: i64,
    /// Admin token hash.
    pub admin_token_hash: Option<String>,
}

/// `design_artifact` row (viewer serves `sanitized_html` only).
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DesignArtifactRow {
    /// Run id.
    pub run_id: String,
    /// Page path (`index.html`, …).
    pub path: String,
    /// Sanitized HTML.
    pub sanitized_html: String,
    /// Raw HTML (audit only).
    pub raw_html: String,
    /// Raw sha256 hex.
    pub raw_sha256: String,
    /// Byte length of raw.
    pub bytes: i32,
}

/// `design_pair` row.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DesignPairRow {
    /// Pair id.
    pub id: String,
    /// Round.
    pub round_id: i64,
    /// Prompt.
    pub prompt_id: String,
    /// Run A.
    pub run_a_id: String,
    /// Run B.
    pub run_b_id: String,
}

/// `design_annotation` row.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DesignAnnotationRow {
    /// Pair.
    pub pair_id: String,
    /// Annotator id.
    pub annotator_id: String,
    /// Winner run id.
    pub winner_run_id: String,
    /// Created unix secs.
    pub created_at_secs: i64,
}

/// `design_rating` row.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct DesignRatingRow {
    /// Round.
    pub round_id: i64,
    /// Miner.
    pub miner_hotkey: String,
    /// Elo rating (integer).
    pub rating: i32,
    /// Wins.
    pub wins: i32,
    /// Losses.
    pub losses: i32,
    /// Lattice score.
    pub score: Option<i64>,
    /// Absence.
    pub absence_reason: Option<i16>,
    /// `score` | `no_score`.
    pub kind: Option<String>,
}

/// New harness insert.
#[derive(Debug)]
pub struct NewDesignHarness<'a> {
    /// id.
    pub id: &'a str,
    /// miner.
    pub miner_hotkey: &'a str,
    /// agent.py.
    pub agent_py: &'a str,
    /// pyproject.
    pub pyproject_toml: &'a str,
    /// extra files JSON.
    pub extra_files: Value,
}

/// New round insert.
#[derive(Debug)]
pub struct NewDesignRound<'a> {
    /// round id.
    pub round_id: i64,
    /// epoch.
    pub epoch: i64,
    /// netuid.
    pub netuid: i32,
    /// `opens_at` as unix secs.
    pub opens_at_secs: i64,
    /// `closes_at` as unix secs.
    pub closes_at_secs: i64,
    /// digest.
    pub prompt_set_digest: &'a str,
    /// status.
    pub status: &'a str,
}

/// New run insert.
#[derive(Debug)]
pub struct NewDesignRun<'a> {
    /// id.
    pub id: &'a str,
    /// round.
    pub round_id: i64,
    /// harness.
    pub harness_id: &'a str,
    /// prompt.
    pub prompt_id: &'a str,
}

/// New artifact.
#[derive(Debug)]
pub struct NewDesignArtifact<'a> {
    /// run.
    pub run_id: &'a str,
    /// path.
    pub path: &'a str,
    /// sanitized.
    pub sanitized_html: &'a str,
    /// raw.
    pub raw_html: &'a str,
    /// sha.
    pub raw_sha256: &'a str,
    /// bytes.
    pub bytes: i32,
}

/// New pair.
#[derive(Debug)]
pub struct NewDesignPair<'a> {
    /// id.
    pub id: &'a str,
    /// round.
    pub round_id: i64,
    /// prompt.
    pub prompt_id: &'a str,
    /// a.
    pub run_a_id: &'a str,
    /// b.
    pub run_b_id: &'a str,
}

/// New annotation.
#[derive(Debug)]
pub struct NewDesignAnnotation<'a> {
    /// pair.
    pub pair_id: &'a str,
    /// annotator.
    pub annotator_id: &'a str,
    /// winner.
    pub winner_run_id: &'a str,
}

/// Stage event.
#[derive(Debug)]
pub struct NewDesignStageEvent<'a> {
    /// run.
    pub run_id: &'a str,
    /// stage.
    pub stage: &'a str,
    /// detail.
    pub detail: Option<Value>,
}

const HARNESS_COLS: &str =
    "id, miner_hotkey, agent_py, pyproject_toml, extra_files, active, eliminated_until_round";
const ROUND_COLS: &str = "round_id, epoch, netuid, prompt_set_digest, status";
const RUN_COLS: &str = "id, round_id, harness_id, prompt_id, status, artifact_digest, \
    sanitize_report, agentic_verdict, error_detail, kind, score, absence_reason, retry_count";
const ARTIFACT_COLS: &str = "run_id, path, sanitized_html, raw_html, raw_sha256, bytes";
const PAIR_COLS: &str = "id, round_id, prompt_id, run_a_id, run_b_id";
const RATING_COLS: &str =
    "round_id, miner_hotkey, rating, wins, losses, score, absence_reason, kind";

/// Insert harness (idempotent on PK conflict → error for caller to treat as duplicate).
///
/// # Errors
/// SQL error.
pub async fn insert_design_harness(pool: &PgPool, n: &NewDesignHarness<'_>) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO design_harness (id, miner_hotkey, agent_py, pyproject_toml, extra_files) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(n.id)
    .bind(n.miner_hotkey)
    .bind(n.agent_py)
    .bind(n.pyproject_toml)
    .bind(&n.extra_files)
    .execute(pool)
    .await?;
    Ok(())
}

/// Fetch harness.
///
/// # Errors
/// SQL error.
pub async fn design_harness(pool: &PgPool, id: &str) -> Result<Option<DesignHarnessRow>, DbError> {
    let q = format!("SELECT {HARNESS_COLS} FROM design_harness WHERE id = $1");
    Ok(sqlx::query_as::<_, DesignHarnessRow>(&q)
        .bind(id)
        .fetch_optional(pool)
        .await?)
}

/// List harnesses for a miner.
///
/// # Errors
/// SQL error.
pub async fn list_design_harnesses(
    pool: &PgPool,
    miner_hotkey: &str,
) -> Result<Vec<DesignHarnessRow>, DbError> {
    let q = format!(
        "SELECT {HARNESS_COLS} FROM design_harness WHERE miner_hotkey = $1 ORDER BY created_at DESC"
    );
    Ok(sqlx::query_as::<_, DesignHarnessRow>(&q)
        .bind(miner_hotkey)
        .fetch_all(pool)
        .await?)
}

/// Recent harnesses (corpus / admin), newest first.
///
/// # Errors
/// SQL error.
pub async fn list_recent_design_harnesses(
    pool: &PgPool,
    limit: i64,
) -> Result<Vec<DesignHarnessRow>, DbError> {
    let q = format!("SELECT {HARNESS_COLS} FROM design_harness ORDER BY created_at DESC LIMIT $1");
    Ok(sqlx::query_as::<_, DesignHarnessRow>(&q)
        .bind(limit)
        .fetch_all(pool)
        .await?)
}

/// Set elimination cooldown.
///
/// # Errors
/// SQL error.
pub async fn set_design_harness_eliminated(
    pool: &PgPool,
    id: &str,
    until_round: i64,
) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE design_harness SET eliminated_until_round = $2, updated_at = now() WHERE id = $1",
    )
    .bind(id)
    .bind(until_round)
    .execute(pool)
    .await?;
    Ok(())
}

/// Insert round.
///
/// # Errors
/// SQL error.
pub async fn insert_design_round(pool: &PgPool, n: &NewDesignRound<'_>) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO design_round (round_id, epoch, netuid, opens_at, closes_at, prompt_set_digest, status) \
         VALUES ($1, $2, $3, to_timestamp($4), to_timestamp($5), $6, $7)",
    )
    .bind(n.round_id)
    .bind(n.epoch)
    .bind(n.netuid)
    .bind(n.opens_at_secs)
    .bind(n.closes_at_secs)
    .bind(n.prompt_set_digest)
    .bind(n.status)
    .execute(pool)
    .await?;
    Ok(())
}

/// Fetch round.
///
/// # Errors
/// SQL error.
pub async fn design_round(pool: &PgPool, round_id: i64) -> Result<Option<DesignRoundRow>, DbError> {
    let q = format!("SELECT {ROUND_COLS} FROM design_round WHERE round_id = $1");
    Ok(sqlx::query_as::<_, DesignRoundRow>(&q)
        .bind(round_id)
        .fetch_optional(pool)
        .await?)
}

/// Update round status.
///
/// # Errors
/// SQL error.
pub async fn update_design_round_status(
    pool: &PgPool,
    round_id: i64,
    status: &str,
) -> Result<(), DbError> {
    sqlx::query("UPDATE design_round SET status = $2, updated_at = now() WHERE round_id = $1")
        .bind(round_id)
        .bind(status)
        .execute(pool)
        .await?;
    Ok(())
}

/// List recent rounds.
///
/// # Errors
/// SQL error.
pub async fn list_design_rounds(pool: &PgPool, limit: i64) -> Result<Vec<DesignRoundRow>, DbError> {
    let q = format!("SELECT {ROUND_COLS} FROM design_round ORDER BY round_id DESC LIMIT $1");
    Ok(sqlx::query_as::<_, DesignRoundRow>(&q)
        .bind(limit)
        .fetch_all(pool)
        .await?)
}

/// Insert queued run.
///
/// # Errors
/// SQL error.
pub async fn insert_design_run(pool: &PgPool, n: &NewDesignRun<'_>) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO design_run (id, round_id, harness_id, prompt_id, status) \
         VALUES ($1, $2, $3, $4, 'queued')",
    )
    .bind(n.id)
    .bind(n.round_id)
    .bind(n.harness_id)
    .bind(n.prompt_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Fetch run.
///
/// # Errors
/// SQL error.
pub async fn design_run(pool: &PgPool, id: &str) -> Result<Option<DesignRunRow>, DbError> {
    let q = format!("SELECT {RUN_COLS} FROM design_run WHERE id = $1");
    Ok(sqlx::query_as::<_, DesignRunRow>(&q)
        .bind(id)
        .fetch_optional(pool)
        .await?)
}

/// Claim next queued run (`SKIP LOCKED` → installing).
///
/// # Errors
/// SQL error.
pub async fn claim_design_run(pool: &PgPool) -> Result<Option<DesignRunRow>, DbError> {
    let q = format!(
        "UPDATE design_run SET status = 'installing', updated_at = now() \
         WHERE id = ( \
           SELECT id FROM design_run WHERE status = 'queued' \
           ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED \
         ) RETURNING {RUN_COLS}"
    );
    Ok(sqlx::query_as::<_, DesignRunRow>(&q)
        .fetch_optional(pool)
        .await?)
}

/// COALESCE update for a run.
///
/// # Errors
/// SQL error / missing row.
#[allow(clippy::too_many_arguments)]
pub async fn update_design_run(
    pool: &PgPool,
    id: &str,
    status: Option<&str>,
    artifact_digest: Option<&str>,
    sanitize_report: Option<Value>,
    agentic_verdict: Option<Value>,
    error_detail: Option<&str>,
    kind: Option<&str>,
    score: Option<i64>,
    absence_reason: Option<i16>,
    retry_bump: i32,
) -> Result<DesignRunRow, DbError> {
    let q = format!(
        "UPDATE design_run SET \
           status = COALESCE($2, status), \
           artifact_digest = COALESCE($3, artifact_digest), \
           sanitize_report = COALESCE($4, sanitize_report), \
           agentic_verdict = COALESCE($5, agentic_verdict), \
           error_detail = COALESCE($6, error_detail), \
           kind = COALESCE($7, kind), \
           score = COALESCE($8, score), \
           absence_reason = COALESCE($9, absence_reason), \
           retry_count = retry_count + $10, \
           updated_at = now() \
         WHERE id = $1 RETURNING {RUN_COLS}"
    );
    Ok(sqlx::query_as::<_, DesignRunRow>(&q)
        .bind(id)
        .bind(status)
        .bind(artifact_digest)
        .bind(sanitize_report)
        .bind(agentic_verdict)
        .bind(error_detail)
        .bind(kind)
        .bind(score)
        .bind(absence_reason)
        .bind(retry_bump)
        .fetch_one(pool)
        .await?)
}

/// Reset failed run to queued.
///
/// # Errors
/// SQL error.
pub async fn reset_design_run_for_retry(pool: &PgPool, id: &str) -> Result<DesignRunRow, DbError> {
    let q = format!(
        "UPDATE design_run SET status = 'queued', artifact_digest = NULL, \
           sanitize_report = NULL, agentic_verdict = NULL, error_detail = NULL, \
           kind = NULL, score = NULL, absence_reason = NULL, \
           retry_count = retry_count + 1, updated_at = now() \
         WHERE id = $1 RETURNING {RUN_COLS}"
    );
    Ok(sqlx::query_as::<_, DesignRunRow>(&q)
        .bind(id)
        .fetch_one(pool)
        .await?)
}

/// List runs (optional status).
///
/// # Errors
/// SQL error.
pub async fn list_design_runs(
    pool: &PgPool,
    status: Option<&str>,
    limit: i64,
) -> Result<Vec<DesignRunRow>, DbError> {
    let q = format!(
        "SELECT {RUN_COLS} FROM design_run \
         WHERE ($1::TEXT IS NULL OR status = $1) \
         ORDER BY created_at DESC LIMIT $2"
    );
    Ok(sqlx::query_as::<_, DesignRunRow>(&q)
        .bind(status)
        .bind(limit)
        .fetch_all(pool)
        .await?)
}

/// Runs for a round.
///
/// # Errors
/// SQL error.
pub async fn design_runs_for_round(
    pool: &PgPool,
    round_id: i64,
) -> Result<Vec<DesignRunRow>, DbError> {
    let q =
        format!("SELECT {RUN_COLS} FROM design_run WHERE round_id = $1 ORDER BY created_at ASC");
    Ok(sqlx::query_as::<_, DesignRunRow>(&q)
        .bind(round_id)
        .fetch_all(pool)
        .await?)
}

/// Stuck non-terminal runs.
///
/// # Errors
/// SQL error.
pub async fn stuck_design_runs(
    pool: &PgPool,
    older_than_secs: i64,
) -> Result<Vec<DesignRunRow>, DbError> {
    let q = format!(
        "SELECT {RUN_COLS} FROM design_run \
         WHERE status IN ('installing','running','sanitizing','agentic_review') \
           AND updated_at < now() - make_interval(secs => $1)"
    );
    Ok(sqlx::query_as::<_, DesignRunRow>(&q)
        .bind(older_than_secs)
        .fetch_all(pool)
        .await?)
}

/// Insert artifact.
///
/// # Errors
/// SQL error.
pub async fn insert_design_artifact(
    pool: &PgPool,
    n: &NewDesignArtifact<'_>,
) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO design_artifact (run_id, path, sanitized_html, raw_html, raw_sha256, bytes) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(n.run_id)
    .bind(n.path)
    .bind(n.sanitized_html)
    .bind(n.raw_html)
    .bind(n.raw_sha256)
    .bind(n.bytes)
    .execute(pool)
    .await?;
    Ok(())
}

/// List artifacts for a run (includes raw for audit; callers must not serve it).
///
/// # Errors
/// SQL error.
pub async fn design_artifacts(
    pool: &PgPool,
    run_id: &str,
) -> Result<Vec<DesignArtifactRow>, DbError> {
    let q = format!("SELECT {ARTIFACT_COLS} FROM design_artifact WHERE run_id = $1 ORDER BY path");
    Ok(sqlx::query_as::<_, DesignArtifactRow>(&q)
        .bind(run_id)
        .fetch_all(pool)
        .await?)
}

/// One sanitized page.
///
/// # Errors
/// SQL error.
pub async fn design_artifact_page(
    pool: &PgPool,
    run_id: &str,
    path: &str,
) -> Result<Option<(String, String)>, DbError> {
    let row: Option<(String, String)> = sqlx::query_as(
        "SELECT path, sanitized_html FROM design_artifact WHERE run_id = $1 AND path = $2",
    )
    .bind(run_id)
    .bind(path)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Insert pair.
///
/// # Errors
/// SQL error.
pub async fn insert_design_pair(pool: &PgPool, n: &NewDesignPair<'_>) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO design_pair (id, round_id, prompt_id, run_a_id, run_b_id) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(n.id)
    .bind(n.round_id)
    .bind(n.prompt_id)
    .bind(n.run_a_id)
    .bind(n.run_b_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Fetch pair.
///
/// # Errors
/// SQL error.
pub async fn design_pair(pool: &PgPool, id: &str) -> Result<Option<DesignPairRow>, DbError> {
    let q = format!("SELECT {PAIR_COLS} FROM design_pair WHERE id = $1");
    Ok(sqlx::query_as::<_, DesignPairRow>(&q)
        .bind(id)
        .fetch_optional(pool)
        .await?)
}

/// Pairs for a round.
///
/// # Errors
/// SQL error.
pub async fn design_pairs_for_round(
    pool: &PgPool,
    round_id: i64,
) -> Result<Vec<DesignPairRow>, DbError> {
    let q =
        format!("SELECT {PAIR_COLS} FROM design_pair WHERE round_id = $1 ORDER BY created_at ASC");
    Ok(sqlx::query_as::<_, DesignPairRow>(&q)
        .bind(round_id)
        .fetch_all(pool)
        .await?)
}

/// Pairs for a round lacking enough annotations for `annotator_id`.
///
/// # Errors
/// SQL error.
pub async fn next_design_pair_for_annotator(
    pool: &PgPool,
    round_id: i64,
    annotator_id: &str,
    min_annotations: i64,
) -> Result<Option<DesignPairRow>, DbError> {
    let q = "SELECT p.id, p.round_id, p.prompt_id, p.run_a_id, p.run_b_id \
         FROM design_pair p \
         WHERE p.round_id = $1 \
           AND NOT EXISTS ( \
             SELECT 1 FROM design_annotation a \
             WHERE a.pair_id = p.id AND a.annotator_id = $2 \
           ) \
           AND (SELECT COUNT(*) FROM design_annotation a2 WHERE a2.pair_id = p.id) < $3 \
         ORDER BY p.created_at ASC LIMIT 1";
    Ok(sqlx::query_as::<_, DesignPairRow>(q)
        .bind(round_id)
        .bind(annotator_id)
        .bind(min_annotations)
        .fetch_optional(pool)
        .await?)
}

/// Insert annotation.
///
/// # Errors
/// SQL error.
pub async fn insert_design_annotation(
    pool: &PgPool,
    n: &NewDesignAnnotation<'_>,
) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO design_annotation (pair_id, annotator_id, winner_run_id) VALUES ($1, $2, $3)",
    )
    .bind(n.pair_id)
    .bind(n.annotator_id)
    .bind(n.winner_run_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Annotations for a round (deterministic order).
///
/// # Errors
/// SQL error.
pub async fn design_annotations_for_round(
    pool: &PgPool,
    round_id: i64,
) -> Result<Vec<DesignAnnotationRow>, DbError> {
    let rows = sqlx::query_as::<_, DesignAnnotationRow>(
        "SELECT a.pair_id, a.annotator_id, a.winner_run_id, \
                (EXTRACT(EPOCH FROM a.created_at)::BIGINT) AS created_at_secs \
         FROM design_annotation a \
         JOIN design_pair p ON p.id = a.pair_id \
         WHERE p.round_id = $1 \
         ORDER BY a.created_at ASC, a.id ASC",
    )
    .bind(round_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Annotation count for a pair.
///
/// # Errors
/// SQL error.
pub async fn design_annotation_count(pool: &PgPool, pair_id: &str) -> Result<i64, DbError> {
    let n: i64 =
        sqlx::query_scalar("SELECT COUNT(*)::bigint FROM design_annotation WHERE pair_id = $1")
            .bind(pair_id)
            .fetch_one(pool)
            .await?;
    Ok(n)
}

/// Upsert rating.
///
/// # Errors
/// SQL error.
#[allow(clippy::too_many_arguments)]
pub async fn upsert_design_rating(
    pool: &PgPool,
    round_id: i64,
    miner_hotkey: &str,
    rating: i32,
    wins: i32,
    losses: i32,
    kind: Option<&str>,
    score: Option<i64>,
    absence_reason: Option<i16>,
) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO design_rating (round_id, miner_hotkey, rating, wins, losses, kind, score, absence_reason) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8) \
         ON CONFLICT (round_id, miner_hotkey) DO UPDATE SET \
           rating = EXCLUDED.rating, wins = EXCLUDED.wins, losses = EXCLUDED.losses, \
           kind = EXCLUDED.kind, score = EXCLUDED.score, absence_reason = EXCLUDED.absence_reason, \
           updated_at = now()",
    )
    .bind(round_id)
    .bind(miner_hotkey)
    .bind(rating)
    .bind(wins)
    .bind(losses)
    .bind(kind)
    .bind(score)
    .bind(absence_reason)
    .execute(pool)
    .await?;
    Ok(())
}

/// Ratings for a round.
///
/// # Errors
/// SQL error.
pub async fn design_ratings_for_round(
    pool: &PgPool,
    round_id: i64,
) -> Result<Vec<DesignRatingRow>, DbError> {
    let q = format!(
        "SELECT {RATING_COLS} FROM design_rating WHERE round_id = $1 ORDER BY rating DESC, miner_hotkey ASC"
    );
    Ok(sqlx::query_as::<_, DesignRatingRow>(&q)
        .bind(round_id)
        .fetch_all(pool)
        .await?)
}

/// Scores for epoch (D24 leaf projection from ratings).
///
/// # Errors
/// SQL error.
pub async fn design_scores_for_epoch(
    pool: &PgPool,
    netuid: i32,
    epoch: i64,
) -> Result<Vec<(String, String, Option<i64>, Option<i16>)>, DbError> {
    let rows: Vec<(String, String, Option<i64>, Option<i16>)> = sqlx::query_as(
        "SELECT DISTINCT ON (r.miner_hotkey) r.miner_hotkey, r.kind, r.score, r.absence_reason \
         FROM design_rating r \
         JOIN design_round dr ON dr.round_id = r.round_id \
         WHERE dr.netuid = $1 AND dr.epoch = $2 AND r.kind IS NOT NULL \
         ORDER BY r.miner_hotkey, r.updated_at DESC",
    )
    .bind(netuid)
    .bind(epoch)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Get / bump daily quota. Returns `runs_used` after bump (0 bump = read).
///
/// # Errors
/// SQL error.
pub async fn design_quota_bump(
    pool: &PgPool,
    miner_hotkey: &str,
    day: &str,
    bump: i32,
) -> Result<i32, DbError> {
    let row: (i32,) = sqlx::query_as(
        "INSERT INTO design_quota (miner_hotkey, day, runs_used) VALUES ($1, $2::date, $3) \
         ON CONFLICT (miner_hotkey, day) DO UPDATE SET runs_used = design_quota.runs_used + $3 \
         RETURNING runs_used",
    )
    .bind(miner_hotkey)
    .bind(day)
    .bind(bump)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// Read quota without bump.
///
/// # Errors
/// SQL error.
pub async fn design_quota_get(
    pool: &PgPool,
    miner_hotkey: &str,
    day: &str,
) -> Result<i32, DbError> {
    let n: Option<i32> = sqlx::query_scalar(
        "SELECT runs_used FROM design_quota WHERE miner_hotkey = $1 AND day = $2::date",
    )
    .bind(miner_hotkey)
    .bind(day)
    .fetch_optional(pool)
    .await?;
    Ok(n.unwrap_or(0))
}

/// Append stage event.
///
/// # Errors
/// SQL error.
pub async fn insert_design_stage_event(
    pool: &PgPool,
    e: &NewDesignStageEvent<'_>,
) -> Result<(), DbError> {
    sqlx::query("INSERT INTO design_stage_event (run_id, stage, detail) VALUES ($1, $2, $3)")
        .bind(e.run_id)
        .bind(e.stage)
        .bind(&e.detail)
        .execute(pool)
        .await?;
    Ok(())
}

/// Ascending journal.
///
/// # Errors
/// SQL error.
pub async fn design_stage_events(
    pool: &PgPool,
    run_id: &str,
) -> Result<Vec<(String, Option<Value>, i64)>, DbError> {
    let rows: Vec<(String, Option<Value>, i64)> = sqlx::query_as(
        "SELECT stage, detail, (EXTRACT(EPOCH FROM created_at)::BIGINT) \
         FROM design_stage_event WHERE run_id = $1 ORDER BY created_at ASC",
    )
    .bind(run_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Insert round award (1–2 winner harness ids). Duplicate PK → error.
///
/// # Errors
/// SQL error.
pub async fn insert_design_round_award(
    pool: &PgPool,
    round_id: i64,
    winner_harness_ids: &Value,
    admin_token_hash: Option<&str>,
) -> Result<(), DbError> {
    sqlx::query(
        "INSERT INTO design_round_award (round_id, winner_harness_ids, admin_token_hash) \
         VALUES ($1, $2, $3)",
    )
    .bind(round_id)
    .bind(winner_harness_ids)
    .bind(admin_token_hash)
    .execute(pool)
    .await?;
    Ok(())
}

/// Load round award if present.
///
/// # Errors
/// SQL error.
pub async fn design_round_award(
    pool: &PgPool,
    round_id: i64,
) -> Result<Option<DesignRoundAwardRow>, DbError> {
    Ok(sqlx::query_as::<_, DesignRoundAwardRow>(
        "SELECT round_id, winner_harness_ids, \
            (EXTRACT(EPOCH FROM awarded_at)::BIGINT) AS awarded_at_secs, \
            admin_token_hash \
         FROM design_round_award WHERE round_id = $1",
    )
    .bind(round_id)
    .fetch_optional(pool)
    .await?)
}
