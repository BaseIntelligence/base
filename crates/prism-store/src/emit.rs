//! Postgres queries for the epoch-close leaf emission outbox (migration 0012).
//!
//! Lives here (not in `db`) because the `db` crate is at its non-test LOC
//! cap; these queries only serve the prism store adapter. Runtime-checked
//! SQL like the rest of the workspace (no `sqlx prepare`).
//!
//! Outbox contract (see `docs/PRISM.md` § Leaf emission):
//! - [`assign_emit_batch`] stamps every scored-not-yet-emitted row with the
//!   target leaf epoch and returns the batch (sticky assignment);
//! - [`active_score_rows`] returns every positive lattice score for the netuid
//!   so the emitter can re-include the live champion set each epoch (carry);
//! - the emitter submits one D24-complete set for that epoch, then advances
//!   [`set_emit_cursor`];
//! - [`pending_emit_epochs`] finds assigned-but-incomplete epochs so a crash
//!   mid-submit replays the identical set before any new work.

use sqlx::PgPool;

use prism_store_types::{EpochScoreRow, FinalScore, StoreError};

#[allow(clippy::needless_pass_by_value)] // map_err passes the error by value
fn backend(e: sqlx::Error) -> StoreError {
    StoreError::Backend(e.to_string())
}

type EmitSqlRow = (String, Option<String>, String, Option<i64>, Option<i16>);

fn rows_to_epoch(rows: Vec<EmitSqlRow>) -> Vec<EpochScoreRow> {
    rows.into_iter()
        .filter_map(|(hk, arch_id, kind, score, absence)| {
            let final_score = match kind.as_str() {
                "score" => score.map(|s| FinalScore::Score(s.cast_unsigned())),
                "no_score" => absence.map(|a| FinalScore::NoScore(u8::try_from(a).unwrap_or(0))),
                _ => None,
            }?;
            Some(EpochScoreRow {
                miner_hotkey: hk,
                arch_id,
                final_score,
            })
        })
        .collect()
}

/// Assign every scored-not-yet-emitted row of `netuid` to leaf `epoch` and
/// return the batch. A concurrent assigner re-evaluates the predicate after
/// waiting on row locks (READ COMMITTED), so each row is assigned exactly once.
pub(crate) async fn assign_emit_batch(
    pool: &PgPool,
    netuid: i32,
    epoch: i64,
) -> Result<Vec<EpochScoreRow>, StoreError> {
    let rows: Vec<EmitSqlRow> = sqlx::query_as(
        "UPDATE prism_submission SET emitted_epoch = $2, updated_at = now() \
         WHERE netuid = $1 AND kind IS NOT NULL AND emitted_epoch IS NULL \
         RETURNING miner_hotkey, arch_id, kind, score, absence_reason",
    )
    .bind(netuid)
    .bind(epoch)
    .fetch_all(pool)
    .await
    .map_err(backend)?;
    Ok(rows_to_epoch(rows))
}

/// Sticky re-read of the batch assigned to `epoch` (recovery resubmit).
pub(crate) async fn emit_batch(
    pool: &PgPool,
    netuid: i32,
    epoch: i64,
) -> Result<Vec<EpochScoreRow>, StoreError> {
    let rows: Vec<EmitSqlRow> = sqlx::query_as(
        "SELECT miner_hotkey, arch_id, kind, score, absence_reason \
         FROM prism_submission \
         WHERE netuid = $1 AND emitted_epoch = $2 AND kind IS NOT NULL",
    )
    .bind(netuid)
    .bind(epoch)
    .fetch_all(pool)
    .await
    .map_err(backend)?;
    Ok(rows_to_epoch(rows))
}

/// Positive lattice scores still eligible for epoch-close competition carry.
///
/// `Score(0)` rejects and `NoScore` absences are excluded — they must not
/// displace a prior valid winner when an epoch's fresh outbox is empty or
/// burn-only. Competition aggregation takes `max` over the union of the
/// fresh batch and this set, so a better later score supersedes naturally.
pub(crate) async fn active_score_rows(
    pool: &PgPool,
    netuid: i32,
) -> Result<Vec<EpochScoreRow>, StoreError> {
    let rows: Vec<EmitSqlRow> = sqlx::query_as(
        "SELECT miner_hotkey, arch_id, kind, score, absence_reason \
         FROM prism_submission \
         WHERE netuid = $1 AND kind = 'score' AND score > 0",
    )
    .bind(netuid)
    .fetch_all(pool)
    .await
    .map_err(backend)?;
    Ok(rows_to_epoch(rows))
}

/// Highest leaf epoch whose set fully landed for `netuid` (`None` = never).
pub(crate) async fn emit_cursor(pool: &PgPool, netuid: i32) -> Result<Option<u64>, StoreError> {
    let row: Option<(i64,)> =
        sqlx::query_as("SELECT epoch FROM prism_emit_cursor WHERE netuid = $1")
            .bind(netuid)
            .fetch_optional(pool)
            .await
            .map_err(backend)?;
    Ok(row.map(|(e,)| e.max(0).cast_unsigned()))
}

/// Monotonic cursor advance after a fully-submitted set.
pub(crate) async fn set_emit_cursor(
    pool: &PgPool,
    netuid: i32,
    epoch: i64,
) -> Result<(), StoreError> {
    sqlx::query(
        "INSERT INTO prism_emit_cursor (netuid, epoch) VALUES ($1, $2) \
         ON CONFLICT (netuid) DO UPDATE \
         SET epoch = GREATEST(prism_emit_cursor.epoch, EXCLUDED.epoch), updated_at = now()",
    )
    .bind(netuid)
    .bind(epoch)
    .execute(pool)
    .await
    .map_err(backend)?;
    Ok(())
}

/// Assigned-but-not-completed leaf epochs (above the cursor), ascending.
pub(crate) async fn pending_emit_epochs(
    pool: &PgPool,
    netuid: i32,
) -> Result<Vec<u64>, StoreError> {
    let rows: Vec<(i64,)> = sqlx::query_as(
        "SELECT DISTINCT emitted_epoch FROM prism_submission \
         WHERE netuid = $1 AND emitted_epoch IS NOT NULL \
           AND emitted_epoch > COALESCE(\
               (SELECT epoch FROM prism_emit_cursor WHERE netuid = $1), -1) \
         ORDER BY emitted_epoch",
    )
    .bind(netuid)
    .fetch_all(pool)
    .await
    .map_err(backend)?;
    Ok(rows
        .into_iter()
        .map(|(e,)| e.max(0).cast_unsigned())
        .collect())
}
