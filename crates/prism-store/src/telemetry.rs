//! `prism_telemetry` per-step rows (migration 0009).
//!
//! The SQL lives in this crate — not in `db::prism_store` — because the `db`
//! package is at its non-test LOC cap; the table is owned by the PRISM
//! challenge the same way `prism_stage_event` is.

use db::PgPool;
use prism_lium::TelemetryPoint;

use crate::store::StoreError;

/// Replace the whole series for one submission (idempotent re-score/retry).
///
/// # Errors
/// SQL error.
pub async fn replace_telemetry(
    pool: &PgPool,
    submission_id: &str,
    series: &[TelemetryPoint],
) -> Result<(), StoreError> {
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
    sqlx::query("DELETE FROM prism_telemetry WHERE submission_id = $1")
        .bind(submission_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
    if !series.is_empty() {
        let mut qb = sqlx::QueryBuilder::new(
            "INSERT INTO prism_telemetry \
             (submission_id, step, loss, grad_norm, layer_stats, reported_at_secs) ",
        );
        qb.push_values(series, |mut b, pt| {
            b.push_bind(submission_id)
                .push_bind(i64::try_from(pt.step).unwrap_or(i64::MAX))
                .push_bind(pt.loss)
                .push_bind(pt.grad_norm)
                .push_bind(pt.layer_stats.clone())
                .push_bind(pt.at_secs);
        });
        qb.build()
            .execute(&mut *tx)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
    }
    tx.commit()
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
    Ok(())
}

/// Delete the series for one submission (retry reset path).
///
/// # Errors
/// SQL error.
pub async fn delete_telemetry(pool: &PgPool, submission_id: &str) -> Result<(), StoreError> {
    sqlx::query("DELETE FROM prism_telemetry WHERE submission_id = $1")
        .bind(submission_id)
        .execute(pool)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
    Ok(())
}

/// Raw `prism_telemetry` row (`step`, `loss`, `grad_norm`, `layer_stats`, `at_secs`).
type TelemetryRow = (
    i64,
    f64,
    Option<f64>,
    Option<serde_json::Value>,
    Option<f64>,
);

/// Ascending series for one submission (step, then insertion order).
///
/// # Errors
/// SQL error.
pub async fn telemetry_for(
    pool: &PgPool,
    submission_id: &str,
) -> Result<Vec<TelemetryPoint>, StoreError> {
    let rows: Vec<TelemetryRow> = sqlx::query_as(
        "SELECT step, loss, grad_norm, layer_stats, reported_at_secs \
             FROM prism_telemetry WHERE submission_id = $1 \
             ORDER BY step ASC, created_at ASC",
    )
    .bind(submission_id)
    .fetch_all(pool)
    .await
    .map_err(|e| StoreError::Backend(e.to_string()))?;
    Ok(rows
        .into_iter()
        .map(
            |(step, loss, grad_norm, layer_stats, at_secs)| TelemetryPoint {
                step: u64::try_from(step).unwrap_or(0),
                loss,
                grad_norm,
                at_secs,
                layer_stats,
            },
        )
        .collect())
}
