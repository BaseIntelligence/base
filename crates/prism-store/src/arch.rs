//! Postgres queries for the architecture registry + top-model journal
//! (migration 0010) plus submission-row `arch_id` / `created_at` fill-ins.
//!
//! Lives here (not in `db`) because the `db` crate is at its non-test LOC
//! cap; these queries only serve the prism store adapter. Runtime-checked
//! SQL like the rest of the workspace (no `sqlx prepare`).

use std::collections::HashMap;

use sqlx::PgPool;

use prism_store_types::{
    ArchitectureRecord, PublishArchOutcome, StoreError, SubmissionState, TopModelPublication,
};

#[allow(clippy::needless_pass_by_value)] // map_err passes the error by value
fn backend(e: sqlx::Error) -> StoreError {
    StoreError::Backend(e.to_string())
}

/// Batch-fill `arch_id` + `created_at_ms` onto states read through the plain
/// `db::prism_store` row shape (which predates migration 0010).
pub(crate) async fn fill_arch_meta(
    pool: &PgPool,
    rows: &mut [SubmissionState],
) -> Result<(), StoreError> {
    if rows.is_empty() {
        return Ok(());
    }
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let meta: Vec<(String, Option<String>, i64)> = sqlx::query_as(
        "SELECT id, arch_id, (FLOOR(EXTRACT(EPOCH FROM created_at) * 1000))::BIGINT \
         FROM prism_submission WHERE id = ANY($1)",
    )
    .bind(&ids)
    .fetch_all(pool)
    .await
    .map_err(backend)?;
    let map: HashMap<String, (Option<String>, i64)> =
        meta.into_iter().map(|(id, a, ms)| (id, (a, ms))).collect();
    for r in rows.iter_mut() {
        if let Some((arch, ms)) = map.get(&r.id) {
            r.arch_id = arch.clone();
            r.created_at_ms = (*ms).max(0).cast_unsigned();
        }
    }
    Ok(())
}

/// Set-once `arch_id` write (intake for training-only rows, publish
/// back-link for architecture submissions).
pub(crate) async fn set_row_arch_id(
    pool: &PgPool,
    id: &str,
    arch_id: &str,
) -> Result<(), StoreError> {
    sqlx::query("UPDATE prism_submission SET arch_id = $2, updated_at = now() WHERE id = $1")
        .bind(id)
        .bind(arch_id)
        .execute(pool)
        .await
        .map_err(backend)?;
    Ok(())
}

type ArchSqlRow = (String, String, String, String, String, Option<f64>, i64);

const ARCH_COLS: &str =
    "arch_id, owner_hotkey, arch_digest, architecture_py, source_submission, best_bpb, \
     (FLOOR(EXTRACT(EPOCH FROM created_at) * 1000))::BIGINT AS created_ms";

fn arch_from_sql(r: ArchSqlRow) -> ArchitectureRecord {
    ArchitectureRecord {
        arch_id: r.0,
        owner_hotkey: r.1,
        arch_digest: r.2,
        architecture_py: r.3,
        source_submission: r.4,
        best_bpb: r.5,
        created_at_ms: r.6.max(0).cast_unsigned(),
    }
}

/// Insert, idempotent on the digest unique key (simultaneous identical
/// architectures share the first registration).
pub(crate) async fn insert_arch(
    pool: &PgPool,
    rec: &ArchitectureRecord,
) -> Result<PublishArchOutcome, StoreError> {
    let inserted: Option<(String,)> = sqlx::query_as(
        "INSERT INTO prism_architecture \
           (arch_id, owner_hotkey, arch_digest, architecture_py, source_submission, best_bpb) \
         VALUES ($1, $2, $3, $4, $5, $6) \
         ON CONFLICT (arch_digest) DO NOTHING RETURNING arch_id",
    )
    .bind(&rec.arch_id)
    .bind(&rec.owner_hotkey)
    .bind(&rec.arch_digest)
    .bind(&rec.architecture_py)
    .bind(&rec.source_submission)
    .bind(rec.best_bpb)
    .fetch_optional(pool)
    .await
    .map_err(backend)?;
    if inserted.is_some() {
        return Ok(PublishArchOutcome::Published);
    }
    let existing: Option<(String,)> =
        sqlx::query_as("SELECT arch_id FROM prism_architecture WHERE arch_digest = $1")
            .bind(&rec.arch_digest)
            .fetch_optional(pool)
            .await
            .map_err(backend)?;
    Ok(PublishArchOutcome::Duplicate(
        existing.map_or_else(|| rec.arch_id.clone(), |(id,)| id),
    ))
}

pub(crate) async fn get_arch(
    pool: &PgPool,
    arch_id: &str,
) -> Result<Option<ArchitectureRecord>, StoreError> {
    let q = format!("SELECT {ARCH_COLS} FROM prism_architecture WHERE arch_id = $1");
    sqlx::query_as::<_, ArchSqlRow>(&q)
        .bind(arch_id)
        .fetch_optional(pool)
        .await
        .map(|o| o.map(arch_from_sql))
        .map_err(backend)
}

pub(crate) async fn list_archs(
    pool: &PgPool,
    limit: i64,
) -> Result<Vec<ArchitectureRecord>, StoreError> {
    let q = format!("SELECT {ARCH_COLS} FROM prism_architecture ORDER BY created_at DESC LIMIT $1");
    sqlx::query_as::<_, ArchSqlRow>(&q)
        .bind(limit)
        .fetch_all(pool)
        .await
        .map(|v| v.into_iter().map(arch_from_sql).collect())
        .map_err(backend)
}

/// Lower-wins compare-and-set on `best_bpb`.
pub(crate) async fn note_arch_best_bpb(
    pool: &PgPool,
    arch_id: &str,
    bpb: f64,
) -> Result<bool, StoreError> {
    let res = sqlx::query(
        "UPDATE prism_architecture SET best_bpb = $2 \
         WHERE arch_id = $1 AND (best_bpb IS NULL OR best_bpb > $2)",
    )
    .bind(arch_id)
    .bind(bpb)
    .execute(pool)
    .await
    .map_err(backend)?;
    Ok(res.rows_affected() > 0)
}

pub(crate) async fn arch_owners(pool: &PgPool) -> Result<Vec<(String, String)>, StoreError> {
    sqlx::query_as("SELECT arch_id, owner_hotkey FROM prism_architecture")
        .fetch_all(pool)
        .await
        .map_err(backend)
}

pub(crate) async fn record_publication(
    pool: &PgPool,
    p: &TopModelPublication,
) -> Result<(), StoreError> {
    sqlx::query(
        "INSERT INTO prism_topmodel_publication \
           (submission_id, arch_id, owner_hotkey, bpb, repo_path, commit_sha) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(&p.submission_id)
    .bind(&p.arch_id)
    .bind(&p.owner_hotkey)
    .bind(p.bpb)
    .bind(&p.repo_path)
    .bind(&p.commit_sha)
    .execute(pool)
    .await
    .map_err(backend)?;
    Ok(())
}

pub(crate) async fn last_publication_bpb(pool: &PgPool) -> Result<Option<f64>, StoreError> {
    Ok(last_publication(pool).await?.map(|p| p.bpb))
}

pub(crate) async fn last_publication(
    pool: &PgPool,
) -> Result<Option<TopModelPublication>, StoreError> {
    type PubRow = (String, Option<String>, String, f64, String, Option<String>);
    let row: Option<PubRow> = sqlx::query_as(
        "SELECT submission_id, arch_id, owner_hotkey, bpb, repo_path, commit_sha \
         FROM prism_topmodel_publication ORDER BY published_at DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .map_err(backend)?;
    Ok(row.map(
        |(submission_id, arch_id, owner_hotkey, bpb, repo_path, commit_sha)| TopModelPublication {
            submission_id,
            arch_id,
            owner_hotkey,
            bpb,
            repo_path,
            commit_sha,
        },
    ))
}

pub(crate) async fn best_scored_bpb(pool: &PgPool) -> Result<Option<f64>, StoreError> {
    let row: (Option<f64>,) = sqlx::query_as(
        "SELECT MIN(bpb) FROM prism_submission WHERE kind = 'score' AND score > 0 AND bpb IS NOT NULL",
    )
    .fetch_one(pool)
    .await
    .map_err(backend)?;
    Ok(row.0)
}
