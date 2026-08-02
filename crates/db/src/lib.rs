//! Postgres access for base: migrations, pool helpers, and typed row shapes.
//!
//! # Roles
//!
//! Migrations run as the database owner (superuser in tests). Application
//! connections should use the `base_app` role, which has **no** `UPDATE`
//! privilege on the append-only tables `raw_weight_snapshot`, `epoch_bundle`,
//! and `peer_root_statement`.
//!
//! # D18
//!
//! `challenge_backends` stores operational routing only. It must never gain
//! signing-key columns; keys live in owner-signed `config/challenges.toml`.

#![forbid(unsafe_code)]

pub mod prism_store;
mod store;

use std::str::FromStr;
use std::time::Duration;

use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use sqlx::{Postgres, Transaction};
use thiserror::Error;
use uuid::Uuid;

pub use sqlx::PgPool;
pub use store::{
    attestation_for_miner, count_raw_weights, get_epoch_bundle, get_epoch_bundle_by_root,
    get_raw_weight, insert_attestation, insert_epoch_bundle, insert_raw_weight,
    latest_bundle_epoch, list_raw_weights_for_epoch, miner_endpoints, upsert_miner_endpoint,
    AttestationRecord, EpochBundleRecord, MinerEndpointRow, NewAttestation, NewEpochBundle,
    NewMinerEndpoint, NewRawWeight, RawWeightRecord, RECEIPT_PK_LEN,
};

/// Tables that the application role may insert into but never update.
pub const APPEND_ONLY_TABLES: &[&str] =
    &["raw_weight_snapshot", "epoch_bundle", "peer_root_statement"];

/// Application DB role created by migrations (no UPDATE on append-only tables).
pub const APP_ROLE: &str = "base_app";

/// Failures while connecting, migrating, or isolating a test schema.
#[derive(Debug, Error)]
pub enum DbError {
    /// `DATABASE_URL` (or override) was missing or empty.
    #[error("database URL is missing or empty")]
    MissingDatabaseUrl,
    /// Underlying sqlx / Postgres error.
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
    /// Migration runner failed.
    #[error("migrate failed: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    /// A required environment variable was missing.
    #[error("missing env var {0}")]
    MissingEnv(&'static str),
}

/// Connect with the default pool settings (max 10, 5s acquire timeout).
///
/// # Errors
///
/// Returns [`DbError`] when the URL is empty or the pool cannot connect.
pub async fn connect(database_url: &str) -> Result<PgPool, DbError> {
    connect_with(database_url, 10).await
}

/// Connect with an explicit max pool size.
///
/// # Errors
///
/// Returns [`DbError`] when the URL is empty or the pool cannot connect.
pub async fn connect_with(database_url: &str, max_connections: u32) -> Result<PgPool, DbError> {
    let url = database_url.trim();
    if url.is_empty() {
        return Err(DbError::MissingDatabaseUrl);
    }
    let opts = PgConnectOptions::from_str(url)?;
    let pool = PgPoolOptions::new()
        .max_connections(max_connections.max(1))
        .acquire_timeout(Duration::from_secs(5))
        .connect_with(opts)
        .await?;
    Ok(pool)
}

/// Connect using the `DATABASE_URL` environment variable.
///
/// # Errors
///
/// [`DbError::MissingEnv`] when unset; otherwise same as [`connect`].
pub async fn connect_from_env() -> Result<PgPool, DbError> {
    let url = std::env::var("DATABASE_URL").map_err(|_| DbError::MissingEnv("DATABASE_URL"))?;
    connect(&url).await
}

/// Run embedded migrations against `pool` (owner / migration role).
///
/// # Errors
///
/// Propagates sqlx migrate errors.
pub async fn migrate(pool: &PgPool) -> Result<(), DbError> {
    sqlx::migrate!("./migrations").run(pool).await?;
    Ok(())
}

/// Begin a transaction on `pool`.
///
/// # Errors
///
/// Propagates sqlx begin errors.
pub async fn begin(pool: &PgPool) -> Result<Transaction<'static, Postgres>, DbError> {
    Ok(pool.begin().await?)
}

/// Count rows in `miners` (compile-time-checked probe query).
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn count_miners(pool: &PgPool) -> Result<i64, DbError> {
    let n: i64 = sqlx::query_scalar!("SELECT COUNT(*)::bigint AS \"count!\" FROM miners")
        .fetch_one(pool)
        .await?;
    Ok(n)
}

/// Insert a miner by hotkey; returns the new id. Idempotent on unique hotkey
/// via `ON CONFLICT DO NOTHING` returning `None` when the hotkey already exists.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn insert_miner_if_absent(pool: &PgPool, hotkey: &str) -> Result<Option<Uuid>, DbError> {
    let id: Option<Uuid> = sqlx::query_scalar!(
        r#"
        INSERT INTO miners (hotkey)
        VALUES ($1)
        ON CONFLICT (hotkey) DO NOTHING
        RETURNING id
        "#,
        hotkey
    )
    .fetch_optional(pool)
    .await?;
    Ok(id)
}

/// List challenge backend base URLs for a challenge (routing only — D18).
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn list_backend_urls(pool: &PgPool, challenge_id: &str) -> Result<Vec<String>, DbError> {
    let rows = sqlx::query_scalar!(
        r#"
        SELECT base_url
        FROM challenge_backends
        WHERE challenge_id = $1 AND healthy = TRUE
        ORDER BY base_url
        "#,
        challenge_id
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Assert that `challenge_backends` has no signing-key-like columns (D18 guard).
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn challenge_backends_has_no_key_columns(pool: &PgPool) -> Result<bool, DbError> {
    let forbidden: Vec<String> = sqlx::query_scalar!(
        r#"
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'challenge_backends'
          AND (
                column_name ILIKE '%signing%key%'
             OR column_name ILIKE '%private%key%'
             OR column_name ILIKE '%secret%key%'
             OR column_name ILIKE 'secret'
             OR column_name ILIKE 'private_key'
             OR column_name ILIKE 'signing_key'
          )
        "#
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .flatten()
    .collect();
    Ok(forbidden.is_empty())
}

/// Whether the current DB user may `UPDATE` `table` (via `has_table_privilege`).
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn current_user_can_update(pool: &PgPool, table: &str) -> Result<bool, DbError> {
    let allowed: bool = sqlx::query_scalar!(
        r#"SELECT has_table_privilege(current_user, $1::text, 'UPDATE') AS "allowed!""#,
        table
    )
    .fetch_one(pool)
    .await?;
    Ok(allowed)
}

/// Current Postgres role name for this connection.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn current_user(pool: &PgPool) -> Result<String, DbError> {
    let name: String = sqlx::query_scalar!(r#"SELECT current_user::text AS "user!""#)
        .fetch_one(pool)
        .await?;
    Ok(name)
}

/// Build an application-role URL from an owner `DATABASE_URL`.
///
/// Replaces the username/password with `base_app` / `base_app` (migration default).
///
/// # Errors
///
/// [`DbError::MissingDatabaseUrl`] or parse failures.
pub fn app_role_database_url(owner_url: &str) -> Result<String, DbError> {
    let url = owner_url.trim();
    if url.is_empty() {
        return Err(DbError::MissingDatabaseUrl);
    }
    let opts = PgConnectOptions::from_str(url)?;
    let host = opts.get_host();
    let port = opts.get_port();
    let db = opts.get_database().unwrap_or("postgres");
    Ok(format!(
        "postgres://{APP_ROLE}:{APP_ROLE}@{host}:{port}/{db}"
    ))
}

/// Per-test isolated schema + migrated pool (owner role).
///
/// Behind `feature = "testing"`. Creates `base_test_<uuid>`, sets
/// `search_path`, runs migrations, and returns a pool whose connections use
/// that schema. Drop the schema when the test finishes via [`TestPool::drop_schema`].
///
/// Requires `DATABASE_URL` pointing at a Postgres 16+ instance where the user
/// can `CREATE SCHEMA` and run migrations.
///
/// # Errors
///
/// [`DbError::MissingEnv`] when `DATABASE_URL` is unset; connection or migrate failures.
#[cfg(feature = "testing")]
pub async fn test_pool() -> Result<TestPool, DbError> {
    let owner_url =
        std::env::var("DATABASE_URL").map_err(|_| DbError::MissingEnv("DATABASE_URL"))?;
    test_pool_with_url(&owner_url).await
}

/// Same as [`test_pool`] but with an explicit owner URL.
///
/// # Errors
///
/// Connection, schema creation, grant, or migrate failures.
#[cfg(feature = "testing")]
pub async fn test_pool_with_url(owner_url: &str) -> Result<TestPool, DbError> {
    let schema = format!("base_test_{}", Uuid::new_v4().simple());
    let base = connect_with(owner_url, 5).await?;

    let create = format!("CREATE SCHEMA {schema}");
    sqlx::query(&create).execute(&base).await?;

    let opts = PgConnectOptions::from_str(owner_url)?.options([("search_path", schema.as_str())]);
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(5))
        .after_connect({
            let schema = schema.clone();
            move |conn, _meta| {
                let schema = schema.clone();
                Box::pin(async move {
                    let q = format!("SET search_path TO {schema}, public");
                    sqlx::query(&q).execute(&mut *conn).await?;
                    Ok(())
                })
            }
        })
        .connect_with(opts)
        .await?;

    sqlx::query("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        .execute(&pool)
        .await?;

    migrate(&pool).await?;

    let grant_schema = format!("GRANT USAGE ON SCHEMA {schema} TO {APP_ROLE}");
    sqlx::query(&grant_schema).execute(&pool).await?;
    let grant_all = format!(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {APP_ROLE}"
    );
    sqlx::query(&grant_all).execute(&pool).await?;
    for table in APPEND_ONLY_TABLES {
        let revoke = format!("REVOKE UPDATE, DELETE ON TABLE {schema}.{table} FROM {APP_ROLE}");
        sqlx::query(&revoke).execute(&pool).await?;
        let grant_ai = format!("GRANT SELECT, INSERT ON TABLE {schema}.{table} TO {APP_ROLE}");
        sqlx::query(&grant_ai).execute(&pool).await?;
    }

    Ok(TestPool {
        pool,
        owner_url: owner_url.to_owned(),
        schema,
    })
}

/// Owner-role pool bound to an isolated test schema.
#[cfg(feature = "testing")]
pub struct TestPool {
    pool: PgPool,
    owner_url: String,
    schema: String,
}

#[cfg(feature = "testing")]
impl TestPool {
    /// Borrow the owner pool (`search_path` = test schema).
    #[must_use]
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Schema name (`base_test_<uuid>`).
    #[must_use]
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Connect as `base_app` with `search_path` set to this test schema.
    ///
    /// # Errors
    ///
    /// Connection failures.
    pub async fn app_pool(&self) -> Result<PgPool, DbError> {
        let app_url = app_role_database_url(&self.owner_url)?;
        let opts = PgConnectOptions::from_str(&app_url)?
            .username(APP_ROLE)
            .password(APP_ROLE);
        let schema = self.schema.clone();
        let pool = PgPoolOptions::new()
            .max_connections(2)
            .acquire_timeout(Duration::from_secs(5))
            .after_connect(move |conn, _meta| {
                let schema = schema.clone();
                Box::pin(async move {
                    let q = format!("SET search_path TO {schema}, public");
                    sqlx::query(&q).execute(&mut *conn).await?;
                    Ok(())
                })
            })
            .connect_with(opts)
            .await?;
        Ok(pool)
    }

    /// Drop the isolated schema (CASCADE).
    ///
    /// # Errors
    ///
    /// Propagates drop failures.
    pub async fn drop_schema(self) -> Result<(), DbError> {
        let drop = format!("DROP SCHEMA IF EXISTS {} CASCADE", self.schema);
        sqlx::query(&drop).execute(&self.pool).await?;
        self.pool.close().await;
        Ok(())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn append_only_table_list_is_exact() {
        assert_eq!(
            APPEND_ONLY_TABLES,
            &["raw_weight_snapshot", "epoch_bundle", "peer_root_statement"]
        );
    }

    #[test]
    fn app_role_url_rewrites_user() {
        let url = app_role_database_url("postgres://postgres:postgres@127.0.0.1:15433/base")
            .expect("url");
        assert!(url.contains("base_app:base_app@"), "url={url}");
        assert!(url.contains("127.0.0.1:15433"));
        assert!(url.ends_with("/base"));
    }

    #[test]
    fn empty_url_rejected() {
        let err = app_role_database_url("  ").expect_err("empty");
        assert!(matches!(err, DbError::MissingDatabaseUrl));
    }
}
