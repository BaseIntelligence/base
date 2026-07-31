//! Integration tests for db (require Postgres via `DATABASE_URL`).
//!
//! Scenarios:
//! - S1 happy: migrate + insert miner + count
//! - S2 edge: app role cannot UPDATE append-only tables
//! - S3 regression: `challenge_backends` has no signing-key columns (D18)

#![cfg(feature = "testing")]
#![allow(clippy::too_many_lines, clippy::expect_used, clippy::unwrap_used)]

use db::{
    challenge_backends_has_no_key_columns, count_miners, current_user, current_user_can_update,
    insert_miner_if_absent, list_backend_urls, test_pool, APPEND_ONLY_TABLES, APP_ROLE,
};

/// Returns `false` when `DATABASE_URL` is unset so default CI (no Postgres) skips.
fn database_url_present() -> bool {
    std::env::var_os("DATABASE_URL").is_some()
}

fn digest32() -> Vec<u8> {
    vec![0u8; 32]
}

fn nonce32() -> Vec<u8> {
    vec![1u8; 32]
}

fn sig64() -> Vec<u8> {
    vec![2u8; 64]
}

#[tokio::test]
async fn s1_migrate_insert_miner_and_count() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test_pool");
    let pool = tp.pool();

    assert_eq!(count_miners(pool).await.expect("count"), 0);

    let id = insert_miner_if_absent(pool, "5GminerHotkeyTestAAAA")
        .await
        .expect("insert")
        .expect("new id");
    assert!(!id.is_nil());

    let again = insert_miner_if_absent(pool, "5GminerHotkeyTestAAAA")
        .await
        .expect("conflict");
    assert!(again.is_none(), "hotkey unique → DO NOTHING");

    assert_eq!(count_miners(pool).await.expect("count"), 1);

    sqlx::query(
        r"
        INSERT INTO challenge_backends (challenge_id, base_url)
        VALUES ($1, $2)
        ",
    )
    .bind("agent-challenge")
    .bind("http://127.0.0.1:8080")
    .execute(pool)
    .await
    .expect("backend insert");

    let urls = list_backend_urls(pool, "agent-challenge")
        .await
        .expect("list");
    assert_eq!(urls, vec!["http://127.0.0.1:8080".to_owned()]);

    tp.drop_schema().await.expect("drop");
}

async fn seed_append_only(owner: &sqlx::PgPool) {
    let digest = digest32();
    let nonce = nonce32();
    let sig = sig64();
    let payload = b"payload".to_vec();

    sqlx::query(
        r"
        INSERT INTO raw_weight_snapshot (
            challenge_id, epoch, miner_hotkey, kind, score,
            payload, payload_digest, signature, nonce
        ) VALUES (
            'c1', 1, '5Gminer', 'score', 42,
            $1, $2, $3, $4
        )
        ",
    )
    .bind(&payload)
    .bind(&digest)
    .bind(&sig)
    .bind(&nonce)
    .execute(owner)
    .await
    .expect("seed raw_weight_snapshot");

    sqlx::query(
        r"
        INSERT INTO epoch_bundle (
            epoch, protocol_version, block_number,
            block_hash, metagraph_root, merkle_root,
            measurements_digest, vector_hash, payload, signature
        ) VALUES (
            1, 1, 100,
            $1, $1, $1,
            $1, $1, $2, $3
        )
        ",
    )
    .bind(&digest)
    .bind(&payload)
    .bind(&sig)
    .execute(owner)
    .await
    .expect("seed epoch_bundle");

    sqlx::query(
        r"
        INSERT INTO peer_root_statement (
            epoch, peer_hotkey, merkle_root, payload, signature
        ) VALUES (
            1, '5Gpeer', $1, $2, $3
        )
        ",
    )
    .bind(&digest)
    .bind(&payload)
    .bind(&sig)
    .execute(owner)
    .await
    .expect("seed peer_root_statement");
}

#[tokio::test]
async fn s2_app_role_cannot_update_append_only_tables() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test_pool");
    seed_append_only(tp.pool()).await;

    let app = tp.app_pool().await.expect("app_pool");
    assert_eq!(current_user(&app).await.expect("user"), APP_ROLE);

    for table in APPEND_ONLY_TABLES {
        let can = current_user_can_update(&app, table)
            .await
            .unwrap_or_else(|e| panic!("privilege check {table}: {e}"));
        assert!(
            !can,
            "base_app must NOT have UPDATE on append-only table {table}"
        );
    }

    for (table, sql) in [
        (
            "raw_weight_snapshot",
            "UPDATE raw_weight_snapshot SET kind = 'score' WHERE epoch = 1",
        ),
        (
            "epoch_bundle",
            "UPDATE epoch_bundle SET protocol_version = 2 WHERE epoch = 1",
        ),
        (
            "peer_root_statement",
            "UPDATE peer_root_statement SET peer_hotkey = 'x' WHERE epoch = 1",
        ),
    ] {
        let err = sqlx::query(sql)
            .execute(&app)
            .await
            .expect_err("UPDATE must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("permission denied")
                || msg.contains("42501")
                || msg.to_lowercase().contains("insufficient"),
            "table={table} err={msg}"
        );
    }

    let digest = digest32();
    let nonce = nonce32();
    let sig = sig64();
    let payload = b"payload".to_vec();
    sqlx::query(
        r"
        INSERT INTO raw_weight_snapshot (
            challenge_id, epoch, miner_hotkey, kind, score,
            payload, payload_digest, signature, nonce
        ) VALUES (
            'c1', 2, '5Gminer2', 'score', 7,
            $1, $2, $3, $4
        )
        ",
    )
    .bind(&payload)
    .bind(&digest)
    .bind(&sig)
    .bind(&nonce)
    .execute(&app)
    .await
    .expect("app INSERT on append-only must succeed");

    sqlx::query(r"INSERT INTO miners (hotkey) VALUES ('5Gmutable')")
        .execute(&app)
        .await
        .expect("app insert miner");
    sqlx::query(r"UPDATE miners SET status = 'inactive' WHERE hotkey = '5Gmutable'")
        .execute(&app)
        .await
        .expect("app update miner");

    tp.drop_schema().await.expect("drop");
}

#[tokio::test]
async fn s3_challenge_backends_has_no_signing_key_columns() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test_pool");
    let ok = challenge_backends_has_no_key_columns(tp.pool())
        .await
        .expect("cols");
    assert!(ok, "D18: challenge_backends must not store signing keys");

    let cols: Vec<String> = sqlx::query_scalar(
        r"
        SELECT column_name::text
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'challenge_backends'
        ORDER BY ordinal_position
        ",
    )
    .fetch_all(tp.pool())
    .await
    .expect("list cols");

    for forbidden in [
        "signing_key",
        "private_key",
        "secret",
        "secret_key",
        "seed",
        "mnemonic",
    ] {
        assert!(
            !cols.iter().any(|c| c == forbidden || c.contains(forbidden)),
            "forbidden column {forbidden} in {cols:?}"
        );
    }

    for required in [
        "id",
        "challenge_id",
        "base_url",
        "weight",
        "healthy",
        "created_at",
    ] {
        assert!(
            cols.iter().any(|c| c == required),
            "missing required column {required} in {cols:?}"
        );
    }

    tp.drop_schema().await.expect("drop");
}

#[tokio::test]
async fn s1_all_tables_have_created_at() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test_pool");
    let tables = [
        "miners",
        "challenge_backends",
        "raw_weight_snapshot",
        "epoch_bundle",
        "peer_root_statement",
        "attestation",
        "attestation_nonce",
        "dissent",
        "promotion",
    ];
    for table in tables {
        let n: i64 = sqlx::query_scalar(
            r"
            SELECT COUNT(*)::bigint
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = $1
              AND column_name = 'created_at'
              AND data_type = 'timestamp with time zone'
              AND is_nullable = 'NO'
            ",
        )
        .bind(table)
        .fetch_one(tp.pool())
        .await
        .expect("created_at");
        assert_eq!(
            n, 1,
            "table {table} must have created_at timestamptz not null"
        );
    }
    tp.drop_schema().await.expect("drop");
}
