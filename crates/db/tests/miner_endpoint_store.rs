//! Integration tests for the miner endpoint store (require Postgres via `DATABASE_URL`).
//!
//! Scenarios:
//! - S1 happy: an announcement round-trips and the newest epoch per hotkey wins
//! - S2 edge: re-announcing inside one epoch replaces the row instead of erroring
//! - S3 edge: `min_epoch` is a staleness floor, not a filter on the newest row
//! - S4 regression: schema `CHECK`s reject a non-canonical hotkey spelling

#![cfg(feature = "testing")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use db::{miner_endpoints, test_pool, MinerEndpointRow, NewMinerEndpoint};

/// Returns `false` when `DATABASE_URL` is unset so default CI (no Postgres) skips.
fn database_url_present() -> bool {
    std::env::var_os("DATABASE_URL").is_some()
}

const MINER_A: &str = "aa11223344556677889900aabbccddeeff00112233445566778899aabbccddee";
const MINER_B: &str = "bb11223344556677889900aabbccddeeff00112233445566778899aabbccddee";

fn row<'a>(epoch: i64, miner: &'a str, base_url: &'a str, sig: &'a [u8]) -> NewMinerEndpoint<'a> {
    NewMinerEndpoint {
        epoch,
        netuid: 1,
        miner_hotkey: miner,
        base_url,
        signature: sig,
    }
}

fn sig(fill: u8) -> Vec<u8> {
    vec![fill; 64]
}

#[tokio::test]
async fn s1_newest_epoch_per_hotkey_is_returned() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test pool");
    let s = sig(0x01);
    for (epoch, url) in [
        (5i64, "https://a-old.example.com"),
        (7, "https://a.example.com"),
    ] {
        db::upsert_miner_endpoint(tp.pool(), &row(epoch, MINER_A, url, &s))
            .await
            .expect("insert");
    }
    db::upsert_miner_endpoint(tp.pool(), &row(6, MINER_B, "https://b.example.com", &s))
        .await
        .expect("insert");

    let rows = miner_endpoints(tp.pool(), 1, 0).await.expect("read");
    assert_eq!(
        rows,
        vec![
            MinerEndpointRow {
                miner_hotkey: MINER_A.to_owned(),
                base_url: "https://a.example.com".to_owned(),
                epoch: 7,
            },
            MinerEndpointRow {
                miner_hotkey: MINER_B.to_owned(),
                base_url: "https://b.example.com".to_owned(),
                epoch: 6,
            },
        ]
    );

    // Another netuid shares the table but never the result set.
    assert!(miner_endpoints(tp.pool(), 2, 0)
        .await
        .expect("read")
        .is_empty());
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s2_reannouncing_inside_one_epoch_replaces_the_row() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test pool");
    db::upsert_miner_endpoint(
        tp.pool(),
        &row(7, MINER_A, "https://old.example.com", &sig(0x01)),
    )
    .await
    .expect("first");
    db::upsert_miner_endpoint(
        tp.pool(),
        &row(7, MINER_A, "https://new.example.com", &sig(0x02)),
    )
    .await
    .expect("re-announce must not conflict");

    let rows = miner_endpoints(tp.pool(), 1, 0).await.expect("read");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].base_url, "https://new.example.com");
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s3_min_epoch_hides_stale_announcements() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test pool");
    let s = sig(0x03);
    db::upsert_miner_endpoint(tp.pool(), &row(5, MINER_A, "https://a.example.com", &s))
        .await
        .expect("insert");
    db::upsert_miner_endpoint(tp.pool(), &row(9, MINER_B, "https://b.example.com", &s))
        .await
        .expect("insert");

    let rows = miner_endpoints(tp.pool(), 1, 6).await.expect("read");
    assert_eq!(rows.len(), 1, "the epoch-5 miner is below the floor");
    assert_eq!(rows[0].miner_hotkey, MINER_B);
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s4_schema_rejects_a_non_canonical_hotkey_spelling() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test pool");
    let s = sig(0x04);
    // Uppercase and 0x-prefixed hotkeys would silently fail to join
    // `attestation.miner_hotkey`, so the schema refuses them outright.
    for bad in [
        MINER_A.to_uppercase(),
        format!("0x{MINER_A}"),
        MINER_A[..63].to_owned(),
    ] {
        let err = db::upsert_miner_endpoint(tp.pool(), &row(7, &bad, "https://a.example.com", &s))
            .await
            .expect_err("non-canonical hotkey must be rejected");
        assert!(
            err.to_string().contains("miner_endpoint_miner_hotkey_hex"),
            "unexpected error for {bad}: {err}"
        );
    }

    let err = db::upsert_miner_endpoint(
        tp.pool(),
        &row(7, MINER_A, "https://a.example.com/v1/agent", &s),
    )
    .await
    .expect_err("a URL with a path must be rejected");
    assert!(
        err.to_string().contains("miner_endpoint_base_url_scheme"),
        "unexpected error: {err}"
    );
    tp.drop_schema().await.expect("drop schema");
}
