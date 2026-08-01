//! Integration tests for the attestation store (require Postgres via `DATABASE_URL`).
//!
//! Scenarios:
//! - S1 happy: verified insert round-trips the receipt key
//! - S2 edge: a later verified attempt overtakes an earlier park
//! - S3 edge: a verified row from one validator wins over another's reject
//! - S4 regression: schema `CHECK`s reject a receipt key on a non-verified row

#![cfg(feature = "testing")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use db::{
    attestation_for_miner, insert_attestation, test_pool, AttestationRecord, NewAttestation,
    RECEIPT_PK_LEN,
};

/// Returns `false` when `DATABASE_URL` is unset so default CI (no Postgres) skips.
fn database_url_present() -> bool {
    std::env::var_os("DATABASE_URL").is_some()
}

fn nonce32() -> Vec<u8> {
    vec![3u8; 32]
}

fn receipt_pk(fill: u8) -> Vec<u8> {
    vec![fill; RECEIPT_PK_LEN]
}

fn row<'a>(
    epoch: i64,
    miner: &'a str,
    validator: &'a str,
    outcome: &'a str,
    nonce: &'a [u8],
    receipt: Option<&'a [u8]>,
) -> NewAttestation<'a> {
    NewAttestation {
        epoch,
        miner_hotkey: miner,
        validator_hotkey: validator,
        nonce,
        outcome,
        quote: None,
        reason: if outcome == "verified" {
            None
        } else {
            Some("pcs_timeout")
        },
        receipt_pk: receipt,
    }
}

#[tokio::test]
async fn s1_verified_attestation_round_trips_the_receipt_key() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test pool");
    let nonce = nonce32();
    let pk = receipt_pk(0x5a);
    let attempt = insert_attestation(
        tp.pool(),
        &row(7, "miner-a", "val-1", "verified", &nonce, Some(&pk)),
    )
    .await
    .expect("insert");
    assert_eq!(attempt, 1);

    let got = attestation_for_miner(tp.pool(), 7, "miner-a")
        .await
        .expect("read");
    assert_eq!(
        got,
        Some(AttestationRecord {
            outcome: "verified".to_owned(),
            receipt_pk: Some([0x5a; RECEIPT_PK_LEN]),
        })
    );
    assert_eq!(
        attestation_for_miner(tp.pool(), 7, "miner-unknown")
            .await
            .expect("read"),
        None
    );
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s2_a_later_verified_attempt_overtakes_a_park() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test pool");
    let nonce = nonce32();
    let pk = receipt_pk(0x11);
    insert_attestation(tp.pool(), &row(9, "miner-b", "val-1", "park", &nonce, None))
        .await
        .expect("park insert");
    let attempt = insert_attestation(
        tp.pool(),
        &row(9, "miner-b", "val-1", "verified", &nonce, Some(&pk)),
    )
    .await
    .expect("verified insert");
    assert_eq!(attempt, 2, "retry lands as a new attempt, never an UPDATE");

    let got = attestation_for_miner(tp.pool(), 9, "miner-b")
        .await
        .expect("read")
        .expect("row");
    assert_eq!(got.outcome, "verified");
    assert_eq!(got.receipt_pk, Some([0x11; RECEIPT_PK_LEN]));
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s3_verified_wins_over_another_validators_reject() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test pool");
    let nonce = nonce32();
    let pk = receipt_pk(0x22);
    insert_attestation(
        tp.pool(),
        &row(11, "miner-c", "val-1", "verified", &nonce, Some(&pk)),
    )
    .await
    .expect("verified insert");
    insert_attestation(
        tp.pool(),
        &row(11, "miner-c", "val-2", "reject", &nonce, None),
    )
    .await
    .expect("reject insert");

    let got = attestation_for_miner(tp.pool(), 11, "miner-c")
        .await
        .expect("read")
        .expect("row");
    assert_eq!(got.outcome, "verified");
    assert_eq!(got.receipt_pk, Some([0x22; RECEIPT_PK_LEN]));

    // A miner with only non-verified rows exposes no key.
    insert_attestation(
        tp.pool(),
        &row(11, "miner-d", "val-1", "park", &nonce, None),
    )
    .await
    .expect("park insert");
    let parked = attestation_for_miner(tp.pool(), 11, "miner-d")
        .await
        .expect("read")
        .expect("row");
    assert_eq!(parked.outcome, "park");
    assert_eq!(parked.receipt_pk, None);
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s4_schema_rejects_a_receipt_key_on_a_non_verified_row() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test pool");
    let nonce = nonce32();
    let pk = receipt_pk(0x33);
    let err = insert_attestation(
        tp.pool(),
        &row(13, "miner-e", "val-1", "park", &nonce, Some(&pk)),
    )
    .await
    .expect_err("park must not carry a receipt key");
    assert!(
        err.to_string()
            .contains("attestation_receipt_pk_verified_only"),
        "unexpected error: {err}"
    );

    let short = vec![0x44u8; 8];
    let err = insert_attestation(
        tp.pool(),
        &row(13, "miner-f", "val-1", "verified", &nonce, Some(&short)),
    )
    .await
    .expect_err("short key must fail the length check");
    assert!(
        err.to_string().contains("attestation_receipt_pk_len"),
        "unexpected error: {err}"
    );
    tp.drop_schema().await.expect("drop schema");
}
