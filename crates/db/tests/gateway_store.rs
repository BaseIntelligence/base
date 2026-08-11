//! Integration tests for the gateway store helpers (require Postgres via `DATABASE_URL`).
//!
//! Scenarios:
//! - S1 happy: raw-weight insert + read back + list + count
//! - S2 edge: identical digest is a conflict; digest change tip-supersedes in place
//! - S3 happy: sealed bundle insert, read by epoch / root, and re-seal bumps `revision`
//! - S4 regression: schema `CHECK`s still reject a malformed raw weight

#![cfg(feature = "testing")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use db::{
    count_raw_weights, get_epoch_bundle, get_epoch_bundle_by_root, get_raw_weight,
    insert_epoch_bundle, insert_raw_weight, latest_bundle_epoch, list_raw_weights_for_epoch,
    test_pool, NewEpochBundle, NewRawWeight,
};
use uuid::Uuid;

/// Returns `false` when `DATABASE_URL` is unset so default CI (no Postgres) skips.
fn database_url_present() -> bool {
    std::env::var_os("DATABASE_URL").is_some()
}

fn digest32() -> Vec<u8> {
    vec![7u8; 32]
}

fn nonce32() -> Vec<u8> {
    vec![8u8; 32]
}

fn sig64() -> Vec<u8> {
    vec![9u8; 64]
}

#[allow(clippy::too_many_arguments)]
fn score_row<'a>(
    id: Uuid,
    challenge_id: &'a str,
    epoch: i64,
    miner: &'a str,
    payload: &'a [u8],
    digest: &'a [u8],
    sig: &'a [u8],
    nonce: &'a [u8],
) -> NewRawWeight<'a> {
    NewRawWeight {
        id,
        challenge_id,
        epoch,
        miner_hotkey: miner,
        kind: "score",
        score: Some(42),
        absence_reason: None,
        payload,
        payload_digest: digest,
        signature: sig,
        nonce,
    }
}

#[tokio::test]
async fn s1_raw_weight_round_trip() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test_pool");
    let pool = tp.pool();
    let (digest, sig, nonce) = (digest32(), sig64(), nonce32());
    let payload = b"scale-body".to_vec();
    let id = Uuid::new_v4();

    let inserted = insert_raw_weight(
        pool,
        &score_row(id, "c1", 7, "aa", &payload, &digest, &sig, &nonce),
    )
    .await
    .expect("insert");
    assert_eq!(inserted, Some(id), "caller-chosen id is the row id");

    let read = get_raw_weight(pool, "c1", 7, "aa")
        .await
        .expect("get")
        .expect("row");
    assert_eq!(read.id, id);
    assert_eq!(read.kind, "score");
    assert_eq!(read.score, Some(42));
    assert_eq!(read.absence_reason, None);
    assert_eq!(read.payload, payload);
    assert_eq!(read.payload_digest, digest);
    assert_eq!(read.signature, sig);

    let absent = NewRawWeight {
        id: Uuid::new_v4(),
        kind: "no_score",
        score: None,
        absence_reason: Some("3"),
        miner_hotkey: "bb",
        ..score_row(
            Uuid::new_v4(),
            "c1",
            7,
            "bb",
            &payload,
            &digest,
            &sig,
            &nonce,
        )
    };
    insert_raw_weight(pool, &absent)
        .await
        .expect("insert absent");

    let rows = list_raw_weights_for_epoch(pool, 7).await.expect("list");
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0].miner_hotkey, "aa",
        "ordered by challenge then miner"
    );
    assert_eq!(rows[1].absence_reason.as_deref(), Some("3"));
    assert!(list_raw_weights_for_epoch(pool, 8)
        .await
        .expect("other epoch")
        .is_empty());
    assert_eq!(count_raw_weights(pool).await.expect("count"), 2);

    tp.drop_schema().await.expect("drop");
}

#[tokio::test]
async fn s2_duplicate_raw_weight_conflicts() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test_pool");
    let pool = tp.pool();
    let (digest, sig, nonce) = (digest32(), sig64(), nonce32());
    let payload = b"scale-body".to_vec();

    let first = insert_raw_weight(
        pool,
        &score_row(
            Uuid::new_v4(),
            "c1",
            1,
            "aa",
            &payload,
            &digest,
            &sig,
            &nonce,
        ),
    )
    .await
    .expect("first");
    assert!(first.is_some());

    let retry = insert_raw_weight(
        pool,
        &score_row(
            Uuid::new_v4(),
            "c1",
            1,
            "aa",
            &payload,
            &digest,
            &sig,
            &nonce,
        ),
    )
    .await
    .expect("retry must not error");
    assert!(retry.is_none(), "identical digest → no second row");
    assert_eq!(count_raw_weights(pool).await.expect("count"), 1);

    // Tip supersede: different digest replaces in place.
    let digest2 = vec![9u8; 32];
    let payload2 = b"scale-body-v2".to_vec();
    let supersede = insert_raw_weight(
        pool,
        &score_row(
            Uuid::new_v4(),
            "c1",
            1,
            "aa",
            &payload2,
            &digest2,
            &sig,
            &nonce,
        ),
    )
    .await
    .expect("supersede");
    assert!(supersede.is_some());
    assert_eq!(count_raw_weights(pool).await.expect("count"), 1);
    let row = get_raw_weight(pool, "c1", 1, "aa")
        .await
        .expect("get")
        .expect("row");
    assert_eq!(row.payload, payload2);
    assert_eq!(row.payload_digest, digest2);

    tp.drop_schema().await.expect("drop");
}

#[tokio::test]
async fn s3_epoch_bundle_revisions() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test_pool");
    let pool = tp.pool();
    let root = vec![3u8; 32];
    let other = vec![4u8; 32];

    let mut bundle = NewEpochBundle {
        epoch: 12,
        protocol_version: 1,
        block_number: 900,
        block_hash: &other,
        metagraph_root: &other,
        merkle_root: &root,
        measurements_digest: &other,
        vector_hash: &other,
        payload: b"first-seal",
        signature: &other,
    };

    let first = insert_epoch_bundle(pool, &bundle).await.expect("seal");
    assert_eq!(first.revision, 1);
    assert!(
        first.sealed_at_micros > 0,
        "computed_at is the seal instant"
    );

    bundle.payload = b"second-seal";
    let second = insert_epoch_bundle(pool, &bundle).await.expect("re-seal");
    assert_eq!(second.revision, 2, "re-seal bumps revision");
    assert!(second.sealed_at_micros >= first.sealed_at_micros);

    let latest = get_epoch_bundle(pool, 12).await.expect("get").expect("row");
    assert_eq!(latest.revision, 2);
    assert_eq!(latest.payload, b"second-seal");

    let by_root = get_epoch_bundle_by_root(pool, &root)
        .await
        .expect("by root")
        .expect("row");
    assert_eq!(by_root.revision, 2);

    assert_eq!(latest_bundle_epoch(pool).await.expect("latest"), Some(12));
    assert!(get_epoch_bundle(pool, 13).await.expect("missing").is_none());

    tp.drop_schema().await.expect("drop");
}

#[tokio::test]
async fn s4_schema_rejects_malformed_raw_weight() {
    if !database_url_present() {
        return;
    }
    let tp = test_pool().await.expect("test_pool");
    let pool = tp.pool();
    let (digest, sig, nonce) = (digest32(), sig64(), nonce32());
    let payload = b"scale-body".to_vec();

    let bad_kind = NewRawWeight {
        kind: "maybe",
        ..score_row(
            Uuid::new_v4(),
            "c1",
            1,
            "aa",
            &payload,
            &digest,
            &sig,
            &nonce,
        )
    };
    assert!(
        insert_raw_weight(pool, &bad_kind).await.is_err(),
        "kind check"
    );

    let bad_shape = NewRawWeight {
        score: None,
        ..score_row(
            Uuid::new_v4(),
            "c1",
            1,
            "aa",
            &payload,
            &digest,
            &sig,
            &nonce,
        )
    };
    assert!(
        insert_raw_weight(pool, &bad_shape).await.is_err(),
        "score/absence shape check"
    );

    let short_nonce = vec![1u8; 16];
    let bad_nonce = NewRawWeight {
        nonce: &short_nonce,
        ..score_row(
            Uuid::new_v4(),
            "c1",
            1,
            "aa",
            &payload,
            &digest,
            &sig,
            &nonce,
        )
    };
    assert!(
        insert_raw_weight(pool, &bad_nonce).await.is_err(),
        "nonce length check"
    );

    assert_eq!(count_raw_weights(pool).await.expect("count"), 0);

    tp.drop_schema().await.expect("drop");
}
