//! Integration tests for the Postgres gateway stores (require `DATABASE_URL`).
//!
//! Scenarios:
//! - S1 happy: raw-weight insert → read back → list → count
//! - S2 edge: resubmitting the same key conflicts and echoes the stored original
//! - S3 happy: seal → read by epoch / root / latest, with the real seal instant
//! - S4 regression: a fresh store over the same pool still serves a sealed
//!   bundle (what a gateway restart looks like), and a re-seal bumps `revision`

#![allow(clippy::expect_used, clippy::unwrap_used)]

use bundle::{EpochBundleBodyV1, EpochBundleV1, ALGORITHM_VERSION, PROTOCOL_VERSION};
use db::{test_pool, NewEpochBundle, TestPool};
use gateway::RawWeightRow;
use uuid::Uuid;

/// Returns `false` when `DATABASE_URL` is unset so default CI (no Postgres) skips.
fn database_url_present() -> bool {
    std::env::var_os("DATABASE_URL").is_some()
}

async fn pool() -> Option<TestPool> {
    if !database_url_present() {
        return None;
    }
    Some(test_pool().await.expect("test_pool"))
}

/// Owner URL pinned to the test's isolated schema, so the stores open their own
/// pool exactly the way production does.
fn schema_url(tp: &TestPool) -> String {
    let base = std::env::var("DATABASE_URL").expect("DATABASE_URL");
    format!("{base}?options=-c%20search_path%3D{}", tp.schema())
}

fn row(challenge_id: &str, epoch: u64, miner: &str) -> RawWeightRow {
    RawWeightRow {
        id: Uuid::new_v4(),
        challenge_id: challenge_id.to_owned(),
        epoch,
        miner_hotkey: miner.to_owned(),
        kind: "score".to_owned(),
        score: Some(11),
        absence_reason: None,
        payload: b"scale-body".to_vec(),
        payload_digest: [5u8; 32],
        challenge_sig: vec![6u8; 64],
    }
}

fn sealed_bundle(epoch: u64, merkle_root: [u8; 32]) -> Vec<u8> {
    let body = EpochBundleBodyV1 {
        protocol_version: PROTOCOL_VERSION,
        epoch,
        netuid: 1,
        block_b: 900,
        block_hash: [1u8; 32],
        metagraph_root: [2u8; 32],
        algorithm_version: ALGORITHM_VERSION,
        emission_shares: vec![(b"c1".to_vec(), 10_000)],
        measurements_digest: [3u8; 32],
        uid_map: vec![([9u8; 32], 1)],
        leaves: Vec::new(),
        merkle_root,
        final_vector: vec![(0, 100), (1, 65_435)],
        gateway_hotkey: [4u8; 32],
    };
    EpochBundleV1 {
        body,
        gateway_sig: [7u8; 64],
    }
    .encode_bytes()
}

#[tokio::test]
async fn s1_raw_weight_round_trip() {
    let Some(tp) = pool().await else { return };
    let (weights, _bundles) = gateway_store_pg::stores(&schema_url(&tp)).expect("stores");

    assert!(weights.is_empty());
    let stored = weights.insert(row("c1", 4, "aa")).expect("insert");
    let read = weights.get("c1", 4, "aa").expect("read back");
    assert_eq!(read, stored, "read back is byte-identical to the ack");
    assert_eq!(read.score, Some(11));
    assert_eq!(read.payload_digest, [5u8; 32]);

    weights.insert(row("c1", 4, "bb")).expect("second miner");
    weights.insert(row("c1", 5, "aa")).expect("other epoch");
    assert_eq!(weights.len(), 3);
    assert_eq!(weights.list_for_epoch(4).len(), 2);
    assert!(weights.get("c1", 9, "aa").is_none());

    tp.drop_schema().await.expect("drop");
}

#[tokio::test]
async fn s2_duplicate_submission_conflicts() {
    let Some(tp) = pool().await else { return };
    let (weights, _bundles) = gateway_store_pg::stores(&schema_url(&tp)).expect("stores");

    let first = weights.insert(row("c1", 4, "aa")).expect("insert");
    let retry = row("c1", 4, "aa");
    let err = weights.insert(retry).expect_err("duplicate must conflict");
    match err {
        gateway::StoreError::Conflict { original } => {
            assert_eq!(original.id, first.id, "409 echoes the stored original");
        }
        other @ gateway::StoreError::Backend(_) => panic!("expected conflict, got {other:?}"),
    }
    assert_eq!(weights.len(), 1, "conflict must not append a second row");

    tp.drop_schema().await.expect("drop");
}

#[tokio::test]
async fn s3_seal_and_serve() {
    let Some(tp) = pool().await else { return };
    let (_weights, bundles) = gateway_store_pg::stores(&schema_url(&tp)).expect("stores");

    let root = [8u8; 32];
    let bytes = sealed_bundle(12, root);
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros();

    let stored = bundles.put_if_absent(12, bytes.clone());
    assert_eq!(stored, bytes);
    assert_eq!(bundles.get_by_epoch(12), Some(bytes.clone()));
    assert_eq!(bundles.get_by_root(&root), Some(bytes.clone()));
    assert_eq!(bundles.latest_epoch(), Some(12));

    let seal = bundles.seal_record(12).expect("seal record");
    assert_eq!(seal.revision, 1, "first seal is revision 1");
    assert!(
        u128::from(seal.sealed_at_micros) >= before,
        "computed_at is the real seal instant"
    );

    // Idempotent: a second seal of the same epoch returns the stored bytes.
    let again = bundles.put_if_absent(12, sealed_bundle(12, [9u8; 32]));
    assert_eq!(again, bytes);
    assert_eq!(bundles.seal_record(12).expect("record").revision, 1);
    assert!(bundles.get_by_root(&[9u8; 32]).is_none());

    tp.drop_schema().await.expect("drop");
}

#[tokio::test]
async fn s4_survives_restart_and_reseal_bumps_revision() {
    let Some(tp) = pool().await else { return };
    let bytes = sealed_bundle(20, [1u8; 32]);
    {
        let (weights, bundles) = gateway_store_pg::stores(&schema_url(&tp)).expect("stores");
        weights.insert(row("c1", 20, "aa")).expect("insert");
        bundles.put_if_absent(20, bytes.clone());
    }

    // A new store over the same database is what a gateway restart looks like.
    let (weights, bundles) = gateway_store_pg::stores(&schema_url(&tp)).expect("stores");
    assert_eq!(bundles.get_by_epoch(20), Some(bytes.clone()));
    assert_eq!(bundles.latest_epoch(), Some(20));
    assert_eq!(weights.len(), 1);
    assert!(weights.get("c1", 20, "aa").is_some());
    assert_eq!(bundles.seal_record(20).expect("record").revision, 1);

    // A genuine re-seal appends the next revision; the store then serves it.
    let resealed = sealed_bundle(20, [2u8; 32]);
    let zeros = [0u8; 32];
    db::insert_epoch_bundle(
        tp.pool(),
        &NewEpochBundle {
            epoch: 20,
            protocol_version: 1,
            block_number: 900,
            block_hash: &zeros,
            metagraph_root: &zeros,
            merkle_root: &[2u8; 32],
            measurements_digest: &zeros,
            vector_hash: &zeros,
            payload: &resealed,
            signature: &[7u8; 64],
        },
    )
    .await
    .expect("re-seal");

    assert_eq!(bundles.seal_record(20).expect("record").revision, 2);
    assert_eq!(bundles.get_by_epoch(20), Some(resealed));

    tp.drop_schema().await.expect("drop");
}
