#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
//! Task 27 VERIFY: epoch sealer + bundle publication.
//!
//! S1 `FakeChain` + 3 seeded leaves → served bundle verifies; merkle matches; vector == aggregate
//! S2 Re-seal is idempotent (identical bytes + signature)
//! S3 Missing participant + no `NoScore` → seal fails (no publish)
//! S4 GET /v1/bundle/{epoch}, /v1/bundle/root/{root}, /v1/weights/latest

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bundle::{
    compute_merkle_root, make_signed_leaf, sort_leaves, verify_bundle, EpochBundleV1, LeafV1,
    LocalTrustRoot, ScoreOrAbsence, ALGORITHM_VERSION,
};
use chain::{FakeChain, FakeChainConfig};
use crypto::{secret_from_bytes, KEY_LEN};
use gateway::{
    build_app_with_bundles, seal_epoch, BundleStore, ChallengeEntry, ChallengesBody,
    MemoryBundleStore, MemoryRawWeightStore, ParticipantPolicy, RawWeightRow, RawWeightStore,
    Registry, RegistryConfig, SealError, SealParams, SharedBundleStore, TlsConfig, BPS_DENOM,
};
use sha2::{Digest, Sha256};
use telemetry::init_metrics;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use trustroot::{measurements_digest, MeasurementsBody};
use uuid::Uuid;

fn sk(tag: u8) -> [u8; KEY_LEN] {
    let dig = Sha256::digest([0x5A, tag, 0xA5, tag]);
    let mut seed = [0u8; KEY_LEN];
    seed.copy_from_slice(&dig);
    seed
}

fn pk_of(secret: &[u8; KEY_LEN]) -> [u8; KEY_LEN] {
    secret_from_bytes(secret)
        .expect("sk")
        .to_public()
        .to_bytes()
}

fn hk(tag: u8) -> [u8; KEY_LEN] {
    let mut h = [0u8; KEY_LEN];
    h[0] = tag;
    h
}

fn trust(cid: &[u8], challenge_pk: [u8; KEY_LEN]) -> LocalTrustRoot {
    LocalTrustRoot {
        challenges: ChallengesBody {
            challenges: vec![ChallengeEntry {
                id: cid.to_vec(),
                public_key: challenge_pk,
                emission_share_bps: BPS_DENOM,
                policy: ParticipantPolicy::AllMetagraphHotkeys,
            }],
        },
        measurements_digest: measurements_digest(&MeasurementsBody::default()),
    }
}

fn chain_with(hotkeys: Vec<[u8; KEY_LEN]>, block_b: u64) -> FakeChain {
    FakeChain::new(FakeChainConfig {
        current_block: block_b.max(10),
        hotkeys: hotkeys.into_iter().map(|h| h.to_vec()).collect(),
        owner_hotkey: vec![0xA1; 32],
        ..FakeChainConfig::default()
    })
}

fn seed_leaf_row(
    store: &MemoryRawWeightStore,
    csk: &[u8; KEY_LEN],
    cid: &[u8],
    miner: [u8; KEY_LEN],
    epoch: u64,
    soa: ScoreOrAbsence,
) {
    let leaf = make_signed_leaf(csk, cid, miner, epoch, soa).expect("leaf");
    insert_leaf(store, &leaf);
}

fn insert_leaf(store: &MemoryRawWeightStore, leaf: &LeafV1) {
    let payload = bundle::raw_weight_payload(
        &leaf.challenge_id,
        &leaf.miner_hotkey,
        leaf.epoch,
        &leaf.score_or_absence,
    );
    let digest = Sha256::digest(&payload);
    let mut payload_digest = [0u8; 32];
    payload_digest.copy_from_slice(&digest);
    let (kind, score_val, absence) = match &leaf.score_or_absence {
        ScoreOrAbsence::Score { value } => ("score".to_owned(), Some(*value), None),
        ScoreOrAbsence::NoScore { reason } => (
            "no_score".to_owned(),
            None,
            Some((*reason as u8).to_string()),
        ),
    };
    let row = RawWeightRow {
        id: Uuid::new_v4(),
        challenge_id: String::from_utf8(leaf.challenge_id.clone()).expect("utf8"),
        epoch: leaf.epoch,
        miner_hotkey: hex::encode(leaf.miner_hotkey),
        kind,
        score: score_val,
        absence_reason: absence,
        payload,
        payload_digest,
        challenge_sig: leaf.challenge_sig.to_vec(),
    };
    store.insert(row).expect("insert");
}

fn seal_fixture() -> (
    FakeChain,
    ChallengesBody,
    Arc<MemoryRawWeightStore>,
    Arc<MemoryBundleStore>,
    SealParams,
    LocalTrustRoot,
    [u8; KEY_LEN],
) {
    let csk = sk(1);
    let gsk = sk(2);
    let cid = b"dummy";
    let epoch = 7u64;
    let block_b = 100u64;
    let miners = [hk(0xA1), hk(0xB2), hk(0xC3)];
    let scores = [50u64, 30, 20];

    let trust = trust(cid, pk_of(&csk));
    let chain = chain_with(miners.to_vec(), block_b);
    let store = Arc::new(MemoryRawWeightStore::new());
    for (m, s) in miners.iter().zip(scores.iter()) {
        seed_leaf_row(
            store.as_ref(),
            &csk,
            cid,
            *m,
            epoch,
            ScoreOrAbsence::Score { value: *s },
        );
    }
    let bundles = Arc::new(MemoryBundleStore::new());
    let params = SealParams {
        epoch,
        netuid: 1,
        block_b,
        gateway_secret: gsk,
        measurements_digest: trust.measurements_digest,
    };
    (
        chain,
        trust.challenges.clone(),
        store,
        bundles,
        params,
        trust,
        gsk,
    )
}

#[test]
fn s1_seal_three_leaves_verifies_merkle_and_aggregate() {
    let (chain, challenges, weights, bundles, params, trust, _gsk) = seal_fixture();
    let bundle = seal_epoch(
        &chain,
        &challenges,
        weights.as_ref(),
        bundles.as_ref(),
        &params,
    )
    .expect("seal");

    verify_bundle(&bundle, &chain, &trust).expect("verify");

    let mut leaves = bundle.body.leaves.clone();
    sort_leaves(&mut leaves);
    let independent = compute_merkle_root(&leaves);
    assert_eq!(bundle.body.merkle_root, independent, "merkle_root mismatch");

    let expected = bundle::python_weights_from_parts(
        &leaves,
        &bundle.body.emission_shares,
        &bundle.body.uid_map,
    )
    .expect("agg")
    .final_vector;
    assert_eq!(bundle.body.final_vector, expected);
    // Python authority for this fixture (scores 50/30/20 on uids 0/1/2, uid 0 being the
    // burn sink): weights [0.5, 0.3, 0.2] -> round-half-even [32768, 19660, 13107].
    assert_eq!(
        bundle.body.final_vector,
        vec![(0, 32_768), (1, 19_660), (2, 13_107)]
    );
    assert_eq!(bundle.body.algorithm_version, ALGORITHM_VERSION);
    assert_eq!(bundle.body.leaves.len(), 3);

    // Served bytes round-trip
    let served = bundles.get_by_epoch(params.epoch).expect("stored");
    let decoded = EpochBundleV1::decode_bytes(&served).expect("decode");
    assert_eq!(decoded.encode_bytes(), served);
    assert_eq!(decoded.gateway_sig, bundle.gateway_sig);
}

#[test]
fn s2_reseal_idempotent_identical_bytes_and_signature() {
    let (chain, challenges, weights, bundles, params, _trust, _gsk) = seal_fixture();
    let b1 = seal_epoch(
        &chain,
        &challenges,
        weights.as_ref(),
        bundles.as_ref(),
        &params,
    )
    .expect("seal1");
    let bytes1 = b1.encode_bytes();
    let b2 = seal_epoch(
        &chain,
        &challenges,
        weights.as_ref(),
        bundles.as_ref(),
        &params,
    )
    .expect("seal2");
    let bytes2 = b2.encode_bytes();
    assert_eq!(bytes1, bytes2, "re-seal must be byte-identical");
    assert_eq!(b1.gateway_sig, b2.gateway_sig);
    assert_eq!(bundles.get_by_epoch(params.epoch).unwrap(), bytes1);
}

#[test]
fn s3_missing_participant_no_noscore_fails_no_publish() {
    let csk = sk(1);
    let gsk = sk(2);
    let cid = b"dummy";
    let epoch = 7u64;
    let block_b = 100u64;
    // Metagraph has 3 miners; only seed 2 leaves → incomplete.
    let miners = [hk(0xA1), hk(0xB2), hk(0xC3)];
    let trust = trust(cid, pk_of(&csk));
    let chain = chain_with(miners.to_vec(), block_b);
    let store = MemoryRawWeightStore::new();
    seed_leaf_row(
        &store,
        &csk,
        cid,
        miners[0],
        epoch,
        ScoreOrAbsence::Score { value: 10 },
    );
    seed_leaf_row(
        &store,
        &csk,
        cid,
        miners[1],
        epoch,
        ScoreOrAbsence::Score { value: 20 },
    );
    // miners[2] missing and no `NoScore`
    let bundles = MemoryBundleStore::new();
    let params = SealParams {
        epoch,
        netuid: 1,
        block_b,
        gateway_secret: gsk,
        measurements_digest: trust.measurements_digest,
    };
    let err = seal_epoch(&chain, &trust.challenges, &store, &bundles, &params)
        .expect_err("must fail closed");
    assert!(
        matches!(err, SealError::IncompleteParticipantSet),
        "got {err}"
    );
    assert!(
        bundles.get_by_epoch(epoch).is_none(),
        "must not publish incomplete bundle"
    );
}

async fn spawn_with_bundles(
    challenges: ChallengesBody,
    weights: Arc<MemoryRawWeightStore>,
    bundles: SharedBundleStore,
) -> (SocketAddr, oneshot::Sender<()>) {
    let _ = telemetry::init_tracing();
    let metrics = init_metrics().expect("metrics");
    let registry = Registry::shared(RegistryConfig::default());
    let chain: gateway::SharedChain =
        Arc::new(validator_sync::SyncChain::new(FakeChain::with_defaults()));
    let app = build_app_with_bundles(
        metrics,
        registry,
        chain,
        &TlsConfig::default(),
        Arc::new(challenges),
        weights as Arc<dyn RawWeightStore>,
        bundles,
    )
    .expect("router");

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await
            .expect("serve");
    });

    let client = reqwest::Client::new();
    for _ in 0..80 {
        if client
            .get(format!("http://{addr}/healthz"))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    (addr, tx)
}

#[tokio::test]
async fn s4_http_bundle_routes_and_weights_latest() {
    let (chain, challenges, weights, bundles, params, trust, _gsk) = seal_fixture();
    let bundle = seal_epoch(
        &chain,
        &challenges,
        weights.as_ref(),
        bundles.as_ref(),
        &params,
    )
    .expect("seal");
    let expected_bytes = bundle.encode_bytes();
    let root_hex = hex::encode(bundle.body.merkle_root);

    let (addr, shutdown) =
        spawn_with_bundles(challenges, weights, bundles as SharedBundleStore).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();

    // GET /v1/bundle/{epoch}
    let resp = client
        .get(format!("http://{addr}/v1/bundle/{}", params.epoch))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("application/octet-stream")
    );
    let body = resp.bytes().await.unwrap();
    assert_eq!(body.as_ref(), expected_bytes.as_slice());
    let fetched = EpochBundleV1::decode_bytes(&body).unwrap();
    verify_bundle(&fetched, &chain, &trust).unwrap();

    // GET /v1/bundle/root/{root}
    let resp = client
        .get(format!("http://{addr}/v1/bundle/root/{root_hex}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let body2 = resp.bytes().await.unwrap();
    assert_eq!(body2.as_ref(), expected_bytes.as_slice());

    // GET /v1/weights/latest
    let resp = client
        .get(format!("http://{addr}/v1/weights/latest"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["epoch"], params.epoch);
    assert_eq!(json["merkle_root"], root_hex);
    assert!(!json["final_vector"].as_array().unwrap().is_empty());
    // Served body carries the full master-weights contract, not just the extras.
    for key in [
        "protocol_version",
        "vector_id",
        "vector_digest",
        "revision",
        "netuid",
        "chain_endpoint",
        "uids",
        "weights",
        "hotkey_weights",
        "chain_domain_bytes",
        "computed_at",
        "expires_at",
        "source_challenges",
        "source_snapshots",
        "source_outcomes",
        "emission_policy_version",
        "emission_shares",
        "burn_policy_version",
        "mapping_policy_version",
        "metagraph_identity",
        "metagraph_hash",
        "metagraph_block",
        "burn_outcome",
        "metagraph_updated_at",
    ] {
        assert!(json.get(key).is_some(), "missing {key} in served body");
    }
    assert_eq!(json["revision"], 1);
    assert!(json["computed_at"].as_str().unwrap().ends_with('Z'));

    // 404 unknown epoch
    let resp = client
        .get(format!("http://{addr}/v1/bundle/999999"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);

    let _ = shutdown.send(());
}
