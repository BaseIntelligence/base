#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]
//! Admin seal HTTP path: POST /v1/admin/seal → GET /v1/weights/latest 200.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use bundle::{
    compute_metagraph_root, make_signed_leaf, metagraph_rows_from_chain, uid_map_from_rows,
    EpochBundleV1, ScoreOrAbsence,
};
use chain::{FakeChain, FakeChainConfig};
use crypto::{secret_from_bytes, KEY_LEN};
use gateway::{
    admin_seal_router, bundle_router, weights_router, BundleStore, ChallengeEntry, ChallengesBody,
    GatewayState, MemoryBundleStore, MemoryRawWeightStore, ParticipantPolicy, RawWeightRow,
    RawWeightStore, Registry, RegistryConfig, SharedBundleStore, SharedChain, BPS_DENOM,
};
use sha2::{Digest, Sha256};
use telemetry::init_metrics;
use tokio::net::TcpListener;
use trustroot::{measurements_digest, MeasurementsBody};
use uuid::Uuid;

/// In-repo chain double for the seal path; production wires the live client.
fn fake_chain(hotkeys: Vec<Vec<u8>>, tip: u64) -> SharedChain {
    Arc::new(validator_sync::SyncChain::new(FakeChain::new(
        FakeChainConfig {
            hotkeys,
            owner_hotkey: vec![0xA1; 32],
            current_block: tip.max(10),
            ..FakeChainConfig::default()
        },
    )))
}

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

fn insert_leaf(
    store: &MemoryRawWeightStore,
    csk: &[u8; KEY_LEN],
    cid: &[u8],
    miner: [u8; KEY_LEN],
    epoch: u64,
    value: u64,
) {
    let leaf =
        make_signed_leaf(csk, cid, miner, epoch, ScoreOrAbsence::Score { value }).expect("leaf");
    let payload = bundle::raw_weight_payload(
        &leaf.challenge_id,
        &leaf.miner_hotkey,
        leaf.epoch,
        &leaf.score_or_absence,
    );
    let digest = Sha256::digest(&payload);
    let mut payload_digest = [0u8; 32];
    payload_digest.copy_from_slice(&digest);
    store
        .insert(RawWeightRow {
            id: Uuid::new_v4(),
            challenge_id: String::from_utf8(leaf.challenge_id.clone()).expect("utf8"),
            epoch: leaf.epoch,
            miner_hotkey: hex::encode(leaf.miner_hotkey),
            kind: "score".into(),
            score: Some(value),
            absence_reason: None,
            payload,
            payload_digest,
            challenge_sig: leaf.challenge_sig.to_vec(),
        })
        .expect("insert");
}

async fn serve(app: Router) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await
            .ok();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (addr, tx)
}

#[tokio::test]
async fn s1_admin_seal_happy_latest_200() {
    std::env::set_var("BASE_GATEWAY_SK", hex::encode(sk(9)));
    let _ = init_metrics();
    let csk = sk(1);
    let cid = b"prism";
    let miners = [hk(1), hk(2), hk(3)];
    let epoch = 91u64;
    let block_b = 920u64;
    let challenges = Arc::new(ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: cid.to_vec(),
            public_key: pk_of(&csk),
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    });
    let mdigest = measurements_digest(&MeasurementsBody::default());
    let weights = Arc::new(MemoryRawWeightStore::new());
    for (i, m) in miners.iter().enumerate() {
        insert_leaf(weights.as_ref(), &csk, cid, *m, epoch, 10 * (i as u64 + 1));
    }
    let bundles = Arc::new(MemoryBundleStore::new());
    let hotkeys: Vec<Vec<u8>> = miners.iter().map(|h| h.to_vec()).collect();
    let registry = Registry::shared(RegistryConfig::default());
    let state = GatewayState::with_parts_seal(
        registry,
        fake_chain(hotkeys, block_b),
        challenges,
        weights,
        bundles as SharedBundleStore,
        mdigest,
        1,
    )
    .expect("state");

    let metrics = init_metrics().expect("metrics");
    let health = telemetry::health_router(metrics).expect("health");
    let app = health
        .merge(weights_router(state.clone()))
        .merge(bundle_router(state.clone()))
        .merge(admin_seal_router(state));

    let (addr, shutdown) = serve(app).await;
    let client = reqwest::Client::new();

    let pre = client
        .get(format!("http://{addr}/v1/weights/latest"))
        .send()
        .await
        .unwrap();
    assert_eq!(pre.status().as_u16(), 404);

    let seal = client
        .post(format!("http://{addr}/v1/admin/seal"))
        .json(&serde_json::json!({ "epoch": epoch, "block_b": block_b, "netuid": 1 }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        seal.status().as_u16(),
        200,
        "body={}",
        seal.text().await.unwrap_or_default()
    );
    let body: serde_json::Value = client
        .post(format!("http://{addr}/v1/admin/seal"))
        .json(&serde_json::json!({ "epoch": epoch, "block_b": block_b }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // idempotent re-seal still 200
    assert_eq!(body["epoch"], epoch);

    let latest = client
        .get(format!("http://{addr}/v1/weights/latest"))
        .send()
        .await
        .unwrap();
    assert_eq!(latest.status().as_u16(), 200);
    let json: serde_json::Value = latest.json().await.unwrap();
    assert_eq!(json["epoch"], epoch);
    assert!(json["merkle_root"].as_str().unwrap().len() == 64);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn s2_admin_seal_incomplete_409() {
    std::env::set_var("BASE_GATEWAY_SK", hex::encode(sk(9)));
    let _ = init_metrics();
    let csk = sk(1);
    let cid = b"prism";
    let epoch = 92u64;
    let challenges = Arc::new(ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: cid.to_vec(),
            public_key: pk_of(&csk),
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    });
    // Only one leaf but metagraph has 3 hotkeys → incomplete
    let weights = Arc::new(MemoryRawWeightStore::new());
    insert_leaf(weights.as_ref(), &csk, cid, hk(1), epoch, 1);
    let bundles = Arc::new(MemoryBundleStore::new());
    let state = GatewayState::with_parts_seal(
        Registry::shared(RegistryConfig::default()),
        fake_chain(vec![hk(1).to_vec(), hk(2).to_vec(), hk(3).to_vec()], 1000),
        challenges,
        weights,
        bundles as SharedBundleStore,
        measurements_digest(&MeasurementsBody::default()),
        1,
    )
    .expect("state");
    let metrics = init_metrics().expect("m");
    let app = telemetry::health_router(metrics)
        .expect("h")
        .merge(admin_seal_router(state));
    let (addr, shutdown) = serve(app).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/admin/seal"))
        .json(&serde_json::json!({ "epoch": epoch, "block_b": 900 }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 409);
    let _ = shutdown.send(());
}

/// The sealed body must be pinned to the chain handed to the gateway, not to a
/// default or env-derived metagraph: UID order, metagraph root, block hash and
/// tip all have to come back out of the injected chain.
#[tokio::test]
async fn s3_admin_seal_pins_injected_chain() {
    std::env::set_var("BASE_GATEWAY_SK", hex::encode(sk(9)));
    let _ = init_metrics();
    let csk = sk(1);
    let cid = b"prism";
    let epoch = 93u64;
    let tip = 777u64;
    // Deliberately not sorted: a fabricated metagraph would not reproduce it.
    let uid_order = vec![hk(3).to_vec(), hk(1).to_vec(), hk(2).to_vec()];
    let challenges = Arc::new(ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: cid.to_vec(),
            public_key: pk_of(&csk),
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    });
    let weights = Arc::new(MemoryRawWeightStore::new());
    for (i, m) in [hk(1), hk(2), hk(3)].iter().enumerate() {
        insert_leaf(weights.as_ref(), &csk, cid, *m, epoch, 10 * (i as u64 + 1));
    }
    let bundles = Arc::new(MemoryBundleStore::new());
    let chain = fake_chain(uid_order.clone(), tip);
    let state = GatewayState::with_parts_seal(
        Registry::shared(RegistryConfig::default()),
        Arc::clone(&chain),
        challenges,
        weights,
        bundles.clone() as SharedBundleStore,
        measurements_digest(&MeasurementsBody::default()),
        1,
    )
    .expect("state");
    let metrics = init_metrics().expect("metrics");
    let app = telemetry::health_router(metrics)
        .expect("health")
        .merge(admin_seal_router(state));
    let (addr, shutdown) = serve(app).await;

    // No `block_b`: the tip must be read from the chain, not from a constant.
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/admin/seal"))
        .json(&serde_json::json!({ "epoch": epoch }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "body={}",
        resp.text().await.unwrap_or_default()
    );
    let _ = shutdown.send(());

    let sealed = bundles.get_by_epoch(epoch).expect("sealed bytes");
    let body = EpochBundleV1::decode_bytes(&sealed).expect("decode").body;
    let rows = metagraph_rows_from_chain(&uid_order, None).expect("rows");
    assert_eq!(body.block_b, tip);
    assert_eq!(body.block_hash, chain.block_hash(tip).expect("hash"));
    assert_eq!(body.metagraph_root, compute_metagraph_root(&rows));
    assert_eq!(body.uid_map, uid_map_from_rows(&rows));
    // hk(3) is uid 0 only because the injected chain says so.
    assert_eq!(
        body.uid_map
            .iter()
            .find(|(h, _)| *h == hk(3))
            .map(|(_, uid)| *uid),
        Some(0)
    );
}
