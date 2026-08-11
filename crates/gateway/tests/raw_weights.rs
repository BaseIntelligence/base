#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]
//! Task 26 VERIFY: signed raw-weight ingress (`POST /v1/weights/raw`).
//!
//! S1 valid → 202 + row
//! S2 wrong key → 401 + NO row
//! S3 unknown challenge → 404 + no row
//! S4 replay identical digest → 409 + original unchanged
//! S5 digest change for same key → 202 tip supersede

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use crypto::{domain, sign_raw, KEY_LEN, SIGNATURE_LEN};
use gateway::{
    build_app_with, ChallengeEntry, ChallengesBody, MemoryRawWeightStore, ParticipantPolicy,
    RawWeightStore, Registry, RegistryConfig, TlsConfig, BPS_DENOM,
};
use parity_scale_codec::Encode;
use rand_core::OsRng;
use telemetry::init_metrics;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

fn mini_keypair() -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
    let mini = schnorrkel::MiniSecretKey::generate_with(OsRng);
    let secret = mini.to_bytes();
    let public = mini
        .expand(schnorrkel::ExpansionMode::Ed25519)
        .to_public()
        .to_bytes();
    (secret, public)
}

fn challenges_body(id: &str, public_key: [u8; KEY_LEN]) -> ChallengesBody {
    ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: id.as_bytes().to_vec(),
            public_key,
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    }
}

/// SCALE body matching `BUNDLE_SPEC` §3.4 `RawWeightBodyV1`.
#[derive(Encode)]
struct RawWeightBodyV1 {
    challenge_id: Vec<u8>,
    miner_hotkey: [u8; 32],
    epoch: u64,
    score_or_absence: ScoreOrAbsenceScale,
}

#[derive(Encode)]
enum ScoreOrAbsenceScale {
    Score { value: u64 },
    NoScore { reason: u8 },
}

fn sign_leaf(
    secret: &[u8; KEY_LEN],
    challenge_id: &str,
    miner: [u8; 32],
    epoch: u64,
    soa: ScoreOrAbsenceScale,
) -> (Vec<u8>, [u8; SIGNATURE_LEN]) {
    let body = RawWeightBodyV1 {
        challenge_id: challenge_id.as_bytes().to_vec(),
        miner_hotkey: miner,
        epoch,
        score_or_absence: soa,
    };
    let payload = body.encode();
    let sig = sign_raw(secret, domain::RAW_WEIGHT, &payload).expect("sign");
    (payload, sig)
}

async fn spawn_gateway(
    challenges: ChallengesBody,
    store: Arc<MemoryRawWeightStore>,
) -> (SocketAddr, oneshot::Sender<()>) {
    let _ = telemetry::init_tracing();
    let metrics = init_metrics().expect("metrics");
    let registry = Registry::shared(RegistryConfig::default());
    let chain: gateway::SharedChain = Arc::new(validator_sync::SyncChain::new(
        chain::FakeChain::with_defaults(),
    ));
    let app = build_app_with(
        metrics,
        registry,
        chain,
        &TlsConfig::default(),
        Arc::new(challenges),
        store as Arc<dyn RawWeightStore>,
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

fn json_score(
    challenge_id: &str,
    miner: [u8; 32],
    epoch: u64,
    value: u64,
    sig: &[u8; SIGNATURE_LEN],
) -> serde_json::Value {
    serde_json::json!({
        "challenge_id": challenge_id,
        "miner_hotkey": hex::encode(miner),
        "epoch": epoch,
        "score_or_absence": { "score": { "value": value } },
        "challenge_sig": hex::encode(sig),
    })
}

fn json_noscore(
    challenge_id: &str,
    miner: [u8; 32],
    epoch: u64,
    reason: u8,
    sig: &[u8; SIGNATURE_LEN],
) -> serde_json::Value {
    serde_json::json!({
        "challenge_id": challenge_id,
        "miner_hotkey": hex::encode(miner),
        "epoch": epoch,
        "score_or_absence": { "no_score": { "reason": reason } },
        "challenge_sig": hex::encode(sig),
    })
}

#[tokio::test]
async fn s1_valid_score_returns_202_and_row() {
    let (sk, pk) = mini_keypair();
    let cid = "dummy";
    let miner = [0x11u8; 32];
    let epoch = 7u64;
    let value = 42u64;
    let (_payload, sig) = sign_leaf(&sk, cid, miner, epoch, ScoreOrAbsenceScale::Score { value });

    let store = Arc::new(MemoryRawWeightStore::new());
    let (addr, shutdown) = spawn_gateway(challenges_body(cid, pk), Arc::clone(&store)).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&json_score(cid, miner, epoch, value, &sig))
        .send()
        .await
        .expect("post");

    assert_eq!(
        resp.status().as_u16(),
        202,
        "body={}",
        resp.text().await.unwrap_or_default()
    );
    assert_eq!(store.len(), 1);
    let row = store
        .get(cid, epoch, &hex::encode(miner))
        .expect("row present");
    assert_eq!(row.kind, "score");
    assert_eq!(row.score, Some(value));
    assert_eq!(row.challenge_sig, sig.to_vec());

    // NoScore path also accepted.
    let miner2 = [0x22u8; 32];
    let (_p2, sig2) = sign_leaf(
        &sk,
        cid,
        miner2,
        epoch,
        ScoreOrAbsenceScale::NoScore { reason: 1 },
    );
    let resp2 = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&json_noscore(cid, miner2, epoch, 1, &sig2))
        .send()
        .await
        .expect("post noscore");
    assert_eq!(resp2.status().as_u16(), 202);
    assert_eq!(store.len(), 2);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn s2_wrong_key_returns_401_and_no_row() {
    let (_sk_good, pk_good) = mini_keypair();
    let (sk_bad, _pk_bad) = mini_keypair();
    let cid = "dummy";
    let miner = [0x33u8; 32];
    let epoch = 3u64;
    let value = 9u64;
    let (_payload, sig) = sign_leaf(
        &sk_bad,
        cid,
        miner,
        epoch,
        ScoreOrAbsenceScale::Score { value },
    );

    let store = Arc::new(MemoryRawWeightStore::new());
    let (addr, shutdown) = spawn_gateway(challenges_body(cid, pk_good), Arc::clone(&store)).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&json_score(cid, miner, epoch, value, &sig))
        .send()
        .await
        .expect("post");

    assert_eq!(
        resp.status().as_u16(),
        401,
        "body={}",
        resp.text().await.unwrap_or_default()
    );
    assert_eq!(store.len(), 0);
    assert!(store.get(cid, epoch, &hex::encode(miner)).is_none());

    let _ = shutdown.send(());
}

#[tokio::test]
async fn s3_unknown_challenge_returns_404_and_no_row() {
    let (sk, pk) = mini_keypair();
    let registered = "dummy";
    let unknown = "not-registered";
    let miner = [0x44u8; 32];
    let epoch = 1u64;
    let value = 1u64;
    // Sign under unknown id (would verify if key were registered under that id).
    let (_payload, sig) = sign_leaf(
        &sk,
        unknown,
        miner,
        epoch,
        ScoreOrAbsenceScale::Score { value },
    );

    let store = Arc::new(MemoryRawWeightStore::new());
    let (addr, shutdown) = spawn_gateway(challenges_body(registered, pk), Arc::clone(&store)).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&json_score(unknown, miner, epoch, value, &sig))
        .send()
        .await
        .expect("post");

    assert_eq!(
        resp.status().as_u16(),
        404,
        "body={}",
        resp.text().await.unwrap_or_default()
    );
    assert_eq!(store.len(), 0);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn s4_replay_challenge_epoch_miner_returns_409_original_unchanged() {
    let (sk, pk) = mini_keypair();
    let cid = "dummy";
    let miner = [0x55u8; 32];
    let epoch = 11u64;
    let value = 100u64;
    let (_payload, sig) = sign_leaf(&sk, cid, miner, epoch, ScoreOrAbsenceScale::Score { value });

    let store = Arc::new(MemoryRawWeightStore::new());
    let (addr, shutdown) = spawn_gateway(challenges_body(cid, pk), Arc::clone(&store)).await;
    let client = reqwest::Client::new();
    let body = json_score(cid, miner, epoch, value, &sig);

    let first = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&body)
        .send()
        .await
        .expect("first");
    assert_eq!(first.status().as_u16(), 202);
    let original = store
        .get(cid, epoch, &hex::encode(miner))
        .expect("original")
        .clone();

    // Same key again (identical payload) → 409, row bytes unchanged.
    let second = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&body)
        .send()
        .await
        .expect("second");
    assert_eq!(
        second.status().as_u16(),
        409,
        "body={}",
        second.text().await.unwrap_or_default()
    );
    assert_eq!(store.len(), 1);
    let after = store
        .get(cid, epoch, &hex::encode(miner))
        .expect("still one");
    assert_eq!(after.payload, original.payload);
    assert_eq!(after.challenge_sig, original.challenge_sig);
    assert_eq!(after.score, original.score);
    assert_eq!(after.id, original.id);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn s5_digest_change_tip_supersedes() {
    let (sk, pk) = mini_keypair();
    let cid = "dummy";
    let miner = [0x66u8; 32];
    let epoch = 12u64;
    let (_payload, sig) = sign_leaf(
        &sk,
        cid,
        miner,
        epoch,
        ScoreOrAbsenceScale::Score { value: 100 },
    );

    let store = Arc::new(MemoryRawWeightStore::new());
    let (addr, shutdown) = spawn_gateway(challenges_body(cid, pk), Arc::clone(&store)).await;
    let client = reqwest::Client::new();

    let first = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&json_score(cid, miner, epoch, 100, &sig))
        .send()
        .await
        .expect("first");
    assert_eq!(first.status().as_u16(), 202);

    let (_p2, sig2) = sign_leaf(
        &sk,
        cid,
        miner,
        epoch,
        ScoreOrAbsenceScale::Score { value: 999 },
    );
    let second = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&json_score(cid, miner, epoch, 999, &sig2))
        .send()
        .await
        .expect("supersede");
    let status = second.status().as_u16();
    let body = second.text().await.unwrap_or_default();
    assert_eq!(status, 202, "body={body}");
    assert!(body.contains("\"superseded\":true"), "body={body}");
    let final_row = store.get(cid, epoch, &hex::encode(miner)).expect("final");
    assert_eq!(final_row.score, Some(999));
    assert_eq!(store.len(), 1);

    // Identical digest after supersede → 409.
    let third = client
        .post(format!("http://{addr}/v1/weights/raw"))
        .json(&json_score(cid, miner, epoch, 999, &sig2))
        .send()
        .await
        .expect("replay");
    assert_eq!(third.status().as_u16(), 409);

    let _ = shutdown.send(());
}
