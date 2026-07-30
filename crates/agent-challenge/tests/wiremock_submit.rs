//! Wiremock integration: POST /v1/weights/raw, 5xx retry, idempotent retry after 202.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use agent_challenge::{
    make_signed_leaf, public_key_from_secret, AgentV1Challenge, Challenge, GatewayClient,
    GatewayClientConfig, NoScoreReasonCode, ScoreOrAbsence, SubmitOutcome, CHALLENGE_ID,
};
use crypto::KEY_LEN;
use rand_core::OsRng;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn mini_secret() -> [u8; KEY_LEN] {
    schnorrkel::MiniSecretKey::generate_with(OsRng).to_bytes()
}

fn signed_score(
    sk: &[u8; KEY_LEN],
    miner: [u8; 32],
    epoch: u64,
    value: u64,
) -> agent_challenge::LeafV1 {
    make_signed_leaf(
        sk,
        CHALLENGE_ID.as_bytes(),
        miner,
        epoch,
        ScoreOrAbsence::Score { value },
    )
    .expect("sign")
}

fn signed_noscore(
    sk: &[u8; KEY_LEN],
    miner: [u8; 32],
    epoch: u64,
    reason: NoScoreReasonCode,
) -> agent_challenge::LeafV1 {
    make_signed_leaf(
        sk,
        CHALLENGE_ID.as_bytes(),
        miner,
        epoch,
        ScoreOrAbsence::NoScore { reason },
    )
    .expect("sign")
}

#[tokio::test]
async fn well_formed_signed_submission_posts_202() {
    let sk = mini_secret();
    let pk = public_key_from_secret(&sk).unwrap();
    let miner = [0x11u8; 32];
    let leaf = signed_score(&sk, miner, 7, 1_000_000);

    let server = MockServer::start().await;
    let hits = Arc::new(AtomicU32::new(0));
    let hits_c = Arc::clone(&hits);

    Mock::given(method("POST"))
        .and(path("/v1/weights/raw"))
        .respond_with(move |req: &Request| {
            hits_c.fetch_add(1, Ordering::SeqCst);
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
            assert_eq!(body["challenge_id"], CHALLENGE_ID);
            assert_eq!(body["epoch"], 7);
            assert_eq!(body["miner_hotkey"], hex::encode(miner));
            assert_eq!(body["score_or_absence"]["score"]["value"], 1_000_000);
            assert!(body["challenge_sig"].as_str().unwrap().len() == 128);
            // Public key of signer is the committed challenge key (caller checks separately).
            let _ = pk;
            ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "id": "00000000-0000-0000-0000-000000000001",
                "challenge_id": CHALLENGE_ID,
                "epoch": 7,
                "miner_hotkey": hex::encode(miner),
                "kind": "score",
                "score": 1_000_000
            }))
        })
        .mount(&server)
        .await;

    let client = GatewayClient::new(GatewayClientConfig {
        base_url: server.uri(),
        max_attempts: 3,
        backoff: std::time::Duration::from_millis(5),
    })
    .unwrap();

    let outcome = client.submit_leaf(&leaf).await.unwrap();
    assert_eq!(outcome, SubmitOutcome::Accepted);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn five_xx_is_retried_then_succeeds() {
    let sk = mini_secret();
    let miner = [0x33u8; 32];
    let leaf = signed_score(&sk, miner, 9, 42);

    let server = MockServer::start().await;
    let hits = Arc::new(AtomicU32::new(0));
    let hits_c = Arc::clone(&hits);

    Mock::given(method("POST"))
        .and(path("/v1/weights/raw"))
        .respond_with(move |_req: &Request| {
            let n = hits_c.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                ResponseTemplate::new(503).set_body_string("unavailable")
            } else {
                ResponseTemplate::new(202).set_body_json(serde_json::json!({
                    "id": "00000000-0000-0000-0000-000000000002",
                    "challenge_id": CHALLENGE_ID,
                    "epoch": 9,
                    "miner_hotkey": hex::encode(miner),
                    "kind": "score",
                    "score": 42
                }))
            }
        })
        .expect(3)
        .mount(&server)
        .await;

    let client = GatewayClient::new(GatewayClientConfig {
        base_url: server.uri(),
        max_attempts: 5,
        backoff: std::time::Duration::from_millis(5),
    })
    .unwrap();

    let outcome = client.submit_leaf(&leaf).await.unwrap();
    assert_eq!(outcome, SubmitOutcome::Accepted);
    assert_eq!(hits.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retry_after_202_does_not_duplicate() {
    // First POST → 202; second POST same leaf → 409 AlreadyPresent (idempotent).
    let sk = mini_secret();
    let miner = [0x44u8; 32];
    let leaf = signed_score(&sk, miner, 11, 7);

    let server = MockServer::start().await;
    let hits = Arc::new(AtomicU32::new(0));
    let hits_c = Arc::clone(&hits);

    Mock::given(method("POST"))
        .and(path("/v1/weights/raw"))
        .respond_with(move |_req: &Request| {
            let n = hits_c.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(202).set_body_json(serde_json::json!({
                    "id": "00000000-0000-0000-0000-000000000003",
                    "challenge_id": CHALLENGE_ID,
                    "epoch": 11,
                    "miner_hotkey": hex::encode(miner),
                    "kind": "score",
                    "score": 7
                }))
            } else {
                // Gateway unique key conflict — client must treat as success, not re-POST a different body.
                ResponseTemplate::new(409).set_body_json(serde_json::json!({
                    "error": "conflict: raw weight already stored",
                    "original": {
                        "id": "00000000-0000-0000-0000-000000000003",
                        "challenge_id": CHALLENGE_ID,
                        "epoch": 11,
                        "miner_hotkey": hex::encode(miner),
                        "kind": "score",
                        "score": 7
                    }
                }))
            }
        })
        .expect(2)
        .mount(&server)
        .await;

    let client = GatewayClient::new(GatewayClientConfig {
        base_url: server.uri(),
        max_attempts: 3,
        backoff: std::time::Duration::from_millis(5),
    })
    .unwrap();

    assert_eq!(
        client.submit_leaf(&leaf).await.unwrap(),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        client.submit_leaf(&leaf).await.unwrap(),
        SubmitOutcome::AlreadyPresent
    );
    assert_eq!(hits.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn noscore_leaf_posts_and_pubkey_matches_secret() {
    let sk = mini_secret();
    let pk = public_key_from_secret(&sk).unwrap();
    let miner = [0x55u8; 32];
    let leaf = signed_noscore(&sk, miner, 3, NoScoreReasonCode::AttestationNotVerified);

    // Verify signature under derived public key (same as challenges.toml commitment).
    let payload =
        bundle::raw_weight_payload(CHALLENGE_ID.as_bytes(), &miner, 3, &leaf.score_or_absence);
    crypto::verify_raw(
        &pk,
        crypto::domain::RAW_WEIGHT,
        &payload,
        &leaf.challenge_sig,
    )
    .expect("sig verifies under challenge public key");

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/weights/raw"))
        .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
            "id": "00000000-0000-0000-0000-000000000004",
            "challenge_id": CHALLENGE_ID,
            "epoch": 3,
            "miner_hotkey": hex::encode(miner),
            "kind": "no_score",
            "absence_reason": "3"
        })))
        .mount(&server)
        .await;

    let client = GatewayClient::new(GatewayClientConfig {
        base_url: server.uri(),
        max_attempts: 2,
        backoff: std::time::Duration::from_millis(5),
    })
    .unwrap();
    assert_eq!(
        client.submit_leaf(&leaf).await.unwrap(),
        SubmitOutcome::Accepted
    );

    // Challenge trait path also signs NoScore.
    let ch = AgentV1Challenge::new();
    let leaf2 = ch
        .sign_leaf(
            &sk,
            miner,
            3,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::Timeout,
            },
        )
        .unwrap();
    assert!(matches!(
        leaf2.score_or_absence,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout
        }
    ));
}

#[tokio::test]
async fn committed_dummy_secret_pubkey_matches_challenges_toml_entry() {
    // /root/.gbase-secrets/challenge-dummy.age decrypts to the key whose pub is in challenges.toml.
    // Prefer decrypted raw path if present; else skip-friendly load of known pub file.
    let pub_path = std::path::Path::new("/root/.gbase-secrets/challenge-dummy.pub");
    let sk_path = std::path::Path::new("/tmp/challenge-dummy.sk");
    if !sk_path.exists() || !pub_path.exists() {
        // Environment without secrets — still assert hex file content if pub exists alone.
        if pub_path.exists() {
            let pub_hex = std::fs::read_to_string(pub_path).unwrap();
            assert_eq!(
                pub_hex.trim(),
                "f2e4965a6a99b75b4212bd45790c496e9665c0e1247e373d9dca3b36413fbd45"
            );
        }
        return;
    }
    let sk = agent_challenge::load_challenge_secret(sk_path).unwrap();
    let pk = public_key_from_secret(&sk).unwrap();
    let pub_hex = std::fs::read_to_string(pub_path).unwrap();
    assert_eq!(hex::encode(pk), pub_hex.trim());
    // challenges.toml (agent-v1 or dummy) commits this public key.
    let toml = std::fs::read_to_string("/root/gbase/config/challenges.toml").unwrap();
    assert!(
        toml.contains(&hex::encode(pk)),
        "challenges.toml must commit challenge public key"
    );
}
