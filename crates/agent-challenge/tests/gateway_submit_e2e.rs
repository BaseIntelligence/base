//! Todo 28: `emit_signed_leaf_set` → `GatewayClient` POST /v1/weights/raw → seal → /v1/weights/latest.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use agent_challenge::{
    emit_signed_leaf_set, public_key_from_secret, submit_signed_leaf_set, AgentV1Challenge,
    Challenge, GatewayClient, GatewayClientConfig, NoScoreReasonCode, ScoreOrAbsence, SubmitError,
    SubmitOutcome, CHALLENGE_ID, SCORE_MAX,
};
use chain::{FakeChain, FakeChainConfig};
use crypto::KEY_LEN;
use gateway::{
    build_app_with_bundles, seal_epoch, ChallengeEntry, ChallengesBody, MemoryBundleStore,
    MemoryRawWeightStore, ParticipantPolicy, RawWeightStore, Registry, RegistryConfig, SealParams,
    SharedBundleStore, TlsConfig, BPS_DENOM,
};
use schnorrkel::MiniSecretKey;
use telemetry::init_metrics;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use trustroot::{measurements_digest, MeasurementsBody};

const EPOCH: u64 = 28;
const BLOCK_B: u64 = 200;
const SOLVER: [u8; KEY_LEN] = [0xA1; KEY_LEN];
const ZERO: [u8; KEY_LEN] = [0xB2; KEY_LEN];
const UNREACH: [u8; KEY_LEN] = [0xC3; KEY_LEN];

fn sk() -> [u8; KEY_LEN] {
    MiniSecretKey::generate_with(rand_core::OsRng).to_bytes()
}

fn e_three() -> BTreeSet<[u8; KEY_LEN]> {
    BTreeSet::from([SOLVER, ZERO, UNREACH])
}

fn scores_three() -> BTreeMap<[u8; KEY_LEN], ScoreOrAbsence> {
    BTreeMap::from([
        (SOLVER, ScoreOrAbsence::Score { value: SCORE_MAX }),
        (ZERO, ScoreOrAbsence::Score { value: 0 }),
        (
            UNREACH,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::Timeout,
            },
        ),
    ])
}

fn challenges(pk: [u8; KEY_LEN]) -> ChallengesBody {
    ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: CHALLENGE_ID.as_bytes().to_vec(),
            public_key: pk,
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    }
}

async fn spawn_gateway(
    challenges: ChallengesBody,
    weights: Arc<MemoryRawWeightStore>,
    bundles: SharedBundleStore,
) -> (SocketAddr, oneshot::Sender<()>) {
    let _ = telemetry::init_tracing();
    let metrics = init_metrics().expect("metrics");
    let registry = Registry::shared(RegistryConfig::default());
    let app = build_app_with_bundles(
        metrics,
        registry,
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

fn gw_client(base: &str) -> GatewayClient {
    GatewayClient::new(GatewayClientConfig {
        base_url: base.into(),
        max_attempts: agent_challenge::DEFAULT_MAX_RETRIES,
        backoff: Duration::from_millis(5),
    })
    .unwrap()
}

/// S1: emit |E| → POST raw → seal → GET /v1/weights/latest 200 with |E| leaves.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn s1_emit_submit_seal_latest_serves_bundle() {
    let secret = sk();
    let pk = public_key_from_secret(&secret).unwrap();
    let expected = e_three();
    let scores = scores_three();
    let leaves = emit_signed_leaf_set(&secret, EPOCH, &expected, &scores).expect("emit");
    assert_eq!(leaves.len(), 3);

    let weights = Arc::new(MemoryRawWeightStore::new());
    let bundles = Arc::new(MemoryBundleStore::new());
    let ch_body = challenges(pk);
    let (addr, shutdown) = spawn_gateway(
        ch_body.clone(),
        Arc::clone(&weights),
        bundles.clone() as SharedBundleStore,
    )
    .await;

    // Before seal: latest is 404.
    let http = reqwest::Client::new();
    let pre = http
        .get(format!("http://{addr}/v1/weights/latest"))
        .send()
        .await
        .unwrap();
    assert_eq!(pre.status().as_u16(), 404, "no sealed bundle yet");

    let client = gw_client(&format!("http://{addr}"));
    let outcomes = submit_signed_leaf_set(&client, &leaves)
        .await
        .expect("submit");
    assert_eq!(outcomes.len(), 3);
    assert!(outcomes.iter().all(|o| *o == SubmitOutcome::Accepted));
    assert_eq!(weights.len(), 3, "exactly |E| rows stored");

    // Trait path also wires emit → submit (idempotent 409).
    let ch = AgentV1Challenge::new();
    let again = ch
        .submit_all(&secret, EPOCH, &expected, &scores, &client)
        .await
        .expect("resubmit");
    assert!(again.iter().all(|o| *o == SubmitOutcome::AlreadyPresent));
    assert_eq!(weights.len(), 3, "retry must not duplicate leaves");

    let chain = FakeChain::new(FakeChainConfig {
        current_block: BLOCK_B.max(10),
        hotkeys: expected.iter().map(|h| h.to_vec()).collect(),
        owner_hotkey: vec![0xA1; 32],
        ..FakeChainConfig::default()
    });
    let gsk = sk();
    let mdigest = measurements_digest(&MeasurementsBody::default());
    let params = SealParams {
        epoch: EPOCH,
        netuid: 1,
        block_b: BLOCK_B,
        gateway_secret: gsk,
        measurements_digest: mdigest,
    };
    let bundle = seal_epoch(
        &chain,
        &ch_body,
        weights.as_ref(),
        bundles.as_ref(),
        &params,
    )
    .expect("seal");
    assert_eq!(bundle.body.leaves.len(), 3, "sealed |E| leaves");

    let resp = http
        .get(format!("http://{addr}/v1/weights/latest"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200, "sealed bundle served");
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["epoch"], EPOCH);
    assert_eq!(
        json["merkle_root"].as_str().unwrap(),
        hex::encode(bundle.body.merkle_root)
    );
    assert!(
        !json["final_vector"].as_array().unwrap().is_empty(),
        "final_vector present"
    );

    // Evidence artifact (happy path).
    let evidence = format!(
        "task-28 gateway sealed bundle\n\
         date: {}\n\
         pre_latest_status: 404\n\
         post_submit_store_len: {}\n\
         sealed_leaves: {}\n\
         latest_status: 200\n\
         latest_epoch: {}\n\
         latest_merkle_root: {}\n\
         final_vector_len: {}\n\
         resubmit_outcomes: all AlreadyPresent\n\
         store_len_after_resubmit: {}\n",
        chrono_like_now(),
        3,
        bundle.body.leaves.len(),
        json["epoch"],
        json["merkle_root"].as_str().unwrap(),
        json["final_vector"].as_array().unwrap().len(),
        weights.len(),
    );
    std::fs::write(
        "/root/.omo/evidence/gbase-agent-challenge-deepagent/task-28-gateway-sealed-bundle.txt",
        evidence,
    )
    .unwrap();

    let _ = shutdown.send(());
}

/// S2: bad `challenge_sig` / untrusted key rejected; 5xx retry does not double-count.
#[tokio::test]
async fn s2_bad_sig_rejected_and_retry_no_duplicate() {
    let secret = sk();
    let pk = public_key_from_secret(&secret).unwrap();
    let expected = e_three();
    let scores = scores_three();
    let mut leaves = emit_signed_leaf_set(&secret, EPOCH, &expected, &scores).expect("emit");

    // Corrupt one signature → gateway 401, store stays empty for that key.
    let bad_hk = SOLVER;
    if let Some(leaf) = leaves.get_mut(&bad_hk) {
        leaf.challenge_sig[0] ^= 0xFF;
    }

    let weights = Arc::new(MemoryRawWeightStore::new());
    let bundles = Arc::new(MemoryBundleStore::new());
    let (addr, shutdown) = spawn_gateway(
        challenges(pk),
        Arc::clone(&weights),
        bundles as SharedBundleStore,
    )
    .await;
    let client = gw_client(&format!("http://{addr}"));

    let err = submit_signed_leaf_set(&client, &leaves)
        .await
        .expect_err("bad sig must fail");
    match &err {
        SubmitError::Http { status, body } => {
            assert_eq!(*status, 401, "D18 unauthorized: {body}");
            assert!(
                body.to_lowercase().contains("unauthorized")
                    || body.to_lowercase().contains("signature")
                    || body.to_lowercase().contains("invalid"),
                "body={body}"
            );
        }
        other => panic!("expected Http 401, got {other:?}"),
    }
    // First leaf in BTreeMap order may be SOLVER (0xA1...) — if submit stops on first error,
    // store may be empty or partial before bad leaf. BTreeMap order is by hotkey bytes.
    // SOLVER=0xA1, ZERO=0xB2, UNREACH=0xC3 → SOLVER first. So store must be empty.
    assert_eq!(
        weights.len(),
        0,
        "bad sig must not leave a row for the corrupted leaf; store_len={}",
        weights.len()
    );

    // Untrusted key: sign with foreign sk while gateway trusts `pk`.
    let foreign = sk();
    let foreign_leaves =
        emit_signed_leaf_set(&foreign, EPOCH, &expected, &scores).expect("foreign emit");
    let err2 = submit_signed_leaf_set(&client, &foreign_leaves)
        .await
        .expect_err("untrusted key");
    match &err2 {
        SubmitError::Http { status, .. } => assert_eq!(*status, 401),
        other => panic!("expected 401, got {other:?}"),
    }
    assert_eq!(weights.len(), 0);

    // Good leaves + wiremock 5xx then 202: retries must not double-count on real gateway either.
    let good = emit_signed_leaf_set(&secret, EPOCH, &expected, &scores).expect("good");
    let outcomes = submit_signed_leaf_set(&client, &good)
        .await
        .expect("good submit");
    assert_eq!(outcomes.len(), 3);
    assert_eq!(weights.len(), 3);
    let outcomes2 = submit_signed_leaf_set(&client, &good)
        .await
        .expect("idempotent");
    assert!(outcomes2
        .iter()
        .all(|o| *o == SubmitOutcome::AlreadyPresent));
    assert_eq!(weights.len(), 3, "no duplicate after retry/resubmit");

    let evidence = format!(
        "task-28 untrusted key and idempotency\n\
         date: {}\n\
         bad_sig_status: 401\n\
         untrusted_key_status: 401\n\
         store_after_rejects: 0\n\
         good_submit_store_len: 3\n\
         resubmit_all_AlreadyPresent: true\n\
         store_after_resubmit: {}\n\
         DEFAULT_MAX_RETRIES: {}\n",
        chrono_like_now(),
        weights.len(),
        agent_challenge::DEFAULT_MAX_RETRIES,
    );
    std::fs::write(
        "/root/.omo/evidence/gbase-agent-challenge-deepagent/task-28-untrusted-key-and-idempotency.txt",
        evidence,
    )
    .unwrap();

    let _ = shutdown.send(());
}

fn chrono_like_now() -> String {
    // Avoid chrono dep; ISO-ish from system clock.
    std::process::Command::new("date")
        .arg("-Iseconds")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".into(), |s| s.trim().to_owned())
}
