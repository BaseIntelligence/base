#![allow(clippy::too_many_lines, clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
//! Task 31 VERIFY: peer root cross-check with min sample, FakeChain + in-process peers.
//!
//! Peer hotkeys are real sr25519 public keys (identity = signature key). Metagraph
//! neurons double as scored miners so `AllMetagraphHotkeys` stays complete.

use std::sync::Arc;
use std::time::Duration;

use gbase_aggregate::{
    aggregate, ScoreOrAbsence as AggScore, VerifiedLeaf, ALGORITHM_VERSION as AGG_V,
};
use gbase_bundle::{
    compute_merkle_root, compute_metagraph_root, make_signed_leaf, metagraph_rows_from_chain,
    sign_bundle, sort_leaves, uid_map_from_rows, EpochBundleBodyV1, EpochBundleV1, ScoreOrAbsence,
    ALGORITHM_VERSION, PROTOCOL_VERSION,
};
use gbase_chain::{ChainClient, FakeChain, FakeChainConfig};
use gbase_crypto::secret_from_bytes;
use gbase_trustroot::{
    measurements_digest, ChallengeEntry, ChallengesBody, MeasurementsBody, ParticipantPolicy,
};
use sha2::{Digest, Sha256};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::{
    fetch_compare_and_crosscheck, spawn_validator_with_ok_db, ComparisonOutcome, CrossCheckConfig,
    CrossCheckFailKind, CrossCheckOutcome, CrossCheckRun, ExpectedBundle, NoSubmissionReason,
    PeerBook, PeerEndpoint, SignedRootStatement, SyncChain, ValidatorRuntime,
};

fn sk(tag: u8) -> [u8; 32] {
    let dig = Sha256::digest([0x5A, tag, 0xA5, tag]);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&dig);
    seed
}

fn pk_of(secret: &[u8; 32]) -> [u8; 32] {
    secret_from_bytes(secret)
        .expect("sk")
        .to_public()
        .to_bytes()
}

fn to_agg(s: &ScoreOrAbsence) -> AggScore {
    match s {
        ScoreOrAbsence::Score { value } => AggScore::Score { value: *value },
        ScoreOrAbsence::NoScore { reason } => AggScore::NoScore {
            reason: *reason as u8,
        },
    }
}

/// Build a valid bundle. `miners` are `(hotkey, score)` — hotkeys form the metagraph.
fn valid_bundle(
    csk: &[u8; 32],
    gsk: &[u8; 32],
    cid: &[u8],
    miners: &[([u8; 32], u64)],
    block_b: u64,
    epoch: u64,
) -> (EpochBundleV1, gbase_bundle::LocalTrustRoot, FakeChain) {
    let cpk = pk_of(csk);
    let gpk = pk_of(gsk);
    let trust = gbase_bundle::LocalTrustRoot {
        challenges: ChallengesBody {
            challenges: vec![ChallengeEntry {
                id: cid.to_vec(),
                public_key: cpk,
                emission_share_bps: 10_000,
                policy: ParticipantPolicy::AllMetagraphHotkeys,
            }],
        },
        measurements_digest: measurements_digest(&MeasurementsBody::default()),
    };
    let hotkeys: Vec<[u8; 32]> = miners.iter().map(|(h, _)| *h).collect();
    let chain = FakeChain::new(FakeChainConfig {
        current_block: block_b.max(10),
        hotkeys: hotkeys.iter().map(|h| h.to_vec()).collect(),
        owner_hotkey: vec![0xA1; 32],
        ..FakeChainConfig::default()
    });
    let block_hash = chain.block_hash(block_b).expect("hash");
    let rows = metagraph_rows_from_chain(
        &hotkeys.iter().map(|h| h.to_vec()).collect::<Vec<_>>(),
        None,
    )
    .expect("rows");
    let mut leaves = Vec::new();
    for (hotkey, score) in miners {
        leaves.push(
            make_signed_leaf(
                csk,
                cid,
                *hotkey,
                epoch,
                ScoreOrAbsence::Score { value: *score },
            )
            .expect("leaf"),
        );
    }
    sort_leaves(&mut leaves);
    let merkle_root = compute_merkle_root(&leaves);
    let uid_map = uid_map_from_rows(&rows);
    let shares = trust.challenges.emission_shares();
    let verified: Vec<VerifiedLeaf> = leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    let final_vector = aggregate(&verified, &shares, &uid_map, AGG_V).expect("agg");
    let body = EpochBundleBodyV1 {
        protocol_version: PROTOCOL_VERSION,
        epoch,
        netuid: 1,
        block_b,
        block_hash,
        metagraph_root: compute_metagraph_root(&rows),
        algorithm_version: ALGORITHM_VERSION,
        emission_shares: shares,
        measurements_digest: trust.measurements_digest,
        uid_map,
        leaves,
        merkle_root,
        final_vector,
        gateway_hotkey: gpk,
    };
    (sign_bundle(gsk, body).expect("sign"), trust, chain)
}

async fn mount_bundle(gateway: &MockServer, epoch: u64, bytes: &[u8]) {
    Mock::given(method("GET"))
        .and(path(format!("/v1/bundle/{epoch}")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(bytes.to_vec()),
        )
        .mount(gateway)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/weights/latest"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(gateway)
        .await;
}

/// S1: A and B agree on root; C samples them → Agreed / Match gated pass.
#[tokio::test]
async fn s1_unanimous_peer_sample_pass() {
    let epoch = 11u64;
    let sk_a = sk(10);
    let sk_b = sk(11);
    let sk_c = sk(12);
    let hk_a = pk_of(&sk_a);
    let hk_b = pk_of(&sk_b);
    let hk_c = pk_of(&sk_c);

    let (bundle, trust, chain_inner) = valid_bundle(
        &sk(1),
        &sk(2),
        b"dummy",
        &[(hk_a, 50), (hk_b, 30), (hk_c, 20)],
        100,
        epoch,
    );
    let root = bundle.body.merkle_root;
    let bytes = bundle.encode_bytes();
    let chain = Arc::new(SyncChain::new(chain_inner));

    let gateway = MockServer::start().await;
    mount_bundle(&gateway, epoch, &bytes).await;

    let rt_a = ValidatorRuntime {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        gateway_endpoint: Some(gateway.uri()),
        root_signing_secret: Some(sk_a),
        own_hotkey: Some(hk_a.to_vec()),
        min_peer_sample: 1,
        peers: PeerBook::from_peers(vec![]).with_own_hotkey(hk_a.to_vec()),
        ..ValidatorRuntime::default()
    };
    let a = spawn_validator_with_ok_db(rt_a, Arc::clone(&chain))
        .await
        .expect("A");

    let rt_b = ValidatorRuntime {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        gateway_endpoint: Some(gateway.uri()),
        root_signing_secret: Some(sk_b),
        own_hotkey: Some(hk_b.to_vec()),
        min_peer_sample: 1,
        peers: PeerBook::from_peers(vec![]).with_own_hotkey(hk_b.to_vec()),
        ..ValidatorRuntime::default()
    };
    let b = spawn_validator_with_ok_db(rt_b, Arc::clone(&chain))
        .await
        .expect("B");

    tokio::time::sleep(Duration::from_millis(30)).await;

    let (out_a, _) = fetch_compare_and_crosscheck(
        &a.coordination,
        epoch,
        chain.as_ref(),
        &trust,
        &a.mirror,
        &a.peers,
        Some(ExpectedBundle { epoch, root }),
        &CrossCheckRun {
            root_store: &a.root_store,
            cfg: &CrossCheckConfig {
            min_peer_sample: 0,
            own_hotkey: Some(hk_a.to_vec()),
        },
            signing_secret: Some(&sk_a),
            own_hotkey_pk: Some(hk_a),
        },
        )
    .await;
    assert!(
        matches!(out_a, ComparisonOutcome::Match { .. }),
        "A: {out_a:?}"
    );
    assert!(a.root_store.get_local(epoch).is_some());

    let (out_b, _) = fetch_compare_and_crosscheck(
        &b.coordination,
        epoch,
        chain.as_ref(),
        &trust,
        &b.mirror,
        &b.peers,
        Some(ExpectedBundle { epoch, root }),
        &CrossCheckRun {
            root_store: &b.root_store,
            cfg: &CrossCheckConfig {
            min_peer_sample: 0,
            own_hotkey: Some(hk_b.to_vec()),
        },
            signing_secret: Some(&sk_b),
            own_hotkey_pk: Some(hk_b),
        },
        )
    .await;
    assert!(matches!(out_b, ComparisonOutcome::Match { .. }), "B: {out_b:?}");

    let url = format!("{}/v1/consensus/root/{epoch}", a.base_url());
    let resp = reqwest::get(&url).await.expect("get");
    assert_eq!(resp.status().as_u16(), 200);
    let body: crate::RootStatementJson = resp.json().await.expect("json");
    let stmt = body.into_statement().unwrap();
    stmt.verify().unwrap();
    assert_eq!(stmt.merkle_root, root);
    assert_eq!(stmt.hotkey, hk_a);

    let peers_c = PeerBook::from_peers(vec![
        PeerEndpoint {
            hotkey: hk_a.to_vec(),
            base_url: a.base_url(),
        },
        PeerEndpoint {
            hotkey: hk_b.to_vec(),
            base_url: b.base_url(),
        },
    ])
    .with_own_hotkey(hk_c.to_vec());

    let rt_c = ValidatorRuntime {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        gateway_endpoint: Some(gateway.uri()),
        root_signing_secret: Some(sk_c),
        own_hotkey: Some(hk_c.to_vec()),
        min_peer_sample: 1,
        peers: peers_c.clone(),
        ..ValidatorRuntime::default()
    };
    let c = spawn_validator_with_ok_db(rt_c, Arc::clone(&chain))
        .await
        .expect("C");

    let (out_c, cross_c) = fetch_compare_and_crosscheck(
        &c.coordination,
        epoch,
        chain.as_ref(),
        &trust,
        &c.mirror,
        &peers_c,
        Some(ExpectedBundle { epoch, root }),
        &CrossCheckRun {
            root_store: &c.root_store,
            cfg: &CrossCheckConfig {
            min_peer_sample: 1,
            own_hotkey: Some(hk_c.to_vec()),
        },
            signing_secret: Some(&sk_c),
            own_hotkey_pk: Some(hk_c),
        },
        )
    .await;
    assert!(
        matches!(out_c, ComparisonOutcome::Match { merkle_root: r, .. } if r == root),
        "C expected Match, got {out_c:?}"
    );
    match cross_c {
        CrossCheckOutcome::Agreed { sample_size, .. } => assert!(sample_size >= 1),
        other => panic!("expected Agreed, got {other:?}"),
    }

    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
    c.shutdown().await.unwrap();
}

/// S2: peers report a different root → fail closed (no Match).
#[tokio::test]
async fn s2_minority_disagree_fail_closed() {
    let epoch = 12u64;
    let sk_a = sk(20);
    let sk_c = sk(22);
    let hk_a = pk_of(&sk_a);
    let hk_c = pk_of(&sk_c);

    let (honest, trust, chain_inner) = valid_bundle(
        &sk(1),
        &sk(2),
        b"dummy",
        &[(hk_a, 50), (hk_c, 30)],
        100,
        epoch,
    );
    let (evil, _, _) = valid_bundle(
        &sk(1),
        &sk(2),
        b"dummy",
        &[(hk_a, 10), (hk_c, 90)],
        100,
        epoch,
    );
    assert_ne!(honest.body.merkle_root, evil.body.merkle_root);
    let honest_root = honest.body.merkle_root;
    let evil_root = evil.body.merkle_root;
    let chain = Arc::new(SyncChain::new(chain_inner));

    let rt_a = ValidatorRuntime {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        gateway_endpoint: None,
        root_signing_secret: Some(sk_a),
        own_hotkey: Some(hk_a.to_vec()),
        peers: PeerBook::from_peers(vec![]).with_own_hotkey(hk_a.to_vec()),
        ..ValidatorRuntime::default()
    };
    let a = spawn_validator_with_ok_db(rt_a, Arc::clone(&chain))
        .await
        .expect("A");
    a.root_store
        .put_local(SignedRootStatement::sign(&sk_a, hk_a, epoch, honest_root).unwrap());

    let gateway = MockServer::start().await;
    mount_bundle(&gateway, epoch, &evil.encode_bytes()).await;

    let peers_c = PeerBook::from_peers(vec![PeerEndpoint {
        hotkey: hk_a.to_vec(),
        base_url: a.base_url(),
    }])
    .with_own_hotkey(hk_c.to_vec());

    let client = crate::CoordinationClient::new(Some(gateway.uri())).unwrap();
    let store = crate::MemoryMirrorStore::shared();
    let root_store = crate::RootStatementStore::shared();
    let (out, cross) = fetch_compare_and_crosscheck(
        &client,
        epoch,
        chain.as_ref(),
        &trust,
        &store,
        &peers_c,
        Some(ExpectedBundle {
            epoch,
            root: evil_root,
        }),
        &CrossCheckRun {
            root_store: &root_store,
            cfg: &CrossCheckConfig {
                min_peer_sample: 1,
                own_hotkey: Some(hk_c.to_vec()),
            },
            signing_secret: Some(&sk_c),
            own_hotkey_pk: Some(hk_c),
        },
    )
    .await;

    assert!(
        matches!(
            out,
            ComparisonOutcome::NoSubmission {
                reason: NoSubmissionReason::PeerRootDisagreement { .. }
            }
        ),
        "expected PeerRootDisagreement gate, got {out:?}"
    );
    assert!(cross.is_cross_check_failed());
    assert!(!cross.allows_submission());
    let ev = root_store.evidence_for_epoch(epoch);
    assert!(ev.iter().any(|s| s.merkle_root == honest_root));
    assert!(ev.iter().any(|s| s.merkle_root == evil_root));

    a.shutdown().await.unwrap();
}

/// S3: eclipse — others exist on metagraph but zero reachable peers → Degraded.
#[tokio::test]
async fn s3_eclipse_insufficient_peers_fail_closed() {
    let epoch = 13u64;
    let sk_c = sk(30);
    let sk_other = sk(31);
    let hk_c = pk_of(&sk_c);
    let hk_other = pk_of(&sk_other);

    let (bundle, trust, chain_inner) =
        valid_bundle(&sk(1), &sk(2), b"dummy", &[(hk_c, 50), (hk_other, 30)], 100, epoch);
    let root = bundle.body.merkle_root;
    let chain = Arc::new(SyncChain::new(chain_inner));

    let gateway = MockServer::start().await;
    mount_bundle(&gateway, epoch, &bundle.encode_bytes()).await;

    let peers = PeerBook::from_peers(vec![PeerEndpoint {
        hotkey: hk_other.to_vec(),
        base_url: "http://127.0.0.1:1".into(),
    }])
    .with_own_hotkey(hk_c.to_vec());

    let client = crate::CoordinationClient::new(Some(gateway.uri())).unwrap();
    let store = crate::MemoryMirrorStore::shared();
    let root_store = crate::RootStatementStore::shared();
    let (out, cross) = fetch_compare_and_crosscheck(
        &client,
        epoch,
        chain.as_ref(),
        &trust,
        &store,
        &peers,
        Some(ExpectedBundle { epoch, root }),
        &CrossCheckRun {
            root_store: &root_store,
            cfg: &CrossCheckConfig {
                min_peer_sample: 1,
                own_hotkey: Some(hk_c.to_vec()),
            },
            signing_secret: Some(&sk_c),
            own_hotkey_pk: Some(hk_c),
        },
    )
    .await;

    assert!(
        matches!(
            out,
            ComparisonOutcome::NoSubmission {
                reason: NoSubmissionReason::PeerSampleInsufficient {
                    reachable: 0,
                    required: 1
                }
            }
        ),
        "got {out:?}"
    );
    match cross {
        CrossCheckOutcome::Failed {
            kind: CrossCheckFailKind::InsufficientPeers { .. },
            ..
        } => {}
        other => panic!("expected InsufficientPeers, got {other:?}"),
    }
}

/// S4: self is excluded — only own statement does not satisfy min_sample.
#[tokio::test]
async fn s4_self_excluded_from_sample() {
    let epoch = 14u64;
    let sk_a = sk(40);
    let sk_other = sk(41);
    let hk_a = pk_of(&sk_a);
    let hk_other = pk_of(&sk_other);

    let (bundle, trust, chain_inner) =
        valid_bundle(&sk(1), &sk(2), b"dummy", &[(hk_a, 50), (hk_other, 30)], 100, epoch);
    let root = bundle.body.merkle_root;
    let chain = Arc::new(SyncChain::new(chain_inner));

    let gateway = MockServer::start().await;
    mount_bundle(&gateway, epoch, &bundle.encode_bytes()).await;

    let peers = PeerBook::from_peers(vec![PeerEndpoint {
        hotkey: hk_a.to_vec(),
        base_url: "http://127.0.0.1:9".into(),
    }])
    .with_own_hotkey(hk_a.to_vec());
    assert!(peers
        .discover_from_chain(chain.as_ref())
        .unwrap()
        .is_empty());

    let client = crate::CoordinationClient::new(Some(gateway.uri())).unwrap();
    let store = crate::MemoryMirrorStore::shared();
    let root_store = crate::RootStatementStore::shared();
    let (out, _) = fetch_compare_and_crosscheck(
        &client,
        epoch,
        chain.as_ref(),
        &trust,
        &store,
        &peers,
        Some(ExpectedBundle { epoch, root }),
        &CrossCheckRun {
            root_store: &root_store,
            cfg: &CrossCheckConfig {
                min_peer_sample: 1,
                own_hotkey: Some(hk_a.to_vec()),
            },
            signing_secret: Some(&sk_a),
            own_hotkey_pk: Some(hk_a),
        },
    )
    .await;
    assert!(
        matches!(
            out,
            ComparisonOutcome::NoSubmission {
                reason: NoSubmissionReason::PeerSampleInsufficient { .. }
            }
        ),
        "self must not count; got {out:?}"
    );
}

/// S5: single-validator metagraph → exempt, Match allowed without peers.
#[tokio::test]
async fn s5_single_validator_exempt() {
    let epoch = 15u64;
    let sk_a = sk(50);
    let hk_a = pk_of(&sk_a);

    let (bundle, trust, chain_inner) =
        valid_bundle(&sk(1), &sk(2), b"dummy", &[(hk_a, 50)], 100, epoch);
    let root = bundle.body.merkle_root;
    let chain = Arc::new(SyncChain::new(chain_inner));

    let gateway = MockServer::start().await;
    mount_bundle(&gateway, epoch, &bundle.encode_bytes()).await;

    let client = crate::CoordinationClient::new(Some(gateway.uri())).unwrap();
    let store = crate::MemoryMirrorStore::shared();
    let root_store = crate::RootStatementStore::shared();
    let peers = PeerBook::from_peers(vec![]).with_own_hotkey(hk_a.to_vec());
    let (out, cross) = fetch_compare_and_crosscheck(
        &client,
        epoch,
        chain.as_ref(),
        &trust,
        &store,
        &peers,
        Some(ExpectedBundle { epoch, root }),
        &CrossCheckRun {
            root_store: &root_store,
            cfg: &CrossCheckConfig {
                min_peer_sample: 1,
                own_hotkey: Some(hk_a.to_vec()),
            },
            signing_secret: Some(&sk_a),
            own_hotkey_pk: Some(hk_a),
        },
    )
    .await;
    assert!(
        matches!(out, ComparisonOutcome::Match { .. }),
        "single-validator must Match, got {out:?}"
    );
    assert!(matches!(
        cross,
        CrossCheckOutcome::SingleValidatorExempt { .. }
    ));
}
