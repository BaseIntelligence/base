#![allow(clippy::too_many_lines, clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
//! Task 32 VERIFY: three-outcome policy, quarantine (multi-drop), dissent serve.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use gbase_aggregate::{
    aggregate, renormalize_after_quarantine, ScoreOrAbsence as AggScore, VerifiedLeaf,
    ALGORITHM_VERSION as AGG_V,
};
use gbase_bundle::{
    compute_merkle_root, compute_metagraph_root, make_signed_leaf, metagraph_rows_from_chain,
    sign_bundle, sort_leaves, uid_map_from_rows, EpochBundleBodyV1, EpochBundleV1, LocalTrustRoot,
    ScoreOrAbsence, ALGORITHM_VERSION, PROTOCOL_VERSION,
};
use gbase_chain::{ChainClient, FakeChain, FakeChainConfig};
use gbase_crosscheck::{CrossCheckFailKind, CrossCheckOutcome};
use gbase_crypto::secret_from_bytes;
use gbase_dissent::{
    apply_three_outcome_policy, DissentReasonCode, DissentSigner, DissentStore, EpochDecision,
    SubmissionSource,
};
use gbase_trustroot::{
    measurements_digest, ChallengeEntry, ChallengesBody, MeasurementsBody, ParticipantPolicy,
};
use sha2::{Digest, Sha256};

use crate::{
    recompute_view_from_comparison, spawn_validator_with_ok_db, compare_bundle,
    ComparisonOutcome, SyncChain, ValidatorRuntime,
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

/// Multi-challenge bundle. `challenges`: (id, challenge_sk, share_bps, score_per_miner).
fn multi_challenge_bundle(
    challenges: &[(&[u8], [u8; 32], u16, u64)],
    gsk: &[u8; 32],
    miners: &[[u8; 32]],
    block_b: u64,
    epoch: u64,
    // Challenge ids whose leaves are signed with a wrong key (bad).
    bad_challenge_ids: &[&[u8]],
) -> (EpochBundleV1, LocalTrustRoot, FakeChain) {
    let gpk = pk_of(gsk);
    let entries: Vec<ChallengeEntry> = challenges
        .iter()
        .map(|(id, csk, bps, _)| ChallengeEntry {
            id: id.to_vec(),
            public_key: pk_of(csk),
            emission_share_bps: *bps,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        })
        .collect();
    let trust = LocalTrustRoot {
        challenges: ChallengesBody {
            challenges: entries,
        },
        measurements_digest: measurements_digest(&MeasurementsBody::default()),
    };
    let chain = FakeChain::new(FakeChainConfig {
        current_block: block_b.max(10),
        hotkeys: miners.iter().map(|h| h.to_vec()).collect(),
        owner_hotkey: vec![0xA1; 32],
        ..FakeChainConfig::default()
    });
    let block_hash = chain.block_hash(block_b).expect("hash");
    let rows = metagraph_rows_from_chain(
        &miners.iter().map(|h| h.to_vec()).collect::<Vec<_>>(),
        None,
    )
    .expect("rows");
    let mut leaves = Vec::new();
    let wrong_sk = sk(0xEE);
    for (id, csk, _, score) in challenges {
        let sign_sk = if bad_challenge_ids.iter().any(|b| b == id) {
            &wrong_sk
        } else {
            csk
        };
        for hotkey in miners {
            leaves.push(
                make_signed_leaf(
                    sign_sk,
                    id,
                    *hotkey,
                    epoch,
                    ScoreOrAbsence::Score { value: *score },
                )
                .expect("leaf"),
            );
        }
    }
    sort_leaves(&mut leaves);
    let merkle_root = compute_merkle_root(&leaves);
    let uid_map = uid_map_from_rows(&rows);
    let shares = trust.challenges.emission_shares();
    // Gateway final_vector: honest aggregate of *all* leaves as if signatures were good
    // (for bad leaves we still put scores in the body; verify will fail on sig).
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

fn single_challenge_honest(
    epoch: u64,
) -> (EpochBundleV1, LocalTrustRoot, FakeChain, [u8; 32], [u8; 32]) {
    let csk = sk(1);
    let gsk = sk(2);
    let m1 = pk_of(&sk(10));
    let m2 = pk_of(&sk(11));
    let (bundle, trust, chain) = multi_challenge_bundle(
        &[(b"ch-a", csk, 10_000, 100)],
        &gsk,
        &[m1, m2],
        100,
        epoch,
        &[],
    );
    (bundle, trust, chain, csk, gsk)
}

fn agreed_cross(epoch: u64, root: [u8; 32]) -> CrossCheckOutcome {
    CrossCheckOutcome::Agreed {
        epoch,
        merkle_root: root,
        sample_size: 1,
        statements: vec![],
    }
}

/// S1 — class A: VectorMismatch → exactly one submission = local recompute + one dissent.
#[test]
fn s1_class_a_one_submission_and_dissent() {
    let epoch = 32;
    let (mut bundle, trust, chain, _csk, gsk) = single_challenge_honest(epoch);
    // Tamper final_vector only (inputs still verify via FinalVectorMismatch path).
    bundle.body.final_vector = vec![(0, 1), (1, 2)];
    // Re-sign body after tamper.
    let body = bundle.body.clone();
    let bundle = sign_bundle(&gsk, body).expect("resign");

    let comparison = compare_bundle(&bundle, &chain, &trust);
    assert!(
        matches!(comparison, ComparisonOutcome::VectorMismatch { .. }),
        "{comparison:?}"
    );
    let local = match &comparison {
        ComparisonOutcome::VectorMismatch { local_vector, .. } => local_vector.clone(),
        _ => unreachable!(),
    };
    let root = bundle.body.merkle_root;
    let view = recompute_view_from_comparison(&comparison);
    let secret = sk(99);
    let hotkey = pk_of(&secret);
    let store = DissentStore::new();
    let decision = apply_three_outcome_policy(
        &view,
        &agreed_cross(epoch, root),
        Some(&bundle),
        &chain,
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey,
        },
        Some(&store),
    )
    .expect("policy");

    match decision {
        EpochDecision::ClassA { ref intent, ref dissent } => {
            assert_eq!(intent.vector, local);
            assert_eq!(intent.source, SubmissionSource::Recompute);
            assert_eq!(decision.submissions().len(), 1);
            dissent.verify().expect("dissent verify");
            assert_eq!(
                dissent.reason_code(),
                Some(DissentReasonCode::VectorMismatch)
            );
            assert!(DissentReasonCode::ALL.contains(&dissent.reason_code().unwrap()));
        }
        other => panic!("expected ClassA, got {other:?}"),
    }
    assert_eq!(store.count_epoch(epoch), 1);
}

/// S2 — class B: peer root conflict → zero submissions + dissent + metric path.
#[test]
fn s2_class_b_zero_submissions_dissent() {
    let epoch = 33;
    let (bundle, trust, chain, _, _) = single_challenge_honest(epoch);
    let comparison = compare_bundle(&bundle, &chain, &trust);
    assert!(matches!(comparison, ComparisonOutcome::Match { .. }));
    let view = recompute_view_from_comparison(&comparison);
    let secret = sk(98);
    let hotkey = pk_of(&secret);
    let store = DissentStore::new();
    let cross = CrossCheckOutcome::Failed {
        epoch,
        kind: CrossCheckFailKind::RootDisagreement {
            candidate: bundle.body.merkle_root,
        },
        statements: vec![],
    };
    let decision = apply_three_outcome_policy(
        &view,
        &cross,
        Some(&bundle),
        &chain,
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey,
        },
        Some(&store),
    )
    .expect("policy");

    assert!(decision.submissions().is_empty());
    match decision {
        EpochDecision::ClassB { ref dissent, reason } => {
            dissent.verify().expect("verify");
            assert_eq!(reason, DissentReasonCode::PeerRootConflict);
            assert_eq!(dissent.reason_code(), Some(DissentReasonCode::PeerRootConflict));
        }
        other => panic!("expected ClassB, got {other:?}"),
    }
    assert_eq!(store.count_epoch(epoch), 1);
}

/// S3 — quarantine one bad challenge, 60% surviving → submit without it.
#[test]
fn s3_quarantine_one_bad_60pct_submits() {
    let epoch = 34;
    let gsk = sk(2);
    let c_good = sk(3);
    let c_bad = sk(4);
    let m1 = pk_of(&sk(20));
    let m2 = pk_of(&sk(21));
    // 6000 + 4000 = 10000; drop 4000 → 60% survive.
    let (bundle, trust, chain) = multi_challenge_bundle(
        &[
            (b"good", c_good, 6000, 50),
            (b"bad", c_bad, 4000, 50),
        ],
        &gsk,
        &[m1, m2],
        200,
        epoch,
        &[b"bad"],
    );
    let comparison = compare_bundle(&bundle, &chain, &trust);
    assert!(
        matches!(comparison, ComparisonOutcome::InputInvalid { .. }),
        "{comparison:?}"
    );
    let view = recompute_view_from_comparison(&comparison);
    let secret = sk(97);
    let hotkey = pk_of(&secret);
    let store = DissentStore::new();
    let decision = apply_three_outcome_policy(
        &view,
        &agreed_cross(epoch, bundle.body.merkle_root),
        Some(&bundle),
        &chain,
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey,
        },
        Some(&store),
    )
    .expect("policy");

    match decision {
        EpochDecision::Quarantine {
            ref intent,
            ref dropped,
            surviving_mass_bps,
        } => {
            assert_eq!(surviving_mass_bps, 6000);
            assert_eq!(dropped, &vec![b"bad".to_vec()]);
            assert_eq!(intent.source, SubmissionSource::Quarantine);
            assert_eq!(decision.submissions().len(), 1);
            // Expected vector: only good challenge leaves, renormalized shares.
            let new_shares =
                renormalize_after_quarantine(&bundle.body.emission_shares, dropped).unwrap();
            let verified: Vec<VerifiedLeaf> = bundle
                .body
                .leaves
                .iter()
                .filter(|l| l.challenge_id == b"good")
                .map(|l| VerifiedLeaf {
                    challenge_id: l.challenge_id.clone(),
                    miner_hotkey: l.miner_hotkey,
                    score_or_absence: to_agg(&l.score_or_absence),
                })
                .collect();
            let expected =
                aggregate(&verified, &new_shares, &bundle.body.uid_map, AGG_V).unwrap();
            assert_eq!(intent.vector, expected);
        }
        other => panic!("expected Quarantine, got {other:?}"),
    }
}

/// S4 — quarantine **two** bad challenges, 55% surviving → still submits (Momus multi-drop).
#[test]
fn s4_quarantine_two_bad_55pct_still_submits() {
    let epoch = 35;
    let gsk = sk(2);
    let c_good = sk(5);
    let c_b1 = sk(6);
    let c_b2 = sk(7);
    let m1 = pk_of(&sk(30));
    // 5500 + 2500 + 2000; drop two → 55% survive (>= 5000 default floor).
    let (bundle, trust, chain) = multi_challenge_bundle(
        &[
            (b"keep", c_good, 5500, 80),
            (b"drop1", c_b1, 2500, 80),
            (b"drop2", c_b2, 2000, 80),
        ],
        &gsk,
        &[m1],
        300,
        epoch,
        &[b"drop1", b"drop2"],
    );
    let comparison = compare_bundle(&bundle, &chain, &trust);
    assert!(matches!(
        comparison,
        ComparisonOutcome::InputInvalid { .. }
    ));
    let view = recompute_view_from_comparison(&comparison);
    let secret = sk(96);
    let hotkey = pk_of(&secret);
    let decision = apply_three_outcome_policy(
        &view,
        &agreed_cross(epoch, bundle.body.merkle_root),
        Some(&bundle),
        &chain,
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey,
        },
        None,
    )
    .expect("policy");

    match decision {
        EpochDecision::Quarantine {
            ref intent,
            ref dropped,
            surviving_mass_bps,
        } => {
            assert_eq!(surviving_mass_bps, 5500);
            assert_eq!(dropped.len(), 2);
            assert!(dropped.contains(&b"drop1".to_vec()));
            assert!(dropped.contains(&b"drop2".to_vec()));
            assert_eq!(decision.submissions().len(), 1);
            assert!(!intent.vector.is_empty());
        }
        other => panic!("expected Quarantine multi-drop, got {other:?}"),
    }
}

/// S5 — 30% surviving → class B, zero submissions.
#[test]
fn s5_quarantine_30pct_escalates_class_b() {
    let epoch = 36;
    let gsk = sk(2);
    let c_good = sk(8);
    let c_bad = sk(9);
    let m1 = pk_of(&sk(40));
    // 3000 + 7000; drop 7000 → 30% < 5000 floor.
    let (bundle, trust, chain) = multi_challenge_bundle(
        &[
            (b"tiny", c_good, 3000, 10),
            (b"huge-bad", c_bad, 7000, 10),
        ],
        &gsk,
        &[m1],
        400,
        epoch,
        &[b"huge-bad"],
    );
    let comparison = compare_bundle(&bundle, &chain, &trust);
    let view = recompute_view_from_comparison(&comparison);
    let secret = sk(95);
    let hotkey = pk_of(&secret);
    let store = DissentStore::new();
    let decision = apply_three_outcome_policy(
        &view,
        &agreed_cross(epoch, bundle.body.merkle_root),
        Some(&bundle),
        &chain,
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey,
        },
        Some(&store),
    )
    .expect("policy");

    assert!(decision.submissions().is_empty());
    match decision {
        EpochDecision::ClassB { ref dissent, reason } => {
            dissent.verify().expect("verify");
            assert_eq!(reason, DissentReasonCode::ShareMassBelowThreshold);
            assert!(DissentReasonCode::from_u8(dissent.body.reason_code).is_some());
        }
        other => panic!("expected ClassB mass, got {other:?}"),
    }
}

/// S6 — every produced DissentV1 verifies; reason in enum; HTTP serve.
#[tokio::test]
async fn s6_dissent_verifies_and_http_serves() {
    let epoch = 37;
    let (mut bundle, trust, chain, _, gsk) = single_challenge_honest(epoch);
    bundle.body.final_vector = vec![(0, 9)];
    let bundle = sign_bundle(&gsk, bundle.body.clone()).expect("resign");
    let comparison = compare_bundle(&bundle, &chain, &trust);
    let view = recompute_view_from_comparison(&comparison);
    let secret = sk(94);
    let hotkey = pk_of(&secret);
    let store = DissentStore::shared();
    let decision = apply_three_outcome_policy(
        &view,
        &agreed_cross(epoch, bundle.body.merkle_root),
        Some(&bundle),
        &chain,
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey,
        },
        Some(store.as_ref()),
    )
    .expect("policy");
    let d = decision.dissent().expect("dissent");
    d.verify().expect("verify");
    assert!(d.reason_code().unwrap().is_spec_enum());

    let runtime = ValidatorRuntime {
        listen_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
        dissent_store: Some(Arc::clone(&store)),
        root_signing_secret: Some(secret),
        own_hotkey: Some(hotkey.to_vec()),
        ..ValidatorRuntime::default()
    };
    let chain = Arc::new(SyncChain::new(chain));
    let v = spawn_validator_with_ok_db(runtime, chain).await.expect("spawn");
    let url = format!("{}/v1/dissent/{epoch}", v.base_url());
    let client = reqwest::Client::new();
    let mut body = None;
    for _ in 0..20 {
        let resp = client.get(&url).send().await.expect("get");
        if resp.status().is_success() {
            body = Some(resp.text().await.expect("text"));
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let body = body.expect("dissent HTTP 200");
    assert!(body.contains("reason_code"), "{body}");
    assert!(body.contains(&format!("\"epoch\":{epoch}")), "{body}");
    v.shutdown().await.expect("shutdown");
}

/// Match path: one submission, no dissent.
#[test]
fn s0_match_submits_no_dissent() {
    let epoch = 31;
    let (bundle, trust, chain, _, _) = single_challenge_honest(epoch);
    let comparison = compare_bundle(&bundle, &chain, &trust);
    let view = recompute_view_from_comparison(&comparison);
    let secret = sk(93);
    let hotkey = pk_of(&secret);
    let decision = apply_three_outcome_policy(
        &view,
        &agreed_cross(epoch, bundle.body.merkle_root),
        Some(&bundle),
        &chain,
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey,
        },
        None,
    )
    .expect("policy");
    assert!(matches!(decision, EpochDecision::Match { .. }));
    assert_eq!(decision.submissions().len(), 1);
    assert!(decision.dissent().is_none());
}
