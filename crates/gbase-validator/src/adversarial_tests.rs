#![allow(
    clippy::too_many_lines,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::pedantic,
    clippy::cognitive_complexity
)]
//! Task 48 adversarial suite — offline proofs via FakeChain + in-process HTTP.
//!
//! Each `a48_*` test maps 1:1 to a VERIFY scenario in the plan. Live staging
//! (real TAO / droplets) remains PENDING_LIVE behind task 47; these tests
//! exercise the same code paths without network.

use std::sync::Arc;
use std::time::Duration;

use gbase_aggregate::{
    aggregate, renormalize_after_quarantine, ScoreOrAbsence as AggScore, VerifiedLeaf,
    ALGORITHM_VERSION as AGG_V,
};
use gbase_bundle::{
    compute_merkle_root, compute_metagraph_root, finalize_body_merkle, make_signed_leaf,
    metagraph_rows_from_chain, sign_bundle, sort_leaves, uid_map_from_rows, BundleError,
    EpochBundleBodyV1, EpochBundleV1, LocalTrustRoot, ScoreOrAbsence, ALGORITHM_VERSION,
    PROTOCOL_VERSION,
};
use gbase_chain::{ChainClient, FakeChain, FakeChainConfig};
use gbase_crosscheck::{CrossCheckFailKind, CrossCheckOutcome};
use gbase_crypto::secret_from_bytes;
use gbase_dissent::{
    apply_three_outcome_policy, DissentReasonCode, DissentSigner, EpochDecision, SubmissionSource,
};
use gbase_submit::{
    submit_intent, FailingDrand, FixedClock, SubmitConfig, SubmitOutcome, MECID_MAIN,
};
use gbase_trustroot::{
    measurements_digest, ChallengeEntry, ChallengesBody, MeasurementsBody, ParticipantPolicy,
};
use sha2::{Digest, Sha256};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::{
    compare_bundle, fetch_compare_and_crosscheck, recompute_view_from_comparison,
    spawn_validator_with_ok_db, ComparisonOutcome, CrossCheckConfig, CrossCheckRun, ExpectedBundle,
    NoSubmissionReason, PeerBook, PeerEndpoint, SignedRootStatement, SyncChain, ValidatorRuntime,
};

fn sk(tag: u8) -> [u8; 32] {
    let dig = Sha256::digest([0x48, tag, 0xA5, tag.wrapping_add(1)]);
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

fn agreed_cross(epoch: u64, root: [u8; 32]) -> CrossCheckOutcome {
    CrossCheckOutcome::Agreed {
        epoch,
        merkle_root: root,
        sample_size: 2,
        statements: vec![],
    }
}

/// Multi-challenge bundle builder. `bad_challenge_ids` get leaves signed with a wrong key.
fn multi_challenge_bundle(
    challenges: &[(&[u8], [u8; 32], u16, u64)],
    gsk: &[u8; 32],
    miners: &[[u8; 32]],
    block_b: u64,
    epoch: u64,
    bad_challenge_ids: &[&[u8]],
    policies: Option<&[ParticipantPolicy]>,
) -> (EpochBundleV1, LocalTrustRoot, FakeChain) {
    let gpk = pk_of(gsk);
    let entries: Vec<ChallengeEntry> = challenges
        .iter()
        .enumerate()
        .map(|(i, (id, csk, bps, _))| ChallengeEntry {
            id: id.to_vec(),
            public_key: pk_of(csk),
            emission_share_bps: *bps,
            policy: policies
                .and_then(|p| p.get(i).cloned())
                .unwrap_or(ParticipantPolicy::AllMetagraphHotkeys),
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
    let rows =
        metagraph_rows_from_chain(&miners.iter().map(|h| h.to_vec()).collect::<Vec<_>>(), None)
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

fn single_honest(epoch: u64) -> (EpochBundleV1, LocalTrustRoot, FakeChain, [u8; 32]) {
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
        None,
    );
    (bundle, trust, chain, gsk)
}

fn decide(
    view: &gbase_dissent::RecomputeView,
    cross: &CrossCheckOutcome,
    bundle: &EpochBundleV1,
    chain: &FakeChain,
    trust: &LocalTrustRoot,
    tag: u8,
) -> EpochDecision {
    let secret = sk(tag);
    let hotkey = pk_of(&secret);
    apply_three_outcome_policy(
        view,
        cross,
        Some(bundle),
        chain,
        trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey,
        },
        None,
    )
    .expect("policy")
}

// ─── (i) inconsistent gateway vector → Class A, identical local vectors ───

/// (i) Gateway publishes vector ≠ honest recompute. Two independent validators
/// both Class A with **byte-identical** local vectors ≠ gateway.
#[test]
fn a48_i_inconsistent_vector_class_a_identical_locals() {
    let epoch = 4801;
    let (mut bundle, trust, chain, gsk) = single_honest(epoch);
    let honest = bundle.body.final_vector.clone();
    bundle.body.final_vector = vec![(0, 1), (1, 9999)];
    let bundle = sign_bundle(&gsk, bundle.body.clone()).expect("resign");

    let c_a = compare_bundle(&bundle, &chain, &trust);
    let c_b = compare_bundle(&bundle, &chain, &trust);
    let (local_a, gw_a) = match (&c_a, &c_b) {
        (
            ComparisonOutcome::VectorMismatch {
                local_vector: la,
                gateway_vector: ga,
                ..
            },
            ComparisonOutcome::VectorMismatch {
                local_vector: lb,
                gateway_vector: gb,
                ..
            },
        ) => {
            assert_eq!(la, lb, "validators must recompute identical locals");
            assert_eq!(la, &honest);
            assert_ne!(la, ga);
            assert_eq!(ga, gb);
            (la.clone(), ga.clone())
        }
        _ => panic!("expected VectorMismatch x2: {c_a:?} / {c_b:?}"),
    };

    let d_a = decide(
        &recompute_view_from_comparison(&c_a),
        &agreed_cross(epoch, bundle.body.merkle_root),
        &bundle,
        &chain,
        &trust,
        90,
    );
    let d_b = decide(
        &recompute_view_from_comparison(&c_b),
        &agreed_cross(epoch, bundle.body.merkle_root),
        &bundle,
        &chain,
        &trust,
        91,
    );
    match (&d_a, &d_b) {
        (
            EpochDecision::ClassA {
                intent: ia,
                dissent: da,
            },
            EpochDecision::ClassA {
                intent: ib,
                dissent: db,
            },
        ) => {
            assert_eq!(ia.vector, ib.vector);
            assert_eq!(ia.vector, local_a);
            assert_ne!(ia.vector, gw_a);
            assert_eq!(ia.source, SubmissionSource::Recompute);
            assert_eq!(d_a.submissions().len(), 1);
            assert_eq!(d_b.submissions().len(), 1);
            da.verify().unwrap();
            db.verify().unwrap();
            assert_eq!(da.reason_code(), Some(DissentReasonCode::VectorMismatch));
        }
        other => panic!("expected ClassA x2, got {other:?}"),
    }
    assert!(chain.submissions().is_empty());
}

// ─── (ii) different bundles A/B → conflicting roots, both signed, Class B ───

/// (ii) Gateway serves A and B different bundles → root disagreement, both
/// signed statements persisted, Class B, zero submissions.
#[tokio::test]
async fn a48_ii_equivocation_class_b_zero_submissions() {
    let epoch = 4802u64;
    let sk_a = sk(20);
    let sk_c = sk(22);
    let hk_a = pk_of(&sk_a);
    let hk_c = pk_of(&sk_c);

    let (honest, trust, chain_inner) = multi_challenge_bundle(
        &[(b"dummy", sk(1), 10_000, 50)],
        &sk(2),
        &[hk_a, hk_c],
        100,
        epoch,
        &[],
        None,
    );
    // Different scores → different merkle root (equivocation payload).
    let (evil, _, _) = multi_challenge_bundle(
        &[(b"dummy", sk(1), 10_000, 10)],
        &sk(2),
        &[hk_a, hk_c],
        100,
        epoch,
        &[],
        None,
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
    Mock::given(method("GET"))
        .and(path(format!("/v1/bundle/{epoch}")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(evil.encode_bytes()),
        )
        .mount(&gateway)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/weights/latest"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&gateway)
        .await;

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
        "got {out:?}"
    );
    assert!(!cross.allows_submission());
    let ev = root_store.evidence_for_epoch(epoch);
    assert!(
        ev.iter().any(|s| s.merkle_root == honest_root),
        "honest root statement missing"
    );
    assert!(
        ev.iter().any(|s| s.merkle_root == evil_root),
        "evil root statement missing"
    );
    for s in &ev {
        s.verify().expect("signed statement must verify");
    }

    // Policy path: Class B, zero submissions.
    let view = recompute_view_from_comparison(&out);
    let secret = sk(88);
    let decision = apply_three_outcome_policy(
        &view,
        &cross,
        Some(&evil),
        chain.as_ref(),
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey: pk_of(&secret),
        },
        None,
    )
    .expect("policy");
    assert!(decision.submissions().is_empty());
    match decision {
        EpochDecision::ClassB { reason, dissent } => {
            assert_eq!(reason, DissentReasonCode::PeerRootConflict);
            dissent.verify().unwrap();
        }
        other => panic!("expected ClassB, got {other:?}"),
    }

    a.shutdown().await.unwrap();
}

// ─── (iii) invalid challenge signature → Class B at signature check ───

/// (iii) Forged leaf with invalid challenge signature → InputInvalid then Class B
/// (single-challenge quarantine exhausts).
#[test]
fn a48_iii_invalid_challenge_sig_class_b() {
    let epoch = 4803;
    let (mut bundle, trust, chain, gsk) = single_honest(epoch);
    bundle.body.leaves[0].challenge_sig[0] ^= 0xff;
    finalize_body_merkle(&mut bundle.body);
    let bundle = sign_bundle(&gsk, bundle.body).unwrap();

    let comparison = compare_bundle(&bundle, &chain, &trust);
    match &comparison {
        ComparisonOutcome::InputInvalid {
            error: BundleError::LeafSignatureInvalid,
        } => {}
        other => panic!("expected LeafSignatureInvalid, got {other:?}"),
    }
    let decision = decide(
        &recompute_view_from_comparison(&comparison),
        &agreed_cross(epoch, bundle.body.merkle_root),
        &bundle,
        &chain,
        &trust,
        87,
    );
    assert!(decision.submissions().is_empty());
    match decision {
        EpochDecision::ClassB { reason, dissent } => {
            assert!(
                matches!(
                    reason,
                    DissentReasonCode::LeafSignatureInvalid
                        | DissentReasonCode::QuarantineExhausted
                ),
                "got {reason:?}"
            );
            dissent.verify().unwrap();
        }
        other => panic!("expected ClassB, got {other:?}"),
    }
}

// ─── (iv) invented key absent from local trust root (D18) ───

/// (iv) Leaf signed by a key invented only on the gateway side — validator
/// verifies against **local** trust root public key → reject (D18).
#[test]
fn a48_iv_invented_key_absent_from_local_trust_d18() {
    let epoch = 4804;
    let csk_real = sk(1);
    let csk_invented = sk(0xD1); // gateway-only key, never in local trust
    let gsk = sk(2);
    let m1 = pk_of(&sk(10));
    // Bundle leaves signed with invented key; trust root holds real key.
    let (mut bundle, trust, chain) = multi_challenge_bundle(
        &[(b"ch-a", csk_real, 10_000, 100)],
        &gsk,
        &[m1],
        100,
        epoch,
        &[],
        None,
    );
    assert_eq!(trust.challenges.challenges[0].public_key, pk_of(&csk_real));
    assert_ne!(pk_of(&csk_real), pk_of(&csk_invented));

    // Replace leaves with ones signed by invented key (same challenge id).
    let leaf = make_signed_leaf(
        &csk_invented,
        b"ch-a",
        m1,
        epoch,
        ScoreOrAbsence::Score { value: 100 },
    )
    .unwrap();
    bundle.body.leaves = vec![leaf];
    finalize_body_merkle(&mut bundle.body);
    let verified: Vec<VerifiedLeaf> = bundle
        .body
        .leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    bundle.body.final_vector = aggregate(
        &verified,
        &bundle.body.emission_shares,
        &bundle.body.uid_map,
        AGG_V,
    )
    .unwrap();
    let bundle = sign_bundle(&gsk, bundle.body).unwrap();

    let comparison = compare_bundle(&bundle, &chain, &trust);
    match &comparison {
        ComparisonOutcome::InputInvalid {
            error: BundleError::LeafSignatureInvalid,
        } => {
            // Local trust root key used for verify — invented gateway key never consulted (D18).
        }
        other => panic!("D18: expected LeafSignatureInvalid vs local trust key, got {other:?}"),
    }

    // Unknown challenge id path (key/id absent entirely from local trust).
    let (mut b2, trust2, chain2, gsk2) = single_honest(epoch + 1);
    let ghost = make_signed_leaf(
        &csk_invented,
        b"gateway-only-challenge",
        m1,
        epoch + 1,
        ScoreOrAbsence::Score { value: 1 },
    )
    .unwrap();
    b2.body.leaves = vec![ghost];
    // Keep emission_shares matching trust so we fail on key unknown, not share mismatch.
    finalize_body_merkle(&mut b2.body);
    let b2 = sign_bundle(&gsk2, b2.body).unwrap();
    let c2 = compare_bundle(&b2, &chain2, &trust2);
    match c2 {
        ComparisonOutcome::InputInvalid {
            error: BundleError::LeafChallengeKeyUnknown,
        } => {}
        other => panic!("D18 unknown id: expected LeafChallengeKeyUnknown, got {other:?}"),
    }

    let decision = decide(
        &recompute_view_from_comparison(&comparison),
        &agreed_cross(epoch, bundle.body.merkle_root),
        &bundle,
        &chain,
        &trust,
        86,
    );
    assert!(
        decision.submissions().is_empty(),
        "D18 must not produce a weight submission"
    );
}

// ─── (v) silent omit of declared participant (D24) ───

/// (v) Gateway silently omits a declared participant → IncompleteParticipantSet.
#[test]
fn a48_v_omit_participant_d24() {
    let epoch = 4805;
    let csk = sk(1);
    let gsk = sk(2);
    let m1 = pk_of(&sk(10));
    let m2 = pk_of(&sk(11));
    let (mut bundle, trust, chain) = multi_challenge_bundle(
        &[(b"ch-a", csk, 10_000, 50)],
        &gsk,
        &[m1, m2],
        100,
        epoch,
        &[],
        None,
    );
    // Drop second miner leaf (censorship).
    bundle.body.leaves.retain(|l| l.miner_hotkey == m1);
    finalize_body_merkle(&mut bundle.body);
    let verified: Vec<VerifiedLeaf> = bundle
        .body
        .leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    bundle.body.final_vector = aggregate(
        &verified,
        &bundle.body.emission_shares,
        &bundle.body.uid_map,
        AGG_V,
    )
    .unwrap();
    let bundle = sign_bundle(&gsk, bundle.body).unwrap();

    let comparison = compare_bundle(&bundle, &chain, &trust);
    match &comparison {
        ComparisonOutcome::InputInvalid {
            error: BundleError::IncompleteParticipantSet,
        } => {}
        other => panic!("D24 omit: expected IncompleteParticipantSet, got {other:?}"),
    }
    let decision = decide(
        &recompute_view_from_comparison(&comparison),
        &agreed_cross(epoch, bundle.body.merkle_root),
        &bundle,
        &chain,
        &trust,
        85,
    );
    // Single challenge incomplete → quarantine drops it → Class B, zero submissions.
    assert!(decision.submissions().is_empty());
    match decision {
        EpochDecision::ClassB { reason, dissent } => {
            assert!(
                matches!(
                    reason,
                    DissentReasonCode::IncompleteParticipantSet
                        | DissentReasonCode::QuarantineExhausted
                ),
                "got {reason:?}"
            );
            dissent.verify().unwrap();
        }
        other => panic!("expected ClassB on D24 omit, got {other:?}"),
    }
}

// ─── (vi) challenge shrinks declared participant set (D24 second half) ───

/// (vi) Owner-signed policy + metagraph imply full set; gateway presents a
/// proper subset of leaves → validator derives expected set and rejects.
#[test]
fn a48_vi_set_shrink_rejected_d24() {
    let epoch = 4806;
    let csk = sk(1);
    let gsk = sk(2);
    let m1 = pk_of(&sk(30));
    let m2 = pk_of(&sk(31));
    let m3 = pk_of(&sk(32));
    // Explicit allowlist of three (owner-signed local trust) ∩ metagraph.
    let mut allow = [m1, m2, m3];
    allow.sort_by(|a, b| a.as_slice().cmp(b.as_slice()));
    let policy = ParticipantPolicy::ExplicitAllowlist {
        hotkeys: allow.to_vec(),
    };
    let (mut bundle, trust, chain) = multi_challenge_bundle(
        &[(b"ch-a", csk, 10_000, 40)],
        &gsk,
        &[m1, m2, m3],
        120,
        epoch,
        &[],
        Some(&[policy]),
    );
    // Shrink: drop m3 entirely (proper subset of expected set).
    bundle.body.leaves.retain(|l| l.miner_hotkey != m3);
    finalize_body_merkle(&mut bundle.body);
    let verified: Vec<VerifiedLeaf> = bundle
        .body
        .leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    bundle.body.final_vector = aggregate(
        &verified,
        &bundle.body.emission_shares,
        &bundle.body.uid_map,
        AGG_V,
    )
    .unwrap();
    let bundle = sign_bundle(&gsk, bundle.body).unwrap();

    // Expected set is derived from local trust policy + metagraph — not gateway claims.
    let comparison = compare_bundle(&bundle, &chain, &trust);
    match comparison {
        ComparisonOutcome::InputInvalid {
            error: BundleError::IncompleteParticipantSet,
        } => {}
        other => panic!("D24 shrink: expected IncompleteParticipantSet, got {other:?}"),
    }
}

// ─── (vii) one garbage challenge, others healthy → quarantine, still submit ───

/// (vii) One challenge garbage, 60% mass survives → quarantine that challenge, submit.
#[test]
fn a48_vii_one_garbage_quarantine_still_submit() {
    let epoch = 4807;
    let gsk = sk(2);
    let c_good = sk(3);
    let c_bad = sk(4);
    let m1 = pk_of(&sk(20));
    let m2 = pk_of(&sk(21));
    let (bundle, trust, chain) = multi_challenge_bundle(
        &[(b"good", c_good, 6000, 50), (b"bad", c_bad, 4000, 50)],
        &gsk,
        &[m1, m2],
        200,
        epoch,
        &[b"bad"],
        None,
    );
    let comparison = compare_bundle(&bundle, &chain, &trust);
    assert!(matches!(comparison, ComparisonOutcome::InputInvalid { .. }));
    let decision = decide(
        &recompute_view_from_comparison(&comparison),
        &agreed_cross(epoch, bundle.body.merkle_root),
        &bundle,
        &chain,
        &trust,
        84,
    );
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
            let expected = aggregate(&verified, &new_shares, &bundle.body.uid_map, AGG_V).unwrap();
            assert_eq!(intent.vector, expected);
        }
        other => panic!("expected Quarantine, got {other:?}"),
    }
}

// ─── (viii) two challenges fail, 55% mass survives → both quarantined, submit ───

/// (viii) Two challenges fail, 55% mass survives → both quarantined, still submit.
#[test]
fn a48_viii_two_fail_55pct_still_submit() {
    let epoch = 4808;
    let gsk = sk(2);
    let c_good = sk(5);
    let c_b1 = sk(6);
    let c_b2 = sk(7);
    let m1 = pk_of(&sk(30));
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
        None,
    );
    let comparison = compare_bundle(&bundle, &chain, &trust);
    assert!(matches!(comparison, ComparisonOutcome::InputInvalid { .. }));
    let decision = decide(
        &recompute_view_from_comparison(&comparison),
        &agreed_cross(epoch, bundle.body.merkle_root),
        &bundle,
        &chain,
        &trust,
        83,
    );
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

// ─── (ix) drand unreachable → epoch skipped, set_weights never called ───

/// (ix) drand unreachable at submission → SkippedDrand, no set_weights / no extrinsic.
#[test]
fn a48_ix_drand_unreachable_skip_no_set_weights() {
    let chain = FakeChain::with_defaults();
    let clock = FixedClock {
        now_unix_secs: 1_700_000_000.0,
    };
    let tip = chain.current_block().unwrap();
    let cfg = SubmitConfig {
        netuid: 1,
        mecid: MECID_MAIN,
        version_key: 1,
        hotkey: vec![0xA1; 32],
        epoch_deadline_block: tip.saturating_sub(1),
        max_rate_limit_retries: 0,
        max_drand_retries: 0,
    };
    let intent = gbase_dissent::SubmissionIntent {
        epoch: 4809,
        vector: vec![(0, 100), (1, 200)],
        source: SubmissionSource::Recompute,
    };
    let out = submit_intent(&intent, &chain, &FailingDrand, &clock, &cfg).expect("skip ok");
    assert_eq!(out, SubmitOutcome::SkippedDrand { epoch: 4809 });
    assert!(chain.call_log().is_empty(), "no extrinsics on drand skip");
    assert!(chain.submissions().is_empty());
    assert!(
        chain.set_weights_log().is_empty(),
        "set_weights must never be called on drand failure (D5)"
    );
}

// ─── (x) validator eclipsed from all peers → Degraded, zero submissions ───

/// (x) Eclipsed from all peers → PeerSampleInsufficient (Degraded), zero submissions.
#[tokio::test]
async fn a48_x_eclipse_degraded_zero_submissions() {
    let epoch = 4810u64;
    let sk_c = sk(30);
    let sk_other = sk(31);
    let hk_c = pk_of(&sk_c);
    let hk_other = pk_of(&sk_other);

    let (bundle, trust, chain_inner) = multi_challenge_bundle(
        &[(b"dummy", sk(1), 10_000, 50)],
        &sk(2),
        &[hk_c, hk_other],
        100,
        epoch,
        &[],
        None,
    );
    let root = bundle.body.merkle_root;
    let chain = Arc::new(SyncChain::new(chain_inner));

    let gateway = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(format!("/v1/bundle/{epoch}")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(bundle.encode_bytes()),
        )
        .mount(&gateway)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/weights/latest"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&gateway)
        .await;

    // Peer exists on metagraph but is unreachable (eclipse).
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
        "Degraded eclipse expected, got {out:?}"
    );
    match cross {
        CrossCheckOutcome::Failed {
            kind: CrossCheckFailKind::InsufficientPeers { .. },
            ..
        } => {}
        other => panic!("expected InsufficientPeers, got {other:?}"),
    }

    let view = recompute_view_from_comparison(&out);
    let secret = sk(82);
    let decision = apply_three_outcome_policy(
        &view,
        &cross,
        Some(&bundle),
        chain.as_ref(),
        &trust,
        5000,
        DissentSigner {
            secret: &secret,
            hotkey: pk_of(&secret),
        },
        None,
    )
    .expect("policy");
    assert!(
        decision.submissions().is_empty(),
        "eclipse must yield zero submissions"
    );
    match decision {
        EpochDecision::ClassB { reason, .. } => {
            assert_eq!(reason, DissentReasonCode::PeerSampleInsufficient);
        }
        other => panic!("expected ClassB Degraded, got {other:?}"),
    }
}

// ─── ops: compose profile isolation (unit-level) ───

/// Ops: evil-gateway must never appear on default/master service path without profile.
#[test]
fn a48_ops_evil_gateway_profile_isolated() {
    let compose = include_str!("../../../docker-compose.yml");
    assert!(
        compose.contains("evil-gateway") || compose.contains("profiles: [\"evil-gateway\"]"),
        "evil-gateway profile must exist in docker-compose.yml"
    );
    // Default services block must not start evil-gateway without profile.
    // Parse lightly: every `evil-gateway:` service stanza must list the profile.
    let mut in_evil = false;
    let mut saw_profile = false;
    for line in compose.lines() {
        let t = line.trim();
        if t.starts_with("evil-gateway:") {
            in_evil = true;
            saw_profile = false;
            continue;
        }
        if in_evil {
            if t.starts_with("profiles:") && t.contains("evil-gateway") {
                saw_profile = true;
            }
            // Next top-level service or volumes/networks ends the stanza.
            if !line.starts_with(' ')
                && !line.starts_with('\t')
                && !t.is_empty()
                && !t.starts_with('#')
            {
                assert!(
                    saw_profile,
                    "evil-gateway service must declare profiles: [\"evil-gateway\"]"
                );
                in_evil = false;
            }
        }
    }
    if in_evil {
        assert!(saw_profile, "evil-gateway must be profile-gated");
    }
    // master profile must not include evil-gateway.
    assert!(
        !compose.contains("profiles: [\"master\", \"evil-gateway\"]")
            && !compose.contains("profiles: [\"evil-gateway\", \"master\"]"),
        "evil-gateway must not share master profile"
    );
}

/// Ops: gateway service uses `restart: unless-stopped` (auto-restart basis for kill test).
#[test]
fn a48_ops_gateway_restart_policy_unless_stopped() {
    let compose = include_str!("../../../docker-compose.yml");
    // Find gateway: block and ensure restart: unless-stopped appears before next service.
    let mut lines = compose.lines().peekable();
    let mut found = false;
    while let Some(line) = lines.next() {
        if line.trim() == "gateway:" {
            for next in lines.by_ref() {
                let t = next.trim();
                if t == "restart: unless-stopped" {
                    found = true;
                    break;
                }
                if !next.starts_with(' ')
                    && !next.starts_with('\t')
                    && !t.is_empty()
                    && !t.starts_with('#')
                {
                    break;
                }
            }
            break;
        }
    }
    assert!(
        found,
        "gateway must have restart: unless-stopped for <60s auto-restart"
    );
}

// ─── smoke: policy Match still works (regression adjacent) ───

#[test]
fn a48_regression_honest_match_submits() {
    let epoch = 4899;
    let (bundle, trust, chain, _) = single_honest(epoch);
    let comparison = compare_bundle(&bundle, &chain, &trust);
    assert!(matches!(comparison, ComparisonOutcome::Match { .. }));
    let decision = decide(
        &recompute_view_from_comparison(&comparison),
        &agreed_cross(epoch, bundle.body.merkle_root),
        &bundle,
        &chain,
        &trust,
        80,
    );
    assert!(matches!(decision, EpochDecision::Match { .. }));
    assert_eq!(decision.submissions().len(), 1);
}

/// Ensure wiremock path still compiles with Duration import used by async tests.
#[tokio::test]
async fn a48_async_runtime_smoke() {
    tokio::time::sleep(Duration::from_millis(1)).await;
}
