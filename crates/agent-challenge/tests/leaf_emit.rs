//! Todo 27: leaf emission with D24 completeness (exact E cover).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};

use agent_challenge::{
    emit_signed_leaf_set, score_map_covering_expected, AttestationStatus, CallOutcome,
    LeafEmitError, MinerEpochOutcome, NoScoreReasonCode, ScoreOrAbsence, CHALLENGE_ID, SCORE_MAX,
};
use agent_dispatch::{TaskResultV1, TaskStatusV1, DISPATCH_PROTOCOL};
use crypto::KEY_LEN;
use schnorrkel::MiniSecretKey;

const EPOCH: u64 = 42;
const SOLVER: [u8; KEY_LEN] = [0x11; KEY_LEN];
const ZERO: [u8; KEY_LEN] = [0x22; KEY_LEN];
const UNREACH: [u8; KEY_LEN] = [0x33; KEY_LEN];
const UNATT: [u8; KEY_LEN] = [0x44; KEY_LEN];

fn sk() -> [u8; KEY_LEN] {
    MiniSecretKey::generate_with(rand_core::OsRng).to_bytes()
}

fn e_four() -> BTreeSet<[u8; KEY_LEN]> {
    BTreeSet::from([SOLVER, ZERO, UNREACH, UNATT])
}

fn four_outcomes() -> BTreeMap<[u8; KEY_LEN], ScoreOrAbsence> {
    BTreeMap::from([
        (SOLVER, ScoreOrAbsence::Score { value: SCORE_MAX }),
        (ZERO, ScoreOrAbsence::Score { value: 0 }),
        (
            UNREACH,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::Timeout,
            },
        ),
        (
            UNATT,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::AttestationNotVerified,
            },
        ),
    ])
}

/// S1: solver + zero + unreachable + unattested → four signed leaves, four distinct outcomes.
#[test]
fn s1_leaf_completeness_four_distinct_outcomes() {
    let secret = sk();
    let expected = e_four();
    let scores = four_outcomes();
    let leaves = emit_signed_leaf_set(&secret, EPOCH, &expected, &scores).expect("emit");
    assert_eq!(leaves.len(), expected.len());
    assert_eq!(leaves.len(), 4);
    // At most one leaf per hotkey (BTreeMap construction).
    assert_eq!(leaves.keys().copied().collect::<BTreeSet<_>>(), expected);
    let mut kinds = BTreeSet::new();
    for h in &expected {
        let leaf = leaves.get(h).expect("leaf");
        assert_eq!(leaf.epoch, EPOCH);
        assert_eq!(leaf.miner_hotkey, *h);
        assert_eq!(leaf.challenge_id, CHALLENGE_ID.as_bytes());
        assert_eq!(&leaf.score_or_absence, scores.get(h).unwrap());
        kinds.insert(format!("{:?}", leaf.score_or_absence));
        assert_ne!(leaf.challenge_sig, [0u8; 64], "must be signed");
    }
    assert_eq!(
        kinds.len(),
        4,
        "four distinct ScoreOrAbsence outcomes: {kinds:?}"
    );
}

/// S2: proper subset of E → named missing hotkey; nothing emitted (Err, no partial).
#[test]
fn s2_subset_refused_names_missing_hotkey() {
    let secret = sk();
    let expected = e_four();
    let mut scores = four_outcomes();
    scores.remove(&UNREACH);
    let err = emit_signed_leaf_set(&secret, EPOCH, &expected, &scores).expect_err("subset");
    match &err {
        LeafEmitError::MissingHotkeys(s) => {
            let hx = hex::encode(UNREACH);
            assert!(s.contains(&hx), "must name missing hotkey {hx}: {s}");
            assert!(
                !s.contains(&hex::encode(SOLVER)),
                "must not list present keys as missing"
            );
        }
        other => panic!("expected MissingHotkeys, got {other:?}"),
    }
}

#[test]
fn s2b_unknown_hotkey_refused() {
    let secret = sk();
    let expected = BTreeSet::from([SOLVER, ZERO]);
    let outsider = [0x99u8; KEY_LEN];
    let scores = BTreeMap::from([
        (SOLVER, ScoreOrAbsence::Score { value: SCORE_MAX }),
        (ZERO, ScoreOrAbsence::Score { value: 0 }),
        (outsider, ScoreOrAbsence::Score { value: 1 }),
    ]);
    let err = emit_signed_leaf_set(&secret, EPOCH, &expected, &scores).expect_err("superset");
    match &err {
        LeafEmitError::UnknownHotkeys(s) => {
            assert!(s.contains(&hex::encode(outsider)), "{s}");
        }
        other => panic!("expected UnknownHotkeys, got {other:?}"),
    }
}

/// Total runner failure still yields |E| leaves after cover + emit.
#[test]
fn s1b_total_failure_still_covers_e() {
    let secret = sk();
    let expected = e_four();
    let outcomes: BTreeMap<_, _> = expected
        .iter()
        .map(|h| {
            (
                *h,
                MinerEpochOutcome::TimedOut {
                    pack_id: "p".into(),
                },
            )
        })
        .collect();
    let scores = score_map_covering_expected(&expected, &BTreeMap::new(), &outcomes);
    assert_eq!(scores.len(), 4);
    for soa in scores.values() {
        assert_eq!(
            soa,
            &ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::Timeout
            }
        );
    }
    let leaves = emit_signed_leaf_set(&secret, EPOCH, &expected, &scores).expect("emit");
    assert_eq!(leaves.len(), 4);
}

#[test]
fn cover_maps_completed_graded_and_capacity() {
    let expected = BTreeSet::from([SOLVER, ZERO, UNREACH]);
    let graded = BTreeMap::from([(SOLVER, ScoreOrAbsence::Score { value: SCORE_MAX })]);
    let dummy = TaskResultV1 {
        protocol: DISPATCH_PROTOCOL.into(),
        challenge_id: CHALLENGE_ID.into(),
        scoring_version: 2,
        epoch: EPOCH,
        miner_hotkey_hex: hex::encode(ZERO),
        pack_id: "p".into(),
        status: TaskStatusV1::Completed,
        model_patch: None,
        patch_sha256_hex: String::new(),
        receipt_sig_hex: String::new(),
    };
    let outcomes = BTreeMap::from([
        (
            ZERO,
            MinerEpochOutcome::Completed {
                pack_id: "p".into(),
                result: dummy,
            },
        ),
        (
            UNREACH,
            MinerEpochOutcome::CapacityExhausted {
                pack_id: "p".into(),
            },
        ),
    ]);
    let scores = score_map_covering_expected(&expected, &graded, &outcomes);
    assert_eq!(scores[&SOLVER], ScoreOrAbsence::Score { value: SCORE_MAX });
    // Completed without grade → ChallengeInternal (silence is a bug).
    assert_eq!(
        scores[&ZERO],
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    assert_eq!(
        scores[&UNREACH],
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    let _ = (AttestationStatus::Missing, CallOutcome::Timeout);
}
