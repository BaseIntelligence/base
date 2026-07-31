//! Fixture-driven offline tests F1–F11 v2 successors (`AGENT_CHALLENGE` `scoring_version` = 2).
//!
//! Each F1–F11 case is a named v2 successor. Latency decay is removed: correct answers
//! always score `SCORE_MAX` regardless of `duration_ms`. Digests match
//! `tests/fixtures/v2_*.hex` (pack-fixture-001 + `FIXTURE_MODEL_PATCH`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use agent_challenge::{
    answer_digest, answer_digest_v2, score_from_outcome, task_blob, task_blob_v2, task_id,
    task_id_v2, AttestationStatus, CallOutcome, NoScoreReasonCode, ScoreInputs, ScoreOrAbsence,
    FIXTURE_MODEL_PATCH, FIXTURE_PACK_ID, SCORE_MAX, SCORING_VERSION,
};

/// F1 v2 successor — miner 0x11 `task_id` golden.
const F1_V2_TASK_ID: &str = "b1c18e56abe993e20e8dadcb72c7a7cadee8975e5741d15d1acb37f5ea367644";
/// F1 v2 successor — `task_blob` golden.
const F1_V2_TASK_BLOB: &str = "c563caca4fa3a7c5e834a88b0dae9eb1ef87f90fcddc9973e38d2730b347c441";
/// F1/F11 v2 successor — `answer_digest_v2(model.patch)` golden (patch-only preimage).
const F1_V2_ANSWER: &str = "703b806158d655e5d37a5b45e3cbdf1e04735517805377199d108ae2a45ead5d";
/// F11 v2 successor — miner 0x22 `task_id` golden.
const F11_V2_TASK_ID: &str = "b99762643336fbf7abeb2c07085ff3d64ee1fd8d1c98b149c57a36ec0396228f";

fn miner11() -> [u8; 32] {
    [0x11u8; 32]
}

fn miner22() -> [u8; 32] {
    [0x22u8; 32]
}

fn correct_inputs_v2(miner: [u8; 32], duration_ms: u64, att: AttestationStatus) -> ScoreInputs {
    let tid = task_id_v2(1, 7, &miner, FIXTURE_PACK_ID, SCORING_VERSION);
    let ans = answer_digest_v2(FIXTURE_MODEL_PATCH);
    ScoreInputs {
        netuid: 1,
        epoch: 7,
        miner_hotkey: miner,
        pack_id: FIXTURE_PACK_ID.to_vec(),
        expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
        attestation: att,
        duration_ms,
        outcome: CallOutcome::Http200 {
            challenge_id: "agent-v1".into(),
            epoch: 7,
            task_id: tid,
            answer_digest: ans,
            agent_version: "1".into(),
        },
    }
}

#[test]
fn f1_v2_digests_and_score_max() {
    let m = miner11();
    let tid = task_id_v2(1, 7, &m, FIXTURE_PACK_ID, SCORING_VERSION);
    let blob = task_blob_v2(&tid, SCORING_VERSION, FIXTURE_PACK_ID);
    let ans = answer_digest_v2(FIXTURE_MODEL_PATCH);
    assert_eq!(hex::encode(tid), F1_V2_TASK_ID);
    assert_eq!(
        hex::encode(tid),
        include_str!("fixtures/v2_task_id_a.hex").trim()
    );
    assert_eq!(hex::encode(blob), F1_V2_TASK_BLOB);
    assert_eq!(
        hex::encode(blob),
        include_str!("fixtures/v2_task_blob_a.hex").trim()
    );
    assert_eq!(hex::encode(ans), F1_V2_ANSWER);
    assert_eq!(
        hex::encode(ans),
        include_str!("fixtures/v2_answer_digest_a.hex").trim()
    );
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(m, 2000, AttestationStatus::Verified)),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
}

/// F2 v2 — `duration_ms=0` still full credit (latency ignored).
#[test]
fn f2_v2_duration_zero_score_max() {
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            0,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
}

/// F3 v2 — former midpoint (6000 ms) is full credit under correctness-only.
#[test]
fn f3_v2_duration_ignored_full_credit() {
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            6000,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
}

/// F4 v2 — former hard boundary (`10_000` ms) is full credit under correctness-only.
#[test]
fn f4_v2_duration_ignored_full_credit() {
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            10_000,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
}

/// F5 v2 — former over-hard duration is full credit when Http200 is correct;
/// Timeout is only via `CallOutcome::Timeout`.
#[test]
fn f5_v2_duration_ignored_timeout_only_via_outcome() {
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            10_001,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
    let timeout = ScoreInputs {
        netuid: 1,
        epoch: 7,
        miner_hotkey: miner11(),
        pack_id: FIXTURE_PACK_ID.to_vec(),
        expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
        attestation: AttestationStatus::Verified,
        duration_ms: 10_001,
        outcome: CallOutcome::Timeout,
    };
    assert_eq!(
        score_from_outcome(&timeout),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout
        }
    );
}

#[test]
fn f6_v2_wrong_answer_score_zero() {
    let mut inp = correct_inputs_v2(miner11(), 2000, AttestationStatus::Verified);
    if let CallOutcome::Http200 {
        ref mut answer_digest,
        ..
    } = inp.outcome
    {
        *answer_digest = [0xffu8; 32];
    }
    assert_eq!(score_from_outcome(&inp), ScoreOrAbsence::Score { value: 0 });
}

#[test]
fn f7_v2_parked_attestation() {
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            2000,
            AttestationStatus::Parked
        )),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified
        }
    );
}

#[test]
fn f8_v2_missing_attestation() {
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            2000,
            AttestationStatus::Missing
        )),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified
        }
    );
}

#[test]
fn f9_v2_rejected_attestation() {
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            2000,
            AttestationStatus::Rejected
        )),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified
        }
    );
}

#[test]
fn f10_v2_schema_invalid_response() {
    let inp = ScoreInputs {
        netuid: 1,
        epoch: 7,
        miner_hotkey: miner11(),
        pack_id: FIXTURE_PACK_ID.to_vec(),
        expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
        attestation: AttestationStatus::Verified,
        duration_ms: 50,
        outcome: CallOutcome::InvalidResponse,
    };
    assert_eq!(
        score_from_outcome(&inp),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::InvalidResponse
        }
    );
}

#[test]
fn f11_v2_second_miner_digests_and_score() {
    let m = miner22();
    let tid = task_id_v2(1, 7, &m, FIXTURE_PACK_ID, SCORING_VERSION);
    let ans = answer_digest_v2(FIXTURE_MODEL_PATCH);
    assert_eq!(hex::encode(tid), F11_V2_TASK_ID);
    assert_eq!(
        hex::encode(tid),
        include_str!("fixtures/v2_task_id_b.hex").trim()
    );
    // Same patch → same answer_digest_v2 regardless of miner.
    assert_eq!(hex::encode(ans), F1_V2_ANSWER);
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(m, 2000, AttestationStatus::Verified)),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
}

#[test]
fn reference_assertions_section_5_7_v2() {
    assert_eq!(SCORING_VERSION, 2);
    // assert score(F1_v2) == Score(1_000_000)
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            2000,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: 1_000_000 }
    );
    // F3 former half-credit → full credit under correctness-only
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            6000,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
    // F5 duration alone does not Timeout; CallOutcome::Timeout does
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            10_001,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
    // assert score(F7_v2) == NoScore(AttestationNotVerified)
    assert_eq!(
        score_from_outcome(&correct_inputs_v2(
            miner11(),
            2000,
            AttestationStatus::Parked
        )),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified
        }
    );
    assert_eq!(
        hex::encode(task_id_v2(
            1,
            7,
            &miner11(),
            FIXTURE_PACK_ID,
            SCORING_VERSION
        )),
        F1_V2_TASK_ID
    );
    assert_eq!(
        hex::encode(answer_digest_v2(FIXTURE_MODEL_PATCH)),
        F1_V2_ANSWER
    );
}

/// Retired v1 echo fixture must not validate under live `scoring_version` 2.
#[test]
fn f_echo_retired_v1_answer_does_not_validate() {
    let m = miner11();
    let tid_v2 = task_id_v2(1, 7, &m, FIXTURE_PACK_ID, SCORING_VERSION);
    let echo = answer_digest(&task_blob(&task_id(1, 7, &m), 1));
    let inp = ScoreInputs {
        netuid: 1,
        epoch: 7,
        miner_hotkey: m,
        pack_id: FIXTURE_PACK_ID.to_vec(),
        expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
        attestation: AttestationStatus::Verified,
        duration_ms: 2000,
        outcome: CallOutcome::Http200 {
            challenge_id: "agent-v1".into(),
            epoch: 7,
            task_id: tid_v2,
            answer_digest: echo,
            agent_version: "1".into(),
        },
    };
    assert_eq!(
        score_from_outcome(&inp),
        ScoreOrAbsence::Score { value: 0 },
        "v1 echo answer must not earn credit under scoring_version 2"
    );
}

/// Integration truth table for QA evidence (reward × attestation).
#[test]
fn truth_table_matrix_v2_correctness() {
    let correct = |att: AttestationStatus| correct_inputs_v2(miner11(), 99_999, att);
    let mut wrong = correct(AttestationStatus::Verified);
    if let CallOutcome::Http200 {
        ref mut answer_digest,
        ..
    } = wrong.outcome
    {
        *answer_digest = [0xAAu8; 32];
    }
    let no_result = ScoreInputs {
        netuid: 1,
        epoch: 7,
        miner_hotkey: miner11(),
        pack_id: FIXTURE_PACK_ID.to_vec(),
        expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
        attestation: AttestationStatus::Verified,
        duration_ms: 1,
        outcome: CallOutcome::MinerError,
    };
    let deadline = ScoreInputs {
        netuid: 1,
        epoch: 7,
        miner_hotkey: miner11(),
        pack_id: FIXTURE_PACK_ID.to_vec(),
        expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
        attestation: AttestationStatus::Verified,
        duration_ms: 1,
        outcome: CallOutcome::Timeout,
    };

    // Verified × reward 1
    assert_eq!(
        score_from_outcome(&correct(AttestationStatus::Verified)),
        ScoreOrAbsence::Score { value: 1_000_000 }
    );
    // Verified × reward 0
    assert_eq!(
        score_from_outcome(&wrong),
        ScoreOrAbsence::Score { value: 0 }
    );
    // Verified × no result
    assert_eq!(
        score_from_outcome(&no_result),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::MinerError
        }
    );
    // Verified × deadline
    assert_eq!(
        score_from_outcome(&deadline),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout
        }
    );

    for att in [
        AttestationStatus::Rejected,
        AttestationStatus::Parked,
        AttestationStatus::Missing,
    ] {
        for mut case in [
            correct(att),
            {
                let mut w = wrong.clone();
                w.attestation = att;
                w
            },
            {
                let mut n = no_result.clone();
                n.attestation = att;
                n
            },
            {
                let mut d = deadline.clone();
                d.attestation = att;
                d
            },
        ] {
            case.attestation = att;
            assert_eq!(
                score_from_outcome(&case),
                ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::AttestationNotVerified
                },
                "non-Verified must gate all rewards"
            );
        }
    }
}
