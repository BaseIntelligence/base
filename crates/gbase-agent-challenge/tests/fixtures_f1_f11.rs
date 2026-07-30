//! Fixture-driven offline tests F1–F11 (`AGENT_CHALLENGE` §5.5–§5.7).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gbase_agent_challenge::{
    answer_digest, score_from_outcome, task_blob, task_id, AttestationStatus, CallOutcome,
    NoScoreReasonCode, ScoreInputs, ScoreOrAbsence, SCORE_MAX, SCORING_VERSION,
};

const F1_TASK_ID: &str = "4a590b2abf87da6bccd97d8fbe5d2e774bdbda3ad421119688010537be2b31ec";
const F1_TASK_BLOB: &str = "8c5430ceb95b9e422026baf2eaddb4c9c723923c6353164fe9b0905a47f9a29f";
const F1_ANSWER: &str = "83180b08e05630496531a158d174ce69ba857d854d8692087947706c159a487c";
const F11_TASK_ID: &str = "d954306fba3943a86bb69aedfd08f2bca850eb2adabaaf5efe2ad2728dbf3412";
const F11_ANSWER: &str = "05157d001bb1ec9ef5acc7140d0221141d2fbc14a830ce32893793f30470c0aa";

fn miner11() -> [u8; 32] {
    [0x11u8; 32]
}

fn miner22() -> [u8; 32] {
    [0x22u8; 32]
}

fn correct_inputs(miner: [u8; 32], duration_ms: u64, att: AttestationStatus) -> ScoreInputs {
    let tid = task_id(1, 7, &miner);
    let blob = task_blob(&tid, SCORING_VERSION);
    let ans = answer_digest(&blob);
    ScoreInputs {
        netuid: 1,
        epoch: 7,
        miner_hotkey: miner,
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
fn f1_digests_and_score_max() {
    let m = miner11();
    let tid = task_id(1, 7, &m);
    let blob = task_blob(&tid, SCORING_VERSION);
    let ans = answer_digest(&blob);
    assert_eq!(hex::encode(tid), F1_TASK_ID);
    assert_eq!(hex::encode(blob), F1_TASK_BLOB);
    assert_eq!(hex::encode(ans), F1_ANSWER);
    assert_eq!(
        score_from_outcome(&correct_inputs(m, 2000, AttestationStatus::Verified)),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
}

#[test]
fn f2_duration_zero_score_max() {
    assert_eq!(
        score_from_outcome(&correct_inputs(miner11(), 0, AttestationStatus::Verified)),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
}

#[test]
fn f3_duration_6000_half() {
    assert_eq!(
        score_from_outcome(&correct_inputs(
            miner11(),
            6000,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: 500_000 }
    );
}

#[test]
fn f4_duration_hard_score_zero() {
    assert_eq!(
        score_from_outcome(&correct_inputs(
            miner11(),
            10_000,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: 0 }
    );
}

#[test]
fn f5_duration_over_hard_timeout() {
    assert_eq!(
        score_from_outcome(&correct_inputs(
            miner11(),
            10_001,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout
        }
    );
}

#[test]
fn f6_wrong_answer_score_zero() {
    let mut inp = correct_inputs(miner11(), 2000, AttestationStatus::Verified);
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
fn f7_parked_attestation() {
    assert_eq!(
        score_from_outcome(&correct_inputs(miner11(), 2000, AttestationStatus::Parked)),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified
        }
    );
}

#[test]
fn f8_missing_attestation() {
    assert_eq!(
        score_from_outcome(&correct_inputs(miner11(), 2000, AttestationStatus::Missing)),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified
        }
    );
}

#[test]
fn f9_rejected_attestation() {
    assert_eq!(
        score_from_outcome(&correct_inputs(
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
fn f10_schema_invalid_response() {
    let inp = ScoreInputs {
        netuid: 1,
        epoch: 7,
        miner_hotkey: miner11(),
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
fn f11_second_miner_digests_and_score() {
    let m = miner22();
    let tid = task_id(1, 7, &m);
    let blob = task_blob(&tid, SCORING_VERSION);
    let ans = answer_digest(&blob);
    assert_eq!(hex::encode(tid), F11_TASK_ID);
    assert_eq!(hex::encode(ans), F11_ANSWER);
    assert_eq!(
        score_from_outcome(&correct_inputs(m, 2000, AttestationStatus::Verified)),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
}

#[test]
fn reference_assertions_section_5_7() {
    // assert score(F1) == Score(1_000_000)
    assert_eq!(
        score_from_outcome(&correct_inputs(
            miner11(),
            2000,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: 1_000_000 }
    );
    // assert score(F3) == Score(500_000)
    assert_eq!(
        score_from_outcome(&correct_inputs(
            miner11(),
            6000,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::Score { value: 500_000 }
    );
    // assert score(F5) == NoScore(Timeout)
    assert_eq!(
        score_from_outcome(&correct_inputs(
            miner11(),
            10_001,
            AttestationStatus::Verified
        )),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout
        }
    );
    // assert score(F7) == NoScore(AttestationNotVerified)
    assert_eq!(
        score_from_outcome(&correct_inputs(miner11(), 2000, AttestationStatus::Parked)),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified
        }
    );
    assert_eq!(hex::encode(task_id(1, 7, &miner11())), F1_TASK_ID);
    assert_eq!(
        hex::encode(answer_digest(&task_blob(
            &task_id(1, 7, &miner11()),
            SCORING_VERSION
        ))),
        F1_ANSWER
    );
}
