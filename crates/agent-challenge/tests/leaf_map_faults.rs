//! Verifier-fault → leaf mapping (todo 14).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use agent_challenge::{
    attempts_within_seal_budget, cover_expected_verify_leaves, grade_to_score_or_absence,
    grade_to_score_or_absence_budgeted, is_operator_fault, map_reward, map_verify_error,
    score_from_outcome, AttestationStatus, CallOutcome, Reward, ScoreInputs, ScoreOrAbsence,
    Verifier, VerifyError, ZeroReason, FIXTURE_MODEL_PATCH, FIXTURE_PACK_ID, MAX_VERIFY_ATTEMPTS,
    MAX_VERIFY_RETRIES, SCORE_MAX, NoScoreReasonCode,
};
use agent_pack::{HarborPack, HeldOutMaterials};
use docker_engine::DockerError;

fn docker_api_err() -> VerifyError {
    VerifyError::Docker(DockerError::Api("connection refused".into()))
}

fn malformed() -> VerifyError {
    VerifyError::MalformedOutput {
        message: "junit: not well-formed".into(),
    }
}

fn apply_failed() -> VerifyError {
    VerifyError::ApplyFailed {
        message: "git apply failed".into(),
    }
}

#[test]
fn map_operator_faults_are_challenge_internal() {
    for err in [
        VerifyError::Timeout { timeout_sec: 30 },
        docker_api_err(),
        malformed(),
        VerifyError::Staging {
            message: "disk full".into(),
        },
        VerifyError::MissingHeldOut {
            message: "no tests".into(),
        },
    ] {
        assert!(is_operator_fault(&err), "{err:?}");
        assert_eq!(
            map_verify_error(&err),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::ChallengeInternal
            },
            "{err:?}"
        );
    }
}

#[test]
fn map_apply_failed_is_score_zero_not_challenge_internal() {
    let err = apply_failed();
    assert!(!is_operator_fault(&err));
    assert_eq!(
        map_verify_error(&err),
        ScoreOrAbsence::Score { value: 0 },
        "ApplyFailed is miner-attributable Score(0)"
    );
}

#[test]
fn map_reward_zero_is_score_zero() {
    let err = VerifyError::RewardZero {
        reason: ZeroReason::TestsFailed {
            f2p_failed: 1,
            p2p_failed: 0,
            detail: "x".into(),
        },
    };
    assert_eq!(map_verify_error(&err), ScoreOrAbsence::Score { value: 0 });
}

#[test]
fn map_never_emits_park_or_attestation_not_verified() {
    let samples = [
        VerifyError::Timeout { timeout_sec: 1 },
        docker_api_err(),
        malformed(),
        apply_failed(),
        VerifyError::RewardZero {
            reason: ZeroReason::Unspecified {
                detail: "z".into(),
            },
        },
    ];
    for err in samples {
        let soa = map_verify_error(&err);
        match soa {
            ScoreOrAbsence::NoScore { reason } => {
                assert_ne!(
                    reason,
                    NoScoreReasonCode::AttestationNotVerified,
                    "Park/attest path forbidden for verify faults: {err:?}"
                );
                assert_ne!(reason, NoScoreReasonCode::NotAttempted);
                assert_ne!(reason, NoScoreReasonCode::PolicySkip);
            }
            ScoreOrAbsence::Score { .. } => {}
        }
    }
}

#[test]
fn verifier_timeout_differs_from_miner_call_timeout() {
    let verify_t = map_verify_error(&VerifyError::Timeout { timeout_sec: 60 });
    let miner = [0xAAu8; 32];
    let miner_t = score_from_outcome(&ScoreInputs {
        netuid: 1,
        epoch: 1,
        miner_hotkey: miner,
        pack_id: FIXTURE_PACK_ID.to_vec(),
        expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
        attestation: AttestationStatus::Verified,
        duration_ms: 0,
        outcome: CallOutcome::Timeout,
    });
    assert_eq!(
        verify_t,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    assert_eq!(
        miner_t,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout
        }
    );
    assert_ne!(verify_t, miner_t);
}

#[test]
fn malformed_junit_not_coerced_to_reward_zero_score_path() {
    let soa = map_verify_error(&malformed());
    assert_eq!(
        soa,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    assert!(!matches!(soa, ScoreOrAbsence::Score { value: 0 }));
}

#[test]
fn map_reward_resolve_is_score_max() {
    assert_eq!(
        map_reward(Reward::try_new(1).unwrap()),
        ScoreOrAbsence::Score { value: SCORE_MAX }
    );
    assert_eq!(
        map_reward(Reward::try_new(0).unwrap()),
        ScoreOrAbsence::Score { value: 0 }
    );
}

struct FailThenOk {
    fails_left: AtomicU32,
    calls: Arc<AtomicU32>,
}
impl Verifier for FailThenOk {
    fn grade(&self, _pack: &HarborPack, _model_patch: &[u8]) -> Result<Reward, VerifyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let left = self.fails_left.load(Ordering::SeqCst);
        if left > 0 {
            self.fails_left.fetch_sub(1, Ordering::SeqCst);
            return Err(docker_api_err());
        }
        Reward::try_new(1)
    }
}

struct AlwaysApplyFail;
impl Verifier for AlwaysApplyFail {
    fn grade(&self, _pack: &HarborPack, _model_patch: &[u8]) -> Result<Reward, VerifyError> {
        Err(apply_failed())
    }
}

struct AlwaysMalformed {
    calls: Arc<AtomicU32>,
}
impl Verifier for AlwaysMalformed {
    fn grade(&self, _pack: &HarborPack, _model_patch: &[u8]) -> Result<Reward, VerifyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(malformed())
    }
}

fn empty_pack() -> HarborPack {
    HarborPack {
        task_id: "t".into(),
        schema_version: "1".into(),
        repository_url: "https://example.invalid/r".into(),
        base_commit_hash: "abc".into(),
        instruction: "fix".into(),
        dockerfile: b"FROM scratch\n".to_vec(),
        agent_timeout_sec: 60,
        verifier_timeout_sec: Some(30),
        held_out: HeldOutMaterials {
            solution_patch: None,
            test_patch: None,
            grader_py: None,
        },
        files: vec![],
    }
}

#[test]
fn grade_retries_bounded_for_operator_outage() {
    let calls = Arc::new(AtomicU32::new(0));
    let v = FailThenOk {
        fails_left: AtomicU32::new(100),
        calls: Arc::clone(&calls),
    };
    let pack = empty_pack();
    let soa = grade_to_score_or_absence(&v, &pack, b"");
    assert_eq!(
        soa,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        MAX_VERIFY_ATTEMPTS,
        "must not outrun retry bound"
    );
}

#[test]
fn grade_retries_then_success() {
    let calls = Arc::new(AtomicU32::new(0));
    let v = FailThenOk {
        fails_left: AtomicU32::new(1),
        calls: Arc::clone(&calls),
    };
    let soa = grade_to_score_or_absence(&v, &empty_pack(), b"");
    assert_eq!(soa, ScoreOrAbsence::Score { value: SCORE_MAX });
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn grade_apply_failed_no_retry() {
    let soa = grade_to_score_or_absence(&AlwaysApplyFail, &empty_pack(), b"bad");
    assert_eq!(soa, ScoreOrAbsence::Score { value: 0 });
}

#[test]
fn grade_malformed_single_attempt() {
    let calls = Arc::new(AtomicU32::new(0));
    let v = AlwaysMalformed {
        calls: Arc::clone(&calls),
    };
    let soa = grade_to_score_or_absence(&v, &empty_pack(), b"");
    assert_eq!(
        soa,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "non-retryable");
}

#[test]
fn cover_expected_verifier_outage_all_challenge_internal() {
    let m1 = [0x11u8; 32];
    let m2 = [0x22u8; 32];
    let m3 = [0x33u8; 32];
    let expected: BTreeSet<_> = [m1, m2, m3].into_iter().collect();
    let results = BTreeMap::new();
    let leaves = cover_expected_verify_leaves(&expected, &results);
    assert_eq!(leaves.len(), expected.len());
    for h in &expected {
        assert_eq!(
            leaves.get(h),
            Some(&ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::ChallengeInternal
            })
        );
    }
}

#[test]
fn cover_expected_mixed_operator_and_miner() {
    let m1 = [0x11u8; 32];
    let m2 = [0x22u8; 32];
    let expected: BTreeSet<_> = [m1, m2].into_iter().collect();
    let mut results = BTreeMap::new();
    results.insert(m1, Err(docker_api_err()));
    results.insert(m2, Err(apply_failed()));
    let leaves = cover_expected_verify_leaves(&expected, &results);
    assert_eq!(
        leaves[&m1],
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    assert_eq!(leaves[&m2], ScoreOrAbsence::Score { value: 0 });
}

#[test]
fn cover_ignores_hotkeys_outside_e() {
    let m1 = [0x11u8; 32];
    let outsider = [0x99u8; 32];
    let expected: BTreeSet<_> = [m1].into_iter().collect();
    let mut results = BTreeMap::new();
    results.insert(outsider, Ok(Reward::try_new(1).unwrap()));
    let leaves = cover_expected_verify_leaves(&expected, &results);
    assert_eq!(leaves.len(), 1);
    assert!(!leaves.contains_key(&outsider));
}

#[test]
fn seal_budget_caps_attempts() {
    assert_eq!(attempts_within_seal_budget(0, 1000), 0);
    assert_eq!(attempts_within_seal_budget(500, 1000), 0);
    assert_eq!(attempts_within_seal_budget(2500, 1000), 2);
    assert_eq!(
        attempts_within_seal_budget(u64::MAX / 2, 1),
        MAX_VERIFY_ATTEMPTS
    );
}

#[test]
fn budgeted_grade_cannot_outrun_seal() {
    let calls = Arc::new(AtomicU32::new(0));
    let v = FailThenOk {
        fails_left: AtomicU32::new(100),
        calls: Arc::clone(&calls),
    };
    let soa = grade_to_score_or_absence_budgeted(&v, &empty_pack(), b"", 1000, 1000);
    assert_eq!(
        soa,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn constants_match_doc() {
    assert_eq!(MAX_VERIFY_RETRIES, 2);
    assert_eq!(MAX_VERIFY_ATTEMPTS, 3);
}
