//! Pure integer scoring rule (`AGENT_CHALLENGE` §5.4).
//!
//! Live path uses pack-bound v2 task identity and `answer_digest_v2(model.patch)`.
//! The v1 echo answer (`gbase-agent-answer-v1` ‖ task_blob) is retired.

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use crypto::KEY_LEN;

use crate::task_gen::{answer_digest_v2, task_id_v2, CHALLENGE_ID, SCORING_VERSION};

/// Maximum score value.
pub const SCORE_MAX: u64 = 1_000_000;
/// Soft latency bound (ms) — full credit at or below.
pub const SOFT_MS: u64 = 2_000;
/// Hard latency bound (ms) — zero credit at boundary; timeout above.
pub const HARD_MS: u64 = 10_000;
/// Connect timeout (ms) — protocol constant (not used in pure scorer).
pub const CONNECT_MS: u64 = 3_000;
/// Max HTTP attempts per miner.
pub const MAX_ATTEMPTS: u32 = 2;

/// Attestation outcome for `(netuid, epoch, miner)` this epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttestationStatus {
    /// Same-epoch Verified — may emit `Score`.
    Verified,
    /// Rejected — must `NoScore(AttestationNotVerified)`.
    Rejected,
    /// Parked — must `NoScore(AttestationNotVerified)` (D13).
    Parked,
    /// Missing / undecided — must `NoScore(AttestationNotVerified)`.
    #[default]
    Missing,
}

/// Inputs for the pure scoring function after a miner attempt (or terminal failure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreInputs {
    /// Subnet netuid (for expected digests).
    pub netuid: u16,
    /// Epoch.
    pub epoch: u64,
    /// Miner hotkey.
    pub miner_hotkey: [u8; KEY_LEN],
    /// Pack id bound into v2 task identity (UTF-8 bytes).
    pub pack_id: Vec<u8>,
    /// Oracle / fixture expected `model.patch` for [`answer_digest_v2`].
    pub expected_model_patch: Vec<u8>,
    /// Attestation status this epoch.
    pub attestation: AttestationStatus,
    /// Challenge-side wall time of the successful attempt, or time to final failure.
    pub duration_ms: u64,
    /// Terminal miner/call outcome.
    pub outcome: CallOutcome,
}

/// Terminal outcome of the challenge↔miner hop (after retries exhausted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallOutcome {
    /// HTTP 200 body accepted for scoring.
    Http200 {
        /// Response `challenge_id`.
        challenge_id: String,
        /// Response epoch.
        epoch: u64,
        /// Response `task_id` (32 bytes).
        task_id: [u8; 32],
        /// Response `answer_digest` (32 bytes) — must equal `answer_digest_v2(model.patch)`.
        answer_digest: [u8; 32],
        /// Response `agent_version`.
        agent_version: String,
    },
    /// Timeout / transport exhausted / deadline.
    Timeout,
    /// HTTP 500 / `agent_internal`.
    MinerError,
    /// Schema / 400 / 403 / hotkey mismatch / field disagree.
    InvalidResponse,
    /// HTTP 429 exhausted.
    RateLimited,
    /// Challenge-side fault after retries.
    ChallengeInternal,
}

/// Score a miner from pure inputs (`AGENT_CHALLENGE` §5.4).
///
/// Bare floating point is forbidden — integer lattice only.
#[must_use]
pub fn score_from_outcome(input: &ScoreInputs) -> ScoreOrAbsence {
    if input.attestation != AttestationStatus::Verified {
        return ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified,
        };
    }

    match &input.outcome {
        CallOutcome::Timeout => ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout,
        },
        CallOutcome::MinerError => ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::MinerError,
        },
        CallOutcome::InvalidResponse => ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::InvalidResponse,
        },
        CallOutcome::RateLimited => ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::RateLimited,
        },
        CallOutcome::ChallengeInternal => ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal,
        },
        CallOutcome::Http200 {
            challenge_id,
            epoch,
            task_id: resp_task_id,
            answer_digest: resp_answer,
            agent_version,
        } => {
            let expected_tid = task_id_v2(
                input.netuid,
                input.epoch,
                &input.miner_hotkey,
                &input.pack_id,
                SCORING_VERSION,
            );
            let expected_answer = answer_digest_v2(&input.expected_model_patch);

            if challenge_id != CHALLENGE_ID
                || *epoch != input.epoch
                || resp_task_id.as_slice() != expected_tid.as_slice()
                || agent_version != "1"
            {
                return ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::InvalidResponse,
                };
            }

            if resp_answer.as_slice() != expected_answer.as_slice() {
                return ScoreOrAbsence::Score { value: 0 };
            }

            score_latency(input.duration_ms)
        }
    }
}

/// Latency credit on the integer lattice after a correct answer.
///
/// ```text
/// if duration_ms > HARD_MS → NoScore(Timeout)
/// if duration_ms <= SOFT_MS → Score(SCORE_MAX)
/// else value = floor(SCORE_MAX * (HARD_MS - duration_ms) / (HARD_MS - SOFT_MS))
/// ```
#[must_use]
pub fn score_latency(duration_ms: u64) -> ScoreOrAbsence {
    if duration_ms > HARD_MS {
        return ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout,
        };
    }
    if duration_ms <= SOFT_MS {
        return ScoreOrAbsence::Score { value: SCORE_MAX };
    }
    // span = HARD_MS - SOFT_MS = 8000
    let span = HARD_MS - SOFT_MS;
    let remaining = HARD_MS - duration_ms;
    // value = (SCORE_MAX * remaining) / span  — u128 intermediate avoids overflow
    let value = (u128::from(SCORE_MAX) * u128::from(remaining)) / u128::from(span);
    // value <= SCORE_MAX by construction (remaining <= span).
    let value_u64 = u64::try_from(value).unwrap_or(0);
    ScoreOrAbsence::Score { value: value_u64 }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::task_gen::{
        answer_digest, answer_digest_v2, task_blob, task_id, task_id_v2, FIXTURE_MODEL_PATCH,
        FIXTURE_PACK_ID, SCORING_VERSION,
    };

    fn base_ok(duration_ms: u64) -> ScoreInputs {
        let miner = [0x11u8; 32];
        let tid = task_id_v2(1, 7, &miner, FIXTURE_PACK_ID, SCORING_VERSION);
        let ans = answer_digest_v2(FIXTURE_MODEL_PATCH);
        ScoreInputs {
            netuid: 1,
            epoch: 7,
            miner_hotkey: miner,
            pack_id: FIXTURE_PACK_ID.to_vec(),
            expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
            attestation: AttestationStatus::Verified,
            duration_ms,
            outcome: CallOutcome::Http200 {
                challenge_id: CHALLENGE_ID.to_owned(),
                epoch: 7,
                task_id: tid,
                answer_digest: ans,
                agent_version: "1".into(),
            },
        }
    }

    #[test]
    fn f1_v2_score_max_at_soft() {
        assert_eq!(
            score_from_outcome(&base_ok(2000)),
            ScoreOrAbsence::Score { value: SCORE_MAX }
        );
    }

    #[test]
    fn f2_v2_score_max_at_zero() {
        assert_eq!(
            score_from_outcome(&base_ok(0)),
            ScoreOrAbsence::Score { value: SCORE_MAX }
        );
    }

    #[test]
    fn f3_v2_midpoint_half() {
        assert_eq!(
            score_from_outcome(&base_ok(6000)),
            ScoreOrAbsence::Score { value: 500_000 }
        );
    }

    #[test]
    fn f4_v2_hard_boundary_zero_score() {
        assert_eq!(
            score_from_outcome(&base_ok(10_000)),
            ScoreOrAbsence::Score { value: 0 }
        );
    }

    #[test]
    fn f5_v2_over_hard_timeout() {
        assert_eq!(
            score_from_outcome(&base_ok(10_001)),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::Timeout
            }
        );
    }

    #[test]
    fn f6_v2_wrong_answer_score_zero() {
        let mut inp = base_ok(2000);
        if let CallOutcome::Http200 {
            ref mut answer_digest,
            ..
        } = inp.outcome
        {
            *answer_digest = [0u8; 32];
        }
        assert_eq!(score_from_outcome(&inp), ScoreOrAbsence::Score { value: 0 });
    }

    #[test]
    fn f7_v2_parked() {
        let mut inp = base_ok(2000);
        inp.attestation = AttestationStatus::Parked;
        assert_eq!(
            score_from_outcome(&inp),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::AttestationNotVerified
            }
        );
    }

    #[test]
    fn f8_v2_missing() {
        let mut inp = base_ok(2000);
        inp.attestation = AttestationStatus::Missing;
        assert_eq!(
            score_from_outcome(&inp),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::AttestationNotVerified
            }
        );
    }

    #[test]
    fn f9_v2_rejected() {
        let mut inp = base_ok(2000);
        inp.attestation = AttestationStatus::Rejected;
        assert_eq!(
            score_from_outcome(&inp),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::AttestationNotVerified
            }
        );
    }

    #[test]
    fn f10_v2_schema_fail() {
        let inp = ScoreInputs {
            netuid: 1,
            epoch: 7,
            miner_hotkey: [0x11u8; 32],
            pack_id: FIXTURE_PACK_ID.to_vec(),
            expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
            attestation: AttestationStatus::Verified,
            duration_ms: 100,
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
    fn f11_v2_second_miner() {
        let miner = [0x22u8; 32];
        let tid = task_id_v2(1, 7, &miner, FIXTURE_PACK_ID, SCORING_VERSION);
        let ans = answer_digest_v2(FIXTURE_MODEL_PATCH);
        let inp = ScoreInputs {
            netuid: 1,
            epoch: 7,
            miner_hotkey: miner,
            pack_id: FIXTURE_PACK_ID.to_vec(),
            expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
            attestation: AttestationStatus::Verified,
            duration_ms: 2000,
            outcome: CallOutcome::Http200 {
                challenge_id: CHALLENGE_ID.to_owned(),
                epoch: 7,
                task_id: tid,
                answer_digest: ans,
                agent_version: "1".into(),
            },
        };
        assert_eq!(
            score_from_outcome(&inp),
            ScoreOrAbsence::Score { value: SCORE_MAX }
        );
    }

    #[test]
    fn latency_integer_only_no_float() {
        // 3000 ms → remaining 7000 → 1_000_000 * 7000 / 8000 = 875_000
        assert_eq!(
            score_latency(3000),
            ScoreOrAbsence::Score { value: 875_000 }
        );
    }

    /// Retired v1 echo answer must not receive full credit under scoring_version 2.
    #[test]
    fn v1_echo_answer_no_longer_validates() {
        let miner = [0x11u8; 32];
        let tid_v2 = task_id_v2(1, 7, &miner, FIXTURE_PACK_ID, SCORING_VERSION);
        // Historical echo: sha256(gbase-agent-answer-v1 ‖ task_blob_v1)
        let tid_v1 = task_id(1, 7, &miner);
        let echo_ans = answer_digest(&task_blob(&tid_v1, 1));
        let inp = ScoreInputs {
            netuid: 1,
            epoch: 7,
            miner_hotkey: miner,
            pack_id: FIXTURE_PACK_ID.to_vec(),
            expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
            attestation: AttestationStatus::Verified,
            duration_ms: 2000,
            outcome: CallOutcome::Http200 {
                challenge_id: CHALLENGE_ID.to_owned(),
                epoch: 7,
                task_id: tid_v2,
                answer_digest: echo_ans,
                agent_version: "1".into(),
            },
        };
        assert_eq!(
            score_from_outcome(&inp),
            ScoreOrAbsence::Score { value: 0 },
            "v1 echo digest must not validate as a correct v2 model.patch answer"
        );
    }

    /// Full v1 identity (v1 task_id + echo answer) is InvalidResponse under v2.
    #[test]
    fn v1_task_id_and_echo_rejected_as_invalid_response() {
        let miner = [0x11u8; 32];
        let tid_v1 = task_id(1, 7, &miner);
        let echo_ans = answer_digest(&task_blob(&tid_v1, 1));
        let inp = ScoreInputs {
            netuid: 1,
            epoch: 7,
            miner_hotkey: miner,
            pack_id: FIXTURE_PACK_ID.to_vec(),
            expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
            attestation: AttestationStatus::Verified,
            duration_ms: 2000,
            outcome: CallOutcome::Http200 {
                challenge_id: CHALLENGE_ID.to_owned(),
                epoch: 7,
                task_id: tid_v1,
                answer_digest: echo_ans,
                agent_version: "1".into(),
            },
        };
        assert_eq!(
            score_from_outcome(&inp),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::InvalidResponse
            }
        );
    }
}
