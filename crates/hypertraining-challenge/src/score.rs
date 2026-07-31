//! Attestation gate + integer score mapping into leaf `ScoreOrAbsence`.

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use hypertraining_pay::{score_from_pay_inputs, PayInputs, SCORE_MAX};

/// Re-export pay lattice max for callers / tests.
pub use hypertraining_pay::SCORE_MAX as HT_SCORE_MAX;

/// Attestation outcome for `(netuid, epoch, miner)` this epoch (I1-style gate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttestationStatus {
    /// Same-epoch Verified — may emit `Score`.
    Verified,
    /// Rejected — must `NoScore(AttestationNotVerified)`.
    Rejected,
    /// Parked — must `NoScore(AttestationNotVerified)`.
    Parked,
    /// Missing / undecided — must `NoScore(AttestationNotVerified)` when required.
    #[default]
    Missing,
}

impl AttestationStatus {
    /// Whether Score emission is allowed under attestation policy.
    #[must_use]
    pub const fn allows_score(self) -> bool {
        matches!(self, Self::Verified)
    }
}

/// Terminal measured outcome for one miner after the tournament pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineOutcome {
    /// Guards passed; wall-clocks available for marginal pay.
    Measured {
        /// Champion segment wall-clock (ms).
        t_champ_ms: u64,
        /// Candidate segment wall-clock (ms).
        t_cand_ms: u64,
        /// True when guards 1–3 (and anti-noise allow-measure) passed.
        guards_passed: bool,
    },
    /// Miner-attributable failure → `Score { 0 }`.
    MinerZero,
    /// Operator / challenge fault → `NoScore(ChallengeInternal)`.
    ChallengeInternal,
    /// Explicit reason (timeout, invalid, policy, …).
    NoScore {
        /// Leaf reason code.
        reason: NoScoreReasonCode,
    },
    /// Already-resolved leaf payload (fixture / offline).
    Resolved(ScoreOrAbsence),
}

/// Map a pipeline outcome + attestation into a leaf payload (integer scores only).
#[must_use]
pub fn score_from_pipeline(
    outcome: &PipelineOutcome,
    attestation: AttestationStatus,
    require_attestation: bool,
) -> ScoreOrAbsence {
    if require_attestation && !attestation.allows_score() {
        return ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified,
        };
    }
    match outcome {
        PipelineOutcome::Resolved(s) => s.clone(),
        PipelineOutcome::MinerZero => ScoreOrAbsence::Score { value: 0 },
        PipelineOutcome::ChallengeInternal => ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal,
        },
        PipelineOutcome::NoScore { reason } => ScoreOrAbsence::NoScore { reason: *reason },
        PipelineOutcome::Measured {
            t_champ_ms,
            t_cand_ms,
            guards_passed,
        } => {
            let value = score_from_pay_inputs(&PayInputs {
                t_champ_ms: *t_champ_ms,
                t_cand_ms: *t_cand_ms,
                guards_passed: *guards_passed,
            });
            debug_assert!(value <= SCORE_MAX);
            ScoreOrAbsence::Score { value }
        }
    }
}

/// Silence-is-bug leaf for missing call coverage (D24).
#[must_use]
pub fn missing_call_noscore() -> ScoreOrAbsence {
    ScoreOrAbsence::NoScore {
        reason: NoScoreReasonCode::ChallengeInternal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faster_cand_positive_integer_score() {
        let s = score_from_pipeline(
            &PipelineOutcome::Measured {
                t_champ_ms: 10_000,
                t_cand_ms: 8_000,
                guards_passed: true,
            },
            AttestationStatus::Verified,
            true,
        );
        match s {
            ScoreOrAbsence::Score { value } => {
                assert!(value > 0 && value <= SCORE_MAX);
            }
            ScoreOrAbsence::NoScore { reason } => panic!("expected Score, got NoScore({reason:?})"),
        }
    }

    #[test]
    fn missing_attest_blocks_when_required() {
        let s = score_from_pipeline(
            &PipelineOutcome::Measured {
                t_champ_ms: 10_000,
                t_cand_ms: 1_000,
                guards_passed: true,
            },
            AttestationStatus::Missing,
            true,
        );
        assert_eq!(
            s,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::AttestationNotVerified
            }
        );
    }

    #[test]
    fn missing_attest_allowed_when_not_required() {
        let s = score_from_pipeline(
            &PipelineOutcome::Measured {
                t_champ_ms: 10_000,
                t_cand_ms: 9_000,
                guards_passed: true,
            },
            AttestationStatus::Missing,
            false,
        );
        assert!(matches!(s, ScoreOrAbsence::Score { value } if value > 0));
    }
}
