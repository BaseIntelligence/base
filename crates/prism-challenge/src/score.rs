//! Map BPB / pipeline outcomes to integer leaf scores.

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use prism_challenge_task::SCORE_MAX;

/// Terminal measured outcome after master eval.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineOutcome {
    /// Measured BPB (lower is better).
    Measured { bpb: f64 },
    /// Miner-attributable zero.
    MinerZero,
    /// Operator / challenge fault.
    ChallengeInternal,
    /// Explicit NoScore.
    NoScore { reason: NoScoreReasonCode },
    /// Already resolved.
    Resolved(ScoreOrAbsence),
}

/// Invert BPB into lattice score: lower bpb → higher score.
///
/// Uses a soft map: `score = SCORE_MAX * (1 / (1 + bpb))` clamped.
#[must_use]
pub fn score_from_bpb(bpb: f64) -> u64 {
    if !bpb.is_finite() || bpb < 0.0 {
        return 0;
    }
    let quality = 1.0 / (1.0 + bpb);
    let v = (quality * (SCORE_MAX as f64)).round();
    if v <= 0.0 {
        0
    } else if v >= SCORE_MAX as f64 {
        SCORE_MAX
    } else {
        v as u64
    }
}

/// Map pipeline outcome to leaf payload.
#[must_use]
pub fn score_from_pipeline(outcome: &PipelineOutcome) -> ScoreOrAbsence {
    match outcome {
        PipelineOutcome::Resolved(s) => s.clone(),
        PipelineOutcome::MinerZero => ScoreOrAbsence::Score { value: 0 },
        PipelineOutcome::ChallengeInternal => ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal,
        },
        PipelineOutcome::NoScore { reason } => ScoreOrAbsence::NoScore { reason: *reason },
        PipelineOutcome::Measured { bpb } => ScoreOrAbsence::Score {
            value: score_from_bpb(*bpb),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_bpb_higher_score() {
        let a = score_from_bpb(1.0);
        let b = score_from_bpb(4.0);
        assert!(a > b);
        assert!(a <= SCORE_MAX);
    }

    #[test]
    fn non_finite_zero() {
        assert_eq!(score_from_bpb(f64::NAN), 0);
        assert_eq!(score_from_bpb(-1.0), 0);
    }
}
