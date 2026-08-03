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

/// Final outcomes after LLM review + similarity gates (orchestrator v2).
#[derive(Debug, Clone, PartialEq)]
pub enum FinalOutcome {
    /// Measured bpb + LLM quality (0..1000) + similarity class.
    Measured {
        /// Bits-per-byte on the pinned val cut (lower is better).
        bpb: f64,
        /// LLM quality verdict 0..1000.
        quality: u16,
        /// Similarity class.
        similarity: prism_review::SimilarityKind,
    },
    /// Operator fault (any pipeline/review/similarity failure).
    ChallengeInternal,
}

/// Map the measured outcome into the integer lattice (scoring v2).
///
/// The score is **pure bpb**: the LLM review is an anti-cheat / coherence
/// GATE, never a grader — its quality vote and issues are recorded as audit
/// events but never add nor remove points. Similarity `Copied`/`Suspicious`
/// is the hard gate (miner-attributable `Score{0}`).
///
/// # Panics
/// Never.
#[must_use]
pub fn combine_final(outcome: &FinalOutcome) -> prism_store::FinalScore {
    use prism_store::FinalScore;
    match outcome {
        FinalOutcome::ChallengeInternal => {
            FinalScore::NoScore(NoScoreReasonCode::ChallengeInternal as u8)
        }
        FinalOutcome::Measured {
            bpb,
            quality: _,
            similarity,
        } => {
            if matches!(
                similarity,
                prism_review::SimilarityKind::Copied | prism_review::SimilarityKind::Suspicious
            ) {
                return FinalScore::Score(0);
            }
            FinalScore::Score(score_from_bpb(*bpb))
        }
    }
}

#[cfg(test)]
mod final_tests {
    use super::*;
    use prism_review::SimilarityKind::Copied;
    use prism_review::SimilarityKind::Original;

    #[test]
    fn copied_is_hard_zero() {
        let o = FinalOutcome::Measured {
            bpb: 1.0,
            quality: 900,
            similarity: Copied,
        };
        assert_eq!(combine_final(&o), prism_store::FinalScore::Score(0));
    }

    #[test]
    fn quality_never_moves_the_score() {
        // Anti-cheat review gates eligibility only; the quality vote must not
        // shift the integer score by a single point.
        let hi = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 900,
            similarity: Original,
        };
        let lo_same_bpb = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 0,
            similarity: Original,
        };
        assert_eq!(combine_final(&hi), combine_final(&lo_same_bpb));

        let worse_bpb = FinalOutcome::Measured {
            bpb: 4.0,
            quality: 1000,
            similarity: Original,
        };
        match (combine_final(&hi), combine_final(&worse_bpb)) {
            (prism_store::FinalScore::Score(a), prism_store::FinalScore::Score(b)) => {
                assert!(a > b);
                assert!(a <= SCORE_MAX);
            }
            _ => panic!("unexpected no_score"),
        }
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
