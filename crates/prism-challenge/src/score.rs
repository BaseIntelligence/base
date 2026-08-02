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

/// Combine the three signals into the integer lattice (scoring v2).
///
/// `quality_part = quality/1000`, `bpb_part = 1/(1+bpb)`; combined as
/// `w_llm * quality_part + (1 - w_llm) * bpb_part`, clamped [0,1].
/// Similarity `Copied`/`Suspicious` IS a hard gate handled by the caller
/// (miner-attributable `Score{0}`), documented in `docs/PRISM.md`.
///
/// # Panics
/// Never; weights are clamped.
#[must_use]
pub fn combine_final(outcome: &FinalOutcome, llm_weight: f64) -> prism_store::FinalScore {
    use prism_store::FinalScore;
    let w = llm_weight.clamp(0.0, 1.0);
    match outcome {
        FinalOutcome::ChallengeInternal => {
            FinalScore::NoScore(NoScoreReasonCode::ChallengeInternal as u8)
        }
        FinalOutcome::Measured {
            bpb,
            quality,
            similarity,
        } => {
            if matches!(
                similarity,
                prism_review::SimilarityKind::Copied | prism_review::SimilarityKind::Suspicious
            ) {
                return FinalScore::Score(0);
            }
            let q_bpb = score_from_bpb(*bpb) as f64 / (SCORE_MAX as f64);
            let q_llm = f64::from(*quality) / 1000.0;
            let q = (w * q_llm + (1.0 - w) * q_bpb).clamp(0.0, 1.0);
            FinalScore::Score((q * SCORE_MAX as f64).round() as u64)
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
        assert_eq!(combine_final(&o, 0.3), prism_store::FinalScore::Score(0));
    }

    #[test]
    fn combine_uses_both_signals() {
        let hi = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 900,
            similarity: Original,
        };
        let lo = FinalOutcome::Measured {
            bpb: 4.0,
            quality: 100,
            similarity: Original,
        };
        let hi_s = match combine_final(&hi, 0.3) {
            prism_store::FinalScore::Score(v) => v,
            prism_store::FinalScore::NoScore(_) => panic!("unexpected no_score"),
        };
        let lo_s = match combine_final(&lo, 0.3) {
            prism_store::FinalScore::Score(v) => v,
            prism_store::FinalScore::NoScore(_) => panic!("unexpected no_score"),
        };
        assert!(hi_s > lo_s);
        assert!(hi_s <= SCORE_MAX);
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
