//! Map review/agentic outcomes to integer leaf scores (orchestrator path).
//!
//! The pipeline-path scoring (`PipelineOutcome` / `score_from_pipeline` /
//! `score_from_bpb`) lives in `prism-pipeline` and is re-exported by the crate
//! root.

use bundle::NoScoreReasonCode;
use prism_pipeline::score_from_bpb;

/// Final outcomes after LLM review + similarity + agentic gates (orchestrator v2).
#[derive(Debug, Clone, PartialEq)]
pub enum FinalOutcome {
    /// Measured bpb + LLM quality (0..1000) + similarity + agentic verdict.
    Measured {
        /// Bits-per-byte on the pinned val cut (lower is better).
        bpb: f64,
        /// LLM quality verdict 0..1000.
        quality: u16,
        /// Cheap single-shot similarity class.
        similarity: prism_review::SimilarityKind,
        /// Agentic anti-cheat verdict (primary gate).
        agentic: challenge_agentic::VerdictKind,
    },
    /// Operator fault (any pipeline/review/similarity/agentic failure).
    ChallengeInternal,
}

/// Map the measured outcome into the integer lattice (scoring v2).
///
/// The score is **pure bpb**: the LLM review is an anti-cheat / coherence
/// GATE, never a grader — its quality vote and issues are recorded as audit
/// events but never add nor remove points. Agentic `Cheat`/`Suspicious` and
/// cheap similarity `Copied`/`Suspicious` are hard gates (miner-attributable
/// `Score{0}`). Missing agentic verdict is fail-closed upstream as
/// [`FinalOutcome::ChallengeInternal`].
///
/// # Panics
/// Never.
#[must_use]
pub fn combine_final(outcome: &FinalOutcome) -> prism_store::FinalScore {
    use challenge_agentic::VerdictKind;
    use prism_store::FinalScore;
    match outcome {
        FinalOutcome::ChallengeInternal => {
            FinalScore::NoScore(NoScoreReasonCode::ChallengeInternal as u8)
        }
        FinalOutcome::Measured {
            bpb,
            quality: _,
            similarity,
            agentic,
        } => {
            if matches!(agentic, VerdictKind::Cheat | VerdictKind::Suspicious) {
                return FinalScore::Score(0);
            }
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
    use prism_challenge_task::SCORE_MAX;
    use prism_review::SimilarityKind::Copied;
    use prism_review::SimilarityKind::Original;

    #[test]
    fn copied_is_hard_zero() {
        let o = FinalOutcome::Measured {
            bpb: 1.0,
            quality: 900,
            similarity: Copied,
            agentic: challenge_agentic::VerdictKind::Clean,
        };
        assert_eq!(combine_final(&o), prism_store::FinalScore::Score(0));
    }

    #[test]
    fn agentic_cheat_is_hard_zero() {
        let o = FinalOutcome::Measured {
            bpb: 1.0,
            quality: 900,
            similarity: Original,
            agentic: challenge_agentic::VerdictKind::Cheat,
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
            agentic: challenge_agentic::VerdictKind::Clean,
        };
        let lo_same_bpb = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 0,
            similarity: Original,
            agentic: challenge_agentic::VerdictKind::Clean,
        };
        assert_eq!(combine_final(&hi), combine_final(&lo_same_bpb));

        let worse_bpb = FinalOutcome::Measured {
            bpb: 4.0,
            quality: 1000,
            similarity: Original,
            agentic: challenge_agentic::VerdictKind::Clean,
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
    use prism_challenge_task::SCORE_MAX;

    #[test]
    fn lower_bpb_higher_score() {
        let a = score_from_bpb(1.0);
        let b = score_from_bpb(4.0);
        assert!(a > b);
        assert!(a <= SCORE_MAX);
    }
}
