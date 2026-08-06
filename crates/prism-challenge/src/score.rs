//! Map review/agentic outcomes to integer leaf scores (orchestrator path).
//!
//! The pipeline-path scoring (`PipelineOutcome` / `score_from_pipeline` /
//! `score_from_bpb`) lives in `prism-pipeline` and is re-exported by the crate
//! root.

use bundle::NoScoreReasonCode;
use prism_pipeline::{final_lattice, CompositeOutcome, ScoringMode};

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
        /// v3 G1–G8 composite when the battery ran; `None` on the v2 path.
        /// Under `PRISM_SCORING_MODE=shadow` (default) it never moves the
        /// score; under `composite` its lattice becomes the score.
        composite: Option<CompositeOutcome>,
    },
    /// Operator fault (any pipeline/review/similarity/agentic failure).
    ChallengeInternal,
}

/// Map the measured outcome into the integer lattice (scoring v2; v3
/// composite only under `PRISM_SCORING_MODE=composite`).
///
/// The score is **pure bpb** in shadow mode: the LLM review is an
/// anti-cheat / coherence GATE, never a grader — its quality vote and issues
/// are recorded as audit events but never add nor remove points. Agentic
/// `Cheat`/`Suspicious` and cheap similarity `Copied`/`Suspicious` are hard
/// gates (miner-attributable `Score{0}`) checked before any scoring-mode
/// logic. Missing agentic verdict is fail-closed upstream as
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
            similarity,
            agentic,
            composite,
            ..
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
            FinalScore::Score(final_lattice(
                *bpb,
                composite.as_ref(),
                ScoringMode::from_env(),
            ))
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
            composite: None,
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
            composite: None,
        };
        assert_eq!(combine_final(&o), prism_store::FinalScore::Score(0));
    }

    #[test]
    fn hard_gates_precede_composite_scoring() {
        // Even with a scored v3 composite attached, the v2 hard gates fire
        // first, independent of scoring mode.
        let o = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 900,
            similarity: Copied,
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: Some(scored_composite(999_999)),
        };
        assert_eq!(combine_final(&o), prism_store::FinalScore::Score(0));
    }

    #[test]
    fn shadow_mode_ignores_attached_composite() {
        // Default env (PRISM_SCORING_MODE unset → shadow): the v2 number is
        // bit-identical whether or not a composite is attached.
        let bare = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 900,
            similarity: Original,
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
        };
        let with = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 900,
            similarity: Original,
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: Some(scored_composite(1)),
        };
        assert_eq!(combine_final(&bare), combine_final(&with));
        assert_eq!(
            combine_final(&with),
            prism_store::FinalScore::Score(prism_pipeline::score_from_bpb(0.5))
        );
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
            composite: None,
        };
        let lo_same_bpb = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 0,
            similarity: Original,
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
        };
        assert_eq!(combine_final(&hi), combine_final(&lo_same_bpb));

        let worse_bpb = FinalOutcome::Measured {
            bpb: 4.0,
            quality: 1000,
            similarity: Original,
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
        };
        match (combine_final(&hi), combine_final(&worse_bpb)) {
            (prism_store::FinalScore::Score(a), prism_store::FinalScore::Score(b)) => {
                assert!(a > b);
                assert!(a <= SCORE_MAX);
            }
            _ => panic!("unexpected no_score"),
        }
    }

    fn scored_composite(lattice: u64) -> CompositeOutcome {
        CompositeOutcome::Scored(prism_pipeline::CompositeScore {
            groups: Vec::new(),
            composite: 0.5,
            se: 0.0,
            lcb: 0.5,
            lattice,
            gates: prism_pipeline::GateReport::default(),
            anchor_version: 0,
            prereg_hash: None,
            bootstrap: prism_pipeline::BootstrapInfo { b: 1000, seed: 0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use prism_challenge_task::SCORE_MAX;

    #[test]
    fn lower_bpb_higher_score() {
        let a = prism_pipeline::score_from_bpb(1.0);
        let b = prism_pipeline::score_from_bpb(4.0);
        assert!(a > b);
        assert!(a <= SCORE_MAX);
    }
}
