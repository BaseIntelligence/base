//! Map review/agentic outcomes to integer leaf scores (orchestrator path).
//!
//! The pipeline-path scoring (`PipelineOutcome` / `score_from_pipeline` /
//! `score_from_bpb`) lives in `prism-pipeline` and is re-exported by the crate
//! root.

#![allow(clippy::doc_markdown)]

mod agentic;
pub use agentic::{build_review_request, corpus_from_rows, gate_corpus_from_rows, same_miner};

use bundle::NoScoreReasonCode;
use prism_pipeline::{final_lattice, CompositeOutcome, ScoringMode};
use prism_review::cheap_similarity_hard_zeros;

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
        /// Cheap LLM similarity confidence `0.0..1.0` (compared for
        /// `Suspicious`; see [`prism_review::SUSPICIOUS_HARD_ZERO_THRESHOLD`]).
        similarity_score: f64,
        /// Cheap LLM evidence lines (trope-only → no Score wipe).
        similarity_evidence: Vec<String>,
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

/// Map the measured outcome into the integer lattice under `mode`
/// (`OrchestratorConfig.scoring_mode`; v2 bit-identical under
/// [`ScoringMode::Shadow`]).
///
/// The score is **pure bpb** in shadow mode: the LLM review is an
/// anti-cheat / coherence GATE, never a grader — its quality vote and issues
/// are recorded as audit events but never add nor remove points.
///
/// Hard gates (miner-attributable `Score{0}`), all checked *before* any
/// scoring-mode logic:
/// - Agentic `Cheat` / `Suspicious` (AST bands already applied upstream).
/// - Cheap LLM `Copied`.
/// - Cheap LLM `Suspicious` **only** when
///   `similarity_score >=` [`prism_review::SUSPICIOUS_HARD_ZERO_THRESHOLD`]
///   `(0.9)` **and** evidence is not generic-trope-only (RMSNorm / SwiGLU /
///   LayerNorm / …). Below the threshold (e.g. 0.7 tropes) → no score wipe.
///
/// Missing agentic verdict is fail-closed upstream as
/// [`FinalOutcome::ChallengeInternal`].
///
/// # Panics
/// Never.
#[must_use]
pub fn combine_final(outcome: &FinalOutcome, mode: ScoringMode) -> prism_store::FinalScore {
    use challenge_agentic::VerdictKind;
    use prism_store::FinalScore;
    match outcome {
        FinalOutcome::ChallengeInternal => {
            FinalScore::NoScore(NoScoreReasonCode::ChallengeInternal as u8)
        }
        FinalOutcome::Measured {
            bpb,
            similarity,
            similarity_score,
            similarity_evidence,
            agentic,
            composite,
            ..
        } => {
            if matches!(agentic, VerdictKind::Cheat | VerdictKind::Suspicious) {
                return FinalScore::Score(0);
            }
            if cheap_similarity_hard_zeros(*similarity, *similarity_score, similarity_evidence) {
                return FinalScore::Score(0);
            }
            FinalScore::Score(final_lattice(*bpb, composite.as_ref(), mode))
        }
    }
}

#[cfg(test)]
mod final_tests {
    use super::*;
    use prism_challenge_task::SCORE_MAX;
    use prism_review::SimilarityKind::Copied;
    use prism_review::SimilarityKind::Original;
    use prism_review::SimilarityKind::Suspicious;

    fn measured(
        similarity: prism_review::SimilarityKind,
        similarity_score: f64,
        evidence: &[&str],
        agentic: challenge_agentic::VerdictKind,
    ) -> FinalOutcome {
        FinalOutcome::Measured {
            bpb: 1.0,
            quality: 900,
            similarity,
            similarity_score,
            similarity_evidence: evidence.iter().map(|s| (*s).to_owned()).collect(),
            agentic,
            composite: None,
        }
    }

    /// Clean-gate row at bpb 0.5 with a v3 composite attached (or not), for the
    /// scoring-mode tests. `similarity` lets one case trip the v2 hard gate.
    fn measured_composite(
        similarity: prism_review::SimilarityKind,
        composite: Option<CompositeOutcome>,
    ) -> FinalOutcome {
        FinalOutcome::Measured {
            bpb: 0.5,
            quality: 900,
            similarity,
            similarity_score: 0.0,
            similarity_evidence: vec![],
            agentic: challenge_agentic::VerdictKind::Clean,
            composite,
        }
    }

    #[test]
    fn copied_is_hard_zero() {
        let o = measured(Copied, 0.5, &[], challenge_agentic::VerdictKind::Clean);
        assert_eq!(
            combine_final(&o, ScoringMode::Shadow),
            prism_store::FinalScore::Score(0)
        );
    }

    #[test]
    fn suspicious_0_7_tropes_not_hard_zero() {
        let o = measured(
            Suspicious,
            0.7,
            &[
                "RMSNorm usage",
                "SwiGLU feed-forward",
                "Layer normalization",
            ],
            challenge_agentic::VerdictKind::Clean,
        );
        assert!(matches!(
            combine_final(&o, ScoringMode::Shadow),
            prism_store::FinalScore::Score(v) if v > 0
        ));
    }

    #[test]
    fn suspicious_0_99_real_copy_is_hard_zero() {
        let o = measured(
            Suspicious,
            0.99,
            &["same custom DualPathBlock wiring as subm:aabbccdd"],
            challenge_agentic::VerdictKind::Clean,
        );
        assert_eq!(
            combine_final(&o, ScoringMode::Shadow),
            prism_store::FinalScore::Score(0)
        );
    }

    #[test]
    fn agentic_cheat_is_hard_zero() {
        let o = measured(Original, 0.0, &[], challenge_agentic::VerdictKind::Cheat);
        assert_eq!(
            combine_final(&o, ScoringMode::Shadow),
            prism_store::FinalScore::Score(0)
        );
    }

    #[test]
    fn hard_gates_precede_composite_scoring() {
        // Even with a scored v3 composite attached, the v2 hard gates fire
        // first, independent of scoring mode.
        let o = measured_composite(Copied, Some(scored_composite(999_999)));
        assert_eq!(
            combine_final(&o, ScoringMode::Composite),
            prism_store::FinalScore::Score(0)
        );
    }

    #[test]
    fn shadow_mode_ignores_attached_composite() {
        // Shadow (the default): the v2 number is bit-identical whether or
        // not a composite is attached.
        let bare = measured_composite(Original, None);
        let with = measured_composite(Original, Some(scored_composite(1)));
        assert_eq!(
            combine_final(&bare, ScoringMode::Shadow),
            combine_final(&with, ScoringMode::Shadow)
        );
        assert_eq!(
            combine_final(&with, ScoringMode::Shadow),
            prism_store::FinalScore::Score(prism_pipeline::score_from_bpb(0.5))
        );
    }

    #[test]
    fn composite_mode_uses_attached_lattice_else_zero() {
        let o = measured_composite(Original, Some(scored_composite(777)));
        assert_eq!(
            combine_final(&o, ScoringMode::Composite),
            prism_store::FinalScore::Score(777)
        );
        // Missing composite fails closed to 0 under composite mode.
        let bare = measured_composite(Original, None);
        assert_eq!(
            combine_final(&bare, ScoringMode::Composite),
            prism_store::FinalScore::Score(0)
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
            similarity_score: 0.0,
            similarity_evidence: vec![],
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
        };
        let lo_same_bpb = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 0,
            similarity: Original,
            similarity_score: 0.0,
            similarity_evidence: vec![],
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
        };
        assert_eq!(
            combine_final(&hi, ScoringMode::Shadow),
            combine_final(&lo_same_bpb, ScoringMode::Shadow)
        );

        let worse_bpb = FinalOutcome::Measured {
            bpb: 4.0,
            quality: 1000,
            similarity: Original,
            similarity_score: 0.0,
            similarity_evidence: vec![],
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
        };
        match (
            combine_final(&hi, ScoringMode::Shadow),
            combine_final(&worse_bpb, ScoringMode::Shadow),
        ) {
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
