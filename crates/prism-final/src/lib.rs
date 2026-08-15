//! Map review/agentic outcomes to integer leaf scores (orchestrator path).
//!
//! The pipeline-path scoring (`PipelineOutcome` / `score_from_pipeline` /
//! `score_from_bpb`) lives in `prism-pipeline` and is re-exported by the crate
//! root.

#![allow(clippy::doc_markdown)]

mod agentic;
mod g2;

pub use agentic::{build_review_request, corpus_from_rows, gate_corpus_from_rows, same_miner};
pub use g2::{extract_g2_accuracies, score_from_g2_benchmarks, G2_PUBLIC_BENCHES};

use bundle::NoScoreReasonCode;
use prism_pipeline::{final_lattice, CompositeOutcome, ScoringMode};
use prism_review::cheap_similarity_hard_zeros;
use serde_json::Value;

/// Final outcomes after LLM review + similarity + agentic gates (orchestrator).
#[derive(Debug, Clone, PartialEq)]
pub enum FinalOutcome {
    /// Measured metrics + review gates. Leaf uses [`ScoringMode`].
    Measured {
        /// Bits-per-byte on the pinned val cut (lower is better; audit / G1).
        bpb: f64,
        /// LLM quality verdict 0..1000.
        quality: u16,
        /// Cheap single-shot similarity class.
        similarity: prism_review::SimilarityKind,
        /// Cheap LLM similarity confidence `0.0..1.0`.
        similarity_score: f64,
        /// Cheap LLM evidence lines (trope-only → no Score wipe).
        similarity_evidence: Vec<String>,
        /// Agentic anti-cheat verdict (primary gate).
        agentic: challenge_agentic::VerdictKind,
        /// v3 G1–G8 composite when the battery ran; `None` on the v2 path.
        composite: Option<CompositeOutcome>,
        /// METRICS_JSON blob (G2 benches for [`ScoringMode::Benchmarks`]).
        metrics: Option<Value>,
    },
    /// Operator fault (any pipeline/review/similarity/agentic failure).
    ChallengeInternal,
}

/// Map the measured outcome into the integer lattice under `mode`.
///
/// Live default is [`ScoringMode::Benchmarks`] (G2 equal-weight accuracies).
/// Hard gates (miner-attributable `Score{0}`) run *before* scoring-mode logic:
/// - Agentic `Cheat` / `Suspicious`
/// - Cheap LLM `Copied`
/// - Cheap LLM `Suspicious` when
///   `similarity_score >=` [`prism_review::SUSPICIOUS_HARD_ZERO_THRESHOLD`]
///   and evidence is not generic-trope-only
///
/// Missing agentic is fail-closed upstream as [`FinalOutcome::ChallengeInternal`].
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
            metrics,
            ..
        } => {
            if matches!(agentic, VerdictKind::Cheat | VerdictKind::Suspicious) {
                return FinalScore::Score(0);
            }
            if cheap_similarity_hard_zeros(*similarity, *similarity_score, similarity_evidence) {
                return FinalScore::Score(0);
            }
            let g2 = metrics.as_ref().map(score_from_g2_benchmarks);
            FinalScore::Score(final_lattice(*bpb, composite.as_ref(), mode, g2))
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
    use serde_json::json;

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
            metrics: None,
        }
    }

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
            metrics: None,
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
        let o = measured_composite(Copied, Some(scored_composite(999_999)));
        assert_eq!(
            combine_final(&o, ScoringMode::Composite),
            prism_store::FinalScore::Score(0)
        );
    }

    #[test]
    fn shadow_mode_ignores_attached_composite() {
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
        let bare = measured_composite(Original, None);
        assert_eq!(
            combine_final(&bare, ScoringMode::Composite),
            prism_store::FinalScore::Score(0)
        );
    }

    #[test]
    fn benchmarks_mode_uses_g2_never_bpb() {
        let o = FinalOutcome::Measured {
            bpb: 0.01, // would dominate under shadow
            quality: 900,
            similarity: Original,
            similarity_score: 0.0,
            similarity_evidence: vec![],
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
            metrics: Some(json!({
                "org.g2.hellaswag_acc": 0.2,
                "org.g2.piqa_acc": 0.4,
            })),
        };
        assert_eq!(
            combine_final(&o, ScoringMode::Benchmarks),
            prism_store::FinalScore::Score(300_000)
        );
        let no_benches = FinalOutcome::Measured {
            bpb: 0.01,
            quality: 900,
            similarity: Original,
            similarity_score: 0.0,
            similarity_evidence: vec![],
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
            metrics: Some(json!({"bpb": 0.01})),
        };
        assert_eq!(
            combine_final(&no_benches, ScoringMode::Benchmarks),
            prism_store::FinalScore::Score(0),
            "no G2 → fail-closed; never bpb"
        );
    }

    #[test]
    fn quality_never_moves_the_score() {
        let hi = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 900,
            similarity: Original,
            similarity_score: 0.0,
            similarity_evidence: vec![],
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
            metrics: None,
        };
        let lo_same_bpb = FinalOutcome::Measured {
            bpb: 0.5,
            quality: 0,
            similarity: Original,
            similarity_score: 0.0,
            similarity_evidence: vec![],
            agentic: challenge_agentic::VerdictKind::Clean,
            composite: None,
            metrics: None,
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
            metrics: None,
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
