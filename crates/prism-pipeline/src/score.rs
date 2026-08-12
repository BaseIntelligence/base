//! Map BPB / pipeline outcomes to integer leaf scores (pipeline path).

use std::sync::OnceLock;

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use prism_challenge_task::{SCORE_MAX, SCORING_VERSION, SCORING_VERSION_V3};

use crate::composite::CompositeOutcome;

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

/// v3 scoring-mode selection, read once from `PRISM_SCORING_MODE`
/// (`shadow` default | `composite`).
///
/// In `shadow` mode the composite (when present) is computed/stored by
/// callers but the leaf score stays `score_from_bpb` — v2 remains live until
/// anchors are calibrated and governance flips the mode. In `composite` mode
/// the v3 lattice is the leaf score and rows carry [`SCORING_VERSION_V3`]
/// (via [`ScoringMode::scoring_version`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringMode {
    /// v2 stays the score; v3 composite is observed only.
    Shadow,
    /// v3 lattice is the score (fail-closed to 0 without a scored composite).
    Composite,
}

impl ScoringMode {
    /// Parse the raw env value: exactly `composite` selects composite mode.
    #[must_use]
    pub fn parse(raw: Option<&str>) -> Self {
        match raw {
            Some("composite") => Self::Composite,
            _ => Self::Shadow,
        }
    }

    /// Mode for this process, read from `PRISM_SCORING_MODE` once.
    #[must_use]
    pub fn from_env() -> Self {
        static MODE: OnceLock<ScoringMode> = OnceLock::new();
        *MODE.get_or_init(|| Self::parse(std::env::var("PRISM_SCORING_MODE").ok().as_deref()))
    }

    /// `challenge_scoring_version` stamped on rows scored under this mode.
    #[must_use]
    pub const fn scoring_version(self) -> u16 {
        match self {
            Self::Shadow => SCORING_VERSION,
            Self::Composite => SCORING_VERSION_V3,
        }
    }

    /// Stable label for logs, run rows, and terminal events.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Composite => "composite",
        }
    }
}

/// Final integer lattice score after the (caller-applied) hard-zero gates,
/// honoring the scoring mode.
///
/// - `shadow`: always the v2 `score_from_bpb` number (bit-identical).
/// - `composite`: the v3 lattice of an attached scored composite; an
///   ineligible or missing composite fails closed to 0 (emission burns).
#[must_use]
pub fn final_lattice(bpb: f64, composite: Option<&CompositeOutcome>, mode: ScoringMode) -> u64 {
    match mode {
        ScoringMode::Shadow => score_from_bpb(bpb),
        ScoringMode::Composite => match composite {
            Some(CompositeOutcome::Scored(s)) => s.lattice,
            Some(CompositeOutcome::Ineligible(_)) => 0,
            None => {
                tracing::warn!(
                    "composite scoring mode but no CompositeOutcome attached; scoring 0"
                );
                0
            }
        },
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

    #[test]
    fn scoring_mode_parse_defaults_to_shadow() {
        assert_eq!(ScoringMode::parse(None), ScoringMode::Shadow);
        assert_eq!(ScoringMode::parse(Some("shadow")), ScoringMode::Shadow);
        assert_eq!(
            ScoringMode::parse(Some("composite")),
            ScoringMode::Composite
        );
        assert_eq!(ScoringMode::parse(Some("COMPOSITE")), ScoringMode::Shadow);
        assert_eq!(ScoringMode::parse(Some("garbage")), ScoringMode::Shadow);
    }

    #[test]
    fn scoring_version_marks_mode() {
        assert_eq!(ScoringMode::Shadow.scoring_version(), SCORING_VERSION);
        assert_eq!(ScoringMode::Composite.scoring_version(), SCORING_VERSION_V3);
        assert_eq!(ScoringMode::Shadow.scoring_version(), 2);
        assert_eq!(ScoringMode::Composite.scoring_version(), 3);
    }

    #[test]
    fn shadow_mode_never_changes_v2_number() {
        let composite = sample_scored(424_242);
        for bpb in [0.0, 0.5, 1.0, 4.0] {
            assert_eq!(
                final_lattice(bpb, Some(&composite), ScoringMode::Shadow),
                score_from_bpb(bpb),
                "shadow ignores the composite"
            );
            assert_eq!(
                final_lattice(bpb, None, ScoringMode::Shadow),
                score_from_bpb(bpb)
            );
        }
    }

    #[test]
    fn composite_mode_uses_lattice_and_fails_closed() {
        let composite = sample_scored(424_242);
        assert_eq!(
            final_lattice(4.0, Some(&composite), ScoringMode::Composite),
            424_242,
            "composite mode emits the v3 lattice, not bpb"
        );
        let ineligible = CompositeOutcome::Ineligible(sample_ineligible());
        assert_eq!(
            final_lattice(0.5, Some(&ineligible), ScoringMode::Composite),
            0
        );
        assert_eq!(final_lattice(0.5, None, ScoringMode::Composite), 0);
    }

    fn sample_scored(lattice: u64) -> CompositeOutcome {
        CompositeOutcome::Scored(crate::composite::CompositeScore {
            groups: Vec::new(),
            composite: 0.8,
            se: 0.01,
            lcb: 0.78,
            lattice,
            gates: crate::composite::GateReport::default(),
            anchor_version: 0,
            prereg_hash: None,
            bootstrap: crate::composite::BootstrapInfo { b: 1000, seed: 1 },
        })
    }

    fn sample_ineligible() -> crate::composite::Ineligible {
        crate::composite::Ineligible {
            reasons: Vec::new(),
            groups: Vec::new(),
            composite: 0.3,
            gates: crate::composite::GateReport::default(),
            anchor_version: 0,
            prereg_hash: None,
        }
    }
}
