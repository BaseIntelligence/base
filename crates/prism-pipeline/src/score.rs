//! Map BPB / pipeline outcomes / G2 benches to integer leaf scores.

use std::sync::OnceLock;

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use prism_challenge_task::{SCORE_MAX, SCORING_VERSION, SCORING_VERSION_V3, SCORING_VERSION_V4};

use crate::composite::CompositeOutcome;

/// Terminal measured outcome after master eval.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineOutcome {
    Measured { bpb: f64 },
    MinerZero,
    ChallengeInternal,
    NoScore { reason: NoScoreReasonCode },
    Resolved(ScoreOrAbsence),
}

/// Soft map: `SCORE_MAX * (1 / (1 + bpb))` clamped.
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

/// `PRISM_SCORING_MODE`: default `benchmarks` (v4); `shadow`=v2 bpb; `composite`=v3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringMode {
    Shadow,
    Composite,
    Benchmarks,
}

impl ScoringMode {
    #[must_use]
    pub fn parse(raw: Option<&str>) -> Self {
        match raw.map(str::trim) {
            Some("shadow") => Self::Shadow,
            Some("composite") => Self::Composite,
            // Default + unknown strings → benchmarks (v4 live leaf).
            _ => Self::Benchmarks,
        }
    }

    #[must_use]
    pub fn from_env() -> Self {
        static MODE: OnceLock<ScoringMode> = OnceLock::new();
        *MODE.get_or_init(|| Self::parse(std::env::var("PRISM_SCORING_MODE").ok().as_deref()))
    }

    #[must_use]
    pub const fn scoring_version(self) -> u16 {
        match self {
            Self::Shadow => SCORING_VERSION,
            Self::Composite => SCORING_VERSION_V3,
            Self::Benchmarks => SCORING_VERSION_V4,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Composite => "composite",
            Self::Benchmarks => "benchmarks",
        }
    }
}

/// Lattice for `mode`. Benchmarks uses `g2_lattice` only (never bpb).
#[must_use]
pub fn final_lattice(
    bpb: f64,
    composite: Option<&CompositeOutcome>,
    mode: ScoringMode,
    g2_lattice: Option<u64>,
) -> u64 {
    match mode {
        ScoringMode::Shadow => score_from_bpb(bpb),
        ScoringMode::Composite => match composite {
            Some(CompositeOutcome::Scored(s)) => s.lattice,
            Some(CompositeOutcome::Ineligible(_)) => 0,
            None => {
                tracing::warn!("composite mode without CompositeOutcome; scoring 0");
                0
            }
        },
        ScoringMode::Benchmarks => g2_lattice.unwrap_or(0),
    }
}

/// Map pipeline outcome to leaf payload (legacy bpb path).
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
    fn scoring_mode_parse_defaults_to_benchmarks() {
        assert_eq!(ScoringMode::parse(None), ScoringMode::Benchmarks);
        assert_eq!(
            ScoringMode::parse(Some("benchmarks")),
            ScoringMode::Benchmarks
        );
        assert_eq!(ScoringMode::parse(Some("shadow")), ScoringMode::Shadow);
        assert_eq!(
            ScoringMode::parse(Some("composite")),
            ScoringMode::Composite
        );
        assert_eq!(
            ScoringMode::parse(Some("COMPOSITE")),
            ScoringMode::Benchmarks
        );
        assert_eq!(ScoringMode::parse(Some("garbage")), ScoringMode::Benchmarks);
    }

    #[test]
    fn scoring_version_marks_mode() {
        assert_eq!(ScoringMode::Shadow.scoring_version(), 2);
        assert_eq!(ScoringMode::Composite.scoring_version(), 3);
        assert_eq!(ScoringMode::Benchmarks.scoring_version(), 4);
    }

    #[test]
    fn shadow_mode_never_changes_v2_number() {
        let composite = sample_scored(424_242);
        for bpb in [0.0, 0.5, 1.0, 4.0] {
            assert_eq!(
                final_lattice(bpb, Some(&composite), ScoringMode::Shadow, Some(1)),
                score_from_bpb(bpb),
            );
            assert_eq!(
                final_lattice(bpb, None, ScoringMode::Shadow, None),
                score_from_bpb(bpb)
            );
        }
    }

    #[test]
    fn composite_mode_uses_lattice_and_fails_closed() {
        let composite = sample_scored(424_242);
        assert_eq!(
            final_lattice(4.0, Some(&composite), ScoringMode::Composite, None),
            424_242,
        );
        let ineligible = CompositeOutcome::Ineligible(sample_ineligible());
        assert_eq!(
            final_lattice(0.5, Some(&ineligible), ScoringMode::Composite, None),
            0
        );
        assert_eq!(final_lattice(0.5, None, ScoringMode::Composite, None), 0);
    }

    #[test]
    fn benchmarks_mode_uses_g2_never_bpb() {
        assert_eq!(
            final_lattice(0.01, None, ScoringMode::Benchmarks, Some(123_456)),
            123_456
        );
        assert_eq!(final_lattice(0.01, None, ScoringMode::Benchmarks, None), 0);
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
