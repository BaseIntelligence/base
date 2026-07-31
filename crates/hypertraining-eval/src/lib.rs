//! Hypertraining Guards 2–3: quality non-inferiority + physical plausibility.
//!
//! # Guard 2 (brief §8.2)
//!
//! ```text
//! d_i = L_champion^(i) − L_candidate^(i)     paired seeds, i = 1..K
//! H0 : E[d] ≤ −ε
//! Promotion allowed if H0 is rejected — one-sided paired test, α = 0.05
//! ε = min(0.25% · L, 0.5 · σ̂_d)
//! ```
//!
//! Primary metric is **continuous validation loss** supplied by the validator
//! harness (fixture numbers in tests). Miner-reported metrics are never inputs.
//!
//! # Guard 3 (brief §8.3)
//!
//! Compare sim / Nsight telemetry counters to an analytic model:
//! DRAM bytes, tensor-core ops, MMA family, roofline speedup vs peak bandwidth.
//!
//! # Numeric policy
//!
//! Public loss / score surfaces use fixed-point micro-units ([`LossMicro`]).
//! Internal statistics (mean, sd, t) may use `f64`; that path is documented and
//! never exposed as a leaf score (leaf scoring lives in `hypertraining-pay`).

#![forbid(unsafe_code)]
// Guard 2 internal stats widen fixed-point micro-loss to f64; not a leaf score path.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

mod epsilon;
mod error;
pub mod fixtures;
mod physics;
mod quality;
mod types;
mod verdict;

pub use epsilon::{
    compute_epsilon_micro, EpsilonParams, DEFAULT_SIGMA_D_ABS_MICRO, DEFAULT_SIGMA_D_REL_BPS,
    LOSS_REL_BUDGET_BPS, MUST_CALIBRATE_NOTE,
};
pub use error::EvalError;
pub use physics::{
    check_physics, AnalyticModel, PhysicsTelemetry, DRAM_MIN_RATIO_BPS, TENSOR_MIN_RATIO_BPS,
};
pub use quality::{paired_differences, quality_non_inferiority, QualityReport, ALPHA};
pub use types::{EvalRun, LossMicro, MICRO_PER_UNIT};
pub use verdict::{evaluate_candidate, EvalVerdict, RejectReason};

pub use hypertraining_cluster::MmaFamily;

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "hypertraining-eval"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        fixture_equal_quality_pairs, fixture_implausible_physics, fixture_plausible_physics,
        fixture_worse_candidate_pairs,
    };

    #[test]
    fn crate_name_is_hypertraining_eval() {
        assert_eq!(crate_name(), "hypertraining-eval");
    }

    #[test]
    fn promote_allowed_when_equal_loss_and_plausible_physics() {
        let (champ, cand) = fixture_equal_quality_pairs();
        let tel = fixture_plausible_physics();
        let model = AnalyticModel::from_telemetry_baseline(&tel, 3_000);
        let v = evaluate_candidate(
            &champ,
            &cand,
            &tel,
            &model,
            &EpsilonParams::must_calibrate_defaults(),
        )
        .expect("eval");
        assert!(
            v.quality_ok,
            "equal loss must reject H0 (non-inferior): {:?}",
            v.reasons
        );
        assert!(v.physics_ok, "plausible physics: {:?}", v.reasons);
        assert!(v.promote_allowed());
        assert!(v.reasons.is_empty());
    }

    #[test]
    fn worse_loss_rejects_quality_guard() {
        let (champ, cand) = fixture_worse_candidate_pairs();
        let tel = fixture_plausible_physics();
        let model = AnalyticModel::from_telemetry_baseline(&tel, 3_000);
        let v = evaluate_candidate(
            &champ,
            &cand,
            &tel,
            &model,
            &EpsilonParams::must_calibrate_defaults(),
        )
        .expect("eval");
        assert!(!v.quality_ok, "worse candidate must fail Guard 2: {v:?}");
        assert!(v.physics_ok);
        assert!(!v.promote_allowed());
        assert!(
            v.reasons
                .iter()
                .any(|r| matches!(r, RejectReason::QualityInferior { .. })),
            "expected QualityInferior, got {:?}",
            v.reasons
        );
    }

    #[test]
    fn implausible_speedup_fails_guard_3() {
        let (champ, cand) = fixture_equal_quality_pairs();
        let tel = fixture_implausible_physics();
        let model = AnalyticModel {
            expected_dram_bytes: tel.dram_bytes,
            expected_tensor_ops: tel.tensor_ops,
            required_mma: tel.mma_family,
            reference_wallclock_ms: 100_000,
            max_plausible_speedup_milli: 2_000,
        };
        let v = evaluate_candidate(
            &champ,
            &cand,
            &tel,
            &model,
            &EpsilonParams::must_calibrate_defaults(),
        )
        .expect("eval");
        assert!(v.quality_ok, "quality should still pass: {:?}", v.reasons);
        assert!(!v.physics_ok, "roofline must fail: {:?}", v.reasons);
        assert!(!v.promote_allowed());
        assert!(
            v.reasons
                .iter()
                .any(|r| matches!(r, RejectReason::RooflineImplausible { .. })),
            "expected RooflineImplausible, got {:?}",
            v.reasons
        );
    }

    #[test]
    fn primary_metric_is_validator_val_loss_not_miner_fields() {
        let run = EvalRun {
            seed: 1,
            val_loss_micro: 1_500_000,
        };
        assert_eq!(run.val_loss_micro, 1_500_000);
    }

    #[test]
    fn epsilon_uses_min_of_relative_and_half_sigma() {
        let l = 2_000_000_i64;
        let sigma = 100_000_i64;
        let eps = compute_epsilon_micro(l, sigma, &EpsilonParams::must_calibrate_defaults());
        assert_eq!(eps, 5_000);
        let eps2 = compute_epsilon_micro(l, 1_000, &EpsilonParams::must_calibrate_defaults());
        assert_eq!(eps2, 500);
    }

    #[test]
    fn must_calibrate_defaults_are_documented_constants() {
        assert_eq!(LOSS_REL_BUDGET_BPS, 25);
        assert_eq!(DEFAULT_SIGMA_D_REL_BPS, 50);
        assert!(MUST_CALIBRATE_NOTE.contains("MUST_CALIBRATE"));
        let p = EpsilonParams::must_calibrate_defaults();
        assert_eq!(p.loss_rel_budget_bps, 25);
    }

    #[test]
    fn skipped_dram_work_fails_physics() {
        let (champ, cand) = fixture_equal_quality_pairs();
        let mut tel = fixture_plausible_physics();
        let model = AnalyticModel::from_telemetry_baseline(&tel, 3_000);
        tel.dram_bytes /= 10;
        let v = evaluate_candidate(
            &champ,
            &cand,
            &tel,
            &model,
            &EpsilonParams::must_calibrate_defaults(),
        )
        .expect("eval");
        assert!(!v.physics_ok);
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::DramBytesImplausible { .. })));
    }

    #[test]
    fn mma_family_downgrade_fails_physics() {
        let (champ, cand) = fixture_equal_quality_pairs();
        let mut tel = fixture_plausible_physics();
        let model = AnalyticModel::from_telemetry_baseline(&tel, 3_000);
        tel.mma_family = MmaFamily::Tf32;
        let v = evaluate_candidate(
            &champ,
            &cand,
            &tel,
            &model,
            &EpsilonParams::must_calibrate_defaults(),
        )
        .expect("eval");
        assert!(!v.physics_ok);
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::MmaFamilyMismatch { .. })));
    }

    #[test]
    fn length_mismatch_is_error() {
        let champ = vec![EvalRun {
            seed: 1,
            val_loss_micro: 1,
        }];
        let cand = vec![
            EvalRun {
                seed: 1,
                val_loss_micro: 1,
            },
            EvalRun {
                seed: 2,
                val_loss_micro: 1,
            },
        ];
        let tel = fixture_plausible_physics();
        let model = AnalyticModel::from_telemetry_baseline(&tel, 3_000);
        let err = evaluate_candidate(
            &champ,
            &cand,
            &tel,
            &model,
            &EpsilonParams::must_calibrate_defaults(),
        )
        .expect_err("mismatch");
        assert!(matches!(err, EvalError::PairedLengthMismatch { .. }));
    }
}
