//! Hypertraining Guard 1: κ=2 numeric kernel gate + precision attestation.
//!
//! # Numeric rule (brief §8.1)
//!
//! ```text
//! max|candidate − ref_fp32|  ≤  κ · max|baseline_same_dtype − ref_fp32|
//! κ = 2
//! ```
//!
//! Evaluated on synthetic CPU GEMV forward outputs and backward-style grads
//! (no real GPU). Precision attestation (`allow_tf32`, accumulate dtype/interval,
//! scaling recipe) is validated mechanically under [`AttestationPolicy`].
//!
//! # Fixtures
//!
//! - [`fixtures::fixture_good_kernel`] — passes κ=2
//! - [`fixtures::fixture_degraded_kernel`] — truncated reduction, fails κ=2
//! - [`fixtures::fixture_attestation_tf32_flagged`] — `allow_tf32=true`, rejected

#![forbid(unsafe_code)]

mod attestation;
mod error;
pub mod fixtures;
mod gate;
pub mod ops;

pub use attestation::{
    validate_attestation, AccumulateDtype, AttestationPolicy, PrecisionAttestation,
    PrecisionFormat, ScalingRecipe,
};
pub use error::{AttestationError, GateError};
pub use gate::{
    evaluate_kappa_surface, gate_kernel, max_abs_diff, require_kappa_pass, KappaSurfaceReport,
    KernelTensors, KAPPA,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        fixture_attestation_ok, fixture_attestation_tf32_flagged, fixture_baseline,
        fixture_degraded_kernel, fixture_good_kernel, fixture_reference,
    };

    #[test]
    fn kappa_constant_is_two() {
        assert!((KAPPA - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn good_kernel_passes_kappa_gate() {
        let reference = fixture_reference();
        let baseline = fixture_baseline();
        let candidate = fixture_good_kernel();
        let reports = gate_kernel(&candidate, &baseline, &reference, KAPPA)
            .expect("good kernel must pass κ=2");
        for r in &reports {
            assert!(r.passed, "{} should pass", r.surface);
            assert!(
                r.candidate_error <= r.budget + f32::EPSILON,
                "{} cand {} budget {}",
                r.surface,
                r.candidate_error,
                r.budget
            );
        }
    }

    #[test]
    fn degraded_truncated_fails_kappa_gate() {
        let reference = fixture_reference();
        let baseline = fixture_baseline();
        let candidate = fixture_degraded_kernel();
        let err = gate_kernel(&candidate, &baseline, &reference, KAPPA)
            .expect_err("truncated reduction must fail κ=2");
        match err {
            GateError::KappaExceeded {
                surface,
                candidate_error,
                baseline_error,
                budget,
                kappa,
            } => {
                assert_eq!(surface, "forward_output");
                assert!(candidate_error > budget);
                assert!((kappa - KAPPA).abs() < f64::EPSILON);
                assert!(baseline_error >= 0.0);
            }
            other => panic!("expected KappaExceeded, got {other:?}"),
        }
    }

    #[test]
    fn degraded_forward_error_exceeds_twice_baseline() {
        let reference = fixture_reference();
        let baseline = fixture_baseline();
        let candidate = fixture_degraded_kernel();
        let report = evaluate_kappa_surface(
            "forward_output",
            &candidate.output,
            &baseline.output,
            &reference.output,
            KAPPA,
        )
        .expect("lengths ok");
        assert!(!report.passed);
        assert!(report.candidate_error > 2.0 * report.baseline_error);
    }

    #[test]
    fn attestation_ok_passes_default_policy() {
        let att = fixture_attestation_ok();
        att.validate(&AttestationPolicy::default())
            .expect("valid attestation");
    }

    #[test]
    fn attestation_tf32_flagged_rejected_when_forced_false() {
        let att = fixture_attestation_tf32_flagged();
        let policy = AttestationPolicy {
            require_allow_tf32_false: true,
            ..AttestationPolicy::default()
        };
        let err = att.validate(&policy).expect_err("tf32 must be rejected");
        assert_eq!(err, AttestationError::Tf32NotAllowed);
    }

    #[test]
    fn attestation_tf32_allowed_when_policy_permits() {
        let att = fixture_attestation_tf32_flagged();
        let policy = AttestationPolicy {
            require_allow_tf32_false: false,
            ..AttestationPolicy::default()
        };
        att.validate(&policy)
            .expect("policy permits allow_tf32=true");
    }

    #[test]
    fn attestation_zero_interval_rejected() {
        let mut att = fixture_attestation_ok();
        att.accumulate_interval = 0;
        let err = att
            .validate(&AttestationPolicy::default())
            .expect_err("interval 0");
        assert_eq!(err, AttestationError::InvalidAccumulateInterval);
    }

    #[test]
    fn length_mismatch_is_error() {
        let err =
            require_kappa_pass("t", &[1.0], &[1.0, 2.0], &[1.0], KAPPA).expect_err("mismatch");
        assert!(matches!(err, GateError::LengthMismatch { .. }));
    }

    #[test]
    fn exact_match_passes_with_zero_baseline_error() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let report = require_kappa_pass("exact", &v, &v, &v, KAPPA).expect("exact");
        assert!(report.passed);
        assert!(report.candidate_error.abs() < f32::EPSILON);
        assert!(report.baseline_error.abs() < f32::EPSILON);
    }
}
