//! κ numeric gate: max|cand − ref| ≤ κ · max|baseline − ref|.

use crate::error::GateError;

/// Normative κ for Guard 1 (`scoring_version = 1`).
pub const KAPPA: f64 = 2.0;

/// One surface compared under the κ rule (forward out, input grad, or param grad).
#[derive(Debug, Clone, PartialEq)]
pub struct KappaSurfaceReport {
    /// Human-readable surface name.
    pub surface: &'static str,
    /// `max|candidate − ref_fp32|`.
    pub candidate_error: f32,
    /// `max|baseline_same_dtype − ref_fp32|`.
    pub baseline_error: f32,
    /// `κ · baseline_error` (pass budget).
    pub budget: f32,
    /// Whether `candidate_error ≤ budget`.
    pub passed: bool,
}

/// Forward outputs plus backward-style gradients for a synthetic kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelTensors {
    /// Forward outputs.
    pub output: Vec<f32>,
    /// ∂L/∂input (bwd-style).
    pub grad_input: Vec<f32>,
    /// ∂L/∂weight (bwd-style).
    pub grad_weight: Vec<f32>,
}

/// Elementwise max absolute difference for equal-length finite slices.
///
/// # Errors
///
/// [`GateError::LengthMismatch`] or [`GateError::NonFinite`].
pub fn max_abs_diff(a: &[f32], b: &[f32], which_a: &'static str) -> Result<f32, GateError> {
    if a.len() != b.len() {
        return Err(GateError::LengthMismatch {
            candidate: a.len(),
            baseline: b.len(),
            reference: b.len(),
        });
    }
    let mut max = 0.0_f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if !x.is_finite() {
            return Err(GateError::NonFinite {
                which: which_a,
                index: i,
            });
        }
        if !y.is_finite() {
            return Err(GateError::NonFinite {
                which: "reference_or_peer",
                index: i,
            });
        }
        max = max.max((x - y).abs());
    }
    Ok(max)
}

/// Evaluate the FlashAttention-style κ rule on one tensor surface.
///
/// # Errors
///
/// Length / non-finite failures from [`max_abs_diff`].
pub fn evaluate_kappa_surface(
    surface: &'static str,
    candidate: &[f32],
    baseline: &[f32],
    reference_fp32: &[f32],
    kappa: f64,
) -> Result<KappaSurfaceReport, GateError> {
    if candidate.len() != baseline.len() || candidate.len() != reference_fp32.len() {
        return Err(GateError::LengthMismatch {
            candidate: candidate.len(),
            baseline: baseline.len(),
            reference: reference_fp32.len(),
        });
    }
    let candidate_error = max_abs_diff(candidate, reference_fp32, "candidate")?;
    let baseline_error = max_abs_diff(baseline, reference_fp32, "baseline")?;
    #[allow(clippy::cast_possible_truncation)] // κ is 2.0; budget stays in f32 error space
    let budget = (kappa as f32) * baseline_error;
    let passed = (f64::from(candidate_error)) <= kappa * f64::from(baseline_error);
    Ok(KappaSurfaceReport {
        surface,
        candidate_error,
        baseline_error,
        budget,
        passed,
    })
}

/// Require a single surface to pass the κ gate.
///
/// # Errors
///
/// [`GateError::KappaExceeded`] when the candidate is outside budget, or length errors.
pub fn require_kappa_pass(
    surface: &'static str,
    candidate: &[f32],
    baseline: &[f32],
    reference_fp32: &[f32],
    kappa: f64,
) -> Result<KappaSurfaceReport, GateError> {
    let report = evaluate_kappa_surface(surface, candidate, baseline, reference_fp32, kappa)?;
    if report.passed {
        Ok(report)
    } else {
        Err(GateError::KappaExceeded {
            surface,
            candidate_error: report.candidate_error,
            baseline_error: report.baseline_error,
            budget: report.budget,
            kappa,
        })
    }
}

/// Gate forward + backward tensors (`output`, `grad_input`, `grad_weight`).
///
/// # Errors
///
/// First failing surface, or length / non-finite errors.
pub fn gate_kernel(
    candidate: &KernelTensors,
    baseline: &KernelTensors,
    reference: &KernelTensors,
    kappa: f64,
) -> Result<[KappaSurfaceReport; 3], GateError> {
    let out = require_kappa_pass(
        "forward_output",
        &candidate.output,
        &baseline.output,
        &reference.output,
        kappa,
    )?;
    let gin = require_kappa_pass(
        "grad_input",
        &candidate.grad_input,
        &baseline.grad_input,
        &reference.grad_input,
        kappa,
    )?;
    let gw = require_kappa_pass(
        "grad_weight",
        &candidate.grad_weight,
        &baseline.grad_weight,
        &reference.grad_weight,
        kappa,
    )?;
    Ok([out, gin, gw])
}
