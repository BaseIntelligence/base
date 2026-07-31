//! Errors for the kernel numeric gate and precision attestation.

use thiserror::Error;

/// Failures while validating a [`crate::PrecisionAttestation`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AttestationError {
    /// Policy requires `allow_tf32 = false` but the attestation set it true.
    #[error("allow_tf32=true rejected when policy requires allow_tf32=false")]
    Tf32NotAllowed,
    /// Policy requires FP32 accumulation.
    #[error("accumulate_dtype must be fp32 under current policy")]
    AccumulateDtypeRejected,
    /// `accumulate_interval` must be strictly positive.
    #[error("accumulate_interval must be > 0")]
    InvalidAccumulateInterval,
    /// Interval below the policy minimum.
    #[error("accumulate_interval {got} below policy minimum {min}")]
    AccumulateIntervalTooSmall { got: u32, min: u32 },
}

/// Failures while comparing tensors under the κ gate.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GateError {
    /// Candidate / baseline / reference lengths disagree.
    #[error(
        "tensor length mismatch: candidate={candidate} baseline={baseline} reference={reference}"
    )]
    LengthMismatch {
        candidate: usize,
        baseline: usize,
        reference: usize,
    },
    /// Non-finite value in a compared tensor.
    #[error("non-finite value in {which} at index {index}")]
    NonFinite { which: &'static str, index: usize },
    /// Candidate error exceeds κ · baseline error.
    #[error(
        "kappa gate failed on {surface}: cand_err={candidate_error} baseline_err={baseline_error} budget={budget} kappa={kappa}"
    )]
    KappaExceeded {
        surface: &'static str,
        candidate_error: f32,
        baseline_error: f32,
        budget: f32,
        kappa: f64,
    },
}
