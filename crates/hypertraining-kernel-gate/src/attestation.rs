//! Precision attestation types and policy validation (brief §7 / §8.3).

use crate::error::AttestationError;

/// Declared compute format for a miner submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionFormat {
    /// FP8 E4M3 primary path.
    Fp8E4m3,
    /// BF16 primary path.
    Bf16,
    /// Mixed precision recipe.
    Mixed,
}

/// Dtype used for loss / gradient accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccumulateDtype {
    /// IEEE FP32 accumulation (required by default policy).
    Fp32,
}

/// Loss-scaling recipe name (binding declaration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalingRecipe {
    /// Delayed scaling.
    Delayed,
    /// Current / instantaneous scaling.
    Current,
    /// Block-wise scaling.
    Block,
}

/// Binding precision attestation from the miner submit body (brief §7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecisionAttestation {
    /// Primary compute format.
    pub format: PrecisionFormat,
    /// Accumulation dtype.
    pub accumulate_dtype: AccumulateDtype,
    /// Steps between scaling updates / accumulate flushes.
    pub accumulate_interval: u32,
    /// Named scaling recipe.
    pub scaling_recipe: ScalingRecipe,
    /// Whether TF32 tensor-core paths are allowed.
    pub allow_tf32: bool,
}

/// Mechanical checks applied to [`PrecisionAttestation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationPolicy {
    /// When true, reject any attestation with `allow_tf32 = true`.
    pub require_allow_tf32_false: bool,
    /// When true, require [`AccumulateDtype::Fp32`].
    pub require_accumulate_fp32: bool,
    /// Minimum accepted `accumulate_interval` (inclusive).
    pub min_accumulate_interval: u32,
}

impl Default for AttestationPolicy {
    fn default() -> Self {
        Self {
            require_allow_tf32_false: true,
            require_accumulate_fp32: true,
            min_accumulate_interval: 1,
        }
    }
}

impl PrecisionAttestation {
    /// Validate this attestation against `policy`.
    ///
    /// # Errors
    ///
    /// Returns [`AttestationError`] when a field violates the policy.
    pub fn validate(&self, policy: &AttestationPolicy) -> Result<(), AttestationError> {
        validate_attestation(self, policy)
    }
}

/// Validate `attestation` under `policy`.
///
/// # Errors
///
/// See [`AttestationError`].
pub fn validate_attestation(
    attestation: &PrecisionAttestation,
    policy: &AttestationPolicy,
) -> Result<(), AttestationError> {
    if attestation.accumulate_interval == 0 {
        return Err(AttestationError::InvalidAccumulateInterval);
    }
    if attestation.accumulate_interval < policy.min_accumulate_interval {
        return Err(AttestationError::AccumulateIntervalTooSmall {
            got: attestation.accumulate_interval,
            min: policy.min_accumulate_interval,
        });
    }
    if policy.require_accumulate_fp32 && attestation.accumulate_dtype != AccumulateDtype::Fp32 {
        return Err(AttestationError::AccumulateDtypeRejected);
    }
    if policy.require_allow_tf32_false && attestation.allow_tf32 {
        return Err(AttestationError::Tf32NotAllowed);
    }
    Ok(())
}
