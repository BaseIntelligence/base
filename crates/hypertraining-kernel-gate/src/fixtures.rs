//! Fixed synthetic fixtures: good kernel, degraded truncated, tf32-flagged attestation.

use crate::attestation::{AccumulateDtype, PrecisionAttestation, PrecisionFormat, ScalingRecipe};
use crate::gate::KernelTensors;
use crate::ops::{
    kernel_tensors_baseline, kernel_tensors_degraded, kernel_tensors_good, kernel_tensors_ref,
};

/// Deterministic small problem size for unit tests (no GPU).
pub const FIXTURE_ROWS: usize = 4;
/// Input / weight column count.
pub const FIXTURE_COLS: usize = 16;

/// Fixed weight matrix (`rows * cols`), mildly varied values (no index casts).
#[must_use]
pub fn fixture_weight() -> Vec<f32> {
    // 4 x 16 = 64 entries; generated offline as 0.125 + 0.01 * k
    let mut w = Vec::with_capacity(FIXTURE_ROWS * FIXTURE_COLS);
    let mut v = 0.125_f32;
    for _ in 0..(FIXTURE_ROWS * FIXTURE_COLS) {
        w.push(v);
        v += 0.01_f32;
    }
    w
}

/// Fixed input vector.
#[must_use]
pub fn fixture_input() -> Vec<f32> {
    let mut x = Vec::with_capacity(FIXTURE_COLS);
    let mut v = -0.5_f32;
    for _ in 0..FIXTURE_COLS {
        x.push(v);
        v += 0.03_f32;
    }
    x
}

/// Reference FP32 tensors for the fixture problem.
#[must_use]
pub fn fixture_reference() -> KernelTensors {
    kernel_tensors_ref(
        &fixture_weight(),
        &fixture_input(),
        FIXTURE_ROWS,
        FIXTURE_COLS,
    )
}

/// Baseline same-dtype (BF16-style) tensors.
#[must_use]
pub fn fixture_baseline() -> KernelTensors {
    kernel_tensors_baseline(
        &fixture_weight(),
        &fixture_input(),
        FIXTURE_ROWS,
        FIXTURE_COLS,
    )
}

/// Good candidate: within κ of baseline vs ref.
#[must_use]
pub fn fixture_good_kernel() -> KernelTensors {
    kernel_tensors_good(
        &fixture_weight(),
        &fixture_input(),
        FIXTURE_ROWS,
        FIXTURE_COLS,
    )
}

/// Degraded candidate: truncated reduction — must fail κ=2.
#[must_use]
pub fn fixture_degraded_kernel() -> KernelTensors {
    kernel_tensors_degraded(
        &fixture_weight(),
        &fixture_input(),
        FIXTURE_ROWS,
        FIXTURE_COLS,
    )
}

/// Valid precision attestation (`allow_tf32 = false`).
#[must_use]
pub fn fixture_attestation_ok() -> PrecisionAttestation {
    PrecisionAttestation {
        format: PrecisionFormat::Bf16,
        accumulate_dtype: AccumulateDtype::Fp32,
        accumulate_interval: 128,
        scaling_recipe: ScalingRecipe::Delayed,
        allow_tf32: false,
    }
}

/// TF32-flagged attestation: `allow_tf32 = true` (reject under default policy).
#[must_use]
pub fn fixture_attestation_tf32_flagged() -> PrecisionAttestation {
    PrecisionAttestation {
        format: PrecisionFormat::Mixed,
        accumulate_dtype: AccumulateDtype::Fp32,
        accumulate_interval: 128,
        scaling_recipe: ScalingRecipe::Current,
        allow_tf32: true,
    }
}
