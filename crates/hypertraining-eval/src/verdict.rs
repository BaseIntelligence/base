//! Combined eval verdict API.

use hypertraining_cluster::MmaFamily;

use crate::epsilon::EpsilonParams;
use crate::error::EvalError;
use crate::physics::{check_physics, AnalyticModel, PhysicsTelemetry};
use crate::quality::quality_non_inferiority;
use crate::types::EvalRun;

/// Why a guard rejected (empty reasons ⇒ both guards passed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// Guard 2: failed to reject `H0` (candidate significantly worse).
    QualityInferior {
        mean_d_micro: i64,
        epsilon_micro: i64,
        /// t statistic × 1000 for fixed-point logging (not a leaf score).
        t_stat_milli: i64,
        t_critical_milli: i64,
    },
    /// Guard 3: DRAM bytes far below analytic expectation (skipped work).
    DramBytesImplausible {
        observed: u64,
        expected: u64,
        min_accepted: u64,
    },
    /// Guard 3: tensor-core ops far below analytic expectation.
    TensorOpsImplausible {
        observed: u64,
        expected: u64,
        min_accepted: u64,
    },
    /// Guard 3: MMA family does not match harness contract.
    MmaFamilyMismatch {
        observed: MmaFamily,
        required: MmaFamily,
    },
    /// Guard 3: speedup exceeds physical / configured roofline bound.
    RooflineImplausible {
        speedup_milli: u64,
        max_plausible_speedup_milli: u64,
        reference_wallclock_ms: u64,
        candidate_wallclock_ms: u64,
    },
}

/// Result of [`evaluate_candidate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalVerdict {
    /// Guard 2 passed (`H0` rejected at `α=0.05`).
    pub quality_ok: bool,
    /// Guard 3 passed (telemetry vs analytic model).
    pub physics_ok: bool,
    /// Human/machine-readable reject causes (empty if both ok).
    pub reasons: Vec<RejectReason>,
}

impl EvalVerdict {
    /// Both guards green — eligible for promotion machine (subject to BH / holdout).
    #[must_use]
    pub const fn promote_allowed(&self) -> bool {
        self.quality_ok && self.physics_ok
    }
}

/// Evaluate candidate quality + physics against champion paired runs.
///
/// - `champ_runs` / `cand_runs`: validator continuous val loss, paired by `seed`
/// - `telemetry`: candidate sim/Nsight counters (not miner-reported)
/// - `model`: analytic expectations for Guard 3
/// - `eps`: `ε` parameters (`MUST_CALIBRATE` defaults via [`EpsilonParams::must_calibrate_defaults`])
///
/// # Errors
///
/// Returns [`EvalError`] when paired runs are malformed or the analytic model is invalid.
pub fn evaluate_candidate(
    champ_runs: &[EvalRun],
    cand_runs: &[EvalRun],
    telemetry: &PhysicsTelemetry,
    model: &AnalyticModel,
    eps: &EpsilonParams,
) -> Result<EvalVerdict, EvalError> {
    let q = quality_non_inferiority(champ_runs, cand_runs, eps)?;
    let mut reasons = Vec::new();
    if !q.quality_ok {
        reasons.push(RejectReason::QualityInferior {
            mean_d_micro: q.mean_d_micro,
            epsilon_micro: q.epsilon_micro,
            t_stat_milli: (q.t_stat * 1000.0).round() as i64,
            t_critical_milli: (q.t_critical * 1000.0).round() as i64,
        });
    }
    let phys_reasons = check_physics(telemetry, model)?;
    let physics_ok = phys_reasons.is_empty();
    reasons.extend(phys_reasons);

    Ok(EvalVerdict {
        quality_ok: q.quality_ok,
        physics_ok,
        reasons,
    })
}
