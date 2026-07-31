//! Guard 3: physical plausibility from sim / Nsight telemetry vs analytic model.

use hypertraining_cluster::MmaFamily;

use crate::error::EvalError;
use crate::verdict::RejectReason;

/// Minimum acceptable DRAM bytes as basis points of analytic expectation (50% = 5000).
pub const DRAM_MIN_RATIO_BPS: u32 = 5_000;

/// Minimum acceptable tensor-ops as basis points of analytic expectation.
pub const TENSOR_MIN_RATIO_BPS: u32 = 5_000;

/// Validator-side physics counters for one candidate segment (sim fixture or Nsight).
///
/// Built from [`hypertraining_cluster::SegmentTelemetry`] or test fixtures — never
/// from miner self-reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicsTelemetry {
    /// DRAM bytes moved during the measured segment.
    pub dram_bytes: u64,
    /// Tensor-core operation count.
    pub tensor_ops: u64,
    /// Dominant MMA family observed.
    pub mma_family: MmaFamily,
    /// Candidate wall-clock (ms) for roofline speedup check.
    pub wallclock_ms: u64,
    /// Peak DRAM bandwidth (bytes/s) used for optional BW-bound cross-check; 0 = skip.
    pub peak_dram_bandwidth_bytes_per_s: u64,
}

impl PhysicsTelemetry {
    /// Lift cluster segment telemetry + wallclock into eval physics input.
    #[must_use]
    pub fn from_segment(
        tel: &hypertraining_cluster::SegmentTelemetry,
        wallclock_ms: u64,
    ) -> Self {
        Self {
            dram_bytes: tel.dram_bytes,
            tensor_ops: tel.tensor_ops,
            mma_family: tel.mma_family,
            wallclock_ms,
            peak_dram_bandwidth_bytes_per_s: tel.peak_dram_bandwidth_bytes_per_s,
        }
    }
}

/// Analytic expectations for Guard 3 (fixture or derived model thresholds).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticModel {
    /// Expected DRAM bytes for the segment workload.
    pub expected_dram_bytes: u64,
    /// Expected tensor-core ops for the segment workload.
    pub expected_tensor_ops: u64,
    /// Required MMA family (harness precision contract).
    pub required_mma: MmaFamily,
    /// Reference wallclock (ms) — typically champion / baseline on same seeds.
    pub reference_wallclock_ms: u64,
    /// Max plausible speedup in milli-units (`1000` = 1.0×, `2000` = 2.0×).
    pub max_plausible_speedup_milli: u64,
}

impl AnalyticModel {
    /// Build a model that treats `tel` as the analytic baseline (exact match OK)
    /// with a given max speedup vs `tel.wallclock_ms` as reference.
    #[must_use]
    pub fn from_telemetry_baseline(tel: &PhysicsTelemetry, max_plausible_speedup_milli: u64) -> Self {
        Self {
            expected_dram_bytes: tel.dram_bytes,
            expected_tensor_ops: tel.tensor_ops,
            required_mma: tel.mma_family,
            reference_wallclock_ms: tel.wallclock_ms.max(1),
            max_plausible_speedup_milli,
        }
    }
}

/// Run Guard 3 checks; returns reject reasons (empty = `physics_ok`).
///
/// # Errors
///
/// Returns [`EvalError::InvalidAnalyticModel`] when expected counters or bounds are zero.
pub fn check_physics(
    tel: &PhysicsTelemetry,
    model: &AnalyticModel,
) -> Result<Vec<RejectReason>, EvalError> {
    if model.expected_dram_bytes == 0 {
        return Err(EvalError::InvalidAnalyticModel {
            field: "dram_bytes",
        });
    }
    if model.expected_tensor_ops == 0 {
        return Err(EvalError::InvalidAnalyticModel {
            field: "tensor_ops",
        });
    }
    if model.reference_wallclock_ms == 0 {
        return Err(EvalError::InvalidAnalyticModel {
            field: "reference_wallclock_ms",
        });
    }
    if model.max_plausible_speedup_milli == 0 {
        return Err(EvalError::InvalidAnalyticModel {
            field: "max_plausible_speedup_milli",
        });
    }

    let mut reasons = Vec::new();

    let dram_floor = model
        .expected_dram_bytes
        .saturating_mul(u64::from(DRAM_MIN_RATIO_BPS))
        / 10_000;
    if tel.dram_bytes < dram_floor {
        reasons.push(RejectReason::DramBytesImplausible {
            observed: tel.dram_bytes,
            expected: model.expected_dram_bytes,
            min_accepted: dram_floor,
        });
    }

    let tops_floor = model
        .expected_tensor_ops
        .saturating_mul(u64::from(TENSOR_MIN_RATIO_BPS))
        / 10_000;
    if tel.tensor_ops < tops_floor {
        reasons.push(RejectReason::TensorOpsImplausible {
            observed: tel.tensor_ops,
            expected: model.expected_tensor_ops,
            min_accepted: tops_floor,
        });
    }

    if tel.mma_family != model.required_mma {
        reasons.push(RejectReason::MmaFamilyMismatch {
            observed: tel.mma_family,
            required: model.required_mma,
        });
    }

    let cand_ms = tel.wallclock_ms.max(1);
    let speedup_milli = model
        .reference_wallclock_ms
        .saturating_mul(1000)
        / cand_ms;
    if speedup_milli > model.max_plausible_speedup_milli {
        reasons.push(RejectReason::RooflineImplausible {
            speedup_milli,
            max_plausible_speedup_milli: model.max_plausible_speedup_milli,
            reference_wallclock_ms: model.reference_wallclock_ms,
            candidate_wallclock_ms: tel.wallclock_ms,
        });
    }

    if tel.peak_dram_bandwidth_bytes_per_s > 0 && tel.wallclock_ms > 0 {
        let elapsed_s_milli = tel.wallclock_ms;
        let observed_bps = tel
            .dram_bytes
            .saturating_mul(1000)
            / elapsed_s_milli.max(1);
        let cap = tel.peak_dram_bandwidth_bytes_per_s.saturating_mul(2);
        if observed_bps > cap {
            reasons.push(RejectReason::RooflineImplausible {
                speedup_milli: observed_bps,
                max_plausible_speedup_milli: cap,
                reference_wallclock_ms: model.reference_wallclock_ms,
                candidate_wallclock_ms: tel.wallclock_ms,
            });
        }
    }

    Ok(reasons)
}
