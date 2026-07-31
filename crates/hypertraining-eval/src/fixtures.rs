//! Controllable fixtures for Guard 2 / Guard 3 TDD (no real GPU).

use hypertraining_cluster::MmaFamily;

use crate::physics::PhysicsTelemetry;
use crate::types::{EvalRun, LossMicro};

/// K=5 paired runs with identical continuous val loss (promote-quality path).
#[must_use]
pub fn fixture_equal_quality_pairs() -> (Vec<EvalRun>, Vec<EvalRun>) {
    let loss: LossMicro = 1_420_000; // 1.42
    let champ: Vec<_> = (0..5_u64)
        .map(|i| EvalRun {
            seed: 100 + i,
            val_loss_micro: loss,
        })
        .collect();
    let cand = champ.clone();
    (champ, cand)
}

/// K=5: candidate clearly worse on continuous val loss (Guard 2 reject).
///
/// Champion ~1.40; candidate ~1.55 on every seed → large negative `d_i`.
#[must_use]
pub fn fixture_worse_candidate_pairs() -> (Vec<EvalRun>, Vec<EvalRun>) {
    let champ: Vec<_> = (0..5_i64)
        .map(|i| EvalRun {
            seed: 200 + i as u64,
            val_loss_micro: 1_400_000 + i * 100,
        })
        .collect();
    let cand: Vec<_> = (0..5_i64)
        .map(|i| EvalRun {
            seed: 200 + i as u64,
            // ~10% worse absolute loss — far beyond ε band
            val_loss_micro: 1_550_000 + i * 100,
        })
        .collect();
    (champ, cand)
}

/// Plausible physics: BF16 MMA, DRAM/tensor near Θ, modest wallclock.
#[must_use]
pub fn fixture_plausible_physics() -> PhysicsTelemetry {
    PhysicsTelemetry {
        dram_bytes: 64_000_000_000, // 64 GB moved
        tensor_ops: 128_000_000_000,
        mma_family: MmaFamily::Bf16,
        wallclock_ms: 50_000,
        peak_dram_bandwidth_bytes_per_s: 2_000_000_000_000, // 2 TB/s
    }
}

/// Implausible speedup: tiny wallclock with full DRAM (roofline fail).
#[must_use]
pub fn fixture_implausible_physics() -> PhysicsTelemetry {
    PhysicsTelemetry {
        dram_bytes: 64_000_000_000,
        tensor_ops: 128_000_000_000,
        mma_family: MmaFamily::Bf16,
        wallclock_ms: 100, // 1000× faster than 100_000 ms reference in test
        peak_dram_bandwidth_bytes_per_s: 2_000_000_000_000,
    }
}
