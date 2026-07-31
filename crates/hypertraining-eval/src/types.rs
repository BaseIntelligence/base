//! Validator-owned eval run types (fixed-point loss).

/// Continuous validation loss in micro-units (`1_000_000` = loss `1.0`).
///
/// Fixed-point keeps the public eval surface free of `f64` leaf scores.
/// Internal Guard 2 statistics may widen to `f64` (see `quality` module docs).
pub type LossMicro = i64;

/// Micro-units per unit loss.
pub const MICRO_PER_UNIT: i64 = 1_000_000;

/// One paired seed's validator-measured continuous val loss.
///
/// **Never** populated from miner-reported metrics — only harness / fixture outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvalRun {
    /// Pairing key (must match between champion and candidate).
    pub seed: u64,
    /// Continuous validation loss (micro-units).
    pub val_loss_micro: LossMicro,
}
