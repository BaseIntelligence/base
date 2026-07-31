//! Eval errors (input contract failures, not guard rejects).

use thiserror::Error;

/// Failures that prevent forming a verdict (bad inputs).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EvalError {
    /// Champion and candidate run vectors differ in length.
    #[error("paired run length mismatch: champ={champ} cand={cand}")]
    PairedLengthMismatch { champ: usize, cand: usize },

    /// Fewer than two paired observations (variance undefined for t-test).
    #[error("need at least 2 paired runs for Guard 2, got {got}")]
    InsufficientPairs { got: usize },

    /// Seed ids do not align pairwise at the same index.
    #[error("paired seed mismatch at index {index}: champ={champ_seed} cand={cand_seed}")]
    SeedMismatch {
        index: usize,
        champ_seed: u64,
        cand_seed: u64,
    },

    /// Analytic model has zero expected counters (misconfigured fixture).
    #[error("analytic model expected_{field} must be > 0")]
    InvalidAnalyticModel { field: &'static str },
}
