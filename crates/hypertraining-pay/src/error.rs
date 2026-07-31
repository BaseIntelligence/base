//! Payment, vesting, and commit-reveal errors.

use thiserror::Error;

/// Failures from pay accounting and commit-reveal binding.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PayError {
    /// Vesting segment count `V` must be ≥ 1.
    #[error("invalid vesting segments V={0}: must be >= 1")]
    InvalidVestingSegments(u32),
    /// Unknown grant id on the ledger.
    #[error("unknown vesting grant id={0}")]
    UnknownGrant(u64),
    /// Grant already fully vested or clawed back; no unvested balance.
    #[error("grant id={0} has no unvested balance")]
    NothingToClawback(u64),
    /// Commit-reveal: revealed payload does not match the prior commitment.
    #[error("commit-reveal mismatch: revealed payload does not match commitment")]
    CommitRevealMismatch,
    /// Commit-reveal: empty payload is not allowed.
    #[error("commit-reveal empty payload")]
    EmptyPayload,
}
