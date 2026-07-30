//! Typed policy errors (library boundary).

use thiserror::Error;

/// Failures that abort evaluation before an [`crate::AttestOutcome`] is produced.
///
/// Most policy paths return an outcome (`Reject` / `Park`) instead of `Err`.
/// `Err` is reserved for programmer / invariant mistakes (e.g. bad key lengths
/// already typed away) — currently unused by [`crate::evaluate`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// Internal invariant broken (should be unreachable with typed inputs).
    #[error("policy invariant: {0}")]
    Invariant(&'static str),
}
