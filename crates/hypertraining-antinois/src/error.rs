//! Anti-noise gate errors.

use thiserror::Error;

/// Failures from similarity / dedupe evaluation (not sanctions).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AntinoisError {
    /// Empty source or compiled blob where content is required.
    #[error("empty artifact: {0}")]
    EmptyArtifact(&'static str),
    /// Miner id must be non-empty.
    #[error("miner id must be non-empty")]
    EmptyMinerId,
    /// Segment window N must be > 0 for dedupe.
    #[error("dedupe window N must be > 0")]
    InvalidDedupeWindow,
}
