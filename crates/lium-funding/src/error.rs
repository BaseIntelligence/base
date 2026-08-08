//! Funding errors.

use thiserror::Error;

/// Funding subsystem error.
#[derive(Debug, Error)]
pub enum FundingError {
    /// Policy rejected the hotkey.
    #[error("ineligible: {0}")]
    Ineligible(String),
    /// Quote math / oracle failure.
    #[error("quote: {0}")]
    Quote(String),
    /// Payment not found / insufficient.
    #[error("payment: {0}")]
    Payment(String),
    /// Credit missing or already spent.
    #[error("credit: {0}")]
    Credit(String),
    /// Lium account / HTTP.
    #[error("lium: {0}")]
    Lium(String),
    /// Misconfiguration.
    #[error("config: {0}")]
    Config(String),
    /// Storage fault.
    #[error("store: {0}")]
    Store(String),
}
