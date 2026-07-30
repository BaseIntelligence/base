//! Fail-closed error types for trust-root load and verify.

use std::path::PathBuf;

use thiserror::Error;

/// Trust-root load / verify / policy failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TrustRootError {
    /// Required path is missing (fail closed).
    #[error("trust root missing: {path}")]
    MissingFile {
        /// Path that was required.
        path: PathBuf,
    },
    /// I/O failure reading a local path.
    #[error("trust root I/O {path}: {message}")]
    Io {
        /// Path involved.
        path: PathBuf,
        /// OS / read error text.
        message: String,
    },
    /// TOML body could not be parsed.
    #[error("trust root TOML invalid at {path}: {message}")]
    InvalidToml {
        /// Path of the TOML file.
        path: PathBuf,
        /// Parser message.
        message: String,
    },
    /// Detached signature file missing or empty.
    #[error("trust root unsigned (missing or empty signature): {path}")]
    Unsigned {
        /// Path of the expected signature file.
        path: PathBuf,
    },
    /// Signature bytes malformed or verification failed under owner key.
    #[error("trust root signature rejected: {reason}")]
    SignatureRejected {
        /// Human-readable reason.
        reason: String,
    },
    /// Signature verifies under a key that is not the configured owner.
    #[error("trust root signed by non-owner key")]
    NonOwner,
    /// Hex / key material encoding error.
    #[error("invalid encoding: {0}")]
    InvalidEncoding(String),
    /// Semantic validation failed (bps sum, duplicate ids, …).
    #[error("trust root invalid: {0}")]
    InvalidBody(String),
    /// No trust-root version is active at the requested epoch.
    #[error("no active trust root at epoch {epoch}")]
    NoActiveRoot {
        /// Epoch that was queried.
        epoch: u64,
    },
    /// Crypto helper failure.
    #[error(transparent)]
    Crypto(#[from] crypto::CryptoError),
}
