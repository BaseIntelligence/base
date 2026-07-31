//! Admission errors.

use thiserror::Error;

/// Why a fork failed sealed-surface admission.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AdmitError {
    /// A changed path matches the denylist.
    #[error("denylist path touched: {path}")]
    DenylistPathTouched {
        /// Path that matched the denylist.
        path: String,
    },
    /// A changed path is outside the allowlist (and not denylisted).
    #[error("path not on allowlist: {path}")]
    PathNotAllowlisted {
        /// Path that is neither allowlisted nor denylisted.
        path: String,
    },
    /// Content hash of a denylisted path does not match the manifest pin.
    #[error("denylist hash mismatch: {path}")]
    DenylistHashMismatch {
        /// Denylist path whose content hash diverged.
        path: String,
    },
    /// Required file content missing from the admission input map.
    #[error("missing file content for hash check: {path}")]
    MissingFileContent {
        /// Path that was required but absent.
        path: String,
    },
    /// Sealed-symbol AST fingerprint does not match the manifest.
    #[error("sealed symbol AST hash mismatch: {key}")]
    SealedSymbolMismatch {
        /// Manifest key `path:symbol`.
        key: String,
    },
    /// Sealed symbol could not be located in source.
    #[error("sealed symbol not found: {key}")]
    SealedSymbolNotFound {
        /// Manifest key `path:symbol`.
        key: String,
    },
    /// Manifest kind/version is not `sealed_surface.v1`.
    #[error("unsupported manifest kind: {kind}")]
    UnsupportedManifestKind {
        /// Kind string from the payload.
        kind: String,
    },
    /// Manifest pin does not match the frozen challenge defaults.
    #[error("manifest pin mismatch: {field} expected {expected}, got {got}")]
    PinMismatch {
        /// Field name (`mlm_commit` or `te_version`).
        field: &'static str,
        /// Expected frozen value.
        expected: String,
        /// Value present in the manifest.
        got: String,
    },
    /// Malformed sealed-symbol key (expected `path:symbol`).
    #[error("invalid sealed symbol key: {key}")]
    InvalidSymbolKey {
        /// Bad key string.
        key: String,
    },
    /// Hex digest in the manifest is not 64 lowercase hex chars.
    #[error("invalid hex digest for {context}: {detail}")]
    InvalidDigest {
        /// What the digest was for.
        context: String,
        /// Parse detail.
        detail: String,
    },
}
