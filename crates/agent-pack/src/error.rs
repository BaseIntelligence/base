//! Typed errors for Harbor pack load / project / catalog.

use std::path::PathBuf;

use thiserror::Error;

/// Failures while resolving, parsing, or projecting a pack.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PackError {
    /// Pack id is unknown or not registered.
    #[error("pack not found: {0}")]
    NotFound(String),
    /// Pack bytes / layout failed validation.
    #[error("pack invalid: {0}")]
    Invalid(String),
    /// A required `task.toml` (or layout) field is absent or empty.
    #[error("missing required field `{field}`")]
    MissingField {
        /// Dotted or bare field name (e.g. `base_commit_hash`).
        field: &'static str,
    },
    /// Unsupported `schema_version` in `task.toml`.
    #[error("unsupported schema_version `{found}`; expected `{expected}`")]
    UnsupportedSchema {
        /// Value present in the file.
        found: String,
        /// Version this crate accepts.
        expected: &'static str,
    },
    /// Filesystem I/O while reading the pack directory.
    #[error("pack I/O at {path}: {message}")]
    Io {
        /// Path that failed.
        path: PathBuf,
        /// OS / context message.
        message: String,
    },
    /// `task.toml` is not valid TOML or failed typed decode.
    #[error("task.toml parse error: {0}")]
    Toml(String),
    /// Pack selection received an empty catalog.
    #[error("empty pack catalog")]
    EmptyCatalog,
}

/// Failures while materializing or loading a pinned pack catalog.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CatalogError {
    /// No packs found under the source or in the manifest.
    #[error("empty pack catalog")]
    Empty,
    /// Cached pack content does not match the manifest (tamper / corruption).
    #[error("catalog integrity failure for pack `{pack_id}`: {message}")]
    Integrity {
        /// Pack that failed verification.
        pack_id: String,
        /// Human-readable mismatch detail.
        message: String,
    },
    /// Recomputed catalog digest does not match the manifest field.
    #[error("catalog_digest mismatch: expected {expected}, found {found}")]
    CatalogDigestMismatch {
        /// Digest recomputed from pin + entries.
        expected: String,
        /// Digest stored in the manifest.
        found: String,
    },
    /// Manifest file is missing under the cache root.
    #[error("catalog manifest missing at {path}")]
    ManifestMissing {
        /// Expected path of `manifest.json`.
        path: PathBuf,
    },
    /// Manifest JSON is invalid.
    #[error("catalog manifest invalid: {0}")]
    ManifestInvalid(String),
    /// Pin is empty or a floating reference such as `latest`.
    #[error("floating or empty catalog pin refused: {0:?}")]
    FloatingPin(String),
    /// Filesystem I/O while materializing or loading the catalog.
    #[error("catalog I/O at {path}: {message}")]
    Io {
        /// Path that failed.
        path: PathBuf,
        /// OS / context message.
        message: String,
    },
    /// Manifest JSON serialization failed.
    #[error("catalog manifest serialize error: {0}")]
    Serialize(String),
    /// Underlying pack load / parse failure.
    #[error(transparent)]
    Pack(#[from] PackError),
}
