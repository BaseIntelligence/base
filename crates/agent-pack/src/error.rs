//! Typed errors for Harbor pack load / project.

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
}
