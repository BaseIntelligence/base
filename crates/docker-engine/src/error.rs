//! Docker client errors.

use thiserror::Error;

/// Docker client errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DockerError {
    /// Method/path not on the allowlist.
    #[error("docker API call not allowlisted: {method} {path}")]
    NotAllowlisted {
        /// HTTP method.
        method: String,
        /// Request path.
        path: String,
    },
    /// Transport or HTTP status failure.
    #[error("docker API error: {0}")]
    Api(String),
    /// JSON decode failure.
    #[error("docker API JSON: {0}")]
    Json(String),
}
