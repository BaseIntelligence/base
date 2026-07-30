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
    /// One-shot run exceeded wall-clock timeout (container stopped and removed).
    #[error("docker run timed out after {timeout_sec}s")]
    Timeout {
        /// Configured timeout in whole seconds.
        timeout_sec: u64,
    },
    /// Owned-run name rejected (must use the verifier prefix).
    #[error("container name must start with `{required_prefix}`: {name}")]
    BadName {
        /// Rejected container name.
        name: String,
        /// Required name prefix.
        required_prefix: String,
    },
}
