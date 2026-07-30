//! Updater error types.

use thiserror::Error;

use crate::docker::DockerError;
use crate::health::HealthError;

/// Top-level updater failures.
#[derive(Debug, Error)]
pub enum UpdaterError {
    /// Image is not digest-pinned.
    #[error("image is not digest-pinned (require repo@sha256:<64-hex>): {image}")]
    NotDigestPinned {
        /// Offending image string.
        image: String,
    },
    /// Target container name equals the updater's own name (D14).
    #[error("refusing to update self container {name} (D14; operator-run only)")]
    RefuseSelfUpdate {
        /// Container name that matched self.
        name: String,
    },
    /// Docker API / allowlist failure.
    #[error(transparent)]
    Docker(#[from] DockerError),
    /// Health gate failure.
    #[error(transparent)]
    Health(#[from] HealthError),
    /// Pin store I/O or JSON.
    #[error("pin store: {0}")]
    PinStore(String),
    /// Target container not found for project/service.
    #[error("no running container for project={project} service={service}")]
    TargetNotFound {
        /// Compose project.
        project: String,
        /// Compose service.
        service: String,
    },
    /// Rollout exhausted retries (surface for callers).
    #[error("rollout exhausted: {reason}")]
    Exhausted {
        /// Last error reason.
        reason: String,
    },
}
