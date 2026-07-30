//! Shared Docker Engine value types.

use serde::{Deserialize, Serialize};

/// Minimal container summary from list/inspect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerSummary {
    /// Container id.
    pub id: String,
    /// First name without leading `/`.
    pub name: String,
    /// Image reference as reported by Docker.
    pub image: String,
    /// Compose project label if present.
    pub compose_project: Option<String>,
    /// Compose service label if present.
    pub compose_service: Option<String>,
}

/// Result of create → start → wait → logs → rm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult {
    /// Process exit code from `wait`.
    pub status_code: i64,
    /// Combined stdout/stderr log text.
    pub logs: String,
}

/// Owned one-shot container run (verifier / agent runner).
///
/// Names must start with [`crate::OWNED_NAME_PREFIX`]. Image should be
/// digest-pinned (`repo@sha256:…`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSpec {
    /// Container name (must start with owned prefix).
    pub name: String,
    /// Image reference, preferably `name@sha256:…`.
    pub image: String,
    /// Entrypoint command.
    pub cmd: Vec<String>,
    /// Bind mounts `host:container[:ro]`.
    pub binds: Vec<String>,
    /// Environment `KEY=VALUE` entries.
    pub env: Vec<String>,
    /// Disable networking inside the container.
    pub network_disabled: bool,
    /// Working directory inside the container.
    pub working_dir: Option<String>,
    /// Wall-clock timeout; on expiry → stop/rm + [`crate::DockerError::Timeout`].
    pub timeout_sec: Option<u64>,
}
