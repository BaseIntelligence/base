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
