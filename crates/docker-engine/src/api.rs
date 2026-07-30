//! `DockerApi` trait — operations used by updater and agent runners.

use std::collections::HashMap;

use crate::error::DockerError;
use crate::types::ContainerSummary;

/// Abstraction over Docker Engine operations used by updater and runners.
///
/// # Errors
/// Implementations return [`DockerError`] on allowlist denial, HTTP/API failure, or bad JSON.
#[allow(clippy::missing_errors_doc)]
pub trait DockerApi: Send + Sync {
    /// List containers (`all=true`).
    fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError>;
    /// Inspect one container by id or name.
    fn inspect_container(&self, id_or_name: &str) -> Result<ContainerSummary, DockerError>;
    /// Pull image by reference (`repo@sha256:…`).
    fn pull_image(&self, image: &str) -> Result<(), DockerError>;
    /// Stop container.
    fn stop_container(&self, id_or_name: &str) -> Result<(), DockerError>;
    /// Create container from image with name and labels.
    fn create_container(
        &self,
        name: &str,
        image: &str,
        labels: &HashMap<String, String>,
    ) -> Result<String, DockerError>;
    /// Start container by id or name.
    fn start_container(&self, id_or_name: &str) -> Result<(), DockerError>;
    /// Rename container.
    fn rename_container(&self, id_or_name: &str, new_name: &str) -> Result<(), DockerError>;
    /// Block until exit; return status code.
    fn wait_container(&self, id_or_name: &str) -> Result<i64, DockerError>;
    /// Fetch logs (stdout+stderr).
    fn container_logs(&self, id_or_name: &str) -> Result<String, DockerError>;
    /// Force-remove a container.
    fn remove_container(&self, id_or_name: &str) -> Result<(), DockerError>;
}
