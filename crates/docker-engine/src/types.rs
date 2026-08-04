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

/// Owned one-shot container run (verifier / design sandbox).
///
/// Names must start with an owned prefix (`base-verify-` or `base-design-`).
/// Image should be digest-pinned (`repo@sha256:…`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // HostConfig hardening knobs are intentionally independent.
pub struct RunSpec {
    /// Container name (must start with an owned prefix).
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
    /// Mount root filesystem read-only.
    pub readonly_rootfs: bool,
    /// Drop all Linux capabilities.
    pub cap_drop_all: bool,
    /// Set `no-new-privileges:true` security opt.
    pub no_new_privileges: bool,
    /// PID limit (`HostConfig.PidsLimit`).
    pub pids_limit: Option<i64>,
    /// Memory limit bytes (`HostConfig.Memory`).
    pub memory_bytes: Option<i64>,
    /// Memory+swap limit bytes (`HostConfig.MemorySwap`); `-1` = unlimited swap.
    pub memory_swap_bytes: Option<i64>,
    /// CPU quota in nano-CPUs (`HostConfig.NanoCpus`).
    pub nano_cpus: Option<i64>,
    /// Attach to an existing network (`HostConfig.NetworkMode`).
    pub network_mode: Option<String>,
    /// Run as `uid:gid` (e.g. `65532:65532`).
    pub user: Option<String>,
    /// Tmpfs mounts: `(dest, options)` e.g. `("/tmp", "rw,noexec,nosuid,size=64m")`.
    pub tmpfs: Vec<(String, String)>,
}

impl RunSpec {
    /// Hardened defaults for design-sandbox runs (caller still sets name/image/cmd/binds).
    #[must_use]
    pub fn design_hardened(name: String, image: String, cmd: Vec<String>) -> Self {
        Self {
            name,
            image,
            cmd,
            binds: vec![],
            env: vec![],
            network_disabled: false,
            working_dir: None,
            timeout_sec: Some(600),
            readonly_rootfs: true,
            cap_drop_all: true,
            no_new_privileges: true,
            pids_limit: Some(256),
            memory_bytes: Some(512 * 1024 * 1024),
            memory_swap_bytes: Some(512 * 1024 * 1024),
            nano_cpus: Some(1_000_000_000),
            network_mode: Some("design-sandbox-egress".into()),
            user: Some("65532:65532".into()),
            tmpfs: vec![("/tmp".into(), "rw,noexec,nosuid,size=64m".into())],
        }
    }
}
