//! Updater configuration (proxy URL, compose project, service, health, pins).

use std::path::PathBuf;
use std::time::Duration;

/// Compose services that promote + a dedicated updater instance may roll.
///
/// Keep in lockstep with `deploy/scripts/promote.sh` `--service` allowlist and
/// keys under `deploy/pins/{staging,prod}.json` (D8 packaging).
pub const ROLLABLE_SERVICES: &[&str] = &[
    "validator",
    "gateway",
    "updater",
    "agent-challenge",
];

/// Runtime configuration for one updater target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdaterConfig {
    /// Base URL of docker-socket-proxy (e.g. `http://socket-proxy:2375`).
    pub proxy_url: String,
    /// Compose project name (used to filter containers by label).
    pub compose_project: String,
    /// Compose service name to roll (label `com.docker.compose.service`).
    pub service_name: String,
    /// Full health URL including path (e.g. `http://validator:8080/readyz`).
    pub health_url: String,
    /// Directory holding `current.json` / `previous.json`.
    pub state_dir: PathBuf,
    /// Desired image **must** already be digest-pinned (`repo@sha256:…`).
    pub desired_image: String,
    /// Container name of this updater process — never recreated automatically.
    pub self_container_name: String,
    /// Max wait while polling `/readyz` after recreate.
    pub health_timeout: Duration,
    /// Interval between `/readyz` polls.
    pub health_poll_interval: Duration,
}

/// True when `name` is a known promote/updater roll target.
#[must_use]
pub fn is_rollable_service(name: &str) -> bool {
    ROLLABLE_SERVICES.iter().any(|s| *s == name)
}

impl UpdaterConfig {
    /// Build config with sensible defaults for timeouts.
    #[must_use]
    pub fn new(
        proxy_url: impl Into<String>,
        compose_project: impl Into<String>,
        service_name: impl Into<String>,
        health_url: impl Into<String>,
        state_dir: PathBuf,
        desired_image: impl Into<String>,
        self_container_name: impl Into<String>,
    ) -> Self {
        Self {
            proxy_url: proxy_url.into(),
            compose_project: compose_project.into(),
            service_name: service_name.into(),
            health_url: health_url.into(),
            state_dir,
            desired_image: desired_image.into(),
            self_container_name: self_container_name.into(),
            health_timeout: Duration::from_mins(1),
            health_poll_interval: Duration::from_millis(100),
        }
    }

    /// Load from environment variables used by the binary.
    ///
    /// | Variable | Required | Meaning |
    /// |----------|----------|---------|
    /// | `GBASE_UPDATER_PROXY_URL` | yes | socket-proxy base URL |
    /// | `GBASE_UPDATER_COMPOSE_PROJECT` | yes | compose project |
    /// | `GBASE_UPDATER_SERVICE_NAME` | yes | target service |
    /// | `GBASE_UPDATER_HEALTH_URL` | yes | full `/readyz` URL |
    /// | `GBASE_UPDATER_STATE_DIR` | yes | pin directory |
    /// | `GBASE_UPDATER_DESIRED_IMAGE` | yes | `image@sha256:…` |
    /// | `GBASE_UPDATER_SELF_NAME` | no | defaults to `HOSTNAME` or `updater` |
    ///
    /// # Errors
    /// Returns a static message when a required variable is missing.
    pub fn from_env() -> Result<Self, &'static str> {
        fn req(key: &str) -> Result<String, &'static str> {
            std::env::var(key).map_err(|_| "missing required updater env var")
        }
        let self_name = std::env::var("GBASE_UPDATER_SELF_NAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "updater".to_owned());
        Ok(Self::new(
            req("GBASE_UPDATER_PROXY_URL")?,
            req("GBASE_UPDATER_COMPOSE_PROJECT")?,
            req("GBASE_UPDATER_SERVICE_NAME")?,
            req("GBASE_UPDATER_HEALTH_URL")?,
            PathBuf::from(req("GBASE_UPDATER_STATE_DIR")?),
            req("GBASE_UPDATER_DESIRED_IMAGE")?,
            self_name,
        ))
    }
}


#[cfg(test)]
mod tests {
    use super::{is_rollable_service, ROLLABLE_SERVICES};

    #[test]
    fn rollable_services_include_agent_challenge() {
        assert_eq!(
            ROLLABLE_SERVICES,
            &["validator", "gateway", "updater", "agent-challenge"]
        );
        assert!(is_rollable_service("agent-challenge"));
    }
}
