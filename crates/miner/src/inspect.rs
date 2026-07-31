//! Inspect rendered compose for contract checks (no secrets in env, extract YAML).

use serde_json::Value;
use thiserror::Error;

use crate::template::{AGENT_SERVICE, SOCKET_PROXY_SERVICE};

/// Extract the embedded docker-compose YAML from rendered app-compose JSON text.
#[must_use]
pub fn docker_compose_from_app_compose_json(app_compose_json: &str) -> Option<String> {
    let v: Value = serde_json::from_str(app_compose_json).ok()?;
    v.get("docker_compose_file")?
        .as_str()
        .map(std::string::ToString::to_string)
}

/// True when the YAML `environment:` blocks contain no obvious secret material.
///
/// Allows path-like values and the launch-token **hash** (public/measured).
/// Rejects long hex blobs that look like private keys and common secret keys.
#[must_use]
pub fn environment_block_has_no_secrets(docker_compose_yaml: &str) -> bool {
    let mut in_env = false;
    for line in docker_compose_yaml.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("environment:") {
            in_env = true;
            continue;
        }
        if in_env {
            // left environment block when indentation returns to service key level
            // or a new top-level key under the service (volumes, ports, image, …)
            if !trimmed.is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
                in_env = false;
                continue;
            }
            if trimmed.starts_with("volumes:")
                || trimmed.starts_with("ports:")
                || trimmed.starts_with("image:")
                || trimmed.starts_with("restart:")
                || trimmed.starts_with("command:")
                || trimmed.starts_with("depends_on:")
                || (trimmed.ends_with(':')
                    && !trimmed.contains(' ')
                    && !trimmed.starts_with("BASE_"))
            {
                // next service key
                if !trimmed.starts_with("BASE_") {
                    in_env = false;
                    continue;
                }
            }
            if !in_env {
                continue;
            }
            // Parse `KEY: "value"` or `KEY: value`
            let Some((key, value)) = trimmed.split_once(':') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim().trim_matches('"');
            if key.is_empty() {
                continue;
            }
            let key_upper = key.to_ascii_uppercase();
            let measured_public_key_name = matches!(
                key,
                "BASE_RECEIPT_PUBLIC_KEY" | "BASE_TRUSTED_CHALLENGE_PUBKEY"
            );
            if key_upper.contains("SECRET")
                || key_upper.contains("PRIVATE")
                || key_upper.ends_with("_SK")
                || (key_upper.ends_with("_KEY")
                    && !key_upper.ends_with("_FILE")
                    && !measured_public_key_name)
                || key_upper.contains("PASSWORD")
                || key_upper.contains("TOKEN") && !key_upper.ends_with("_HASH")
            {
                return false;
            }
            // Reject raw 64-byte hex that is NOT a known measured public field
            let measured_public_hex = matches!(
                key,
                "BASE_LAUNCH_TOKEN_HASH"
                    | "BASE_RECEIPT_PUBLIC_KEY"
                    | "BASE_TRUSTED_CHALLENGE_PUBKEY"
            );
            if !measured_public_hex
                && value.len() == 64
                && value.chars().all(|c| c.is_ascii_hexdigit())
                && !value.starts_with('/')
            {
                return false;
            }
            // Reject PEM-looking blobs
            if value.contains("BEGIN") && value.contains("PRIVATE") {
                return false;
            }
        }
    }
    // Must mention agent service (sanity)
    docker_compose_yaml.contains(AGENT_SERVICE)
}

/// Agent service mounts raw `/var/run/docker.sock` (forbidden; use measured socket-proxy).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("agent service must not mount /var/run/docker.sock (use measured {SOCKET_PROXY_SERVICE})")]
pub struct RawDockerSockOnAgent;

/// True when the `agent` service block mounts `/var/run/docker.sock`.
#[must_use]
pub fn agent_service_mounts_docker_sock(docker_compose_yaml: &str) -> bool {
    let mut in_agent = false;
    for line in docker_compose_yaml.lines() {
        // Service keys are indented exactly two spaces: `  name:`
        if let Some(rest) = line.strip_prefix("  ") {
            if !rest.starts_with(' ') && !rest.starts_with('\t') {
                let trimmed = rest.trim();
                if trimmed.ends_with(':') && !trimmed[..trimmed.len() - 1].contains(' ') {
                    let name = &trimmed[..trimmed.len() - 1];
                    in_agent = name == AGENT_SERVICE;
                    continue;
                }
            }
        }
        if in_agent && line.contains("/var/run/docker.sock") {
            return true;
        }
    }
    false
}

/// Reject compose YAML where the long-lived `agent` mounts the raw Docker socket.
///
/// # Errors
/// Returns [`RawDockerSockOnAgent`] when the agent service mounts `docker.sock`.
pub fn reject_raw_docker_sock_on_agent(
    docker_compose_yaml: &str,
) -> Result<(), RawDockerSockOnAgent> {
    if agent_service_mounts_docker_sock(docker_compose_yaml) {
        Err(RawDockerSockOnAgent)
    } else {
        Ok(())
    }
}
