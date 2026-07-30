//! Inspect rendered compose for contract checks (no secrets in env, extract YAML).

use serde_json::Value;

use crate::template::AGENT_SERVICE;

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
            if !trimmed.is_empty()
                && !line.starts_with(' ')
                && !line.starts_with('\t')
            {
                in_env = false;
                continue;
            }
            if trimmed.starts_with("volumes:")
                || trimmed.starts_with("ports:")
                || trimmed.starts_with("image:")
                || trimmed.starts_with("restart:")
                || trimmed.starts_with("command:")
                || (trimmed.ends_with(':')
                    && !trimmed.contains(' ')
                    && !trimmed.starts_with("GBASE_"))
            {
                // next service key
                if !trimmed.starts_with("GBASE_") {
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
            if key_upper.contains("SECRET")
                || key_upper.contains("PRIVATE")
                || key_upper.ends_with("_SK")
                || key_upper.ends_with("_KEY") && !key_upper.ends_with("_FILE")
                || key_upper.contains("PASSWORD")
                || key_upper.contains("TOKEN") && !key_upper.ends_with("_HASH")
            {
                return false;
            }
            // Reject raw 64-byte hex that is NOT the known launch-token hash field
            if key != "GBASE_LAUNCH_TOKEN_HASH"
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
