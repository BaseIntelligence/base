//! Digest-only image pin for Verda job deployments (miners cannot override).

use prism_lium_harness::resolved_pod_image;
use prism_lium_types::LiumError;
use sha2::{Digest, Sha256};

/// Embedded operator job server (hash-pinned in create payload tests).
pub const JOB_SERVER_PY: &str = include_str!("job_server.py");

/// Health path the replica must expose.
pub const HEALTH_PATH: &str = "/health";
/// Container listen port (Verda health + job POST).
pub const EXPOSED_PORT: u16 = 8000;

/// Resolve the operator image. Tags and miner-supplied refs are rejected.
///
/// # Errors
/// Non-digest `PRISM_POD_IMAGE_REF` / `PRISM_VERDA_IMAGE_REF`.
pub fn pinned_image() -> Result<String, LiumError> {
    if let Some(image) = std::env::var("PRISM_VERDA_IMAGE_REF")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        return require_digest(&image);
    }
    resolved_pod_image().map(|(image, _, _)| image)
}

/// Fail closed unless `repository@sha256:<64 lowercase hex>`.
///
/// # Errors
/// Tag or empty digest.
pub fn require_digest(image: &str) -> Result<String, LiumError> {
    let image = image.trim();
    let Some((repo, digest)) = image.rsplit_once("@sha256:") else {
        return Err(LiumError::Integrity(
            "Verda image must be repository@sha256:<64 lowercase hex>".into(),
        ));
    };
    if repo.is_empty()
        || digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(LiumError::Integrity(
            "Verda image digest must be 64 lowercase hex".into(),
        ));
    }
    Ok(image.to_owned())
}

/// SHA-256 hex of the operator job server (integrity pin).
#[must_use]
pub fn job_server_sha256() -> String {
    hex_sha256(JOB_SERVER_PY.as_bytes())
}

/// Hex SHA-256.
#[must_use]
pub fn hex_sha256(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    format!("{d:x}")
}

/// Verda rejects each `cmd` argument longer than 253 characters.
pub const CMD_ARG_MAX: usize = 253;
/// Hex payload chunk size for `PRISM_JS_C*` env vars.
pub const JOB_SERVER_ENV_CHUNK: usize = 200;
const HEX: &[char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// Operator `command` list: short bootstrap that `exec`s the env-chunked server.
///
/// The full `job_server.py` cannot be a `cmd` argument (253-char cap).
#[must_use]
pub fn job_command() -> Vec<String> {
    let boot = "import os,binascii;exec(binascii.unhexlify(''.join(os.environ[k] for k in sorted(os.environ) if k.startswith('PRISM_JS_C'))).decode())";
    debug_assert!(boot.len() < CMD_ARG_MAX);
    vec!["python3".into(), "-c".into(), boot.into()]
}

/// Hex-chunked job server for Verda `env` (plain).
#[must_use]
pub fn job_server_env_pairs() -> Vec<(String, String)> {
    let mut hex = String::with_capacity(JOB_SERVER_PY.len() * 2);
    for b in JOB_SERVER_PY.bytes() {
        hex.push(HEX[(b >> 4) as usize]);
        hex.push(HEX[(b & 0x0f) as usize]);
    }
    hex.as_bytes()
        .chunks(JOB_SERVER_ENV_CHUNK)
        .enumerate()
        .map(|(i, chunk)| {
            (
                format!("PRISM_JS_C{i:02}"),
                String::from_utf8_lossy(chunk).into_owned(),
            )
        })
        .collect()
}

/// Miner-facing sold-out copy (queued, not Score(0)).
pub const CAPACITY_POLICY: &str = "When Verda has no matching 1× B200 (or documented H200/H100 fallback) compute, the job stays queued and retries until a replica can start (sold out is not Score(0)). Bad ZIP, auth, and image-override errors still fail.";

/// Capacity note for events.
pub const CAPACITY_NOTE: &str =
    "B200s are currently out of capacity on Verda; this job is queued until compute appears.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_ok_and_tag_rejected() {
        let ok =
            "docker.io/x/y@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(require_digest(ok).unwrap(), ok);
        assert!(require_digest("docker.io/x/y:latest").is_err());
        assert!(require_digest("docker.io/x/y@sha256:abc").is_err());
    }

    #[test]
    fn job_server_hash_stable() {
        assert_eq!(job_server_sha256().len(), 64);
        assert!(JOB_SERVER_PY.contains("do_POST"));
        assert!(JOB_SERVER_PY.contains("/health"));
        let cmd = job_command();
        assert!(cmd.iter().all(|a| a.len() < CMD_ARG_MAX));
        assert!(!job_server_env_pairs().is_empty());
    }
}
