//! Render measured app-compose, offline hash, optional `phala deploy`.

use std::path::{Path, PathBuf};
use std::process::Command;

use compose_hash::{compose_hash_hex, ComposeHashError};
use serde_json::{json, Value};
use thiserror::Error;

use crate::inspect::reject_raw_docker_sock_on_agent;
pub use crate::template::DEFAULT_SOCKET_PROXY_IMAGE;
use crate::template::{docker_compose_yaml, ComposeTemplateInput, DOCKER_BASE_ENV};

/// Default digest-pinned agent image (digest from images CI tip 3056ca7).
pub const DEFAULT_AGENT_IMAGE: &str =
    "ghcr.io/baseintelligence/gbase/gbase-agent@sha256:50508825f450c6d1b21e53bf61cda8eeee6373eaced24ec3925555feac3ebc83";
/// Default digest-pinned attest-helper image.
pub const DEFAULT_ATTEST_HELPER_IMAGE: &str =
    "ghcr.io/baseintelligence/gbase/gbase-attest-helper@sha256:9bf28955414f087d27e033e085a16be2fda20290404add83bd919893c24e8d7c";

/// Whether to invoke the Phala CLI after rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployMode {
    /// Render + print compose-hash only (default for tests / dry-run).
    NoDeploy,
    /// Render, print hash, then run `phala deploy`.
    Deploy,
}

/// Parameters for a miner CVM deploy render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeployParams {
    /// CVM / app-compose display name.
    pub name: String,
    /// Digest-pinned agent image.
    pub agent_image: String,
    /// Digest-pinned attest-helper image.
    pub attest_helper_image: String,
    /// Digest-pinned socket-proxy image (measured allowlist path).
    pub socket_proxy_image: String,
    /// Lowercase hex SHA-256 of launch token (measured).
    pub launch_token_hash: String,
    /// Subnet netuid.
    pub netuid: u16,
    /// Work-receipt public key (64 lowercase hex) published in measured compose.
    pub receipt_public_key_hex: String,
    /// Deploy vs dry-run.
    pub mode: DeployMode,
    /// Optional path to write `app-compose.json`.
    pub out_compose: Option<PathBuf>,
    /// `phala` binary (default `phala` on PATH).
    pub phala_bin: PathBuf,
}

impl Default for DeployParams {
    fn default() -> Self {
        Self {
            name: "miner".to_owned(),
            agent_image: DEFAULT_AGENT_IMAGE.to_owned(),
            attest_helper_image: DEFAULT_ATTEST_HELPER_IMAGE.to_owned(),
            socket_proxy_image: DEFAULT_SOCKET_PROXY_IMAGE.to_owned(),
            // Deterministic empty-token hash placeholder for offline dry-runs.
            launch_token_hash: empty_launch_token_hash_hex(),
            netuid: 1,
            // Deterministic placeholder pubkey for offline dry-runs (not a real sk).
            receipt_public_key_hex: "11".repeat(32),
            mode: DeployMode::NoDeploy,
            out_compose: None,
            phala_bin: PathBuf::from("phala"),
        }
    }
}

/// Result of render (+ optional phala).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeployResult {
    /// Canonical compose-hash as 64 lowercase hex chars.
    pub compose_hash_hex: String,
    /// Pretty-printed app-compose JSON (stable field set).
    pub app_compose_json: String,
    /// True when `phala deploy` was invoked successfully.
    pub phala_invoked: bool,
}

/// Deploy / render failures.
#[derive(Debug, Error)]
pub enum DeployError {
    /// Image ref is not digest-pinned or uses `:latest`.
    #[error("image must be digest-pinned (repo@sha256:<64 hex>), got: {0}")]
    ImageNotDigestPinned(String),
    /// Launch token hash is not 64 lowercase hex chars.
    #[error("launch_token_hash must be 64 lowercase hex chars")]
    BadLaunchTokenHash,
    /// Receipt public key is not 64 lowercase hex chars.
    #[error("receipt_public_key_hex must be 64 lowercase hex chars")]
    BadReceiptPublicKey,
    /// Agent service mounts raw docker.sock (must use measured socket-proxy).
    #[error(transparent)]
    RawDockerSock(#[from] crate::inspect::RawDockerSockOnAgent),
    /// Compose JSON hashing failed.
    #[error(transparent)]
    ComposeHash(#[from] ComposeHashError),
    /// JSON serialization failed.
    #[error("serialize app-compose: {0}")]
    Serialize(String),
    /// Filesystem write failed.
    #[error("write {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// `phala` CLI failed or missing.
    #[error("phala deploy failed: {0}")]
    Phala(String),
}

/// SHA-256 of empty bytes as default launch-token hash (offline placeholder).
#[must_use]
pub fn empty_launch_token_hash_hex() -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest([]);
    hex_encode(d.as_slice())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn is_digest_pinned(image: &str) -> bool {
    // repo@sha256:<64 hex> — reject :latest anywhere.
    if image.contains(":latest") {
        return false;
    }
    let Some((_, digest)) = image.split_once("@sha256:") else {
        return false;
    };
    digest.len() == 64 && digest.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_hex64_lower(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

/// Build the full Phala `app-compose.json` document as a [`Value`].
///
/// # Errors
/// Returns [`DeployError`] when images or launch-token hash fail validation.
pub fn render_app_compose(params: &DeployParams) -> Result<Value, DeployError> {
    if !is_digest_pinned(&params.agent_image) {
        return Err(DeployError::ImageNotDigestPinned(
            params.agent_image.clone(),
        ));
    }
    if !is_digest_pinned(&params.attest_helper_image) {
        return Err(DeployError::ImageNotDigestPinned(
            params.attest_helper_image.clone(),
        ));
    }
    if !is_digest_pinned(&params.socket_proxy_image) {
        return Err(DeployError::ImageNotDigestPinned(
            params.socket_proxy_image.clone(),
        ));
    }
    if !is_hex64_lower(&params.launch_token_hash) {
        return Err(DeployError::BadLaunchTokenHash);
    }
    if !is_hex64_lower(&params.receipt_public_key_hex) {
        return Err(DeployError::BadReceiptPublicKey);
    }

    let yaml = docker_compose_yaml(&ComposeTemplateInput {
        agent_image: &params.agent_image,
        attest_helper_image: &params.attest_helper_image,
        socket_proxy_image: &params.socket_proxy_image,
        launch_token_hash: &params.launch_token_hash,
        netuid: params.netuid,
        receipt_public_key_hex: &params.receipt_public_key_hex,
    });
    reject_raw_docker_sock_on_agent(&yaml)?;

    // Field set mirrors dstack/Phala app-compose v2 (see real fixture layout).
    // Null-valued keys are stripped by compose_hash; we omit them for clarity.
    Ok(json!({
        "allowed_envs": [
            "GBASE_NETUID",
            "GBASE_MINER_HOTKEY_FILE",
            "GBASE_LAUNCH_TOKEN_HASH",
            "GBASE_RECEIPT_SK_FILE",
            "GBASE_RECEIPT_PUBLIC_KEY",
            DOCKER_BASE_ENV
        ],

        "docker_compose_file": yaml,
        "features": ["kms", "tproxy-net"],
        "gateway_enabled": true,
        "kms_enabled": true,
        "local_key_provider_enabled": false,
        "manifest_version": 2,
        "name": params.name,
        "no_instance_id": false,
        "public_logs": true,
        "public_sysinfo": true,
        "public_tcbinfo": true,
        "runner": "docker-compose",
        "secure_time": false,
        "storage_fs": "zfs",
        "tproxy_enabled": true
    }))
}

/// Serialize app-compose to compact JSON bytes suitable for [`compose_hash`].
///
/// # Errors
/// Same as [`render_app_compose`], plus serialization errors.
pub fn render_app_compose_bytes(params: &DeployParams) -> Result<Vec<u8>, DeployError> {
    let value = render_app_compose(params)?;
    // Compact form; compose_hash re-canonicalizes key order anyway.
    serde_json::to_vec(&value).map_err(|e| DeployError::Serialize(e.to_string()))
}

/// Render compose, compute offline hash, optionally write file and run phala.
///
/// # Errors
/// Validation, IO, hash, or phala failures.
pub fn deploy_or_dry_run(params: &DeployParams) -> Result<DeployResult, DeployError> {
    let bytes = render_app_compose_bytes(params)?;
    let hash_hex = compose_hash_hex(&bytes)?;
    // Pretty JSON for operator inspection / --out.
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|e| DeployError::Serialize(e.to_string()))?;
    let pretty =
        serde_json::to_string_pretty(&value).map_err(|e| DeployError::Serialize(e.to_string()))?;

    if let Some(path) = &params.out_compose {
        std::fs::write(path, pretty.as_bytes()).map_err(|source| DeployError::Io {
            path: path.clone(),
            source,
        })?;
    }

    let mut phala_invoked = false;
    if params.mode == DeployMode::Deploy {
        let compose_path = if let Some(p) = &params.out_compose {
            p.clone()
        } else {
            let tmp =
                std::env::temp_dir().join(format!("miner-{}-app-compose.json", &hash_hex[..16]));
            std::fs::write(&tmp, pretty.as_bytes()).map_err(|source| DeployError::Io {
                path: tmp.clone(),
                source,
            })?;
            tmp
        };
        run_phala_deploy(&params.phala_bin, &compose_path, &params.name)?;
        phala_invoked = true;
    }

    Ok(DeployResult {
        compose_hash_hex: hash_hex,
        app_compose_json: pretty,
        phala_invoked,
    })
}

/// Invoke `phala deploy --compose <path> --name <name>` (best-effort argv).
///
/// # Errors
/// Missing binary or non-zero exit.
pub fn run_phala_deploy(phala_bin: &Path, compose: &Path, name: &str) -> Result<(), DeployError> {
    let output = Command::new(phala_bin)
        .args([
            "deploy",
            "--compose",
            &compose.display().to_string(),
            "--name",
            name,
        ])
        .output()
        .map_err(|e| {
            DeployError::Phala(format!(
                "failed to spawn `{}` ({e}). Install the Phala CLI and fund your own Phala account (R3).",
                phala_bin.display()
            ))
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Err(DeployError::Phala(format!(
        "exit {:?}: stdout={stdout} stderr={stderr}",
        output.status.code()
    )))
}
