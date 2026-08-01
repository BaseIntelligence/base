//! Render measured app-compose, offline hash, optional `phala deploy`.

use std::path::{Path, PathBuf};
use std::process::Command;

use compose_hash::{compose_hash_hex, ComposeHashError};
use serde_json::{json, Value};
use thiserror::Error;

use crate::inspect::reject_raw_docker_sock_on_agent;
pub use crate::template::DEFAULT_SOCKET_PROXY_IMAGE;
use crate::template::{
    docker_compose_yaml, pre_launch_script, ComposeTemplateInput, DEFAULT_ENVIRONMENT_IMAGE,
    DEFAULT_PACK_ROOT_IN_CVM, LAUNCH_TOKEN_ENV, MINER_HOTKEY_HEX_ENV, RECEIPT_SK_HEX_ENV,
};

/// Default digest-pinned agent image (digest from the images CI run for abab330a).
pub const DEFAULT_AGENT_IMAGE: &str =
    "ghcr.io/baseintelligence/base/base-agent@sha256:d8f7722896156f0f3dda0c50f0f6897f93b6442dfc9d85d8a8c675a50c81216f";
/// Default digest-pinned attest-helper image.
pub const DEFAULT_ATTEST_HELPER_IMAGE: &str =
    "ghcr.io/baseintelligence/base/base-attest-helper@sha256:7fe62eadb9e7f48f63c81586ec5e152626882d9f649e0def8dcb0f2dd847501e";
/// Default pack catalog HTTP base (local/dev; production overlays gateway).
pub const DEFAULT_PACK_CATALOG_URL: &str = "http://127.0.0.1:8090";
/// Default trusted challenge pubkey (`config/challenges.toml` agent-v1 `public_key`).
pub const DEFAULT_TRUSTED_CHALLENGE_PUBKEY_HEX: &str =
    "f2e4965a6a99b75b4212bd45790c496e9665c0e1247e373d9dca3b36413fbd45";

/// Whether to invoke the Phala CLI after rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployMode {
    /// Render + print compose-hash only (default for tests / dry-run).
    NoDeploy,
    /// Render, print hash, then run `phala deploy`.
    Deploy,
}

/// Values handed to the CVM as Phala **encrypted secrets**.
///
/// These are never measured, never written into `app_compose_json`, and never
/// printed: the measured compose only carries their variable names.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct DeploySecrets {
    /// Work-receipt mini-secret as 64 hex chars (`agent-runner` accepts hex).
    pub receipt_sk_hex: String,
    /// Raw launch token whose SHA-256 is measured.
    pub launch_token: String,
    /// Public miner hotkey hex (not secret, but travels the same channel).
    pub miner_hotkey_hex: String,
}

// Derived Debug would print the secrets through any `{params:?}`.
impl std::fmt::Debug for DeploySecrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeploySecrets")
            .field("receipt_sk_hex", &"<redacted>")
            .field("launch_token", &"<redacted>")
            .field("miner_hotkey_hex", &"<redacted>")
            .finish()
    }
}

impl DeploySecrets {
    /// `NAME=VALUE` lines for the `phala deploy -e <file>` env file.
    fn env_file_body(&self) -> String {
        format!(
            "{RECEIPT_SK_HEX_ENV}={}\n{LAUNCH_TOKEN_ENV}={}\n{MINER_HOTKEY_HEX_ENV}={}\n",
            self.receipt_sk_hex, self.launch_token, self.miner_hotkey_hex
        )
    }

    fn require_all(&self) -> Result<(), DeployError> {
        for (name, value) in [
            (RECEIPT_SK_HEX_ENV, &self.receipt_sk_hex),
            (LAUNCH_TOKEN_ENV, &self.launch_token),
            (MINER_HOTKEY_HEX_ENV, &self.miner_hotkey_hex),
        ] {
            if value.trim().is_empty() {
                return Err(DeployError::MissingSecret(name));
            }
        }
        Ok(())
    }
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
    /// Digest-pinned Harbor environment image for pack runs.
    pub environment_image: String,
    /// Pack root path inside the agent container.
    pub pack_root: String,
    /// Pack catalog HTTP base URL.
    pub pack_catalog_url: String,
    /// Trusted challenge public key (64 lowercase hex).
    pub trusted_challenge_pubkey_hex: String,
    /// Encrypted-secret values (required for [`DeployMode::Deploy`] only).
    pub secrets: DeploySecrets,
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
            environment_image: DEFAULT_ENVIRONMENT_IMAGE.to_owned(),
            pack_root: DEFAULT_PACK_ROOT_IN_CVM.to_owned(),
            pack_catalog_url: DEFAULT_PACK_CATALOG_URL.to_owned(),
            trusted_challenge_pubkey_hex: DEFAULT_TRUSTED_CHALLENGE_PUBKEY_HEX.to_owned(),
            secrets: DeploySecrets::default(),
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
    /// Trusted challenge public key is not 64 lowercase hex chars.
    #[error("trusted_challenge_pubkey_hex must be 64 lowercase hex chars")]
    BadTrustedChallengePubkey,
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
    /// An encrypted-secret value required for a real deploy is absent.
    #[error(
        "missing encrypted-secret value for {0}; pass --receipt-sk-host-path, \
         --launch-token-file and --miner-hotkey-hex"
    )]
    MissingSecret(&'static str),
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

/// Measured launch-token hash for a raw token (what the attest-helper compares
/// the bearer credential against).
#[must_use]
pub fn launch_token_hash_hex(token: &str) -> String {
    use sha2::{Digest, Sha256};
    hex_encode(Sha256::digest(token.as_bytes()).as_slice())
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
    if !is_digest_pinned(&params.environment_image) {
        return Err(DeployError::ImageNotDigestPinned(
            params.environment_image.clone(),
        ));
    }
    if !is_hex64_lower(&params.launch_token_hash) {
        return Err(DeployError::BadLaunchTokenHash);
    }
    if !is_hex64_lower(&params.receipt_public_key_hex) {
        return Err(DeployError::BadReceiptPublicKey);
    }
    if !is_hex64_lower(&params.trusted_challenge_pubkey_hex) {
        return Err(DeployError::BadTrustedChallengePubkey);
    }

    let yaml = docker_compose_yaml(&ComposeTemplateInput {
        agent_image: &params.agent_image,
        attest_helper_image: &params.attest_helper_image,
        socket_proxy_image: &params.socket_proxy_image,
        launch_token_hash: &params.launch_token_hash,
        netuid: params.netuid,
        receipt_public_key_hex: &params.receipt_public_key_hex,
        environment_image: &params.environment_image,
        pack_root: &params.pack_root,
        pack_catalog_url: &params.pack_catalog_url,
        trusted_challenge_pubkey_hex: &params.trusted_challenge_pubkey_hex,
    });
    reject_raw_docker_sock_on_agent(&yaml)?;

    // Field set mirrors dstack/Phala app-compose v2 (see real fixture layout).
    // Null-valued keys are stripped by compose_hash; we omit them for clarity.
    Ok(json!({
        // Measured allowlist: a variable the guest was not measured to accept
        // cannot be smuggled in. The CLI rewrites this from the names in the
        // `-e` env file, in file order, and appends its own
        // DSTACK_AUTHORIZED_KEYS, so mirroring that exactly is what makes the
        // hash printed here equal the one Phala measures. Verified against
        // app_id 340ead2af2ff1d950d47a6fae0ffa473854b5d96. Every other BASE_*
        // setting is a literal in the compose and needs no entry.
        "allowed_envs": [
            RECEIPT_SK_HEX_ENV,
            LAUNCH_TOKEN_ENV,
            MINER_HOTKEY_HEX_ENV,
            "DSTACK_AUTHORIZED_KEYS"
        ],

        "docker_compose_file": yaml,
        // Materialises the three bind sources from encrypted-secret values.
        // Nothing else creates them, so without this the guest boots without
        // /run/base/receipt_sk and agent-runner exits.
        "pre_launch_script": pre_launch_script(),
        "features": ["kms", "tproxy-net"],
        "gateway_enabled": true,
        "kms_enabled": true,
        "local_key_provider_enabled": false,
        "manifest_version": 2,
        // The display name travels as `phala deploy --name` and the CLI leaves
        // this empty, so putting `params.name` here would only desync the hash.
        "name": "",
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
        params.secrets.require_all()?;
        run_phala_deploy_with_secrets(params, &value, &hash_hex[..16])?;
        phala_invoked = true;
    }

    Ok(DeployResult {
        compose_hash_hex: hash_hex,
        app_compose_json: pretty,
        phala_invoked,
    })
}

/// Files `phala deploy` reads: the CLI builds the app-compose itself from the
/// docker-compose YAML plus the pre-launch script, so `-c` is *not* the
/// rendered `app-compose.json`.
#[derive(Debug, Clone, Copy)]
pub struct PhalaDeployInvocation<'a> {
    /// `phala` binary.
    pub phala_bin: &'a Path,
    /// docker-compose YAML (the `-c` argument).
    pub docker_compose: &'a Path,
    /// Rendered pre-launch script.
    pub pre_launch_script: &'a Path,
    /// Mode-0600 `.env` file with the encrypted-secret values.
    pub env_file: &'a Path,
    /// CVM display name.
    pub name: &'a str,
}

/// Write the CLI inputs to temp files, deploy, then shred the secret env file.
fn run_phala_deploy_with_secrets(
    params: &DeployParams,
    app_compose: &Value,
    stem: &str,
) -> Result<(), DeployError> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let dir = std::env::temp_dir();
    let compose_path = dir.join(format!("miner-{stem}-docker-compose.yml"));
    let script_path = dir.join(format!("miner-{stem}-pre-launch.sh"));
    let env_path = dir.join(format!("miner-{stem}.env"));

    let yaml = app_compose["docker_compose_file"]
        .as_str()
        .unwrap_or_default();
    write_file(&compose_path, yaml.as_bytes())?;
    let script = app_compose["pre_launch_script"]
        .as_str()
        .unwrap_or_default();
    write_file(&script_path, script.as_bytes())?;

    // Secrets go through a 0600 file rather than argv, because argv is world
    // readable via /proc for the lifetime of the child.
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true).mode(0o600);
    let write_env = opts
        .open(&env_path)
        .and_then(|mut f| f.write_all(params.secrets.env_file_body().as_bytes()));

    let outcome = write_env
        .map_err(|source| DeployError::Io {
            path: env_path.clone(),
            source,
        })
        .and_then(|()| {
            run_phala_deploy(&PhalaDeployInvocation {
                phala_bin: &params.phala_bin,
                docker_compose: &compose_path,
                pre_launch_script: &script_path,
                env_file: &env_path,
                name: &params.name,
            })
        });
    // Also on the error path: the file holds the receipt private key.
    let _ = std::fs::remove_file(&env_path);
    outcome
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), DeployError> {
    std::fs::write(path, bytes).map_err(|source| DeployError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Invoke `phala deploy --name <n> -c <yml> --pre-launch-script <sh> -e <env>`.
///
/// # Errors
/// Missing binary or non-zero exit. The argv is never quoted back into the
/// error, because `--pre-launch-script` and `-e` name files holding secrets.
pub fn run_phala_deploy(inv: &PhalaDeployInvocation<'_>) -> Result<(), DeployError> {
    let phala_bin = inv.phala_bin;
    let output = Command::new(phala_bin)
        .args([
            "deploy",
            "--name",
            inv.name,
            "-c",
            &inv.docker_compose.display().to_string(),
            "--pre-launch-script",
            &inv.pre_launch_script.display().to_string(),
            "-e",
            &inv.env_file.display().to_string(),
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
