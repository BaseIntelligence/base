//! `miner` — miner deploy + certify CLI (`AGENT_CHALLENGE.md` §9).
//!
//! ```text
//! miner deploy --no-deploy
//! miner certify --fixture-mode --validator-url http://127.0.0.1:PORT ...
//! ```

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use crypto::{generate_mini_secret, public_key_from_mini_secret, KEY_LEN};
use miner::{
    certify, deploy_or_dry_run, empty_launch_token_hash_hex, parse_hotkey_hex, CertifyParams,
    DeployMode, DeployParams, QuoteSource, DEFAULT_AGENT_IMAGE, DEFAULT_ATTEST_HELPER_IMAGE,
    DEFAULT_SOCKET_PROXY_IMAGE,
};

#[derive(Debug, Parser)]
#[command(
    name = "miner",
    about = "Miner CVM deploy/certify CLI (AGENT_CHALLENGE.md section 9)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Render measured app-compose, print offline compose-hash, optionally phala deploy.
    Deploy {
        /// CVM / app-compose name.
        #[arg(long, default_value = "miner")]
        name: String,
        /// Digest-pinned agent image (`repo@sha256:<64 hex>`).
        #[arg(long, default_value = DEFAULT_AGENT_IMAGE, env = "GBASE_AGENT_IMAGE")]
        agent_image: String,
        /// Digest-pinned attest-helper image.
        #[arg(
            long,
            default_value = DEFAULT_ATTEST_HELPER_IMAGE,
            env = "GBASE_ATTEST_HELPER_IMAGE"
        )]
        attest_helper_image: String,
        /// Digest-pinned socket-proxy image (measured allowlist).
        #[arg(
            long,
            default_value = DEFAULT_SOCKET_PROXY_IMAGE,
            env = "GBASE_SOCKET_PROXY_IMAGE"
        )]
        socket_proxy_image: String,
        /// Lowercase hex SHA-256 of the launch token (measured; not the raw token).
        #[arg(long, env = "GBASE_LAUNCH_TOKEN_HASH")]
        launch_token_hash: Option<String>,
        /// Subnet netuid embedded as non-secret env.
        #[arg(long, default_value_t = 1, env = "GBASE_NETUID")]
        netuid: u16,
        /// Host path for the CVM-local receipt mini-secret (mode 0600). Generated if missing.
        #[arg(long, env = "GBASE_RECEIPT_SK_HOST_PATH", default_value = "receipt_sk")]
        receipt_sk_host_path: PathBuf,
        /// Optional pre-known receipt public key (64 hex). When omitted, derived from the secret file.
        #[arg(long, env = "GBASE_RECEIPT_PUBLIC_KEY")]
        receipt_public_key: Option<String>,
        /// Write rendered app-compose.json here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Skip `phala deploy` (default). Print compose-hash only.
        #[arg(long, conflicts_with = "deploy")]
        no_deploy: bool,
        /// Actually invoke `phala deploy` after hashing.
        #[arg(long)]
        deploy: bool,
        /// Path to `phala` binary.
        #[arg(long, default_value = "phala", env = "GBASE_PHALA_BIN")]
        phala_bin: PathBuf,
    },
    /// Request nonce, obtain D10-bound quote, submit to validator (task 38).
    Certify {
        /// Validator base URL (`http://host:port`).
        #[arg(long, env = "GBASE_VALIDATOR_URL")]
        validator_url: String,
        /// Subnet netuid.
        #[arg(long, default_value_t = 1, env = "GBASE_NETUID")]
        netuid: u16,
        /// Epoch to bind into `report_data`.
        #[arg(long, env = "GBASE_EPOCH")]
        epoch: u64,
        /// Miner hotkey (64 hex).
        #[arg(long, env = "GBASE_MINER_HOTKEY_HEX")]
        miner_hotkey_hex: String,
        /// Use embedded/real fixtures instead of a live CVM.
        #[arg(long, default_value_t = false)]
        fixture_mode: bool,
        /// Fixture directory with `quote.bin` + `event_log.json` (implies fixture mode).
        #[arg(long)]
        fixture_dir: Option<PathBuf>,
        /// Live agent / attest-helper base URL (ignored when `--fixture-mode`).
        #[arg(long, env = "GBASE_AGENT_URL")]
        agent_url: Option<String>,
        /// Optional validator hotkey override (defaults to nonce response).
        #[arg(long, env = "GBASE_VALIDATOR_HOTKEY_HEX")]
        validator_hotkey_hex: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("miner: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.cmd {
        Cmd::Deploy {
            name,
            agent_image,
            attest_helper_image,
            socket_proxy_image,
            launch_token_hash,
            netuid,
            receipt_sk_host_path,
            receipt_public_key,
            out,
            no_deploy: _,
            deploy,
            phala_bin,
        } => {
            let mode = if deploy {
                DeployMode::Deploy
            } else {
                DeployMode::NoDeploy
            };
            let receipt_public_key_hex =
                provision_receipt_key(&receipt_sk_host_path, receipt_public_key.as_deref())?;
            let params = DeployParams {
                name,
                agent_image,
                attest_helper_image,
                socket_proxy_image,
                launch_token_hash: launch_token_hash.unwrap_or_else(empty_launch_token_hash_hex),
                netuid,
                receipt_public_key_hex: receipt_public_key_hex.clone(),
                mode,
                out_compose: out,
                phala_bin,
            };
            let result = deploy_or_dry_run(&params).map_err(|e| e.to_string())?;
            println!("compose-hash={}", result.compose_hash_hex);
            println!("receipt-public-key={}", receipt_public_key_hex);
            println!("receipt-sk-host-path={}", receipt_sk_host_path.display());
            println!("phala_invoked={}", result.phala_invoked);
            println!("mode={mode:?}");
            println!("note=miner_funds_own_phala_account secrets_are_file_mounts_under_/run/gbase");
            Ok(())
        }
        Cmd::Certify {
            validator_url,
            netuid,
            epoch,
            miner_hotkey_hex,
            fixture_mode,
            fixture_dir,
            agent_url,
            validator_hotkey_hex,
        } => {
            let miner_hotkey = parse_hotkey_hex(&miner_hotkey_hex).map_err(|e| e.to_string())?;
            let validator_hotkey_override = match validator_hotkey_hex {
                Some(h) => Some(parse_hotkey_hex(&h).map_err(|e| e.to_string())?),
                None => None,
            };
            let use_fixture = fixture_mode || fixture_dir.is_some();
            let quote_source = if use_fixture {
                QuoteSource::Fixture { dir: fixture_dir }
            } else {
                let agent_base = agent_url.ok_or_else(|| {
                    "certify: --agent-url required unless --fixture-mode".to_owned()
                })?;
                QuoteSource::Live { agent_base }
            };
            let params = CertifyParams {
                validator_url,
                netuid,
                epoch,
                miner_hotkey,
                quote_source,
                validator_hotkey_override,
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            let result = rt.block_on(certify(&params)).map_err(|e| e.to_string())?;
            println!("nonce={}", result.nonce_hex);
            println!("outcome={}", result.outcome);
            if let Some(r) = &result.reason {
                println!("reason={r}");
            }
            println!("grants_credit={}", result.grants_credit);
            println!("carries_prior_verified={}", result.carries_prior_verified);
            println!("validator_hotkey={}", result.validator_hotkey_hex);
            println!("fixture_mode={}", result.fixture_mode);
            Ok(())
        }
    }
}

/// Load or generate a CVM-local receipt mini-secret on the host; return public hex.
///
/// Private bytes are written mode 0600 and never printed.
fn provision_receipt_key(
    path: &std::path::Path,
    known_public_hex: Option<&str>,
) -> Result<String, String> {
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let secret = if path.is_file() {
        let raw = fs::read(path).map_err(|e| e.to_string())?;
        parse_mini_secret(&raw)?
    } else {
        let secret = generate_mini_secret();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true).mode(0o600);
        let mut f = opts.open(path).map_err(|e| e.to_string())?;
        f.write_all(&secret).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        let mut perms = fs::metadata(path).map_err(|e| e.to_string())?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(path, perms).map_err(|e| e.to_string())?;
        secret
    };
    let derived = public_key_from_mini_secret(&secret).map_err(|e| e.to_string())?;
    let derived_hex = hex::encode(derived);
    if let Some(known) = known_public_hex {
        let known = known.trim().to_ascii_lowercase();
        if known != derived_hex {
            return Err(format!(
                "receipt public key mismatch: file derives {derived_hex}, flag has {known}"
            ));
        }
    }
    Ok(derived_hex)
}

fn parse_mini_secret(raw: &[u8]) -> Result<[u8; KEY_LEN], String> {
    if raw.len() == KEY_LEN {
        let mut out = [0u8; KEY_LEN];
        out.copy_from_slice(raw);
        // Validate expandable.
        let _ = public_key_from_mini_secret(&out).map_err(|e| e.to_string())?;
        return Ok(out);
    }
    let text = std::str::from_utf8(raw).map_err(|e| e.to_string())?;
    let trimmed = text.trim();
    let hex_s = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    let bytes = hex::decode(hex_s).map_err(|e| e.to_string())?;
    if bytes.len() != KEY_LEN {
        return Err(format!("receipt secret must be {KEY_LEN} bytes"));
    }
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&bytes);
    let _ = public_key_from_mini_secret(&out).map_err(|e| e.to_string())?;
    Ok(out)
}
