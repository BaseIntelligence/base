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
    announce, attest_grant, certify, deploy_or_dry_run, empty_launch_token_hash_hex,
    launch_token_hash_hex, parse_hotkey_hex, AnnounceParams, AttestGrantParams, CertifyParams,
    DeployMode, DeployParams, DeploySecrets, QuoteSource, DEFAULT_AGENT_IMAGE,
    DEFAULT_ATTEST_HELPER_IMAGE, DEFAULT_ENVIRONMENT_IMAGE, DEFAULT_PACK_CATALOG_URL,
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

#[derive(Debug, clap::Args)]
struct DeployArgs {
    /// CVM / app-compose name.
    #[arg(long, default_value = "miner")]
    name: String,
    /// Digest-pinned agent image (`repo@sha256:<64 hex>`).
    #[arg(long, default_value = DEFAULT_AGENT_IMAGE, env = "BASE_AGENT_IMAGE")]
    agent_image: String,
    /// Digest-pinned attest-helper image.
    #[arg(
            long,
            default_value = DEFAULT_ATTEST_HELPER_IMAGE,
            env = "BASE_ATTEST_HELPER_IMAGE"
        )]
    attest_helper_image: String,
    /// Digest-pinned socket-proxy image (measured allowlist).
    #[arg(
            long,
            default_value = DEFAULT_SOCKET_PROXY_IMAGE,
            env = "BASE_SOCKET_PROXY_IMAGE"
        )]
    socket_proxy_image: String,
    /// Lowercase hex SHA-256 of the launch token (measured; not the raw token).
    #[arg(long, env = "BASE_LAUNCH_TOKEN_HASH")]
    launch_token_hash: Option<String>,
    /// Host path for the raw launch token (mode 0600). Generated if missing.
    #[arg(long, env = "BASE_LAUNCH_TOKEN_FILE")]
    launch_token_file: Option<PathBuf>,
    /// Subnet netuid embedded as non-secret env.
    #[arg(long, default_value_t = 1, env = "BASE_NETUID")]
    netuid: u16,
    /// Host path for the CVM-local receipt mini-secret (mode 0600). Generated if missing.
    #[arg(long, env = "BASE_RECEIPT_SK_HOST_PATH", default_value = "receipt_sk")]
    receipt_sk_host_path: PathBuf,
    /// Optional pre-known receipt public key (64 hex). When omitted, derived from the secret file.
    #[arg(long, env = "BASE_RECEIPT_PUBLIC_KEY")]
    receipt_public_key: Option<String>,
    /// Public miner hotkey (64 hex). Required for `--deploy`.
    #[arg(long, env = "BASE_MINER_HOTKEY_HEX")]
    miner_hotkey_hex: Option<String>,
    /// Operator-published pack catalog base URL, as reachable from inside the CVM.
    #[arg(long, env = "BASE_PACK_CATALOG_URL", default_value = DEFAULT_PACK_CATALOG_URL)]
    pack_catalog_url: String,
    /// Digest-pinned Harbor environment image the agent runs pack tasks in.
    #[arg(long, env = "BASE_ENVIRONMENT_IMAGE", default_value = DEFAULT_ENVIRONMENT_IMAGE)]
    environment_image: String,
    /// Real agent command as a JSON array. Measured into the compose when set;
    /// omitted, the runner falls back to its built-in reference placeholder.
    #[arg(long, env = "BASE_AGENT_CMD")]
    agent_cmd: Option<String>,
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
    #[arg(long, default_value = "phala", env = "BASE_PHALA_BIN")]
    phala_bin: PathBuf,
    /// Pin deployment to a specific Phala node id (placement only; auto-selected
    /// when omitted). Useful when the auto-selected node never materialises
    /// containers, as prod9 proved on 2026-08-01.
    #[arg(long, env = "BASE_PHALA_NODE_ID")]
    node_id: Option<String>,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Render measured app-compose, print offline compose-hash, optionally phala deploy.
    Deploy(DeployArgs),
    /// Request nonce, obtain D10-bound quote, submit to validator (task 38).
    Certify {
        /// Validator base URL (`http://host:port`).
        #[arg(long, env = "BASE_VALIDATOR_URL")]
        validator_url: String,
        /// Subnet netuid.
        #[arg(long, default_value_t = 1, env = "BASE_NETUID")]
        netuid: u16,
        /// Epoch to bind into `report_data`.
        #[arg(long, env = "BASE_EPOCH")]
        epoch: u64,
        /// Miner hotkey (64 hex).
        #[arg(long, env = "BASE_MINER_HOTKEY_HEX")]
        miner_hotkey_hex: String,
        /// Use embedded/real fixtures instead of a live CVM.
        #[arg(long, default_value_t = false)]
        fixture_mode: bool,
        /// Fixture directory with `quote.bin` + `event_log.json` (implies fixture mode).
        #[arg(long)]
        fixture_dir: Option<PathBuf>,
        /// Live agent / attest-helper base URL (ignored when `--fixture-mode`).
        #[arg(long, env = "BASE_AGENT_URL")]
        agent_url: Option<String>,
        /// Optional validator hotkey override (defaults to nonce response).
        #[arg(long, env = "BASE_VALIDATOR_HOTKEY_HEX")]
        validator_hotkey_hex: Option<String>,
        /// Host path of the raw launch token presented to the CVM attest-helper.
        #[arg(long, env = "BASE_LAUNCH_TOKEN_FILE")]
        launch_token_file: Option<PathBuf>,
        /// `app-compose.json` written by `deploy --out`, submitted as the
        /// RTMR3 compose preimage when the quote response omits it.
        #[arg(long, env = "BASE_APP_COMPOSE_FILE")]
        app_compose_file: Option<PathBuf>,
    },
    /// Announce the CVM's public base URL to the gateway (§9.3 step 5).
    Announce {
        /// Gateway base URL (`https://host`).
        #[arg(long, env = "BASE_GATEWAY_URL")]
        gateway_url: String,
        /// Subnet netuid.
        #[arg(long, default_value_t = 1, env = "BASE_NETUID")]
        netuid: u16,
        /// Current chain epoch; the gate rejects any other value.
        #[arg(long, env = "BASE_EPOCH")]
        epoch: u64,
        /// Public CVM base URL to announce (origin only, no path).
        #[arg(long, env = "BASE_MINER_BASE_URL")]
        base_url: String,
    },
    /// Owner-issued credit for a non-TEE runtime (§9.6, master only).
    AttestGrant {
        /// Master gateway base URL (internal network / VPC only).
        #[arg(long, env = "BASE_GATEWAY_URL")]
        gateway_url: String,
        /// Epoch the credit binds to (attestation rows are epoch-keyed).
        #[arg(long, env = "BASE_EPOCH")]
        epoch: u64,
        /// Miner hotkey (64 hex). Defaults to the keystore's public key.
        #[arg(long, env = "BASE_MINER_HOTKEY_HEX")]
        miner_hotkey_hex: Option<String>,
        /// Host path of the receipt mini-secret (raw 32 bytes or hex, mode 0600).
        #[arg(long, env = "BASE_RECEIPT_SK_FILE", default_value = "receipt_sk")]
        receipt_sk_file: PathBuf,
        /// Mandatory audit note stored as `reason: admin-exempt: …`.
        #[arg(long, env = "BASE_ATTEST_GRANT_REASON")]
        reason: String,
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
        Cmd::Deploy(args) => run_deploy(args),
        Cmd::Certify {
            validator_url,
            netuid,
            epoch,
            miner_hotkey_hex,
            fixture_mode,
            fixture_dir,
            agent_url,
            validator_hotkey_hex,
            launch_token_file,
            app_compose_file,
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
            let token_file = launch_token_file.as_deref();
            let params = CertifyParams {
                validator_url,
                netuid,
                epoch,
                miner_hotkey,
                quote_source,
                validator_hotkey_override,
                launch_token: token_file.map(read_launch_token).transpose()?,
                app_compose_override: app_compose_file
                    .as_deref()
                    .map(std::fs::read_to_string)
                    .transpose()
                    .map_err(|e| format!("app-compose file: {e}"))?,
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            let result = rt.block_on(certify(&params)).map_err(|e| e.to_string())?;
            print_certify_result(&result);
            Ok(())
        }
        Cmd::Announce {
            gateway_url,
            netuid,
            epoch,
            base_url,
        } => run_announce(gateway_url, netuid, epoch, base_url),
        Cmd::AttestGrant {
            gateway_url,
            epoch,
            miner_hotkey_hex,
            receipt_sk_file,
            reason,
        } => run_attest_grant(
            &gateway_url,
            epoch,
            miner_hotkey_hex,
            &receipt_sk_file,
            &reason,
        ),
    }
}

/// POST the owner grant for the local (non-TEE) runtime (§9.6).
fn run_attest_grant(
    gateway_url: &str,
    epoch: u64,
    miner_hotkey_hex: Option<String>,
    receipt_sk_file: &PathBuf,
    reason: &str,
) -> Result<(), String> {
    let miner_hotkey = if let Some(h) = miner_hotkey_hex {
        parse_hotkey_hex(&h.trim().to_ascii_lowercase()).map_err(|e| e.to_string())?
    } else {
        let sk = resolve_miner_hotkey_secret()?;
        public_key_from_mini_secret(&sk).map_err(|e| e.to_string())?
    };
    let raw = std::fs::read(receipt_sk_file)
        .map_err(|e| format!("receipt secret {}: {e}", receipt_sk_file.display()))?;
    let receipt_secret = parse_mini_secret(&raw)?;
    let receipt_public_key =
        public_key_from_mini_secret(&receipt_secret).map_err(|e| e.to_string())?;
    if reason.trim().is_empty() {
        return Err("--reason must be a non-empty audit note".to_owned());
    }
    let params = AttestGrantParams {
        gateway_url: gateway_url.to_owned(),
        epoch,
        miner_hotkey,
        receipt_public_key,
        reason: reason.trim().to_owned(),
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let out = rt
        .block_on(attest_grant(&params))
        .map_err(|e| e.to_string())?;
    println!("miner_hotkey={}", out.miner_hotkey_hex);
    println!("receipt_pk={}", out.receipt_pk_hex);
    println!("epoch={}", out.epoch);
    println!("attempt={}", out.attempt);
    println!("outcome={}", out.outcome);
    Ok(())
}

/// Render the measured compose, then optionally hand the secrets to Phala.
fn run_deploy(args: DeployArgs) -> Result<(), String> {
    let DeployArgs {
        name,
        agent_image,
        attest_helper_image,
        socket_proxy_image,
        launch_token_hash,
        launch_token_file,
        netuid,
        receipt_sk_host_path,
        receipt_public_key,
        miner_hotkey_hex,
        pack_catalog_url,
        environment_image,
        agent_cmd,
        out,
        no_deploy: _,
        deploy,
        phala_bin,
        node_id,
    } = args;
    let mode = if deploy {
        DeployMode::Deploy
    } else {
        DeployMode::NoDeploy
    };
    let (receipt_public_key_hex, receipt_secret) =
        provision_receipt_key(&receipt_sk_host_path, receipt_public_key.as_deref())?;
    let launch_token_hash =
        resolve_launch_token_hash(launch_token_hash, launch_token_file.as_deref())?;
    let secrets = collect_deploy_secrets(
        &receipt_secret,
        launch_token_file.as_deref(),
        miner_hotkey_hex.as_deref(),
    )?;
    let agent_cmd_json = agent_cmd
        .as_deref()
        .map(validate_agent_cmd_json)
        .transpose()?;
    let params = DeployParams {
        name,
        agent_image,
        attest_helper_image,
        socket_proxy_image,
        launch_token_hash: launch_token_hash.clone(),
        netuid,
        receipt_public_key_hex: receipt_public_key_hex.clone(),
        pack_catalog_url,
        environment_image,
        agent_cmd_json,
        secrets,
        mode,
        out_compose: out,
        phala_bin,
        phala_node_id: node_id,
        ..DeployParams::default()
    };
    let result = deploy_or_dry_run(&params).map_err(|e| e.to_string())?;
    println!("compose-hash={}", result.compose_hash_hex);
    println!("receipt-public-key={receipt_public_key_hex}");
    println!("receipt-sk-host-path={}", receipt_sk_host_path.display());
    print_launch_token_report(launch_token_file.as_deref(), &launch_token_hash);
    println!("phala_invoked={}", result.phala_invoked);
    println!("mode={mode:?}");
    println!(
        "note=miner_funds_own_phala_account \
                 secrets_travel_as_phala_encrypted_env_and_land_as_file_mounts_under_/run/base"
    );
    Ok(())
}

fn validate_agent_cmd_json(raw: &str) -> Result<String, String> {
    let value: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| format!("--agent-cmd is not JSON: {e}"))?;
    let items = value
        .as_array()
        .ok_or_else(|| "--agent-cmd must be a JSON array of strings".to_owned())?;
    if items.is_empty() || !items.iter().all(serde_json::Value::is_string) {
        return Err("--agent-cmd must be a JSON array of strings".into());
    }
    Ok(raw.to_owned())
}

fn resolve_launch_token_hash(
    hash: Option<String>,
    token_file: Option<&std::path::Path>,
) -> Result<String, String> {
    match (hash, token_file) {
        // Measuring a hash the operator's own token does not produce yields a
        // CVM that answers 401 to its owner, and only after deploying.
        (Some(h), Some(path)) => {
            let derived = launch_token_hash_hex(&read_launch_token(path)?);
            if derived == h {
                Ok(h)
            } else {
                Err(format!(
                    "--launch-token-hash does not match the token in {}",
                    path.display()
                ))
            }
        }
        (Some(h), None) => Ok(h),
        (None, Some(path)) => provision_launch_token(path),
        (None, None) => Ok(empty_launch_token_hash_hex()),
    }
}

fn print_launch_token_report(launch_token_file: Option<&std::path::Path>, hash: &str) {
    match launch_token_file {
        Some(path) => println!("launch-token-host-path={}", path.display()),
        None => println!("launch-token-host-path="),
    }
    if hash == empty_launch_token_hash_hex() {
        println!(
            "warning=no_launch_token_configured \
             attest_helper_will_refuse_quotes_pass_--launch-token-file"
        );
    }
}

fn print_certify_result(result: &miner::CertifyResult) {
    println!("nonce={}", result.nonce_hex);
    println!("outcome={}", result.outcome);
    if let Some(r) = &result.reason {
        println!("reason={r}");
    }
    println!("grants_credit={}", result.grants_credit);
    println!("carries_prior_verified={}", result.carries_prior_verified);
    println!("validator_hotkey={}", result.validator_hotkey_hex);
    println!("fixture_mode={}", result.fixture_mode);
}

/// Sign and POST the CVM base URL to the gateway (§9.3 step 5).
fn run_announce(
    gateway_url: String,
    netuid: u16,
    epoch: u64,
    base_url: String,
) -> Result<(), String> {
    let params = AnnounceParams {
        gateway_url,
        netuid,
        epoch,
        base_url,
        hotkey_secret: resolve_miner_hotkey_secret()?,
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let out = rt.block_on(announce(&params)).map_err(|e| e.to_string())?;
    println!("miner_hotkey={}", out.miner_hotkey_hex);
    println!("base_url={}", out.base_url);
    println!("epoch={}", out.epoch);
    println!("netuid={}", out.netuid);
    Ok(())
}

/// Load the miner hotkey mini-secret the same way every other role loads its
/// signing key: wallet, then mnemonic file, then raw key file.
///
/// Unlike `certify`, which only *names* the hotkey, announcing has to sign with
/// it, so a public-only `BASE_MINER_HOTKEY` is not enough here.
fn resolve_miner_hotkey_secret() -> Result<[u8; KEY_LEN], String> {
    match keystore::resolve_keypair_from_env("BASE_MINER") {
        Ok(Some(kp)) => Ok(*kp.expose_mini_secret()),
        Ok(None) => Err(
            "no miner hotkey configured: set BASE_MINER_WALLET (Bittensor wallet), \
             BASE_MINER_MNEMONIC_FILE, or BASE_MINER_SK_FILE"
                .to_owned(),
        ),
        Err(e) => Err(format!("miner hotkey resolution failed: {e}")),
    }
}

/// Load or generate a CVM-local receipt mini-secret on the host; return public hex.
///
/// Private bytes are written mode 0600 and never printed.
fn provision_receipt_key(
    path: &std::path::Path,
    known_public_hex: Option<&str>,
) -> Result<(String, [u8; KEY_LEN]), String> {
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
    Ok((derived_hex, secret))
}

/// Gather the values the CVM receives as Phala encrypted secrets.
///
/// Dry-runs leave them empty: only the variable *names* are measured, so the
/// compose-hash an owner signs is identical with or without them.
fn collect_deploy_secrets(
    receipt_secret: &[u8; KEY_LEN],
    launch_token_file: Option<&std::path::Path>,
    miner_hotkey_hex: Option<&str>,
) -> Result<DeploySecrets, String> {
    let miner_hotkey_hex = match miner_hotkey_hex {
        Some(h) => {
            let h = h.trim().to_ascii_lowercase();
            parse_hotkey_hex(&h).map_err(|e| e.to_string())?;
            h
        }
        None => String::new(),
    };
    Ok(DeploySecrets {
        receipt_sk_hex: hex::encode(receipt_secret),
        launch_token: launch_token_file
            .map(read_launch_token)
            .transpose()?
            .unwrap_or_default(),
        miner_hotkey_hex,
    })
}

/// Load or generate the raw launch token on the host; return its SHA-256 hex.
///
/// Token bytes are written mode 0600 and never printed: the compose measures
/// only the hash, so the file is the sole copy of the bearer credential that
/// lets the operator ask their own CVM for a quote.
fn provision_launch_token(path: &std::path::Path) -> Result<String, String> {
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let token = if path.is_file() {
        read_launch_token(path)?
    } else {
        let token = hex::encode(generate_mini_secret());
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true).mode(0o600);
        let mut f = opts.open(path).map_err(|e| e.to_string())?;
        f.write_all(token.as_bytes()).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        let mut perms = fs::metadata(path).map_err(|e| e.to_string())?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(path, perms).map_err(|e| e.to_string())?;
        token
    };
    Ok(launch_token_hash_hex(&token))
}

/// Read a launch token file; trailing newline from `echo`/editors is not part
/// of the credential the CVM hashes.
fn read_launch_token(path: &std::path::Path) -> Result<String, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("launch token {}: {e}", path.display()))?;
    let token = raw.trim_end().to_owned();
    if token.is_empty() {
        return Err(format!("launch token {} is empty", path.display()));
    }
    Ok(token)
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
