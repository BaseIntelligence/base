//! validator — independent weight recomputation process.
//!
//! Listens for `/healthz`, `/readyz`, `/metrics`. Epoch clock from chain tip.
//! Coordination client talks only to allowlisted gateway paths.
//! Bundle fetch/recompute/compare (task 29) and verified-bundle mirror/peer fetch
//! (task 30) live in the library; CRV4 submit and dissent are later tasks.
//!
//! Attestation (`/v1/attest/*`) is always mounted. Measurements allowlist is
//! loaded from `BASE_TRUST_ROOT_DIR` (default `./config` then `/etc/base/config`)
//! so fixture/live certify can return `Verified` against the owner-signed root.
//!
//! F3 continuous path: after start, a background loop probes gateway
//! `/v1/weights/latest`, fetches the sealed bundle, runs `compare_bundle`, and
//! logs a `Match epoch=...` line on success.
//!
//! Chain backend: always [`chain_live::LiveChainClient`] against
//! `BASE_CHAIN_ENDPOINT`. There is no in-memory fake; a validator that cannot
//! reach the chain exits rather than recomputing against invented state.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use bundle::LocalTrustRoot;
use chain::ChainClient;
use config::{keys, load, Role};
use trustroot::{load_config_dir, measurements_digest};
use validator::{
    db_ready_from_fn, db_ready_ok, spawn_coordination_loop, spawn_validator, AttestState,
    RegistrationStub, SyncChain, ValidatorRuntime,
};

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() -> ExitCode {
    if let Err(e) = telemetry::init_tracing() {
        eprintln!("tracing init: {e}");
    }

    let cfg = match load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("validator config error: {e}");
            eprintln!(
                "required: {}=validator {}=<u16> and database url or file",
                keys::ROLE,
                keys::NETUID
            );
            return ExitCode::from(2);
        }
    };

    if cfg.role != Role::Validator {
        eprintln!(
            "validator requires {}=validator (got {})",
            keys::ROLE,
            cfg.role
        );
        return ExitCode::from(2);
    }

    let listen = std::env::var("BASE_LISTEN")
        .ok()
        .and_then(|s| s.parse::<SocketAddr>().ok())
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 8080)));

    let netuid = cfg.netuid.get();
    let (attest, trust) = match load_attest_and_trust(netuid, cfg.rotation_epochs) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("trust root: {e}");
            return ExitCode::from(2);
        }
    };

    tracing::info!(
        endpoint = %cfg.chain_endpoint,
        netuid,
        "validator connecting to live chain"
    );
    let mut client = match chain_live::LiveChainClient::connect(&cfg.chain_endpoint) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("validator: live chain connect failed: {e}");
            return ExitCode::from(1);
        }
    };
    client.set_netuid(netuid);

    // Weight submission needs the sr25519 secret, not just the public hotkey.
    match keystore::resolve_keypair_from_env("BASE_VALIDATOR") {
        Ok(Some(kp)) => {
            client.set_signing_key(*kp.expose_mini_secret());
            tracing::info!(
                hotkey = %kp.ss58_address(),
                "validator signing key loaded; weight submission enabled"
            );
        }
        Ok(None) => tracing::warn!(
            "no validator signing key configured; weight submission will fail closed \
             (set BASE_VALIDATOR_WALLET or BASE_VALIDATOR_MNEMONIC_FILE)"
        ),
        Err(e) => {
            eprintln!("validator: signing key resolution failed: {e}");
            return ExitCode::from(2);
        }
    }

    // Fail fast on a misconfigured netuid rather than looping on empty reads.
    match client.subnet_owner_hotkey(netuid) {
        Ok(owner) => tracing::info!(
            netuid,
            owner = %hex::encode(&owner),
            "live chain reachable; subnet owner resolved"
        ),
        Err(e) => {
            eprintln!("validator: chain preflight failed for netuid {netuid}: {e}");
            return ExitCode::from(1);
        }
    }

    let chain = Arc::new(SyncChain::new(client));
    run_validator_main(cfg, listen, netuid, attest, trust, chain).await
}

/// Generic main body: database, runtime, spawn, coordination loop, ctrl-c, shutdown.
async fn run_validator_main<C: ChainClient + Send + 'static>(
    cfg: config::Config,
    listen: SocketAddr,
    netuid: u16,
    attest: AttestState,
    trust: LocalTrustRoot,
    chain: Arc<SyncChain<C>>,
) -> ExitCode {
    let db_check = match resolve_database_url(&cfg) {
        Ok(Some(url)) => match db::connect(&url).await {
            Ok(pool) => {
                if let Err(e) = db::migrate(&pool).await {
                    tracing::warn!(error = %e, "migrate skipped/failed");
                }
                // Durable attestation record (receipt keys included); without a
                // pool the credit book dies with the process.
                attest.attach_pool(pool.clone()).await;
                db_ready_from_fn(move || {
                    let pool = pool.clone();
                    async move {
                        db::count_miners(&pool)
                            .await
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    }
                })
            }
            Err(e) => {
                eprintln!("database connect failed: {e}");
                return ExitCode::from(1);
            }
        },
        Ok(None) => db_ready_ok(),
        Err(e) => {
            eprintln!("database url: {e}");
            return ExitCode::from(2);
        }
    };

    // Pin attestations to the chain epoch: the challenge service joins on that
    // epoch, so a miner-declared one would let it bank credit for future epochs.
    attest
        .attach_chain(Arc::clone(&chain) as validator::SharedChain)
        .await;

    let hotkey = validator_hotkey_from_env();
    let registration = match validator::registration::resolve_on_chain(chain.as_ref(), &hotkey) {
        Ok(status) => {
            tracing::info!(
                registered = status.registered,
                uid = ?status.uid,
                "resolved on-chain registration"
            );
            RegistrationStub::from_status(status)
        }
        Err(e) => {
            tracing::warn!(error = %e, "registration lookup failed; reporting unregistered");
            RegistrationStub::new()
        }
    };

    let runtime = ValidatorRuntime {
        epoch_length: cfg.epoch_length,
        listen_addr: listen,
        gateway_endpoint: cfg.gateway_endpoint.clone(),
        registration,
        attest: Some(attest),
        min_peer_sample: cfg.min_peer_sample,
        min_share_mass_bps: cfg.min_share_mass_bps,
        ..ValidatorRuntime::default()
    };

    let running = match spawn_validator(runtime, Arc::clone(&chain), db_check, vec![]).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("validator failed to start: {e}");
            return ExitCode::from(1);
        }
    };

    tracing::info!(addr = %running.addr, netuid, "validator ready");

    let interval_secs: u64 = std::env::var("BASE_COORDINATION_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
        .max(1);
    let _coord_loop = spawn_coordination_loop(
        Arc::clone(&running.coordination),
        Arc::clone(&chain),
        trust,
        Duration::from_secs(interval_secs),
    );

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("ctrl-c received, shutting down");
        }
    }

    if let Err(e) = running.shutdown().await {
        tracing::error!(error = %e, "shutdown error");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn resolve_database_url(cfg: &config::Config) -> Result<Option<String>, String> {
    if let Some(url) = cfg.database_url.as_ref() {
        return Ok(Some(url.clone()));
    }
    if let Some(path) = cfg.database_url_file.as_ref() {
        let s =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let t = s.trim().to_owned();
        if t.is_empty() {
            return Err("database_url_file is empty".into());
        }
        return Ok(Some(t));
    }
    Ok(None)
}

/// Resolve trust-root directory: env override, then `./config`, then `/etc/base/config`.
fn trust_root_dir() -> PathBuf {
    if let Ok(p) = std::env::var("BASE_TRUST_ROOT_DIR") {
        let pb = PathBuf::from(p);
        if !pb.as_os_str().is_empty() {
            return pb;
        }
    }
    let cwd = PathBuf::from("config");
    if cwd.join("measurements.toml").is_file() {
        return cwd;
    }
    PathBuf::from("/etc/base/config")
}

/// Load challenges + measurements; build `AttestState` and `LocalTrustRoot`.
/// Chain creation is handled separately in `main()` based on `BASE_CHAIN_BACKEND`.
fn load_attest_and_trust(
    netuid: u16,
    rotation_epochs: u32,
) -> Result<(AttestState, LocalTrustRoot), String> {
    let dir = trust_root_dir();
    if !dir.is_dir() {
        return Err(format!(
            "trust root dir missing: {} (set BASE_TRUST_ROOT_DIR)",
            dir.display()
        ));
    }
    let (ch, ms) = load_config_dir(&dir, 0, rotation_epochs)
        .map_err(|e| format!("load_config_dir {}: {e}", dir.display()))?;
    let primary_ch = ch
        .primary()
        .map_err(|e| format!("challenges primary: {e}"))?;
    let primary_ms = ms
        .primary()
        .map_err(|e| format!("measurements primary: {e}"))?;
    let measurements = primary_ms.body.clone();
    let mdigest = measurements_digest(&measurements);
    let trust = LocalTrustRoot {
        challenges: primary_ch.body.clone(),
        measurements_digest: mdigest,
    };
    tracing::info!(
        dir = %dir.display(),
        challenges = primary_ch.body.challenges.len(),
        measurements = measurements.entries.len(),
        "loaded validator trust root"
    );

    let hotkey = validator_hotkey_from_env();
    // Real Intel DCAP when built with `--features dcap`; see BASE_ATTEST_VERIFIER.
    let attest =
        AttestState::with_verifier(measurements, hotkey, netuid, validator::verifier_from_env());

    Ok((attest, trust))
}

/// Resolve the validator hotkey used for D10 `report_data` binding.
///
/// Order: Bittensor wallet (`BASE_WALLET_NAME` / `BASE_WALLET_HOTKEY`), then a
/// mnemonic file, then raw hex. Zeros only when nothing is configured.
fn validator_hotkey_from_env() -> [u8; 32] {
    match keystore::resolve_public_key_from_env("BASE_VALIDATOR") {
        Ok(Some(pk)) => pk,
        Ok(None) => [0_u8; 32],
        Err(e) => {
            tracing::warn!(error = %e, "validator hotkey resolution failed; using zeros");
            [0_u8; 32]
        }
    }
}
