//! validator — independent weight recomputation process.
//!
//! Listens for `/healthz`, `/readyz`, `/metrics`. Epoch clock from chain tip.
//! Coordination client talks only to allowlisted gateway paths.
//! Bundle fetch/recompute/compare (task 29) and verified-bundle mirror/peer fetch
//! (task 30) live in the library; CRV4 submit and dissent are later tasks.
//!
//! Attestation (`/v1/attest/*`) is always mounted. Measurements allowlist is
//! loaded from `GBASE_TRUST_ROOT_DIR` (default `./config` then `/etc/gbase/config`)
//! so fixture/live certify can return `Verified` against the owner-signed root.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use chain::FakeChain;
use config::{keys, load, Role};
use trustroot::{load_config_dir, MeasurementsBody};
use validator::{
    db_ready_from_fn, db_ready_ok, spawn_validator, AttestState, RegistrationStub, SyncChain,
    ValidatorRuntime,
};

#[tokio::main]
async fn main() -> ExitCode {
    if let Err(e) = telemetry::init_tracing() {
        // Subscriber may already be set; continue with best-effort logs.
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

    let listen = std::env::var("GBASE_LISTEN")
        .ok()
        .and_then(|s| s.parse::<SocketAddr>().ok())
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 8080)));

    // Skeleton chain: FakeChain until live SDK client is wired (task 13 follow-up).
    let chain = Arc::new(SyncChain::new(FakeChain::with_defaults()));

    let db_check = match resolve_database_url(&cfg) {
        Ok(Some(url)) => match db::connect(&url).await {
            Ok(pool) => {
                if let Err(e) = db::migrate(&pool).await {
                    tracing::warn!(error = %e, "migrate skipped/failed");
                }
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

    let netuid = cfg.netuid.get();
    let attest = match build_attest_state(netuid, cfg.rotation_epochs) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("attest allowlist: {e}");
            return ExitCode::from(2);
        }
    };

    let runtime = ValidatorRuntime {
        epoch_length: cfg.epoch_length,
        listen_addr: listen,
        gateway_endpoint: cfg.gateway_endpoint.clone(),
        registration: RegistrationStub::new(),
        attest: Some(attest),
        min_peer_sample: cfg.min_peer_sample,
        min_share_mass_bps: cfg.min_share_mass_bps,
        ..ValidatorRuntime::default()
    };

    let running = match spawn_validator(runtime, chain, db_check, vec![]).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("validator failed to start: {e}");
            return ExitCode::from(1);
        }
    };

    tracing::info!(addr = %running.addr, netuid, "validator ready");

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

/// Resolve trust-root directory: env override, then `./config`, then `/etc/gbase/config`.
fn trust_root_dir() -> PathBuf {
    if let Ok(p) = std::env::var("GBASE_TRUST_ROOT_DIR") {
        let pb = PathBuf::from(p);
        if !pb.as_os_str().is_empty() {
            return pb;
        }
    }
    let cwd = PathBuf::from("config");
    if cwd.join("measurements.toml").is_file() {
        return cwd;
    }
    PathBuf::from("/etc/gbase/config")
}

/// Load owner-signed measurements and build AttestState.
///
/// `GBASE_ATTEST_VERIFIER=pcs_timeout` forces Parked (PCS outage) for negative QA.
/// Default is the happy-path mock verifier (fixture/live quotes still need allowlist match).
fn build_attest_state(netuid: u16, rotation_epochs: u32) -> Result<AttestState, String> {
    let dir = trust_root_dir();
    let measurements = load_measurements_allowlist(&dir, rotation_epochs)?;
    let hotkey = validator_hotkey_from_env();
    let mode = std::env::var("GBASE_ATTEST_VERIFIER").unwrap_or_else(|_| "ok".to_owned());
    let state = match mode.as_str() {
        "pcs_timeout" | "park" | "parked" => {
            tracing::warn!(%mode, "attest verifier = PCS timeout (Parked path)");
            AttestState::with_pcs_timeout(measurements, hotkey, netuid)
        }
        _ => AttestState::with_ok_verifier(measurements, hotkey, netuid),
    };
    Ok(state)
}

fn load_measurements_allowlist(
    dir: &Path,
    rotation_epochs: u32,
) -> Result<MeasurementsBody, String> {
    if !dir.is_dir() {
        return Err(format!(
            "trust root dir missing: {} (set GBASE_TRUST_ROOT_DIR)",
            dir.display()
        ));
    }
    // epoch=0 accepts introduced_epoch=0 roots (committed config).
    let (_ch, ms) = load_config_dir(dir, 0, rotation_epochs)
        .map_err(|e| format!("load_config_dir {}: {e}", dir.display()))?;
    let primary = ms
        .primary()
        .map_err(|e| format!("measurements primary: {e}"))?;
    let n = primary.body.entries.len();
    if n == 0 {
        tracing::warn!(
            dir = %dir.display(),
            "measurements allowlist empty — attest submit fail-closed"
        );
    } else {
        tracing::info!(
            dir = %dir.display(),
            entries = n,
            version = primary.version,
            "loaded measurements allowlist"
        );
    }
    Ok(primary.body.clone())
}

/// Optional 32-byte validator hotkey for D10 report_data binding (hex or zeros).
fn validator_hotkey_from_env() -> [u8; 32] {
    let Ok(hex_str) = std::env::var("GBASE_VALIDATOR_HOTKEY_HEX") else {
        return [0u8; 32];
    };
    let hex_str = hex_str.trim();
    if hex_str.len() != 64 {
        tracing::warn!("GBASE_VALIDATOR_HOTKEY_HEX length != 64; using zeros");
        return [0u8; 32];
    }
    let mut out = [0u8; 32];
    match hex::decode(hex_str) {
        Ok(bytes) if bytes.len() == 32 => {
            out.copy_from_slice(&bytes);
            out
        }
        _ => {
            tracing::warn!("GBASE_VALIDATOR_HOTKEY_HEX decode failed; using zeros");
            [0u8; 32]
        }
    }
}
