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

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use bundle::LocalTrustRoot;
use chain::{FakeChain, FakeChainConfig};
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

    let listen = std::env::var("BASE_LISTEN")
        .ok()
        .and_then(|s| s.parse::<SocketAddr>().ok())
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 8080)));

    let netuid = cfg.netuid.get();
    let (attest, trust, chain) = match build_runtime_trust(netuid, cfg.rotation_epochs) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("trust root / chain: {e}");
            return ExitCode::from(2);
        }
    };

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

    let running = match spawn_validator(runtime, Arc::clone(&chain), db_check, vec![]).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("validator failed to start: {e}");
            return ExitCode::from(1);
        }
    };

    tracing::info!(addr = %running.addr, netuid, "validator ready");

    // F3 continuous Match loop (gateway latest → bundle → compare).
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

/// Load challenges + measurements; build `AttestState`, `LocalTrustRoot`, and `FakeChain`
/// aligned with gateway (`BASE_GATEWAY_HOTKEY` / `BASE_FAKE_METAGRAPH_HOTKEYS`).
fn build_runtime_trust(
    netuid: u16,
    rotation_epochs: u32,
) -> Result<(AttestState, LocalTrustRoot, Arc<SyncChain<FakeChain>>), String> {
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
    let mode = std::env::var("BASE_ATTEST_VERIFIER").unwrap_or_else(|_| "ok".to_owned());
    let attest = match mode.as_str() {
        "pcs_timeout" | "park" | "parked" => {
            tracing::warn!(%mode, "attest verifier = PCS timeout (Parked path)");
            AttestState::with_pcs_timeout(measurements, hotkey, netuid)
        }
        _ => AttestState::with_ok_verifier(measurements, hotkey, netuid),
    };

    let chain = Arc::new(SyncChain::new(fake_chain_aligned(netuid)?));
    Ok((attest, trust, chain))
}

fn fake_chain_aligned(netuid: u16) -> Result<FakeChain, String> {
    let owner = parse_hotkey_env("BASE_GATEWAY_HOTKEY")
        .or_else(|| parse_hotkey_env("BASE_VALIDATOR_HOTKEY_HEX"))
        .unwrap_or([0xA1; 32]);
    let hotkeys = parse_fake_metagraph_hotkeys(&owner)?;
    Ok(FakeChain::new(FakeChainConfig {
        netuid,
        owner_hotkey: owner.to_vec(),
        hotkeys,
        current_block: 10_000,
        ..FakeChainConfig::default()
    }))
}

fn parse_fake_metagraph_hotkeys(owner: &[u8; 32]) -> Result<Vec<Vec<u8>>, String> {
    let Ok(raw) = std::env::var("BASE_FAKE_METAGRAPH_HOTKEYS") else {
        return Ok(vec![owner.to_vec()]);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(vec![owner.to_vec()]);
    }
    let mut out = Vec::new();
    for part in raw.split(',') {
        let h = parse_hotkey_hex(part.trim()).map_err(|e| e.clone())?;
        out.push(h.to_vec());
    }
    if out.is_empty() {
        out.push(owner.to_vec());
    }
    Ok(out)
}

fn parse_hotkey_env(name: &str) -> Option<[u8; 32]> {
    let Ok(s) = std::env::var(name) else {
        return None;
    };
    parse_hotkey_hex(s.trim()).ok()
}

fn parse_hotkey_hex(s: &str) -> Result<[u8; 32], String> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let bytes = hex::decode(s).map_err(|e| format!("hotkey hex: {e}"))?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("hotkey must be 32 bytes, got {}", v.len()))
}

/// Optional 32-byte validator hotkey for D10 `report_data` binding (hex or zeros).
fn validator_hotkey_from_env() -> [u8; 32] {
    parse_hotkey_env("BASE_VALIDATOR_HOTKEY_HEX").unwrap_or([0u8; 32])
}
