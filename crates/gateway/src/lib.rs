//! Master-only gbase gateway (D3) with backend registry, reverse proxy, and
//! signed raw-weight ingress and epoch bundle seal/serve (tasks 24–27).
//!
//! Startup order is intentional and tested:
//! 1. Resolve on-chain [`ChainClient::subnet_owner_hotkey`].
//! 2. Compare to the configured gateway hotkey.
//! 3. On mismatch: structured fatal log + [`GatewayError::MasterMismatch`]
//!    (process exit code [`EXIT_MASTER_MISMATCH`]) **before any listener bind**.
//! 4. On match: install telemetry, mount health + registry + weights + bundles +
//!    challenge proxy, and serve. This process is the sole TLS owner (D20); ACME is task 42.
//!
//! Backend registry is operational routing only (D18) — never signing keys.
//! Raw-weight leaves are verified against the local owner-signed trust root.
//! Epoch seal uses local trust-root shares (D23) and fails closed on incomplete sets (D24).
#![forbid(unsafe_code)]

mod api;
mod app;
mod proxy;
mod sealer;
mod tls;
mod weights;

use std::fmt;
use std::net::SocketAddr;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

use axum::Router;
use chain::{ChainClient, ChainError};
use config::{keys as cfg_keys, Config, Role};
use telemetry::{init_metrics, init_tracing, TelemetryError};
use thiserror::Error;
use tokio::net::TcpListener;

pub use api::GatewayState;
pub use gateway_registry::{
    Backend, BackendView, CreateBackend, Registry, RegistryConfig, RegistryError, DEFAULT_COOLDOWN,
    DEFAULT_FAILURE_THRESHOLD,
};
pub use sealer::{
    admin_seal_router, bundle_router, load_gateway_secret, seal_epoch, BundleStore,
    MemoryBundleStore, SealError, SealParams, SealRequest, SealResponse, SharedBundleStore,
};
pub use tls::TlsConfig;
pub use trustroot::{ChallengeEntry, ChallengesBody, ParticipantPolicy, BPS_DENOM};
pub use weights::{
    accept_raw_weight, weights_router, MemoryRawWeightStore, RawWeightAccepted, RawWeightRequest,
    RawWeightRow, RawWeightStore, ScoreOrAbsenceWire, SharedWeightStore, StoreError,
};

/// Process exit code when configured hotkey ≠ on-chain `SubnetOwnerHotkey`.
pub const EXIT_MASTER_MISMATCH: u8 = 2;

/// Default listen address when `GBASE_GATEWAY_LISTEN` is unset.
pub const DEFAULT_LISTEN: &str = "0.0.0.0:8080";

/// Env keys specific to the gateway binary (in addition to `GBASE_*` from config).
pub mod keys {
    /// Bind address (`host:port`), e.g. `0.0.0.0:8080`.
    pub const LISTEN: &str = "GBASE_GATEWAY_LISTEN";
    /// Gateway hotkey as 64 hex chars (32 raw bytes). Must equal on-chain owner.
    pub const HOTKEY: &str = "GBASE_GATEWAY_HOTKEY";
    /// Consecutive upstream failures before passive ejection (default 3).
    pub const FAIL_THRESHOLD: &str = "GBASE_GATEWAY_FAIL_THRESHOLD";
    /// Ejection cooldown seconds before re-admission (default 30).
    pub const COOLDOWN_SECS: &str = "GBASE_GATEWAY_COOLDOWN_SECS";
    /// Owner-signed trust root directory (`challenges.toml` + `measurements.toml`).
    pub const TRUST_ROOT_DIR: &str = "GBASE_TRUST_ROOT_DIR";
    /// Comma-separated 64-hex hotkeys for `FakeChain` metagraph (UID order).
    pub const FAKE_METAGRAPH_HOTKEYS: &str = "GBASE_FAKE_METAGRAPH_HOTKEYS";
    /// Gateway bundle-signing mini-secret (64 hex).
    pub const GATEWAY_SK: &str = "GBASE_GATEWAY_SK";
    /// Path to gateway mini-secret (32 raw bytes or 64 hex).
    pub const GATEWAY_SK_FILE: &str = "GBASE_GATEWAY_SK_FILE";

    pub use crate::tls::keys as tls;
}

/// Runtime knobs required to start the gateway server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayConfig {
    /// TCP bind address (bound only after master check passes).
    pub listen: SocketAddr,
    /// Subnet netuid used for `subnet_owner_hotkey`.
    pub netuid: u16,
    /// Configured gateway hotkey (32-byte sr25519 public key).
    pub hotkey: [u8; 32],
    /// Passive ejection / re-admission knobs.
    pub registry: RegistryConfig,
    /// Sole TLS owner config (D20); real ACME in task 42.
    pub tls: TlsConfig,
}

impl GatewayConfig {
    /// Build from a validated [`Config`] plus gateway-specific listen/hotkey.
    ///
    /// # Errors
    ///
    /// Role is not gateway, or listen/hotkey parse failures.
    pub fn from_base(
        base: &Config,
        listen: SocketAddr,
        hotkey: [u8; 32],
        registry: RegistryConfig,
        tls: TlsConfig,
    ) -> Result<Self, GatewayError> {
        if base.role != Role::Gateway {
            return Err(GatewayError::Config(format!(
                "role must be gateway, got {}",
                base.role
            )));
        }
        Ok(Self {
            listen,
            netuid: base.netuid.get(),
            hotkey,
            registry,
            tls,
        })
    }

    /// Load layered config + gateway env knobs.
    ///
    /// # Errors
    ///
    /// Config validation, missing hotkey, or bad listen/hotkey encoding.
    pub fn from_env() -> Result<Self, GatewayError> {
        let base = config::load().map_err(|e| GatewayError::Config(e.to_string()))?;
        let listen_raw = std::env::var(keys::LISTEN).unwrap_or_else(|_| DEFAULT_LISTEN.to_owned());
        let listen = SocketAddr::from_str(&listen_raw).map_err(|e| {
            GatewayError::Config(format!("invalid {} `{listen_raw}`: {e}", keys::LISTEN))
        })?;
        let hotkey_raw = std::env::var(keys::HOTKEY)
            .map_err(|_| GatewayError::Config(format!("missing required env {}", keys::HOTKEY)))?;
        let hotkey = parse_hotkey_hex(&hotkey_raw)?;
        let registry = registry_config_from_env()?;
        let tls = TlsConfig::from_env();
        Self::from_base(&base, listen, hotkey, registry, tls)
    }
}

fn registry_config_from_env() -> Result<RegistryConfig, GatewayError> {
    let mut cfg = RegistryConfig::default();
    if let Ok(raw) = std::env::var(keys::FAIL_THRESHOLD) {
        cfg.failure_threshold = raw.parse().map_err(|e| {
            GatewayError::Config(format!("invalid {} `{raw}`: {e}", keys::FAIL_THRESHOLD))
        })?;
        if cfg.failure_threshold == 0 {
            return Err(GatewayError::Config(format!(
                "{} must be >= 1",
                keys::FAIL_THRESHOLD
            )));
        }
    }
    if let Ok(raw) = std::env::var(keys::COOLDOWN_SECS) {
        let secs: u64 = raw.parse().map_err(|e| {
            GatewayError::Config(format!("invalid {} `{raw}`: {e}", keys::COOLDOWN_SECS))
        })?;
        cfg.cooldown = std::time::Duration::from_secs(secs);
    }
    Ok(cfg)
}

/// Parse a 32-byte hotkey from lowercase/uppercase hex (optional `0x` prefix).
///
/// # Errors
///
/// Wrong length or non-hex input.
pub fn parse_hotkey_hex(s: &str) -> Result<[u8; 32], GatewayError> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let bytes = hex::decode(s)
        .map_err(|e| GatewayError::Config(format!("invalid {} hex: {e}", keys::HOTKEY)))?;
    let arr: [u8; 32] = bytes.try_into().map_err(|v: Vec<u8>| {
        GatewayError::Config(format!(
            "{} must be 32 bytes (64 hex chars), got {} bytes",
            keys::HOTKEY,
            v.len()
        ))
    })?;
    Ok(arr)
}

/// Format hotkey bytes as lowercase hex (no `0x`) for structured logs.
#[must_use]
pub fn hotkey_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Gateway failures (config, master check, bind, telemetry, client).
#[derive(Debug, Error)]
pub enum GatewayError {
    /// Configured hotkey is not the on-chain subnet owner (D3).
    #[error("master-only check failed: configured hotkey is not SubnetOwnerHotkey")]
    MasterMismatch {
        /// Configured gateway hotkey (hex).
        configured: String,
        /// On-chain owner hotkey (hex).
        owner: String,
        /// Netuid queried.
        netuid: u16,
    },
    /// Chain read failure while resolving owner.
    #[error("chain error: {0}")]
    Chain(#[from] ChainError),
    /// Operator / env configuration problem.
    #[error("config error: {0}")]
    Config(String),
    /// TCP bind failed (only reached after master check).
    #[error("bind error: {0}")]
    Bind(String),
    /// Telemetry install failure.
    #[error("telemetry error: {0}")]
    Telemetry(#[from] TelemetryError),
    /// Outbound HTTP client build failure.
    #[error("http client error: {0}")]
    HttpClient(String),
}

impl GatewayError {
    /// Map to a process [`ExitCode`] (`2` for master mismatch, `1` otherwise).
    #[must_use]
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::MasterMismatch { .. } => ExitCode::from(EXIT_MASTER_MISMATCH),
            _ => ExitCode::from(1),
        }
    }

    /// Emit a structured fatal log line suitable for JSON tracing collectors.
    pub fn log_fatal(&self) {
        match self {
            Self::MasterMismatch {
                configured,
                owner,
                netuid,
            } => {
                tracing::error!(
                    event = "master_only_mismatch",
                    fatal = true,
                    netuid = *netuid,
                    configured_hotkey = %configured,
                    owner_hotkey = %owner,
                    exit_code = EXIT_MASTER_MISMATCH,
                    "gateway hotkey is not on-chain SubnetOwnerHotkey; refusing to bind"
                );
            }
            other => {
                tracing::error!(
                    event = "gateway_fatal",
                    fatal = true,
                    error = %other,
                    "gateway startup failed"
                );
            }
        }
    }
}

/// Resolve `SubnetOwnerHotkey` and require equality with `configured_hotkey`.
///
/// Does **not** bind any socket. Call this before [`TcpListener::bind`].
///
/// # Errors
///
/// [`GatewayError::MasterMismatch`] or chain transport errors.
pub fn assert_master_only(
    chain: &dyn ChainClient,
    netuid: u16,
    configured_hotkey: &[u8],
) -> Result<(), GatewayError> {
    let owner = chain.subnet_owner_hotkey(netuid)?;
    if owner.as_slice() != configured_hotkey {
        return Err(GatewayError::MasterMismatch {
            configured: hotkey_hex(configured_hotkey),
            owner: hotkey_hex(&owner),
            netuid,
        });
    }
    Ok(())
}

/// Resolve trust-root directory: env, then `./config`, then `/etc/gbase/config`.
#[must_use]
pub fn trust_root_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var(keys::TRUST_ROOT_DIR) {
        let pb = std::path::PathBuf::from(p);
        if !pb.as_os_str().is_empty() {
            return pb;
        }
    }
    let cwd = std::path::PathBuf::from("config");
    if cwd.join("challenges.toml").is_file() {
        return cwd;
    }
    std::path::PathBuf::from("/etc/gbase/config")
}

/// Load owner-signed challenges + measurements digest from the trust-root dir.
///
/// # Errors
///
/// Missing dir or verify failure.
pub fn load_production_trust_root(
    rotation_epochs: u32,
) -> Result<(Arc<trustroot::ChallengesBody>, [u8; 32]), GatewayError> {
    let dir = trust_root_dir();
    if !dir.is_dir() {
        return Err(GatewayError::Config(format!(
            "trust root dir missing: {} (set {})",
            dir.display(),
            keys::TRUST_ROOT_DIR
        )));
    }
    let (ch, ms) = trustroot::load_config_dir(&dir, 0, rotation_epochs)
        .map_err(|e| GatewayError::Config(format!("load_config_dir {}: {e}", dir.display())))?;
    let primary_ch = ch
        .primary()
        .map_err(|e| GatewayError::Config(format!("challenges primary: {e}")))?;
    let primary_ms = ms
        .primary()
        .map_err(|e| GatewayError::Config(format!("measurements primary: {e}")))?;
    let digest = trustroot::measurements_digest(&primary_ms.body);
    tracing::info!(
        dir = %dir.display(),
        challenges = primary_ch.body.challenges.len(),
        measurements = primary_ms.body.entries.len(),
        "loaded gateway trust root"
    );
    Ok((Arc::new(primary_ch.body.clone()), digest))
}

/// Parse `GBASE_FAKE_METAGRAPH_HOTKEYS` (comma-separated 64-hex). Empty → owner only.
///
/// # Errors
///
/// Bad hex encoding.
pub fn parse_fake_metagraph_hotkeys(owner: &[u8; 32]) -> Result<Vec<Vec<u8>>, GatewayError> {
    let Ok(raw) = std::env::var(keys::FAKE_METAGRAPH_HOTKEYS) else {
        return Ok(vec![owner.to_vec()]);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(vec![owner.to_vec()]);
    }
    let mut out = Vec::new();
    for part in raw.split(',') {
        let h = parse_hotkey_hex(part.trim())?;
        out.push(h.to_vec());
    }
    if out.is_empty() {
        out.push(owner.to_vec());
    }
    Ok(out)
}

/// Build the full application router (health + registry + weights + proxy) without binding.
///
/// Production path loads owner-signed challenges from [`trust_root_dir`].
///
/// # Errors
///
/// Telemetry, trust-root load, or HTTP client construction failures.
pub fn build_app(
    metrics: metrics_exporter_prometheus::PrometheusHandle,
    registry: Arc<Registry>,
    tls: &TlsConfig,
) -> Result<Router, GatewayError> {
    // Prefer production trust root; fall back to empty only when dir missing in tests.
    let (challenges, mdigest) = match load_production_trust_root(3) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "gateway trust root unavailable; empty challenges");
            (
                Arc::new(trustroot::ChallengesBody::default()),
                trustroot::measurements_digest(&trustroot::MeasurementsBody::default()),
            )
        }
    };
    let owner = std::env::var(keys::HOTKEY)
        .ok()
        .and_then(|s| parse_hotkey_hex(&s).ok())
        .unwrap_or([0u8; 32]);
    let hotkeys = parse_fake_metagraph_hotkeys(&owner).unwrap_or_else(|_| vec![owner.to_vec()]);
    let netuid = std::env::var("GBASE_NETUID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1u16);
    let state = GatewayState::with_parts_seal(
        registry,
        challenges,
        Arc::new(MemoryRawWeightStore::new()),
        Arc::new(MemoryBundleStore::new()),
        mdigest,
        netuid,
        owner,
        hotkeys,
        10_000,
    )
    .map_err(GatewayError::HttpClient)?;
    app::build_router(metrics, state, tls)
}

/// Like [`build_app`] with injected trust root and weight store (tests / hydrate).
///
/// # Errors
///
/// Telemetry or HTTP client construction failures.
pub fn build_app_with(
    metrics: metrics_exporter_prometheus::PrometheusHandle,
    registry: Arc<Registry>,
    tls: &TlsConfig,
    challenges: Arc<trustroot::ChallengesBody>,
    weights: SharedWeightStore,
) -> Result<Router, GatewayError> {
    build_app_with_bundles(
        metrics,
        registry,
        tls,
        challenges,
        weights,
        Arc::new(MemoryBundleStore::new()),
    )
}

/// Full injection for seal/serve tests: trust root, weights, and bundle store.
///
/// # Errors
///
/// Telemetry or HTTP client construction failures.
pub fn build_app_with_bundles(
    metrics: metrics_exporter_prometheus::PrometheusHandle,
    registry: Arc<Registry>,
    tls: &TlsConfig,
    challenges: Arc<trustroot::ChallengesBody>,
    weights: SharedWeightStore,
    bundles: SharedBundleStore,
) -> Result<Router, GatewayError> {
    let state = GatewayState::with_parts(registry, challenges, weights, bundles)
        .map_err(GatewayError::HttpClient)?;
    app::build_router(metrics, state, tls)
}

/// Master check → telemetry → bind → axum serve until `shutdown` completes.
///
/// On master mismatch this returns **without** opening a listener (port stays closed).
///
/// # Errors
///
/// Master mismatch, chain, telemetry, or bind/serve failures.
pub async fn run_until<F>(
    config: GatewayConfig,
    chain: &dyn ChainClient,
    shutdown: F,
) -> Result<(), GatewayError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let registry = Registry::shared(config.registry.clone());
    run_until_with_registry(config, chain, registry, shutdown).await
}

/// Like [`run_until`] but injects a pre-built registry (tests / DB hydrate).
///
/// # Errors
///
/// Same as [`run_until`].
pub async fn run_until_with_registry<F>(
    config: GatewayConfig,
    chain: &dyn ChainClient,
    registry: Arc<Registry>,
    shutdown: F,
) -> Result<(), GatewayError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    // CRITICAL: master-only before any bind (D3).
    assert_master_only(chain, config.netuid, &config.hotkey)?;

    let _ = init_tracing();
    let metrics = init_metrics()?;
    // Prefer registry knobs from config when the shared handle was default-built.
    let _ = &config.registry;
    let app = build_app(metrics, registry, &config.tls)?;

    let listener = TcpListener::bind(config.listen)
        .await
        .map_err(|e| GatewayError::Bind(format!("{}: {e}", config.listen)))?;

    let bound = listener
        .local_addr()
        .map_err(|e| GatewayError::Bind(e.to_string()))?;
    tracing::info!(
        event = "gateway_listening",
        %bound,
        netuid = config.netuid,
        hotkey = %hotkey_hex(&config.hotkey),
        tls_mode = config.tls.mode_label(),
        "master-only check passed; serving health + registry + weights + challenge proxy"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| GatewayError::Bind(format!("serve: {e}")))?;
    Ok(())
}

/// Production entry: run until ctrl-c / SIGTERM with a fresh in-memory registry.
///
/// # Errors
///
/// Same as [`run_until`].
pub async fn run(config: GatewayConfig, chain: &dyn ChainClient) -> Result<(), GatewayError> {
    let registry = Registry::shared(config.registry.clone());
    run_until_with_registry(config, chain, registry, shutdown_signal()).await
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::warn!(error = %e, "ctrl_c handler failed");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "sigterm handler unavailable");
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

impl fmt::Display for GatewayConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GatewayConfig {{ listen: {}, netuid: {}, hotkey: {}, tls: {} }}",
            self.listen,
            self.netuid,
            hotkey_hex(&self.hotkey),
            self.tls.mode_label()
        )
    }
}

/// Re-export config key names operators may need alongside gateway keys.
pub mod config_keys {
    pub use super::cfg_keys::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use chain::{FakeChain, FakeChainConfig};
    use std::time::Duration;
    use tokio::sync::oneshot;

    fn owner_hotkey() -> [u8; 32] {
        [0xA1; 32]
    }

    fn other_hotkey() -> [u8; 32] {
        [0xB2; 32]
    }

    fn cfg(listen: SocketAddr, hotkey: [u8; 32]) -> GatewayConfig {
        GatewayConfig {
            listen,
            netuid: 1,
            hotkey,
            registry: RegistryConfig::default(),
            tls: TlsConfig::default(),
        }
    }

    #[test]
    fn s1_assert_master_mismatch_no_bind_needed() {
        let chain = FakeChain::with_defaults();
        let err = assert_master_only(&chain, 1, &other_hotkey()).expect_err("mismatch");
        match &err {
            GatewayError::MasterMismatch {
                configured,
                owner,
                netuid,
            } => {
                assert_eq!(*netuid, 1);
                assert_eq!(configured, &hotkey_hex(&other_hotkey()));
                assert_eq!(owner, &hotkey_hex(&owner_hotkey()));
                assert_eq!(err.exit_code(), ExitCode::from(2));
            }
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn s2_assert_master_match_ok() {
        let chain = FakeChain::with_defaults();
        assert_master_only(&chain, 1, &owner_hotkey()).expect("match");
    }

    #[test]
    fn s3_parse_hotkey_hex_roundtrip() {
        let h = owner_hotkey();
        let encoded = hotkey_hex(&h);
        assert_eq!(parse_hotkey_hex(&encoded).unwrap(), h);
        assert_eq!(parse_hotkey_hex(&format!("0x{encoded}")).unwrap(), h);
        assert!(parse_hotkey_hex("dead").is_err());
        assert!(parse_hotkey_hex(&"ab".repeat(31)).is_err()); // 31 bytes
    }

    #[tokio::test]
    async fn s1_mismatch_run_until_exits_2_and_port_refused() {
        let probe = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);

        let listen: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let chain = FakeChain::with_defaults();
        let config = cfg(listen, other_hotkey());

        let err = run_until(config, &chain, async {})
            .await
            .expect_err("must refuse");
        assert!(
            matches!(err, GatewayError::MasterMismatch { .. }),
            "got {err}"
        );
        assert_eq!(err.exit_code(), ExitCode::from(EXIT_MASTER_MISMATCH));

        let connect = tokio::time::timeout(
            Duration::from_millis(500),
            tokio::net::TcpStream::connect(listen),
        )
        .await;
        match connect {
            Ok(Err(e)) => {
                let kind = e.kind();
                assert!(
                    kind == std::io::ErrorKind::ConnectionRefused
                        || kind == std::io::ErrorKind::AddrNotAvailable,
                    "expected connection refused, got {e:?}"
                );
            }
            Ok(Ok(_stream)) => panic!("port should not accept connections after mismatch"),
            Err(elapsed) => panic!("connect timed out unexpectedly: {elapsed:?}"),
        }
    }

    #[tokio::test]
    async fn s2_match_run_until_healthz_200() {
        let reserve = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = reserve.local_addr().unwrap();
        drop(reserve);

        let chain = FakeChain::new(FakeChainConfig {
            owner_hotkey: owner_hotkey().to_vec(),
            ..FakeChainConfig::default()
        });
        let config = cfg(addr, owner_hotkey());

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let server = tokio::task::spawn_local(async move {
                    run_until(config, &chain, async move {
                        let _ = shutdown_rx.await;
                    })
                    .await
                });

                let client = reqwest::Client::builder()
                    .timeout(Duration::from_secs(2))
                    .build()
                    .unwrap();
                let url = format!("http://{addr}/healthz");
                let mut last_err = None;
                let mut ok = false;
                for _ in 0..80 {
                    match client.get(&url).send().await {
                        Ok(resp) if resp.status().as_u16() == 200 => {
                            let body = resp.text().await.unwrap_or_default();
                            assert!(
                                body.contains("ok"),
                                "healthz body should contain ok, got {body:?}"
                            );
                            ok = true;
                            break;
                        }
                        Ok(resp) => last_err = Some(format!("status {}", resp.status())),
                        Err(e) => last_err = Some(e.to_string()),
                    }
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                assert!(ok, "healthz never became ready: {last_err:?}");

                let _ = shutdown_tx.send(());
                let result = server.await.expect("join local server");
                assert!(result.is_ok(), "run_until error: {result:?}");
            })
            .await;
    }

    #[test]
    fn s3_chain_unknown_netuid_does_not_look_like_mismatch() {
        let chain = FakeChain::with_defaults();
        let err = assert_master_only(&chain, 99, &owner_hotkey()).expect_err("bad netuid");
        assert!(matches!(err, GatewayError::Chain(_)), "got {err}");
        assert_eq!(err.exit_code(), ExitCode::from(1));
    }

    #[test]
    fn from_base_rejects_non_gateway_role() {
        use config::{NetUid, DEFAULT_CHAIN_ENDPOINT, DEFAULT_EPOCH_LENGTH};
        let base = Config {
            role: Role::Validator,
            netuid: NetUid::new(1),
            chain_endpoint: DEFAULT_CHAIN_ENDPOINT.to_owned(),
            gateway_endpoint: None,
            database_url: Some("postgres://x".into()),
            database_url_file: None,
            epoch_length: DEFAULT_EPOCH_LENGTH,
            min_share_mass_bps: 5000,
            rotation_epochs: 3,
            min_peer_sample: 1,
            max_collateral_age_secs: 1,
            domain: None,
        };
        let err = GatewayConfig::from_base(
            &base,
            "127.0.0.1:9".parse().unwrap(),
            owner_hotkey(),
            RegistryConfig::default(),
            TlsConfig::default(),
        )
        .expect_err("role");
        assert!(matches!(err, GatewayError::Config(_)));
    }
}
