//! Master-only base gateway (D3) with backend registry, reverse proxy, and
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

use std::process::ExitCode;
use std::sync::Arc;

use axum::Router;
use chain::{ChainClient, ChainError};
use config::keys as cfg_keys;
use telemetry::{init_metrics, init_tracing, TelemetryError};
use thiserror::Error;
use tokio::net::TcpListener;

mod gw_config;

pub use api::{registry_router, GatewayState, SharedChain};
pub use gateway_core::admin_attest::{
    admin_attest_grant_router, AttestGrantRequest, AttestGrantResponse, AttestGrantState,
    ATTEST_GRANT_ROUTE,
};
pub use gateway_core::admin_auth::{
    admin_auth_middleware, AdminAuth, AdminAuthError, ADMIN_TOKEN_ENV, ADMIN_TOKEN_FILE_ENV,
};
pub use gateway_registry::{
    Backend, BackendView, CreateBackend, Registry, RegistryConfig, RegistryError, DEFAULT_COOLDOWN,
    DEFAULT_FAILURE_THRESHOLD,
};
pub use gw_config::{
    hotkey_hex, keys, parse_hotkey_hex, resolve_gateway_hotkey, GatewayConfig, OwnerCheck,
    REQUIRE_OWNER_ENV,
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

/// Raw-weight and sealed-bundle stores injected by the binary.
///
/// Production passes Postgres-backed stores; tests pass the in-memory pair.
pub type Stores = (SharedWeightStore, SharedBundleStore);

/// Process exit code when configured hotkey ≠ on-chain `SubnetOwnerHotkey`.
pub const EXIT_MASTER_MISMATCH: u8 = 2;

/// Default listen address when `BASE_GATEWAY_LISTEN` is unset.
pub const DEFAULT_LISTEN: &str = "0.0.0.0:8080";

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

/// Resolve `SubnetOwnerHotkey` and compare it with `configured_hotkey`.
///
/// Does **not** bind any socket. Call this before [`TcpListener::bind`].
///
/// Under [`OwnerCheck::Advisory`] a mismatch (or an unreadable owner) is logged
/// and startup proceeds: the trust anchor for consumers is the bundle signature
/// under the configured hotkey, not this lookup.
///
/// # Errors
///
/// Under [`OwnerCheck::Required`]: [`GatewayError::MasterMismatch`] on a
/// differing owner, or [`GatewayError::Chain`] when the lookup fails.
pub fn check_master_only(
    chain: &dyn ChainClient,
    netuid: u16,
    configured_hotkey: &[u8],
    mode: OwnerCheck,
) -> Result<(), GatewayError> {
    let strict = mode == OwnerCheck::Required;
    let owner = match chain.subnet_owner_hotkey(netuid) {
        Ok(o) => o,
        Err(e) if strict => return Err(e.into()),
        Err(e) => {
            tracing::warn!(
                netuid,
                error = %e,
                "subnet owner unreadable; continuing (advisory master check)"
            );
            return Ok(());
        }
    };
    if owner.as_slice() == configured_hotkey {
        tracing::info!(netuid, "master-only check passed: hotkey is subnet owner");
        return Ok(());
    }
    if strict {
        return Err(GatewayError::MasterMismatch {
            configured: hotkey_hex(configured_hotkey),
            owner: hotkey_hex(&owner),
            netuid,
        });
    }
    tracing::warn!(
        netuid,
        configured = %hotkey_hex(configured_hotkey),
        owner = %hotkey_hex(&owner),
        "configured hotkey is NOT the on-chain subnet owner; continuing because {REQUIRE_OWNER_ENV} is unset"
    );
    Ok(())
}

/// [`check_master_only`] with the mode taken from [`REQUIRE_OWNER_ENV`].
///
/// # Errors
///
/// See [`check_master_only`].
pub fn assert_master_only(
    chain: &dyn ChainClient,
    netuid: u16,
    configured_hotkey: &[u8],
) -> Result<(), GatewayError> {
    check_master_only(chain, netuid, configured_hotkey, OwnerCheck::from_env())
}

/// Resolve trust-root directory: env, then `./config`, then `/etc/base/config`.
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
    std::path::PathBuf::from("/etc/base/config")
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

/// Build the full application router (health + registry + weights + proxy) without binding.
///
/// Production path loads owner-signed challenges from [`trust_root_dir`].
///
/// `extra` is merged last: services owned by other crates (the miner endpoint
/// announce router) mount through it so this crate does not grow a dependency
/// on each one.
///
/// # Errors
///
/// Telemetry, trust-root load, or HTTP client construction failures.
pub fn build_app(
    metrics: metrics_exporter_prometheus::PrometheusHandle,
    registry: Arc<Registry>,
    chain: SharedChain,
    tls: &TlsConfig,
    stores: Stores,
    extra: Option<Router>,
) -> Result<Router, GatewayError> {
    // Fail closed: an empty trust root would seal bundles with a default
    // measurements digest and no challenges. Tests inject one via `build_app_with`.
    let (challenges, mdigest) = load_production_trust_root(3)?;
    let netuid = std::env::var("BASE_NETUID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1u16);
    let state = GatewayState::with_parts_seal(
        registry, chain, challenges, stores.0, stores.1, mdigest, netuid,
    )
    .map_err(GatewayError::HttpClient)?;
    let app = app::build_router(metrics, state, tls)?;
    let app = match extra {
        Some(extra) => app.merge(extra),
        None => app,
    };
    apply_admin_auth(app)
}

/// Like [`build_app`] with injected trust root and weight store (tests / hydrate).
///
/// # Errors
///
/// Telemetry or HTTP client construction failures.
pub fn build_app_with(
    metrics: metrics_exporter_prometheus::PrometheusHandle,
    registry: Arc<Registry>,
    chain: SharedChain,
    tls: &TlsConfig,
    challenges: Arc<trustroot::ChallengesBody>,
    weights: SharedWeightStore,
) -> Result<Router, GatewayError> {
    build_app_with_bundles(
        metrics,
        registry,
        chain,
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
    chain: SharedChain,
    tls: &TlsConfig,
    challenges: Arc<trustroot::ChallengesBody>,
    weights: SharedWeightStore,
    bundles: SharedBundleStore,
) -> Result<Router, GatewayError> {
    let state = GatewayState::with_parts(registry, chain, challenges, weights, bundles)
        .map_err(GatewayError::HttpClient)?;
    let app = app::build_router(metrics, state, tls)?;
    apply_admin_auth(app)
}

/// Attach `/v1/admin/*` bearer middleware.
///
/// # Errors
///
/// [`AdminAuth::from_env`] fail-closed config errors.
pub fn apply_admin_auth(app: Router) -> Result<Router, GatewayError> {
    let auth = AdminAuth::from_env().map_err(|e| GatewayError::Config(e.to_string()))?;
    Ok(app.layer(axum::middleware::from_fn_with_state(
        auth,
        admin_auth_middleware,
    )))
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
    chain: SharedChain,
    stores: Stores,
    extra: Option<Router>,
    shutdown: F,
) -> Result<(), GatewayError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let registry = Registry::shared(config.registry.clone());
    run_until_with_registry(config, chain, registry, stores, extra, shutdown).await
}

/// Like [`run_until`] but injects a pre-built registry (tests / DB hydrate).
///
/// # Errors
///
/// Same as [`run_until`].
pub async fn run_until_with_registry<F>(
    config: GatewayConfig,
    chain: SharedChain,
    registry: Arc<Registry>,
    stores: Stores,
    extra: Option<Router>,
    shutdown: F,
) -> Result<(), GatewayError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    // CRITICAL: master-only before any bind (D3).
    check_master_only(
        chain.as_ref(),
        config.netuid,
        &config.hotkey,
        config.owner_check,
    )?;

    let _ = init_tracing();
    let metrics = init_metrics()?;
    // Prefer registry knobs from config when the shared handle was default-built.
    let _ = &config.registry;
    // In-memory registry: seed from BASE_GATEWAY_BACKENDS(_FILE) so compose/prod
    // restarts never leave /challenge/* at 503 until a manual admin POST.
    let seeded = registry
        .seed_from_env()
        .map_err(|e| GatewayError::Config(format!("backend seed: {e}")))?;
    if seeded > 0 {
        tracing::info!(
            event = "gateway_backends_seeded",
            created = seeded,
            "boot-seeded challenge backends into in-memory registry"
        );
    }
    let app = build_app(metrics, registry, chain, &config.tls, stores, extra)?;

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
pub async fn run(
    config: GatewayConfig,
    chain: SharedChain,
    stores: Stores,
    extra: Option<Router>,
) -> Result<(), GatewayError> {
    let registry = Registry::shared(config.registry.clone());
    run_until_with_registry(config, chain, registry, stores, extra, shutdown_signal()).await
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

/// Re-export config key names operators may need alongside gateway keys.
pub mod config_keys {
    pub use super::cfg_keys::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use chain::{FakeChain, FakeChainConfig};
    use config::{Config, Role};
    use std::net::SocketAddr;
    use std::time::Duration;
    use tokio::sync::oneshot;
    use validator_sync::SyncChain;

    fn owner_hotkey() -> [u8; 32] {
        [0xA1; 32]
    }

    fn other_hotkey() -> [u8; 32] {
        [0xB2; 32]
    }

    fn mem_stores() -> Stores {
        (
            Arc::new(MemoryRawWeightStore::new()),
            Arc::new(MemoryBundleStore::new()),
        )
    }

    /// Test config with the fail-closed owner check, so the D3 refusal path
    /// stays covered even though production now defaults to advisory.
    fn cfg(listen: SocketAddr, hotkey: [u8; 32]) -> GatewayConfig {
        GatewayConfig {
            listen,
            netuid: 1,
            hotkey,
            registry: RegistryConfig::default(),
            tls: TlsConfig::default(),
            owner_check: OwnerCheck::Required,
        }
    }

    fn cfg_advisory(listen: SocketAddr, hotkey: [u8; 32]) -> GatewayConfig {
        GatewayConfig {
            owner_check: OwnerCheck::Advisory,
            ..cfg(listen, hotkey)
        }
    }

    #[test]
    fn s1_assert_master_mismatch_no_bind_needed() {
        let chain = FakeChain::with_defaults();
        let err = check_master_only(&chain, 1, &other_hotkey(), OwnerCheck::Required)
            .expect_err("mismatch");
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
        check_master_only(&chain, 1, &owner_hotkey(), OwnerCheck::Required).expect("match");
        check_master_only(&chain, 1, &owner_hotkey(), OwnerCheck::Advisory).expect("match");
    }

    /// Advisory mode is the default: a non-owner hotkey must still start.
    #[test]
    fn advisory_mode_allows_non_owner_hotkey() {
        let chain = FakeChain::with_defaults();
        check_master_only(&chain, 1, &other_hotkey(), OwnerCheck::Advisory)
            .expect("advisory must not fail on mismatch");
    }

    /// Advisory mode also tolerates an unreadable owner (unknown netuid).
    #[test]
    fn advisory_mode_tolerates_chain_lookup_failure() {
        let chain = FakeChain::with_defaults();
        check_master_only(&chain, 99, &owner_hotkey(), OwnerCheck::Advisory)
            .expect("advisory must not fail on chain error");
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
        let chain: SharedChain = Arc::new(SyncChain::new(FakeChain::with_defaults()));
        let config = cfg(listen, other_hotkey());

        let err = run_until(config, chain, mem_stores(), None, async {})
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
        run_until_serves_healthz(cfg).await;
    }

    /// Advisory mode (the production default) must bind and serve even when the
    /// configured hotkey is not the on-chain subnet owner.
    #[tokio::test]
    async fn advisory_non_owner_still_serves_healthz() {
        run_until_serves_healthz(|addr, _hk| cfg_advisory(addr, other_hotkey())).await;
    }

    async fn run_until_serves_healthz(
        make_cfg: impl FnOnce(SocketAddr, [u8; 32]) -> GatewayConfig,
    ) {
        // `build_app` fails closed without a trust root; use the committed one.
        std::env::set_var(
            keys::TRUST_ROOT_DIR,
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config"),
        );
        let reserve = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = reserve.local_addr().unwrap();
        drop(reserve);

        let chain: SharedChain = Arc::new(SyncChain::new(FakeChain::new(FakeChainConfig {
            owner_hotkey: owner_hotkey().to_vec(),
            ..FakeChainConfig::default()
        })));
        let config = make_cfg(addr, owner_hotkey());

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let server = tokio::task::spawn_local(async move {
                    run_until(config, chain, mem_stores(), None, async move {
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
        let err = check_master_only(&chain, 99, &owner_hotkey(), OwnerCheck::Required)
            .expect_err("bad netuid");
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
