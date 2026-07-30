//! Validator HTTP app: telemetry + mirror + consensus root + graceful shutdown.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use chain::ChainClient;
use telemetry::{HealthRouterBuilder, ReadyCheck};
use tokio::net::TcpListener;
use tokio::sync::watch;

use crate::coordination::CoordinationClient;
use crate::crosscheck::{consensus_root_router, RootStatementStore, SharedRootStore};
use crate::epoch::{epoch_from_block, EpochSnapshot};
use crate::error::ValidatorError;
use crate::mirror::{mirror_router, MemoryMirrorStore, SharedMirrorStore};
use crate::peers::PeerBook;
use crate::registration::RegistrationStub;
use dissent::{dissent_router, DissentStore, SharedDissentStore};

/// Runtime knobs for the skeleton (injectable for tests).
#[derive(Clone)]
pub struct ValidatorRuntime {
    /// Epoch length in blocks.
    pub epoch_length: u64,
    /// Listen address (e.g. `127.0.0.1:0` for ephemeral).
    pub listen_addr: SocketAddr,
    /// Optional gateway base URL.
    pub gateway_endpoint: Option<String>,
    /// Registration stub.
    pub registration: RegistrationStub,
    /// Peer book for mirror fetch (hotkey → base URL), filtered by metagraph.
    pub peers: PeerBook,
    /// Optional pre-built mirror store (tests share stores across validators).
    pub mirror: Option<SharedMirrorStore>,
    /// Optional pre-built root-statement store (task 31).
    pub root_store: Option<SharedRootStore>,
    /// Optional 32-byte mini-secret for signing local root statements.
    pub root_signing_secret: Option<[u8; 32]>,
    /// This validator's hotkey (excluded from peer sample).
    pub own_hotkey: Option<Vec<u8>>,
    /// Minimum agreeing peer sample (D26); default from config.
    pub min_peer_sample: u32,
    /// Optional pre-built dissent store (task 32).
    pub dissent_store: Option<SharedDissentStore>,
    /// Quarantine surviving-share floor (D6).
    pub min_share_mass_bps: u16,
}

impl Default for ValidatorRuntime {
    fn default() -> Self {
        Self {
            epoch_length: config::DEFAULT_EPOCH_LENGTH,
            listen_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            gateway_endpoint: None,
            registration: RegistrationStub::new(),
            peers: PeerBook::new(),
            mirror: None,
            root_store: None,
            root_signing_secret: None,
            own_hotkey: None,
            min_peer_sample: config::DEFAULT_MIN_PEER_SAMPLE,
            dissent_store: None,
            min_share_mass_bps: config::DEFAULT_MIN_SHARE_MASS_BPS,
        }
    }
}

/// Build readiness checks for chain tip + optional DB ping.
pub fn chain_ready_check<C>(chain: Arc<C>, name: impl Into<String>) -> ReadyCheck
where
    C: ChainClient + Send + Sync + 'static,
{
    let name = name.into();
    ReadyCheck::new(name, move || {
        let chain = Arc::clone(&chain);
        async move { chain.current_block().map(|_| ()).map_err(|e| e.to_string()) }
    })
}

/// Always-ok DB check (unit tests / no postgres).
#[must_use]
pub fn db_ready_ok() -> ReadyCheck {
    ReadyCheck::always_ok("db")
}

/// Injectable DB readiness from an async closure (keeps sqlx out of this crate).
pub fn db_ready_from_fn<F, Fut>(check: F) -> ReadyCheck
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), String>> + Send + 'static,
{
    ReadyCheck::new("db", check)
}

/// Build the observability router with injectable checks.
///
/// # Errors
///
/// Telemetry builder failures.
pub fn build_health_router(checks: Vec<ReadyCheck>) -> Result<Router, ValidatorError> {
    let handle = telemetry::init_metrics()?;
    Ok(HealthRouterBuilder::new()
        .metrics_handle(handle)
        .readiness_checks(checks)
        .build()?)
}

/// Running validator handle (bound port + shutdown + mirror + root store).
pub struct RunningValidator {
    /// Bound local address.
    pub addr: SocketAddr,
    /// Signal graceful shutdown (send `true`).
    pub shutdown_tx: watch::Sender<bool>,
    /// Join handle for the serve task.
    pub join: tokio::task::JoinHandle<Result<(), ValidatorError>>,
    /// Coordination client used by the process.
    pub coordination: Arc<CoordinationClient>,
    /// Verified-bundle mirror store served at `/v1/bundle/root/{root}`.
    pub mirror: SharedMirrorStore,
    /// Signed peer-root statements (local serve + evidence).
    pub root_store: SharedRootStore,
    /// Peer book used for outbound peer fetch.
    pub peers: PeerBook,
    /// Optional mini-secret for signing local roots.
    pub root_signing_secret: Option<[u8; 32]>,
    /// Own hotkey (sample exclusion).
    pub own_hotkey: Option<Vec<u8>>,
    /// Minimum peer sample (D26).
    pub min_peer_sample: u32,
    /// Dissent statements store (task 32).
    pub dissent_store: SharedDissentStore,
    /// Quarantine mass floor (D6).
    pub min_share_mass_bps: u16,
    /// Epoch length.
    pub epoch_length: u64,
    /// Chain tip reader for epoch snapshots.
    tip: Arc<dyn Fn() -> Result<u64, String> + Send + Sync>,
}

impl RunningValidator {
    /// Current epoch snapshot from the injected tip.
    ///
    /// # Errors
    ///
    /// Tip or `epoch_length` failures.
    pub fn epoch_snapshot(&self) -> Result<EpochSnapshot, String> {
        let block = (self.tip)()?;
        epoch_from_block(block, self.epoch_length)
    }

    /// Base URL for this validator's HTTP surface (`http://{addr}`).
    #[must_use]
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Request graceful shutdown and wait for the server task.
    ///
    /// # Errors
    ///
    /// Serve-task failures.
    pub async fn shutdown(self) -> Result<(), ValidatorError> {
        let _ = self.shutdown_tx.send(true);
        match self.join.await {
            Ok(r) => r,
            Err(e) => Err(ValidatorError::Serve(format!("join: {e}"))),
        }
    }
}

/// Spawn the health + mirror + consensus-root server with graceful shutdown.
///
/// # Errors
///
/// Bind, router, or coordination client build failures.
pub async fn spawn_validator<C>(
    runtime: ValidatorRuntime,
    chain: Arc<C>,
    db_check: ReadyCheck,
    extra_checks: Vec<ReadyCheck>,
) -> Result<RunningValidator, ValidatorError>
where
    C: ChainClient + Send + Sync + 'static,
{
    let coordination = Arc::new(
        CoordinationClient::new(runtime.gateway_endpoint.clone())
            .map_err(|e| ValidatorError::Coordination(e.to_string()))?,
    );

    // Optional one-shot coordination tick (allowed path only).
    if let Err(e) = coordination.tick().await {
        tracing::warn!(error = %e, "coordination tick failed (non-fatal for skeleton)");
    }

    let mirror = runtime
        .mirror
        .clone()
        .unwrap_or_else(MemoryMirrorStore::shared);
    let root_store = runtime
        .root_store
        .clone()
        .unwrap_or_else(RootStatementStore::shared);
    let dissent_store = runtime
        .dissent_store
        .clone()
        .unwrap_or_else(DissentStore::shared);
    let peers = runtime.peers.clone();
    let min_share_mass_bps = runtime.min_share_mass_bps;
    let root_signing_secret = runtime.root_signing_secret;
    let own_hotkey = runtime.own_hotkey.clone();
    let min_peer_sample = runtime.min_peer_sample;

    let mut checks = vec![chain_ready_check(Arc::clone(&chain), "chain"), db_check];
    checks.extend(extra_checks);

    let app = build_health_router(checks)?
        .merge(mirror_router(Arc::clone(&mirror)))
        .merge(consensus_root_router(Arc::clone(&root_store)))
        .merge(dissent_router(Arc::clone(&dissent_store)));
    let listener = TcpListener::bind(runtime.listen_addr)
        .await
        .map_err(|e| ValidatorError::Serve(format!("bind: {e}")))?;
    let addr = listener
        .local_addr()
        .map_err(|e| ValidatorError::Serve(format!("local_addr: {e}")))?;

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let join = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                loop {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                    if shutdown_rx.changed().await.is_err() {
                        break;
                    }
                }
            })
            .await
            .map_err(|e| ValidatorError::Serve(format!("serve: {e}")))
    });

    let tip_chain = Arc::clone(&chain);
    let tip = Arc::new(move || tip_chain.current_block().map_err(|e| e.to_string()));

    tracing::info!(
        %addr,
        registered = runtime.registration.status().registered,
        has_gateway = coordination.has_gateway(),
        peer_count = peers.all().len(),
        min_peer_sample,
        "validator listening"
    );

    Ok(RunningValidator {
        addr,
        shutdown_tx,
        join,
        coordination,
        mirror,
        root_store,
        peers,
        root_signing_secret,
        own_hotkey,
        min_peer_sample,
        dissent_store,
        min_share_mass_bps,
        epoch_length: runtime.epoch_length,
        tip,
    })
}

/// Convenience: spawn with always-ok DB check (no postgres required).
///
/// # Errors
///
/// Same as [`spawn_validator`].
pub async fn spawn_validator_with_ok_db<C>(
    runtime: ValidatorRuntime,
    chain: Arc<C>,
) -> Result<RunningValidator, ValidatorError>
where
    C: ChainClient + Send + Sync + 'static,
{
    spawn_validator(runtime, chain, db_ready_ok(), vec![]).await
}
