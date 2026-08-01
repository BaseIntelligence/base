//! `agent-challenge` — operator-side agent-v1 challenge service.
//!
//! Listens on `:8090` for health + pack catalog routes. Challenge secret is
//! loaded from `BASE_CHALLENGE_SK_FILE` (mode 0600 file). Never logs or commits
//! the secret.
//!
//! # Epoch dispatch driver
//!
//! When `BASE_CHALLENGE_DISPATCH=1`, a background tokio task drives one epoch
//! per tick, and every input of that tick comes from the chain:
//!
//! 1. [`chainsnap::read_snapshot`] re-reads the epoch, the `block_B` pin, the
//!    expected set `E` (real uid → hotkey), and each miner's published axon.
//! 2. [`run_epoch_dispatch`] hands the work to [`dispatch::HttpDispatchClient`],
//!    which talks HTTP to those axons under signed dispatch auth.
//! 3. [`attest::ControlPlane`] reads this epoch's attestation outcomes, and
//!    [`grade::grade_outcomes`] gates on them (I1) before verifying each work
//!    receipt against the attested CVM key and running the returned patch
//!    through the pack's held-out Harbor harness.
//! 4. [`score_map_covering_expected`] covers all of `E`, [`emit_signed_leaf_set`]
//!    signs, and [`submit_signed_leaf_set`] POSTs to the gateway.
//!
//! Grading is fail-closed: a miner that cannot be graded gets a `NoScore` with
//! the reason it could not be, never an invented score.

#![forbid(unsafe_code)]

mod attest;
mod chainsnap;
mod dispatch;
mod grade;

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_challenge::{
    emit_signed_leaf_set, load_challenge_secret, pack_routes, public_key_from_secret,
    run_epoch_dispatch, score_map_covering_expected, submit_signed_leaf_set, ActiveSignerRegistry,
    AgentV1Challenge, Challenge, EpochDispatchConfig, GatewayClient, GatewayClientConfig,
    PackCatalogState, CHALLENGE_ID, DEFAULT_MAX_RETRIES, SCORING_VERSION,
};
use agent_pack::PackId;
use axum::routing::get;
use axum::Router;
use chain::ChainClient;
use clap::{Parser, Subcommand};
use tokio::net::TcpListener;
use trustroot::{encode_hex, ParticipantPolicy};

use attest::ControlPlane;
use chainsnap::ChainSnapshot;
use dispatch::HttpDispatchClient;
use grade::{grade_outcomes, GradeEpoch, HarborGradeSource};

/// Operator challenge service CLI.
#[derive(Debug, Parser)]
#[command(name = "agent-challenge", about = "agent-v1 challenge service")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Bind address for health endpoints (default 0.0.0.0:8090).
    #[arg(
        long,
        env = "BASE_CHALLENGE_BIND",
        default_value = "0.0.0.0:8090",
        global = true
    )]
    bind: SocketAddr,
    /// Path to challenge mini-secret (32 raw bytes or hex). Required for ready.
    #[arg(long, env = "BASE_CHALLENGE_SK_FILE", global = true)]
    challenge_sk_file: Option<PathBuf>,
    /// Harbor `tasks/` source directory for pack materialization.
    #[arg(
        long,
        env = "BASE_PACK_SOURCE_DIR",
        default_value = "/var/lib/base/pack-source",
        global = true
    )]
    pack_source_dir: PathBuf,
    /// Materialized pack cache directory.
    #[arg(
        long,
        env = "BASE_PACK_CACHE_DIR",
        default_value = "/var/lib/base/pack-cache",
        global = true
    )]
    pack_cache_dir: PathBuf,
    /// When `1`, bootstrap empty pack source from seed / local HF pull paths.
    #[arg(long, env = "BASE_HF_AUTO_PULL", default_value = "0", global = true)]
    hf_auto_pull: String,
    /// When `1`, enable the background epoch dispatch driver.
    #[arg(
        long,
        env = "BASE_CHALLENGE_DISPATCH",
        default_value = "0",
        global = true
    )]
    challenge_dispatch: String,
    /// Gateway URL for signed leaf submit (default: `BASE_GATEWAY_ENDPOINT` env
    /// or `http://gateway:8080`).
    #[arg(long, env = "BASE_CHALLENGE_GATEWAY_ENDPOINT", global = true)]
    challenge_gateway_endpoint: Option<String>,
    /// Epoch dispatch interval in seconds.
    #[arg(
        long,
        env = "BASE_CHALLENGE_EPOCH_INTERVAL_SECS",
        default_value = "60",
        global = true
    )]
    challenge_epoch_interval_secs: u64,
    /// Substrate JSON-RPC endpoint the epoch, pin, `E`, and axons are read from.
    #[arg(
        long,
        env = "BASE_CHAIN_ENDPOINT",
        default_value = config::DEFAULT_CHAIN_ENDPOINT,
        global = true
    )]
    chain_endpoint: String,
    /// Subnet netuid. Required when dispatch is enabled; there is no safe default.
    #[arg(long, env = "BASE_NETUID", global = true)]
    netuid: Option<u16>,
    /// Docker Engine HTTP base (socket-proxy) used by the held-out verifier.
    #[arg(long, env = "BASE_DOCKER_BASE", global = true)]
    docker_base: Option<String>,
    /// JSON file mapping `pack_id` → digest-pinned verifier image.
    #[arg(long, env = "BASE_VERIFY_IMAGE_MAP", global = true)]
    verify_image_map: Option<PathBuf>,
    /// Verifier image for packs absent from the map (single-pack operators).
    #[arg(long, env = "BASE_ENVIRONMENT_IMAGE", global = true)]
    environment_image: Option<String>,
    /// Staging root for verifier binds.
    #[arg(
        long,
        env = "BASE_VERIFY_WORK_ROOT",
        default_value = "/var/lib/base/verify",
        global = true
    )]
    verify_work_root: PathBuf,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print challenge id, scoring version, and public key derived from the secret file.
    Identity,
    /// Run health + pack HTTP server (default when no subcommand).
    Serve,
}

/// Background epoch dispatch driver state surfaced to `/readyz`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchState {
    /// No dispatch in flight.
    Idle,
    /// Dispatch currently running.
    Active,
    /// Last dispatch failed.
    Error,
}

impl DispatchState {
    fn as_str(self) -> &'static str {
        match self {
            DispatchState::Idle => "idle",
            DispatchState::Active => "active",
            DispatchState::Error => "error",
        }
    }
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("agent-challenge: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let is_identity = matches!(cli.cmd, Some(Cmd::Identity));
    if is_identity {
        return cmd_identity(cli.challenge_sk_file.as_ref());
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(cmd_serve(cli))
}

fn cmd_identity(sk_file: Option<&PathBuf>) -> Result<(), String> {
    let path = sk_file.ok_or("BASE_CHALLENGE_SK_FILE / --challenge-sk-file required")?;
    let sk = load_challenge_secret(path).map_err(|e| e.to_string())?;
    let pk = public_key_from_secret(&sk).map_err(|e| e.to_string())?;
    let ch = AgentV1Challenge::new();
    println!("challenge_id={}", ch.challenge_id());
    println!("scoring_version={}", ch.scoring_version());
    println!("public_key={}", encode_hex(&pk));
    println!("challenge_id_const={CHALLENGE_ID}");
    println!("scoring_version_const={SCORING_VERSION}");
    Ok(())
}

async fn cmd_serve(cli: Cli) -> Result<(), String> {
    // Propagate HF auto-pull flag into ensure_pack_source env contract.
    if matches!(
        cli.hf_auto_pull.as_str(),
        "1" | "true" | "TRUE" | "yes" | "YES"
    ) {
        std::env::set_var("BASE_HF_AUTO_PULL", "1");
    }

    let sk_ready = match cli.challenge_sk_file.as_ref() {
        Some(p) if p.is_file() => load_challenge_secret(p).is_ok(),
        _ => false,
    };

    let pack_state =
        match PackCatalogState::open_from_source(&cli.pack_source_dir, &cli.pack_cache_dir) {
            Ok(s) => {
                tracing::info!(
                    event = "pack_catalog_loaded",
                    source = %cli.pack_source_dir.display(),
                    cache = %cli.pack_cache_dir.display(),
                    packs = s.catalog().len(),
                    pin = %s.catalog().pin(),
                    "pack catalog ready"
                );
                Some(Arc::new(s))
            }
            Err(e) => {
                tracing::warn!(
                    event = "pack_catalog_bootstrap_failed",
                    source = %cli.pack_source_dir.display(),
                    error = %e,
                    "pack catalog not loaded; /v1/* pack routes unavailable"
                );
                None
            }
        };

    // Start the background epoch dispatch driver when enabled. Fail-closed if
    // the challenge signing key is missing or unreadable.
    let dispatch_state = setup_dispatch_driver(&cli, pack_state.as_ref()).await?;

    let catalog_ready = pack_state.as_ref().is_some_and(|s| s.is_ready());
    let ready = sk_ready && catalog_ready;
    let ready_reason = if !sk_ready {
        "challenge secret not loaded"
    } else if !catalog_ready {
        "pack catalog empty or not loaded"
    } else {
        "ready"
    };

    let mut app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route(
            "/readyz",
            get({
                let base_ready = ready;
                let base_reason = ready_reason.to_owned();
                let dispatch_state = dispatch_state.clone();
                move || {
                    let base_reason = base_reason.clone();
                    let dispatch_state = dispatch_state.clone();
                    async move {
                        let (dstate_str, dispatch_on) = match &dispatch_state {
                            Some(st) => {
                                let s =
                                    *st.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                                (s.as_str(), true)
                            }
                            None => ("disabled", false),
                        };
                        let body = if dispatch_on {
                            format!("{base_reason} dispatch={dstate_str}")
                        } else {
                            base_reason
                        };
                        let ok = base_ready && (!dispatch_on || dstate_str != "error");
                        if ok {
                            (axum::http::StatusCode::OK, body)
                        } else {
                            (axum::http::StatusCode::SERVICE_UNAVAILABLE, body)
                        }
                    }
                }
            }),
        );

    if let Some(state) = pack_state {
        app = app.merge(pack_routes(state));
    }

    let listener = TcpListener::bind(cli.bind)
        .await
        .map_err(|e| format!("bind {}: {e}", cli.bind))?;
    let actual = listener.local_addr().map_err(|e| e.to_string())?;
    tracing::info!(
        event = "agent_challenge_listening",
        %actual,
        challenge_id = CHALLENGE_ID,
        scoring_version = SCORING_VERSION,
        ready,
        ready_reason,
        dispatch_enabled = dispatch_state.is_some(),
        "agent-challenge health + pack server"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Whether the epoch dispatch driver is enabled by env/CLI flag.
fn dispatch_enabled(cli: &Cli) -> bool {
    matches!(
        cli.challenge_dispatch.as_str(),
        "1" | "true" | "TRUE" | "yes" | "YES"
    )
}

/// Resolve the gateway endpoint: explicit override → `BASE_GATEWAY_ENDPOINT`
/// env → `http://gateway:8080`.
fn resolve_gateway_endpoint(cli: &Cli) -> String {
    if let Some(ep) = &cli.challenge_gateway_endpoint {
        return ep.clone();
    }
    if let Ok(ep) = std::env::var("BASE_GATEWAY_ENDPOINT") {
        if !ep.trim().is_empty() {
            return ep;
        }
    }
    "http://gateway:8080".to_owned()
}

/// Control-plane Postgres URL from the shared operator config, if configured.
///
/// # Errors
///
/// Unreadable or empty `database_url_file`.
fn resolve_database_url() -> Result<Option<String>, String> {
    let cfg = config::load().map_err(|e| e.to_string())?;
    if let Some(url) = cfg.database_url.as_ref() {
        return Ok(Some(url.clone()));
    }
    let Some(path) = cfg.database_url_file.as_ref() else {
        return Ok(None);
    };
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let trimmed = raw.trim().to_owned();
    if trimmed.is_empty() {
        return Err("database_url_file is empty".into());
    }
    Ok(Some(trimmed))
}

/// Current unix time in milliseconds (best-effort; 0 if the clock is before epoch).
fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0))
}

/// Pack ids eligible for dispatch this epoch.
///
/// # Errors
///
/// Empty or unloaded catalog: without a pack there is nothing to dispatch, and
/// a placeholder id would only produce ungradeable results.
fn dispatch_catalog(pack_state: Option<&Arc<PackCatalogState>>) -> Result<Vec<PackId>, String> {
    let ids = pack_state
        .map(|s| s.catalog().pack_ids())
        .unwrap_or_default();
    if ids.is_empty() {
        return Err("BASE_CHALLENGE_DISPATCH=1 requires a non-empty pack catalog".into());
    }
    Ok(ids)
}

/// Everything the driver needs that does not change between ticks.
struct DriverConfig {
    sk: [u8; 32],
    pk: [u8; 32],
    netuid: u16,
    policy: ParticipantPolicy,
    catalog: Vec<PackId>,
    attest: ControlPlane,
    gateway: GatewayClient,
    signers: Arc<ActiveSignerRegistry>,
    grade_source: HarborGradeSource,
    interval: Duration,
}

/// Wire the background epoch dispatch driver. Returns the shared
/// [`DispatchState`] for `/readyz` when enabled, else `None`.
///
/// # Errors
///
/// Fail-closed when dispatch is enabled but the challenge signing key, netuid,
/// pack catalog, docker base, chain connection, attestation control plane, or
/// gateway client is missing.
async fn setup_dispatch_driver(
    cli: &Cli,
    pack_state: Option<&Arc<PackCatalogState>>,
) -> Result<Option<Arc<Mutex<DispatchState>>>, String> {
    if !dispatch_enabled(cli) {
        return Ok(None);
    }

    let sk_path = cli
        .challenge_sk_file
        .as_ref()
        .ok_or("BASE_CHALLENGE_DISPATCH=1 requires BASE_CHALLENGE_SK_FILE")?;
    let sk = load_challenge_secret(sk_path)
        .map_err(|e| format!("dispatch enabled but challenge secret load failed: {e}"))?;
    let pk = public_key_from_secret(&sk).map_err(|e| format!("challenge public key: {e}"))?;

    let netuid = cli
        .netuid
        .ok_or("BASE_CHALLENGE_DISPATCH=1 requires BASE_NETUID")?;
    let docker_base = cli
        .docker_base
        .clone()
        .ok_or("BASE_CHALLENGE_DISPATCH=1 requires BASE_DOCKER_BASE (grading is not optional)")?;
    let images = match cli.verify_image_map.as_ref() {
        Some(p) => HarborGradeSource::load_image_map(p)?,
        None => BTreeMap::new(),
    };
    if images.is_empty() && cli.environment_image.is_none() {
        return Err(
            "BASE_CHALLENGE_DISPATCH=1 requires BASE_VERIFY_IMAGE_MAP or BASE_ENVIRONMENT_IMAGE"
                .into(),
        );
    }

    // I1 has no offline fallback: without the control plane every miner would
    // be Missing, so refuse to serve rather than zero the whole subnet quietly.
    let database_url = resolve_database_url()?.ok_or(
        "BASE_CHALLENGE_DISPATCH=1 requires a control-plane database for attestation outcomes",
    )?;
    let attest = ControlPlane::connect(&database_url).await?;

    let mut chain = chain_live::LiveChainClient::connect(&cli.chain_endpoint)
        .map_err(|e| format!("chain connect {}: {e}", cli.chain_endpoint))?;
    chain.set_netuid(netuid);

    let endpoint = resolve_gateway_endpoint(cli);
    let gateway = GatewayClient::new(GatewayClientConfig {
        base_url: endpoint.clone(),
        max_attempts: DEFAULT_MAX_RETRIES,
        backoff: Duration::from_millis(50),
    })
    .map_err(|e| format!("gateway client build: {e}"))?;
    let catalog = dispatch_catalog(pack_state)?;
    let state = Arc::new(Mutex::new(DispatchState::Idle));

    tracing::info!(
        event = "epoch_dispatch_driver_start",
        netuid,
        chain = %cli.chain_endpoint,
        interval_secs = cli.challenge_epoch_interval_secs,
        catalog = catalog.len(),
        gateway = %endpoint,
        "epoch dispatch driver enabled"
    );

    let cfg = DriverConfig {
        sk,
        pk,
        netuid,
        policy: ParticipantPolicy::AllMetagraphHotkeys,
        catalog,
        attest,
        gateway,
        signers: ActiveSignerRegistry::new(),
        grade_source: HarborGradeSource {
            docker_base,
            cache_dir: cli.pack_cache_dir.clone(),
            work_root: cli.verify_work_root.clone(),
            images,
            default_image: cli.environment_image.clone(),
        },
        interval: Duration::from_secs(cli.challenge_epoch_interval_secs),
    };

    tokio::spawn(epoch_dispatch_driver(cfg, chain, Arc::clone(&state)));

    Ok(Some(state))
}

/// Run one epoch dispatch + grade + score + sign + submit, returning the leaf count.
///
/// # Errors
///
/// String describing the first failure (dispatch, grade join, emit, or submit).
async fn run_one_epoch(cfg: &DriverConfig, snap: &ChainSnapshot) -> Result<usize, String> {
    let epoch = snap.pin.epoch;
    let expected_keys = snap.expected.hotkeys();
    let deadline = cfg.interval;
    let deadline_unix_ms =
        unix_now_ms().saturating_add(u64::try_from(deadline.as_millis()).unwrap_or(60_000));

    let dispatch_cfg = EpochDispatchConfig {
        challenge_id: CHALLENGE_ID.to_owned(),
        scoring_version: SCORING_VERSION,
        epoch,
        expected: snap.expected.clone(),
        catalog: cfg.catalog.clone(),
        deadline,
        deadline_unix_ms,
    };

    let client = Arc::new(HttpDispatchClient::new(
        snap.endpoints.clone(),
        cfg.sk,
        cfg.pk,
    )?);
    let result = run_epoch_dispatch(&dispatch_cfg, client, &cfg.signers)
        .await
        .map_err(|e| e.to_string())?;

    // I1 gate inputs are read once per epoch, before any grading compute.
    let expected_vec: Vec<_> = expected_keys.iter().copied().collect();
    let attest = cfg.attest.epoch_attestations(epoch, &expected_vec).await;

    // Harbor grading is a long blocking Docker run; keep it off the reactor.
    let (source, expected, endpoints, outcomes, netuid) = (
        cfg.grade_source.clone(),
        snap.expected.clone(),
        snap.endpoints.clone(),
        result.outcomes.clone(),
        cfg.netuid,
    );
    let graded = tokio::task::spawn_blocking(move || {
        grade_outcomes(
            &source,
            &GradeEpoch {
                netuid,
                epoch,
                challenge_id: CHALLENGE_ID,
                scoring_version: SCORING_VERSION,
                expected: &expected,
                endpoints: &endpoints,
                outcomes: &outcomes,
                attest: &attest,
            },
        )
    })
    .await
    .map_err(|e| format!("grade task: {e}"))?;

    let scores = score_map_covering_expected(&expected_keys, &graded, &result.outcomes);

    let leaves = emit_signed_leaf_set(&cfg.sk, epoch, &expected_keys, &scores)
        .map_err(|e| format!("emit leaves: {e}"))?;
    let n = leaves.len();
    submit_signed_leaf_set(&cfg.gateway, &leaves)
        .await
        .map_err(|e| format!("submit leaves: {e}"))?;
    Ok(n)
}

/// Set the shared dispatch state (poisoned locks are recovered, never panic).
fn set_dispatch_state(state: &Mutex<DispatchState>, s: DispatchState) {
    *state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = s;
}

/// Eternal epoch loop: sleep → read chain → dispatch → grade → sign → submit.
///
/// The chain snapshot is taken **inside** the loop: epoch, pin, `E`, and axons
/// all move as the subnet moves.
async fn epoch_dispatch_driver<C: ChainClient + Send + 'static>(
    cfg: DriverConfig,
    chain: C,
    state: Arc<Mutex<DispatchState>>,
) {
    loop {
        tokio::time::sleep(cfg.interval).await;
        set_dispatch_state(&state, DispatchState::Active);

        let snap = match chainsnap::read_snapshot(&chain, cfg.netuid, &cfg.policy) {
            Ok(s) => s,
            Err(e) => {
                set_dispatch_state(&state, DispatchState::Error);
                tracing::warn!(
                    event = "epoch_chain_snapshot_error",
                    netuid = cfg.netuid,
                    error = %e,
                    "chain snapshot failed; no leaves emitted this tick"
                );
                continue;
            }
        };

        tracing::info!(
            event = "epoch_dispatch_start",
            epoch = snap.pin.epoch,
            block_b = snap.pin.block_b,
            block_hash = %hex::encode(snap.pin.block_hash),
            participants = snap.expected.participants.len(),
            reachable = snap.endpoints.len(),
            "epoch dispatch tick"
        );

        match run_one_epoch(&cfg, &snap).await {
            Ok(leaves) => {
                set_dispatch_state(&state, DispatchState::Idle);
                tracing::info!(
                    event = "epoch_dispatch_complete",
                    epoch = snap.pin.epoch,
                    leaves,
                    "epoch dispatch complete"
                );
            }
            Err(e) => {
                set_dispatch_state(&state, DispatchState::Error);
                tracing::warn!(
                    event = "epoch_dispatch_error",
                    epoch = snap.pin.epoch,
                    error = %e,
                    "epoch dispatch failed"
                );
            }
        }
    }
}
