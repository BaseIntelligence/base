//! `agent-challenge` — operator-side agent-v1 challenge service.
//!
//! Listens on `:8090` for health + pack catalog routes. Challenge secret is
//! loaded from `BASE_CHALLENGE_SK_FILE` (mode 0600 file). Never logs or commits
//! the secret.
//!
//! # Epoch dispatch driver
//!
//! When `BASE_CHALLENGE_DISPATCH=1`, a background tokio task periodically drives
//! an epoch dispatch: it builds an [`ExpectedSet`] from
//! `BASE_FAKE_METAGRAPH_HOTKEYS`, runs [`run_epoch_dispatch`] with a stub
//! [`OperatorDispatchClient`] that declares zero runner capacity (the operator
//! host is a scoring coordinator, not a pack runner), scores the outcomes with
//! [`score_map_covering_expected`], signs the leaf set with
//! [`emit_signed_leaf_set`], and POSTs it via [`submit_signed_leaf_set`].
//!
//! The epoch number is a CLI/env argument (`BASE_CHALLENGE_EPOCH`, default 0).
//! The daemon does **not** read the chain for the epoch or the `block_B` pin —
//! that is a future enhancement. `block_hash` is stamped with a fixed
//! placeholder until chain-tip epoch integration lands.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_challenge::{
    emit_signed_leaf_set, load_challenge_secret, pack_routes, public_key_from_secret,
    run_epoch_dispatch, score_map_covering_expected, submit_signed_leaf_set, ActiveSignerRegistry,
    AgentV1Challenge, Challenge, EpochDispatchClient, EpochDispatchConfig, ExpectedParticipant,
    ExpectedSet, GatewayClient, GatewayClientConfig, Hotkey, PackCatalogState, RunnerCapacity,
    CHALLENGE_ID, DEFAULT_MAX_RETRIES, SCORING_VERSION,
};
use agent_dispatch::{TaskDescriptorV1, TaskResultV1};
use agent_pack::PackId;
use axum::routing::get;
use axum::Router;
use clap::{Parser, Subcommand};
use tokio::net::TcpListener;
use trustroot::encode_hex;

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
    /// Epoch number stamped into dispatch + leaves (no chain read yet).
    #[arg(long, env = "BASE_CHALLENGE_EPOCH", default_value = "0", global = true)]
    challenge_epoch: u64,
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

/// Stub dispatch client for the operator-side challenge daemon.
///
/// The operator host is a scoring coordinator, not a pack runner. It declares
/// **zero** runner capacity, so every miner resolves to
/// `CapacityExhausted` → `ChallengeInternal` `NoScore`. Real runner integration
/// replaces this client later (the `run_pack` body is unreachable while capacity
/// stays zero).
#[derive(Debug, Clone, Copy, Default)]
struct OperatorDispatchClient;

impl EpochDispatchClient for OperatorDispatchClient {
    async fn capacity(&self, _miner: Hotkey) -> RunnerCapacity {
        RunnerCapacity {
            max_concurrency: 0,
            current_load: 0,
        }
    }

    async fn run_pack(
        &self,
        _miner: Hotkey,
        _descriptor: TaskDescriptorV1,
    ) -> Result<TaskResultV1, String> {
        // Unreachable: capacity() advertises zero slots, so run_epoch_dispatch
        // short-circuits every miner to CapacityExhausted before calling run_pack.
        Err("operator dispatch client declares no runner capacity".into())
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
    let dispatch_state = setup_dispatch_driver(&cli, pack_state.as_ref())?;

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

/// Build an [`ExpectedSet`] from `BASE_FAKE_METAGRAPH_HOTKEYS` (comma-separated
/// 64-hex hotkeys). `block_hash` is a fixed placeholder — the daemon does not
/// read the chain yet.
///
/// # Errors
///
/// Empty env / malformed hotkey hex / wrong byte length.
fn parse_expected_set_from_env() -> Result<ExpectedSet, String> {
    let raw = std::env::var("BASE_FAKE_METAGRAPH_HOTKEYS").unwrap_or_default();
    // Fixed placeholder pin — replaced by chain-tip `block_B` pin later.
    let mut block_hash = [0u8; 32];
    block_hash[0] = 0xBD;
    block_hash[1] = 0xE0;
    block_hash[2] = 0x0C;

    let mut participants = Vec::new();
    for (idx, part) in raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .enumerate()
    {
        let stripped = part
            .strip_prefix("0x")
            .or_else(|| part.strip_prefix("0X"))
            .unwrap_or(part);
        let bytes =
            hex::decode(stripped).map_err(|e| format!("invalid hotkey hex '{part}': {e}"))?;
        let hk: [u8; 32] = bytes.try_into().map_err(|v: Vec<u8>| {
            format!(
                "hotkey must be 32 bytes (64 hex chars), got {} bytes",
                v.len()
            )
        })?;
        let uid = u16::try_from(idx).unwrap_or(u16::MAX);
        participants.push(ExpectedParticipant { hotkey: hk, uid });
    }

    if participants.is_empty() {
        return Err(
            "BASE_CHALLENGE_DISPATCH=1 requires BASE_FAKE_METAGRAPH_HOTKEYS (comma-separated 64-hex)"
                .into(),
        );
    }

    Ok(ExpectedSet {
        block_hash,
        participants,
    })
}

/// Current unix time in milliseconds (best-effort; 0 if the clock is before epoch).
fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0))
}

/// Catalog for dispatch: real pack ids when the catalog is loaded and non-empty,
/// else a single placeholder (the stub never runs packs, but
/// [`run_epoch_dispatch`] requires a non-empty catalog for `select_pack`).
fn dispatch_catalog(pack_state: Option<&Arc<PackCatalogState>>) -> Vec<PackId> {
    pack_state
        .and_then(|s| {
            let ids = s.catalog().pack_ids();
            if ids.is_empty() {
                None
            } else {
                Some(ids)
            }
        })
        .unwrap_or_else(|| vec![PackId::new("operator-stub")])
}

/// Wire the background epoch dispatch driver. Returns the shared
/// [`DispatchState`] for `/readyz` when enabled, else `None`.
///
/// # Errors
///
/// Fail-closed when dispatch is enabled but the challenge signing key is
/// missing/unreadable, the expected set is empty, or the gateway client cannot
/// be built.
fn setup_dispatch_driver(
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

    let expected = parse_expected_set_from_env()?;
    let endpoint = resolve_gateway_endpoint(cli);
    let gateway = GatewayClient::new(GatewayClientConfig {
        base_url: endpoint.clone(),
        max_attempts: DEFAULT_MAX_RETRIES,
        backoff: Duration::from_millis(50),
    })
    .map_err(|e| format!("gateway client build: {e}"))?;
    let catalog = dispatch_catalog(pack_state);
    let signers = ActiveSignerRegistry::new();
    let state = Arc::new(Mutex::new(DispatchState::Idle));
    let interval = Duration::from_secs(cli.challenge_epoch_interval_secs);

    tracing::info!(
        event = "epoch_dispatch_driver_start",
        epoch = cli.challenge_epoch,
        interval_secs = cli.challenge_epoch_interval_secs,
        participants = expected.participants.len(),
        catalog = catalog.len(),
        gateway = %endpoint,
        "epoch dispatch driver enabled"
    );

    tokio::spawn(epoch_dispatch_driver(
        sk,
        expected,
        catalog,
        gateway,
        signers,
        interval,
        cli.challenge_epoch,
        Arc::clone(&state),
    ));

    Ok(Some(state))
}

/// Run one epoch dispatch + score + sign + submit, returning the leaf count.
///
/// # Errors
///
/// String describing the first failure (dispatch, emit, or submit).
async fn run_one_epoch(
    sk: &[u8; 32],
    expected: &ExpectedSet,
    catalog: &[PackId],
    gateway: &GatewayClient,
    signers: &Arc<ActiveSignerRegistry>,
    epoch: u64,
    deadline: Duration,
) -> Result<usize, String> {
    let expected_keys = expected.hotkeys();
    let deadline_unix_ms =
        unix_now_ms().saturating_add(u64::try_from(deadline.as_millis()).unwrap_or(60_000));

    let cfg = EpochDispatchConfig {
        challenge_id: CHALLENGE_ID.to_owned(),
        scoring_version: SCORING_VERSION,
        epoch,
        expected: expected.clone(),
        catalog: catalog.to_vec(),
        deadline,
        deadline_unix_ms,
    };

    let client = Arc::new(OperatorDispatchClient);
    let result = run_epoch_dispatch(&cfg, client, signers)
        .await
        .map_err(|e| e.to_string())?;

    // Empty graded map: every CapacityExhausted outcome → ChallengeInternal NoScore.
    let graded: BTreeMap<[u8; 32], agent_challenge::ScoreOrAbsence> = BTreeMap::new();
    let scores = score_map_covering_expected(&expected_keys, &graded, &result.outcomes);

    let leaves = emit_signed_leaf_set(sk, epoch, &expected_keys, &scores)
        .map_err(|e| format!("emit leaves: {e}"))?;
    let n = leaves.len();
    submit_signed_leaf_set(gateway, &leaves)
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

/// Eternal epoch dispatch loop: sleep → dispatch → score → sign → submit.
#[allow(clippy::too_many_arguments)]
async fn epoch_dispatch_driver(
    sk: [u8; 32],
    expected: ExpectedSet,
    catalog: Vec<PackId>,
    gateway: GatewayClient,
    signers: Arc<ActiveSignerRegistry>,
    interval: Duration,
    epoch: u64,
    state: Arc<Mutex<DispatchState>>,
) {
    loop {
        tokio::time::sleep(interval).await;

        set_dispatch_state(&state, DispatchState::Active);
        tracing::info!(
            event = "epoch_dispatch_start",
            epoch,
            participants = expected.participants.len(),
            interval_secs = interval.as_secs(),
            "epoch dispatch tick"
        );

        match run_one_epoch(
            &sk, &expected, &catalog, &gateway, &signers, epoch, interval,
        )
        .await
        {
            Ok(leaves) => {
                set_dispatch_state(&state, DispatchState::Idle);
                tracing::info!(
                    event = "epoch_dispatch_complete",
                    epoch,
                    leaves,
                    "epoch dispatch complete"
                );
            }
            Err(e) => {
                set_dispatch_state(&state, DispatchState::Error);
                tracing::warn!(
                    event = "epoch_dispatch_error",
                    epoch,
                    error = %e,
                    "epoch dispatch failed"
                );
            }
        }
    }
}
