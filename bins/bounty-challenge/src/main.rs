//! `bounty-challenge` — operator-side bounty challenge service (:8095).
//!
//! Intake → ffmpeg compress → agentic similar-24h → admin approve → D24 emit
//! with `TARGET_BUGS` burn-sink on uid=0.

#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown, clippy::too_many_lines)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use bounty_challenge::{
    bounty_router, emit_epoch, openrouter_model, AppState, BountySimAgent, DbBountyStore,
    GatewayClient, GatewayClientConfig, MemoryBountyStore, Orchestrator, OrchestratorConfig,
    CHALLENGE_ID, DEFAULT_OPENROUTER_MODEL, DRY_RUN_BASE_URL, SCORING_VERSION,
};
use chain::ChainClient;
use challenge_agentic::{load_api_key_file, AgentConfig, AgenticBackend, OpenRouterAgent};
use challenge_common::{expected_set_at_chain, PinnedBlockHash};
use challenge_keys::load_challenge_secret;
use clap::Parser;
use crypto::KEY_LEN;
use sha2::{Digest, Sha256};
use submission_gating::{
    watch_once, GatingStore, MemoryGatingStore, MetagraphCache, PgGatingStore,
};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use trustroot::ParticipantPolicy;

/// Bounty challenge CLI.
#[derive(Debug, Parser)]
#[command(
    name = "bounty-challenge",
    about = "Bounty challenge service (port 8095)"
)]
struct Cli {
    /// Bind address.
    #[arg(long, env = "BASE_CHALLENGE_BIND", default_value = "0.0.0.0:8095")]
    bind: SocketAddr,
    /// Force in-memory store even when `BASE_DATABASE_URL` is set.
    /// Env `BOUNTY_FORCE_MEMORY=1|true` is read via [`env_truthy`] (clap bool
    /// env rejects `1`).
    #[arg(long, default_value_t = false)]
    force_memory: bool,
    /// Challenge mini-secret file (optional — emitter disabled when missing).
    #[arg(long, env = "BASE_CHALLENGE_SK_FILE")]
    challenge_sk_file: Option<PathBuf>,
    /// Force Sim agentic + copy compress (CI / no ffmpeg / no OpenRouter).
    /// Env `BOUNTY_FORCE_SIM=1|true` is read via [`env_truthy`].
    #[arg(long, default_value_t = false)]
    force_sim: bool,
    /// Netuid.
    #[arg(long, env = "BASE_NETUID", default_value_t = 1)]
    netuid: u16,
    /// Max concurrent pipeline workers.
    #[arg(long, env = "BOUNTY_MAX_CONCURRENT", default_value_t = 2)]
    max_concurrent: u32,
    /// Chain WS endpoint.
    #[arg(long, env = "BASE_CHAIN_ENDPOINT")]
    chain_endpoint: Option<String>,
    /// Gateway base URL for leaf submit.
    #[arg(
        long,
        env = "BASE_CHALLENGE_GATEWAY_ENDPOINT",
        default_value = "http://gateway:8080"
    )]
    gateway_endpoint: String,
    /// Admin bearer tokens file (one token per line); hashed at boot.
    #[arg(long, env = "BOUNTY_ADMIN_TOKENS_FILE")]
    admin_tokens_file: Option<PathBuf>,
    /// OpenRouter key file for agentic review.
    #[arg(
        long,
        env = "BOUNTY_OPENROUTER_KEY_FILE",
        default_value = "/run/base/openrouter/api_key"
    )]
    openrouter_key_file: PathBuf,
    /// Artifacts volume root.
    #[arg(
        long,
        env = "BOUNTY_ARTIFACTS_ROOT",
        default_value = "/var/lib/bounty/artifacts"
    )]
    artifacts_root: PathBuf,
    /// Metagraph watcher cadence (seconds).
    #[arg(long, env = "BOUNTY_GATING_WATCH_SECS", default_value_t = 60)]
    gating_watch_secs: u64,
    /// Max raw upload bytes.
    #[arg(long, env = "BOUNTY_MAX_UPLOAD_BYTES", default_value_t = 104_857_600)]
    max_upload_bytes: usize,
    /// Emitter poll seconds.
    #[arg(long, env = "BOUNTY_EMIT_POLL_SECS", default_value_t = 15)]
    emit_poll_secs: u64,
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn load_token_hashes(path: Option<&PathBuf>) -> Vec<String> {
    let Some(p) = path else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(p) else {
        tracing::warn!(path = %p.display(), "admin tokens file unreadable");
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|tok| {
            let mut h = Sha256::new();
            h.update(tok.as_bytes());
            hex::encode(h.finalize())
        })
        .collect()
}

fn build_agentic(force_sim: bool, key_file: &Path) -> (Arc<dyn AgenticBackend>, &'static str) {
    if force_sim {
        return (Arc::new(BountySimAgent::new()), "sim");
    }
    match load_api_key_file(key_file) {
        Ok(key) => {
            let model = openrouter_model();
            match OpenRouterAgent::with_config(
                key,
                challenge_agentic::OPENROUTER_API_BASE,
                model,
                AgentConfig::default(),
            ) {
                Ok(agent) => (Arc::new(agent), "openrouter"),
                Err(e) => {
                    tracing::warn!(error = %e, "OpenRouter agent build failed; using BountySimAgent");
                    (Arc::new(BountySimAgent::new()), "sim")
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %key_file.display(),
                "OpenRouter key missing; using BountySimAgent"
            );
            (Arc::new(BountySimAgent::new()), "sim")
        }
    }
}

fn spawn_gating_watcher(
    chain_ep: &str,
    netuid: u16,
    cache: Arc<MetagraphCache>,
    gating: Arc<dyn GatingStore>,
    poll: Duration,
) {
    let ep = chain_ep.to_owned();
    tokio::spawn(async move {
        let mut client: Option<chain_live::LiveChainClient> = None;
        let stores = vec![(CHALLENGE_ID.to_owned(), gating)];
        loop {
            if client.is_none() {
                let ep2 = ep.clone();
                client = match tokio::task::spawn_blocking(move || {
                    let mut c =
                        chain_live::LiveChainClient::connect(&ep2).map_err(|e| e.to_string())?;
                    c.set_netuid(netuid);
                    Ok::<_, String>(c)
                })
                .await
                {
                    Ok(Ok(c)) => Some(c),
                    Ok(Err(e)) => {
                        tracing::warn!(error = %e, "gating watcher: chain connect failed");
                        None
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "gating watcher: join failed");
                        None
                    }
                };
            }
            if let Some(c) = client.as_mut() {
                if let Err(e) = watch_once(c, netuid, &cache, &stores).await {
                    tracing::warn!(error = %e, "gating watcher tick failed (cache kept)");
                    client = None;
                }
            }
            tokio::time::sleep(poll).await;
        }
    });
}

fn spawn_emitter(
    store: Arc<dyn bounty_challenge::BountyStore>,
    gateway: GatewayClient,
    sk: [u8; KEY_LEN],
    chain_ep: String,
    netuid: u16,
    epoch: Arc<AtomicU64>,
    poll: Duration,
) {
    tokio::spawn(async move {
        let mut client: Option<chain_live::LiveChainClient> = None;
        loop {
            if client.is_none() {
                let ep2 = chain_ep.clone();
                client = match tokio::task::spawn_blocking(move || {
                    let mut c =
                        chain_live::LiveChainClient::connect(&ep2).map_err(|e| e.to_string())?;
                    c.set_netuid(netuid);
                    Ok::<_, String>(c)
                })
                .await
                {
                    Ok(Ok(c)) => Some(c),
                    Ok(Err(e)) => {
                        tracing::warn!(error = %e, "emitter: chain connect failed");
                        None
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "emitter: join failed");
                        None
                    }
                };
            }
            if let Some(c) = client.as_mut() {
                match emitter_tick(&store, &gateway, &sk, c, netuid, &epoch).await {
                    Ok(Some(s)) => tracing::info!(
                        epoch = s.epoch,
                        leaves = s.leaves,
                        burn_units = s.burn_units,
                        "bounty epoch emitted"
                    ),
                    Ok(None) => {}
                    Err(e) => {
                        tracing::warn!(error = %e, "emitter tick error");
                        client = None;
                    }
                }
            }
            tokio::time::sleep(poll).await;
        }
    });
}

async fn emitter_tick(
    store: &Arc<dyn bounty_challenge::BountyStore>,
    gateway: &GatewayClient,
    sk: &[u8; KEY_LEN],
    chain: &chain_live::LiveChainClient,
    netuid: u16,
    epoch_atom: &AtomicU64,
) -> Result<Option<bounty_challenge::EmitSummary>, String> {
    let state =
        chain::gather_schedule_state(chain, netuid).map_err(|e| format!("schedule: {e}"))?;
    let epoch = state.subnet_epoch_index;
    epoch_atom.store(epoch, std::sync::atomic::Ordering::Relaxed);
    let block_hash = chain
        .block_hash(state.last_epoch_block)
        .map_err(|e| format!("block_hash: {e}"))?;
    let expected = expected_set_at_chain(
        &ParticipantPolicy::AllMetagraphHotkeys,
        PinnedBlockHash::new(block_hash),
        chain,
    )
    .map_err(|e| format!("expected set: {e}"))?;
    emit_epoch(Arc::clone(store), gateway, sk, epoch, &expected)
        .await
        .map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let cli = Cli::parse();
    // Clap bool env only accepts true/false; also honor 1/yes like design/prism.
    let force_memory = cli.force_memory || env_truthy("BOUNTY_FORCE_MEMORY");
    let force_sim = cli.force_sim || env_truthy("BOUNTY_FORCE_SIM");

    if let Err(e) = tokio::fs::create_dir_all(&cli.artifacts_root).await {
        tracing::error!(error = %e, path = %cli.artifacts_root.display(), "artifacts root");
        return ExitCode::FAILURE;
    }

    let (store, store_mode): (Arc<dyn bounty_challenge::BountyStore>, String) = if force_memory {
        (Arc::new(MemoryBountyStore::new()), "memory".into())
    } else if let Ok(url) = std::env::var("BASE_DATABASE_URL") {
        if url.trim().is_empty() {
            (Arc::new(MemoryBountyStore::new()), "memory".into())
        } else {
            match db::connect(&url).await {
                Ok(pool) => {
                    if let Err(e) = db::migrate(&pool).await {
                        tracing::error!(error = %e, "db migrate failed");
                        return ExitCode::FAILURE;
                    }
                    (
                        Arc::new(DbBountyStore::new(pool))
                            as Arc<dyn bounty_challenge::BountyStore>,
                        "postgres".into(),
                    )
                }
                Err(e) => {
                    tracing::error!(error = %e, "db connect failed");
                    return ExitCode::FAILURE;
                }
            }
        }
    } else {
        (Arc::new(MemoryBountyStore::new()), "memory".into())
    };

    let (agentic, agentic_mode) = build_agentic(force_sim, &cli.openrouter_key_file);
    let admin_hashes = load_token_hashes(cli.admin_tokens_file.as_ref());

    let gating_enabled = !matches!(std::env::var("BASE_SUBMISSION_GATING").as_deref(), Ok("0"));
    if !gating_enabled {
        tracing::warn!("BASE_SUBMISSION_GATING=0: intake metagraph checks disabled");
    }

    let metagraph = Arc::new(MetagraphCache::new());
    let gating: Arc<dyn GatingStore> = if store_mode == "postgres" {
        if let Ok(url) = std::env::var("BASE_DATABASE_URL") {
            match db::connect(&url).await {
                Ok(pool) => Arc::new(PgGatingStore::new(pool)),
                Err(_) => Arc::new(MemoryGatingStore::new()),
            }
        } else {
            Arc::new(MemoryGatingStore::new())
        }
    } else {
        Arc::new(MemoryGatingStore::new())
    };

    let epoch = Arc::new(AtomicU64::new(0));
    let backend_mode = format!("{store_mode}/{agentic_mode}");

    let state = Arc::new(AppState {
        store: Arc::clone(&store),
        backend_mode: backend_mode.clone(),
        artifacts_root: cli.artifacts_root.clone(),
        admin_token_hashes: admin_hashes,
        metagraph: gating_enabled.then(|| Arc::clone(&metagraph)),
        epoch: Arc::clone(&epoch),
        max_upload_bytes: cli.max_upload_bytes,
    });

    let gateway = match GatewayClient::new(GatewayClientConfig {
        base_url: if force_sim {
            DRY_RUN_BASE_URL.to_owned()
        } else {
            cli.gateway_endpoint.clone()
        },
        ..Default::default()
    }) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!(error = %e, "gateway client");
            return ExitCode::FAILURE;
        }
    };

    if gating_enabled {
        if let Some(ep) = cli.chain_endpoint.as_deref().filter(|s| !s.is_empty()) {
            spawn_gating_watcher(
                ep,
                cli.netuid,
                Arc::clone(&metagraph),
                Arc::clone(&gating),
                Duration::from_secs(cli.gating_watch_secs.max(15)),
            );
        }
    }

    let orch = Arc::new(Orchestrator::new(
        Arc::clone(&store),
        OrchestratorConfig {
            max_concurrent: cli.max_concurrent.max(1),
            force_sim,
            artifacts_root: cli.artifacts_root.clone(),
            ..OrchestratorConfig::default()
        },
        agentic,
    ));
    let permits = Arc::new(Semaphore::new(cli.max_concurrent.max(1) as usize));
    for _ in 0..cli.max_concurrent.max(1) {
        tokio::spawn(Arc::clone(&orch).run_worker(Arc::clone(&permits)));
    }

    match cli.challenge_sk_file.as_ref() {
        Some(p) if p.is_file() => match load_challenge_secret(p) {
            Ok(sk) => {
                if let Some(ep) = cli.chain_endpoint.clone().filter(|s| !s.is_empty()) {
                    spawn_emitter(
                        Arc::clone(&store),
                        gateway,
                        sk,
                        ep,
                        cli.netuid,
                        Arc::clone(&epoch),
                        Duration::from_secs(cli.emit_poll_secs.max(5)),
                    );
                } else {
                    tracing::warn!("challenge sk present but no BASE_CHAIN_ENDPOINT; emitter idle");
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "challenge sk load failed");
                return ExitCode::FAILURE;
            }
        },
        _ => tracing::warn!("no BASE_CHALLENGE_SK_FILE; leaf emitter disabled"),
    }

    let app = bounty_router(state);
    let listener = match TcpListener::bind(cli.bind).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, bind = %cli.bind, "bind failed");
            return ExitCode::FAILURE;
        }
    };
    tracing::info!(
        bind = %cli.bind,
        challenge_id = CHALLENGE_ID,
        scoring_version = SCORING_VERSION,
        backend = %backend_mode,
        default_model = DEFAULT_OPENROUTER_MODEL,
        "bounty-challenge listening"
    );

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        tracing::error!(error = %e, "server error");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
