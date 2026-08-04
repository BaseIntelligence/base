//! `design-challenge` — operator-side design challenge service (:8093).

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use challenge_agentic::{load_api_key_file, AgenticBackend, OpenRouterAgent, SimAgent};
use challenge_keys::load_challenge_secret;
use clap::{Parser, Subcommand};
use design_challenge::{
    design_router, force_sim_refusal_reason, host_sim_allowed, public_key_from_secret, AppState,
    DbDesignStore, DesignStore, GatewayClient, GatewayClientConfig, MemoryDesignStore,
    Orchestrator, OrchestratorConfig, CHALLENGE_ID, SCORING_VERSION,
};
use design_sandbox::{DockerSandbox, SandboxBackend, SimSandbox};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use trustroot::encode_hex;

/// Design challenge CLI.
#[derive(Debug, Parser)]
#[command(
    name = "design-challenge",
    about = "Design challenge service (port 8093)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Bind address.
    #[arg(
        long,
        env = "BASE_CHALLENGE_BIND",
        default_value = "0.0.0.0:8093",
        global = true
    )]
    bind: SocketAddr,
    /// Challenge mini-secret file.
    #[arg(long, env = "BASE_CHALLENGE_SK_FILE", global = true)]
    challenge_sk_file: Option<PathBuf>,
    /// Force host Sim sandbox (requires `BASE_ALLOW_HOST_SIM=1` and non-prod).
    #[arg(long, env = "DESIGN_FORCE_SIM", default_value_t = false, global = true)]
    force_sim: bool,
    /// Netuid.
    #[arg(long, env = "BASE_NETUID", default_value_t = 1, global = true)]
    netuid: u16,
    /// Max concurrent sandbox runs.
    #[arg(
        long,
        env = "DESIGN_MAX_CONCURRENT",
        default_value_t = 2,
        global = true
    )]
    max_concurrent: u32,
    /// Chain WS endpoint.
    #[arg(
        long,
        env = "BASE_CHAIN_ENDPOINT",
        default_value = "wss://test.finney.opentensor.ai:443",
        global = true
    )]
    chain_endpoint: String,
    /// Gateway base URL.
    #[arg(
        long,
        env = "BASE_CHALLENGE_GATEWAY_ENDPOINT",
        default_value = "http://gateway:8080",
        global = true
    )]
    gateway_endpoint: String,
    /// Annotator tokens file (one token per line); hashed at boot.
    #[arg(long, env = "DESIGN_ANNOTATOR_TOKENS_FILE", global = true)]
    annotator_tokens_file: Option<PathBuf>,
    /// Admin bearer tokens file (falls back to annotator tokens when unset).
    #[arg(long, env = "DESIGN_ADMIN_TOKENS_FILE", global = true)]
    admin_tokens_file: Option<PathBuf>,
    /// `OpenRouter` key file for agentic review (`SimAgent` when missing).
    #[arg(
        long,
        env = "DESIGN_AGENTIC_OPENROUTER_KEY_FILE",
        default_value = "/run/base/openrouter/api_key",
        global = true
    )]
    agentic_openrouter_key_file: PathBuf,
    /// Force `SimAgent` for agentic review (CI).
    #[arg(
        long,
        env = "DESIGN_FORCE_AGENTIC_SIM",
        default_value_t = false,
        global = true
    )]
    force_agentic_sim: bool,
    /// Docker engine base URL.
    #[arg(
        long,
        env = "DESIGN_DOCKER_BASE",
        default_value = "http://socket-proxy:2375",
        global = true
    )]
    docker_base: String,
    /// Staging root for sandbox work dirs.
    #[arg(
        long,
        env = "DESIGN_STAGING_ROOT",
        default_value = "/var/lib/design/staging",
        global = true
    )]
    staging_root: PathBuf,
    /// LLM proxy URL.
    #[arg(
        long,
        env = "DESIGN_LLM_PROXY",
        default_value = "http://design-egress-proxy:8094",
        global = true
    )]
    llm_proxy: String,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print identity.
    Identity,
    /// Run server + workers.
    Serve,
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("design-challenge: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    if matches!(cli.cmd, Some(Cmd::Identity)) {
        return cmd_identity(&resolve_sk_path(cli.challenge_sk_file.as_ref())?);
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(cmd_serve(cli))
}

fn resolve_sk_path(cli_path: Option<&PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = cli_path {
        return Ok(p.clone());
    }
    if let Some(p) = std::env::var_os("DESIGN_CHALLENGE_SK_FILE") {
        return Ok(PathBuf::from(p));
    }
    Err("BASE_CHALLENGE_SK_FILE or DESIGN_CHALLENGE_SK_FILE required".into())
}

fn cmd_identity(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("challenge secret file missing: {}", path.display()));
    }
    let sk = load_challenge_secret(path).map_err(|e| e.to_string())?;
    let pk = public_key_from_secret(&sk)?;
    println!("challenge_id={CHALLENGE_ID}");
    println!("scoring_version={SCORING_VERSION}");
    println!("public_key={}", encode_hex(&pk));
    Ok(())
}

fn load_token_hashes(path: Option<&PathBuf>) -> Vec<String> {
    let Some(p) = path else {
        return vec![];
    };
    let Ok(text) = std::fs::read_to_string(p) else {
        return vec![];
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

fn select_agentic(cli: &Cli) -> (Arc<dyn AgenticBackend>, &'static str) {
    if cli.force_agentic_sim || env_truthy("DESIGN_FORCE_AGENTIC_SIM") {
        return (Arc::new(SimAgent::new()), "sim");
    }
    let Ok(key) = load_api_key_file(&cli.agentic_openrouter_key_file) else {
        tracing::info!(
            path = %cli.agentic_openrouter_key_file.display(),
            "no agentic OpenRouter key; using SimAgent"
        );
        return (Arc::new(SimAgent::new()), "sim");
    };
    match OpenRouterAgent::new(key) {
        Ok(agent) => (Arc::new(agent), "openrouter"),
        Err(e) => {
            tracing::warn!(error = %e, "OpenRouter agent init failed; using SimAgent");
            (Arc::new(SimAgent::new()), "sim")
        }
    }
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn deploy_env() -> Option<String> {
    std::env::var("BASE_DEPLOY_ENV").ok()
}

/// Host `SimSandbox` is allowed only with an explicit non-prod opt-in.
fn host_sim_ok(netuid: u16) -> bool {
    host_sim_allowed(
        netuid,
        env_truthy("BASE_ALLOW_HOST_SIM"),
        deploy_env().as_deref(),
    )
}

fn select_sandbox(cli: &Cli) -> Result<(Arc<dyn SandboxBackend>, &'static str), String> {
    let want_sim = cli.force_sim || env_truthy("DESIGN_FORCE_SIM");
    if want_sim {
        if !host_sim_ok(cli.netuid) {
            return Err(force_sim_refusal_reason().into());
        }
        return Ok((Arc::new(SimSandbox::new()), "sim"));
    }

    match DockerSandbox::new(&cli.docker_base, cli.staging_root.clone()) {
        Ok(docker) => {
            // Fail closed at boot if the engine/proxy is unreachable — no silent Sim.
            if let Err(e) = docker.client.list_owned() {
                if host_sim_ok(cli.netuid) {
                    tracing::warn!(
                        error = %e,
                        "docker sandbox unreachable; BASE_ALLOW_HOST_SIM=1 → SimSandbox"
                    );
                    return Ok((Arc::new(SimSandbox::new()), "sim"));
                }
                return Err(format!(
                    "docker sandbox unavailable (fail-closed; set BASE_ALLOW_HOST_SIM=1 \
                     only on non-prod/CI): {e}"
                ));
            }
            Ok((Arc::new(docker), "docker"))
        }
        Err(e) => {
            if host_sim_ok(cli.netuid) {
                tracing::warn!(
                    error = %e,
                    "docker sandbox init failed; BASE_ALLOW_HOST_SIM=1 → SimSandbox"
                );
                Ok((Arc::new(SimSandbox::new()), "sim"))
            } else {
                Err(format!(
                    "docker sandbox unavailable (fail-closed; set BASE_ALLOW_HOST_SIM=1 \
                     only on non-prod/CI): {e}"
                ))
            }
        }
    }
}

async fn cmd_serve(cli: Cli) -> Result<(), String> {
    let sk_path = resolve_sk_path(cli.challenge_sk_file.as_ref())?;
    let sk = load_challenge_secret(&sk_path).map_err(|e| e.to_string())?;

    let store: Arc<dyn DesignStore> = if let Ok(url) = std::env::var("BASE_DATABASE_URL") {
        let pool = db::connect(&url).await.map_err(|e| e.to_string())?;
        db::migrate(&pool).await.map_err(|e| e.to_string())?;
        Arc::new(DbDesignStore::new(pool))
    } else {
        tracing::warn!("no BASE_DATABASE_URL; using MemoryDesignStore");
        Arc::new(MemoryDesignStore::new())
    };

    let (sandbox, backend_mode) = select_sandbox(&cli)?;
    let (agentic, agentic_mode) = select_agentic(&cli);

    let gateway = Arc::new(
        GatewayClient::new(GatewayClientConfig {
            base_url: cli.gateway_endpoint.clone(),
            ..GatewayClientConfig::default()
        })
        .map_err(|e| e.to_string())?,
    );

    let mut chain =
        chain_live::LiveChainClient::connect(&cli.chain_endpoint).map_err(|e| e.to_string())?;
    chain.set_netuid(cli.netuid);

    let orch = Arc::new(Orchestrator::new(
        OrchestratorConfig {
            netuid: cli.netuid,
            claim_poll: Duration::from_millis(750),
            stuck_grace_secs: 3600,
            llm_proxy: cli.llm_proxy.clone(),
            staging_root: cli.staging_root.clone(),
        },
        Arc::clone(&store),
        sandbox,
        agentic,
        gateway,
        Arc::new(chain),
        sk,
    ));

    let annotator_hashes = load_token_hashes(cli.annotator_tokens_file.as_ref());
    let admin_hashes = load_token_hashes(cli.admin_tokens_file.as_ref());
    let state = Arc::new(AppState {
        store,
        epoch: AtomicU64::new(0),
        netuid: cli.netuid,
        backend_mode,
        annotator_token_hashes: annotator_hashes,
        admin_token_hashes: admin_hashes,
        frame_ancestors: std::env::var("DESIGN_FRAME_ANCESTORS")
            .unwrap_or_else(|_| "'none'".into()),
        retry_max: 2,
        award_hook: Some(Arc::clone(&orch) as Arc<dyn design_challenge::AdminAwardHook>),
    });
    tracing::info!(backend_mode, agentic_mode, "design backends selected");

    let permits = cli.max_concurrent.max(1) as usize;
    let sem = Arc::new(Semaphore::new(permits));
    for i in 0..permits {
        let o = Arc::clone(&orch);
        let s = Arc::clone(&sem);
        tokio::spawn(async move {
            let _permit = s.acquire_owned().await.ok();
            tracing::info!(worker = i, "design worker up");
            o.run_worker().await;
        });
    }
    tokio::spawn(Arc::clone(&orch).run_round_loop());
    tokio::spawn(Arc::clone(&orch).run_sweeper());

    let app = design_router(state);
    let listener = TcpListener::bind(cli.bind)
        .await
        .map_err(|e| e.to_string())?;
    tracing::info!(%cli.bind, backend_mode, "design-challenge listening");
    axum::serve(listener, app)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}
