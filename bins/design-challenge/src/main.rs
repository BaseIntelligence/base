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
use design_challenge_bin::resanitize;
use design_challenge::{
    design_router, force_sim_refusal_reason, host_sim_allowed, public_key_from_secret, AppState,
    DbDesignStore, DesignStore, GatewayClient, GatewayClientConfig, MemoryDesignStore,
    Orchestrator, OrchestratorConfig, CHALLENGE_ID, SCORING_VERSION,
};
use design_sandbox::{DockerSandbox, SandboxBackend, SimSandbox};
use review_docker::{DockerAgent, DockerAgentConfig};
use sha2::{Digest, Sha256};
use submission_gating::{
    watch_once, GatingStore, MemoryGatingStore, MetagraphCache, PgGatingStore,
};
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
    /// Local/e2e: ms to hold after each published stage (photograph mid-flight).
    /// Default 0 — never enable on staging/prod.
    #[arg(
        long,
        env = "DESIGN_SIM_STAGE_DELAY_MS",
        default_value_t = 0,
        global = true
    )]
    sim_stage_delay_ms: u64,
    /// Review backend: `docker` (containerized design-review image) or
    /// `inline` (in-process agent; local/CI).
    #[arg(
        long,
        env = "DESIGN_REVIEW_BACKEND",
        default_value = "docker",
        global = true
    )]
    review_backend: String,
    /// Review image ref (digest-pinned in deploy).
    #[arg(
        long,
        env = "DESIGN_REVIEW_IMAGE",
        default_value = "design-review:0.1.0",
        global = true
    )]
    review_image: String,
    /// Auto-retry budget for infra-class run failures (install/AST/LLM).
    #[arg(
        long,
        env = "DESIGN_AUTO_RETRY_MAX",
        default_value_t = 3,
        global = true
    )]
    auto_retry_max: u32,
    /// Install-phase timeout in seconds (`pip install` from `pyproject.toml`).
    #[arg(
        long,
        env = "DESIGN_INSTALL_TIMEOUT_SECS",
        default_value_t = design_sandbox::DEFAULT_INSTALL_TIMEOUT_SECS,
        global = true
    )]
    install_timeout_secs: u64,
    /// Metagraph watcher cadence (gating eligibility resets).
    #[arg(
        long,
        env = "BASE_GATING_WATCH_SECS",
        default_value_t = 120,
        global = true
    )]
    gating_watch_secs: u64,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print identity.
    Identity,
    /// Run server + workers.
    Serve,
    /// (Re)capture missing run screenshots (`index.png`) against Postgres,
    /// then exit. Idempotent: runs that already have a screenshot are skipped.
    BackfillScreenshots {
        /// Scan at most this many recent runs (newest first).
        #[arg(long, env = "DESIGN_BACKFILL_LIMIT", default_value_t = 500)]
        limit: u32,
    },
    /// Re-sanitize from stored `raw_html` (current sanitizer), then force
    /// re-capture `index.png`. Repairs runs where an older sanitizer wiped
    /// `<style>` (existing `backfill-screenshots` cannot — it only re-renders
    /// already-sanitized HTML and skips runs that already have a PNG).
    BackfillResanitize {
        /// Scan at most this many recent runs (newest first). Ignored when
        /// `--run-id` is set.
        #[arg(long, env = "DESIGN_BACKFILL_LIMIT", default_value_t = 500)]
        limit: u32,
        /// Specific run id(s) to repair (repeatable). When set, skips the
        /// newest-N scan.
        #[arg(long = "run-id")]
        run_ids: Vec<String>,
        /// Sleep between Chromium captures so live rounds are not starved.
        #[arg(long, env = "DESIGN_BACKFILL_SLEEP_MS", default_value_t = 2_000)]
        sleep_ms: u64,
        /// List candidates / validate sanitize restore without writing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let mut cli = Cli::parse();
    // Ordered failover list wins over the single-endpoint flag/env.
    if let Ok(list) = std::env::var("BASE_CHAIN_ENDPOINTS") {
        if !list.trim().is_empty() {
            cli.chain_endpoint = list;
        }
    }
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
    match &cli.cmd {
        Some(Cmd::BackfillScreenshots { limit }) => {
            return rt.block_on(cmd_backfill_screenshots(&cli, *limit));
        }
        Some(Cmd::BackfillResanitize {
            limit,
            run_ids,
            sleep_ms,
            dry_run,
        }) => {
            return rt.block_on(cmd_backfill_resanitize(
                &cli,
                *limit,
                run_ids.clone(),
                *sleep_ms,
                *dry_run,
            ));
        }
        _ => {}
    }
    rt.block_on(cmd_serve(cli))
}

/// One-shot screenshot backfill: Postgres store + headless Chromium only (no
/// chain, gateway, or challenge key needed).
async fn cmd_backfill_screenshots(cli: &Cli, limit: u32) -> Result<(), String> {
    let url = std::env::var("BASE_DATABASE_URL")
        .map_err(|_| "backfill-screenshots requires BASE_DATABASE_URL".to_owned())?;
    let pool = db::connect(&url).await.map_err(|e| e.to_string())?;
    let store = DbDesignStore::new(pool);
    let s =
        design_challenge::backfill::backfill_screenshots(&store, &cli.staging_root, limit).await?;
    println!(
        "screenshot backfill: scanned={} missing={} captured={} failed={}",
        s.scanned, s.missing, s.captured, s.failed
    );
    if s.failed > 0 {
        return Err(format!(
            "{} screenshot capture(s) failed; re-run to retry",
            s.failed
        ));
    }
    Ok(())
}

/// Re-sanitize from `raw_html` + force screenshot (see [`Cmd::BackfillResanitize`]).
async fn cmd_backfill_resanitize(
    cli: &Cli,
    limit: u32,
    run_ids: Vec<String>,
    sleep_ms: u64,
    dry_run: bool,
) -> Result<(), String> {
    let url = std::env::var("BASE_DATABASE_URL")
        .map_err(|_| "backfill-resanitize requires BASE_DATABASE_URL".to_owned())?;
    let pool = db::connect(&url).await.map_err(|e| e.to_string())?;
    let store = DbDesignStore::new(pool);
    let s = resanitize::backfill_resanitize(
        &store,
        &cli.staging_root,
        limit,
        &run_ids,
        sleep_ms,
        dry_run,
    )
    .await?;
    println!(
        "resanitize backfill: scanned={} candidates={} resanitized={} screenshots={} failed={} skipped={} dry_run={dry_run}",
        s.scanned, s.candidates, s.resanitized, s.screenshots, s.failed, s.skipped
    );
    if s.failed > 0 {
        return Err(format!(
            "{} resanitize/capture failure(s); re-run to retry",
            s.failed
        ));
    }
    Ok(())
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

/// Review backend: containerized `design-review` image (default, same Docker
/// pattern as the sandbox) or the legacy in-process agent (`inline`).
fn select_review(cli: &Cli) -> Result<(Arc<dyn AgenticBackend>, &'static str), String> {
    if cli.review_backend != "docker" {
        let (agentic, _) = select_agentic(cli);
        return Ok((agentic, "inline"));
    }
    let key = load_api_key_file(&cli.agentic_openrouter_key_file).ok();
    if key.is_none() {
        tracing::info!(
            path = %cli.agentic_openrouter_key_file.display(),
            "no OpenRouter key; review containers will run SimAgent offline"
        );
    }
    let agent = DockerAgent::new(DockerAgentConfig {
        docker_base: cli.docker_base.clone(),
        image: cli.review_image.clone(),
        openrouter_key: key,
        ..DockerAgentConfig::default()
    })
    .map_err(|e| e.to_string())?;
    // Fail closed at boot when the engine is unreachable (no silent inline).
    if let Err(e) = agent.client.list_owned() {
        return Err(format!(
            "docker review unavailable (fail-closed; set DESIGN_REVIEW_BACKEND=inline \
             only on local/CI): {e}"
        ));
    }
    Ok((Arc::new(agent), "docker"))
}

/// Spawn the metagraph watcher: refresh the cached snapshot and reconcile
/// gating eligibility (hotkey deregistered/replaced → back to `open`).
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
            if let Some(c) = &client {
                if let Err(e) = watch_once(c, netuid, &cache, &stores).await {
                    tracing::warn!(error = %e, "gating watcher tick failed (cache kept)");
                    client = None; // reconnect next tick on persistent errors
                }
            }
            tokio::time::sleep(poll).await;
        }
    });
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
        Ok(mut docker) => {
            docker.install_timeout_sec = cli.install_timeout_secs;
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

/// Orchestrator + shared HTTP state (award hook wires back into the orchestrator).
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn build_app_state(
    cli: &Cli,
    store: Arc<dyn DesignStore>,
    gating: Arc<dyn GatingStore>,
    gating_enabled: bool,
    sandbox: Arc<dyn SandboxBackend>,
    agentic: Arc<dyn AgenticBackend>,
    gateway: Arc<GatewayClient>,
    chain: chain_live::LiveChainClient,
    sk: [u8; crypto::KEY_LEN],
    backend_mode: &'static str,
    stage_delay: Duration,
) -> (
    Arc<AppState>,
    Arc<Orchestrator<chain_live::LiveChainClient>>,
) {
    let mut orch = Orchestrator::new(
        OrchestratorConfig {
            netuid: cli.netuid,
            claim_poll: Duration::from_millis(750),
            stuck_grace_secs: 3600,
            llm_proxy: cli.llm_proxy.clone(),
            staging_root: cli.staging_root.clone(),
            stage_delay,
            auto_retry_max: cli.auto_retry_max,
        },
        Arc::clone(&store),
        sandbox,
        agentic,
        gateway,
        Arc::new(chain),
        sk,
    );
    if gating_enabled {
        orch = orch.with_gating(Arc::clone(&gating));
    }
    let orch = Arc::new(orch);

    // Metagraph cache + watcher feed the intake membership check.
    let metagraph = Arc::new(MetagraphCache::new());
    if gating_enabled {
        spawn_gating_watcher(
            &cli.chain_endpoint,
            cli.netuid,
            Arc::clone(&metagraph),
            Arc::clone(&gating),
            Duration::from_secs(cli.gating_watch_secs.max(15)),
        );
    }

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
            .unwrap_or_else(|_| design_sanitize::default_frame_ancestors().into()),
        retry_max: 2,
        award_hook: Some(Arc::clone(&orch) as Arc<dyn design_challenge::AdminAwardHook>),
        gating: gating_enabled.then(|| Arc::clone(&gating)),
        metagraph: gating_enabled.then(|| Arc::clone(&metagraph)),
    });
    (state, orch)
}

async fn build_stores() -> Result<(Arc<dyn DesignStore>, Arc<dyn GatingStore>), String> {
    if let Ok(url) = std::env::var("BASE_DATABASE_URL") {
        let pool = db::connect(&url).await.map_err(|e| e.to_string())?;
        db::migrate(&pool).await.map_err(|e| e.to_string())?;
        return Ok((
            Arc::new(DbDesignStore::new(pool.clone())),
            Arc::new(PgGatingStore::new(pool)) as Arc<dyn GatingStore>,
        ));
    }
    tracing::warn!("no BASE_DATABASE_URL; using MemoryDesignStore + MemoryGatingStore");
    Ok((
        Arc::new(MemoryDesignStore::new()),
        Arc::new(MemoryGatingStore::new()) as Arc<dyn GatingStore>,
    ))
}

async fn cmd_serve(cli: Cli) -> Result<(), String> {
    let sk_path = resolve_sk_path(cli.challenge_sk_file.as_ref())?;
    let sk = load_challenge_secret(&sk_path).map_err(|e| e.to_string())?;

    let (store, gating) = build_stores().await?;
    // BASE_SUBMISSION_GATING=0 disables intake gating (local dev only).
    let gating_enabled = !matches!(std::env::var("BASE_SUBMISSION_GATING").as_deref(), Ok("0"));
    if !gating_enabled {
        tracing::warn!("BASE_SUBMISSION_GATING=0: intake metagraph/1-max checks disabled");
    }

    let (sandbox, backend_mode) = select_sandbox(&cli)?;
    let (agentic, agentic_mode) = select_review(&cli)?;

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

    let stage_delay = Duration::from_millis(cli.sim_stage_delay_ms);
    if !stage_delay.is_zero() {
        tracing::info!(
            delay_ms = cli.sim_stage_delay_ms,
            "design sim stage delay enabled (local evidence only)"
        );
    }
    let (state, orch) = build_app_state(
        &cli,
        store,
        gating,
        gating_enabled,
        sandbox,
        agentic,
        gateway,
        chain,
        sk,
        backend_mode,
        stage_delay,
    );
    tracing::info!(
        backend_mode,
        agentic_mode,
        gating_enabled,
        "design backends selected"
    );

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
