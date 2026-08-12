//! `prism-challenge` — operator-side PRISM challenge service.
//!
//! API: miner submit + full progression surface (AGENT-style list/detail/
//! status/jobs/recipe). Orchestrator: DB-backed Lium job runner + master-side
//! LLM review/similarity + agentic anti-cheat + epoch-close D24 leaf emitter.
//! **No Phala CVM.**
//!
//! Backend selection (fail-closed concept, never invented):
//! - Lium backend: `LIUM_API_KEY` (or credentials file) present and
//!   `PRISM_FORCE_SIM=false` → real pods; otherwise Sim.
//! - LLM reviewer / agentic: `OPENROUTER_API_KEY_FILE` present → real
//!   `OpenRouter`; otherwise `SimReviewer` / `SimAgent` (deterministic).
//! - Store: `BASE_DATABASE_URL` present → Postgres (`prism_submission` +
//!   `prism_stage_event`); otherwise in-memory (dev/sim only).

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use challenge_agentic::{AgenticBackend, OpenRouterAgent, SimAgent};
use challenge_keys::load_challenge_secret;
use clap::{Parser, Subcommand};
use prism_challenge::{
    submission_router, AppState, DbEvalStore, DbPrismStore, EvalStore, MemoryEvalStore,
    MemoryPrismStore, Orchestrator, OrchestratorConfig, PrismStore, CHALLENGE_ID, SCORING_VERSION,
};
use prism_lium::{EvalJobBackend, LiumClient, LiumSshConfig, SimLiumBackend};
use prism_lium_payer::{allow_operator_lium_fallback, PayerBackendFactory, PayerKeyVault};
use prism_review::{OpenRouterClient, ReviewBackend, SimReviewer};
use submission_gating::{
    watch_once, GatingStore, MemoryGatingStore, MetagraphCache, PgGatingStore,
};
/// Manual `/retry` ceiling; keep ≥ `PRISM_AUTO_RETRY_MAX` so infra retries work.
const MAX_ATTEMPTS: u32 = 3;

use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use trustroot::encode_hex;

/// Operator PRISM challenge service CLI.
#[derive(Debug, Parser)]
#[command(
    name = "prism-challenge",
    about = "PRISM challenge service (port 8092, master→Lium, orchestrated, LLM-reviewed)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Bind address for HTTP (default 0.0.0.0:8092).
    #[arg(
        long,
        env = "BASE_CHALLENGE_BIND",
        default_value = "0.0.0.0:8092",
        global = true
    )]
    bind: SocketAddr,
    /// Challenge mini-secret file.
    #[arg(long, env = "BASE_CHALLENGE_SK_FILE", global = true)]
    challenge_sk_file: Option<PathBuf>,
    /// Force Sim backends (local dev).
    #[arg(long, env = "PRISM_FORCE_SIM", default_value_t = false, global = true)]
    force_sim: bool,
    /// Netuid.
    #[arg(long, env = "BASE_NETUID", default_value_t = 1, global = true)]
    netuid: u16,
    /// Max concurrent Lium evals (workers).
    #[arg(
        long,
        env = "PRISM_MAX_CONCURRENT_EVALS",
        default_value_t = 8,
        global = true
    )]
    max_concurrent_evals: u32,
    /// Chain WS endpoint (epoch/E resolution).
    #[arg(long, env = "BASE_CHAIN_ENDPOINT", global = true)]
    chain_endpoint: Option<String>,
    /// Gateway base URL for leaf submit.
    #[arg(
        long,
        env = "BASE_CHALLENGE_GATEWAY_ENDPOINT",
        default_value = "http://gateway:8080",
        global = true
    )]
    gateway_endpoint: String,
    /// Local/e2e: ms to hold after each published stage (photograph mid-flight).
    /// Default 0 — never enable on staging/prod.
    #[arg(
        long,
        env = "PRISM_SIM_STAGE_DELAY_MS",
        default_value_t = 0,
        global = true
    )]
    sim_stage_delay_ms: u64,
    /// Auto-retry budget for infra-class failures (Lium install/AST/LLM).
    #[arg(long, env = "PRISM_AUTO_RETRY_MAX", default_value_t = 3, global = true)]
    auto_retry_max: u32,
    /// Metagraph watcher cadence (gating eligibility resets).
    #[arg(
        long,
        env = "BASE_GATING_WATCH_SECS",
        default_value_t = 120,
        global = true
    )]
    gating_watch_secs: u64,
    /// Operator bearer tokens file. Unset → retry/admin/playground answer 503.
    #[arg(long, env = "PRISM_ADMIN_TOKENS_FILE", global = true)]
    admin_tokens_file: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print identity.
    Identity,
    /// Run server + workers (default).
    Serve,
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let mut cli = Cli::parse();
    // Ordered failover list wins over the single-endpoint flag/env.
    if let Ok(list) = std::env::var("BASE_CHAIN_ENDPOINTS") {
        if !list.trim().is_empty() {
            cli.chain_endpoint = Some(list);
        }
    }
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("prism-challenge: {msg}");
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
    if let Some(p) = std::env::var_os("PRISM_CHALLENGE_SK_FILE") {
        return Ok(PathBuf::from(p));
    }
    Err("BASE_CHALLENGE_SK_FILE or PRISM_CHALLENGE_SK_FILE required".into())
}

fn cmd_identity(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("challenge secret file missing: {}", path.display()));
    }
    let sk = load_challenge_secret(path).map_err(|e| e.to_string())?;
    let pk = prism_challenge::public_key_from_secret(&sk)?;
    println!("challenge_id={CHALLENGE_ID}");
    println!("scoring_version={SCORING_VERSION}");
    println!("public_key={}", encode_hex(&pk));
    Ok(())
}

/// Resolve the Lium API key (file/env/credentials — never logged).
fn load_lium_api_key() -> Option<String> {
    if let Ok(path) = std::env::var("LIUM_API_KEY_FILE") {
        if let Ok(text) = std::fs::read_to_string(PathBuf::from(path)) {
            let t = text.trim().to_owned();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    if let Ok(k) = std::env::var("LIUM_API_KEY") {
        let t = k.trim().to_owned();
        if !t.is_empty() {
            return Some(t);
        }
    }
    let cred = std::env::var("LIUM_CREDENTIALS_FILE")
        .unwrap_or_else(|_| "/root/.config/prism-mission/credentials.env".to_owned());
    let text = std::fs::read_to_string(PathBuf::from(cred)).ok()?;
    for line in text.lines() {
        let line = line.trim().strip_prefix("export ").unwrap_or(line.trim());
        if let Some(rest) = line.strip_prefix("LIUM_API_KEY=") {
            let v = rest.trim().trim_matches('"').trim_matches('\'');
            if !v.is_empty() {
                return Some(v.to_owned());
            }
        }
    }
    None
}

fn load_ssh_public_key() -> Option<String> {
    let p = std::env::var("LIUM_SSH_PUBLIC_KEY_FILE").map_or_else(
        |_| PathBuf::from("/root/.config/prism-mission/lium_ssh_ed25519.pub"),
        PathBuf::from,
    );
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

type BackendBundle = (
    Arc<dyn EvalJobBackend>,
    String,
    Vec<String>,
    LiumSshConfig,
    Option<String>,
);

fn live_ssh_config() -> LiumSshConfig {
    let mut ssh = LiumSshConfig::default_live();
    if let Ok(v) = std::env::var("PRISM_SSH_ATTEMPTS") {
        ssh.ssh_attempts = v.parse().unwrap_or(ssh.ssh_attempts);
    }
    if let Ok(v) = std::env::var("PRISM_SSH_RETRY_SECS") {
        ssh.ssh_retry_secs = v.parse().unwrap_or(ssh.ssh_retry_secs);
    }
    if let Ok(v) = std::env::var("PRISM_SSH_RUNNING_TIMEOUT_SECS") {
        ssh.running_timeout_secs = v.parse().unwrap_or(ssh.running_timeout_secs);
    }
    if let Ok(p) = std::env::var("LIUM_SSH_PRIVATE_KEY") {
        ssh.private_key_path = Some(PathBuf::from(p));
    } else {
        let default = PathBuf::from("/root/.config/prism-mission/lium_ssh_ed25519");
        if default.is_file() {
            ssh.private_key_path = Some(default);
        }
    }
    ssh
}

fn build_backend(force_sim: bool) -> Result<BackendBundle, String> {
    let ssh = live_ssh_config();
    // Live path: SSH must exist so miner-funded pods can still be operated by master.
    // Operator LIUM_API_KEY is optional when miners bring X-Lium-Api-Key (BYOK).
    if force_sim {
        return Ok((
            Arc::new(SimLiumBackend::new()),
            "sim".into(),
            vec![],
            ssh,
            None,
        ));
    }
    let pk = load_ssh_public_key()
        .ok_or("live Lium requires SSH public key (LIUM_SSH_PUBLIC_KEY_FILE or default)")?;
    if let Some(api_key) = load_lium_api_key() {
        let client = LiumClient::with_config(
            api_key.clone(),
            prism_challenge::LIUM_API_BASE_URL,
            ssh.clone(),
        )
        .map_err(|e| e.to_string())?;
        return Ok((
            Arc::new(client),
            "lium".into(),
            vec![pk],
            ssh,
            Some(api_key),
        ));
    }
    // Miner-BYOK only: placeholder backend is unused when the vault supplies a key.
    tracing::info!("lium mode: miner-funded only (no operator LIUM_API_KEY)");
    Ok((
        Arc::new(SimLiumBackend::new()),
        "lium".into(),
        vec![pk],
        ssh,
        None,
    ))
}

fn openrouter_key_path() -> PathBuf {
    let path = std::env::var("OPENROUTER_API_KEY_FILE").map(PathBuf::from);
    match path {
        Ok(p) if p.is_file() => p,
        _ => PathBuf::from("/run/base/openrouter/api_key"),
    }
}

/// Reviewer from `OPENROUTER_API_KEY_FILE`, else sim (deterministic).
#[must_use]
pub(crate) fn build_reviewer() -> (Arc<dyn ReviewBackend>, &'static str) {
    let key_path = openrouter_key_path();
    if let Ok(key) = prism_review::load_api_key_file(&key_path) {
        if let Ok(c) = OpenRouterClient::new(key) {
            return (Arc::new(c), "openrouter");
        }
    }
    (Arc::new(SimReviewer::new()), "sim")
}

/// Agentic anti-cheat: `OpenRouter` tool loop when key present, else `SimAgent`.
#[must_use]
pub(crate) fn build_agentic() -> (Arc<dyn AgenticBackend>, &'static str) {
    let key_path = openrouter_key_path();
    if let Ok(key) = challenge_agentic::load_api_key_file(&key_path) {
        if let Ok(a) = OpenRouterAgent::new(key) {
            return (Arc::new(a), "openrouter");
        }
    }
    (Arc::new(SimAgent::new()), "sim")
}

#[allow(clippy::type_complexity)]
async fn build_store() -> (
    Arc<dyn PrismStore>,
    Arc<dyn GatingStore>,
    Arc<dyn EvalStore>,
    String,
) {
    if let Ok(url) = std::env::var("BASE_DATABASE_URL") {
        if !url.trim().is_empty() {
            let pool = match db::connect(&url).await {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("prism-challenge: db connect {e}");
                    std::process::exit(1);
                }
            };
            if let Err(e) = db::migrate(&pool).await {
                eprintln!("prism-challenge: db migrate {e}");
                std::process::exit(1);
            }
            return (
                Arc::new(DbPrismStore::new(pool.clone())),
                Arc::new(PgGatingStore::new(pool.clone())) as Arc<dyn GatingStore>,
                Arc::new(DbEvalStore::new(pool)) as Arc<dyn EvalStore>,
                "postgres".into(),
            );
        }
    }
    (
        Arc::new(MemoryPrismStore::new()),
        Arc::new(MemoryGatingStore::new()) as Arc<dyn GatingStore>,
        Arc::new(MemoryEvalStore::new()) as Arc<dyn EvalStore>,
        "memory".into(),
    )
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

/// Top-model GitHub publisher: graceful no-op (`None`) unless
/// `PRISM_TOPMODEL_GITHUB_TOKEN_FILE` points at a readable token file.
fn build_topmodel() -> Option<Arc<prism_registry::TopModelPublisher>> {
    let p = prism_registry::TopModelPublisher::from_env().map(Arc::new);
    if p.is_none() {
        tracing::info!("top-model publish disabled (PRISM_TOPMODEL_GITHUB_TOKEN_FILE unset/empty)");
    }
    p
}

fn orchestrator_config(
    cli: &Cli,
    ssh_pks: Vec<String>,
    stage_delay: Duration,
) -> OrchestratorConfig {
    OrchestratorConfig {
        netuid: cli.netuid,
        max_price_per_hour: 2.5,
        max_lifetime_hours: prism_recipe::POD_LIFETIME_HOURS_CAP,
        ssh_public_keys: ssh_pks,
        image_digest: None,
        claim_poll: Duration::from_millis(750),
        emit_poll: Duration::from_secs(15),
        max_attempts: MAX_ATTEMPTS,
        similarity_corpus_limit: 6,
        // > wait_running(15m) + train(6h) + ssh margin(~65m) ≈ 7h20m.
        stuck_grace_secs: 10 * 3600,
        stage_delay,
        auto_retry_max: cli.auto_retry_max,
        // scoring_mode from PRISM_SCORING_MODE (default shadow).
        ..Default::default()
    }
}

fn attach_miner_payer(
    orch: Orchestrator<chain_live::LiveChainClient>,
    vault: Option<Arc<PayerKeyVault>>,
    live_ssh: LiumSshConfig,
    has_operator_key: bool,
) -> Orchestrator<chain_live::LiveChainClient> {
    let Some(vault) = vault else {
        return orch;
    };
    // Never fall back to Sim when "allow operator" is set without a real key.
    let allow_op = allow_operator_lium_fallback() && has_operator_key;
    tracing::info!(
        allow_operator_lium = allow_op,
        "miner-funded Lium enabled (X-Lium-Api-Key)"
    );
    orch.with_payer(PayerBackendFactory::new(vault, live_ssh, allow_op))
}

#[allow(clippy::too_many_lines)]
async fn cmd_serve(cli: Cli) -> Result<(), String> {
    let path = resolve_sk_path(cli.challenge_sk_file.as_ref())?;
    if !path.is_file() {
        return Err(format!("challenge secret file missing: {}", path.display()));
    }
    let sk = load_challenge_secret(&path).map_err(|e| e.to_string())?;
    let (backend, backend_mode, ssh_pks, live_ssh, operator_key) = build_backend(cli.force_sim)?;
    let (reviewer, reviewer_mode) = build_reviewer();
    let (agentic, agentic_mode) = build_agentic();
    let (store, gating, eval_store, store_mode) = build_store().await;
    // BASE_SUBMISSION_GATING=0 disables intake gating (local dev only).
    let gating_enabled = !matches!(std::env::var("BASE_SUBMISSION_GATING").as_deref(), Ok("0"));
    if !gating_enabled {
        tracing::warn!("BASE_SUBMISSION_GATING=0: intake metagraph/1-max checks disabled");
    }
    let gateway = Arc::new(
        prism_challenge::GatewayClient::new(prism_challenge::GatewayClientConfig {
            base_url: cli.gateway_endpoint.clone(),
            ..Default::default()
        })
        .map_err(|e| e.to_string())?,
    );

    // Metagraph cache + watcher feed the intake membership check.
    let metagraph = Arc::new(MetagraphCache::new());
    let payer_vault = (backend_mode == "lium").then(|| Arc::new(PayerKeyVault::from_env()));
    let active_jobs = Arc::new(prism_orphan::ActiveJobs::new());
    let log_buf = Arc::new(prism_orphan::LogBuffer::new());

    let state = Arc::new(AppState {
        store: Arc::clone(&store),
        eval_store: Arc::clone(&eval_store),
        epoch: AtomicU64::new(0),
        netuid: cli.netuid,
        backend_mode: Box::leak(
            format!("{backend_mode}/{reviewer_mode}/{agentic_mode}/{store_mode}").into_boxed_str(),
        ),
        retry_max: MAX_ATTEMPTS,
        gating: gating_enabled.then(|| Arc::clone(&gating)),
        metagraph: gating_enabled.then(|| Arc::clone(&metagraph)),
        admin_token_hashes: prism_intake::load_token_hashes(cli.admin_tokens_file.as_deref()),
        payer_vault: payer_vault.clone(),
        logs: Arc::clone(&log_buf),
    });
    // Zone B miner self-report intake: split into `prism-attribution` for
    // the per-crate LOC cap (same pattern as the attribution planner,
    // which mounts inside `submission_router`); own state, no bearer at
    // this layer — front at the gateway where the deployment needs it.
    let app = submission_router(Arc::clone(&state)).route(
        "/v1/submissions/{id}/zone-b",
        prism_attribution::zone_b_route(Arc::clone(&store), Arc::clone(&eval_store)),
    );

    // Chain (for epoch/E): live only when endpoint configured; otherwise a
    // fixed epoch-0 (local sim posture documented in PRISM.md).
    let chain_ep = cli
        .chain_endpoint
        .clone()
        .unwrap_or_else(|| "wss://test.finney.opentensor.ai:443".to_string());
    let mut chain = chain_live::LiveChainClient::connect(&chain_ep).map_err(|e| e.to_string())?;
    chain.set_netuid(cli.netuid);

    spawn_epoch_feed(&chain_ep, &state);

    let stage_delay = Duration::from_millis(cli.sim_stage_delay_ms);
    if !stage_delay.is_zero() {
        tracing::info!(
            delay_ms = cli.sim_stage_delay_ms,
            "prism sim stage delay enabled (local evidence only)"
        );
    }
    let oc = orchestrator_config(&cli, ssh_pks, stage_delay);
    let topmodel = build_topmodel();
    let mut orchestrator = Orchestrator::new(
        oc,
        store,
        backend,
        reviewer,
        agentic,
        &gateway,
        Arc::new(chain),
        sk,
    )
    .with_topmodel(topmodel)
    // v3: persist the METRICS_JSON v2 battery + compute the composite
    // (scoring mode from PRISM_SCORING_MODE; default shadow keeps the v2
    // score bit-identical).
    .with_eval_store(Some(eval_store))
    .with_runtime(Arc::clone(&active_jobs), Arc::clone(&log_buf));
    orchestrator = attach_miner_payer(orchestrator, payer_vault, live_ssh, operator_key.is_some());
    if gating_enabled {
        orchestrator = orchestrator.with_gating(Arc::clone(&gating));
        spawn_gating_watcher(
            &chain_ep,
            cli.netuid,
            Arc::clone(&metagraph),
            Arc::clone(&gating),
            Duration::from_secs(cli.gating_watch_secs.max(15)),
        );
    }
    let orchestrator = Arc::new(orchestrator);
    spawn_orchestrator(
        &cli,
        &orchestrator,
        Arc::clone(&state.store),
        gating_enabled.then_some(gating),
    );
    let listener = TcpListener::bind(cli.bind)
        .await
        .map_err(|e| format!("bind {}: {e}", cli.bind))?;
    tracing::info!(
        %cli.bind,
        challenge_id = CHALLENGE_ID,
        eval_backend = %backend_mode,
        "prism-challenge listening"
    );
    axum::serve(listener, app)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn spawn_epoch_feed(chain_ep: &str, state: &Arc<AppState>) {
    let st_for_feed = Arc::clone(state);
    let ep = chain_ep.to_owned();
    tokio::spawn(async move {
        loop {
            let netuid = st_for_feed.netuid;
            let ep2 = ep.clone();
            let epoch = tokio::task::spawn_blocking(move || {
                let Ok(mut c) = chain_live::LiveChainClient::connect(&ep2) else {
                    return Err("connect".to_string());
                };
                c.set_netuid(netuid);
                chain::gather_schedule_state(&c, netuid)
                    .map(|s| chain::current_epoch_pre_run_coinbase(&s, s.current_block))
                    .map_err(|e| e.to_string())
            })
            .await;
            match epoch {
                Ok(Ok(e)) => prism_challenge::record_epoch(&st_for_feed, e),
                _ => {
                    tracing::warn!("epoch feed read failed; epoch unchanged");
                }
            }
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    });
}

fn spawn_orchestrator(
    cli: &Cli,
    orchestrator: &Arc<Orchestrator<chain_live::LiveChainClient>>,
    store: Arc<dyn PrismStore>,
    gating: Option<Arc<dyn GatingStore>>,
) {
    let permits = cli.max_concurrent_evals.max(1) as usize;
    let sem = Arc::new(Semaphore::new(permits));
    for i in 0..permits {
        let o = Arc::clone(orchestrator);
        let s = Arc::clone(&sem);
        tokio::spawn(async move {
            let _permit = s.acquire_owned().await.ok();
            tracing::info!(worker = i, "orchestrator worker up");
            o.run_worker().await;
        });
    }
    let o = Arc::clone(orchestrator);
    tokio::spawn(async move { o.run_sweeper().await });
    let o = Arc::clone(orchestrator);
    tokio::spawn(async move {
        if let Err(e) = o.reconcile_orphans(true).await {
            tracing::warn!(error = %e, "recover_on_boot failed");
        }
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            if let Err(e) = o.reconcile_orphans(false).await {
                tracing::warn!(error = %e, "orphan reconcile tick error");
            }
        }
    });
    let o = Arc::clone(orchestrator);
    tokio::spawn(async move { o.run_emitter().await });
    spawn_rate_limit_recovery(store, gating);
}

/// Re-queue last-6h failed rows that died on Lium HTTP 429 (no `retry_count` burn).
async fn recover_rate_limited(
    store: &Arc<dyn PrismStore>,
    gating: Option<&Arc<dyn GatingStore>>,
) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    let Ok(failed) = store.list(Some("failed"), None, 500).await else {
        return 0;
    };
    let mut n = 0u32;
    for row in failed {
        let Some(err) = row.error_detail.as_deref() else {
            continue;
        };
        if !lium_rent_pool::should_recover(err, row.updated_at_ms, now) {
            continue;
        }
        if store.reset_for_retry(&row.id, false).await.is_err() {
            continue;
        }
        n = n.saturating_add(1);
        tracing::info!(submission_id = %row.id, "requeued rate-limited submission");
        if let Some(g) = gating {
            let key = prism_pipeline::gating_key(row.arch_id.as_deref());
            let _ = g.reset_open(&key, &row.miner_hotkey).await;
            let _ = g.mark_registered(&key, &row.miner_hotkey, None).await;
        }
    }
    n
}

fn spawn_rate_limit_recovery(store: Arc<dyn PrismStore>, gating: Option<Arc<dyn GatingStore>>) {
    tokio::spawn(async move {
        loop {
            let n = recover_rate_limited(&store, gating.as_ref()).await;
            if n > 0 {
                tracing::info!(requeued = n, "lium 429 recovery tick");
            }
            tokio::time::sleep(Duration::from_mins(2)).await;
        }
    });
}
