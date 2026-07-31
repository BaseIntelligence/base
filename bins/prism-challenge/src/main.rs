//! `prism-challenge` — operator-side PRISM challenge service.
//!
//! Listens on `:8092` for liveness (`GET /health`) and miner submit
//! (`POST /v1/submissions`). Challenge secret from `BASE_CHALLENGE_SK_FILE` or
//! `PRISM_CHALLENGE_SK_FILE`. **No Phala CVM** — master evals on Lium (or Sim).
//!
//! When `LIUM_API_KEY` is set, the background worker rents live Lium GPUs.
//! Otherwise it uses [`SimLiumBackend`] (CI / local).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use agent_challenge_keys::load_challenge_secret;
use clap::{Parser, Subcommand};
use crypto::KEY_LEN;
use prism_challenge::{
    emit_signed_leaf_set, public_key_from_secret, run_eval_pipeline, submission_router,
    EvalJobBackend, LiumClient, LiumSshConfig, PipelineInput, PrismConfig, SimLiumBackend,
    SubmissionService, CHALLENGE_ID, SCORING_VERSION,
};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use trustroot::encode_hex;

/// Operator PRISM challenge service CLI.
#[derive(Debug, Parser)]
#[command(
    name = "prism-challenge",
    about = "PRISM challenge service (port 8092, master→Lium, no Phala CVM)"
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
    /// Path to challenge mini-secret (32 raw bytes or hex / age).
    #[arg(long, env = "BASE_CHALLENGE_SK_FILE", global = true)]
    challenge_sk_file: Option<PathBuf>,
    /// Force Sim backend even if `LIUM_API_KEY` is present.
    #[arg(long, env = "PRISM_FORCE_SIM", default_value_t = false, global = true)]
    force_sim: bool,
    /// Epoch stamped on emitted leaves (operator-set; default 0).
    #[arg(long, env = "PRISM_EPOCH", default_value_t = 0, global = true)]
    epoch: u64,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print challenge id, scoring version, and public key.
    Identity,
    /// Run HTTP server + eval worker (default).
    Serve,
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let cli = Cli::parse();
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
    Err("BASE_CHALLENGE_SK_FILE or PRISM_CHALLENGE_SK_FILE / --challenge-sk-file required".into())
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

fn load_lium_api_key() -> Option<String> {
    if let Ok(k) = std::env::var("LIUM_API_KEY") {
        let t = k.trim().to_owned();
        if !t.is_empty() {
            return Some(t);
        }
    }
    let cred = PathBuf::from("/root/.config/prism-mission/credentials.env");
    let Ok(text) = std::fs::read_to_string(cred) else {
        return None;
    };
    for line in text.lines() {
        let line = line.trim();
        let line = line.strip_prefix("export ").unwrap_or(line);
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
    let p = match std::env::var("LIUM_SSH_PUBLIC_KEY_FILE") {
        Ok(v) => PathBuf::from(v),
        Err(_) => PathBuf::from("/root/.config/prism-mission/lium_ssh_ed25519.pub"),
    };
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn build_backend(force_sim: bool) -> Result<(Arc<dyn EvalJobBackend>, PrismConfig, &'static str), String> {
    if force_sim || load_lium_api_key().is_none() {
        let cfg = PrismConfig::sim();
        let backend: Arc<dyn EvalJobBackend> = Arc::new(SimLiumBackend::new());
        return Ok((backend, cfg, "sim"));
    }
    let api_key = load_lium_api_key().ok_or_else(|| "LIUM_API_KEY missing".to_string())?;
    let mut ssh = LiumSshConfig::default_live();
    if let Ok(p) = std::env::var("LIUM_SSH_PRIVATE_KEY") {
        ssh.private_key_path = Some(PathBuf::from(p));
    } else {
        let default = PathBuf::from("/root/.config/prism-mission/lium_ssh_ed25519");
        if default.is_file() {
            ssh.private_key_path = Some(default);
        }
    }
    let client = LiumClient::with_config(api_key, prism_challenge::LIUM_API_BASE_URL, ssh)
        .map_err(|e| e.to_string())?;
    let mut cfg = PrismConfig::live_smoke();
    if let Some(pk) = load_ssh_public_key() {
        cfg = cfg.with_ssh_public_keys(vec![pk]);
    } else {
        return Err(
            "live Lium requires SSH public key (LIUM_SSH_PUBLIC_KEY_FILE or default path)".into(),
        );
    }
    let backend: Arc<dyn EvalJobBackend> = Arc::new(client);
    Ok((backend, cfg, "lium"))
}

async fn cmd_serve(cli: Cli) -> Result<(), String> {
    let path = resolve_sk_path(cli.challenge_sk_file.as_ref())?;
    if !path.is_file() {
        return Err(format!("challenge secret file missing: {}", path.display()));
    }
    // Load secret to fail-closed at boot; do not log it.
    let sk = load_challenge_secret(&path).map_err(|e| e.to_string())?;
    let (backend, cfg, mode) = build_backend(cli.force_sim)?;
    let state = Arc::new(SubmissionService::new());
    let app = submission_router(Arc::clone(&state));

    // Single-flight eval worker (max_concurrent_evals = 1).
    let permits = cfg.max_concurrent_evals.max(1);
    let sem = Arc::new(Semaphore::new(permits as usize));
    let worker_state = Arc::clone(&state);
    let worker_backend = Arc::clone(&backend);
    let worker_cfg = cfg.clone();
    let worker_sk = sk;
    let epoch = cli.epoch;
    tokio::spawn(async move {
        eval_worker_loop(worker_state, worker_backend, worker_cfg, worker_sk, epoch, sem).await;
    });

    let listener = TcpListener::bind(cli.bind)
        .await
        .map_err(|e| format!("bind {}: {e}", cli.bind))?;
    tracing::info!(
        %cli.bind,
        challenge_id = CHALLENGE_ID,
        eval_backend = mode,
        "prism-challenge listening"
    );
    axum::serve(listener, app)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

async fn eval_worker_loop(
    svc: Arc<SubmissionService>,
    backend: Arc<dyn EvalJobBackend>,
    cfg: PrismConfig,
    sk: [u8; KEY_LEN],
    epoch: u64,
    sem: Arc<Semaphore>,
) {
    loop {
        let Some(queued) = svc.pop() else {
            tokio::time::sleep(Duration::from_millis(200)).await;
            continue;
        };
        let Ok(permit) = sem.clone().acquire_owned().await else {
            tracing::error!("eval semaphore closed");
            return;
        };
        let backend = Arc::clone(&backend);
        let cfg = cfg.clone();
        let sid = queued.id.clone();
        tokio::spawn(async move {
            let _permit = permit;
            tracing::info!(submission_id = %sid, "prism eval start");
            let input = PipelineInput {
                request: queued.request.clone(),
            };
            match run_eval_pipeline(backend, &cfg, input).await {
                Ok(result) => {
                    // Best-effort single-hotkey leaf set for operator logs / gateway handoff.
                    if let Ok(hk) = hotkey_from_hex(&queued.request.miner_hotkey) {
                        let mut expected = BTreeSet::new();
                        expected.insert(hk);
                        let mut scores = BTreeMap::new();
                        scores.insert(hk, result.score.clone());
                        match emit_signed_leaf_set(&sk, epoch, &expected, &scores) {
                            Ok(leaves) => {
                                tracing::info!(
                                    submission_id = %sid,
                                    pod_id = %result.pod_id,
                                    bpb = ?result.bpb,
                                    termination_verified = result.receipt.termination_verified,
                                    leaves = leaves.len(),
                                    "prism eval complete"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    submission_id = %sid,
                                    error = %e,
                                    "leaf emit failed after eval"
                                );
                            }
                        }
                    } else {
                        tracing::info!(
                            submission_id = %sid,
                            pod_id = %result.pod_id,
                            bpb = ?result.bpb,
                            "prism eval complete (no leaf; bad hotkey hex)"
                        );
                    }
                }
                Err(e) => {
                    tracing::error!(submission_id = %sid, error = %e, "prism eval failed");
                }
            }
        });
    }
}

fn hotkey_from_hex(s: &str) -> Result<[u8; KEY_LEN], ()> {
    let s = s.trim();
    if s.len() != 64 {
        return Err(());
    }
    let mut out = [0u8; KEY_LEN];
    for i in 0..KEY_LEN {
        let byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|_| ())?;
        out[i] = byte;
    }
    Ok(out)
}
