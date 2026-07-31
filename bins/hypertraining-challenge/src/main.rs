//! `hypertraining-challenge` — operator-side hypertraining challenge service.
//!
//! Listens on `:8091` for liveness (`GET /health`) and miner submit
//! (`POST /v1/submissions`). Challenge secret is loaded from
//! `BASE_CHALLENGE_SK_FILE` or `HYPERTRAINING_CHALLENGE_SK_FILE` (mode 0600
//! file). Missing or unloadable secret → non-zero exit. Never logs the secret.

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use agent_challenge_keys::load_challenge_secret;
use clap::{Parser, Subcommand};
use hypertraining_challenge::{
    public_key_from_secret, submission_router, SubmissionService, CHALLENGE_ID, SCORING_VERSION,
};
use tokio::net::TcpListener;
use trustroot::encode_hex;

/// Operator hypertraining challenge service CLI.
#[derive(Debug, Parser)]
#[command(
    name = "hypertraining-challenge",
    about = "hypertraining challenge service (port 8091)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Bind address for HTTP (default 0.0.0.0:8091).
    #[arg(
        long,
        env = "BASE_CHALLENGE_BIND",
        default_value = "0.0.0.0:8091",
        global = true
    )]
    bind: SocketAddr,
    /// Path to challenge mini-secret (32 raw bytes or hex). Required for serve.
    #[arg(long, env = "BASE_CHALLENGE_SK_FILE", global = true)]
    challenge_sk_file: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print challenge id, scoring version, and public key from the secret file.
    Identity,
    /// Run HTTP server (default when no subcommand).
    Serve,
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("hypertraining-challenge: {msg}");
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

/// Prefer CLI / `BASE_CHALLENGE_SK_FILE`, then `HYPERTRAINING_CHALLENGE_SK_FILE`.
fn resolve_sk_path(cli_path: Option<&PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = cli_path {
        return Ok(p.clone());
    }
    if let Some(p) = std::env::var_os("HYPERTRAINING_CHALLENGE_SK_FILE") {
        return Ok(PathBuf::from(p));
    }
    Err(
        "BASE_CHALLENGE_SK_FILE or HYPERTRAINING_CHALLENGE_SK_FILE / --challenge-sk-file required"
            .into(),
    )
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

async fn cmd_serve(cli: Cli) -> Result<(), String> {
    let path = resolve_sk_path(cli.challenge_sk_file.as_ref())?;
    if !path.is_file() {
        return Err(format!("challenge secret file missing: {}", path.display()));
    }
    // Load once at boot — refuse to listen without a valid signing key.
    let sk = load_challenge_secret(&path).map_err(|e| e.to_string())?;
    let pk = public_key_from_secret(&sk)?;

    let app = submission_router(Arc::new(SubmissionService::default()));

    let listener = TcpListener::bind(cli.bind)
        .await
        .map_err(|e| format!("bind {}: {e}", cli.bind))?;
    let actual = listener.local_addr().map_err(|e| e.to_string())?;
    tracing::info!(
        event = "hypertraining_challenge_listening",
        %actual,
        challenge_id = CHALLENGE_ID,
        scoring_version = SCORING_VERSION,
        public_key = %encode_hex(&pk),
        "hypertraining-challenge health + submit server"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}
