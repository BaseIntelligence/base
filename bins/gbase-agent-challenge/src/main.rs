//! `gbase-agent-challenge` — operator-side agent-v1 challenge service.
//!
//! Listens on `:8090` for `/healthz` + `/readyz`. Challenge secret is loaded from
//! `GBASE_CHALLENGE_SK_FILE` (mode 0600 file). Never logs or commits the secret.

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use axum::routing::get;
use axum::Router;
use clap::{Parser, Subcommand};
use gbase_agent_challenge::{
    load_challenge_secret, public_key_from_secret, AgentV1Challenge, Challenge, CHALLENGE_ID,
    SCORING_VERSION,
};
use gbase_trustroot::encode_hex;
use tokio::net::TcpListener;

/// Operator challenge service CLI.
#[derive(Debug, Parser)]
#[command(name = "gbase-agent-challenge", about = "agent-v1 challenge service")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Bind address for health endpoints (default 0.0.0.0:8090).
    #[arg(long, env = "GBASE_CHALLENGE_BIND", default_value = "0.0.0.0:8090")]
    bind: SocketAddr,
    /// Path to challenge mini-secret (32 raw bytes or hex). Required for ready.
    #[arg(long, env = "GBASE_CHALLENGE_SK_FILE")]
    challenge_sk_file: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print challenge id, scoring version, and public key derived from the secret file.
    Identity,
    /// Run health HTTP server (default when no subcommand).
    Serve,
}

fn main() -> ExitCode {
    let _ = gbase_telemetry::init_tracing();
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("gbase-agent-challenge: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let cmd = cli.cmd.unwrap_or(Cmd::Serve);
    match cmd {
        Cmd::Identity => cmd_identity(cli.challenge_sk_file.as_ref()),
        Cmd::Serve => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            rt.block_on(cmd_serve(cli.bind, cli.challenge_sk_file.as_ref()))
        }
    }
}

fn cmd_identity(sk_file: Option<&PathBuf>) -> Result<(), String> {
    let path = sk_file.ok_or("GBASE_CHALLENGE_SK_FILE / --challenge-sk-file required")?;
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

async fn cmd_serve(bind: SocketAddr, sk_file: Option<&PathBuf>) -> Result<(), String> {
    let ready = match sk_file {
        Some(p) if p.is_file() => load_challenge_secret(p).is_ok(),
        _ => false,
    };

    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route(
            "/readyz",
            get(move || async move {
                if ready {
                    (axum::http::StatusCode::OK, "ready")
                } else {
                    (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        "challenge secret not loaded",
                    )
                }
            }),
        );

    let listener = TcpListener::bind(bind)
        .await
        .map_err(|e| format!("bind {bind}: {e}"))?;
    let actual = listener.local_addr().map_err(|e| e.to_string())?;
    tracing::info!(
        event = "agent_challenge_listening",
        %actual,
        challenge_id = CHALLENGE_ID,
        scoring_version = SCORING_VERSION,
        ready,
        "agent-challenge health server"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}
