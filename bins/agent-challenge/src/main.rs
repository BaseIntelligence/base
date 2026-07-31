//! `agent-challenge` — operator-side agent-v1 challenge service.
//!
//! Listens on `:8090` for health + pack catalog routes. Challenge secret is
//! loaded from `BASE_CHALLENGE_SK_FILE` (mode 0600 file). Never logs or commits
//! the secret.

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use agent_challenge::{
    load_challenge_secret, pack_routes, public_key_from_secret, AgentV1Challenge, Challenge,
    PackCatalogState, CHALLENGE_ID, SCORING_VERSION,
};
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
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Print challenge id, scoring version, and public key derived from the secret file.
    Identity,
    /// Run health + pack HTTP server (default when no subcommand).
    Serve,
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
                let reason = ready_reason.to_owned();
                move || {
                    let reason = reason.clone();
                    async move {
                        if reason == "ready" {
                            (axum::http::StatusCode::OK, "ready".to_owned())
                        } else {
                            (axum::http::StatusCode::SERVICE_UNAVAILABLE, reason)
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
