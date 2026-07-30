//! `agent-runner` — miner CVM HTTP task API (`agent:8080`).
//!
//! Loads the CVM-local work-receipt key from `GBASE_RECEIPT_SK_FILE` (mode 0600
//! mount). Dispatch auth (todo 18) is **on by default**. Capacity is advertised
//! only until todo 19.

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use agent_runner::{
    app, load_or_generate, load_required, receipt_sk_path_from_env, RunnerConfig, RunnerState,
    DEFAULT_DISPATCH_NONCE_TTL, DEFAULT_RECEIPT_SK_PATH, RECEIPT_SK_FILE_ENV,
};
use clap::Parser;
use tokio::net::TcpListener;

/// Miner agent-runner CLI.
#[derive(Debug, Parser)]
#[command(
    name = "agent-runner",
    about = "Miner CVM agent task API (capacity + authenticated dispatch + work-receipt signing)"
)]
struct Cli {
    /// Bind address (compose publishes agent:8080).
    #[arg(long, env = "GBASE_RUNNER_BIND", default_value = "0.0.0.0:8080")]
    bind: SocketAddr,
    /// Advertised max concurrency (enforcement is todo 19).
    #[arg(long, env = "GBASE_MAX_CONCURRENCY", default_value_t = 1)]
    max_concurrency: u32,
    /// Path to the CVM-local receipt mini-secret (mode 0600 file).
    #[arg(long, env = "GBASE_RECEIPT_SK_FILE", default_value = DEFAULT_RECEIPT_SK_PATH)]
    receipt_sk_file: PathBuf,
    /// When set, generate the receipt key if the file is missing (local/dev only).
    #[arg(long, env = "GBASE_RECEIPT_SK_GENERATE", default_value_t = false)]
    receipt_sk_generate: bool,
    /// Challenge hotkey (32-byte pubkey as 64 hex chars). Required when auth is on.
    #[arg(long, env = "GBASE_CHALLENGE_PUBKEY_HEX")]
    challenge_pubkey_hex: Option<String>,
    /// Enforce signed dispatch (default true). Set `false` only for local dev.
    #[arg(long, env = "GBASE_DISPATCH_AUTH", default_value_t = true)]
    dispatch_auth: bool,
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("agent-runner: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(serve(cli))
}

async fn serve(cli: Cli) -> Result<(), String> {
    let path = if cli.receipt_sk_file.as_os_str().is_empty() {
        receipt_sk_path_from_env()
    } else {
        cli.receipt_sk_file.clone()
    };
    let key = if cli.receipt_sk_generate {
        load_or_generate(&path).map_err(|e| e.to_string())?
    } else {
        load_required(&path).map_err(|e| {
            format!(
                "{e} (path={}, set {RECEIPT_SK_FILE_ENV} or pass --receipt-sk-generate for local)",
                path.display()
            )
        })?
    };
    let public_hex = key.public_key_hex();

    let trusted = match &cli.challenge_pubkey_hex {
        Some(h) => Some(parse_pubkey_hex(h)?),
        None => None,
    };
    if cli.dispatch_auth && trusted.is_none() {
        return Err(
            "GBASE_CHALLENGE_PUBKEY_HEX required when dispatch auth is enabled".into(),
        );
    }

    let state = RunnerState::new(RunnerConfig {
        max_concurrency: cli.max_concurrency,
        receipt_key: Some(key),
        auth_enabled: cli.dispatch_auth,
        trusted_challenge_pubkey: trusted,
        dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
    });
    let router = app(state);
    let listener = TcpListener::bind(cli.bind)
        .await
        .map_err(|e| format!("bind {}: {e}", cli.bind))?;
    let actual = listener.local_addr().map_err(|e| e.to_string())?;
    tracing::info!(
        event = "agent_runner_listening",
        %actual,
        max_concurrency = cli.max_concurrency,
        receipt_public_key_hex = %public_hex,
        auth_enabled = cli.dispatch_auth,
        "agent-runner HTTP surface"
    );
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn parse_pubkey_hex(s: &str) -> Result<[u8; 32], String> {
    let bytes = hex::decode(s).map_err(|e| format!("challenge pubkey hex: {e}"))?;
    if bytes.len() != 32 {
        return Err(format!(
            "challenge pubkey must be 32 bytes (64 hex chars), got {}",
            bytes.len()
        ));
    }
    let mut out = [0_u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}
