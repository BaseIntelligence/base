//! `agent-runner` — miner CVM HTTP task API (`agent:8080`).
//!
//! Loads the CVM-local work-receipt key from `GBASE_RECEIPT_SK_FILE` (mode 0600
//! mount). Dispatch auth (todo 18) is on by default when a trusted challenge
//! pubkey is configured. Concurrency is clamped to 1..=5 and enforced with a
//! semaphore (todo 19).

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use agent_runner::{
    app, clamp_concurrency, load_or_generate, load_required, receipt_sk_path_from_env,
    RunnerConfig, RunnerState, DEFAULT_DISPATCH_NONCE_TTL, DEFAULT_RECEIPT_SK_PATH,
    RECEIPT_SK_FILE_ENV,
};
use clap::Parser;
use crypto::KEY_LEN;
use tokio::net::TcpListener;

/// Miner agent-runner CLI.
#[derive(Debug, Parser)]
#[command(
    name = "agent-runner",
    about = "Miner CVM agent task API (capacity + dispatch + work-receipt signing)"
)]
struct Cli {
    /// Bind address (compose publishes agent:8080).
    #[arg(long, env = "GBASE_RUNNER_BIND", default_value = "0.0.0.0:8080")]
    bind: SocketAddr,
    /// Miner-declared max concurrency (clamped to 1..=5 at runtime).
    #[arg(long, env = "GBASE_MAX_CONCURRENCY", default_value_t = 1)]
    max_concurrency: u32,
    /// Path to the CVM-local receipt mini-secret (mode 0600 file).
    #[arg(long, env = "GBASE_RECEIPT_SK_FILE", default_value = DEFAULT_RECEIPT_SK_PATH)]
    receipt_sk_file: PathBuf,
    /// When set, generate the receipt key if the file is missing (local/dev only).
    #[arg(long, env = "GBASE_RECEIPT_SK_GENERATE", default_value_t = false)]
    receipt_sk_generate: bool,
    /// Disable dispatch auth (local/dev only). Default: auth on when pubkey set.
    #[arg(long, env = "GBASE_DISPATCH_AUTH_DISABLE", default_value_t = false)]
    dispatch_auth_disable: bool,
    /// Trusted challenge public key (64 hex) for dispatch auth.
    #[arg(long, env = "GBASE_TRUSTED_CHALLENGE_PUBKEY")]
    trusted_challenge_pubkey: Option<String>,
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

fn parse_pubkey_hex(s: &str) -> Result<[u8; KEY_LEN], String> {
    let bytes = hex::decode(s.trim()).map_err(|e| format!("trusted pubkey hex: {e}"))?;
    if bytes.len() != KEY_LEN {
        return Err(format!("trusted pubkey must be {KEY_LEN} bytes"));
    }
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&bytes);
    Ok(out)
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

    let trusted = match &cli.trusted_challenge_pubkey {
        Some(h) => Some(parse_pubkey_hex(h)?),
        None => None,
    };
    let auth_enabled = !cli.dispatch_auth_disable;
    if auth_enabled && trusted.is_none() {
        return Err(
            "dispatch auth enabled but GBASE_TRUSTED_CHALLENGE_PUBKEY unset (or pass --dispatch-auth-disable)"
                .into(),
        );
    }

    let declared = cli.max_concurrency;
    let effective = clamp_concurrency(declared);
    let state = RunnerState::new(RunnerConfig {
        max_concurrency: declared,
        auth_enabled,
        trusted_challenge_pubkey: trusted,
        dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
        receipt_key: Some(key),
        stub_hold: Duration::ZERO,
    });
    let router = app(state);
    let listener = TcpListener::bind(cli.bind)
        .await
        .map_err(|e| format!("bind {}: {e}", cli.bind))?;
    let actual = listener.local_addr().map_err(|e| e.to_string())?;
    tracing::info!(
        event = "agent_runner_listening",
        %actual,
        declared_max_concurrency = declared,
        effective_max_concurrency = effective,
        receipt_public_key_hex = %public_hex,
        auth_enabled,
        "agent-runner HTTP surface"
    );
    let _ = Duration::from_secs(1); // keep import used if ttl changes
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}
