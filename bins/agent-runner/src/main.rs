//! `agent-runner` — miner CVM HTTP task API (`agent:8080`).
//!
//! Auth is intentionally open until todo 18 (signed dispatch). Capacity is
//! advertised only until todo 19 (concurrency clamp).

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::process::ExitCode;

use agent_runner::{app, RunnerConfig, RunnerState};
use clap::Parser;
use tokio::net::TcpListener;

/// Miner agent-runner CLI.
#[derive(Debug, Parser)]
#[command(
    name = "agent-runner",
    about = "Miner CVM agent task API (capacity + dispatch)"
)]
struct Cli {
    /// Bind address (compose publishes agent:8080).
    #[arg(long, env = "GBASE_RUNNER_BIND", default_value = "0.0.0.0:8080")]
    bind: SocketAddr,
    /// Advertised max concurrency (enforcement is todo 19).
    #[arg(long, env = "GBASE_MAX_CONCURRENCY", default_value_t = 1)]
    max_concurrency: u32,
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
    let state = RunnerState::new(RunnerConfig {
        max_concurrency: cli.max_concurrency,
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
        auth = "stub_todo_18",
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
