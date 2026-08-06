//! `design-egress-proxy` — open egress for design sandboxes (internal
//! blocklist) plus the budgeted `OpenRouter` chat path.

#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use design_egress_proxy::{proxy_router, BudgetLedger, ProxyState, DEFAULT_TOKEN_BUDGET};
use tokio::net::TcpListener;

/// Egress proxy CLI.
#[derive(Debug, Parser)]
#[command(name = "design-egress-proxy")]
struct Cli {
    /// Bind address.
    #[arg(long, env = "DESIGN_EGRESS_BIND", default_value = "0.0.0.0:8094")]
    bind: SocketAddr,
    /// OpenRouter API key file.
    #[arg(long, env = "OPENROUTER_API_KEY_FILE")]
    openrouter_key_file: Option<PathBuf>,
    /// Per-run token budget.
    #[arg(long, env = "DESIGN_TOKEN_BUDGET", default_value_t = DEFAULT_TOKEN_BUDGET)]
    token_budget: u64,
    /// Force sim replies (no upstream).
    #[arg(long, env = "DESIGN_EGRESS_SIM", default_value_t = false)]
    sim: bool,
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("design-egress-proxy: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let key = cli.openrouter_key_file.and_then(|p| {
        std::fs::read_to_string(p)
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
    });
    let state = Arc::new(ProxyState {
        openrouter_key: key,
        budgets: BudgetLedger::new(cli.token_budget),
        sim: cli.sim || std::env::var("DESIGN_EGRESS_SIM").as_deref() == Ok("1"),
        enforce_blocklist: true,
    });
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let app = proxy_router(state);
        let listener = TcpListener::bind(cli.bind)
            .await
            .map_err(|e| e.to_string())?;
        tracing::info!(%cli.bind, "design-egress-proxy listening (open egress, internal blocklist)");
        axum::serve(listener, app).await.map_err(|e| e.to_string())
    })
}
