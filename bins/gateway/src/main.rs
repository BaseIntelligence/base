//! gateway — master-only gateway with registry + proxy (D3).
//!
//! Resolves on-chain `SubnetOwnerHotkey` and refuses to bind any listener when
//! the configured hotkey does not match (exit code 2).
//!
//! Chain backend (task 47 cleartext/IP e2e):
//! - default / `GBASE_CHAIN_BACKEND=fake_owner`: [`FakeChain`] whose owner hotkey
//!   equals the configured `GBASE_GATEWAY_HOTKEY` so master check can pass without
//!   a full live SDK client (TLS/ACME still deferred to task 42).
//! - `GBASE_CHAIN_BACKEND=not_implemented`: previous fail-closed stub.

use std::process::ExitCode;

use chain::{FakeChain, FakeChainConfig, NotImplementedChain};
use gateway::{GatewayConfig, GatewayError};

#[tokio::main]
async fn main() -> ExitCode {
    // Tracing before config so structured fatals are JSON when possible.
    let _ = telemetry::init_tracing();

    let config = match GatewayConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            e.log_fatal();
            eprintln!("gateway config error: {e}");
            eprintln!(
                "required: GBASE_ROLE=gateway GBASE_NETUID GBASE_DOMAIN \
                 GBASE_DATABASE_URL (or _FILE) GBASE_GATEWAY_HOTKEY \
                 [GBASE_GATEWAY_LISTEN]"
            );
            return e.exit_code();
        }
    };

    let backend = std::env::var("GBASE_CHAIN_BACKEND")
        .unwrap_or_else(|_| "fake_owner".to_owned());
    let backend = backend.to_ascii_lowercase();

    match backend.as_str() {
        "not_implemented" | "stub" => {
            tracing::warn!(
                backend = %backend,
                "gateway using NotImplementedChain (master check will fail until live SDK)"
            );
            run_with(config, &NotImplementedChain).await
        }
        _ => {
            // fake_owner (default): owner hotkey == configured gateway hotkey.
            let mut fc = FakeChainConfig::default();
            fc.netuid = config.netuid;
            fc.owner_hotkey = config.hotkey.to_vec();
            // UID0 = owner for single-master metagraph smoke.
            fc.hotkeys = vec![config.hotkey.to_vec()];
            let chain = FakeChain::new(fc);
            tracing::info!(
                backend = "fake_owner",
                netuid = config.netuid,
                "gateway chain: FakeChain owner matches GBASE_GATEWAY_HOTKEY (cleartext e2e)"
            );
            run_with(config, &chain).await
        }
    }
}

async fn run_with(config: GatewayConfig, chain: &dyn chain::ChainClient) -> ExitCode {
    match gateway::run(config, chain).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            e.log_fatal();
            if matches!(e, GatewayError::MasterMismatch { .. }) {
                eprintln!("gateway: master-only check failed (exit 2)");
            } else {
                eprintln!("gateway error: {e}");
            }
            e.exit_code()
        }
    }
}
