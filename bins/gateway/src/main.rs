//! gateway — master-only gateway with registry + proxy (D3).
//!
//! Resolves on-chain `SubnetOwnerHotkey` and refuses to bind any listener when
//! the configured hotkey does not match (exit code 2).

use std::process::ExitCode;

use chain::NotImplementedChain;
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

    // Live chain client lands with SDK wiring; skeleton uses NotImplementedChain
    // so production deploys fail closed until task follow-up wires RPC owner read.
    // Tests inject FakeChain via the library `run_until` API.
    let chain = NotImplementedChain;
    match gateway::run(config, &chain).await {
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
