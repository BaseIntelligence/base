//! gateway — master-only gateway with registry + proxy (D3).
//!
//! Chain backend: always [`chain_live::LiveChainClient`] against
//! `BASE_CHAIN_ENDPOINT`. There is no in-memory fake.
//!
//! The gateway resolves the on-chain `SubnetOwnerHotkey` and logs whether the
//! configured hotkey matches it. The mismatch is advisory by default; set
//! `BASE_GATEWAY_REQUIRE_OWNER=1` to restore the fail-closed master-only check.

use std::process::ExitCode;

use config::keys;
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
                "required: BASE_ROLE=gateway BASE_NETUID BASE_DOMAIN \
                 BASE_DATABASE_URL (or _FILE) BASE_GATEWAY_HOTKEY \
                 [BASE_GATEWAY_LISTEN]"
            );
            return e.exit_code();
        }
    };

    let endpoint = std::env::var(keys::CHAIN_ENDPOINT)
        .unwrap_or_else(|_| config::DEFAULT_CHAIN_ENDPOINT.to_owned());
    tracing::info!(
        endpoint = %endpoint,
        netuid = config.netuid,
        "gateway connecting to live chain"
    );
    let mut client = match chain_live::LiveChainClient::connect(&endpoint) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gateway: live chain connect failed: {e}");
            return ExitCode::from(1);
        }
    };
    client.set_netuid(config.netuid);
    run_with(config, &client).await
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
