//! gateway — master-only gateway with registry + proxy (D3).
//!
//! Chain backend: always [`chain_live::LiveChainClient`] against
//! `BASE_CHAIN_ENDPOINT`. There is no in-memory fake.
//!
//! The gateway resolves the on-chain `SubnetOwnerHotkey` and logs whether the
//! configured hotkey matches it. The mismatch is advisory by default; set
//! `BASE_GATEWAY_REQUIRE_OWNER=1` to restore the fail-closed master-only check.

use std::process::ExitCode;
use std::sync::Arc;

use config::keys;
use gateway::{GatewayConfig, GatewayError, MemoryBundleStore, MemoryRawWeightStore, Stores};

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
    // Same live client backs the master-only check and the epoch seal: the
    // sealed metagraph root must come from the chain we actually talked to.
    let chain: gateway::SharedChain = Arc::new(client);

    let stores = match resolve_stores().await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gateway: {e}");
            return ExitCode::from(1);
        }
    };

    // Miners announce their CVM base URL here (AGENT_CHALLENGE.md §9.3 step 5).
    // Without a database there is nowhere to put an announcement, so the route
    // is left unmounted rather than accepting POSTs it would silently discard.
    let announce = match resolve_endpoint_router(&chain, config.netuid).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gateway: {e}");
            return ExitCode::from(1);
        }
    };

    // Owner-issued credit for runtimes without a TEE (AGENT_CHALLENGE.md
    // §9.6). Same mounting rule as the endpoint route: no database, no route.
    let attest_grant = match resolve_attest_grant_router(config.hotkey).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gateway: {e}");
            return ExitCode::from(1);
        }
    };
    let extra = merge_optional_routers(announce, attest_grant);

    run_with(config, chain, stores, extra).await
}

/// Owner-issued attestation-grant router, when a database is configured.
async fn resolve_attest_grant_router(
    gateway_hotkey: [u8; 32],
) -> Result<Option<axum::Router>, String> {
    let base = config::load().map_err(|e| e.to_string())?;
    let Some(url) = resolve_database_url(&base)? else {
        tracing::warn!(
            event = "gateway_attest_grant_disabled",
            "no database configured; {} is not mounted",
            gateway::ATTEST_GRANT_ROUTE
        );
        return Ok(None);
    };
    let pool = db::connect(&url)
        .await
        .map_err(|e| format!("attest-grant pool connect failed: {e}"))?;
    Ok(Some(gateway::admin_attest_grant_router(
        gateway::AttestGrantState::new(pool, gateway_hotkey),
    )))
}

/// Merge up to two optional extra routers into one.
fn merge_optional_routers(
    announce: Option<axum::Router>,
    attest_grant: Option<axum::Router>,
) -> Option<axum::Router> {
    match (announce, attest_grant) {
        (None, None) => None,
        (Some(a), None) => Some(a),
        (None, Some(g)) => Some(g),
        (Some(a), Some(g)) => Some(a.merge(g)),
    }
}

/// Miner endpoint announce router, when a database is configured.
async fn resolve_endpoint_router(
    chain: &gateway::SharedChain,
    netuid: u16,
) -> Result<Option<axum::Router>, String> {
    let base = config::load().map_err(|e| e.to_string())?;
    let Some(url) = resolve_database_url(&base)? else {
        tracing::warn!(
            event = "gateway_miner_endpoint_disabled",
            "no database configured; {} is not mounted",
            miner_endpoint::ENDPOINT_ROUTE
        );
        return Ok(None);
    };
    let pool = db::connect(&url)
        .await
        .map_err(|e| format!("miner endpoint pool connect failed: {e}"))?;
    let state = miner_endpoint::MinerEndpointState::new(Arc::clone(chain), pool, netuid);
    Ok(Some(miner_endpoint::miner_endpoint_router(state)))
}

/// Postgres stores when a database is configured, in-memory otherwise.
///
/// A configured but unreachable database is fatal: falling back to memory would
/// silently drop every raw weight and sealed bundle on restart.
async fn resolve_stores() -> Result<Stores, String> {
    let base = config::load().map_err(|e| e.to_string())?;
    let Some(url) = resolve_database_url(&base)? else {
        tracing::warn!(
            event = "gateway_store_memory",
            "no database configured; raw weights and sealed bundles are not persisted"
        );
        return Ok((
            Arc::new(MemoryRawWeightStore::new()),
            Arc::new(MemoryBundleStore::new()),
        ));
    };
    let pool = db::connect(&url)
        .await
        .map_err(|e| format!("database connect failed: {e}"))?;
    db::migrate(&pool)
        .await
        .map_err(|e| format!("database migrate failed: {e}"))?;
    pool.close().await;
    let stores = gateway_store_pg::stores(&url)?;
    tracing::info!(
        event = "gateway_store_postgres",
        "raw weights and sealed bundles persist to postgres"
    );
    Ok(stores)
}

fn resolve_database_url(cfg: &config::Config) -> Result<Option<String>, String> {
    if let Some(url) = cfg.database_url.as_ref() {
        return Ok(Some(url.clone()));
    }
    if let Some(path) = cfg.database_url_file.as_ref() {
        let raw =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let trimmed = raw.trim().to_owned();
        if trimmed.is_empty() {
            return Err("database_url_file is empty".into());
        }
        return Ok(Some(trimmed));
    }
    Ok(None)
}

async fn run_with(
    config: GatewayConfig,
    chain: gateway::SharedChain,
    stores: Stores,
    extra: Option<axum::Router>,
) -> ExitCode {
    match gateway::run(config, chain, stores, extra).await {
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
