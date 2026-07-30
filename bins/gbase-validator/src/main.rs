//! gbase-validator — independent weight recomputation process (skeleton).
//!
//! Listens for `/healthz`, `/readyz`, `/metrics`. Epoch clock from chain tip.
//! Coordination client talks only to allowlisted gateway paths.

use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::Arc;

use gbase_chain::FakeChain;
use gbase_config::{keys, load, Role};
use gbase_validator::{
    db_ready_from_fn, db_ready_ok, spawn_validator, RegistrationStub, SyncChain, ValidatorRuntime,
};

#[tokio::main]
async fn main() -> ExitCode {
    if let Err(e) = gbase_telemetry::init_tracing() {
        // Subscriber may already be set; continue with best-effort logs.
        eprintln!("tracing init: {e}");
    }

    let cfg = match load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gbase-validator config error: {e}");
            eprintln!(
                "required: {}=validator {}=<u16> and database url or file",
                keys::ROLE,
                keys::NETUID
            );
            return ExitCode::from(2);
        }
    };

    if cfg.role != Role::Validator {
        eprintln!(
            "gbase-validator requires {}=validator (got {})",
            keys::ROLE,
            cfg.role
        );
        return ExitCode::from(2);
    }

    let listen = std::env::var("GBASE_LISTEN")
        .ok()
        .and_then(|s| s.parse::<SocketAddr>().ok())
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 8080)));

    // Skeleton chain: FakeChain until live SDK client is wired (task 13 follow-up).
    let chain = Arc::new(SyncChain::new(FakeChain::with_defaults()));

    let db_check = match resolve_database_url(&cfg) {
        Ok(Some(url)) => match gbase_db::connect(&url).await {
            Ok(pool) => {
                if let Err(e) = gbase_db::migrate(&pool).await {
                    tracing::warn!(error = %e, "migrate skipped/failed");
                }
                db_ready_from_fn(move || {
                    let pool = pool.clone();
                    async move {
                        gbase_db::count_miners(&pool)
                            .await
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    }
                })
            }
            Err(e) => {
                eprintln!("database connect failed: {e}");
                return ExitCode::from(1);
            }
        },
        Ok(None) => db_ready_ok(),
        Err(e) => {
            eprintln!("database url: {e}");
            return ExitCode::from(2);
        }
    };

    let runtime = ValidatorRuntime {
        epoch_length: cfg.epoch_length,
        listen_addr: listen,
        gateway_endpoint: cfg.gateway_endpoint.clone(),
        registration: RegistrationStub::new(),
    };

    let running = match spawn_validator(runtime, chain, db_check, vec![]).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gbase-validator failed to start: {e}");
            return ExitCode::from(1);
        }
    };

    tracing::info!(addr = %running.addr, "gbase-validator ready");

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("ctrl-c received, shutting down");
        }
    }

    if let Err(e) = running.shutdown().await {
        tracing::error!(error = %e, "shutdown error");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn resolve_database_url(cfg: &gbase_config::Config) -> Result<Option<String>, String> {
    if let Some(url) = cfg.database_url.as_ref() {
        return Ok(Some(url.clone()));
    }
    if let Some(path) = cfg.database_url_file.as_ref() {
        let s =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let t = s.trim().to_owned();
        if t.is_empty() {
            return Err("database_url_file is empty".into());
        }
        return Ok(Some(t));
    }
    Ok(None)
}
