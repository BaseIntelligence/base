//! gbase-updater — digest-pinned rollout over docker-socket-proxy (D14).
//!
//! Self-update of this container is **never** automatic; the library refuses
//! when the target name matches `GBASE_UPDATER_SELF_NAME` / `HOSTNAME`.

use std::process::ExitCode;

use gbase_updater::{tick, AllowlistClient, HttpHealthProbe, Updater, UpdaterConfig};

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .json()
        .init();

    let config = match UpdaterConfig::from_env() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("gbase-updater config error: {msg}");
            eprintln!(
                "required: GBASE_UPDATER_PROXY_URL GBASE_UPDATER_COMPOSE_PROJECT \
                 GBASE_UPDATER_SERVICE_NAME GBASE_UPDATER_HEALTH_URL \
                 GBASE_UPDATER_STATE_DIR GBASE_UPDATER_DESIRED_IMAGE"
            );
            return ExitCode::from(2);
        }
    };

    let docker = match AllowlistClient::new(&config.proxy_url) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("docker client: {e}");
            return ExitCode::from(1);
        }
    };
    let health = HttpHealthProbe::new(config.health_url.clone());
    let mut updater = Updater::new(config);

    match tick(&mut updater, &docker, &health) {
        Ok(outcome) => {
            tracing::info!(?outcome, phase = %updater.phase, "tick complete");
            ExitCode::SUCCESS
        }
        Err(e) => {
            tracing::error!(error = %e, "tick failed");
            ExitCode::from(1)
        }
    }
}
