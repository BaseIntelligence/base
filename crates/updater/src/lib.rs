//! Digest-pinned container updater over `tecnativa/docker-socket-proxy` (D14).
//!
//! State machine (ported from BASE `challenge_watcher.py`):
//! `idle → resolving → pulling → recreating → verifying → committing`
//! with `rolling_back` / `backoff` / `exhausted` on failure.
//!
//! - Docker Engine API **only** via HTTP to a socket-proxy allowlist.
//! - Image references are **digests only** (`…@sha256:<64-hex>`), never `:latest`.
//! - Durable pins: `current.json` / `previous.json`.
//! - Health-gate: HTTP GET `{health_url}` (typically `/readyz`); rollback on failure.
//! - Never recreates a container whose name matches `self_container_name`.
//! - Self-update of the updater container is **operator-run only** (not automatic).

#![forbid(unsafe_code)]

mod config;
mod digest;
mod docker;
mod error;
mod health;
mod machine;
mod pin_store;

pub use config::{is_rollable_service, UpdaterConfig, ROLLABLE_SERVICES};
pub use digest::{extract_digest, is_pinned_digest, parse_pinned_image, PinnedImage};
pub use docker::{
    assert_allowlisted, is_allowlisted, Allowlist, AllowlistClient, ContainerSummary, DockerApi,
    DockerError, MockDocker, RunResult, ALLOWED_ROUTES, UPDATER_ROUTES, VERIFIER_ROUTES,
};
pub use error::UpdaterError;
pub use health::{check_readyz, wait_readyz, HealthError, ScriptedHealth};
pub use machine::{tick, HealthProbe, HttpHealthProbe, Phase, TickOutcome, Updater};
pub use pin_store::{commit_pins, load_pins, save_current, save_previous, PinRecord, PinStore};

#[cfg(test)]
mod rollable_lockstep_tests {
    use super::{is_rollable_service, ROLLABLE_SERVICES};

    #[test]
    fn prism_challenge_is_rollable_for_promote_lockstep() {
        assert!(ROLLABLE_SERVICES.contains(&"prism-challenge"));
        assert!(is_rollable_service("prism-challenge"));
        assert!(!is_rollable_service("agent-challenge"));
        assert!(is_rollable_service("validator"));
        assert!(!is_rollable_service("base-agent"));
        assert!(!is_rollable_service("socket-proxy"));
    }
}
