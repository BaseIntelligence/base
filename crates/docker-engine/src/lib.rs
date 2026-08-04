//! Allowlisted Docker Engine HTTP client (shared).
//!
//! Method/path allowlists ([`Allowlist::updater`] / [`Allowlist::verifier`]),
//! [`DockerApi`], live [`AllowlistClient`], and [`MockDocker`] for tests.
//! Talks HTTP to `tecnativa/docker-socket-proxy` (`CONTAINERS=1 IMAGES=1 POST=1`).
//! No bollard.

#![forbid(unsafe_code)]

mod allowlist;
mod api;
mod client;
mod error;
mod mock;
mod types;

pub use allowlist::{
    assert_allowlisted, is_allowlisted, Allowlist, ALLOWED_ROUTES, UPDATER_ROUTES, VERIFIER_ROUTES,
};
pub use api::DockerApi;
pub use client::AllowlistClient;
pub use error::DockerError;
pub use mock::MockDocker;
pub use types::{ContainerSummary, RunResult, RunSpec};

/// Required name prefix for classic verifier / one-shot runs.
pub const OWNED_NAME_PREFIX: &str = "base-verify-";

/// Required name prefix for design-challenge sandbox runs.
pub const DESIGN_OWNED_NAME_PREFIX: &str = "base-design-";

/// True when `name` starts with an owned run prefix.
#[must_use]
pub fn is_owned_name(name: &str) -> bool {
    name.starts_with(OWNED_NAME_PREFIX) || name.starts_with(DESIGN_OWNED_NAME_PREFIX)
}

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "docker-engine"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_is_docker_engine() {
        assert_eq!(crate_name(), "docker-engine");
    }

    #[test]
    fn profiles_differ_on_delete_container() {
        assert!(!Allowlist::updater().is_allowed("DELETE", "/containers/x"));
        assert!(Allowlist::verifier().is_allowed("DELETE", "/containers/x"));
    }

    #[test]
    fn owned_prefixes() {
        assert!(is_owned_name("base-verify-abc"));
        assert!(is_owned_name("base-design-abc"));
        assert!(!is_owned_name("evil"));
    }
}
