//! Docker Engine access for the updater — re-exported from shared `docker-engine`.
//!
//! Implementation lives in `crates/docker-engine`. This module preserves the
//! historical `updater::docker::*` path for callers and tests.

pub use docker_engine::{
    assert_allowlisted, is_allowlisted, Allowlist, AllowlistClient, ContainerSummary, DockerApi,
    DockerError, MockDocker, RunResult, ALLOWED_ROUTES, UPDATER_ROUTES, VERIFIER_ROUTES,
};
