//! Allowlisted Docker Engine HTTP client (shared).
//!
//! # Scope (this crate)
//! Method/path allowlist and the `DockerApi` trait used by `updater` and
//! future agent runners. Concrete HTTP transport will be extracted from
//! `crates/updater/src/docker.rs` in a later task — **no logic moved yet**.
//!
//! # What stays in `agent-challenge`
//! Scoring, NoScore / D24 completeness, signing, and weight submit remain in
//! `agent-challenge`. Docker access is infrastructure, not challenge scoring.
//!
//! # Allowlist intent
//! Matches tecnativa/docker-socket-proxy with `CONTAINERS=1 IMAGES=1 POST=1`.
//! Clients must never issue methods/paths outside [`ALLOWED_ROUTES`].

#![forbid(unsafe_code)]

use thiserror::Error;

/// HTTP method + path-prefix pairs permitted for Engine access.
///
/// Paths are matched as Engine API routes **without** the optional `/v1.xx` prefix.
/// Full matching logic lands with the HTTP client extraction.
pub const ALLOWED_ROUTES: &[(&str, &str)] = &[
    ("GET", "/containers/json"),
    ("GET", "/containers/"),  // /containers/{id}/json
    ("POST", "/containers/"), // create, start, stop, rename, …
    ("POST", "/images/create"),
    ("GET", "/images/"),
];

/// Minimal container summary (list/inspect projection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerSummary {
    /// Container id.
    pub id: String,
    /// First name without leading `/`.
    pub name: String,
    /// Image reference as reported by Docker.
    pub image: String,
}

/// Docker client errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DockerError {
    /// Method/path not on the allowlist.
    #[error("docker API call not allowlisted: {method} {path}")]
    NotAllowlisted {
        /// HTTP method.
        method: String,
        /// Request path.
        path: String,
    },
    /// Transport or HTTP status failure (implementation later).
    #[error("docker API error: {0}")]
    Api(String),
}

/// Abstraction over Docker Engine operations used by updaters and runners.
pub trait DockerApi: Send + Sync {
    /// List containers (`all=true` semantics when implemented).
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError>;

    /// Inspect one container by id or name.
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn inspect_container(&self, id_or_name: &str) -> Result<ContainerSummary, DockerError>;
}

/// Return whether `(method, path)` is covered by [`ALLOWED_ROUTES`] (prefix match).
///
/// Skeleton helper so the allowlist is exercised before the full client lands.
#[must_use]
pub fn is_allowlisted(method: &str, path: &str) -> bool {
    let method = method.to_ascii_uppercase();
    ALLOWED_ROUTES.iter().any(|(m, prefix)| {
        m.eq_ignore_ascii_case(&method) && (path == *prefix || path.starts_with(prefix))
    })
}

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "docker-engine"
}

#[cfg(test)]
mod tests {
    use super::{crate_name, is_allowlisted, DockerApi, DockerError};

    struct NoopDocker;

    impl DockerApi for NoopDocker {
        fn list_containers(&self) -> Result<Vec<super::ContainerSummary>, DockerError> {
            Ok(Vec::new())
        }

        fn inspect_container(
            &self,
            id_or_name: &str,
        ) -> Result<super::ContainerSummary, DockerError> {
            Err(DockerError::Api(format!("missing {id_or_name}")))
        }
    }

    #[test]
    fn crate_name_is_docker_engine() {
        assert_eq!(crate_name(), "docker-engine");
    }

    #[test]
    fn allowlist_accepts_container_list() {
        assert!(is_allowlisted("GET", "/containers/json"));
        assert!(is_allowlisted("get", "/containers/abc/json"));
    }

    #[test]
    fn allowlist_rejects_volumes_delete() {
        assert!(!is_allowlisted("DELETE", "/volumes/x"));
    }

    #[test]
    fn noop_list_is_empty() {
        let d = NoopDocker;
        let list = d.list_containers().expect("noop list");
        assert!(list.is_empty());
    }
}
