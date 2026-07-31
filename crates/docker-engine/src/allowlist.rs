//! Explicit Engine API method/path allowlists (updater vs verifier).

use crate::error::DockerError;

/// HTTP method + path-prefix pairs for one client profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allowlist {
    routes: &'static [(&'static str, &'static str)],
}

/// Updater: list/inspect/stop/rename + pull/create/start.
pub const UPDATER_ROUTES: &[(&str, &str)] = &[
    ("GET", "/containers/json"),
    ("GET", "/containers/"),
    ("POST", "/containers/"),
    ("POST", "/images/create"),
    ("GET", "/images/"),
];

/// Verifier / agent-runner: pull + create/start/wait/logs/rm.
pub const VERIFIER_ROUTES: &[(&str, &str)] = &[
    ("POST", "/images/create"),
    ("POST", "/containers/create"),
    ("POST", "/containers/"),
    ("GET", "/containers/"),
    ("DELETE", "/containers/"),
];

/// Backward-compatible alias for the updater route table.
pub const ALLOWED_ROUTES: &[(&str, &str)] = UPDATER_ROUTES;

impl Allowlist {
    /// Digest-pinned container updater profile.
    #[must_use]
    pub const fn updater() -> Self {
        Self {
            routes: UPDATER_ROUTES,
        }
    }

    /// Challenge verifier / one-shot agent runner profile.
    #[must_use]
    pub const fn verifier() -> Self {
        Self {
            routes: VERIFIER_ROUTES,
        }
    }

    /// Route table for this profile.
    #[must_use]
    pub const fn routes(self) -> &'static [(&'static str, &'static str)] {
        self.routes
    }

    /// True if method+path is permitted.
    #[must_use]
    pub fn is_allowed(self, method: &str, path: &str) -> bool {
        let method = method.to_ascii_uppercase();
        let path = strip_api_version(path);
        self.routes
            .iter()
            .any(|(m, p)| method == *m && path_matches(path, p))
    }

    /// Reject non-allowlisted calls before any HTTP is sent.
    ///
    /// # Errors
    /// [`DockerError::NotAllowlisted`] when blocked.
    pub fn assert_allowed(self, method: &str, path: &str) -> Result<(), DockerError> {
        if self.is_allowed(method, path) {
            Ok(())
        } else {
            Err(DockerError::NotAllowlisted {
                method: method.to_owned(),
                path: path.to_owned(),
            })
        }
    }
}

/// Updater-profile allow check.
#[must_use]
pub fn is_allowlisted(method: &str, path: &str) -> bool {
    Allowlist::updater().is_allowed(method, path)
}

/// Updater-profile guard before HTTP.
///
/// # Errors
/// [`DockerError::NotAllowlisted`] when blocked.
pub fn assert_allowlisted(method: &str, path: &str) -> Result<(), DockerError> {
    Allowlist::updater().assert_allowed(method, path)
}

fn strip_api_version(path: &str) -> &str {
    let b = path.as_bytes();
    if b.len() > 4 && b[0] == b'/' && b[1] == b'v' {
        if let Some(i) = path[1..].find('/') {
            return &path[1 + i..];
        }
    }
    path
}

fn path_matches(path: &str, prefix: &str) -> bool {
    if prefix.ends_with('/') {
        path == prefix.trim_end_matches('/') || path.starts_with(prefix)
    } else {
        path == prefix || path.starts_with(&format!("{prefix}?"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updater_accepts_container_and_image_routes() {
        let a = Allowlist::updater();
        assert!(a.is_allowed("GET", "/containers/json"));
        assert!(a.is_allowed("GET", "/v1.43/containers/json"));
        assert!(a.is_allowed("POST", "/images/create?fromImage=x"));
        assert!(a.is_allowed("POST", "/containers/abc/stop"));
        assert!(a.is_allowed("POST", "/containers/create?name=x"));
        assert!(a.is_allowed("POST", "/containers/abc/wait"));
    }

    #[test]
    fn updater_rejects_volumes_delete_and_networks() {
        let a = Allowlist::updater();
        assert!(!a.is_allowed("DELETE", "/volumes/foo"));
        assert!(!a.is_allowed("GET", "/volumes"));
        assert!(!a.is_allowed("POST", "/networks/create"));
        assert!(!a.is_allowed("DELETE", "/containers/abc"));
        assert!(a.assert_allowed("DELETE", "/volumes/x").is_err());
    }

    #[test]
    fn verifier_allows_run_lifecycle_only() {
        let v = Allowlist::verifier();
        assert!(v.is_allowed("POST", "/images/create?fromImage=alpine%40sha256%3Aab"));
        assert!(v.is_allowed("POST", "/containers/create?name=t"));
        assert!(v.is_allowed("POST", "/containers/id1/start"));
        assert!(v.is_allowed("POST", "/containers/id1/wait"));
        assert!(v.is_allowed("GET", "/containers/id1/logs?stdout=true"));
        assert!(v.is_allowed("DELETE", "/containers/id1?force=true"));
    }

    #[test]
    fn verifier_rejects_updater_only_and_foreign_verbs() {
        let v = Allowlist::verifier();
        assert!(!v.is_allowed("DELETE", "/volumes/x"));
        assert!(!v.is_allowed("POST", "/networks/create"));
        assert!(!v.is_allowed("GET", "/volumes"));
        assert!(!v.is_allowed("POST", "/build"));
        assert!(matches!(
            v.assert_allowed("DELETE", "/volumes/data"),
            Err(DockerError::NotAllowlisted { .. })
        ));
    }

    #[test]
    fn legacy_helpers_match_updater_profile() {
        assert!(is_allowlisted("GET", "/containers/json"));
        assert!(!is_allowlisted("DELETE", "/volumes/x"));
        assert!(assert_allowlisted("POST", "/networks/create").is_err());
    }
}
