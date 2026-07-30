//! Gateway coordination client with a hard allowlist.
//!
//! Validators may only call public bundle / weights read paths. Master-only
//! endpoints (owner assertion admin, backend registry CRUD, raw weight ingress,
//! seal triggers) are refused client-side so a misconfiguration cannot hit them.

use std::sync::Arc;

use reqwest::Client;
use thiserror::Error;

/// Paths the gateway exposes only for the master / challenge plane.
///
/// Hitting any of these from a validator is a bug. Tests mount wiremock stubs
/// for these and assert zero hits + empty unmatched log.
pub const MASTER_ONLY_PATHS: &[&str] = &[
    "/v1/admin/seal",
    "/v1/admin/backends",
    "/v1/weights/raw",
    "/v1/master/status",
    "/challenge/",
];

/// Paths validators are allowed to call on the gateway (read plane).
pub const ALLOWED_GATEWAY_PATHS: &[&str] = &[
    "/healthz",
    "/readyz",
    "/v1/bundle/",
    "/v1/bundle/root/",
    "/v1/weights/latest",
];

/// Coordination client errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CoordinationError {
    /// Request path is master-only or otherwise not allowlisted.
    #[error("gateway path not allowed for validator: {path}")]
    PathNotAllowed {
        /// Requested path.
        path: String,
    },
    /// No gateway endpoint configured.
    #[error("gateway endpoint not configured")]
    NoGateway,
    /// HTTP transport failure.
    #[error("gateway transport: {0}")]
    Transport(String),
    /// Unexpected HTTP status.
    #[error("gateway HTTP {status} for {path}")]
    HttpStatus {
        /// Status code.
        status: u16,
        /// Path attempted.
        path: String,
    },
}

/// Returns true when `path` is a master-only route.
#[must_use]
pub fn is_master_only_path(path: &str) -> bool {
    let path = normalize_path(path);
    MASTER_ONLY_PATHS.iter().any(|p| {
        if p.ends_with('/') {
            path.starts_with(p) || path == p.trim_end_matches('/')
        } else {
            path == *p || path.starts_with(&format!("{p}/"))
        }
    })
}

/// Returns true when `path` is on the validator allowlist.
#[must_use]
pub fn is_allowed_gateway_path(path: &str) -> bool {
    if is_master_only_path(path) {
        return false;
    }
    let path = normalize_path(path);
    ALLOWED_GATEWAY_PATHS.iter().any(|p| {
        if p.ends_with('/') {
            path.starts_with(p) || path == p.trim_end_matches('/')
        } else {
            path == *p
        }
    })
}

fn normalize_path(path: &str) -> String {
    let p = path.split('?').next().unwrap_or(path);
    if p.is_empty() {
        return "/".to_owned();
    }
    if p.starts_with('/') {
        p.to_owned()
    } else {
        format!("/{p}")
    }
}

/// HTTP client that only issues allowlisted gateway requests.
#[derive(Debug, Clone)]
pub struct CoordinationClient {
    base_url: Option<String>,
    http: Client,
}

impl CoordinationClient {
    /// Build a client. `base_url` may be `None` (healthy without co-located gateway).
    ///
    /// # Errors
    ///
    /// Returns [`CoordinationError::Transport`] if the reqwest client cannot be built.
    pub fn new(base_url: Option<String>) -> Result<Self, CoordinationError> {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| CoordinationError::Transport(e.to_string()))?;
        Ok(Self {
            base_url: base_url.and_then(|u| {
                let t = u.trim().trim_end_matches('/').to_owned();
                if t.is_empty() {
                    None
                } else {
                    Some(t)
                }
            }),
            http,
        })
    }

    /// Whether a gateway base URL is configured.
    #[must_use]
    pub fn has_gateway(&self) -> bool {
        self.base_url.is_some()
    }

    /// GET an allowlisted path. Refuses master-only paths before any network I/O.
    ///
    /// # Errors
    ///
    /// Path not allowed, missing gateway, transport, or non-success status.
    pub async fn get_allowed(&self, path: &str) -> Result<reqwest::Response, CoordinationError> {
        let path = normalize_path(path);
        if is_master_only_path(&path) || !is_allowed_gateway_path(&path) {
            return Err(CoordinationError::PathNotAllowed { path });
        }
        let base = self.base_url.as_ref().ok_or(CoordinationError::NoGateway)?;
        let url = format!("{base}{path}");
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| CoordinationError::Transport(e.to_string()))?;
        let status = resp.status();
        if status.is_success() {
            Ok(resp)
        } else {
            Err(CoordinationError::HttpStatus {
                status: status.as_u16(),
                path,
            })
        }
    }

    /// Fetch sealed epoch bundle bytes: `GET /v1/bundle/{epoch}`.
    ///
    /// Returns raw SCALE `EpochBundleV1` octets (`application/octet-stream`).
    /// Does **not** fall back to another epoch (no last-known-good).
    ///
    /// # Errors
    ///
    /// Path not allowed, missing gateway, transport, or non-success status.
    pub async fn fetch_bundle(&self, epoch: u64) -> Result<Vec<u8>, CoordinationError> {
        let path = format!("/v1/bundle/{epoch}");
        let resp = self.get_allowed(&path).await?;
        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| CoordinationError::Transport(e.to_string()))
    }

    /// Lightweight coordination tick: probe `GET /v1/weights/latest` when a gateway is set.
    ///
    /// No-op (Ok) when no gateway is configured — validators stay healthy without one.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::get_allowed`] failures when a gateway is configured.
    pub async fn tick(&self) -> Result<(), CoordinationError> {
        if !self.has_gateway() {
            return Ok(());
        }
        let _ = self.get_allowed("/v1/weights/latest").await?;
        Ok(())
    }
}

/// Shared handle for readiness / runtime.
pub type SharedCoordination = Arc<CoordinationClient>;

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn master_only_paths_detected() {
        assert!(is_master_only_path("/v1/weights/raw"));
        assert!(is_master_only_path("/v1/admin/seal"));
        assert!(is_master_only_path("/v1/admin/backends"));
        assert!(is_master_only_path("/v1/master/status"));
        assert!(is_master_only_path("/challenge/foo/score"));
        assert!(!is_master_only_path("/v1/weights/latest"));
        assert!(!is_master_only_path("/v1/bundle/1"));
    }

    #[test]
    fn allowed_paths() {
        assert!(is_allowed_gateway_path("/v1/weights/latest"));
        assert!(is_allowed_gateway_path("/v1/bundle/42"));
        assert!(is_allowed_gateway_path("/v1/bundle/root/abcd"));
        assert!(is_allowed_gateway_path("/healthz"));
        assert!(!is_allowed_gateway_path("/v1/weights/raw"));
        assert!(!is_allowed_gateway_path("/secret"));
    }

    #[tokio::test]
    async fn refuses_master_only_without_network() {
        let client = CoordinationClient::new(Some("http://127.0.0.1:1".into())).unwrap();
        let err = client
            .get_allowed("/v1/weights/raw")
            .await
            .expect_err("must refuse");
        assert!(matches!(err, CoordinationError::PathNotAllowed { .. }));
    }

    #[tokio::test]
    async fn s2_never_calls_master_only_wiremock_unmatched_empty() {
        let server = MockServer::start().await;

        // Allowed path the coordination tick uses.
        Mock::given(method("GET"))
            .and(path("/v1/weights/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
            .mount(&server)
            .await;

        // Master-only stubs: if the validator hits them, received_requests is non-empty.
        for p in [
            "/v1/weights/raw",
            "/v1/admin/seal",
            "/v1/admin/backends",
            "/v1/master/status",
        ] {
            Mock::given(method("GET"))
                .and(path(p))
                .respond_with(ResponseTemplate::new(200))
                .mount(&server)
                .await;
            Mock::given(method("POST"))
                .and(path(p))
                .respond_with(ResponseTemplate::new(200))
                .mount(&server)
                .await;
        }

        let client = CoordinationClient::new(Some(server.uri())).unwrap();
        client.tick().await.expect("tick");

        // Explicit refuse still works.
        let _ = client.get_allowed("/v1/admin/seal").await.expect_err("no");

        let received = server.received_requests().await.expect("log");
        assert_eq!(
            received.len(),
            1,
            "only weights/latest should be hit, got {received:?}"
        );
        assert!(received[0].url.path().ends_with("/v1/weights/latest"));

        // No master-only path appears in the request log.
        for req in &received {
            let p = req.url.path();
            assert!(!is_master_only_path(p), "master-only path was called: {p}");
        }
    }

    #[tokio::test]
    async fn tick_noop_without_gateway() {
        let client = CoordinationClient::new(None).unwrap();
        client.tick().await.expect("noop");
        assert!(!client.has_gateway());
    }
}
