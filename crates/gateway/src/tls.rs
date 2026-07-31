//! TLS ownership for the gateway process (D20).
//!
//! This process is the **sole** TLS terminator for base. No Caddy, Traefik, or
//! nginx may sit in front. Production ACME (DNS-01 via Cloudflare + `BASE_DOMAIN`)
//! lands in **task 42** using [`rustls-acme`](https://docs.rs/rustls-acme).
//!
//! Until task 42, operators may:
//! - run plain HTTP on a private network (default), or
//! - supply static PEM paths via env (stub — accepted and logged, not yet wired
//!   into `axum_server` / `tokio-rustls`; binding remains cleartext so unit tests
//!   stay hermetic).
//!
//! ## Task 42 checklist (`rustls-acme`)
//!
//! 1. Depend on `rustls-acme` + `tokio-rustls` + `axum-server` (or hyper accept loop).
//! 2. Require `BASE_DOMAIN`; abort if unset (D25).
//! 3. DNS-01 challenge with Cloudflare token (`/root/.cf_api_token` in deploy).
//! 4. Persist certs on a mounted volume; restart must not re-issue unnecessarily.
//! 5. Serve HTTPS only; keep master-only check **before** any listener bind (D3).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Env keys for the TLS stub (task 42 will extend these).
pub mod keys {
    /// When `1`/`true`, prefer TLS mode (task 42 wires real accept).
    pub const ENABLED: &str = "BASE_GATEWAY_TLS";
    /// Path to PEM certificate chain (static stub).
    pub const CERT_PATH: &str = "BASE_GATEWAY_TLS_CERT";
    /// Path to PEM private key (static stub).
    pub const KEY_PATH: &str = "BASE_GATEWAY_TLS_KEY";
    /// Directory for ACME cache (task 42 / rustls-acme).
    pub const ACME_CACHE: &str = "BASE_GATEWAY_ACME_CACHE";
}

/// TLS configuration owned exclusively by the gateway (D20).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Operator requested TLS termination in this process.
    pub enabled: bool,
    /// Optional static certificate path (pre-ACME / test fixtures).
    pub cert_path: Option<PathBuf>,
    /// Optional static private key path.
    pub key_path: Option<PathBuf>,
    /// Future ACME cache directory (rustls-acme, task 42).
    pub acme_cache_dir: Option<PathBuf>,
}

impl TlsConfig {
    /// Load from environment. Missing vars → cleartext defaults.
    #[must_use]
    pub fn from_env() -> Self {
        let enabled = std::env::var(keys::ENABLED)
            .is_ok_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"));
        Self {
            enabled,
            cert_path: std::env::var_os(keys::CERT_PATH).map(PathBuf::from),
            key_path: std::env::var_os(keys::KEY_PATH).map(PathBuf::from),
            acme_cache_dir: std::env::var_os(keys::ACME_CACHE).map(PathBuf::from),
        }
    }

    /// Whether static PEM paths are both present (still not activated until task 42).
    #[must_use]
    pub fn has_static_material(&self) -> bool {
        self.cert_path.is_some() && self.key_path.is_some()
    }

    /// Human-readable mode for logs.
    #[must_use]
    pub fn mode_label(&self) -> &'static str {
        if self.enabled && self.has_static_material() {
            "tls_static_stub"
        } else if self.enabled {
            "tls_requested_pending_task42_rustls_acme"
        } else {
            "cleartext"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_cleartext_sole_owner_stub() {
        let t = TlsConfig::default();
        assert!(!t.enabled);
        assert_eq!(t.mode_label(), "cleartext");
        // Documented contract: gateway owns TLS; stub does not pull in reverse proxies.
        assert!(!t.has_static_material());
    }

    #[test]
    fn serde_roundtrip_has_no_foreign_terminators() {
        let t = TlsConfig {
            enabled: true,
            cert_path: Some(PathBuf::from("/certs/fullchain.pem")),
            key_path: Some(PathBuf::from("/certs/privkey.pem")),
            acme_cache_dir: Some(PathBuf::from("/var/lib/gbase/acme")),
        };
        let v = serde_json::to_value(&t).unwrap();
        assert_eq!(v["enabled"], true);
        assert!(v.get("caddy").is_none());
        assert!(v.get("nginx").is_none());
        assert!(v.get("traefik").is_none());
    }
}
