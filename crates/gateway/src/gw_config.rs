//! Gateway operator knobs: env keys, runtime config, hotkey resolution, and
//! the master-only owner-check mode. Re-exported through `gateway::*`.

use std::fmt;
use std::net::SocketAddr;
use std::str::FromStr;

use config::{Config, Role};
use gateway_registry::RegistryConfig;

use crate::tls::TlsConfig;
use crate::{GatewayError, DEFAULT_LISTEN};

/// Env keys specific to the gateway binary (in addition to `BASE_*` from config).
pub mod keys {
    /// Bind address (`host:port`), e.g. `0.0.0.0:8080`.
    pub const LISTEN: &str = "BASE_GATEWAY_LISTEN";
    /// Gateway hotkey as 64 hex chars (32 raw bytes). Must equal on-chain owner.
    pub const HOTKEY: &str = "BASE_GATEWAY_HOTKEY";
    /// Consecutive upstream failures before passive ejection (default 3).
    pub const FAIL_THRESHOLD: &str = "BASE_GATEWAY_FAIL_THRESHOLD";
    /// Ejection cooldown seconds before re-admission (default 30).
    pub const COOLDOWN_SECS: &str = "BASE_GATEWAY_COOLDOWN_SECS";
    /// Owner-signed trust root directory (`challenges.toml` + `measurements.toml`).
    pub const TRUST_ROOT_DIR: &str = "BASE_TRUST_ROOT_DIR";
    /// Gateway bundle-signing mini-secret (64 hex).
    pub const GATEWAY_SK: &str = "BASE_GATEWAY_SK";
    /// Path to gateway mini-secret (32 raw bytes or 64 hex).
    pub const GATEWAY_SK_FILE: &str = "BASE_GATEWAY_SK_FILE";
    /// `frame-ancestors` allowlist for the viewer lockdown CSP the gateway
    /// re-applies to `/challenge/*/v1/view/*` responses (defense in depth).
    /// Defaults to [`design_sanitize::default_frame_ancestors`].
    pub const VIEW_FRAME_ANCESTORS: &str = "BASE_GATEWAY_VIEW_FRAME_ANCESTORS";

    pub use crate::tls::keys as tls;
}

/// Runtime knobs required to start the gateway server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayConfig {
    /// TCP bind address (bound only after master check passes).
    pub listen: SocketAddr,
    /// Subnet netuid used for `subnet_owner_hotkey`.
    pub netuid: u16,
    /// Configured gateway hotkey (32-byte sr25519 public key).
    pub hotkey: [u8; 32],
    /// Passive ejection / re-admission knobs.
    pub registry: RegistryConfig,
    /// Sole TLS owner config (D20); real ACME in task 42.
    pub tls: TlsConfig,
    /// How strictly to treat an on-chain subnet-owner mismatch.
    pub owner_check: OwnerCheck,
}

impl GatewayConfig {
    /// Build from a validated [`Config`] plus gateway-specific listen/hotkey.
    ///
    /// # Errors
    ///
    /// Role is not gateway, or listen/hotkey parse failures.
    pub fn from_base(
        base: &Config,
        listen: SocketAddr,
        hotkey: [u8; 32],
        registry: RegistryConfig,
        tls: TlsConfig,
    ) -> Result<Self, GatewayError> {
        if base.role != Role::Gateway {
            return Err(GatewayError::Config(format!(
                "role must be gateway, got {}",
                base.role
            )));
        }
        Ok(Self {
            listen,
            netuid: base.netuid.get(),
            hotkey,
            registry,
            tls,
            owner_check: OwnerCheck::from_env(),
        })
    }

    /// Load layered config + gateway env knobs.
    ///
    /// # Errors
    ///
    /// Config validation, missing hotkey, or bad listen/hotkey encoding.
    pub fn from_env() -> Result<Self, GatewayError> {
        let base = config::load().map_err(|e| GatewayError::Config(e.to_string()))?;
        let listen_raw = std::env::var(keys::LISTEN).unwrap_or_else(|_| DEFAULT_LISTEN.to_owned());
        let listen = SocketAddr::from_str(&listen_raw).map_err(|e| {
            GatewayError::Config(format!("invalid {} `{listen_raw}`: {e}", keys::LISTEN))
        })?;
        let hotkey = resolve_gateway_hotkey()?;
        let registry = registry_config_from_env()?;
        let tls = TlsConfig::from_env();
        Self::from_base(&base, listen, hotkey, registry, tls)
    }
}

fn registry_config_from_env() -> Result<RegistryConfig, GatewayError> {
    let mut cfg = RegistryConfig::default();
    if let Ok(raw) = std::env::var(keys::FAIL_THRESHOLD) {
        cfg.failure_threshold = raw.parse().map_err(|e| {
            GatewayError::Config(format!("invalid {} `{raw}`: {e}", keys::FAIL_THRESHOLD))
        })?;
        if cfg.failure_threshold == 0 {
            return Err(GatewayError::Config(format!(
                "{} must be >= 1",
                keys::FAIL_THRESHOLD
            )));
        }
    }
    if let Ok(raw) = std::env::var(keys::COOLDOWN_SECS) {
        let secs: u64 = raw.parse().map_err(|e| {
            GatewayError::Config(format!("invalid {} `{raw}`: {e}", keys::COOLDOWN_SECS))
        })?;
        cfg.cooldown = std::time::Duration::from_secs(secs);
    }
    Ok(cfg)
}

/// Resolve the gateway hotkey from a Bittensor wallet, mnemonic file, or hex.
///
/// Delegates to [`keystore::resolve_public_key_from_env`] with the
/// `BASE_GATEWAY` prefix, so operators may set any of:
/// `BASE_GATEWAY_WALLET` (+ `BASE_GATEWAY_WALLET_HOTKEY`), `BASE_WALLET_NAME`,
/// `BASE_GATEWAY_MNEMONIC_FILE`, `BASE_GATEWAY_SK_FILE`, or the legacy
/// `BASE_GATEWAY_HOTKEY` (64 hex chars or an SS58 address).
///
/// # Errors
///
/// No source configured, or the configured source failed to load.
pub fn resolve_gateway_hotkey() -> Result<[u8; 32], GatewayError> {
    match keystore::resolve_public_key_from_env("BASE_GATEWAY") {
        Ok(Some(pk)) => Ok(pk),
        Ok(None) => Err(GatewayError::Config(format!(
            "no gateway hotkey configured: set BASE_GATEWAY_WALLET (Bittensor wallet), \
             BASE_GATEWAY_MNEMONIC_FILE, BASE_GATEWAY_SK_FILE, or {}",
            keys::HOTKEY
        ))),
        Err(e) => Err(GatewayError::Config(format!(
            "gateway hotkey resolution failed: {e}"
        ))),
    }
}

/// Parse a 32-byte hotkey from lowercase/uppercase hex (optional `0x` prefix).
///
/// # Errors
///
/// Wrong length or non-hex input.
pub fn parse_hotkey_hex(s: &str) -> Result<[u8; 32], GatewayError> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let bytes = hex::decode(s)
        .map_err(|e| GatewayError::Config(format!("invalid {} hex: {e}", keys::HOTKEY)))?;
    let arr: [u8; 32] = bytes.try_into().map_err(|v: Vec<u8>| {
        GatewayError::Config(format!(
            "{} must be 32 bytes (64 hex chars), got {} bytes",
            keys::HOTKEY,
            v.len()
        ))
    })?;
    Ok(arr)
}

/// Format hotkey bytes as lowercase hex (no `0x`) for structured logs.
#[must_use]
pub fn hotkey_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Env flag restoring the fail-closed master-only check.
pub const REQUIRE_OWNER_ENV: &str = "BASE_GATEWAY_REQUIRE_OWNER";

/// How strictly to treat an on-chain subnet-owner mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnerCheck {
    /// Log the result and start anyway (default).
    Advisory,
    /// Refuse to start unless the configured hotkey is the on-chain owner.
    Required,
}

impl OwnerCheck {
    /// Read the mode from [`REQUIRE_OWNER_ENV`]; absent or falsy → advisory.
    #[must_use]
    pub fn from_env() -> Self {
        let on = std::env::var(REQUIRE_OWNER_ENV)
            .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes"));
        if on {
            Self::Required
        } else {
            Self::Advisory
        }
    }
}

impl fmt::Display for GatewayConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GatewayConfig {{ listen: {}, netuid: {}, hotkey: {}, tls: {} }}",
            self.listen,
            self.netuid,
            hotkey_hex(&self.hotkey),
            self.tls.mode_label()
        )
    }
}
