//! Layered configuration loading: defaults → TOML → `BASE_*` env.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::Deserialize;

use crate::error::{Error, Issue, ValidationReport};
use crate::netuid::NetUid;
use crate::role::Role;
use crate::Config;

/// Default chain WebSocket endpoint (Bittensor test finney).
pub const DEFAULT_CHAIN_ENDPOINT: &str = "wss://test.finney.opentensor.ai:443";

/// Default epoch length in blocks (typical subnet tempo order of magnitude).
pub const DEFAULT_EPOCH_LENGTH: u64 = 360;

/// Default quarantine surviving-share floor (D6), basis points.
pub const DEFAULT_MIN_SHARE_MASS_BPS: u16 = 5_000;

/// Default trust-root dual-accept window in epochs (D21).
pub const DEFAULT_ROTATION_EPOCHS: u32 = 3;

/// Default minimum peer sample for cross-check (D26).
pub const DEFAULT_MIN_PEER_SAMPLE: u32 = 1;

/// Default max DCAP collateral age before park (seconds) — 7 days.
pub const DEFAULT_MAX_COLLATERAL_AGE_SECS: u64 = 7 * 24 * 60 * 60;

/// Maximum legal emission-share mass in basis points.
pub const MAX_SHARE_MASS_BPS: u16 = 10_000;

/// Env / file key names (stable contract for operators).
pub mod keys {
    /// Process role.
    pub const ROLE: &str = "BASE_ROLE";
    /// Subnet netuid.
    pub const NETUID: &str = "BASE_NETUID";
    /// Chain WebSocket URL.
    pub const CHAIN_ENDPOINT: &str = "BASE_CHAIN_ENDPOINT";
    /// Gateway base URL (validators / miners).
    pub const GATEWAY_ENDPOINT: &str = "BASE_GATEWAY_ENDPOINT";
    /// Postgres URL.
    pub const DATABASE_URL: &str = "BASE_DATABASE_URL";
    /// Path to a file containing the Postgres URL (secret mount).
    pub const DATABASE_URL_FILE: &str = "BASE_DATABASE_URL_FILE";
    /// Epoch length in blocks.
    pub const EPOCH_LENGTH: &str = "BASE_EPOCH_LENGTH";
    /// Quarantine mass floor (bps).
    pub const MIN_SHARE_MASS_BPS: &str = "BASE_MIN_SHARE_MASS_BPS";
    /// Trust-root rotation dual-accept epochs.
    pub const ROTATION_EPOCHS: &str = "BASE_ROTATION_EPOCHS";
    /// Minimum peer sample size.
    pub const MIN_PEER_SAMPLE: &str = "BASE_MIN_PEER_SAMPLE";
    /// Max collateral age seconds.
    pub const MAX_COLLATERAL_AGE_SECS: &str = "BASE_MAX_COLLATERAL_AGE_SECS";
    /// Cloudflare zone / public domain (D25).
    pub const DOMAIN: &str = "BASE_DOMAIN";
    /// Optional path to TOML config file.
    pub const CONFIG_PATH: &str = "BASE_CONFIG";
}

/// Optional overrides from a single layer (TOML file or env).
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
struct Layer {
    role: Option<String>,
    netuid: Option<u64>,
    chain_endpoint: Option<String>,
    gateway_endpoint: Option<String>,
    database_url: Option<String>,
    database_url_file: Option<String>,
    epoch_length: Option<u64>,
    min_share_mass_bps: Option<u64>,
    rotation_epochs: Option<u64>,
    min_peer_sample: Option<u64>,
    max_collateral_age_secs: Option<u64>,
    domain: Option<String>,
}

/// Mutable merge state before validation.
#[derive(Debug, Clone, Default)]
struct Builder {
    role: Option<Role>,
    netuid: Option<NetUid>,
    chain_endpoint: Option<String>,
    gateway_endpoint: Option<String>,
    database_url: Option<String>,
    database_url_file: Option<PathBuf>,
    epoch_length: Option<u64>,
    min_share_mass_bps: Option<u16>,
    rotation_epochs: Option<u32>,
    min_peer_sample: Option<u32>,
    max_collateral_age_secs: Option<u64>,
    domain: Option<String>,
    early_issues: Vec<Issue>,
}

impl Builder {
    fn apply_defaults(&mut self) {
        self.chain_endpoint = Some(DEFAULT_CHAIN_ENDPOINT.to_owned());
        self.epoch_length = Some(DEFAULT_EPOCH_LENGTH);
        self.min_share_mass_bps = Some(DEFAULT_MIN_SHARE_MASS_BPS);
        self.rotation_epochs = Some(DEFAULT_ROTATION_EPOCHS);
        self.min_peer_sample = Some(DEFAULT_MIN_PEER_SAMPLE);
        self.max_collateral_age_secs = Some(DEFAULT_MAX_COLLATERAL_AGE_SECS);
    }

    fn apply_layer(&mut self, layer: Layer) {
        if let Some(raw) = layer.role {
            match Role::from_str(&raw) {
                Ok(role) => self.role = Some(role),
                Err(e) => self.early_issues.push(Issue::InvalidValue {
                    field: "role",
                    reason: e.to_string(),
                }),
            }
        }
        if let Some(raw) = layer.netuid {
            match u16::try_from(raw) {
                Ok(v) => self.netuid = Some(NetUid::new(v)),
                Err(_) => self.early_issues.push(Issue::InvalidValue {
                    field: "netuid",
                    reason: format!("{raw} does not fit u16"),
                }),
            }
        }
        if let Some(v) = layer.chain_endpoint {
            self.chain_endpoint = Some(v);
        }
        if let Some(v) = layer.gateway_endpoint {
            self.gateway_endpoint = empty_to_none(&v);
        }
        if let Some(v) = layer.database_url {
            self.database_url = empty_to_none(&v);
        }
        if let Some(v) = layer.database_url_file {
            self.database_url_file = empty_to_none(&v).map(PathBuf::from);
        }
        if let Some(v) = layer.epoch_length {
            self.epoch_length = Some(v);
        }
        if let Some(raw) = layer.min_share_mass_bps {
            match u16::try_from(raw) {
                Ok(v) => self.min_share_mass_bps = Some(v),
                Err(_) => self.early_issues.push(Issue::InvalidValue {
                    field: "min_share_mass_bps",
                    reason: format!("{raw} does not fit u16"),
                }),
            }
        }
        if let Some(raw) = layer.rotation_epochs {
            match u32::try_from(raw) {
                Ok(v) => self.rotation_epochs = Some(v),
                Err(_) => self.early_issues.push(Issue::InvalidValue {
                    field: "rotation_epochs",
                    reason: format!("{raw} does not fit u32"),
                }),
            }
        }
        if let Some(raw) = layer.min_peer_sample {
            match u32::try_from(raw) {
                Ok(v) => self.min_peer_sample = Some(v),
                Err(_) => self.early_issues.push(Issue::InvalidValue {
                    field: "min_peer_sample",
                    reason: format!("{raw} does not fit u32"),
                }),
            }
        }
        if let Some(v) = layer.max_collateral_age_secs {
            self.max_collateral_age_secs = Some(v);
        }
        if let Some(v) = layer.domain {
            self.domain = empty_to_none(&v);
        }
    }

    fn apply_env_map(&mut self, env: &BTreeMap<String, String>) {
        let layer = Layer {
            role: env.get(keys::ROLE).cloned(),
            netuid: env
                .get(keys::NETUID)
                .and_then(|s| parse_u64_field("netuid", s, &mut self.early_issues)),
            chain_endpoint: env.get(keys::CHAIN_ENDPOINT).cloned(),
            gateway_endpoint: env.get(keys::GATEWAY_ENDPOINT).cloned(),
            database_url: env.get(keys::DATABASE_URL).cloned(),
            database_url_file: env.get(keys::DATABASE_URL_FILE).cloned(),
            epoch_length: env
                .get(keys::EPOCH_LENGTH)
                .and_then(|s| parse_u64_field("epoch_length", s, &mut self.early_issues)),
            min_share_mass_bps: env
                .get(keys::MIN_SHARE_MASS_BPS)
                .and_then(|s| parse_u64_field("min_share_mass_bps", s, &mut self.early_issues)),
            rotation_epochs: env
                .get(keys::ROTATION_EPOCHS)
                .and_then(|s| parse_u64_field("rotation_epochs", s, &mut self.early_issues)),
            min_peer_sample: env
                .get(keys::MIN_PEER_SAMPLE)
                .and_then(|s| parse_u64_field("min_peer_sample", s, &mut self.early_issues)),
            max_collateral_age_secs: env.get(keys::MAX_COLLATERAL_AGE_SECS).and_then(|s| {
                parse_u64_field("max_collateral_age_secs", s, &mut self.early_issues)
            }),
            domain: env.get(keys::DOMAIN).cloned(),
        };
        self.apply_layer(layer);
    }

    fn finish(self) -> Result<Config, ValidationReport> {
        let mut issues = self.early_issues;

        let role = copy_required(self.role, "role", &mut issues);
        let netuid = copy_required(self.netuid, "netuid", &mut issues);
        let chain_endpoint =
            take_required_string(self.chain_endpoint, "chain_endpoint", &mut issues);
        let epoch_length = copy_required(self.epoch_length, "epoch_length", &mut issues);
        let min_share_mass_bps =
            copy_required(self.min_share_mass_bps, "min_share_mass_bps", &mut issues);
        let rotation_epochs = copy_required(self.rotation_epochs, "rotation_epochs", &mut issues);
        let min_peer_sample = copy_required(self.min_peer_sample, "min_peer_sample", &mut issues);
        let max_collateral_age_secs = copy_required(
            self.max_collateral_age_secs,
            "max_collateral_age_secs",
            &mut issues,
        );

        if let Some(bps) = min_share_mass_bps {
            if bps > MAX_SHARE_MASS_BPS {
                issues.push(Issue::MinShareMassBpsOutOfRange { value: bps });
            }
        }
        if epoch_length == Some(0) {
            issues.push(Issue::EpochLengthZero);
        }

        let has_db_url = self.database_url.as_ref().is_some_and(|s| !s.is_empty());
        let has_db_file = self.database_url_file.is_some();
        if has_db_url && has_db_file {
            issues.push(Issue::MutuallyExclusiveDatabaseUrlSources);
        }

        if let Some(role) = role {
            if role.requires_domain() {
                let domain_ok = self.domain.as_ref().is_some_and(|d| !d.trim().is_empty());
                if !domain_ok {
                    issues.push(Issue::DomainRequiredForGateway);
                }
            }
            if role.requires_database() && !has_db_url && !has_db_file {
                issues.push(Issue::DatabaseRequired {
                    role: role.to_string(),
                });
            }
        }

        if let Some(report) = ValidationReport::from_issues(issues) {
            return Err(report);
        }

        match (
            role,
            netuid,
            chain_endpoint,
            epoch_length,
            min_share_mass_bps,
            rotation_epochs,
            min_peer_sample,
            max_collateral_age_secs,
        ) {
            (
                Some(role),
                Some(netuid),
                Some(chain_endpoint),
                Some(epoch_length),
                Some(min_share_mass_bps),
                Some(rotation_epochs),
                Some(min_peer_sample),
                Some(max_collateral_age_secs),
            ) => Ok(Config {
                role,
                netuid,
                chain_endpoint,
                gateway_endpoint: self.gateway_endpoint,
                database_url: self.database_url,
                database_url_file: self.database_url_file,
                epoch_length,
                min_share_mass_bps,
                rotation_epochs,
                min_peer_sample,
                max_collateral_age_secs,
                domain: self.domain,
            }),
            _ => Err(ValidationReport::invariant_violation()),
        }
    }
}

fn empty_to_none(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_owned())
    }
}

fn copy_required<T: Copy>(
    value: Option<T>,
    field: &'static str,
    issues: &mut Vec<Issue>,
) -> Option<T> {
    if value.is_none() {
        issues.push(Issue::MissingField { field });
    }
    value
}

fn take_required_string(
    value: Option<String>,
    field: &'static str,
    issues: &mut Vec<Issue>,
) -> Option<String> {
    match value {
        Some(s) if !s.trim().is_empty() => Some(s.trim().to_owned()),
        _ => {
            issues.push(Issue::MissingField { field });
            None
        }
    }
}

fn parse_u64_field(field: &'static str, raw: &str, issues: &mut Vec<Issue>) -> Option<u64> {
    match raw.trim().parse::<u64>() {
        Ok(v) => Some(v),
        Err(e) => {
            issues.push(Issue::InvalidValue {
                field,
                reason: e.to_string(),
            });
            None
        }
    }
}

fn layer_from_toml_str(text: &str) -> Result<Layer, Issue> {
    toml::from_str(text).map_err(|e| Issue::InvalidValue {
        field: "base.toml",
        reason: e.to_string(),
    })
}

fn layer_from_toml_path(path: &Path) -> Result<Layer, Issue> {
    let text = fs::read_to_string(path).map_err(|e| Issue::TomlIo {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    match layer_from_toml_str(&text) {
        Ok(layer) => Ok(layer),
        Err(Issue::InvalidValue { reason, .. }) => Err(Issue::TomlIo {
            path: path.to_path_buf(),
            message: reason,
        }),
        Err(other) => Err(other),
    }
}

/// Load configuration from defaults, an optional TOML path, and an env map.
///
/// Layer order (later wins): **defaults → TOML file → `env` map**.
///
/// # Errors
///
/// Returns [`Error::Validation`] with **all** problems discovered.
pub fn load_from(
    toml_path: Option<&Path>,
    env: &BTreeMap<String, String>,
) -> Result<Config, Error> {
    let mut builder = Builder::default();
    builder.apply_defaults();

    if let Some(path) = toml_path {
        match layer_from_toml_path(path) {
            Ok(layer) => builder.apply_layer(layer),
            Err(issue) => builder.early_issues.push(issue),
        }
    }

    builder.apply_env_map(env);
    builder.finish().map_err(Error::Validation)
}

/// Load from defaults, optional TOML string, and env map (filesystem-free tests).
///
/// # Errors
///
/// Returns [`Error::Validation`] with **all** problems discovered.
pub fn load_from_toml_str(
    toml_text: Option<&str>,
    env: &BTreeMap<String, String>,
) -> Result<Config, Error> {
    let mut builder = Builder::default();
    builder.apply_defaults();

    if let Some(text) = toml_text {
        match layer_from_toml_str(text) {
            Ok(layer) => builder.apply_layer(layer),
            Err(issue) => builder.early_issues.push(issue),
        }
    }

    builder.apply_env_map(env);
    builder.finish().map_err(Error::Validation)
}

/// Collect `BASE_*` variables from the process environment.
#[must_use]
pub fn base_env_from_process() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("BASE_"))
        .collect()
}

/// Resolve the TOML path: `BASE_CONFIG` env, else `./base.toml` if it exists.
#[must_use]
pub fn resolve_toml_path(env: &BTreeMap<String, String>) -> Option<PathBuf> {
    if let Some(p) = env.get(keys::CONFIG_PATH) {
        return Some(PathBuf::from(p));
    }
    let default = PathBuf::from("base.toml");
    if default.is_file() {
        Some(default)
    } else {
        None
    }
}

/// Load using the process environment and default path resolution.
///
/// # Errors
///
/// Returns [`Error::Validation`] with **all** problems discovered.
pub fn load() -> Result<Config, Error> {
    let env = base_env_from_process();
    let path = resolve_toml_path(&env);
    load_from(path.as_deref(), &env)
}
