//! Env knobs for funding (no secrets in process env when avoidable).

use crate::types::FundingEconomics;

/// Runtime funding configuration.
#[derive(Debug, Clone)]
pub struct FundingConfig {
    /// When false, rent gate is a no-op (default for Prism until wallet live).
    pub require_funding: bool,
    /// Operator SS58 deposit address (placeholder until secrets wired).
    pub deposit_address: String,
    /// Quote TTL seconds.
    pub quote_ttl_secs: u64,
    /// Economics (rate/hours/buffer).
    pub economics: FundingEconomics,
    /// Optional admin bearer for `/v1/funding/admin/*`.
    pub admin_token: Option<String>,
}

impl Default for FundingConfig {
    fn default() -> Self {
        Self {
            require_funding: false,
            deposit_address: "REPLACE_ME_DEPOSIT_SS58".into(),
            quote_ttl_secs: 900,
            economics: FundingEconomics::prism_default(),
            admin_token: None,
        }
    }
}

impl FundingConfig {
    /// Load from env. Secrets (deposit address file, admin token file) optional.
    ///
    /// `PRISM_REQUIRE_LIUM_FUNDING` — `1`/`true` enables the rent gate (default off).
    #[must_use]
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.require_funding = env_truthy("PRISM_REQUIRE_LIUM_FUNDING");
        if let Ok(addr) = std::env::var("LIUM_FUNDING_DEPOSIT_ADDRESS") {
            if !addr.trim().is_empty() {
                cfg.deposit_address = addr.trim().to_owned();
            }
        } else if let Ok(path) = std::env::var("LIUM_FUNDING_DEPOSIT_ADDRESS_FILE") {
            if let Ok(s) = std::fs::read_to_string(&path) {
                let t = s.trim();
                if !t.is_empty() {
                    cfg.deposit_address = t.to_owned();
                }
            }
        }
        if let Ok(v) = std::env::var("LIUM_FUNDING_QUOTE_TTL_SECS") {
            if let Ok(n) = v.parse::<u64>() {
                cfg.quote_ttl_secs = n;
            }
        }
        cfg.economics = economics_from_env(cfg.economics);
        cfg.admin_token =
            read_secret_env("LIUM_FUNDING_ADMIN_TOKEN", "LIUM_FUNDING_ADMIN_TOKEN_FILE");
        cfg
    }
}

fn economics_from_env(mut e: FundingEconomics) -> FundingEconomics {
    if let Some(v) = env_f64("PRISM_FUNDING_RATE_USD_PER_HOUR")
        .or_else(|| env_f64("LIUM_FUNDING_RATE_USD_PER_HOUR"))
    {
        e.rate_usd_per_hour = v;
    }
    if let Some(v) = env_f64("PRISM_FUNDING_HOURS").or_else(|| env_f64("LIUM_FUNDING_HOURS")) {
        e.hours = v;
    }
    if let Some(v) = env_f64("LIUM_FUNDING_BUFFER") {
        e.buffer = v;
    }
    e
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|s| s.parse().ok())
}

fn env_truthy(key: &str) -> bool {
    matches!(
        std::env::var(key).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn read_secret_env(env_key: &str, file_key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(env_key) {
        let t = v.trim();
        if !t.is_empty() {
            return Some(t.to_owned());
        }
    }
    if let Ok(path) = std::env::var(file_key) {
        if let Ok(s) = std::fs::read_to_string(path) {
            let t = s.trim();
            if !t.is_empty() {
                return Some(t.to_owned());
            }
        }
    }
    None
}
