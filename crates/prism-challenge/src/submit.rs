//! Gateway raw-weights submit client (thin wrap over `challenge-common`).

use std::collections::BTreeMap;
use std::time::Duration;

use bundle::LeafV1;
use challenge_common::{
    submit_signed_leaf_set as submit_common, GatewayClient as CommonGatewayClient,
    GatewayClientConfig as CommonGatewayClientConfig, SubmitError as CommonSubmitError,
    SubmitOutcome as CommonSubmitOutcome, DEFAULT_MAX_RETRIES, DRY_RUN_BASE_URL,
};
use crypto::KEY_LEN;
use thiserror::Error;

/// Gateway client config (prism-facing field names).
#[derive(Debug, Clone)]
pub struct GatewayClientConfig {
    /// Base URL, or [`DRY_RUN_BASE_URL`] for no network.
    pub base_url: String,
    /// Max retries after the first attempt. Converted to common `max_attempts = max_retries + 1`
    /// to match the prior `0..=max_retries` loop.
    pub max_retries: u32,
}

impl Default for GatewayClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8080".into(),
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
}

impl From<GatewayClientConfig> for CommonGatewayClientConfig {
    fn from(cfg: GatewayClientConfig) -> Self {
        CommonGatewayClientConfig {
            base_url: cfg.base_url,
            max_attempts: cfg.max_retries.saturating_add(1).max(1),
            backoff: Duration::from_millis(50),
        }
    }
}

/// Submit errors (prism-facing).
#[derive(Debug, Error)]
pub enum SubmitError {
    #[error("http: {0}")]
    Http(String),
    #[error("gateway rejected: {0}")]
    Rejected(String),
    #[error("serialize: {0}")]
    Serialize(String),
}

impl From<CommonSubmitError> for SubmitError {
    fn from(err: CommonSubmitError) -> Self {
        match err {
            CommonSubmitError::Transport(s) => Self::Http(s),
            CommonSubmitError::Http { status, body } => {
                Self::Rejected(format!("status {status}: {body}"))
            }
            CommonSubmitError::Serialize(s) => Self::Serialize(s),
        }
    }
}

/// Outcome of a submit attempt (prism-facing aggregate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitOutcome {
    Accepted,
    DryRun { leaf_count: usize },
}

/// Thin HTTP client for `POST /v1/weights/raw`.
#[derive(Debug, Clone)]
pub struct GatewayClient {
    inner: CommonGatewayClient,
}

impl GatewayClient {
    /// Build client.
    ///
    /// # Errors
    /// HTTP client build failure.
    pub fn new(cfg: GatewayClientConfig) -> Result<Self, SubmitError> {
        let inner = CommonGatewayClient::new(cfg.into())?;
        Ok(Self { inner })
    }

    /// Dry-run: no network, reports leaf count.
    #[must_use]
    pub fn dry_run(leaves: &BTreeMap<[u8; KEY_LEN], LeafV1>) -> SubmitOutcome {
        match CommonGatewayClient::dry_run(leaves) {
            CommonSubmitOutcome::DryRun { leaf_count } => SubmitOutcome::DryRun { leaf_count },
            _ => SubmitOutcome::DryRun {
                leaf_count: leaves.len(),
            },
        }
    }

    /// Whether this client skips network I/O.
    #[must_use]
    pub fn is_dry_run(&self) -> bool {
        self.inner.is_dry_run()
    }
}

/// Submit signed leaves (or dry-run when `base_url` is `dry-run`).
///
/// `challenge_id` / `epoch` are accepted for call-site compatibility; leaves already
/// carry both.
///
/// # Errors
/// HTTP / rejection.
pub async fn submit_signed_leaf_set(
    client: &GatewayClient,
    _challenge_id: &str,
    _epoch: u64,
    leaves: &BTreeMap<[u8; KEY_LEN], LeafV1>,
) -> Result<SubmitOutcome, SubmitError> {
    if client.is_dry_run() || client.inner.base_url() == DRY_RUN_BASE_URL {
        return Ok(GatewayClient::dry_run(leaves));
    }
    let _outcomes = submit_common(&client.inner, leaves).await?;
    Ok(SubmitOutcome::Accepted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_counts() {
        let m = BTreeMap::new();
        assert_eq!(
            GatewayClient::dry_run(&m),
            SubmitOutcome::DryRun { leaf_count: 0 }
        );
    }
}
