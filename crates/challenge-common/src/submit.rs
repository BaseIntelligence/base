//! Gateway `POST /v1/weights/raw` client with 5xx retry and (challenge, epoch, miner) idempotency.

use std::time::Duration;

use bundle::{LeafV1, ScoreOrAbsence};
use crypto::KEY_LEN;
use serde::Serialize;
use thiserror::Error;

/// Default max attempts on 5xx / transport errors (including the first try).
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Sentinel `base_url` that disables network I/O.
pub const DRY_RUN_BASE_URL: &str = "dry-run";

/// Client configuration.
#[derive(Debug, Clone)]
pub struct GatewayClientConfig {
    /// Base URL, e.g. `http://127.0.0.1:8080` (no trailing slash required).
    /// Use [`DRY_RUN_BASE_URL`] to skip network I/O.
    pub base_url: String,
    /// Max attempts on retryable failures.
    pub max_attempts: u32,
    /// Base backoff between retries.
    pub backoff: Duration,
}

impl Default for GatewayClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8080".into(),
            max_attempts: DEFAULT_MAX_RETRIES,
            backoff: Duration::from_millis(50),
        }
    }
}

/// Outcome of a single leaf submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// Fresh accept (HTTP 202).
    Accepted,
    /// Already stored (HTTP 409) — treated as success for idempotent retry.
    AlreadyPresent,
    /// No network: dry-run mode (`base_url == "dry-run"`).
    DryRun {
        /// Number of leaves that would have been posted.
        leaf_count: usize,
    },
}

/// Submit errors (non-retryable after exhaustion, or client/auth errors).
#[derive(Debug, Error)]
pub enum SubmitError {
    /// HTTP client / transport after retries exhausted.
    #[error("gateway transport: {0}")]
    Transport(String),
    /// Non-success HTTP status that is not 202/409.
    #[error("gateway HTTP {status}: {body}")]
    Http {
        /// Status code.
        status: u16,
        /// Response body snippet.
        body: String,
    },
    /// JSON serialize failure.
    #[error("serialize request: {0}")]
    Serialize(String),
}

/// HTTP client for signed raw-weight leaves.
#[derive(Debug, Clone)]
pub struct GatewayClient {
    http: reqwest::Client,
    cfg: GatewayClientConfig,
}

impl GatewayClient {
    /// Build a client.
    ///
    /// # Errors
    ///
    /// When the underlying `reqwest::Client` cannot be built.
    pub fn new(cfg: GatewayClientConfig) -> Result<Self, SubmitError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| SubmitError::Transport(e.to_string()))?;
        Ok(Self { http, cfg })
    }

    /// Whether this client is in dry-run mode (no network).
    #[must_use]
    pub fn is_dry_run(&self) -> bool {
        self.cfg.base_url == DRY_RUN_BASE_URL
    }

    /// Configured base URL.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.cfg.base_url
    }

    /// Dry-run outcome for a leaf map (no network).
    #[must_use]
    pub fn dry_run(leaves: &std::collections::BTreeMap<[u8; KEY_LEN], LeafV1>) -> SubmitOutcome {
        SubmitOutcome::DryRun {
            leaf_count: leaves.len(),
        }
    }

    /// POST one signed leaf. Retries 5xx and transport errors.
    ///
    /// Idempotency: HTTP 409 (identical digest already present) is success.
    /// Tip re-emits with a changed score/digest return 202 (supersede) and
    /// are also success. Callers may change tip leaves between ticks.
    ///
    /// # Errors
    ///
    /// See [`SubmitError`].
    pub async fn submit_leaf(&self, leaf: &LeafV1) -> Result<SubmitOutcome, SubmitError> {
        if self.is_dry_run() {
            return Ok(SubmitOutcome::DryRun { leaf_count: 1 });
        }
        let body = leaf_to_json(leaf).map_err(SubmitError::Serialize)?;
        let url = format!("{}/v1/weights/raw", self.cfg.base_url.trim_end_matches('/'));
        let attempts = self.cfg.max_attempts.max(1);
        let mut last_err = SubmitError::Transport("no attempts".into());

        for attempt in 0..attempts {
            if attempt > 0 {
                tokio::time::sleep(self.cfg.backoff.saturating_mul(attempt)).await;
            }
            match self.post_once(&url, &body).await {
                Ok(outcome) => return Ok(outcome),
                Err(e) if is_retryable(&e) && attempt + 1 < attempts => {
                    last_err = e;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err)
    }

    /// Submit every leaf; stop on first hard error.
    ///
    /// # Errors
    ///
    /// First non-recoverable [`SubmitError`].
    pub async fn submit_all(&self, leaves: &[LeafV1]) -> Result<Vec<SubmitOutcome>, SubmitError> {
        if self.is_dry_run() {
            return Ok(vec![SubmitOutcome::DryRun {
                leaf_count: leaves.len(),
            }]);
        }
        let mut out = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            out.push(self.submit_leaf(leaf).await?);
        }
        Ok(out)
    }

    async fn post_once(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<SubmitOutcome, SubmitError> {
        let resp = self
            .http
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(|e| SubmitError::Transport(e.to_string()))?;
        let status = resp.status();
        let code = status.as_u16();
        if code == 202 {
            return Ok(SubmitOutcome::Accepted);
        }
        if code == 409 {
            return Ok(SubmitOutcome::AlreadyPresent);
        }
        let body_text = resp.text().await.unwrap_or_default();
        Err(SubmitError::Http {
            status: code,
            body: body_text.chars().take(512).collect(),
        })
    }
}

fn is_retryable(err: &SubmitError) -> bool {
    match err {
        SubmitError::Transport(_) => true,
        SubmitError::Http { status, .. } => (500..600).contains(status),
        SubmitError::Serialize(_) => false,
    }
}

#[derive(Serialize)]
struct RawWeightJson<'a> {
    challenge_id: &'a str,
    miner_hotkey: String,
    epoch: u64,
    score_or_absence: ScoreOrAbsenceJson,
    challenge_sig: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum ScoreOrAbsenceJson {
    Score { value: u64 },
    NoScore { reason: u8 },
}

fn leaf_to_json(leaf: &LeafV1) -> Result<serde_json::Value, String> {
    let challenge_id =
        std::str::from_utf8(&leaf.challenge_id).map_err(|e| format!("challenge_id utf8: {e}"))?;
    let soa = match &leaf.score_or_absence {
        ScoreOrAbsence::Score { value } => ScoreOrAbsenceJson::Score { value: *value },
        ScoreOrAbsence::NoScore { reason } => ScoreOrAbsenceJson::NoScore {
            reason: *reason as u8,
        },
    };
    let req = RawWeightJson {
        challenge_id,
        miner_hotkey: hex::encode(leaf.miner_hotkey),
        epoch: leaf.epoch,
        score_or_absence: soa,
        challenge_sig: hex::encode(leaf.challenge_sig),
    };
    serde_json::to_value(req).map_err(|e| e.to_string())
}

/// Build JSON body for tests / external callers.
///
/// # Errors
///
/// Returns an error string if the leaf cannot be encoded as JSON.
pub fn leaf_request_json(leaf: &LeafV1) -> Result<serde_json::Value, String> {
    leaf_to_json(leaf)
}

/// Encode miner hotkey hex (lowercase).
#[must_use]
pub fn hotkey_hex(hk: &[u8; KEY_LEN]) -> String {
    hex::encode(hk)
}

/// POST every leaf from [`crate::emit_signed_leaf_set`] (`BTreeMap` order).
///
/// Retries use [`DEFAULT_MAX_RETRIES`]; HTTP 409 is success (no duplicate leaf).
/// When `base_url` is [`DRY_RUN_BASE_URL`], returns a single [`SubmitOutcome::DryRun`].
///
/// # Errors
///
/// First non-recoverable [`SubmitError`].
pub async fn submit_signed_leaf_set(
    client: &GatewayClient,
    leaves: &std::collections::BTreeMap<[u8; KEY_LEN], LeafV1>,
) -> Result<Vec<SubmitOutcome>, SubmitError> {
    if client.is_dry_run() {
        return Ok(vec![GatewayClient::dry_run(leaves)]);
    }
    let list: Vec<LeafV1> = leaves.values().cloned().collect();
    client.submit_all(&list).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn dry_run_counts() {
        let m = BTreeMap::new();
        assert_eq!(
            GatewayClient::dry_run(&m),
            SubmitOutcome::DryRun { leaf_count: 0 }
        );
    }
}
