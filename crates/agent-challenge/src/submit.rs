//! Gateway `POST /v1/weights/raw` client with 5xx retry and (challenge, epoch, miner) idempotency.

use std::time::Duration;

use bundle::{LeafV1, NoScoreReasonCode, ScoreOrAbsence};
use crypto::KEY_LEN;
use serde::Serialize;
use thiserror::Error;

/// Default max attempts on 5xx / transport errors (including the first try).
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Client configuration.
#[derive(Debug, Clone)]
pub struct GatewayClientConfig {
    /// Base URL, e.g. `http://127.0.0.1:8080` (no trailing slash required).
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

    /// POST one signed leaf. Retries 5xx and transport errors.
    ///
    /// Idempotency: HTTP 409 (already present) is success — never submits a
    /// conflicting `ScoreOrAbsence` for the same key from this client path;
    /// callers must not change the leaf between retries.
    ///
    /// # Errors
    ///
    /// See [`SubmitError`].
    pub async fn submit_leaf(&self, leaf: &LeafV1) -> Result<SubmitOutcome, SubmitError> {
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
            reason: reason_code_u8(*reason),
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

fn reason_code_u8(r: NoScoreReasonCode) -> u8 {
    match r {
        NoScoreReasonCode::NotAttempted => 0,
        NoScoreReasonCode::Timeout => 1,
        NoScoreReasonCode::InvalidResponse => 2,
        NoScoreReasonCode::AttestationNotVerified => 3,
        NoScoreReasonCode::MinerError => 4,
        NoScoreReasonCode::RateLimited => 5,
        NoScoreReasonCode::ChallengeInternal => 6,
        NoScoreReasonCode::PolicySkip => 7,
    }
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
///
/// # Errors
///
/// First non-recoverable [`SubmitError`].
pub async fn submit_signed_leaf_set(
    client: &GatewayClient,
    leaves: &std::collections::BTreeMap<[u8; KEY_LEN], LeafV1>,
) -> Result<Vec<SubmitOutcome>, SubmitError> {
    let list: Vec<LeafV1> = leaves.values().cloned().collect();
    client.submit_all(&list).await
}
