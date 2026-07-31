//! Gateway raw-weights submit client (hypertraining-compatible leaf JSON).

use std::collections::BTreeMap;
use std::time::Duration;

use bundle::{LeafV1, NoScoreReasonCode, ScoreOrAbsence};
use crypto::KEY_LEN;
use serde::Serialize;
use thiserror::Error;

/// Default retry budget.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Gateway client config.
#[derive(Debug, Clone)]
pub struct GatewayClientConfig {
    pub base_url: String,
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

/// Submit errors.
#[derive(Debug, Error)]
pub enum SubmitError {
    #[error("http: {0}")]
    Http(String),
    #[error("gateway rejected: {0}")]
    Rejected(String),
    #[error("serialize: {0}")]
    Serialize(String),
}

/// Outcome of a submit attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitOutcome {
    Accepted,
    DryRun { leaf_count: usize },
}

/// Thin HTTP client for `POST /v1/weights/raw`.
#[derive(Debug, Clone)]
pub struct GatewayClient {
    pub(crate) cfg: GatewayClientConfig,
    http: reqwest::Client,
}

impl GatewayClient {
    /// Build client.
    ///
    /// # Errors
    /// HTTP client build failure.
    pub fn new(cfg: GatewayClientConfig) -> Result<Self, SubmitError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| SubmitError::Http(e.to_string()))?;
        Ok(Self { cfg, http })
    }

    /// Dry-run: no network, reports leaf count.
    #[must_use]
    pub fn dry_run(leaves: &BTreeMap<[u8; KEY_LEN], LeafV1>) -> SubmitOutcome {
        SubmitOutcome::DryRun {
            leaf_count: leaves.len(),
        }
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

fn reason_code_u8(r: NoScoreReasonCode) -> u8 {
    r as u8
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

/// Submit signed leaves (or dry-run when `base_url` is `dry-run`).
///
/// # Errors
/// HTTP / rejection.
pub async fn submit_signed_leaf_set(
    client: &GatewayClient,
    _challenge_id: &str,
    _epoch: u64,
    leaves: &BTreeMap<[u8; KEY_LEN], LeafV1>,
) -> Result<SubmitOutcome, SubmitError> {
    if client.cfg.base_url == "dry-run" {
        return Ok(GatewayClient::dry_run(leaves));
    }
    let url = format!(
        "{}/v1/weights/raw",
        client.cfg.base_url.trim_end_matches('/')
    );
    let mut last_err = String::new();
    for leaf in leaves.values() {
        let body = leaf_to_json(leaf).map_err(SubmitError::Serialize)?;
        let mut ok = false;
        for _ in 0..=client.cfg.max_retries {
            match client.http.post(&url).json(&body).send().await {
                Ok(resp) if resp.status().as_u16() == 202 || resp.status().as_u16() == 409 => {
                    ok = true;
                    break;
                }
                Ok(resp) => {
                    last_err = format!("status {}", resp.status());
                }
                Err(e) => last_err = e.to_string(),
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        if !ok {
            return Err(SubmitError::Rejected(last_err));
        }
    }
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
