//! Client for the master-only admin attestation grant (`AGENT_CHALLENGE.md`
//! §9.6): records `verified` credit for a non-TEE runtime, epoch-bound.

use crypto::KEY_LEN;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Route path on the master-plane gateway (`gateway::ATTEST_GRANT_ROUTE`,
/// repeated here so this CLI crate keeps its dependency surface axum-free).
pub const ATTEST_GRANT_ROUTE: &str = "/v1/admin/attest-grant";

/// Grant flow errors.
#[derive(Debug, Error)]
pub enum AttestGrantError {
    /// HTTP / transport failure.
    #[error("http: {0}")]
    Http(String),
    /// JSON encode/decode.
    #[error("json: {0}")]
    Json(String),
    /// Gateway returned a non-success HTTP status.
    #[error("gateway status {status}: {body}")]
    GatewayStatus {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },
}

/// Parameters for one admin grant.
#[derive(Debug, Clone)]
pub struct AttestGrantParams {
    /// Master gateway base URL (no trailing slash required).
    pub gateway_url: String,
    /// Chain epoch the credit binds to.
    pub epoch: u64,
    /// Miner hotkey (sr25519 public key) receiving credit.
    pub miner_hotkey: [u8; KEY_LEN],
    /// Receipt public key the runtime signs work receipts with.
    pub receipt_public_key: [u8; KEY_LEN],
    /// Mandatory audit note (stored as `reason: admin-exempt: …`).
    pub reason: String,
}

/// What the gateway stored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttestGrantOutcome {
    /// Granted epoch.
    pub epoch: u64,
    /// Miner hotkey hex as stored.
    pub miner_hotkey_hex: String,
    /// Receipt key hex as stored.
    pub receipt_pk_hex: String,
    /// Attempt slot the row landed in.
    pub attempt: i32,
    /// Always `verified`.
    pub outcome: String,
}

/// POST the grant to the master-plane gateway.
///
/// # Errors
///
/// Transport failure, non-success status, or response decode failure.
pub async fn attest_grant(
    params: &AttestGrantParams,
) -> Result<AttestGrantOutcome, AttestGrantError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| AttestGrantError::Http(e.to_string()))?;
    let url = format!(
        "{}{ATTEST_GRANT_ROUTE}",
        params.gateway_url.trim_end_matches('/')
    );
    let request = serde_json::json!({
        "epoch": params.epoch,
        "miner_hotkey_hex": hex::encode(params.miner_hotkey),
        "receipt_pk_hex": hex::encode(params.receipt_public_key),
        "reason": params.reason,
    });
    let resp = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| AttestGrantError::Http(e.to_string()))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| AttestGrantError::Http(e.to_string()))?;
    if !status.is_success() {
        return Err(AttestGrantError::GatewayStatus {
            status: status.as_u16(),
            body: text,
        });
    }
    serde_json::from_str(&text).map_err(|e| AttestGrantError::Json(e.to_string()))
}
