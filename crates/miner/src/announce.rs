//! Miner endpoint announce client (`AGENT_CHALLENGE.md` §9.3 step 5).
//!
//! Signs [`MinerEndpointBodyV1`] with the miner hotkey and POSTs it to the
//! gateway, which is what lets the challenge service find the CVM this miner
//! just deployed. Nothing here is authoritative: the gateway re-validates the
//! URL, the epoch, the registration, and the signature.

use crypto::KEY_LEN;
use miner_endpoint::{sign_endpoint, validate_base_url, MinerEndpointBodyV1, ENDPOINT_ROUTE};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from the announce flow.
#[derive(Debug, Error)]
pub enum AnnounceError {
    /// The base URL would be refused by the gateway; caught before signing.
    #[error("base_url: {0}")]
    BaseUrl(String),
    /// Signing failed (malformed hotkey mini-secret).
    #[error("sign: {0}")]
    Sign(String),
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

/// Parameters for one announcement.
#[derive(Debug, Clone)]
pub struct AnnounceParams {
    /// Gateway base URL (no trailing slash required).
    pub gateway_url: String,
    /// Subnet netuid.
    pub netuid: u16,
    /// Chain epoch; the gateway rejects anything but the current one.
    pub epoch: u64,
    /// Public CVM base URL to announce, origin only.
    pub base_url: String,
    /// Miner hotkey mini-secret (never logged).
    pub hotkey_secret: [u8; KEY_LEN],
}

/// What the gateway stored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnnounceOutcome {
    /// Subnet netuid.
    pub netuid: u16,
    /// Miner hotkey hex as stored.
    pub miner_hotkey_hex: String,
    /// Base URL as stored.
    pub base_url: String,
    /// Epoch the row was stored under.
    pub epoch: u64,
}

/// Sign and POST the announcement.
///
/// # Errors
///
/// URL rejection, signing failure, transport error, or a gateway error status.
pub async fn announce(params: &AnnounceParams) -> Result<AnnounceOutcome, AnnounceError> {
    // Fail locally on a URL the gateway would 400 anyway, so the operator sees
    // the rejection class instead of an HTTP body.
    validate_base_url(&params.base_url).map_err(|e| AnnounceError::BaseUrl(e.to_string()))?;

    let miner_hotkey = crypto::public_key_from_mini_secret(&params.hotkey_secret)
        .map_err(|e| AnnounceError::Sign(e.to_string()))?;
    let body = MinerEndpointBodyV1 {
        netuid: params.netuid,
        miner_hotkey,
        base_url: params.base_url.clone().into_bytes(),
        epoch: params.epoch,
    };
    let signature = sign_endpoint(&params.hotkey_secret, &body)
        .map_err(|e| AnnounceError::Sign(e.to_string()))?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| AnnounceError::Http(e.to_string()))?;
    let url = format!(
        "{}{ENDPOINT_ROUTE}",
        params.gateway_url.trim_end_matches('/')
    );
    let request = serde_json::json!({
        "netuid": params.netuid,
        "miner_hotkey_hex": hex::encode(miner_hotkey),
        "base_url": params.base_url,
        "epoch": params.epoch,
        "signature_hex": hex::encode(signature),
    });
    let resp = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| AnnounceError::Http(e.to_string()))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| AnnounceError::Http(e.to_string()))?;
    if !status.is_success() {
        return Err(AnnounceError::GatewayStatus {
            status: status.as_u16(),
            body: text,
        });
    }
    serde_json::from_str(&text).map_err(|e| AnnounceError::Json(e.to_string()))
}
