//! Miner certify client (task 38): nonce → quote → submit.
//!
//! Live path talks to a CVM `GET /v1/quote` and a validator
//! `POST /v1/attest/{nonce,submit}`. Fixture mode loads real Phala quote +
//! event log, patches D10 `report_data` into the quote, and submits without a
//! live CVM.

use std::path::{Path, PathBuf};

use attest_parse::{patch_report_data, REPORT_DATA_LEN};
use attest_policy::{compute_report_data, ReportDataBinding};
use crypto::KEY_LEN;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from the certify flow.
#[derive(Debug, Error)]
pub enum CertifyError {
    /// HTTP / transport failure.
    #[error("http: {0}")]
    Http(String),
    /// JSON encode/decode.
    #[error("json: {0}")]
    Json(String),
    /// Hex decode/encode.
    #[error("hex: {0}")]
    Hex(String),
    /// Fixture IO.
    #[error("fixture: {0}")]
    Fixture(String),
    /// Quote patch / length.
    #[error("quote: {0}")]
    Quote(String),
    /// Validator returned a non-success HTTP status.
    #[error("validator status {status}: {body}")]
    ValidatorStatus {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },
}

/// How to obtain the quote + event log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuoteSource {
    /// Load real fixtures and patch D10 `report_data` (no live CVM).
    Fixture {
        /// Directory containing `quote.bin` + `event_log.json` (optional).
        dir: Option<PathBuf>,
    },
    /// `GET {agent_base}/v1/quote?...` on a live CVM / attest-helper.
    Live {
        /// Agent or attest-helper base URL.
        agent_base: String,
    },
}

/// Parameters for one certify attempt.
#[derive(Debug, Clone)]
pub struct CertifyParams {
    /// Validator base URL (no trailing slash required).
    pub validator_url: String,
    /// Subnet netuid.
    pub netuid: u16,
    /// Epoch to bind.
    pub epoch: u64,
    /// Miner hotkey (32 bytes).
    pub miner_hotkey: [u8; KEY_LEN],
    /// Quote source.
    pub quote_source: QuoteSource,
    /// Optional override for claimed validator hotkey (defaults to nonce response).
    pub validator_hotkey_override: Option<[u8; KEY_LEN]>,
}

/// Result printed by the CLI / returned to callers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CertifyResult {
    /// Hex nonce issued by the validator.
    pub nonce_hex: String,
    /// Outcome label: `verified` | `rejected` | `parked`.
    pub outcome: String,
    /// Machine reason when not verified.
    pub reason: Option<String>,
    /// Whether credit was granted.
    pub grants_credit: bool,
    /// Always false under D13.
    pub carries_prior_verified: bool,
    /// Validator hotkey used in the D10 binding.
    pub validator_hotkey_hex: String,
    /// Whether fixture mode was used.
    pub fixture_mode: bool,
}

#[derive(Debug, Deserialize)]
struct NonceResp {
    nonce_hex: String,
    epoch: u64,
    netuid: u16,
    validator_hotkey_hex: String,
    #[allow(dead_code)]
    ttl_secs: u64,
}

#[derive(Debug, Deserialize)]
struct SubmitResp {
    outcome: String,
    reason: Option<String>,
    grants_credit: bool,
    carries_prior_verified: bool,
}

#[derive(Debug, Deserialize)]
struct QuoteJson {
    quote_hex: String,
    event_log_json: String,
}

/// Run the full certify flow (async HTTP).
///
/// # Errors
///
/// Transport, fixture, or validator error responses.
pub async fn certify(params: &CertifyParams) -> Result<CertifyResult, CertifyError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| CertifyError::Http(e.to_string()))?;

    let base = params.validator_url.trim_end_matches('/');
    let nonce_url = format!("{base}/v1/attest/nonce");
    let submit_url = format!("{base}/v1/attest/submit");

    let nonce_body = serde_json::json!({
        "miner_hotkey_hex": hex::encode(params.miner_hotkey),
        "epoch": params.epoch,
        "netuid": params.netuid,
    });
    let nonce_http = client
        .post(&nonce_url)
        .json(&nonce_body)
        .send()
        .await
        .map_err(|e| CertifyError::Http(e.to_string()))?;
    let status = nonce_http.status();
    let nonce_text = nonce_http
        .text()
        .await
        .map_err(|e| CertifyError::Http(e.to_string()))?;
    if !status.is_success() {
        return Err(CertifyError::ValidatorStatus {
            status: status.as_u16(),
            body: nonce_text,
        });
    }
    let nonce_resp: NonceResp =
        serde_json::from_str(&nonce_text).map_err(|e| CertifyError::Json(e.to_string()))?;

    let nonce = decode_key(&nonce_resp.nonce_hex)?;
    let validator_hotkey = match params.validator_hotkey_override {
        Some(v) => v,
        None => decode_key(&nonce_resp.validator_hotkey_hex)?,
    };

    let binding = ReportDataBinding {
        netuid: params.netuid,
        epoch: params.epoch,
        miner_pubkey: params.miner_hotkey,
        nonce,
        validator_hotkey,
    };
    let report_data = compute_report_data(&binding);

    let fixture_mode = matches!(params.quote_source, QuoteSource::Fixture { .. });
    let (quote, event_log_json) = match &params.quote_source {
        QuoteSource::Fixture { dir } => load_fixture_quote(dir.as_deref(), &report_data)?,
        QuoteSource::Live { agent_base } => {
            fetch_live_quote(&client, agent_base, &binding, &report_data).await?
        }
    };

    let submit_body = serde_json::json!({
        "miner_hotkey_hex": hex::encode(params.miner_hotkey),
        "epoch": params.epoch,
        "netuid": params.netuid,
        "nonce_hex": hex::encode(nonce),
        "quote_hex": hex::encode(&quote),
        "event_log_json": event_log_json,
        "validator_hotkey_hex": hex::encode(validator_hotkey),
    });
    let submit_http = client
        .post(&submit_url)
        .json(&submit_body)
        .send()
        .await
        .map_err(|e| CertifyError::Http(e.to_string()))?;
    let status = submit_http.status();
    let submit_text = submit_http
        .text()
        .await
        .map_err(|e| CertifyError::Http(e.to_string()))?;
    if !status.is_success() {
        return Err(CertifyError::ValidatorStatus {
            status: status.as_u16(),
            body: submit_text,
        });
    }
    let submit: SubmitResp =
        serde_json::from_str(&submit_text).map_err(|e| CertifyError::Json(e.to_string()))?;

    let _ = nonce_resp.epoch;
    let _ = nonce_resp.netuid;

    Ok(CertifyResult {
        nonce_hex: nonce_resp.nonce_hex,
        outcome: submit.outcome,
        reason: submit.reason,
        grants_credit: submit.grants_credit,
        carries_prior_verified: submit.carries_prior_verified,
        validator_hotkey_hex: hex::encode(validator_hotkey),
        fixture_mode,
    })
}

/// Embedded real fixtures (task 34/35) for `--fixture-mode` without a path.
const EMBEDDED_QUOTE: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/quote.bin");
const EMBEDDED_EVENT_LOG: &[u8] =
    include_bytes!("../../attest-parse/tests/fixtures/real/event_log.json");

fn load_fixture_quote(
    dir: Option<&Path>,
    report_data: &[u8; REPORT_DATA_LEN],
) -> Result<(Vec<u8>, String), CertifyError> {
    let (mut quote, event_raw) = if let Some(d) = dir {
        let q = std::fs::read(d.join("quote.bin"))
            .map_err(|e| CertifyError::Fixture(format!("quote.bin: {e}")))?;
        let e = std::fs::read(d.join("event_log.json"))
            .map_err(|e| CertifyError::Fixture(format!("event_log.json: {e}")))?;
        (q, e)
    } else {
        (EMBEDDED_QUOTE.to_vec(), EMBEDDED_EVENT_LOG.to_vec())
    };
    patch_report_data(&mut quote, report_data).map_err(|e| CertifyError::Quote(e.to_string()))?;
    let event_log_json = String::from_utf8(event_raw)
        .map_err(|e| CertifyError::Fixture(format!("event_log utf8: {e}")))?;
    Ok((quote, event_log_json))
}

async fn fetch_live_quote(
    client: &reqwest::Client,
    agent_base: &str,
    binding: &ReportDataBinding,
    report_data: &[u8; REPORT_DATA_LEN],
) -> Result<(Vec<u8>, String), CertifyError> {
    let base = agent_base.trim_end_matches('/');
    // Attest-helper contract: GET /v1/quote with binding fields; response JSON.
    let url = format!(
        "{base}/v1/quote?netuid={}&epoch={}&nonce_hex={}&validator_hotkey_hex={}&miner_hotkey_hex={}",
        binding.netuid,
        binding.epoch,
        hex::encode(binding.nonce),
        hex::encode(binding.validator_hotkey),
        hex::encode(binding.miner_pubkey),
    );
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| CertifyError::Http(e.to_string()))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| CertifyError::Http(e.to_string()))?;
    if !status.is_success() {
        return Err(CertifyError::ValidatorStatus {
            status: status.as_u16(),
            body: text,
        });
    }
    let parsed: QuoteJson =
        serde_json::from_str(&text).map_err(|e| CertifyError::Json(e.to_string()))?;
    let mut quote =
        hex::decode(parsed.quote_hex.trim()).map_err(|e| CertifyError::Hex(e.to_string()))?;
    // Ensure D10 binding is present even if helper omitted it.
    patch_report_data(&mut quote, report_data).map_err(|e| CertifyError::Quote(e.to_string()))?;
    Ok((quote, parsed.event_log_json))
}

fn decode_key(s: &str) -> Result<[u8; KEY_LEN], CertifyError> {
    let b = hex::decode(s.trim()).map_err(|e| CertifyError::Hex(e.to_string()))?;
    if b.len() != KEY_LEN {
        return Err(CertifyError::Hex(format!(
            "expected {KEY_LEN} bytes, got {}",
            b.len()
        )));
    }
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&b);
    Ok(out)
}

/// Parse a 64-char hex hotkey.
///
/// # Errors
///
/// Bad hex or wrong length.
pub fn parse_hotkey_hex(s: &str) -> Result<[u8; KEY_LEN], CertifyError> {
    decode_key(s)
}
