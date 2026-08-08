//! Operator Lium account client (balance / optional invoice helpers).
//!
//! Auth: `X-API-Key` against `https://lium.io/api` (same as `prism-lium`).
//! See <https://docs.lium.io/developers/agents.md> and OpenAPI
//! `GET /users/me`, `POST /nowpayments/create-invoice`.

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;

use crate::error::FundingError;

/// Default Lium API base.
pub const LIUM_API_BASE_URL: &str = "https://lium.io/api";

/// Operator-facing Lium account operations used by funding (not pod rent).
#[async_trait]
pub trait LiumAccountClient: Send + Sync {
    /// Account USD balance (`GET /users/me` → `balance`).
    async fn balance_usd(&self) -> Result<f64, FundingError>;
}

/// No-op / test account with a fixed balance.
#[derive(Debug, Clone)]
pub struct FakeLiumAccount {
    /// Reported USD balance.
    pub balance_usd: f64,
}

#[async_trait]
impl LiumAccountClient for FakeLiumAccount {
    async fn balance_usd(&self) -> Result<f64, FundingError> {
        Ok(self.balance_usd)
    }
}

/// HTTPS Lium account client. API key never appears in `Debug`.
pub struct HttpLiumAccount {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl std::fmt::Debug for HttpLiumAccount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpLiumAccount")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl HttpLiumAccount {
    /// Build against the public Lium API.
    ///
    /// # Errors
    /// Empty key or HTTP client build failure.
    pub fn new(api_key: impl Into<String>) -> Result<Self, FundingError> {
        Self::with_base_url(api_key, LIUM_API_BASE_URL)
    }

    /// Custom base URL (tests).
    ///
    /// # Errors
    /// Empty key or HTTP client build failure.
    pub fn with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self, FundingError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(FundingError::Config("empty LIUM_API_KEY".into()));
        }
        let mut headers = HeaderMap::new();
        let mut hv = HeaderValue::from_str(&api_key)
            .map_err(|e| FundingError::Config(format!("api key header: {e}")))?;
        hv.set_sensitive(true);
        headers.insert("X-API-Key", hv);
        headers.insert(
            reqwest::header::USER_AGENT,
            HeaderValue::from_static("lium-funding/0.1 (base)"),
        );
        headers.insert(
            reqwest::header::ACCEPT,
            HeaderValue::from_static("application/json"),
        );
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| FundingError::Lium(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
        })
    }
}

#[async_trait]
impl LiumAccountClient for HttpLiumAccount {
    async fn balance_usd(&self) -> Result<f64, FundingError> {
        let url = format!("{}/users/me", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| FundingError::Lium(sanitize(&e.to_string(), &self.api_key)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| FundingError::Lium(sanitize(&e.to_string(), &self.api_key)))?;
        if !status.is_success() {
            return Err(FundingError::Lium(format!(
                "GET /users/me -> {status}: {}",
                sanitize(&text, &self.api_key)
            )));
        }
        let v: Value =
            serde_json::from_str(&text).map_err(|e| FundingError::Lium(format!("json: {e}")))?;
        v.get("balance")
            .and_then(|x| x.as_f64())
            .or_else(|| {
                v.get("balance")
                    .and_then(|x| x.as_str())
                    .and_then(|s| s.parse().ok())
            })
            .ok_or_else(|| FundingError::Lium("users/me missing balance".into()))
    }
}

fn sanitize(msg: &str, key: &str) -> String {
    if key.is_empty() {
        msg.to_owned()
    } else {
        msg.replace(key, "<redacted>")
    }
}
