//! Allowlisted HTTP egress for design sandbox (`PyPI` install / `OpenRouter` run).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

/// Default per-run token budget.
pub const DEFAULT_TOKEN_BUDGET: u64 = 8_000;

/// Proxy mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProxyMode {
    /// PyPI allowlist only.
    Pypi,
    /// OpenRouter chat only.
    Llm,
}

/// Host allowlists.
#[must_use]
pub fn pypi_hosts() -> &'static [&'static str] {
    &["pypi.org", "files.pythonhosted.org", "pypi.python.org"]
}

/// OpenRouter host.
#[must_use]
pub fn openrouter_hosts() -> &'static [&'static str] {
    &["openrouter.ai"]
}

/// True if host is allowed for mode.
#[must_use]
pub fn host_allowed(mode: ProxyMode, host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    let list = match mode {
        ProxyMode::Pypi => pypi_hosts(),
        ProxyMode::Llm => openrouter_hosts(),
    };
    list.iter()
        .any(|h| host == *h || host.ends_with(&format!(".{h}")))
}

/// Budget tracker per run.
#[derive(Debug, Default)]
pub struct BudgetLedger {
    used: Mutex<HashMap<String, u64>>,
    /// Per-run token limit.
    pub limit: u64,
}

impl BudgetLedger {
    /// New with per-run limit.
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self {
            used: Mutex::new(HashMap::new()),
            limit,
        }
    }

    /// Remaining budget.
    pub fn remaining(&self, run_id: &str) -> u64 {
        let used = self.used.lock().map_or(0, |g| *g.get(run_id).unwrap_or(&0));
        self.limit.saturating_sub(used)
    }

    /// Charge tokens; Err if over.
    ///
    /// # Errors
    /// Over budget.
    pub fn charge(&self, run_id: &str, tokens: u64) -> Result<(), ProxyError> {
        let mut g = self
            .used
            .lock()
            .map_err(|_| ProxyError::Internal("poison".into()))?;
        let e = g.entry(run_id.to_owned()).or_insert(0);
        if e.saturating_add(tokens) > self.limit {
            return Err(ProxyError::BudgetExceeded);
        }
        *e = e.saturating_add(tokens);
        Ok(())
    }
}

/// Proxy errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProxyError {
    /// Host not allowlisted.
    #[error("host not allowlisted")]
    HostDenied,
    /// Budget.
    #[error("token budget exceeded")]
    BudgetExceeded,
    /// Internal.
    #[error("internal: {0}")]
    Internal(String),
}

/// Shared state.
#[derive(Debug)]
pub struct ProxyState {
    /// Mode (install vs run). Can be overridden per-request via header.
    pub mode: ProxyMode,
    /// OpenRouter API key (never returned).
    pub openrouter_key: Option<String>,
    /// Budgets.
    pub budgets: BudgetLedger,
    /// When true, chat returns deterministic stub (CI).
    pub sim: bool,
}

/// Chat request from harness SDK.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Run id for budget.
    pub run_id: String,
    /// Model.
    pub model: String,
    /// Messages.
    pub messages: Vec<Value>,
}

/// Build router.
pub fn proxy_router(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat", post(chat))
        .route("/v1/budget/{run_id}", get(budget))
        .with_state(state)
}

async fn health(State(st): State<Arc<ProxyState>>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "mode": st.mode,
        "sim": st.sim,
    }))
}

async fn budget(
    State(st): State<Arc<ProxyState>>,
    axum::extract::Path(run_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    Json(json!({
        "run_id": run_id,
        "remaining": st.budgets.remaining(&run_id),
        "limit": st.budgets.limit,
    }))
}

async fn chat(
    State(st): State<Arc<ProxyState>>,
    body: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.body_text()})),
            )
                .into_response();
        }
    };
    if st.mode != ProxyMode::Llm && !st.sim {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "llm disabled in pypi mode"})),
        )
            .into_response();
    }
    // Rough token estimate: chars/4.
    let est = req
        .messages
        .iter()
        .map(|m| m.to_string().len() as u64 / 4)
        .sum::<u64>()
        .saturating_add(64);
    if let Err(e) = st.budgets.charge(&req.run_id, est) {
        return (
            StatusCode::PAYMENT_REQUIRED,
            Json(json!({"error": e.to_string()})),
        )
            .into_response();
    }
    if st.sim || st.openrouter_key.is_none() {
        let content = format!("sim-reply model={} msgs={}", req.model, req.messages.len());
        return Json(json!({"content": content, "usage": {"tokens": est}})).into_response();
    }
    // Live OpenRouter path (key injected server-side).
    match forward_openrouter(st.openrouter_key.as_deref().unwrap_or(""), &req).await {
        Ok(content) => Json(json!({"content": content, "usage": {"tokens": est}})).into_response(),
        Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({"error": e}))).into_response(),
    }
}

async fn forward_openrouter(key: &str, req: &ChatRequest) -> Result<String, String> {
    let client = reqwest::Client::new();
    let body = json!({
        "model": req.model,
        "messages": req.messages,
    });
    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("openrouter HTTP {}", resp.status()));
    }
    let v: Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(v.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn allowlists() {
        assert!(host_allowed(ProxyMode::Pypi, "pypi.org"));
        assert!(host_allowed(ProxyMode::Pypi, "files.pythonhosted.org"));
        assert!(!host_allowed(ProxyMode::Pypi, "evil.test"));
        assert!(host_allowed(ProxyMode::Llm, "openrouter.ai"));
        assert!(!host_allowed(ProxyMode::Llm, "pypi.org"));
    }

    #[test]
    fn budget_enforced() {
        let b = BudgetLedger::new(100);
        b.charge("r1", 60).unwrap();
        assert!(b.charge("r1", 50).is_err());
        assert_eq!(b.remaining("r1"), 40);
    }
}
