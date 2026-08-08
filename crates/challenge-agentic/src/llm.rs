//! OpenRouter-compatible chat + tools client (master-side; key never logged).

use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use serde_json::{json, Value};

use crate::types::{AgenticError, OPENROUTER_API_BASE};

/// Default chat model for tool-calling reviews.
pub const DEFAULT_MODEL: &str = "openai/gpt-4o-mini";

/// Read the API key from a file (mode 0400), never from env text.
///
/// # Errors
/// Missing/empty file.
pub fn load_api_key_file(path: &std::path::Path) -> Result<String, AgenticError> {
    let text = std::fs::read_to_string(path).map_err(|_| AgenticError::NoKey)?;
    let key = text.trim().to_owned();
    if key.is_empty() || key.len() < 12 {
        return Err(AgenticError::NoKey);
    }
    Ok(key)
}

/// Take `OPENROUTER_API_KEY` from the live process environment into memory and
/// remove the env slot.
///
/// Prefer `OPENROUTER_API_KEY_FILE` in review containers: Linux
/// `/proc/<pid>/environ` is a **boot-time snapshot**, so `unsetenv` cannot
/// hide a key that was injected into the container's initial environ. This
/// helper remains for legacy/local injects and for clearing the live environ
/// map (e.g. accidental inheritance). Always removes the variable.
#[must_use]
pub fn take_openrouter_api_key() -> Option<String> {
    let key = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .map(|k| k.trim().to_owned())
        .filter(|k| !k.is_empty());
    // Single-threaded at challenge-review startup; libtest serializes the
    // unit test that mutates this var.
    std::env::remove_var("OPENROUTER_API_KEY");
    key
}

/// HTTP client for `/chat/completions` with tools.
pub struct ChatClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl std::fmt::Debug for ChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl ChatClient {
    /// Build with default base + model.
    ///
    /// # Errors
    /// Empty key.
    pub fn new(api_key: impl Into<String>) -> Result<Self, AgenticError> {
        Self::with_config(api_key, OPENROUTER_API_BASE, DEFAULT_MODEL)
    }

    /// Explicit base + model (tests / overrides).
    ///
    /// # Errors
    /// Empty key.
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, AgenticError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(AgenticError::NoKey);
        }
        let mut headers = HeaderMap::new();
        let mut hv = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| AgenticError::Transport(e.to_string()))?;
        hv.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, hv);
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("base-challenge-agentic/0.1"),
        );
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_mins(2))
            .build()
            .map_err(|e| AgenticError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
            model: model.into(),
        })
    }

    /// Provider base URL (never includes the key).
    #[must_use]
    pub fn base_url(&self) -> &str {
        self.base_url.as_str()
    }

    /// Configured model id.
    #[must_use]
    pub fn model(&self) -> &str {
        self.model.as_str()
    }

    /// One chat turn; returns assistant message JSON + usage tokens.
    ///
    /// # Errors
    /// Transport / provider / parse.
    pub async fn chat_tools(
        &self,
        messages: &[Value],
        tools: &Value,
        max_tokens: u32,
    ) -> Result<ChatTurn, AgenticError> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = json!({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": max_tokens,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgenticError::Transport(sanitize(&e.to_string(), &self.api_key)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| AgenticError::Transport(sanitize(&e.to_string(), &self.api_key)))?;
        if !status.is_success() {
            return Err(AgenticError::Provider(format!(
                "{status}: {}",
                sanitize(&truncate(&text, 240), &self.api_key)
            )));
        }
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| AgenticError::Parse(format!("envelope json: {e}")))?;
        let message = v
            .pointer("/choices/0/message")
            .cloned()
            .ok_or_else(|| AgenticError::Parse("missing choices[0].message".into()))?;
        let tokens = v
            .pointer("/usage/total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        Ok(ChatTurn { message, tokens })
    }
}

/// One provider turn.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    /// Assistant message object (may include `tool_calls`).
    pub message: Value,
    /// `usage.total_tokens` when present.
    pub tokens: u64,
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_owned()
    } else {
        format!("{}…", &s[..n])
    }
}

fn sanitize(msg: &str, key: &str) -> String {
    if key.is_empty() {
        msg.to_owned()
    } else {
        msg.replace(key, "<redacted>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize tests that mutate `OPENROUTER_API_KEY`.
    static OPENROUTER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn take_openrouter_api_key_scrubs_live_env() {
        let _guard = OPENROUTER_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("OPENROUTER_API_KEY", "  sk-test-secret-value-xx  ");
        let key = take_openrouter_api_key().expect("key");
        assert_eq!(key, "sk-test-secret-value-xx");
        assert!(std::env::var("OPENROUTER_API_KEY").is_err());
        // Absent after scrub → None.
        assert!(take_openrouter_api_key().is_none());
    }
}
