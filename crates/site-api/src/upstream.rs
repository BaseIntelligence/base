//! HTTP fetch helpers against registry-picked challenge backends.

use gateway_registry::RegistryError;
use serde_json::Value;

use crate::state::SiteState;

/// Challenge ids registered on the gateway.
pub const DESIGN: &str = "design";
pub const PRISM: &str = "prism";

/// Upstream fetch error.
#[derive(Debug)]
pub enum UpstreamError {
    /// No healthy backend.
    Unavailable(String),
    /// Transport / decode.
    Transport(String),
}

impl std::fmt::Display for UpstreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(s) | Self::Transport(s) => write!(f, "{s}"),
        }
    }
}

/// GET JSON from a challenge backend path (e.g. `/v1/dashboard`).
pub async fn get_json(
    st: &SiteState,
    challenge_id: &str,
    path: &str,
) -> Result<Value, UpstreamError> {
    let backend = st.registry.pick(challenge_id).map_err(|e| match e {
        RegistryError::NoBackends(id) => {
            UpstreamError::Unavailable(format!("no healthy backends for challenge_id={id}"))
        }
        other => UpstreamError::Transport(other.to_string()),
    })?;
    let base = backend.base_url.trim_end_matches('/');
    let path = if path.starts_with('/') {
        path.to_owned()
    } else {
        format!("/{path}")
    };
    let url = format!("{base}{path}");
    let resp = st
        .client
        .get(&url)
        .send()
        .await
        .map_err(|e| UpstreamError::Transport(e.to_string()))?;
    let status = resp.status();
    if !status.is_success() {
        st.registry.record_failure(backend.id);
        return Err(UpstreamError::Transport(format!(
            "upstream {challenge_id} {path} → {status}"
        )));
    }
    st.registry.record_success(backend.id);
    resp.json::<Value>()
        .await
        .map_err(|e| UpstreamError::Transport(e.to_string()))
}

/// Soft GET: `None` when the backend is missing or errors (never panics).
pub async fn get_json_opt(st: &SiteState, challenge_id: &str, path: &str) -> Option<Value> {
    match get_json(st, challenge_id, path).await {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::debug!(challenge_id, path, error = %e, "site-api upstream miss");
            None
        }
    }
}
