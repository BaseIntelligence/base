//! Fail-closed bearer gates for Prism operator routes.

#![allow(clippy::result_large_err)]

use std::path::Path;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use crypto::ct_eq;
use serde_json::json;
use sha2::{Digest, Sha256};

/// Accepted bearer hashes for one route group, plus the role name used in logs.
pub type BearerGate = (Arc<[String]>, &'static str);

/// sha256 hex of the caller's bearer, published to handlers by [`require_bearer`].
#[derive(Clone, Debug)]
pub struct CallerToken(pub String);

/// sha256 hex of a raw bearer token.
#[must_use]
pub fn token_hash(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    hex::encode(h.finalize())
}

/// Load one-token-per-line file into sha256 hex hashes (empty / missing → empty).
#[must_use]
pub fn load_token_hashes(path: Option<&Path>) -> Vec<String> {
    let Some(p) = path else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(p) else {
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(token_hash)
        .collect()
}

/// Constant-time membership over sha256 hex hashes.
#[must_use]
pub fn hash_matches(hashes: &[String], candidate: &str) -> bool {
    hashes.iter().fold(false, |hit, h| {
        hit | ct_eq(h.as_bytes(), candidate.as_bytes())
    })
}

fn json_err(code: StatusCode, kind: &str, msg: &str) -> Response {
    (code, Json(json!({"error": msg, "code": kind}))).into_response()
}

fn bearer_token(headers: &HeaderMap) -> Result<&str, Response> {
    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let token = auth
        .strip_prefix("Bearer ")
        .or_else(|| auth.strip_prefix("bearer "))
        .unwrap_or("")
        .trim();
    if token.is_empty() {
        return Err(json_err(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing bearer",
        ));
    }
    Ok(token)
}

/// Route-layer bearer check. Empty config → **503**, never open.
pub async fn require_bearer(
    State((hashes, role)): State<BearerGate>,
    mut request: Request,
    next: Next,
) -> Response {
    if hashes.is_empty() {
        let msg = format!("{role} bearer tokens are not configured");
        return json_err(StatusCode::SERVICE_UNAVAILABLE, "auth_unconfigured", &msg);
    }
    let hash = match bearer_token(request.headers()) {
        Ok(token) => token_hash(token),
        Err(unauthorized) => return unauthorized,
    };
    if !hash_matches(&hashes, &hash) {
        return json_err(StatusCode::UNAUTHORIZED, "unauthorized", "bad token");
    }
    request.extensions_mut().insert(CallerToken(hash));
    next.run(request).await
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn empty_file_is_fail_closed_config() {
        assert!(load_token_hashes(None).is_empty());
    }

    #[test]
    fn hash_roundtrip() {
        let h = token_hash("secret");
        assert!(hash_matches(std::slice::from_ref(&h), &h));
        assert!(!hash_matches(&[h], "other"));
    }
}
