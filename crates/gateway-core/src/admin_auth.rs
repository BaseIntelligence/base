//! Bearer auth for master-only `/v1/admin/*` control-plane routes.
//!
//! Prod historically published these on `:80`/`:443` with no app auth. Policy:
//! - token env/file set → require matching `Authorization: Bearer`
//! - `BASE_GATEWAY_REQUIRE_OWNER=1` without token → fail closed at load
//! - otherwise (local/test) → open + warn

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use thiserror::Error;

/// Env: raw admin bearer token (prefer the file form in deploy).
pub const ADMIN_TOKEN_ENV: &str = "BASE_GATEWAY_ADMIN_TOKEN";
/// Env: path to a single-line admin bearer token file (mode 0400).
pub const ADMIN_TOKEN_FILE_ENV: &str = "BASE_GATEWAY_ADMIN_TOKEN_FILE";
/// Same flag the gateway uses for master-only owner check.
pub const REQUIRE_OWNER_ENV: &str = "BASE_GATEWAY_REQUIRE_OWNER";

/// Admin auth configuration errors.
#[derive(Debug, Error)]
pub enum AdminAuthError {
    /// Missing/empty/unreadable token configuration.
    #[error("{0}")]
    Config(String),
}

/// Shared admin-auth state for the axum middleware.
#[derive(Clone, Debug)]
pub struct AdminAuth {
    /// When `Some`, Bearer must match. When `None`, admin routes are open.
    token: Option<Arc<[u8]>>,
}

impl AdminAuth {
    /// Load from env. Fails closed when owner-check is required but no token.
    ///
    /// # Errors
    ///
    /// Missing token under `BASE_GATEWAY_REQUIRE_OWNER=1`, or unreadable file.
    pub fn from_env() -> Result<Self, AdminAuthError> {
        let token = load_token()?;
        let require_owner = std::env::var(REQUIRE_OWNER_ENV)
            .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES"));
        if token.is_none() {
            if require_owner {
                return Err(AdminAuthError::Config(format!(
                    "{REQUIRE_OWNER_ENV}=1 requires {ADMIN_TOKEN_ENV} or {ADMIN_TOKEN_FILE_ENV} \
                     (refusing to expose open /v1/admin/* on a public listener)"
                )));
            }
            tracing::warn!(
                event = "gateway_admin_auth_open",
                "BASE_GATEWAY_ADMIN_TOKEN unset; /v1/admin/* is unauthenticated \
                 (OK for local tests only)"
            );
        } else {
            tracing::info!(
                event = "gateway_admin_auth_enabled",
                "Bearer auth required for /v1/admin/*"
            );
        }
        Ok(Self {
            token: token.map(Into::into),
        })
    }

    /// Test helper: require this exact bearer token.
    #[must_use]
    pub fn require_token(token: impl Into<String>) -> Self {
        Self {
            token: Some(Arc::from(token.into().into_bytes().into_boxed_slice())),
        }
    }

    /// Test helper: leave admin routes open.
    #[must_use]
    pub fn open() -> Self {
        Self { token: None }
    }
}

fn load_token() -> Result<Option<Vec<u8>>, AdminAuthError> {
    if let Ok(raw) = std::env::var(ADMIN_TOKEN_ENV) {
        let t = raw.trim();
        if t.is_empty() {
            return Err(AdminAuthError::Config(format!(
                "{ADMIN_TOKEN_ENV} is set but empty"
            )));
        }
        return Ok(Some(t.as_bytes().to_vec()));
    }
    if let Ok(path) = std::env::var(ADMIN_TOKEN_FILE_ENV) {
        let raw = std::fs::read_to_string(&path).map_err(|e| {
            AdminAuthError::Config(format!("read {ADMIN_TOKEN_FILE_ENV} ({path}): {e}"))
        })?;
        let t = raw.trim();
        if t.is_empty() {
            return Err(AdminAuthError::Config(format!(
                "{ADMIN_TOKEN_FILE_ENV} ({path}) is empty"
            )));
        }
        return Ok(Some(t.as_bytes().to_vec()));
    }
    Ok(None)
}

/// Axum middleware: gate `/v1/admin` and `/v1/admin/*`.
pub async fn admin_auth_middleware(
    State(auth): State<AdminAuth>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    if !is_admin_path(path) {
        return next.run(request).await;
    }
    let Some(expected) = auth.token.as_ref() else {
        return next.run(request).await;
    };
    let provided = bearer_from_headers(request.headers());
    match provided {
        Some(got) if ct_eq(got.as_bytes(), expected.as_ref()) => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "admin bearer token required",
                "code": "admin_unauthorized",
            })),
        )
            .into_response(),
    }
}

fn is_admin_path(path: &str) -> bool {
    path == "/v1/admin" || path.starts_with("/v1/admin/")
}

fn bearer_from_headers(headers: &axum::http::HeaderMap) -> Option<String> {
    let raw = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let token = raw
        .strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))?;
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_owned())
    }
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::{ct_eq, is_admin_path};

    #[test]
    fn admin_path_prefix() {
        assert!(is_admin_path("/v1/admin/seal"));
        assert!(is_admin_path("/v1/admin/backends"));
        assert!(is_admin_path("/v1/admin"));
        assert!(!is_admin_path("/v1/weights/latest"));
        assert!(!is_admin_path("/v1/administration"));
    }

    #[test]
    fn ct_eq_basic() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
    }
}
