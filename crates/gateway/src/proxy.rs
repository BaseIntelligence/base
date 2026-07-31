//! Reverse proxy for `/challenge/{id}/*` with registry round-robin (D20 path).
//!
//! Forwards method, filtered headers, and body to the selected upstream.
//! Transport errors and 5xx responses increment fail counters and may passively
//! eject a backend after N failures.

use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::Router;
use bytes::Bytes;

use crate::api::GatewayState;
use gateway_registry::RegistryError;

/// Hop-by-hop headers that must not be forwarded (RFC 7230).
fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
            | "host"
    )
}

/// Mount challenge reverse-proxy routes.
pub fn proxy_router(state: GatewayState) -> Router {
    Router::new()
        .route("/challenge/{challenge_id}", any(proxy_root))
        .route("/challenge/{challenge_id}/{*rest}", any(proxy_rest))
        .with_state(state)
}

async fn proxy_root(
    State(st): State<GatewayState>,
    Path(challenge_id): Path<String>,
    req: Request,
) -> Response {
    let query = req.uri().query().map(str::to_owned);
    proxy_inner(st, challenge_id, String::new(), query, req).await
}

async fn proxy_rest(
    State(st): State<GatewayState>,
    Path((challenge_id, rest)): Path<(String, String)>,
    req: Request,
) -> Response {
    let query = req.uri().query().map(str::to_owned);
    proxy_inner(st, challenge_id, rest, query, req).await
}

async fn proxy_inner(
    st: GatewayState,
    challenge_id: String,
    rest: String,
    query: Option<String>,
    req: Request,
) -> Response {
    let method = req.method().clone();
    let headers = req.headers().clone();
    let body = match axum::body::to_bytes(req.into_body(), 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("failed to read body: {e}")).into_response();
        }
    };

    let mut last_status = StatusCode::BAD_GATEWAY;
    let mut last_msg = String::from("no upstream attempt");
    let mut attempted = Vec::new();

    for _ in 0..2 {
        let backend = match st.registry.pick(&challenge_id) {
            Ok(b) => b,
            Err(RegistryError::NoBackends(_)) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("no healthy backends for challenge_id={challenge_id}"),
                )
                    .into_response();
            }
            Err(e) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
            }
        };

        if attempted.contains(&backend.id) {
            break;
        }
        attempted.push(backend.id);

        let url = upstream_url(&backend.base_url, &rest, query.as_deref());
        match forward(&st.client, method.clone(), &url, &headers, body.clone()).await {
            ForwardResult::Ok(upstream_resp) => {
                let status = upstream_resp.status();
                if status.is_server_error() {
                    st.registry.record_failure(backend.id);
                    last_status = status;
                    last_msg = format!("upstream {status}");
                    continue;
                }
                st.registry.record_success(backend.id);
                return upstream_resp;
            }
            ForwardResult::Err(msg) => {
                st.registry.record_failure(backend.id);
                last_status = StatusCode::BAD_GATEWAY;
                last_msg = msg;
            }
        }
    }

    (last_status, last_msg).into_response()
}

/// Join base URL, remaining path, and optional query.
#[must_use]
pub fn upstream_url(base: &str, rest: &str, query: Option<&str>) -> String {
    let base = base.trim_end_matches('/');
    let mut url = if rest.is_empty() {
        format!("{base}/")
    } else {
        let rest = rest.trim_start_matches('/');
        format!("{base}/{rest}")
    };
    if let Some(q) = query {
        if !q.is_empty() {
            url.push('?');
            url.push_str(q);
        }
    }
    url
}

enum ForwardResult {
    Ok(Response),
    Err(String),
}

async fn forward(
    client: &reqwest::Client,
    method: Method,
    url: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> ForwardResult {
    let method = match reqwest::Method::from_bytes(method.as_str().as_bytes()) {
        Ok(m) => m,
        Err(e) => return ForwardResult::Err(format!("bad method: {e}")),
    };

    let mut rb = client.request(method, url);
    for (name, value) in headers {
        if is_hop_by_hop(name) || name == header::CONTENT_LENGTH {
            continue;
        }
        if let Ok(v) = value.to_str() {
            rb = rb.header(name.as_str(), v);
        }
    }

    if !body.is_empty() {
        rb = rb.body(body);
    }

    let resp = match rb.send().await {
        Ok(r) => r,
        Err(e) => return ForwardResult::Err(format!("upstream error: {e}")),
    };

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut out_headers = HeaderMap::new();
    for (name, value) in resp.headers() {
        if is_hop_by_hop(name) {
            continue;
        }
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            out_headers.insert(n, v);
        }
    }

    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return ForwardResult::Err(format!("upstream body: {e}")),
    };

    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    *response.headers_mut() = out_headers;
    ForwardResult::Ok(response)
}

/// Build upstream URL including the original request query string.
#[must_use]
#[allow(dead_code)]
pub fn upstream_uri(base: &str, rest: &str, original: &Uri) -> String {
    upstream_url(base, rest, original.query())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_by_hop_detects_connection() {
        assert!(is_hop_by_hop(&header::CONNECTION));
        assert!(is_hop_by_hop(&header::HOST));
        assert!(!is_hop_by_hop(&header::CONTENT_TYPE));
    }

    #[test]
    fn upstream_uri_joins_path_and_query() {
        let u: Uri = "http://gw/challenge/c1/v1/score?x=1".parse().unwrap();
        let out = upstream_uri("http://127.0.0.1:9", "v1/score", &u);
        assert_eq!(out, "http://127.0.0.1:9/v1/score?x=1");
    }
}
