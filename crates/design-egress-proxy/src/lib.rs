//! Open HTTP egress for design sandboxes: forward proxy (`CONNECT` + plain
//! HTTP) with an internal-target blocklist, plus the budgeted `OpenRouter`
//! chat path. Both sandbox phases (install + run) reach the public internet
//! through this proxy; the proxy refuses loopback / link-local / RFC1918 /
//! CGNAT and control-plane service names, checked **after** DNS resolution
//! (DNS-rebinding safe).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;

/// Default per-run token budget.
pub const DEFAULT_TOKEN_BUDGET: u64 = 8_000;

/// Hop-by-hop headers never forwarded in either direction.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
];

/// Control-plane service names resolvable on the compose networks. The
/// post-resolution IP block is the real guard (all of these land in RFC1918);
/// the name list is defense-in-depth against odd DNS setups.
const BLOCKED_HOSTNAMES: &[&str] = &[
    "gateway",
    "postgres",
    "socket-proxy",
    "design-challenge",
    "prism-challenge",
    "design-egress-proxy",
    "updater",
    "site-api",
    "evil-gateway",
];

/// True when `host` names a loopback / internal control-plane target.
#[must_use]
pub fn host_blocked_by_name(host: &str) -> bool {
    let h = host.trim_end_matches('.').to_ascii_lowercase();
    h == "localhost" || h.ends_with(".internal") || BLOCKED_HOSTNAMES.contains(&h.as_str())
}

/// True when `ip` is loopback, link-local (cloud metadata 169.254.169.254),
/// RFC1918/VPC, CGNAT, multicast, reserved, or the IPv6 equivalents.
#[must_use]
pub fn ip_blocked(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            o[0] == 0 // 0.0.0.0/8 "this network"
                || o[0] == 10 // RFC1918
                || o[0] == 127 // loopback
                || (o[0] == 100 && (64..=127).contains(&o[1])) // 100.64.0.0/10 CGNAT
                || (o[0] == 169 && o[1] == 254) // link-local + cloud metadata
                || (o[0] == 172 && (16..=31).contains(&o[1])) // RFC1918
                || (o[0] == 192 && o[1] == 168) // RFC1918
                || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0.0/24 IETF
                || (o[0] == 192 && o[1] == 0 && o[2] == 2) // TEST-NET-1
                || (o[0] == 198 && (18..=19).contains(&o[1])) // benchmark
                || (o[0] == 198 && o[1] == 51 && o[2] == 100) // TEST-NET-2
                || (o[0] == 203 && o[1] == 0 && o[2] == 113) // TEST-NET-3
                || o[0] >= 224 // multicast + reserved
        }
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return ip_blocked(&IpAddr::V4(mapped));
            }
            let seg = v6.segments();
            v6.is_unspecified()
                || v6.is_loopback()
                || (seg[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
                || (seg[0] & 0xfe00) == 0xfc00 // fc00::/7 ULA
                || (seg[0] & 0xff00) == 0xff00 // ff00::/8 multicast
        }
    }
}

/// Resolve `host:port` and refuse internal targets. Fail-closed: when any
/// resolved address is blocked, the whole request is denied.
///
/// # Errors
/// [`ProxyError::HostDenied`] for blocked/unresolvable hosts,
/// [`ProxyError::Internal`] on DNS failure.
pub async fn resolve_egress(
    host: &str,
    port: u16,
    enforce: bool,
) -> Result<Vec<SocketAddr>, ProxyError> {
    if enforce && host_blocked_by_name(host) {
        return Err(ProxyError::HostDenied);
    }
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| ProxyError::Internal(format!("dns {host}: {e}")))?
        .collect();
    if addrs.is_empty() {
        return Err(ProxyError::HostDenied);
    }
    if enforce && addrs.iter().any(|a| ip_blocked(&a.ip())) {
        return Err(ProxyError::HostDenied);
    }
    Ok(addrs)
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
    /// Host blocked by egress policy.
    #[error("host blocked by egress policy")]
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
    /// OpenRouter API key (never returned).
    pub openrouter_key: Option<String>,
    /// Budgets.
    pub budgets: BudgetLedger,
    /// When true, chat returns deterministic stub (CI).
    pub sim: bool,
    /// Enforce the internal-target blocklist. Always on in prod; disabled
    /// only by loopback tunnel tests.
    pub enforce_blocklist: bool,
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

/// Build router: operator API routes plus the forward-proxy fallback.
pub fn proxy_router(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat", post(chat))
        .route("/v1/budget/{run_id}", get(budget))
        .fallback(forward)
        .with_state(state)
}

async fn health(State(st): State<Arc<ProxyState>>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "sim": st.sim,
        "blocklist": st.enforce_blocklist,
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

fn deny(e: &ProxyError) -> Response {
    let status = match e {
        ProxyError::HostDenied => StatusCode::FORBIDDEN,
        _ => StatusCode::BAD_GATEWAY,
    };
    (status, Json(json!({"error": e.to_string()}))).into_response()
}

/// Forward-proxy fallback: `CONNECT` tunnels (HTTPS) and absolute-URI plain
/// HTTP forwarding, both gated by the internal-target blocklist.
async fn forward(State(st): State<Arc<ProxyState>>, req: Request<Body>) -> Response {
    if req.method() == Method::CONNECT {
        return connect_tunnel(st, req).await;
    }
    if req.uri().authority().is_some() {
        return forward_http(st, req).await;
    }
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": "absolute URI or CONNECT required"})),
    )
        .into_response()
}

async fn connect_tunnel(st: Arc<ProxyState>, mut req: Request<Body>) -> Response {
    let Some(authority) = req.uri().authority().cloned() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "CONNECT authority required"})),
        )
            .into_response();
    };
    let host = authority.host().to_owned();
    let port = authority.port_u16().unwrap_or(443);
    let addrs = match resolve_egress(&host, port, st.enforce_blocklist).await {
        Ok(a) => a,
        Err(e) => return deny(&e),
    };
    let mut upstream = None;
    for addr in addrs {
        let attempt = tokio::time::timeout(
            Duration::from_secs(15),
            tokio::net::TcpStream::connect(addr),
        );
        if let Ok(Ok(s)) = attempt.await {
            upstream = Some(s);
            break;
        }
    }
    let Some(mut upstream) = upstream else {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": "upstream connect failed"})),
        )
            .into_response();
    };
    tokio::spawn(async move {
        match hyper::upgrade::on(&mut req).await {
            Ok(upgraded) => {
                let mut client = hyper_util::rt::TokioIo::new(upgraded);
                let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
            }
            Err(e) => tracing::debug!(error = %e, "CONNECT upgrade failed"),
        }
    });
    Response::new(Body::empty())
}

async fn forward_http(st: Arc<ProxyState>, req: Request<Body>) -> Response {
    let uri = req.uri().clone();
    let Some(authority) = uri.authority().cloned() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "absolute URI required"})),
        )
            .into_response();
    };
    let scheme = uri.scheme_str().unwrap_or("http").to_owned();
    if scheme != "http" && scheme != "https" {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "only http/https URIs"})),
        )
            .into_response();
    }
    let host = authority.host().to_owned();
    let port = authority
        .port_u16()
        .unwrap_or(if scheme == "https" { 443 } else { 80 });
    let addrs = match resolve_egress(&host, port, st.enforce_blocklist).await {
        Ok(a) => a,
        Err(e) => return deny(&e),
    };
    // Pin the validated addresses: reqwest connects only to these (no re-DNS,
    // no rebinding window). Redirects stay client-side so each hop re-checks.
    let client = match reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .resolve_to_addrs(&host, &addrs)
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_mins(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => return deny(&ProxyError::Internal(e.to_string())),
    };
    let path_q = uri.path_and_query().map_or("/", |pq| pq.as_str());
    let url = format!("{scheme}://{authority}{path_q}");
    let mut out = client.request(req.method().clone(), url);
    for (name, value) in req.headers() {
        if HOP_BY_HOP.contains(&name.as_str()) {
            continue;
        }
        out = out.header(name, value);
    }
    let resp = match out
        .body(reqwest::Body::wrap_stream(
            req.into_body().into_data_stream(),
        ))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return deny(&ProxyError::Internal(e.to_string())),
    };
    let mut b = Response::builder().status(resp.status());
    for (name, value) in resp.headers() {
        if HOP_BY_HOP.contains(&name.as_str()) {
            continue;
        }
        b = b.header(name, value);
    }
    b.body(Body::from_stream(resp.bytes_stream()))
        .unwrap_or_else(|_| deny(&ProxyError::Internal("response build".into())))
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
    fn blocklist_covers_internal_ranges() {
        for blocked in [
            "169.254.169.254", // cloud metadata
            "127.0.0.1",
            "10.0.0.5",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "100.64.0.1", // CGNAT (some VPCs)
            "0.0.0.0",
            "224.0.0.1",    // multicast
            "240.0.0.1",    // reserved
            "192.0.2.1",    // TEST-NET-1
            "198.51.100.1", // TEST-NET-2
            "203.0.113.1",  // TEST-NET-3
            "::1",
            "fe80::1",
            "fd00::1",
            "ff02::1",
            "::ffff:10.1.2.3", // v4-mapped private
            "::ffff:169.254.169.254",
        ] {
            let ip: IpAddr = blocked.parse().unwrap();
            assert!(ip_blocked(&ip), "{blocked} must be blocked");
        }
        for allowed in [
            "8.8.8.8",
            "1.1.1.1",
            "140.82.112.3",  // github
            "151.101.0.223", // pypi fastly
            "::ffff:8.8.8.8",
            "2606:4700:4700::1111",
        ] {
            let ip: IpAddr = allowed.parse().unwrap();
            assert!(!ip_blocked(&ip), "{allowed} must be allowed");
        }
    }

    #[test]
    fn control_plane_names_blocked() {
        for name in ["gateway", "postgres", "socket-proxy", "design-challenge"] {
            assert!(host_blocked_by_name(name));
        }
        assert!(host_blocked_by_name("localhost"));
        assert!(host_blocked_by_name("foo.internal"));
        assert!(!host_blocked_by_name("pypi.org"));
        assert!(!host_blocked_by_name("gateway.example.com"));
    }

    #[tokio::test]
    async fn resolve_egress_refuses_internal() {
        assert!(matches!(
            resolve_egress("localhost", 80, true).await,
            Err(ProxyError::HostDenied)
        ));
        assert!(matches!(
            resolve_egress("169.254.169.254", 443, true).await,
            Err(ProxyError::HostDenied)
        ));
        assert!(matches!(
            resolve_egress("socket-proxy", 2375, true).await,
            Err(ProxyError::HostDenied)
        ));
        // Literal public IP needs no DNS.
        let addrs = resolve_egress("8.8.8.8", 443, true).await.unwrap();
        assert_eq!(addrs[0].port(), 443);
        // Enforcement off (loopback tunnel tests): localhost resolves.
        let addrs = resolve_egress("127.0.0.1", 9, false).await.unwrap();
        assert_eq!(addrs[0].port(), 9);
    }

    #[test]
    fn budget_enforced() {
        let b = BudgetLedger::new(100);
        b.charge("r1", 60).unwrap();
        assert!(b.charge("r1", 50).is_err());
        assert_eq!(b.remaining("r1"), 40);
    }
}
