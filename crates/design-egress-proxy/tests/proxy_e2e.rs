//! End-to-end proxy behavior over real TCP: internal targets refused,
//! `CONNECT` tunnel + plain-HTTP forward proven against loopback servers
//! (blocklist enforcement disabled for those two only).

#![allow(clippy::unwrap_used)]

use std::io::{Read, Write};
use std::net::TcpListener as StdListener;
use std::sync::Arc;

use design_egress_proxy::{proxy_router, BudgetLedger, ProxyState};
use tokio::net::TcpListener;

fn state(enforce: bool) -> Arc<ProxyState> {
    Arc::new(ProxyState {
        openrouter_key: None,
        budgets: BudgetLedger::new(8_000),
        sim: true,
        enforce_blocklist: enforce,
    })
}

async fn bind_proxy(enforce: bool) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, proxy_router(state(enforce)))
            .await
            .unwrap();
    });
    addr.to_string()
}

/// Raw HTTP/1.1 exchange: send `request`, return the status line + body text.
fn raw_exchange(proxy: &str, request: &str) -> String {
    let mut s = std::net::TcpStream::connect(proxy).unwrap();
    s.write_all(request.as_bytes()).unwrap();
    let mut buf = String::new();
    // Read whatever arrives within a short window; enough for status + small body.
    s.set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    let mut chunk = [0_u8; 8192];
    loop {
        match s.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.push_str(&String::from_utf8_lossy(&chunk[..n]));
                if buf.contains("\r\n\r\n") && !buf.starts_with("HTTP/1.1 200") {
                    break;
                }
                if buf.contains("hello-body") || buf.contains("sim-reply") {
                    break;
                }
            }
        }
    }
    buf
}

#[test]
fn connect_to_metadata_refused() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let proxy = rt.block_on(bind_proxy(true));
    let resp = raw_exchange(
        &proxy,
        "CONNECT 169.254.169.254:443 HTTP/1.1\r\nhost: 169.254.169.254:443\r\n\r\n",
    );
    assert!(resp.starts_with("HTTP/1.1 403"), "got: {resp:?}");
    assert!(resp.contains("egress policy"));
}

#[test]
fn connect_to_loopback_refused() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let proxy = rt.block_on(bind_proxy(true));
    let resp = raw_exchange(
        &proxy,
        "CONNECT 127.0.0.1:8093 HTTP/1.1\r\nhost: 127.0.0.1:8093\r\n\r\n",
    );
    assert!(resp.starts_with("HTTP/1.1 403"), "got: {resp:?}");
}

#[test]
fn plain_http_to_metadata_refused() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let proxy = rt.block_on(bind_proxy(true));
    let resp = raw_exchange(
        &proxy,
        "GET http://169.254.169.254/latest/meta-data HTTP/1.1\r\nhost: 169.254.169.254\r\n\r\n",
    );
    assert!(resp.starts_with("HTTP/1.1 403"), "got: {resp:?}");
}

#[test]
fn control_plane_service_name_refused() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let proxy = rt.block_on(bind_proxy(true));
    let resp = raw_exchange(
        &proxy,
        "CONNECT postgres:5432 HTTP/1.1\r\nhost: postgres:5432\r\n\r\n",
    );
    assert!(resp.starts_with("HTTP/1.1 403"), "got: {resp:?}");
}

/// Loopback echo server; returns its port.
fn spawn_echo() -> u16 {
    let l = StdListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        while let Ok((mut s, _)) = l.accept() {
            std::thread::spawn(move || {
                let mut buf = [0_u8; 4096];
                while let Ok(n) = s.read(&mut buf) {
                    if n == 0 || s.write_all(&buf[..n]).is_err() {
                        break;
                    }
                }
            });
        }
    });
    port
}

#[test]
fn connect_tunnel_passes_bytes_when_allowed() {
    let echo = spawn_echo();
    let rt = tokio::runtime::Runtime::new().unwrap();
    // Blocklist enforcement off: proves the CONNECT machinery against a
    // loopback target (the enforced-block cases are covered above).
    let proxy = rt.block_on(bind_proxy(false));
    let mut s = std::net::TcpStream::connect(&proxy).unwrap();
    write!(
        s,
        "CONNECT 127.0.0.1:{echo} HTTP/1.1\r\nhost: 127.0.0.1:{echo}\r\n\r\n"
    )
    .unwrap();
    let mut hdr = vec![0_u8; 128];
    let n = s.read(&mut hdr).unwrap();
    let head = String::from_utf8_lossy(&hdr[..n]);
    assert!(head.starts_with("HTTP/1.1 200"), "got: {head:?}");
    s.write_all(b"tunnel-ping").unwrap();
    let mut pong = [0_u8; 11];
    s.read_exact(&mut pong).unwrap();
    assert_eq!(&pong, b"tunnel-ping");
}

/// Minimal loopback HTTP origin; returns its port.
fn spawn_origin() -> u16 {
    let l = StdListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        while let Ok((mut s, _)) = l.accept() {
            let mut buf = [0_u8; 4096];
            let _ = s.read(&mut buf);
            let _ = s.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: 10\r\nconnection: close\r\n\r\nhello-body",
            );
        }
    });
    port
}

#[test]
fn plain_http_forward_when_allowed() {
    let origin = spawn_origin();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let proxy = rt.block_on(bind_proxy(false));
    let resp = raw_exchange(
        &proxy,
        &format!("GET http://127.0.0.1:{origin}/some/path?q=1 HTTP/1.1\r\nhost: 127.0.0.1:{origin}\r\n\r\n"),
    );
    assert!(resp.starts_with("HTTP/1.1 200"), "got: {resp:?}");
    assert!(resp.contains("hello-body"), "got: {resp:?}");
}

#[test]
fn sim_chat_still_served() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let proxy = rt.block_on(bind_proxy(true));
    let body = r#"{"run_id":"r1","model":"m","messages":[{"role":"u"}]}"#;
    let req = format!(
        "POST /v1/chat HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    let resp = raw_exchange(&proxy, &req);
    assert!(resp.starts_with("HTTP/1.1 200"), "got: {resp:?}");
    assert!(resp.contains("sim-reply"), "got: {resp:?}");
}

#[test]
#[ignore = "requires internet"]
fn live_public_endpoints_reachable() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let proxy = rt.block_on(bind_proxy(true));
    // CONNECT path (what pip/HTTPS clients use).
    let resp = raw_exchange(
        &proxy,
        "CONNECT pypi.org:443 HTTP/1.1\r\nhost: pypi.org:443\r\n\r\n",
    );
    assert!(resp.starts_with("HTTP/1.1 200"), "CONNECT pypi: {resp:?}");
    // Plain-HTTP forward path.
    let resp = raw_exchange(
        &proxy,
        "GET http://example.com/ HTTP/1.1\r\nhost: example.com\r\n\r\n",
    );
    assert!(
        resp.starts_with("HTTP/1.1 200"),
        "GET example.com: {resp:?}"
    );
    // Metadata stays refused even live.
    let resp = raw_exchange(
        &proxy,
        "CONNECT 169.254.169.254:443 HTTP/1.1\r\nhost: 169.254.169.254:443\r\n\r\n",
    );
    assert!(resp.starts_with("HTTP/1.1 403"), "metadata: {resp:?}");
}
