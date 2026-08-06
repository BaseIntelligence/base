#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Viewer lockdown defense in depth: miner-controlled HTML proxied through
//! `/challenge/{id}/v1/view/*` must leave the gateway with the CSP `sandbox`
//! floor (opaque origin, no scripts) and without `Set-Cookie` — even when the
//! upstream challenge service sends weak headers.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use chain::FakeChain;
use gateway::{
    build_app_with_bundles, CreateBackend, MemoryBundleStore, MemoryRawWeightStore, Registry,
    RegistryConfig, SharedChain, TlsConfig,
};
use telemetry::init_metrics;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn spawn_gateway(registry: Arc<Registry>) -> (SocketAddr, oneshot::Sender<()>) {
    let _ = telemetry::init_tracing();
    let metrics = init_metrics().expect("metrics");
    let chain: SharedChain = Arc::new(validator_sync::SyncChain::new(FakeChain::with_defaults()));
    let app = build_app_with_bundles(
        metrics,
        registry,
        chain,
        &TlsConfig::default(),
        Arc::new(trustroot::ChallengesBody::default()),
        Arc::new(MemoryRawWeightStore::new()),
        Arc::new(MemoryBundleStore::new()),
    )
    .expect("router");

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (tx, rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await
            .expect("serve");
    });

    let client = reqwest::Client::new();
    for _ in 0..80 {
        if client
            .get(format!("http://{addr}/healthz"))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    (addr, tx)
}

#[tokio::test]
async fn view_response_leaves_gateway_sandboxed_without_cookies() {
    let upstream = MockServer::start().await;
    // Worst-case upstream: weak CSP, a cookie, and a script payload.
    Mock::given(method("GET"))
        .and(path("/v1/view/run1/index.html"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html; charset=utf-8")
                .insert_header("content-security-policy", "default-src *")
                .insert_header("set-cookie", "session=evil; Path=/")
                .set_body_string("<html><script>alert(1)</script>miner page</html>"),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/stats"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string("{\"ok\":true}"),
        )
        .mount(&upstream)
        .await;

    let reg = Registry::shared(RegistryConfig {
        failure_threshold: 2,
        cooldown: Duration::from_millis(120),
    });
    reg.create(&CreateBackend {
        challenge_id: "design".into(),
        base_url: upstream.uri(),
        weight: 1,
    })
    .unwrap();

    let (addr, shutdown) = spawn_gateway(reg).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!(
            "http://{addr}/challenge/design/v1/view/run1/index.html"
        ))
        .send()
        .await
        .expect("proxy view");
    assert_eq!(resp.status().as_u16(), 200);
    let headers = resp.headers().clone();
    let body = resp.text().await.unwrap();

    // Body and 200 semantics survive untouched.
    assert_eq!(body, "<html><script>alert(1)</script>miner page</html>");

    // Lockdown floor re-applied over the weak upstream CSP.
    let csp = headers
        .get("content-security-policy")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(csp.starts_with("sandbox;"), "{csp}");
    assert!(!csp.contains("allow-scripts"), "{csp}");
    assert!(!csp.contains("allow-same-origin"), "{csp}");
    assert!(csp.contains("default-src 'none'"), "{csp}");
    assert!(csp.contains("frame-ancestors"), "{csp}");
    assert!(csp.contains("https://joinbase.ai"), "{csp}");

    assert_eq!(
        headers
            .get("x-content-type-options")
            .and_then(|v| v.to_str().ok()),
        Some("nosniff")
    );
    assert_eq!(
        headers.get("referrer-policy").and_then(|v| v.to_str().ok()),
        Some("no-referrer")
    );
    // Never cookies on miner-content responses.
    assert!(headers.get("set-cookie").is_none());

    // Non-view proxy paths are not rewritten.
    let stats = client
        .get(format!("http://{addr}/challenge/design/v1/stats"))
        .send()
        .await
        .expect("proxy stats");
    assert_eq!(stats.status().as_u16(), 200);
    assert!(stats.headers().get("content-security-policy").is_none());

    let _ = shutdown.send(());
}
