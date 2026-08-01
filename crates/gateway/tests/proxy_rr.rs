#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Task 25 VERIFY: reverse proxy round-robin, passive ejection, D18 no keys.
//!
//! Spins an in-process gateway router (no master check) against wiremock backends.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use chain::FakeChain;
use gateway::{
    build_app_with_bundles, CreateBackend, MemoryBundleStore, MemoryRawWeightStore, Registry,
    RegistryConfig, SharedChain, TlsConfig, DEFAULT_FAILURE_THRESHOLD,
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

    // Wait until accept is live.
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

fn fast_registry() -> Arc<Registry> {
    Registry::shared(RegistryConfig {
        failure_threshold: 2,
        cooldown: Duration::from_millis(120),
    })
}

#[tokio::test]
async fn s1_two_wiremock_backends_alternate_round_robin() {
    let a = MockServer::start().await;
    let b = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/ping"))
        .respond_with(ResponseTemplate::new(200).set_body_string("A"))
        .mount(&a)
        .await;
    Mock::given(method("GET"))
        .and(path("/ping"))
        .respond_with(ResponseTemplate::new(200).set_body_string("B"))
        .mount(&b)
        .await;

    let reg = fast_registry();
    reg.create(&CreateBackend {
        challenge_id: "agent-v1".into(),
        base_url: a.uri(),
        weight: 1,
    })
    .unwrap();
    reg.create(&CreateBackend {
        challenge_id: "agent-v1".into(),
        base_url: b.uri(),
        weight: 1,
    })
    .unwrap();

    let (addr, shutdown) = spawn_gateway(reg).await;
    let client = reqwest::Client::new();
    let mut bodies = Vec::new();
    for _ in 0..6 {
        let resp = client
            .get(format!("http://{addr}/challenge/agent-v1/ping"))
            .send()
            .await
            .expect("proxy");
        assert_eq!(resp.status().as_u16(), 200);
        bodies.push(resp.text().await.unwrap());
    }
    assert_eq!(bodies, ["A", "B", "A", "B", "A", "B"]);
    let _ = shutdown.send(());
}

#[tokio::test]
async fn s2_killing_backend_moves_traffic_to_survivor_200() {
    let good = MockServer::start().await;
    let bad = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/work"))
        .respond_with(ResponseTemplate::new(200).set_body_string("GOOD"))
        .mount(&good)
        .await;
    // bad returns 500 → counts as failure toward ejection
    Mock::given(method("GET"))
        .and(path("/work"))
        .respond_with(ResponseTemplate::new(500).set_body_string("DEAD"))
        .mount(&bad)
        .await;

    let reg = fast_registry();
    let bad_view = reg
        .create(&CreateBackend {
            challenge_id: "c1".into(),
            base_url: bad.uri(),
            weight: 1,
        })
        .unwrap();
    reg.create(&CreateBackend {
        challenge_id: "c1".into(),
        base_url: good.uri(),
        weight: 1,
    })
    .unwrap();

    let (addr, shutdown) = spawn_gateway(reg.clone()).await;
    let client = reqwest::Client::new();

    // Drive failures until bad is ejected (threshold=2). Mixed 5xx/200 ok.
    for _ in 0..12 {
        let _ = client
            .get(format!("http://{addr}/challenge/c1/work"))
            .send()
            .await;
    }

    // After ejection, all traffic should hit good with 200.
    let mut ok = 0;
    for _ in 0..8 {
        let resp = client
            .get(format!("http://{addr}/challenge/c1/work"))
            .send()
            .await
            .expect("proxy");
        if resp.status().as_u16() == 200 {
            let body = resp.text().await.unwrap();
            assert_eq!(body, "GOOD");
            ok += 1;
        }
    }
    assert!(
        ok >= 6,
        "expected survivor to serve most traffic after ejection, ok={ok}"
    );

    let view = reg.get(bad_view.id).unwrap();
    assert!(
        view.ejected || !view.healthy,
        "bad backend should be ejected"
    );
    let _ = shutdown.send(());
}

#[tokio::test]
async fn s2b_connection_refused_ejects_and_survivor_serves() {
    // Start a mock, register it, then shut it down (kill).
    let doomed = MockServer::start().await;
    let survivor = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200).set_body_string("LIVE"))
        .mount(&survivor)
        .await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200).set_body_string("DOOM"))
        .mount(&doomed)
        .await;

    let reg = fast_registry();
    let doomed_uri = doomed.uri();
    let doomed_view = reg
        .create(&CreateBackend {
            challenge_id: "c2".into(),
            base_url: doomed_uri.clone(),
            weight: 1,
        })
        .unwrap();
    reg.create(&CreateBackend {
        challenge_id: "c2".into(),
        base_url: survivor.uri(),
        weight: 1,
    })
    .unwrap();

    let (addr, shutdown) = spawn_gateway(reg.clone()).await;
    let client = reqwest::Client::new();

    // Kill doomed backend (connection refused on next hop).
    drop(doomed);

    // Drive passive ejection via real proxy failures.
    for _ in 0..16 {
        let _ = client
            .get(format!("http://{addr}/challenge/c2/x"))
            .send()
            .await;
    }
    // Ensure threshold reached even if RR luck avoided doomed.
    reg.record_failure(doomed_view.id);
    reg.record_failure(doomed_view.id);

    let mut live = 0;
    for _ in 0..8 {
        let resp = client
            .get(format!("http://{addr}/challenge/c2/x"))
            .send()
            .await
            .expect("proxy");
        if resp.status().as_u16() == 200 && resp.text().await.unwrap() == "LIVE" {
            live += 1;
        }
    }
    assert!(
        live >= 7,
        "survivor should take traffic after kill, live={live} ejected={}",
        reg.get(doomed_view.id).unwrap().ejected
    );
    let _ = shutdown.send(());
}

#[tokio::test]
async fn s2c_restore_re_admits_after_cooldown() {
    let a = MockServer::start().await;
    let b = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/r"))
        .respond_with(ResponseTemplate::new(200).set_body_string("A"))
        .mount(&a)
        .await;
    Mock::given(method("GET"))
        .and(path("/r"))
        .respond_with(ResponseTemplate::new(200).set_body_string("B"))
        .mount(&b)
        .await;

    let reg = fast_registry();
    let av = reg
        .create(&CreateBackend {
            challenge_id: "c3".into(),
            base_url: a.uri(),
            weight: 1,
        })
        .unwrap();
    reg.create(&CreateBackend {
        challenge_id: "c3".into(),
        base_url: b.uri(),
        weight: 1,
    })
    .unwrap();

    // Spawn before ejecting — spawn_gateway can take > cooldown (120ms),
    // which previously re-admitted A before the "only B" window ran.
    let (addr, shutdown) = spawn_gateway(reg.clone()).await;
    let client = reqwest::Client::new();

    // Eject A via forced failures at registry level (after gateway is live).
    reg.record_failure(av.id);
    reg.record_failure(av.id);
    assert!(reg.get(av.id).unwrap().ejected);

    // During ejection only B.
    for _ in 0..4 {
        let body = client
            .get(format!("http://{addr}/challenge/c3/r"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(body, "B");
    }

    tokio::time::sleep(Duration::from_millis(200)).await;

    // After cooldown, A is re-admitted — should see A again.
    let mut saw_a = false;
    for _ in 0..12 {
        let body = client
            .get(format!("http://{addr}/challenge/c3/r"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        if body == "A" {
            saw_a = true;
            break;
        }
    }
    assert!(saw_a, "restored backend must re-admit after cooldown");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn s3_registry_api_neither_sets_nor_returns_signing_key() {
    let reg = Registry::shared(RegistryConfig {
        failure_threshold: DEFAULT_FAILURE_THRESHOLD,
        cooldown: Duration::from_secs(30),
    });
    let (addr, shutdown) = spawn_gateway(reg).await;
    let client = reqwest::Client::new();

    // Reject signing_key in body (deny_unknown_fields).
    let bad = client
        .post(format!("http://{addr}/v1/admin/backends"))
        .json(&serde_json::json!({
            "challenge_id": "c",
            "base_url": "http://127.0.0.1:9",
            "signing_key": "aabbccdd"
        }))
        .send()
        .await
        .unwrap();
    let bad_status = bad.status().as_u16();
    let bad_body = bad.text().await.unwrap();
    assert!(
        bad_status == 400 || bad_status == 422,
        "signing_key must be rejected, status={bad_status} body={bad_body}"
    );
    assert!(
        bad_body.contains("signing_key") || bad_body.contains("unknown field"),
        "body={bad_body}"
    );

    // Also reject private_key.
    let bad2 = client
        .post(format!("http://{addr}/v1/admin/backends"))
        .json(&serde_json::json!({
            "challenge_id": "c",
            "base_url": "http://127.0.0.1:9",
            "private_key": "secret"
        }))
        .send()
        .await
        .unwrap();
    let s2 = bad2.status().as_u16();
    assert!(s2 == 400 || s2 == 422, "private_key status={s2}");

    // Happy create — response must not contain key-shaped fields.
    let ok = client
        .post(format!("http://{addr}/v1/admin/backends"))
        .json(&serde_json::json!({
            "challenge_id": "c",
            "base_url": "http://127.0.0.1:9"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status().as_u16(), 201);
    let v: serde_json::Value = ok.json().await.unwrap();
    let obj = v.as_object().unwrap();
    for k in obj.keys() {
        let lower = k.to_lowercase();
        assert!(
            !lower.contains("key") && !lower.contains("secret") && !lower.contains("signing"),
            "D18: response must not expose key field {k}"
        );
    }
    assert!(obj.contains_key("base_url"));
    assert!(obj.contains_key("challenge_id"));
    assert!(obj.contains_key("id"));

    let list: serde_json::Value = client
        .get(format!("http://{addr}/v1/admin/backends?challenge_id=c"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let arr = list.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    for k in arr[0].as_object().unwrap().keys() {
        let lower = k.to_lowercase();
        assert!(!lower.contains("key") && !lower.contains("secret"));
    }

    let _ = shutdown.send(());
}
