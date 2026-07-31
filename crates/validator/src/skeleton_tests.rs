//! Integration scenarios for task-28 validator skeleton.

use std::sync::Arc;
use std::time::Duration;

use crate::{
    is_master_only_path, spawn_validator_with_ok_db, CoordinationClient, SyncChain,
    ValidatorRuntime, MASTER_ONLY_PATHS,
};
use chain::{fake_defaults, FakeChain, FakeChainConfig};
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn get_status(base: &str, route: &str) -> (u16, String) {
    let url = format!("http://{base}{route}");
    let resp = reqwest::get(&url).await.expect("http");
    let status = resp.status().as_u16();
    let body = resp.text().await.expect("body");
    (status, body)
}

/// S1: /readyz 200 against `FakeChain` + wiremock gateway + injectable db ok.
#[tokio::test]
async fn s1_readyz_200_fake_chain_and_wiremock_gateway() {
    let gateway = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/weights/latest"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .mount(&gateway)
        .await;

    // Master-only stubs — must receive zero traffic.
    for p in MASTER_ONLY_PATHS.iter().filter(|p| !p.ends_with('/')) {
        Mock::given(path(*p))
            .respond_with(ResponseTemplate::new(599))
            .mount(&gateway)
            .await;
    }
    Mock::given(path_regex(r"^/challenge/.*"))
        .respond_with(ResponseTemplate::new(599))
        .mount(&gateway)
        .await;

    let chain = Arc::new(SyncChain::new(FakeChain::with_defaults()));
    let runtime = ValidatorRuntime {
        epoch_length: fake_defaults::TEMPO,
        listen_addr: "127.0.0.1:0".parse().expect("addr"),
        gateway_endpoint: Some(gateway.uri()),
        registration: crate::RegistrationStub::new(),
        ..ValidatorRuntime::default()
    };

    let running = spawn_validator_with_ok_db(runtime, chain)
        .await
        .expect("spawn");
    let base = running.addr.to_string();

    tokio::time::sleep(Duration::from_millis(50)).await;

    let (hz, hz_body) = get_status(&base, "/healthz").await;
    assert_eq!(hz, 200, "healthz body={hz_body}");

    let (rz, rz_body) = get_status(&base, "/readyz").await;
    assert_eq!(rz, 200, "readyz body={rz_body}");
    assert!(
        rz_body.contains("chain") && rz_body.contains("db"),
        "readyz should list chain+db checks: {rz_body}"
    );

    let (mz, mz_body) = get_status(&base, "/metrics").await;
    assert_eq!(mz, 200);
    assert!(mz_body.contains("base_up"), "{mz_body}");

    // S2: no master-only hits; only allowlisted coordination traffic.
    let received = gateway.received_requests().await.expect("recv");
    for req in &received {
        let p = req.url.path();
        assert!(
            !is_master_only_path(p),
            "validator called master-only path {p}"
        );
    }
    assert!(
        received
            .iter()
            .any(|r| r.url.path().ends_with("/v1/weights/latest")),
        "expected coordination tick to hit weights/latest, got {received:?}"
    );
    // wiremock unmatched log empty: every hit is the allowlisted path only.
    assert!(
        received
            .iter()
            .all(|r| r.url.path() == "/v1/weights/latest"),
        "unexpected paths in gateway log: {received:?}"
    );

    running.shutdown().await.expect("shutdown");
}

/// S3: healthy with no co-located gateway.
#[tokio::test]
async fn s3_readyz_ok_without_gateway() {
    let chain = Arc::new(SyncChain::new(FakeChain::with_defaults()));
    let runtime = ValidatorRuntime {
        gateway_endpoint: None,
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        ..ValidatorRuntime::default()
    };
    let running = spawn_validator_with_ok_db(runtime, chain)
        .await
        .expect("spawn");
    let base = running.addr.to_string();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let (rz, body) = get_status(&base, "/readyz").await;
    assert_eq!(rz, 200, "{body}");
    assert!(!running.coordination.has_gateway());
    running.shutdown().await.expect("shutdown");
}

/// S4: epoch clock tracks `FakeChain` tip.
#[tokio::test]
async fn s4_epoch_clock_matches_fake_chain() {
    let cfg = FakeChainConfig {
        current_block: 720,
        ..FakeChainConfig::default()
    };
    let chain = Arc::new(SyncChain::new(FakeChain::new(cfg)));
    let runtime = ValidatorRuntime {
        epoch_length: 360,
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        ..ValidatorRuntime::default()
    };
    let running = spawn_validator_with_ok_db(runtime, chain)
        .await
        .expect("spawn");
    let snap = running.epoch_snapshot().expect("epoch");
    assert_eq!(snap.current_block, 720);
    assert_eq!(snap.epoch_index, 2);
    assert_eq!(snap.epoch_start_block, 720);
    running.shutdown().await.expect("shutdown");
}

/// S5: graceful shutdown stops accepting.
#[tokio::test]
async fn s5_graceful_shutdown() {
    let chain = Arc::new(SyncChain::new(FakeChain::with_defaults()));
    let runtime = ValidatorRuntime {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        ..ValidatorRuntime::default()
    };
    let running = spawn_validator_with_ok_db(runtime, chain)
        .await
        .expect("spawn");
    let base = running.addr.to_string();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let (rz, _) = get_status(&base, "/readyz").await;
    assert_eq!(rz, 200);
    running.shutdown().await.expect("shutdown");
    let url = format!("http://{base}/readyz");
    let result = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_millis(300))
        .send()
        .await;
    assert!(result.is_err(), "server should be down after shutdown");
}

/// S2 unit: coordination client path guard (no network for refuse).
#[tokio::test]
async fn s2_coordination_refuses_master_only() {
    let c = CoordinationClient::new(Some("http://example.invalid".into())).unwrap();
    for p in [
        "/v1/weights/raw",
        "/v1/admin/seal",
        "/v1/admin/backends",
        "/v1/master/status",
        "/challenge/x",
    ] {
        let err = c.get_allowed(p).await.expect_err(p);
        assert!(
            matches!(err, crate::CoordinationError::PathNotAllowed { .. }),
            "{p} => {err}"
        );
    }
}
