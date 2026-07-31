#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Todo 15: register hypertraining challenge backend when trust root has the row.
//!
//! S1 trust-root load includes `hypertraining`
//! S2 POST /v1/admin/backends → 201 with compose `base_url`
//! S3 agent-v1 registration still works (multi-challenge, not agent-v1-only)

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use gateway::{
    build_app_with, ChallengesBody, MemoryRawWeightStore, RawWeightStore, Registry, RegistryConfig,
    TlsConfig,
};
use telemetry::init_metrics;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use trustroot::load_config_dir;

const HYPERTRAINING_ID: &str = "hypertraining";
const HYPERTRAINING_BASE: &str = "http://hypertraining-challenge:8091";
const AGENT_V1_ID: &str = "agent-v1";
const AGENT_V1_BASE: &str = "http://agent-challenge:8090";

fn load_repo_challenges() -> ChallengesBody {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config");
    assert!(
        root.join("challenges.toml").is_file(),
        "repo config/challenges.toml must exist"
    );
    let (ch, _ms) = load_config_dir(&root, 0, 3).expect("committed config must verify");
    let primary = ch.primary().expect("active challenges root");
    primary
        .body
        .get(HYPERTRAINING_ID.as_bytes())
        .expect("trust root must include hypertraining row before gateway backend registration");
    primary
        .body
        .get(AGENT_V1_ID.as_bytes())
        .expect("trust root must still include agent-v1");
    primary.body.clone()
}

async fn spawn_gateway(challenges: ChallengesBody) -> (SocketAddr, oneshot::Sender<()>) {
    let _ = telemetry::init_tracing();
    let metrics = init_metrics().expect("metrics");
    let registry = Registry::shared(RegistryConfig::default());
    let store = Arc::new(MemoryRawWeightStore::new());
    let app = build_app_with(
        metrics,
        registry,
        &TlsConfig::default(),
        Arc::new(challenges),
        store as Arc<dyn RawWeightStore>,
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
async fn s1_post_admin_backends_hypertraining_when_trust_root_has_row() {
    let challenges = load_repo_challenges();
    let (addr, shutdown) = spawn_gateway(challenges).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://{addr}/v1/admin/backends"))
        .json(&serde_json::json!({
            "challenge_id": HYPERTRAINING_ID,
            "base_url": HYPERTRAINING_BASE,
        }))
        .send()
        .await
        .expect("POST backends");

    assert_eq!(
        resp.status().as_u16(),
        201,
        "body={}",
        resp.text().await.unwrap_or_default()
    );
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["challenge_id"], HYPERTRAINING_ID);
    assert_eq!(body["base_url"], HYPERTRAINING_BASE);
    assert_eq!(body["healthy"], true);
    assert_eq!(body["ejected"], false);
    assert!(body.get("id").is_some());
    // D18: no key-shaped fields
    let obj = body.as_object().expect("object");
    for k in obj.keys() {
        assert!(
            !k.to_lowercase().contains("key") && !k.to_lowercase().contains("secret"),
            "D18 unexpected field {k}"
        );
    }

    let listed = client
        .get(format!(
            "http://{addr}/v1/admin/backends?challenge_id={HYPERTRAINING_ID}"
        ))
        .send()
        .await
        .expect("list")
        .json::<Vec<serde_json::Value>>()
        .await
        .expect("list json");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["challenge_id"], HYPERTRAINING_ID);
    assert_eq!(listed[0]["base_url"], HYPERTRAINING_BASE);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn s2_multi_challenge_agent_v1_and_hypertraining_register() {
    let challenges = load_repo_challenges();
    let (addr, shutdown) = spawn_gateway(challenges).await;
    let client = reqwest::Client::new();

    for (cid, base) in [
        (HYPERTRAINING_ID, HYPERTRAINING_BASE),
        (AGENT_V1_ID, AGENT_V1_BASE),
    ] {
        let resp = client
            .post(format!("http://{addr}/v1/admin/backends"))
            .json(&serde_json::json!({
                "challenge_id": cid,
                "base_url": base,
            }))
            .send()
            .await
            .expect("POST");
        assert_eq!(
            resp.status().as_u16(),
            201,
            "challenge_id={cid} body={}",
            resp.text().await.unwrap_or_default()
        );
    }

    let all = client
        .get(format!("http://{addr}/v1/admin/backends"))
        .send()
        .await
        .expect("list all")
        .json::<Vec<serde_json::Value>>()
        .await
        .expect("json");
    assert_eq!(all.len(), 2);
    let ids: Vec<&str> = all
        .iter()
        .filter_map(|v| v["challenge_id"].as_str())
        .collect();
    assert!(ids.contains(&HYPERTRAINING_ID));
    assert!(ids.contains(&AGENT_V1_ID));

    let _ = shutdown.send(());
}
