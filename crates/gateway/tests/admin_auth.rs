//! `/v1/admin/*` bearer gate (incident: public unauthenticated seal/backends).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use axum::Router;
use chain::{FakeChain, FakeChainConfig};
use gateway::{
    admin_auth_middleware, admin_seal_router, registry_router, AdminAuth, ChallengeEntry,
    ChallengesBody, GatewayState, MemoryBundleStore, MemoryRawWeightStore, ParticipantPolicy,
    Registry, RegistryConfig, SharedChain, BPS_DENOM,
};
use tokio::net::TcpListener;
use validator_sync::SyncChain;

fn fake_chain() -> SharedChain {
    Arc::new(SyncChain::new(FakeChain::new(FakeChainConfig {
        hotkeys: vec![vec![1u8; 32]],
        owner_hotkey: vec![0xA1; 32],
        current_block: 100,
        ..FakeChainConfig::default()
    })))
}

fn challenges() -> Arc<ChallengesBody> {
    Arc::new(ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: b"c".to_vec(),
            public_key: [9u8; 32],
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    })
}

async fn spawn(app: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    format!("http://{addr}")
}

fn app_with_auth(auth: AdminAuth) -> Router {
    let reg = Registry::shared(RegistryConfig::default());
    let state = GatewayState::with_parts(
        reg,
        fake_chain(),
        challenges(),
        Arc::new(MemoryRawWeightStore::new()),
        Arc::new(MemoryBundleStore::new()),
    )
    .expect("state");
    Router::new()
        .merge(registry_router(state.clone()))
        .merge(admin_seal_router(state))
        .layer(axum::middleware::from_fn_with_state(
            auth,
            admin_auth_middleware,
        ))
}

#[tokio::test]
async fn admin_backends_open_without_token_config() {
    let base = spawn(app_with_auth(AdminAuth::open())).await;
    let res = reqwest::get(format!("{base}/v1/admin/backends"))
        .await
        .expect("get");
    assert_eq!(res.status(), 200);
}

#[tokio::test]
async fn admin_backends_rejects_missing_bearer() {
    let base = spawn(app_with_auth(AdminAuth::require_token("s3cret"))).await;
    let res = reqwest::get(format!("{base}/v1/admin/backends"))
        .await
        .expect("get");
    assert_eq!(res.status(), 401);
    let body: serde_json::Value = res.json().await.expect("json");
    assert_eq!(body["code"], "admin_unauthorized");
}

#[tokio::test]
async fn admin_backends_accepts_matching_bearer() {
    let base = spawn(app_with_auth(AdminAuth::require_token("s3cret"))).await;
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{base}/v1/admin/backends"))
        .header("Authorization", "Bearer s3cret")
        .json(&serde_json::json!({
            "challenge_id": "c",
            "base_url": "http://127.0.0.1:9",
            "weight": 1
        }))
        .send()
        .await
        .expect("post");
    assert_eq!(res.status(), 201);
}

#[tokio::test]
async fn non_admin_paths_stay_open() {
    let base = spawn(app_with_auth(AdminAuth::require_token("s3cret"))).await;
    // No weights router mounted → 404, not 401.
    let res = reqwest::get(format!("{base}/v1/weights/latest"))
        .await
        .expect("get");
    assert_eq!(res.status(), 404);
}
