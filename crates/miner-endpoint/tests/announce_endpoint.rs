//! `POST /v1/miners/endpoint` end-to-end against a real Postgres.
//!
//! Skipped (not failed) when `DATABASE_URL` is unset, the same gate
//! `db/tests/attestation_store.rs` uses, so default CI without a database is
//! unaffected.
//!
//! Scenarios:
//! - happy: a registered miner's signed announcement is stored and read back
//! - 401: a signature from a key other than the announced hotkey
//! - 403: a hotkey that is not in the metagraph
//! - 400: an epoch behind or ahead of the chain's
//! - idempotence: re-announcing in the same epoch replaces, never conflicts
//! - supersede: a later epoch wins for the same hotkey

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use chain::{
    current_epoch_pre_run_coinbase, gather_schedule_state, ChainClient, FakeChain, FakeChainConfig,
};
use db::{miner_endpoints, MinerEndpointRow, PgPool, TestPool};
use miner_endpoint::{
    miner_endpoint_router, sign_endpoint, MinerEndpointBodyV1, MinerEndpointState, SharedChain,
};
use tower::ServiceExt;
use validator_sync::SyncChain;

const NETUID: u16 = 1;

/// Returns `false` when `DATABASE_URL` is unset so default CI (no Postgres) skips.
fn database_url_present() -> bool {
    std::env::var_os("DATABASE_URL").is_some()
}

/// A deterministic mini-secret per test miner (never a real key).
fn secret(fill: u8) -> [u8; 32] {
    [fill; 32]
}

fn public(secret_bytes: &[u8; 32]) -> [u8; 32] {
    crypto::public_key_from_mini_secret(secret_bytes).expect("derive public key")
}

/// A chain whose metagraph contains `hotkeys` and whose epoch index is `epoch`.
fn chain_with(hotkeys: Vec<[u8; 32]>, epoch: u64) -> SharedChain {
    Arc::new(SyncChain::new(FakeChain::new(FakeChainConfig {
        subnet_epoch_index: epoch,
        hotkeys: hotkeys.into_iter().map(|h| h.to_vec()).collect(),
        ..FakeChainConfig::default()
    })))
}

/// The epoch the handler will accept for `chain`.
fn current_epoch(chain: &SharedChain) -> u64 {
    let state = gather_schedule_state(chain.as_ref(), NETUID).expect("schedule");
    current_epoch_pre_run_coinbase(&state, state.current_block)
}

fn router(chain: SharedChain, pool: &PgPool) -> Router {
    miner_endpoint_router(MinerEndpointState::new(chain, pool.clone(), NETUID))
}

/// Body signed by `signer`, announcing `base_url` for `hotkey`.
fn request_json(
    signer: &[u8; 32],
    hotkey: [u8; 32],
    base_url: &str,
    epoch: u64,
) -> serde_json::Value {
    let body = MinerEndpointBodyV1 {
        netuid: NETUID,
        miner_hotkey: hotkey,
        base_url: base_url.as_bytes().to_vec(),
        epoch,
    };
    let sig = sign_endpoint(signer, &body).expect("sign");
    serde_json::json!({
        "netuid": NETUID,
        "miner_hotkey_hex": hex::encode(hotkey),
        "base_url": base_url,
        "epoch": epoch,
        "signature_hex": hex::encode(sig),
    })
}

async fn post(app: Router, payload: &serde_json::Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(miner_endpoint::ENDPOINT_ROUTE)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("request");
    let resp = app.oneshot(req).await.expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn pool() -> TestPool {
    db::test_pool().await.expect("test pool")
}

#[tokio::test]
async fn s1_registered_miner_announcement_is_stored() {
    if !database_url_present() {
        return;
    }
    let tp = pool().await;
    let sk = secret(0x11);
    let hk = public(&sk);
    let chain = chain_with(vec![hk], 42);
    let epoch = current_epoch(&chain);

    let payload = request_json(&sk, hk, "https://cvm.example.com:8443", epoch);
    let (status, body) = post(router(chain, tp.pool()), &payload).await;
    assert_eq!(status, StatusCode::OK, "body={body}");
    let out: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert_eq!(out["epoch"].as_u64(), Some(epoch));
    assert_eq!(
        out["miner_hotkey_hex"].as_str(),
        Some(hex::encode(hk).as_str())
    );

    let rows = miner_endpoints(tp.pool(), i32::from(NETUID), 0)
        .await
        .expect("read");
    assert_eq!(
        rows,
        vec![MinerEndpointRow {
            miner_hotkey: hex::encode(hk),
            base_url: "https://cvm.example.com:8443".to_owned(),
            epoch: i64::try_from(epoch).unwrap(),
        }]
    );
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s2_signature_from_the_wrong_key_is_401() {
    if !database_url_present() {
        return;
    }
    let tp = pool().await;
    let victim = public(&secret(0x11));
    let attacker = secret(0x22);
    // The attacker is registered too: only the signature is wrong, so this
    // isolates the 401 from the 403.
    let chain = chain_with(vec![victim, public(&attacker)], 42);
    let epoch = current_epoch(&chain);

    let payload = request_json(&attacker, victim, "https://evil.example.com", epoch);
    let (status, _) = post(router(chain, tp.pool()), &payload).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let rows = miner_endpoints(tp.pool(), i32::from(NETUID), 0)
        .await
        .expect("read");
    assert!(rows.is_empty(), "a bad signature must store nothing");
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s3_hotkey_absent_from_the_metagraph_is_403() {
    if !database_url_present() {
        return;
    }
    let tp = pool().await;
    let sk = secret(0x33);
    let hk = public(&sk);
    // Metagraph holds somebody else; the signature itself is perfectly valid.
    let chain = chain_with(vec![public(&secret(0x44))], 42);
    let epoch = current_epoch(&chain);

    let payload = request_json(&sk, hk, "https://cvm.example.com", epoch);
    let (status, _) = post(router(chain, tp.pool()), &payload).await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let rows = miner_endpoints(tp.pool(), i32::from(NETUID), 0)
        .await
        .expect("read");
    assert!(
        rows.is_empty(),
        "an unregistered key must not fill the table"
    );
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s4_stale_or_future_epoch_is_400() {
    if !database_url_present() {
        return;
    }
    let tp = pool().await;
    let sk = secret(0x55);
    let hk = public(&sk);
    let chain = chain_with(vec![hk], 42);
    let epoch = current_epoch(&chain);

    for claimed in [epoch - 1, epoch + 1] {
        let payload = request_json(&sk, hk, "https://cvm.example.com", claimed);
        let (status, body) = post(router(Arc::clone(&chain), tp.pool()), &payload).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "claimed={claimed}");
        assert!(
            body.contains(&format!(
                "epoch {claimed} is not the current chain epoch {epoch}"
            )),
            "message must quote both epochs, got {body}"
        );
    }

    let rows = miner_endpoints(tp.pool(), i32::from(NETUID), 0)
        .await
        .expect("read");
    assert!(rows.is_empty());
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s5_reannouncing_in_the_same_epoch_is_idempotent() {
    if !database_url_present() {
        return;
    }
    let tp = pool().await;
    let sk = secret(0x66);
    let hk = public(&sk);
    let chain = chain_with(vec![hk], 42);
    let epoch = current_epoch(&chain);

    let first = request_json(&sk, hk, "https://old.example.com", epoch);
    let (status, _) = post(router(Arc::clone(&chain), tp.pool()), &first).await;
    assert_eq!(status, StatusCode::OK);
    // Byte-identical retry (a lost response) must not 409.
    let (status, _) = post(router(Arc::clone(&chain), tp.pool()), &first).await;
    assert_eq!(status, StatusCode::OK);

    // A mid-epoch redeploy moves the URL instead of erroring.
    let moved = request_json(&sk, hk, "https://new.example.com", epoch);
    let (status, _) = post(router(chain, tp.pool()), &moved).await;
    assert_eq!(status, StatusCode::OK);

    let rows = miner_endpoints(tp.pool(), i32::from(NETUID), 0)
        .await
        .expect("read");
    assert_eq!(rows.len(), 1, "one row per (epoch, hotkey)");
    assert_eq!(rows[0].base_url, "https://new.example.com");
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s6_a_later_epoch_supersedes_an_earlier_url() {
    if !database_url_present() {
        return;
    }
    let tp = pool().await;
    let sk = secret(0x77);
    let hk = public(&sk);

    let early_chain = chain_with(vec![hk], 42);
    let early = current_epoch(&early_chain);
    let payload = request_json(&sk, hk, "https://epoch-a.example.com", early);
    let (status, _) = post(router(early_chain, tp.pool()), &payload).await;
    assert_eq!(status, StatusCode::OK);

    let late_chain = chain_with(vec![hk], 43);
    let late = current_epoch(&late_chain);
    assert!(late > early);
    let payload = request_json(&sk, hk, "https://epoch-b.example.com", late);
    let (status, _) = post(router(late_chain, tp.pool()), &payload).await;
    assert_eq!(status, StatusCode::OK);

    let rows = miner_endpoints(tp.pool(), i32::from(NETUID), 0)
        .await
        .expect("read");
    assert_eq!(
        rows,
        vec![MinerEndpointRow {
            miner_hotkey: hex::encode(hk),
            base_url: "https://epoch-b.example.com".to_owned(),
            epoch: i64::try_from(late).unwrap(),
        }],
        "the newest announcement per hotkey wins"
    );

    // The staleness floor hides an announcement the caller considers too old.
    let floor = i64::try_from(late).unwrap() + 1;
    assert!(miner_endpoints(tp.pool(), i32::from(NETUID), floor)
        .await
        .expect("read")
        .is_empty());
    tp.drop_schema().await.expect("drop schema");
}

#[tokio::test]
async fn s7_an_ssrf_url_never_reaches_the_table() {
    if !database_url_present() {
        return;
    }
    let tp = pool().await;
    let sk = secret(0x88);
    let hk = public(&sk);
    let chain = chain_with(vec![hk], 42);
    let epoch = current_epoch(&chain);

    // Correctly signed by a registered miner — only the URL is hostile.
    let payload = request_json(&sk, hk, "http://169.254.169.254", epoch);
    let (status, body) = post(router(chain, tp.pool()), &payload).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("base_url"), "got {body}");

    let rows = miner_endpoints(tp.pool(), i32::from(NETUID), 0)
        .await
        .expect("read");
    assert!(rows.is_empty());
    tp.drop_schema().await.expect("drop schema");
}

/// `ChainClient` is only used through the trait here; keep the import honest.
#[allow(dead_code)]
fn _assert_chain_client(_: &dyn ChainClient) {}
