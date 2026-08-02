//! `POST /v1/admin/attest-grant` HTTP contract: validation classes and the
//! pool-failure surface, served by a real listener. Row-insertion correctness
//! is covered by `db` integration tests against real Postgres; the challenge
//! gate consumption is covered by `bins/agent-challenge` attestation lookup
//! tests.

#![allow(clippy::expect_used)]

use gateway::{admin_attest_grant_router, AttestGrantState, ATTEST_GRANT_ROUTE};
use tokio::net::TcpListener;

/// Bind the router on an ephemeral port; the lazy pool is dialed only when a
/// query actually runs, which keeps validation tests database-free.
async fn spawn() -> String {
    // `connect_lazy` succeeds without a server; the first query then fails,
    // exercising the 500 surface deterministically.
    let pool = db::PgPool::connect_lazy("postgres://127.0.0.1:1/base").expect("lazy pool");
    let state = AttestGrantState::new(pool, [0x9A; 32]);
    let app = admin_attest_grant_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}{ATTEST_GRANT_ROUTE}")
}

fn body(miner_hotkey_hex: &str, receipt_pk_hex: &str, reason: &str) -> serde_json::Value {
    serde_json::json!({
        "epoch": 55,
        "miner_hotkey_hex": miner_hotkey_hex,
        "receipt_pk_hex": receipt_pk_hex,
        "reason": reason,
    })
}

const HK: &str = "a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5";
const PK: &str = "6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f6f";

#[tokio::test]
async fn happy_shape_reaches_the_database() {
    let url = spawn().await;
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body(HK, PK, "testnet 541: no TEE backend available"))
        .send()
        .await
        .expect("post");
    // The pool has no server behind it, so the insert fails: 500 with the db
    // error proves every validator passed (400 would mean a field was refused).
    assert_eq!(resp.status(), 500);
    let json: serde_json::Value = resp.json().await.expect("json");
    assert!(json["error"].as_str().expect("error").contains("pool"));
}

#[tokio::test]
async fn rejects_bad_hex_keys_and_length_classes() {
    let url = spawn().await;
    for (hk, pk, reason) in [
        ("zz", PK, "ok"),
        ("aa", PK, "ok"),
        (HK, "zz", "ok"),
        (HK, &PK[..62], "ok"),
    ] {
        let resp = reqwest::Client::new()
            .post(&url)
            .json(&body(hk, pk, reason))
            .send()
            .await
            .expect("post");
        assert_eq!(resp.status(), 400, "hk={hk} pk={pk} must 400");
    }
}

#[tokio::test]
async fn rejects_empty_and_oversized_reason() {
    let url = spawn().await;
    for reason in ["", "   ", &"x".repeat(600)] {
        let resp = reqwest::Client::new()
            .post(&url)
            .json(&body(HK, PK, reason))
            .send()
            .await
            .expect("post");
        assert_eq!(resp.status(), 400, "reason must 400: len {}", reason.len());
    }
}

#[tokio::test]
async fn rejects_unknown_fields_and_surrounding_whitespace() {
    let url = spawn().await;
    let mut b = body(HK, PK, "note");
    b["surprise"] = serde_json::json!(true);
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&b)
        .send()
        .await
        .expect("post");
    assert_eq!(resp.status(), 422);

    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body(&format!(" {HK}"), PK, "note"))
        .send()
        .await
        .expect("post");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn accepts_0x_prefixed_hex() {
    let url = spawn().await;
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body(&format!("0x{HK}"), &format!("0x{PK}"), "note"))
        .send()
        .await
        .expect("post");
    assert_eq!(resp.status(), 500); // validation passed; pool unreachable
}

#[test]
fn response_serializes_expected_fields() {
    let r = gateway::AttestGrantResponse {
        epoch: 55,
        miner_hotkey_hex: "aa".repeat(32),
        receipt_pk_hex: "bb".repeat(32),
        attempt: 3,
        outcome: "verified".to_owned(),
    };
    let json = serde_json::to_value(&r).expect("serialize");
    assert_eq!(json["outcome"], "verified");
    assert_eq!(json["attempt"], 3);
    assert_eq!(json["epoch"], 55);
}
