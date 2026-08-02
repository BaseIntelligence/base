//! Manual-surface QA: `miner attest-grant` against a stub master gateway.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use tokio::process::Command;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const MINER_HK: &str = "a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5";

fn miner_bin() -> PathBuf {
    // Same-package bin: cargo sets CARGO_BIN_EXE_miner for integration tests.
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_miner") {
        return PathBuf::from(p);
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../target/debug/miner");
    p
}

/// Write a fresh receipt mini-secret somewhere unique; content never printed.
fn receipt_sk_file() -> (PathBuf, [u8; 32]) {
    let unique = format!(
        "base-attest-grant-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    );
    let path = std::env::temp_dir().join(unique);
    let secret = crypto::generate_mini_secret();
    std::fs::write(&path, secret).expect("write receipt_sk");
    (path, secret)
}

fn expected_pk_hex(secret: &[u8; 32]) -> String {
    hex::encode(crypto::public_key_from_mini_secret(secret).expect("derive receipt pk"))
}

fn expected_outcome(hk: &str, pk: &str) -> serde_json::Value {
    serde_json::json!({
        "epoch": 55,
        "miner_hotkey_hex": hk,
        "receipt_pk_hex": pk,
        "attempt": 1,
        "outcome": "verified",
    })
}

#[tokio::test]
async fn cli_grant_happy_path_posts_and_prints_outcome() {
    let server = MockServer::start().await;
    let (receipt_path, receipt_secret) = receipt_sk_file();
    let expected_pk = expected_pk_hex(&receipt_secret);
    Mock::given(method("POST"))
        .and(path("/v1/admin/attest-grant"))
        .and(body_json(serde_json::json!({
            "epoch": 55,
            "miner_hotkey_hex": MINER_HK,
            "receipt_pk_hex": expected_pk,
            "reason": "testnet 541: spec test",
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(expected_outcome(MINER_HK, &expected_pk)),
        )
        .expect(1)
        .mount(&server)
        .await;

    let out = Command::new(miner_bin())
        .args([
            "attest-grant",
            "--gateway-url",
            &server.uri(),
            "--epoch",
            "55",
            "--miner-hotkey-hex",
            MINER_HK,
            "--receipt-sk-file",
            receipt_path.to_str().expect("utf8 path"),
            "--reason",
            "testnet 541: spec test",
        ])
        .output()
        .await
        .expect("spawn miner");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let _ = std::fs::remove_file(&receipt_path);
    assert!(
        out.status.success(),
        "cli failed:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains(&format!("miner_hotkey={MINER_HK}")),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains(&format!("receipt_pk={expected_pk}")),
        "stdout: {stdout}"
    );
    assert!(stdout.contains("epoch=55"), "stdout: {stdout}");
    assert!(stdout.contains("attempt=1"), "stdout: {stdout}");
    assert!(stdout.contains("outcome=verified"), "stdout: {stdout}");
    server.verify().await;
}

#[tokio::test]
async fn cli_grant_gateway_400_exits_nonzero() {
    let server = MockServer::start().await;
    let (receipt_path, _secret) = receipt_sk_file();
    Mock::given(method("POST"))
        .and(path("/v1/admin/attest-grant"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": "reason must be a non-empty audit note"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let out = Command::new(miner_bin())
        .args([
            "attest-grant",
            "--gateway-url",
            &server.uri(),
            "--epoch",
            "55",
            "--miner-hotkey-hex",
            MINER_HK,
            "--receipt-sk-file",
            receipt_path.to_str().expect("utf8 path"),
            "--reason",
            "still non-empty but gateway refuses",
        ])
        .output()
        .await
        .expect("spawn miner");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let _ = std::fs::remove_file(&receipt_path);
    assert!(!out.status.success(), "expected failure: {:?}", out.status);
    assert!(stderr.contains("gateway status 400"), "stderr: {stderr}");
    server.verify().await;
}
