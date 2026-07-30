//! Attest router mounted on the main validator HTTP app (task-01).
//!
//! S1: POST /v1/attest/nonce → 200 + 64-hex nonce
//! S2: known-good fixture submit → verified + credit
//! S3: replay same nonce → rejected, credit retained
//! S4: /healthz still answers on the same process

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use attest_parse::{parse_tdx_quote_v4, patch_report_data};
use attest_policy::{compute_report_data, replay_compose_hash, ReportDataBinding};
use attest_replay::events_from_json;
use chain::FakeChain;
use crypto::KEY_LEN;
use trustroot::{MeasurementEntry, MeasurementsBody};

use crate::{spawn_validator_with_ok_db, AttestState, SyncChain, ValidatorRuntime};

const QUOTE: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/quote.bin");
const EVENT_LOG: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/event_log.json");

fn real_measurements() -> MeasurementsBody {
    let parsed = parse_tdx_quote_v4(QUOTE).expect("parse");
    let events = events_from_json(EVENT_LOG).expect("events");
    let (compose_hash, replay) = replay_compose_hash(&events).expect("replay");
    assert_eq!(replay.rtmr3, parsed.td_report.rtmr3);
    MeasurementsBody {
        entries: vec![MeasurementEntry {
            mr_td: parsed.td_report.mr_td,
            rtmr0: parsed.td_report.rtmr0,
            rtmr1: parsed.td_report.rtmr1,
            rtmr2: parsed.td_report.rtmr2,
            rtmr3: parsed.td_report.rtmr3,
            compose_hash,
        }],
    }
}

fn keys() -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
    ([0xaa; KEY_LEN], [0xcc; KEY_LEN])
}

async fn boot_validator_with_attest() -> crate::RunningValidator {
    let (_miner, validator_hk) = keys();
    let attest = AttestState::with_ok_verifier(real_measurements(), validator_hk, 1);
    let chain = Arc::new(SyncChain::new(FakeChain::with_defaults()));
    let runtime = ValidatorRuntime {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        attest: Some(attest),
        ..ValidatorRuntime::default()
    };
    let running = spawn_validator_with_ok_db(runtime, chain)
        .await
        .expect("spawn");
    tokio::time::sleep(Duration::from_millis(30)).await;
    running
}

/// S1 — nonce issue on the main validator app returns 200 + 64-hex nonce.
#[tokio::test]
async fn s1_attest_nonce_200_on_validator_app() {
    let running = boot_validator_with_attest().await;
    let base = running.base_url();
    let (miner, _) = keys();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/attest/nonce"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner),
            "epoch": 1,
        }))
        .send()
        .await
        .expect("http");
    assert_eq!(resp.status().as_u16(), 200, "nonce status");
    let body: serde_json::Value = resp.json().await.expect("json");
    let nonce_hex = body["nonce_hex"].as_str().expect("nonce_hex");
    assert_eq!(nonce_hex.len(), 64, "64 hex chars for 32-byte nonce");
    assert!(
        nonce_hex.chars().all(|c| c.is_ascii_hexdigit()),
        "nonce must be hex: {nonce_hex}"
    );
    assert_eq!(body["epoch"], 1);
    running.shutdown().await.expect("shutdown");
}

/// S2 — known-good fixture submit → verified + credit on the book.
#[tokio::test]
async fn s2_known_good_submit_verified() {
    let running = boot_validator_with_attest().await;
    let base = running.base_url();
    let (miner, validator_hk) = keys();
    let client = reqwest::Client::new();

    let nonce_resp: serde_json::Value = client
        .post(format!("{base}/v1/attest/nonce"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner),
            "epoch": 1,
            "netuid": 1,
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let nonce_hex = nonce_resp["nonce_hex"].as_str().unwrap().to_owned();
    let mut nonce = [0u8; KEY_LEN];
    nonce.copy_from_slice(&hex::decode(&nonce_hex).unwrap());

    let binding = ReportDataBinding {
        netuid: 1,
        epoch: 1,
        miner_pubkey: miner,
        nonce,
        validator_hotkey: validator_hk,
    };
    let rd = compute_report_data(&binding);
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &rd).unwrap();
    let event_log = String::from_utf8(EVENT_LOG.to_vec()).unwrap();

    let submit: serde_json::Value = client
        .post(format!("{base}/v1/attest/submit"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner),
            "epoch": 1,
            "netuid": 1,
            "nonce_hex": nonce_hex,
            "quote_hex": hex::encode(&quote),
            "event_log_json": event_log,
            "validator_hotkey_hex": hex::encode(validator_hk),
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(submit["outcome"], "verified", "submit={submit}");
    assert_eq!(submit["grants_credit"], true);
    assert!(
        running.attest.has_credit(1, 1, miner).await,
        "credit book must record Verified"
    );
    running.shutdown().await.expect("shutdown");
}

/// S3 — replaying the same nonce is rejected; prior credit stays.
#[tokio::test]
async fn s3_nonce_replay_rejected() {
    let running = boot_validator_with_attest().await;
    let base = running.base_url();
    let (miner, validator_hk) = keys();
    let client = reqwest::Client::new();

    let nonce_resp: serde_json::Value = client
        .post(format!("{base}/v1/attest/nonce"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner),
            "epoch": 1,
            "netuid": 1,
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let nonce_hex = nonce_resp["nonce_hex"].as_str().unwrap().to_owned();
    let mut nonce = [0u8; KEY_LEN];
    nonce.copy_from_slice(&hex::decode(&nonce_hex).unwrap());

    let binding = ReportDataBinding {
        netuid: 1,
        epoch: 1,
        miner_pubkey: miner,
        nonce,
        validator_hotkey: validator_hk,
    };
    let rd = compute_report_data(&binding);
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &rd).unwrap();
    let event_log = String::from_utf8(EVENT_LOG.to_vec()).unwrap();
    let body = serde_json::json!({
        "miner_hotkey_hex": hex::encode(miner),
        "epoch": 1,
        "netuid": 1,
        "nonce_hex": nonce_hex,
        "quote_hex": hex::encode(&quote),
        "event_log_json": event_log,
        "validator_hotkey_hex": hex::encode(validator_hk),
    });

    let first: serde_json::Value = client
        .post(format!("{base}/v1/attest/submit"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(first["outcome"], "verified");

    let second: serde_json::Value = client
        .post(format!("{base}/v1/attest/submit"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(second["outcome"], "rejected", "replay={second}");
    assert_eq!(second["reason"], "nonce_invalid");
    assert_eq!(second["grants_credit"], false);
    assert!(
        running.attest.has_credit(1, 1, miner).await,
        "replay must not wipe Verified credit"
    );
    running.shutdown().await.expect("shutdown");
}

/// S4 — pre-existing health surface still works alongside attest.
#[tokio::test]
async fn s4_healthz_still_works() {
    let running = boot_validator_with_attest().await;
    let base = running.base_url();
    let resp = reqwest::get(format!("{base}/healthz"))
        .await
        .expect("healthz");
    assert_eq!(resp.status().as_u16(), 200);
    running.shutdown().await.expect("shutdown");
}
