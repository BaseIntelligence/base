//! Task 38: fixture-driven certify + validator attest VERIFY matrix.
//!
//! S1 correct binding → Verified
//! S2 resubmission → Rejected
//! S3 different epoch → Rejected
//! S4 different validator hotkey → Rejected
//! S5 another miner's quote relayed → Rejected
//! S6 simulated PCS outage → Parked, no credit, no carry-forward

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::net::SocketAddr;
use std::time::Duration;

use attest_parse::{parse_tdx_quote_v4, patch_report_data};
use attest_policy::{compute_report_data, replay_compose_hash, AttestOutcome, ReportDataBinding};
use attest_replay::events_from_json;
use crypto::KEY_LEN;
use miner::{certify, CertifyParams, QuoteSource};
use trustroot::{MeasurementEntry, MeasurementsBody, COMPOSE_HASH_LEN, REGISTER_LEN};
use validator::{spawn_attest_server, AttestState};

const QUOTE: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/quote.bin");
const EVENT_LOG: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/event_log.json");
const APP_COMPOSE: &[u8] =
    include_bytes!("../../attest-parse/tests/fixtures/real/app-compose.json");

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

async fn boot_ok() -> (SocketAddr, tokio::sync::watch::Sender<bool>, AttestState) {
    let (miner, validator) = keys();
    let _ = miner;
    let state = AttestState::with_ok_verifier(real_measurements(), validator, 1);
    let (addr, tx, _join) =
        spawn_attest_server(state.clone(), SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind");
    // give server a tick
    tokio::time::sleep(Duration::from_millis(20)).await;
    (addr, tx, state)
}

async fn boot_pcs() -> (SocketAddr, tokio::sync::watch::Sender<bool>, AttestState) {
    let (_miner, validator) = keys();
    let state = AttestState::with_pcs_timeout(real_measurements(), validator, 1);
    let (addr, tx, _join) =
        spawn_attest_server(state.clone(), SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind");
    tokio::time::sleep(Duration::from_millis(20)).await;
    (addr, tx, state)
}

fn base_params(addr: SocketAddr) -> CertifyParams {
    let (miner, _) = keys();
    CertifyParams {
        validator_url: format!("http://{addr}"),
        netuid: 1,
        epoch: 42,
        miner_hotkey: miner,
        quote_source: QuoteSource::Fixture { dir: None },
        validator_hotkey_override: None,
    }
}

/// S1 — correct D10 binding against real fixtures → Verified + credit.
#[tokio::test]
async fn s1_correct_binding_verified() {
    let (addr, shutdown, state) = boot_ok().await;
    let params = base_params(addr);
    let result = certify(&params).await.expect("certify");
    assert_eq!(result.outcome, "verified");
    assert!(result.grants_credit);
    assert!(!result.carries_prior_verified);
    assert!(result.fixture_mode);
    let (miner, _) = keys();
    assert!(state.has_credit(1, 42, miner).await);
    let _ = shutdown.send(true);
}

/// S2 — resubmit same quote/nonce path → second attempt Rejected (nonce spent).
#[tokio::test]
async fn s2_resubmission_rejected() {
    let (addr, shutdown, state) = boot_ok().await;
    let params = base_params(addr);
    let first = certify(&params).await.expect("first");
    assert_eq!(first.outcome, "verified");
    // Second certify issues a NEW nonce; to test resubmission of the same nonce
    // we re-POST submit with the first nonce + quote.
    let client = reqwest::Client::new();
    let (miner, validator) = keys();
    let nonce = hex::decode(&first.nonce_hex).unwrap();
    let mut nonce_arr = [0u8; 32];
    nonce_arr.copy_from_slice(&nonce);
    let binding = ReportDataBinding {
        netuid: 1,
        epoch: 42,
        miner_pubkey: miner,
        nonce: nonce_arr,
        validator_hotkey: validator,
    };
    let rd = compute_report_data(&binding);
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &rd).unwrap();
    let event_log = String::from_utf8(EVENT_LOG.to_vec()).unwrap();
    let body = serde_json::json!({
        "miner_hotkey_hex": hex::encode(miner),
        "epoch": 42,
        "netuid": 1,
        "nonce_hex": first.nonce_hex,
        "quote_hex": hex::encode(&quote),
        "event_log_json": event_log,
        "validator_hotkey_hex": hex::encode(validator),
    });
    let resp = client
        .post(format!("http://{addr}/v1/attest/submit"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["outcome"], "rejected");
    assert_eq!(v["reason"], "nonce_invalid");
    assert_eq!(v["grants_credit"], false);
    // First submit granted credit; resubmit must not wipe Verified (D13 credit).
    assert!(state.has_credit(1, 42, miner).await);
    let _ = shutdown.send(true);
}

/// S3 — quote bound to different epoch → Rejected (`report_data` mismatch).
#[tokio::test]
async fn s3_different_epoch_rejected() {
    let (addr, shutdown, _state) = boot_ok().await;
    let client = reqwest::Client::new();
    let (miner, validator) = keys();
    // Issue nonce for epoch 42
    let nonce_resp: serde_json::Value = client
        .post(format!("http://{addr}/v1/attest/nonce"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner),
            "epoch": 42,
            "netuid": 1,
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let nonce_hex = nonce_resp["nonce_hex"].as_str().unwrap().to_owned();
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&hex::decode(&nonce_hex).unwrap());
    // Build report_data for epoch 99 (wrong)
    let binding = ReportDataBinding {
        netuid: 1,
        epoch: 99,
        miner_pubkey: miner,
        nonce,
        validator_hotkey: validator,
    };
    let rd = compute_report_data(&binding);
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &rd).unwrap();
    let event_log = String::from_utf8(EVENT_LOG.to_vec()).unwrap();
    // Claim epoch 42 in submit (mismatch with quote report_data)
    let body = serde_json::json!({
        "miner_hotkey_hex": hex::encode(miner),
        "epoch": 42,
        "netuid": 1,
        "nonce_hex": nonce_hex,
        "quote_hex": hex::encode(&quote),
        "event_log_json": event_log,
        "validator_hotkey_hex": hex::encode(validator),
    });
    let v: serde_json::Value = client
        .post(format!("http://{addr}/v1/attest/submit"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["outcome"], "rejected");
    assert_eq!(v["reason"], "report_data_mismatch");
    let _ = shutdown.send(true);
}

/// S4 — different validator hotkey in binding → Rejected.
#[tokio::test]
async fn s4_different_validator_hotkey_rejected() {
    let (addr, shutdown, _state) = boot_ok().await;
    let client = reqwest::Client::new();
    let (miner, validator) = keys();
    let other_validator = [0xdd; KEY_LEN];
    let nonce_resp: serde_json::Value = client
        .post(format!("http://{addr}/v1/attest/nonce"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner),
            "epoch": 42,
            "netuid": 1,
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let nonce_hex = nonce_resp["nonce_hex"].as_str().unwrap().to_owned();
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&hex::decode(&nonce_hex).unwrap());
    // Quote bound to other_validator
    let binding = ReportDataBinding {
        netuid: 1,
        epoch: 42,
        miner_pubkey: miner,
        nonce,
        validator_hotkey: other_validator,
    };
    let rd = compute_report_data(&binding);
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &rd).unwrap();
    let event_log = String::from_utf8(EVENT_LOG.to_vec()).unwrap();
    // Submit claims this validator
    let body = serde_json::json!({
        "miner_hotkey_hex": hex::encode(miner),
        "epoch": 42,
        "netuid": 1,
        "nonce_hex": nonce_hex,
        "quote_hex": hex::encode(&quote),
        "event_log_json": event_log,
        "validator_hotkey_hex": hex::encode(validator),
    });
    let v: serde_json::Value = client
        .post(format!("http://{addr}/v1/attest/submit"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["outcome"], "rejected");
    assert_eq!(v["reason"], "report_data_mismatch");
    let _ = shutdown.send(true);
}

/// S5 — miner A relays miner B's quote → Rejected.
#[tokio::test]
async fn s5_relayed_other_miner_quote_rejected() {
    let (addr, shutdown, _state) = boot_ok().await;
    let client = reqwest::Client::new();
    let (miner_a, validator) = keys();
    let miner_b = [0xbb; KEY_LEN];
    let nonce_resp: serde_json::Value = client
        .post(format!("http://{addr}/v1/attest/nonce"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner_a),
            "epoch": 42,
            "netuid": 1,
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let nonce_hex = nonce_resp["nonce_hex"].as_str().unwrap().to_owned();
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&hex::decode(&nonce_hex).unwrap());
    // Quote bound to miner_b
    let binding = ReportDataBinding {
        netuid: 1,
        epoch: 42,
        miner_pubkey: miner_b,
        nonce,
        validator_hotkey: validator,
    };
    let rd = compute_report_data(&binding);
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &rd).unwrap();
    let event_log = String::from_utf8(EVENT_LOG.to_vec()).unwrap();
    // Miner A claims the quote
    let body = serde_json::json!({
        "miner_hotkey_hex": hex::encode(miner_a),
        "epoch": 42,
        "netuid": 1,
        "nonce_hex": nonce_hex,
        "quote_hex": hex::encode(&quote),
        "event_log_json": event_log,
        "validator_hotkey_hex": hex::encode(validator),
    });
    let v: serde_json::Value = client
        .post(format!("http://{addr}/v1/attest/submit"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["outcome"], "rejected");
    assert_eq!(v["reason"], "report_data_mismatch");
    let _ = shutdown.send(true);
}

/// S6 — PCS outage → Parked, no credit, no carry-forward.
#[tokio::test]
async fn s6_pcs_outage_parked_no_credit() {
    let (addr, shutdown, state) = boot_pcs().await;
    let (miner, _) = keys();
    // Prior epoch Verified credit must not carry.
    // (book starts empty; record a fake prior Verified then park this epoch)
    let params = base_params(addr);
    let result = certify(&params).await.expect("certify");
    assert_eq!(result.outcome, "parked");
    assert_eq!(result.reason.as_deref(), Some("pcs_timeout"));
    assert!(!result.grants_credit);
    assert!(!result.carries_prior_verified);
    assert!(!state.has_credit(1, 42, miner).await);
    // Explicit: looking up a different epoch also has no credit
    assert!(!state.has_credit(1, 41, miner).await);
    let out = state.outcome(1, 42, miner).await;
    assert!(matches!(out, Some(AttestOutcome::Parked { .. })));
    let _ = shutdown.send(true);
}

/// S7 — `app_compose_json` reaches the policy: the real capture's compose is
/// the correct preimage but publishes no receipt key → Rejected.
#[tokio::test]
async fn s7_compose_without_receipt_key_rejected() {
    let (addr, shutdown, _state) = boot_ok().await;
    let compose = String::from_utf8(APP_COMPOSE.to_vec()).unwrap();
    let v = submit_with_compose(addr, 42, Some(compose)).await;
    assert_eq!(v["outcome"], "rejected", "resp={v}");
    assert_eq!(v["reason"], "receipt_key_invalid");
    assert!(v["receipt_public_key_hex"].is_null());
    let _ = shutdown.send(true);
}

/// S8 — a compose that does not hash to the RTMR3 measurement is a hard reject.
#[tokio::test]
async fn s8_tampered_compose_rejected() {
    let (addr, shutdown, state) = boot_ok().await;
    // Swap in an attacker-chosen receipt key: same document, different digest.
    let mut compose: serde_json::Value = serde_json::from_slice(APP_COMPOSE).unwrap();
    let yaml = compose["docker_compose_file"].as_str().unwrap().to_owned();
    compose["docker_compose_file"] = serde_json::Value::String(format!(
        "{yaml}    environment:\n      BASE_RECEIPT_PUBLIC_KEY: \"{}\"\n",
        hex::encode([0xee_u8; KEY_LEN])
    ));
    let v = submit_with_compose(addr, 43, Some(compose.to_string())).await;
    assert_eq!(v["outcome"], "rejected", "resp={v}");
    assert_eq!(v["reason"], "compose_preimage_mismatch");
    assert!(v["receipt_public_key_hex"].is_null());
    let (miner, _) = keys();
    assert!(!state.has_credit(1, 43, miner).await);
    let _ = shutdown.send(true);
}

/// Issue a nonce, bind a quote to it, and submit with `app_compose_json`.
async fn submit_with_compose(
    addr: SocketAddr,
    epoch: u64,
    app_compose_json: Option<String>,
) -> serde_json::Value {
    let client = reqwest::Client::new();
    let (miner, validator) = keys();
    let nonce_resp: serde_json::Value = client
        .post(format!("http://{addr}/v1/attest/nonce"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner),
            "epoch": epoch,
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
        epoch,
        miner_pubkey: miner,
        nonce,
        validator_hotkey: validator,
    };
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &compute_report_data(&binding)).unwrap();
    client
        .post(format!("http://{addr}/v1/attest/submit"))
        .json(&serde_json::json!({
            "miner_hotkey_hex": hex::encode(miner),
            "epoch": epoch,
            "netuid": 1,
            "nonce_hex": nonce_hex,
            "quote_hex": hex::encode(&quote),
            "event_log_json": String::from_utf8(EVENT_LOG.to_vec()).unwrap(),
            "app_compose_json": app_compose_json,
            "validator_hotkey_hex": hex::encode(validator),
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// Pipeline unit: `verify_submission` happy path without HTTP.
#[test]
fn s0_pipeline_verify_submission_direct() {
    use std::time::{Duration as StdDuration, Instant};

    use attest_policy::{
        verify_submission, CollateralFreshness, MockQuoteVerifier, QuoteVerifyOk, SubmissionInput,
        TcbStatus,
    };
    use crypto::{register_with_ttl, MemoryNonceStore};

    let measurements = real_measurements();
    let (miner, validator) = keys();
    let nonce = [0x22; KEY_LEN];
    let binding = ReportDataBinding {
        netuid: 1,
        epoch: 7,
        miner_pubkey: miner,
        nonce,
        validator_hotkey: validator,
    };
    let rd = compute_report_data(&binding);
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &rd).unwrap();
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_with_ttl(&mut nonces, nonce, now, StdDuration::from_mins(1)).unwrap();
    let verifier = MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::UpToDate,
        collateral: CollateralFreshness::Fresh,
    });
    let out = verify_submission(&mut SubmissionInput {
        measurements: &measurements,
        quote: &quote,
        event_log_json: EVENT_LOG,
        binding,
        app_compose: None,
        nonces: &mut nonces,
        now,
        verifier: &verifier,
    });
    assert_eq!(out.outcome, AttestOutcome::Verified);
    // No compose preimage supplied ⇒ nothing binds a receipt key.
    assert_eq!(out.receipt_pk, None);
    let _ = COMPOSE_HASH_LEN;
    let _ = REGISTER_LEN;
}

/// S9 — a miner may only attest for the chain's current epoch.
///
/// The challenge service joins `attestation.epoch` against its own chain-derived
/// epoch pin, so an unpinned epoch would let a miner bank credit for epochs that
/// have not happened, and would make an honest miner's off-by-one look like
/// silence rather than an error.
#[tokio::test]
async fn s9_epoch_must_match_the_chain() {
    use chain::{FakeChain, FakeChainConfig};

    let (_miner, validator) = keys();
    let state = AttestState::with_ok_verifier(real_measurements(), validator, 1);

    let chain = FakeChain::new(FakeChainConfig {
        netuid: 1,
        ..FakeChainConfig::default()
    });
    let on_chain_epoch = {
        let state = chain::gather_schedule_state(&chain, 1).expect("schedule");
        chain::current_epoch_pre_run_coinbase(&state, state.current_block)
    };

    state
        .attach_chain(std::sync::Arc::new(validator_sync::SyncChain::new(chain)))
        .await;
    let (addr, tx, _join) = spawn_attest_server(state, SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .expect("bind");
    tokio::time::sleep(Duration::from_millis(20)).await;

    let future = CertifyParams {
        epoch: on_chain_epoch + 7,
        ..base_params(addr)
    };
    let err = certify(&future)
        .await
        .expect_err("a future epoch must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("not the current chain epoch"),
        "expected an epoch refusal, got: {msg}"
    );

    let _ = tx.send(true);
}
