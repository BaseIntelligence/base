//! Binding the receipt key to the measurement: the compose preimage a miner
//! ships must hash to the RTMR3 compose hash before anything is read from it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use attest_parse::{parse_tdx_quote_v4, patch_report_data};
use attest_policy::{
    compute_report_data, verify_submission, AttestOutcome, CollateralFreshness, MockQuoteVerifier,
    QuoteVerifyOk, RejectReason, ReportDataBinding, SubmissionInput, SubmissionVerdict, TcbStatus,
    RECEIPT_PUBLIC_KEY_ENV,
};
use compose_hash::{compose_hash, ComposeHash};
use crypto::{register_with_ttl, MemoryNonceStore, KEY_LEN};
use trustroot::{MeasurementEntry, MeasurementsBody};

const QUOTE: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/quote.bin");
const EVENT_LOG: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/event_log.json");
const REAL_APP_COMPOSE: &[u8] =
    include_bytes!("../../attest-parse/tests/fixtures/real/app-compose.json");

const MINER_PK: [u8; KEY_LEN] = [0x5a; KEY_LEN];
const ATTACKER_PK: [u8; KEY_LEN] = [0xee; KEY_LEN];

/// A miner `app-compose.json` publishing `receipt_pk` the way `miner deploy` does.
fn miner_compose(receipt_pk: [u8; KEY_LEN]) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "manifest_version": 2,
        "name": "base-miner",
        "runner": "docker-compose",
        "docker_compose_file": format!(
            "services:\n  agent:\n    image: agent@sha256:aa\n    environment:\n      BASE_NETUID: \"1\"\n      {RECEIPT_PUBLIC_KEY_ENV}: \"{}\"\n",
            hex::encode(receipt_pk)
        ),
    }))
    .expect("compose json")
}

/// Real event log with the `compose-hash` payload repointed at `hash`.
///
/// The logged per-event digest is dropped so replay recomputes it; the RTMR3
/// value that comes out is not compared against the quote by the policy, only
/// the allowlist entry is, and these tests build that entry themselves.
fn event_log_measuring(hash: &ComposeHash) -> Vec<u8> {
    let mut events: serde_json::Value = serde_json::from_slice(EVENT_LOG).expect("event log");
    for e in events.as_array_mut().expect("array") {
        if e.get("event").and_then(serde_json::Value::as_str) == Some("compose-hash") {
            e["event_payload"] = serde_json::Value::String(hex::encode(hash));
            e["digest"] = serde_json::Value::String(String::new());
        }
    }
    serde_json::to_vec(&events).expect("event log json")
}

fn allowlist(compose: &ComposeHash) -> MeasurementsBody {
    let td = parse_tdx_quote_v4(QUOTE).expect("parse").td_report;
    MeasurementsBody {
        entries: vec![MeasurementEntry {
            mr_td: td.mr_td,
            rtmr0: td.rtmr0,
            rtmr1: td.rtmr1,
            rtmr2: td.rtmr2,
            rtmr3: td.rtmr3,
            compose_hash: *compose,
        }],
    }
}

/// Submit `app_compose` against a measurement that pins `measured`.
fn submit(
    measured: &ComposeHash,
    event_log: &[u8],
    app_compose: Option<&[u8]>,
) -> SubmissionVerdict {
    let binding = ReportDataBinding {
        netuid: 1,
        epoch: 42,
        miner_pubkey: [0xaa; KEY_LEN],
        nonce: [0xbb; KEY_LEN],
        validator_hotkey: [0xcc; KEY_LEN],
    };
    let mut quote = QUOTE.to_vec();
    patch_report_data(&mut quote, &compute_report_data(&binding)).expect("patch");

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_with_ttl(&mut nonces, binding.nonce, now, Duration::from_hours(1)).unwrap();

    verify_submission(&mut SubmissionInput {
        measurements: &allowlist(measured),
        quote: &quote,
        event_log_json: event_log,
        binding,
        app_compose,
        nonces: &mut nonces,
        now,
        verifier: &MockQuoteVerifier::Ok(QuoteVerifyOk {
            tcb_status: TcbStatus::UpToDate,
            collateral: CollateralFreshness::Fresh,
        }),
    })
}

#[test]
fn measured_compose_binds_the_receipt_key() {
    let compose = miner_compose(MINER_PK);
    let measured = compose_hash(&compose).expect("hash");
    let verdict = submit(&measured, &event_log_measuring(&measured), Some(&compose));
    assert_eq!(verdict.outcome, AttestOutcome::Verified);
    assert_eq!(verdict.receipt_pk, Some(MINER_PK));
}

/// The whole point: a compose the miner swapped after boot never binds a key.
#[test]
fn tampered_compose_is_rejected_not_verified_with_the_attacker_key() {
    let honest = miner_compose(MINER_PK);
    let measured = compose_hash(&honest).expect("hash");
    let event_log = event_log_measuring(&measured);

    let tampered = miner_compose(ATTACKER_PK);
    assert_ne!(honest, tampered);

    let verdict = submit(&measured, &event_log, Some(&tampered));
    assert_eq!(
        verdict.outcome,
        AttestOutcome::Rejected {
            reason: RejectReason::ComposePreimageMismatch
        }
    );
    assert_eq!(verdict.receipt_pk, None);
    assert!(!verdict.outcome.grants_credit());
}

#[test]
fn measured_compose_without_a_receipt_key_is_rejected() {
    // The real task-34 capture CVM predates BASE_RECEIPT_PUBLIC_KEY: correct
    // preimage, nothing to bind.
    let measured = compose_hash(REAL_APP_COMPOSE).expect("hash");
    let verdict = submit(&measured, EVENT_LOG, Some(REAL_APP_COMPOSE));
    assert_eq!(
        verdict.outcome,
        AttestOutcome::Rejected {
            reason: RejectReason::ReceiptKeyInvalid
        }
    );
    assert_eq!(verdict.receipt_pk, None);
}

#[test]
fn real_fixture_compose_matches_the_real_event_log() {
    let measured = compose_hash(REAL_APP_COMPOSE).expect("hash");
    let mut poisoned = REAL_APP_COMPOSE.to_vec();
    // Same document plus an attacker-chosen key ⇒ different digest.
    let injected = format!(
        ",\"{RECEIPT_PUBLIC_KEY_ENV}\":\"{}\"}}",
        hex::encode(ATTACKER_PK)
    );
    let last = poisoned.len() - 1;
    poisoned.splice(last.., injected.into_bytes());

    let verdict = submit(&measured, EVENT_LOG, Some(&poisoned));
    assert_eq!(
        verdict.outcome,
        AttestOutcome::Rejected {
            reason: RejectReason::ComposePreimageMismatch
        }
    );
}

#[test]
fn no_compose_supplied_verifies_without_a_key() {
    let measured = compose_hash(REAL_APP_COMPOSE).expect("hash");
    let verdict = submit(&measured, EVENT_LOG, None);
    assert_eq!(verdict.outcome, AttestOutcome::Verified);
    assert_eq!(verdict.receipt_pk, None);
}
