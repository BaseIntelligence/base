//! Task-35: real Phala fixtures through policy allowlist (mock quote crypto).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use gbase_attest_parse::parse_tdx_quote_v4;
use gbase_attest_policy::{
    evaluate, mr_config_id_matches, replay_compose_hash, AttestOutcome, CollateralFreshness,
    MockQuoteVerifier, PolicyInput, QuoteVerifyOk, RejectReason, ReportDataBinding, TcbStatus,
};
use gbase_attest_replay::events_from_json;
use gbase_compose_hash::{compose_hash as hash_app_compose, mr_config_id};
use gbase_crypto::{register_with_ttl, MemoryNonceStore, KEY_LEN};
use gbase_trustroot::{MeasurementEntry, MeasurementsBody, COMPOSE_HASH_LEN, REGISTER_LEN};

const QUOTE: &[u8] = include_bytes!("../../gbase-attest-parse/tests/fixtures/real/quote.bin");
const EVENT_LOG: &[u8] =
    include_bytes!("../../gbase-attest-parse/tests/fixtures/real/event_log.json");
const APP_COMPOSE: &[u8] =
    include_bytes!("../../gbase-attest-parse/tests/fixtures/real/app-compose.json");

fn real_entry() -> (MeasurementEntry, [u8; COMPOSE_HASH_LEN]) {
    let parsed = parse_tdx_quote_v4(QUOTE).expect("parse quote");
    let events = events_from_json(EVENT_LOG).expect("event log");
    let (compose_hash, replay) = replay_compose_hash(&events).expect("replay");
    assert_eq!(replay.rtmr3, parsed.td_report.rtmr3);
    assert!(mr_config_id_matches(&parsed.td_report, &compose_hash));
    let from_file = hash_app_compose(APP_COMPOSE).expect("compose file hash");
    assert_eq!(from_file, compose_hash);
    assert_eq!(mr_config_id(&compose_hash), parsed.td_report.mr_config_id);

    let entry = MeasurementEntry {
        mr_td: parsed.td_report.mr_td,
        rtmr0: parsed.td_report.rtmr0,
        rtmr1: parsed.td_report.rtmr1,
        rtmr2: parsed.td_report.rtmr2,
        rtmr3: parsed.td_report.rtmr3,
        compose_hash,
    };
    (entry, compose_hash)
}

fn binding() -> ReportDataBinding {
    ReportDataBinding {
        netuid: 1,
        epoch: 42,
        miner_pubkey: [0xaa; KEY_LEN],
        nonce: [0xbb; KEY_LEN],
        validator_hotkey: [0xcc; KEY_LEN],
    }
}

#[test]
fn real_quote_allowlisted_passes_policy() {
    let (entry, compose) = real_entry();
    let body = MeasurementsBody {
        entries: vec![entry.clone()],
    };
    let mut td = parse_tdx_quote_v4(QUOTE).expect("td").td_report;
    let b = binding();
    td.report_data = gbase_attest_policy::compute_report_data(&b);

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_with_ttl(&mut nonces, b.nonce, now, Duration::from_hours(1)).unwrap();
    let verifier = MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::UpToDate,
        collateral: CollateralFreshness::Fresh,
    });

    let out = evaluate(&mut PolicyInput {
        measurements: &body,
        td_report: &td,
        compose_hash: &compose,
        binding: b,
        quote: QUOTE,
        nonces: &mut nonces,
        now,
        verifier: &verifier,
    });
    assert_eq!(out, AttestOutcome::Verified);
    assert!(out.grants_credit());
}

#[test]
fn real_quote_mutated_measurement_rejected() {
    let (mut entry, compose) = real_entry();
    // Flip one byte of mr_td — must fail allowlist.
    entry.mr_td[0] ^= 0xff;
    let body = MeasurementsBody {
        entries: vec![entry],
    };
    let mut td = parse_tdx_quote_v4(QUOTE).expect("td").td_report;
    let b = binding();
    td.report_data = gbase_attest_policy::compute_report_data(&b);

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_with_ttl(&mut nonces, b.nonce, now, Duration::from_hours(1)).unwrap();
    let verifier = MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::UpToDate,
        collateral: CollateralFreshness::Fresh,
    });

    let out = evaluate(&mut PolicyInput {
        measurements: &body,
        td_report: &td,
        compose_hash: &compose,
        binding: b,
        quote: QUOTE,
        nonces: &mut nonces,
        now,
        verifier: &verifier,
    });
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::MeasurementNotAllowlisted
        }
    );
}

#[test]
fn real_measurements_body_allows_quote_exact() {
    let (entry, compose) = real_entry();
    let body = MeasurementsBody {
        entries: vec![entry.clone()],
    };
    assert!(body.allows_quote(
        &entry.mr_td,
        &entry.rtmr0,
        &entry.rtmr1,
        &entry.rtmr2,
        &entry.rtmr3,
        &compose
    ));
    let mut bad = entry.rtmr3;
    bad[0] ^= 1;
    assert!(!body.allows_quote(
        &entry.mr_td,
        &entry.rtmr0,
        &entry.rtmr1,
        &entry.rtmr2,
        &bad,
        &compose
    ));
    // silence unused REGISTER_LEN if needed
    let _: usize = REGISTER_LEN;
}
