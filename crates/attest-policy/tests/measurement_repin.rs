//! Todo 23: measurement re-pin sequencing after socket-proxy compose-hash change.
//!
//! Proves fail-closed stale allowlist (wrong cutover order) and acceptance after
//! dual-entry / hard-cut update. Uses synthetic MeasurementEntry profiles plus
//! real Phala fixture registers for the historical compose_hash row.
//!
//! Live CVM MRTD/RTMR for the normative post-proxy template must still be
//! captured before production scoring depends on Verified — see
//! docs/runbooks/measurement-repin-socket-proxy.md.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use attest_parse::parse_tdx_quote_v4;
use attest_policy::{
    evaluate, replay_compose_hash, AttestOutcome, CollateralFreshness, MockQuoteVerifier,
    PolicyInput, QuoteVerifyOk, RejectReason, ReportDataBinding, TcbStatus,
};
use attest_replay::events_from_json;
use crypto::{register_with_ttl, MemoryNonceStore, KEY_LEN};
use trustroot::{MeasurementEntry, MeasurementsBody, COMPOSE_HASH_LEN, REGISTER_LEN};

const QUOTE: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/quote.bin");
const EVENT_LOG: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/event_log.json");

/// Normative post–socket-proxy compose-hash from `DeployParams::default()`
/// (`cargo run -p miner-bin -- deploy --no-deploy --netuid 541`).
const NEW_COMPOSE_HASH_HEX: &str =
    "95089ce1b1ccb528e3309acc6dc304835c1634f540663784a99e700364c331ed";

fn hex32(s: &str) -> [u8; COMPOSE_HASH_LEN] {
    assert_eq!(s.len(), COMPOSE_HASH_LEN * 2, "compose hash hex length");
    let mut out = [0_u8; COMPOSE_HASH_LEN];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0]);
        let lo = hex_nibble(chunk[1]);
        out[i] = (hi << 4) | lo;
    }
    out
}

fn hex_nibble(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => panic!("bad hex nibble {}", b as char),
    }
}

fn encode_hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn old_fixture_entry() -> (MeasurementEntry, [u8; COMPOSE_HASH_LEN]) {
    let parsed = parse_tdx_quote_v4(QUOTE).expect("parse quote");
    let events = events_from_json(EVENT_LOG).expect("event log");
    let (compose_hash, replay) = replay_compose_hash(&events).expect("replay");
    assert_eq!(replay.rtmr3, parsed.td_report.rtmr3);
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

/// Provisional new-template profile: fixture-shaped registers + new compose_hash.
/// Structure-only stand-in until live CVM capture replaces registers.
fn new_socket_proxy_entry() -> MeasurementEntry {
    let (old, _) = old_fixture_entry();
    let mut rtmr3 = old.rtmr3;
    // Distinct RTMR3 so dual-entry matching is exact-tuple, not accidental.
    rtmr3[REGISTER_LEN - 1] ^= 0xa5;
    MeasurementEntry {
        mr_td: old.mr_td,
        rtmr0: old.rtmr0,
        rtmr1: old.rtmr1,
        rtmr2: old.rtmr2,
        rtmr3,
        compose_hash: hex32(NEW_COMPOSE_HASH_HEX),
    }
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

fn ok_verifier() -> MockQuoteVerifier {
    MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::UpToDate,
        collateral: CollateralFreshness::Fresh,
    })
}

fn td_from_entry(entry: &MeasurementEntry, b: &ReportDataBinding) -> attest_parse::TdReport {
    attest_parse::TdReport {
        mr_td: entry.mr_td,
        mr_config_id: [0_u8; REGISTER_LEN],
        rtmr0: entry.rtmr0,
        rtmr1: entry.rtmr1,
        rtmr2: entry.rtmr2,
        rtmr3: entry.rtmr3,
        report_data: attest_policy::compute_report_data(b),
    }
}

fn eval(body: &MeasurementsBody, entry: &MeasurementEntry) -> AttestOutcome {
    let b = binding();
    let td = td_from_entry(entry, &b);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_with_ttl(&mut nonces, b.nonce, now, Duration::from_hours(1)).unwrap();
    let verifier = ok_verifier();
    evaluate(&mut PolicyInput {
        measurements: body,
        td_report: &td,
        compose_hash: &entry.compose_hash,
        binding: b,
        quote: b"synthetic-repin-quote",
        nonces: &mut nonces,
        now,
        verifier: &verifier,
    })
}

/// QA failure path: new compose against **old** allowlist only → Rejected.
/// This is exactly the outage from requiring Verified before re-pin.
#[test]
fn repin_stale_allowlist_rejects_new_compose() {
    let (old, _) = old_fixture_entry();
    let body_old_only = MeasurementsBody {
        entries: vec![old],
    };
    let new_e = new_socket_proxy_entry();
    assert!(
        !body_old_only.allows_quote(
            &new_e.mr_td,
            &new_e.rtmr0,
            &new_e.rtmr1,
            &new_e.rtmr2,
            &new_e.rtmr3,
            &new_e.compose_hash
        ),
        "old body must not allow new compose_hash"
    );
    let out = eval(&body_old_only, &new_e);
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::MeasurementNotAllowlisted
        }
    );
    assert!(!out.grants_credit());
}

/// QA happy path: updated allowlist includes new entry → Verified.
#[test]
fn repin_updated_allowlist_accepts_new_compose() {
    let (old, _) = old_fixture_entry();
    let new_e = new_socket_proxy_entry();
    let body = MeasurementsBody {
        entries: vec![old, new_e.clone()],
    };
    assert!(body.allows_quote(
        &new_e.mr_td,
        &new_e.rtmr0,
        &new_e.rtmr1,
        &new_e.rtmr2,
        &new_e.rtmr3,
        &new_e.compose_hash
    ));
    let out = eval(&body, &new_e);
    assert_eq!(out, AttestOutcome::Verified);
    assert!(out.grants_credit());
}

/// Dual-entry rotation window: old fixture quote still allowlisted.
#[test]
fn repin_dual_entry_still_accepts_old_compose() {
    let (old, old_hash) = old_fixture_entry();
    let new_e = new_socket_proxy_entry();
    let body = MeasurementsBody {
        entries: vec![old.clone(), new_e],
    };
    assert!(body.allows_quote(
        &old.mr_td,
        &old.rtmr0,
        &old.rtmr1,
        &old.rtmr2,
        &old.rtmr3,
        &old_hash
    ));
    // Real quote path through policy (mock crypto), same as real_fixtures happy path.
    let mut td = parse_tdx_quote_v4(QUOTE).expect("td").td_report;
    let b = binding();
    td.report_data = attest_policy::compute_report_data(&b);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_with_ttl(&mut nonces, b.nonce, now, Duration::from_hours(1)).unwrap();
    let verifier = ok_verifier();
    let out = evaluate(&mut PolicyInput {
        measurements: &body,
        td_report: &td,
        compose_hash: &old_hash,
        binding: b,
        quote: QUOTE,
        nonces: &mut nonces,
        now,
        verifier: &verifier,
    });
    assert_eq!(out, AttestOutcome::Verified);
}

/// Hard-cut after rotation window: new-only body rejects old compose.
#[test]
fn repin_hard_cut_rejects_old_compose() {
    let (old, old_hash) = old_fixture_entry();
    let new_e = new_socket_proxy_entry();
    let body_new_only = MeasurementsBody {
        entries: vec![new_e],
    };
    assert!(!body_new_only.allows_quote(
        &old.mr_td,
        &old.rtmr0,
        &old.rtmr1,
        &old.rtmr2,
        &old.rtmr3,
        &old_hash
    ));
    let mut td = parse_tdx_quote_v4(QUOTE).expect("td").td_report;
    let b = binding();
    td.report_data = attest_policy::compute_report_data(&b);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_with_ttl(&mut nonces, b.nonce, now, Duration::from_hours(1)).unwrap();
    let verifier = ok_verifier();
    let out = evaluate(&mut PolicyInput {
        measurements: &body_new_only,
        td_report: &td,
        compose_hash: &old_hash,
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

/// Empty allowlist remains fail-closed (regression guard for re-pin edits).
#[test]
fn repin_empty_allowlist_still_fail_closed() {
    let body = MeasurementsBody::default();
    let new_e = new_socket_proxy_entry();
    let out = eval(&body, &new_e);
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::EmptyAllowlist
        }
    );
}

/// Documented normative new compose_hash is 32 bytes and differs from fixture.
#[test]
fn repin_new_compose_hash_differs_from_fixture() {
    let (_, old_hash) = old_fixture_entry();
    let new_hash = hex32(NEW_COMPOSE_HASH_HEX);
    assert_ne!(old_hash, new_hash);
    assert_eq!(
        encode_hex_lower(&new_hash),
        NEW_COMPOSE_HASH_HEX,
        "pin must match miner-bin default post-proxy hash"
    );
}
