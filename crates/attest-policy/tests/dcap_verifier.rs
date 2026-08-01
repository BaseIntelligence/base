//! DCAP verifier mapping tables (offline) + the real `dcap-qvl` path.
//!
//! The mapping tests run in the default, feature-less, network-free CI path.
//! The real-quote test needs `--features dcap` **and** the network, so it is
//! `#[ignore]`d:
//!
//! ```text
//! cargo test -p attest-policy --features dcap -- --ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use attest_parse::parse_tdx_quote_v4;
use attest_policy::{
    classify_dcap_collateral_error, classify_dcap_verify_error, compute_report_data, evaluate,
    is_dcap_revoked_error, map_dcap_tcb_status, replay_compose_hash, AttestOutcome,
    CollateralFreshness, MockQuoteVerifier, ParkReason, PolicyInput, RejectReason,
    ReportDataBinding, TcbStatus, VerifierFailureKind,
};
use attest_replay::events_from_json;
use crypto::{register_with_ttl, MemoryNonceStore, KEY_LEN};
use trustroot::{MeasurementEntry, MeasurementsBody, COMPOSE_HASH_LEN};

const QUOTE: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/quote.bin");
const EVENT_LOG: &[u8] = include_bytes!("../../attest-parse/tests/fixtures/real/event_log.json");

// ---------------------------------------------------------------- mappings

#[test]
fn dcap_status_strings_map_onto_policy_tcb_status() {
    let expected: &[(&str, TcbStatus)] = &[
        ("UpToDate", TcbStatus::UpToDate),
        ("SWHardeningNeeded", TcbStatus::SWHardeningNeeded),
        ("ConfigurationNeeded", TcbStatus::ConfigurationNeeded),
        // No policy variant: falls back to the most conservative one (Park).
        ("ConfigurationAndSWHardeningNeeded", TcbStatus::OutOfDate),
        ("OutOfDate", TcbStatus::OutOfDate),
        (
            "OutOfDateConfigurationNeeded",
            TcbStatus::OutOfDateConfigurationNeeded,
        ),
        ("Revoked", TcbStatus::Revoked),
    ];
    for (name, want) in expected {
        assert_eq!(map_dcap_tcb_status(name), Some(*want), "status {name}");
    }
}

#[test]
fn unknown_dcap_status_never_accepts() {
    for name in ["", "uptodate", "TDRelaunchAdvised", "Whatever"] {
        assert_eq!(map_dcap_tcb_status(name), None, "status {name}");
    }
}

#[test]
fn dcap_verify_errors_classify_expiry_as_park_and_the_rest_as_reject() {
    let park = [
        "TCBInfo expired",
        "QE Identity expired",
        "Collateral expired: earliest_expiration 1 < now 2",
        "TCBInfo issue date is in the future",
    ];
    for msg in park {
        assert_eq!(
            classify_dcap_verify_error(msg),
            VerifierFailureKind::CollateralExpired,
            "msg {msg}"
        );
    }
    let reject = [
        "Signature is invalid for qe_report in quote",
        "QE report hash mismatch",
        "Certificate chain is too short in quote",
        "Unsupported DCAP quote version",
        "No matching TCB level found",
    ];
    for msg in reject {
        assert_eq!(
            classify_dcap_verify_error(msg),
            VerifierFailureKind::CryptoInvalid,
            "msg {msg}"
        );
    }
}

#[test]
fn dcap_collateral_errors_park_unless_the_quote_itself_is_bad() {
    assert_eq!(
        classify_dcap_collateral_error("Failed to get PCK certificate chain: connection refused"),
        VerifierFailureKind::PcsTimeout
    );
    assert_eq!(
        classify_dcap_collateral_error("Failed to fetch https://pcs/tcb: HTTP 503"),
        VerifierFailureKind::PcsTimeout
    );
    assert_eq!(
        classify_dcap_collateral_error("Failed to parse quote: unexpected end of input"),
        VerifierFailureKind::CryptoInvalid
    );
}

#[test]
fn revoked_is_detected_from_dcap_error_text() {
    assert!(is_dcap_revoked_error("TCB status is invalid: Revoked"));
    assert!(!is_dcap_revoked_error("TCBInfo expired"));
}

// ------------------------------------------- failure kind -> park / reject

fn real_entry() -> (MeasurementEntry, [u8; COMPOSE_HASH_LEN]) {
    let parsed = parse_tdx_quote_v4(QUOTE).expect("parse quote");
    let events = events_from_json(EVENT_LOG).expect("event log");
    let (compose_hash, _replay) = replay_compose_hash(&events).expect("replay");
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
        epoch: 7,
        miner_pubkey: [0x11; KEY_LEN],
        nonce: [0x22; KEY_LEN],
        validator_hotkey: [0x33; KEY_LEN],
    }
}

fn outcome_for(kind: VerifierFailureKind) -> AttestOutcome {
    let (entry, compose) = real_entry();
    let body = MeasurementsBody {
        entries: vec![entry],
    };
    let mut td = parse_tdx_quote_v4(QUOTE).expect("td").td_report;
    let b = binding();
    td.report_data = compute_report_data(&b);

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_with_ttl(&mut nonces, b.nonce, now, Duration::from_hours(1)).unwrap();
    let verifier = MockQuoteVerifier::Err(kind);
    evaluate(&mut PolicyInput {
        measurements: &body,
        td_report: &td,
        compose_hash: &compose,
        binding: b,
        quote: QUOTE,
        nonces: &mut nonces,
        now,
        verifier: &verifier,
    })
}

#[test]
fn verifier_failure_kinds_map_to_park_or_reject() {
    assert_eq!(
        outcome_for(VerifierFailureKind::CryptoInvalid),
        AttestOutcome::Rejected {
            reason: RejectReason::QuoteCryptoInvalid
        }
    );
    assert_eq!(
        outcome_for(VerifierFailureKind::PcsTimeout),
        AttestOutcome::Parked {
            reason: ParkReason::PcsTimeout
        }
    );
    assert_eq!(
        outcome_for(VerifierFailureKind::CollateralExpired),
        AttestOutcome::Parked {
            reason: ParkReason::CollateralExpired
        }
    );
    assert_eq!(
        outcome_for(VerifierFailureKind::Unavailable),
        AttestOutcome::Parked {
            reason: ParkReason::VerifierUnavailable
        }
    );
    // A DCAP `Revoked` verdict is reported as a status, not a failure kind.
    assert_eq!(
        attest_policy::classify_tcb(TcbStatus::Revoked, CollateralFreshness::Fresh),
        attest_policy::TcbAction::Reject
    );
}

// ------------------------------------------------- real dcap-qvl (network)

#[cfg(feature = "dcap")]
mod real {
    use attest_policy::{
        CollateralFreshness, DcapQuoteVerifier, QuoteVerifier, TcbStatus, VerifierFailureKind,
        DEFAULT_COLLATERAL_MAX_AGE, DEFAULT_PCS_URL,
    };

    /// Run the REAL Intel DCAP verifier against the real TDX quote fixture.
    ///
    /// Requires outbound HTTPS to `BASE_PCCS_URL` (default Intel PCS), hence
    /// `#[ignore]`. Observed against Intel PCS
    /// (`https://api.trustedservices.intel.com`):
    /// `Ok(QuoteVerifyOk { tcb_status: UpToDate, collateral: Fresh })` — the
    /// PCK chain, QE report and ISV report signatures all verify and the
    /// fixture platform's TCB level is current.
    ///
    /// The point of this test is that real cryptography runs. Any of
    /// `UpToDate` / a soft status / `OutOfDate` / `CollateralExpired` is an
    /// acceptable observation for an aging fixture; a `CryptoInvalid` would
    /// mean the trust chain genuinely failed and is treated as a failure here.
    #[test]
    #[ignore = "needs network access to Intel PCS / PCCS"]
    fn real_quote_runs_real_dcap_crypto() {
        let quote = super::QUOTE;
        let url = std::env::var("BASE_PCCS_URL").unwrap_or_else(|_| DEFAULT_PCS_URL.to_owned());
        let verifier = DcapQuoteVerifier::new(&url, DEFAULT_COLLATERAL_MAX_AGE)
            .expect("build DcapQuoteVerifier");

        let first = verifier.verify(quote);
        println!("dcap verify (pccs={url}) => {first:?}");

        match &first {
            Ok(ok) => {
                assert_ne!(
                    ok.tcb_status,
                    TcbStatus::Revoked,
                    "fixture platform must not be revoked"
                );
                // Collateral we just fetched is by construction younger than
                // the configured max age.
                assert_eq!(ok.collateral, CollateralFreshness::Fresh);
                assert!(
                    matches!(
                        ok.tcb_status,
                        TcbStatus::UpToDate
                            | TcbStatus::SWHardeningNeeded
                            | TcbStatus::ConfigurationNeeded
                            | TcbStatus::OutOfDate
                            | TcbStatus::OutOfDateConfigurationNeeded
                    ),
                    "unexpected tcb status {:?}",
                    ok.tcb_status
                );
            }
            Err(e) => {
                assert_ne!(
                    e.kind,
                    VerifierFailureKind::CryptoInvalid,
                    "real quote must not fail trust-chain verification"
                );
                assert!(
                    matches!(
                        e.kind,
                        VerifierFailureKind::PcsTimeout
                            | VerifierFailureKind::CollateralExpired
                            | VerifierFailureKind::Unavailable
                    ),
                    "unexpected failure {:?}",
                    e.kind
                );
            }
        }

        // Second call must be served from the per-platform collateral cache
        // (same verdict, no second PCS round-trip).
        let second = verifier.verify(quote);
        println!("dcap verify (cached) => {second:?}");
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
    }

    /// A corrupted quote must be a genuine cryptographic rejection.
    #[test]
    #[ignore = "needs network access to Intel PCS / PCCS"]
    fn tampered_quote_is_crypto_rejected() {
        let mut quote = super::QUOTE.to_vec();
        // Flip a bit inside the TD report body (after the 48-byte header).
        let idx = 100;
        quote[idx] ^= 0xff;
        let url = std::env::var("BASE_PCCS_URL").unwrap_or_else(|_| DEFAULT_PCS_URL.to_owned());
        let verifier = DcapQuoteVerifier::new(&url, DEFAULT_COLLATERAL_MAX_AGE)
            .expect("build DcapQuoteVerifier");
        let out = verifier.verify(&quote);
        println!("dcap verify (tampered) => {out:?}");
        let err = out.expect_err("tampered quote must not verify");
        assert_eq!(err.kind, VerifierFailureKind::CryptoInvalid);
    }
}
