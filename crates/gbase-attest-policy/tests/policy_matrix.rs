#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Integration-style unit tests for gbase-attest-policy (task 23 VERIFY matrix).

use std::time::{Duration, Instant};

use gbase_attest_parse::TdReport;
use gbase_crypto::{register_with_ttl, MemoryNonceStore, KEY_LEN};
use gbase_trustroot::{MeasurementEntry, MeasurementsBody, COMPOSE_HASH_LEN, REGISTER_LEN};

use gbase_attest_policy::{
    classify_tcb, compute_report_data, evaluate, AttestCreditBook, AttestOutcome,
    CollateralFreshness, CreditKey, MockQuoteVerifier, ParkReason, PolicyInput, QuoteVerifyOk,
    RejectReason, ReportDataBinding, TcbAction, TcbStatus, VerifierFailureKind, REPORT_DATA_LEN,
};

fn zeros_reg() -> [u8; REGISTER_LEN] {
    [0_u8; REGISTER_LEN]
}

fn entry_a() -> MeasurementEntry {
    MeasurementEntry {
        mr_td: [1_u8; REGISTER_LEN],
        rtmr0: [2_u8; REGISTER_LEN],
        rtmr1: [3_u8; REGISTER_LEN],
        rtmr2: [4_u8; REGISTER_LEN],
        rtmr3: [5_u8; REGISTER_LEN],
        compose_hash: [6_u8; COMPOSE_HASH_LEN],
    }
}

fn body_with(entry: MeasurementEntry) -> MeasurementsBody {
    MeasurementsBody {
        entries: vec![entry],
    }
}

fn binding_base() -> ReportDataBinding {
    ReportDataBinding {
        netuid: 1,
        epoch: 100,
        miner_pubkey: [0x11; KEY_LEN],
        nonce: [0x22; KEY_LEN],
        validator_hotkey: [0x33; KEY_LEN],
    }
}

fn td_for(binding: &ReportDataBinding, entry: &MeasurementEntry) -> TdReport {
    TdReport {
        mr_td: entry.mr_td,
        mr_config_id: zeros_reg(),
        rtmr0: entry.rtmr0,
        rtmr1: entry.rtmr1,
        rtmr2: entry.rtmr2,
        rtmr3: entry.rtmr3,
        report_data: compute_report_data(binding),
    }
}

fn ok_verifier() -> MockQuoteVerifier {
    MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::UpToDate,
        collateral: CollateralFreshness::Fresh,
    })
}

fn register_nonce(store: &mut MemoryNonceStore, nonce: [u8; KEY_LEN], now: Instant) {
    register_with_ttl(store, nonce, now, Duration::from_hours(1)).expect("register nonce");
}

fn run(
    measurements: &MeasurementsBody,
    td: &TdReport,
    compose: &[u8; COMPOSE_HASH_LEN],
    binding: ReportDataBinding,
    nonces: &mut MemoryNonceStore,
    now: Instant,
    verifier: &MockQuoteVerifier,
) -> AttestOutcome {
    evaluate(&mut PolicyInput {
        measurements,
        td_report: td,
        compose_hash: compose,
        binding,
        quote: b"synthetic-quote",
        nonces,
        now,
        verifier,
    })
}

/// Happy path → Verified + credit.
#[test]
fn s0_happy_verified_grants_credit() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = ok_verifier();

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(out, AttestOutcome::Verified);
    assert!(out.grants_credit());
    assert!(!out.carries_prior_verified());

    let mut book = AttestCreditBook::new();
    let key = CreditKey {
        netuid: binding.netuid,
        epoch: binding.epoch,
        miner: binding.miner_pubkey,
    };
    book.record(key, out);
    assert!(book.has_credit(&key));
}

/// Missing/empty allowlist → fail-closed Reject.
#[test]
fn s1_empty_allowlist_fail_closed() {
    let body = MeasurementsBody::default();
    assert!(body.entries.is_empty());
    let binding = binding_base();
    let entry = entry_a();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = ok_verifier();

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::EmptyAllowlist
        }
    );
    assert!(!out.grants_credit());
}

/// Wrong nonce in `report_data` (binding nonce ≠ registered / hash) → Reject.
#[test]
fn s2_wrong_nonce_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let mut binding = binding_base();
    let good_nonce = binding.nonce;
    // Quote was built with a different nonce than we claim at redeem time:
    // build report_data with wrong_nonce, then try to redeem good_nonce.
    let wrong = [0xEE; KEY_LEN];
    binding.nonce = wrong;
    let td = td_for(&binding, &entry);
    // Claim the good nonce at policy time (mismatch vs report_data).
    binding.nonce = good_nonce;

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, good_nonce, now);
    let v = ok_verifier();

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::ReportDataMismatch
        }
    );
}

/// Replayed nonce → Reject.
#[test]
fn s3_replayed_nonce_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = ok_verifier();

    let first = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(first, AttestOutcome::Verified);

    // Second evaluation: same nonce already redeemed.
    let second = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        second,
        AttestOutcome::Rejected {
            reason: RejectReason::NonceInvalid
        }
    );
}

/// Quote bound to a different epoch → Reject.
#[test]
fn s4_different_epoch_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let mut binding = binding_base();
    let td = td_for(&binding, &entry);
    binding.epoch += 1; // claim different epoch

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = ok_verifier();

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::ReportDataMismatch
        }
    );
}

/// Quote bound to a different validator hotkey → Reject.
#[test]
fn s5_different_validator_hotkey_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let mut binding = binding_base();
    let td = td_for(&binding, &entry);
    binding.validator_hotkey = [0xFF; KEY_LEN];

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = ok_verifier();

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::ReportDataMismatch
        }
    );
}

/// Miner A relaying miner B's quote (pubkey mismatch) → Reject.
#[test]
fn s6_miner_a_relays_b_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let mut binding = binding_base();
    let miner_b = binding.miner_pubkey;
    let td = td_for(&binding, &entry);
    // Claim miner A while quote binds miner B.
    binding.miner_pubkey = [0xAA; KEY_LEN];
    assert_ne!(binding.miner_pubkey, miner_b);

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = ok_verifier();

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::ReportDataMismatch
        }
    );
}

/// Simulated PCS timeout → Park.
#[test]
fn s7_pcs_timeout_park() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = MockQuoteVerifier::Err(VerifierFailureKind::PcsTimeout);

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Parked {
            reason: ParkReason::PcsTimeout
        }
    );
    assert!(!out.grants_credit());
}

/// Expired collateral → Park.
#[test]
fn s8_expired_collateral_park() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::UpToDate,
        collateral: CollateralFreshness::Expired,
    });

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Parked {
            reason: ParkReason::CollateralExpired
        }
    );
    assert!(!out.grants_credit());
}

/// Revoked TCB → Reject.
#[test]
fn s9_revoked_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::Revoked,
        collateral: CollateralFreshness::Fresh,
    });

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::TcbRevoked
        }
    );
}

/// `OutOfDate` TCB → Park.
#[test]
fn s10_out_of_date_park() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::OutOfDate,
        collateral: CollateralFreshness::Fresh,
    });

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Parked {
            reason: ParkReason::TcbOutOfDate
        }
    );
}

/// Parked miner: no credit and does not inherit last epoch's Verified (D13).
#[test]
fn s11_parked_no_credit_no_carry_forward() {
    let mut book = AttestCreditBook::new();
    let miner = [0x42; KEY_LEN];
    let prev = CreditKey {
        netuid: 1,
        epoch: 10,
        miner,
    };
    let curr = CreditKey {
        netuid: 1,
        epoch: 11,
        miner,
    };

    book.record(prev, AttestOutcome::Verified);
    assert!(book.has_credit(&prev));

    // This epoch: Parked — must not grant credit and must not use epoch 10.
    book.record(
        curr,
        AttestOutcome::Parked {
            reason: ParkReason::TcbOutOfDate,
        },
    );
    assert!(
        !book.has_credit(&curr),
        "parked epoch must not grant credit"
    );
    assert!(
        book.has_credit(&prev),
        "prior epoch record remains but is a different key"
    );
    // Explicit: credit check for current epoch does not fall back to prev.
    assert!(!book.has_credit(&curr));
    assert!(!AttestOutcome::Parked {
        reason: ParkReason::PcsTimeout
    }
    .carries_prior_verified());
    assert!(!AttestOutcome::Verified.carries_prior_verified());
}

/// TCB table unit mapping (D13).
#[test]
fn s12_tcb_table_mapping() {
    assert_eq!(
        classify_tcb(TcbStatus::UpToDate, CollateralFreshness::Fresh),
        TcbAction::Accept { warn: false }
    );
    assert_eq!(
        classify_tcb(TcbStatus::SWHardeningNeeded, CollateralFreshness::Fresh),
        TcbAction::Accept { warn: true }
    );
    assert_eq!(
        classify_tcb(TcbStatus::ConfigurationNeeded, CollateralFreshness::Fresh),
        TcbAction::Accept { warn: true }
    );
    assert_eq!(
        classify_tcb(TcbStatus::OutOfDate, CollateralFreshness::Fresh),
        TcbAction::Park
    );
    assert_eq!(
        classify_tcb(
            TcbStatus::OutOfDateConfigurationNeeded,
            CollateralFreshness::Fresh
        ),
        TcbAction::Park
    );
    assert_eq!(
        classify_tcb(TcbStatus::Revoked, CollateralFreshness::Fresh),
        TcbAction::Reject
    );
    // Expired collateral parks even if UpToDate.
    assert_eq!(
        classify_tcb(TcbStatus::UpToDate, CollateralFreshness::Expired),
        TcbAction::Park
    );
}

/// D10 preimage is sensitive to each field; length is 64.
#[test]
fn s13_report_data_field_sensitivity() {
    let base = binding_base();
    let h0 = compute_report_data(&base);
    assert_eq!(h0.len(), REPORT_DATA_LEN);

    let mut b = base;
    b.netuid = 2;
    assert_ne!(compute_report_data(&b), h0);

    let mut b = base;
    b.epoch = 999;
    assert_ne!(compute_report_data(&b), h0);

    let mut b = base;
    b.miner_pubkey[0] ^= 1;
    assert_ne!(compute_report_data(&b), h0);

    let mut b = base;
    b.nonce[0] ^= 1;
    assert_ne!(compute_report_data(&b), h0);

    let mut b = base;
    b.validator_hotkey[0] ^= 1;
    assert_ne!(compute_report_data(&b), h0);
}

/// Measurement not on allowlist → Reject (after binding + nonce ok).
#[test]
fn s14_measurement_not_allowlisted_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let mut td = td_for(&binding, &entry);
    td.mr_td = [0xAB; REGISTER_LEN]; // tamper

    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = ok_verifier();

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::MeasurementNotAllowlisted
        }
    );
}

/// Crypto invalid quote → Reject.
#[test]
fn s15_quote_crypto_invalid_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = MockQuoteVerifier::Err(VerifierFailureKind::CryptoInvalid);

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::QuoteCryptoInvalid
        }
    );
}

/// Unknown nonce (never issued) → Reject.
#[test]
fn s16_unknown_nonce_reject() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    // deliberately do not register
    let v = ok_verifier();

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(
        out,
        AttestOutcome::Rejected {
            reason: RejectReason::NonceInvalid
        }
    );
}

/// `SWHardeningNeeded` still Verified (accept + warn).
#[test]
fn s17_sw_hardening_accept() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let binding = binding_base();
    let td = td_for(&binding, &entry);
    let mut nonces = MemoryNonceStore::new();
    let now = Instant::now();
    register_nonce(&mut nonces, binding.nonce, now);
    let v = MockQuoteVerifier::Ok(QuoteVerifyOk {
        tcb_status: TcbStatus::SWHardeningNeeded,
        collateral: CollateralFreshness::Fresh,
    });

    let out = run(
        &body,
        &td,
        &entry.compose_hash,
        binding,
        &mut nonces,
        now,
        &v,
    );
    assert_eq!(out, AttestOutcome::Verified);
}

/// End-to-end: parked this epoch after prior Verified — scoring credit only if Verified now.
#[test]
fn s18_full_pipeline_park_clears_credit_path() {
    let entry = entry_a();
    let body = body_with(entry.clone());
    let mut binding = binding_base();
    let mut book = AttestCreditBook::new();

    // Epoch 100 Verified
    {
        let td = td_for(&binding, &entry);
        let mut nonces = MemoryNonceStore::new();
        let now = Instant::now();
        register_nonce(&mut nonces, binding.nonce, now);
        let v = ok_verifier();
        let out = run(
            &body,
            &td,
            &entry.compose_hash,
            binding,
            &mut nonces,
            now,
            &v,
        );
        assert_eq!(out, AttestOutcome::Verified);
        book.record(
            CreditKey {
                netuid: binding.netuid,
                epoch: binding.epoch,
                miner: binding.miner_pubkey,
            },
            out,
        );
    }

    // Epoch 101 Parked (OutOfDate) — new nonce
    binding.epoch = 101;
    binding.nonce = [0x99; KEY_LEN];
    {
        let td = td_for(&binding, &entry);
        let mut nonces = MemoryNonceStore::new();
        let now = Instant::now();
        register_nonce(&mut nonces, binding.nonce, now);
        let v = MockQuoteVerifier::Ok(QuoteVerifyOk {
            tcb_status: TcbStatus::OutOfDate,
            collateral: CollateralFreshness::Fresh,
        });
        let out = run(
            &body,
            &td,
            &entry.compose_hash,
            binding,
            &mut nonces,
            now,
            &v,
        );
        assert!(matches!(out, AttestOutcome::Parked { .. }));
        let key = CreditKey {
            netuid: binding.netuid,
            epoch: binding.epoch,
            miner: binding.miner_pubkey,
        };
        book.record(key, out);
        assert!(
            !book.has_credit(&key),
            "parked miner has no attestation credit this epoch"
        );
        // Must not inherit epoch 100
        assert!(book.has_credit(&CreditKey {
            netuid: 1,
            epoch: 100,
            miner: binding.miner_pubkey,
        }));
        assert!(!book.has_credit(&key));
    }
}

/// Glue: `mr_config_id` v1 matches compose hash.
#[test]
fn s19_mr_config_id_matches_compose_hash() {
    use gbase_attest_policy::{compose_hash, mr_config_id_matches};
    let entry = entry_a();
    let binding = binding_base();
    let mut td = td_for(&binding, &entry);
    td.mr_config_id = compose_hash::mr_config_id(&entry.compose_hash);
    assert!(mr_config_id_matches(&td, &entry.compose_hash));
    td.mr_config_id[0] ^= 1;
    assert!(!mr_config_id_matches(&td, &entry.compose_hash));
}

/// Glue: `replay_compose_hash` extracts 32-byte payload.
#[test]
fn s20_replay_compose_hash_ok() {
    use gbase_attest_policy::{replay, replay_compose_hash};
    use gbase_attest_replay::{
        event_digest, APP_IMR, COMPOSE_HASH_EVENT, DSTACK_RUNTIME_EVENT_TYPE,
    };
    let payload = [7u8; 32];
    let digest = event_digest(DSTACK_RUNTIME_EVENT_TYPE, COMPOSE_HASH_EVENT, &payload);
    let events = [replay::Event {
        imr: APP_IMR,
        event_type: DSTACK_RUNTIME_EVENT_TYPE,
        name: COMPOSE_HASH_EVENT.into(),
        payload: payload.to_vec(),
        digest: Some(digest),
    }];
    let (hash, r) = replay_compose_hash(&events).expect("replay");
    assert_eq!(hash, payload);
    assert_eq!(r.compose_hash.as_deref(), Some(payload.as_slice()));
}
