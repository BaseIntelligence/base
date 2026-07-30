//! Core policy evaluation: allowlist → `report_data` → nonce → quote → TCB.

use std::time::Instant;

use gbase_attest_parse::TdReport;
use gbase_crypto::{NonceError, NonceStore};
use gbase_trustroot::{MeasurementsBody, COMPOSE_HASH_LEN};

use crate::report_data::{verify_report_data, ReportDataBinding};
use crate::tcb::{classify_tcb, CollateralFreshness, TcbAction, TcbStatus};
use crate::verifier::{QuoteVerifier, VerifierFailureKind};
use crate::{AttestOutcome, ParkReason, RejectReason};

/// Inputs for one attestation policy evaluation.
pub struct PolicyInput<'a, S: NonceStore, V: QuoteVerifier> {
    /// Owner-signed measurement allowlist (empty ⇒ fail closed).
    pub measurements: &'a MeasurementsBody,
    /// Parsed TD report fields (from `gbase-attest-parse`).
    pub td_report: &'a TdReport,
    /// Compose hash from RTMR3 replay / event log (32 bytes).
    pub compose_hash: &'a [u8; COMPOSE_HASH_LEN],
    /// Claimed D10 binding (must match `td_report.report_data`).
    pub binding: ReportDataBinding,
    /// Raw quote bytes passed to the verifier.
    pub quote: &'a [u8],
    /// Single-use nonce store.
    pub nonces: &'a mut S,
    /// Evaluation time for nonce redeem.
    pub now: Instant,
    /// Quote verifier (mock or dcap).
    pub verifier: &'a V,
}

/// Run the full policy pipeline. Always returns an [`AttestOutcome`] (never panics).
#[must_use]
pub fn evaluate<S: NonceStore, V: QuoteVerifier>(
    input: &mut PolicyInput<'_, S, V>,
) -> AttestOutcome {
    // 1. Allowlist fail-closed (empty body rejects every quote).
    if input.measurements.entries.is_empty() {
        return reject(RejectReason::EmptyAllowlist);
    }

    // 2. D10 report_data binding (epoch / validator / miner / nonce all in hash).
    if !verify_report_data(&input.td_report.report_data, &input.binding) {
        return reject(RejectReason::ReportDataMismatch);
    }

    // 3. Single-use nonce redeem (unknown / expired / replay → Reject).
    match input.nonces.redeem(&input.binding.nonce, input.now) {
        Ok(()) => {}
        Err(NonceError::Unknown | NonceError::Expired | NonceError::AlreadyPresent) => {
            return reject(RejectReason::NonceInvalid);
        }
    }

    // 4. Measurement allowlist exact match.
    if !allows(input.measurements, input.td_report, input.compose_hash) {
        return reject(RejectReason::MeasurementNotAllowlisted);
    }

    // 5. Quote crypto + TCB / collateral.
    match input.verifier.verify(input.quote) {
        Ok(ok) => map_tcb(ok.tcb_status, ok.collateral),
        Err(e) => map_verifier_err(e.kind),
    }
}

fn allows(
    body: &MeasurementsBody,
    td: &TdReport,
    compose_hash: &[u8; COMPOSE_HASH_LEN],
) -> bool {
    body.allows_quote(
        &td.mr_td,
        &td.rtmr0,
        &td.rtmr1,
        &td.rtmr2,
        &td.rtmr3,
        compose_hash,
    )
}

fn map_tcb(status: TcbStatus, collateral: CollateralFreshness) -> AttestOutcome {
    match classify_tcb(status, collateral) {
        TcbAction::Accept { warn: _ } => AttestOutcome::Verified,
        TcbAction::Park => {
            // Distinguish collateral vs TCB out-of-date for metrics/reasons.
            if matches!(collateral, CollateralFreshness::Expired) {
                park(ParkReason::CollateralExpired)
            } else {
                park(ParkReason::TcbOutOfDate)
            }
        }
        TcbAction::Reject => reject(RejectReason::TcbRevoked),
    }
}

fn map_verifier_err(kind: VerifierFailureKind) -> AttestOutcome {
    match kind {
        VerifierFailureKind::CryptoInvalid => reject(RejectReason::QuoteCryptoInvalid),
        VerifierFailureKind::PcsTimeout => park(ParkReason::PcsTimeout),
        VerifierFailureKind::CollateralExpired => park(ParkReason::CollateralExpired),
        VerifierFailureKind::Unavailable => park(ParkReason::VerifierUnavailable),
    }
}

fn reject(reason: RejectReason) -> AttestOutcome {
    AttestOutcome::Rejected { reason }
}

fn park(reason: ParkReason) -> AttestOutcome {
    AttestOutcome::Parked { reason }
}

