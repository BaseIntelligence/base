//! Full certify verify path: parse → replay → compose-hash → policy.
//!
//! Used by the validator attestation endpoint (task 38). Callers supply a
//! [`NonceStore`] and [`QuoteVerifier`] (mock or dcap).

use std::time::Instant;

use attest_parse::parse_tdx_quote_v4;
use attest_replay::events_from_json;
use crypto::NonceStore;
use trustroot::MeasurementsBody;

use crate::glue::replay_compose_hash;
use crate::policy::{evaluate, PolicyInput};
use crate::report_data::ReportDataBinding;
use crate::verifier::QuoteVerifier;
use crate::{AttestOutcome, RejectReason};

/// Inputs for one miner certify submission (quote + event log + D10 claims).
pub struct SubmissionInput<'a, S: NonceStore, V: QuoteVerifier + ?Sized> {
    /// Owner-signed measurement allowlist.
    pub measurements: &'a MeasurementsBody,
    /// Raw TDX quote bytes.
    pub quote: &'a [u8],
    /// RTMR3 event log JSON (Phala / dstack shape).
    pub event_log_json: &'a [u8],
    /// Claimed D10 binding (must match quote `report_data` after parse).
    pub binding: ReportDataBinding,
    /// Single-use nonce store (nonce must already be registered).
    pub nonces: &'a mut S,
    /// Evaluation time for nonce redeem.
    pub now: Instant,
    /// Quote crypto verifier.
    pub verifier: &'a V,
}

/// Parse → replay → compose-hash → [`evaluate`]. Always returns an outcome.
///
/// Structural failures (malformed quote / event log / missing compose-hash)
/// map to [`AttestOutcome::Rejected`] without redeeming the nonce when parse
/// fails before the policy stage. Once policy runs, nonce redeem follows
/// [`evaluate`] rules (invalid nonce → Reject after attempted redeem).
#[must_use]
pub fn verify_submission<S: NonceStore, V: QuoteVerifier + ?Sized>(
    input: &mut SubmissionInput<'_, S, V>,
) -> AttestOutcome {
    let Ok(parsed) = parse_tdx_quote_v4(input.quote) else {
        return AttestOutcome::Rejected {
            reason: RejectReason::QuoteMalformed,
        };
    };

    let Ok(events) = events_from_json(input.event_log_json) else {
        return AttestOutcome::Rejected {
            reason: RejectReason::EventLogInvalid,
        };
    };

    let Ok((compose_hash, _replay)) = replay_compose_hash(&events) else {
        return AttestOutcome::Rejected {
            reason: RejectReason::EventLogInvalid,
        };
    };

    evaluate(&mut PolicyInput {
        measurements: input.measurements,
        td_report: &parsed.td_report,
        compose_hash: &compose_hash,
        binding: input.binding,
        quote: input.quote,
        nonces: input.nonces,
        now: input.now,
        verifier: input.verifier,
    })
}
