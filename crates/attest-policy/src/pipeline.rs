//! Full certify verify path: parse → replay → compose-hash → policy.
//!
//! Used by the validator attestation endpoint (task 38). Callers supply a
//! [`NonceStore`] and [`QuoteVerifier`] (mock or dcap).

use std::time::Instant;

use attest_parse::parse_tdx_quote_v4;
use attest_replay::events_from_json;
use crypto::{NonceStore, KEY_LEN};
use trustroot::MeasurementsBody;

use crate::glue::replay_compose_hash;
use crate::policy::{evaluate, PolicyInput};
use crate::receipt_key::{receipt_key_from_compose, verify_compose_preimage, ComposeBindError};
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
    /// Measured `app-compose.json` preimage, when the miner supplied one.
    ///
    /// Checked against the RTMR3 compose hash before anything is read from it.
    /// `None` keeps the outcome unchanged but yields no receipt key, so the
    /// miner's work receipts stay unverifiable downstream.
    pub app_compose: Option<&'a [u8]>,
    /// Single-use nonce store (nonce must already be registered).
    pub nonces: &'a mut S,
    /// Evaluation time for nonce redeem.
    pub now: Instant,
    /// Quote crypto verifier.
    pub verifier: &'a V,
}

/// Outcome of one submission plus what the measured compose bound to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmissionVerdict {
    /// Policy decision.
    pub outcome: AttestOutcome,
    /// Receipt public key read out of the measured compose. Only ever set for
    /// [`AttestOutcome::Verified`].
    pub receipt_pk: Option<[u8; KEY_LEN]>,
}

/// Parse → replay → compose-hash → [`evaluate`]. Always returns a verdict.
///
/// Structural failures (malformed quote / event log / missing compose-hash)
/// map to [`AttestOutcome::Rejected`] without redeeming the nonce when parse
/// fails before the policy stage. Once policy runs, nonce redeem follows
/// [`evaluate`] rules (invalid nonce → Reject after attempted redeem).
///
/// A supplied `app_compose` is bound to RTMR3 before the policy runs: a
/// preimage that does not hash to the measured compose hash is a hard
/// [`RejectReason::ComposePreimageMismatch`], never a park, because accepting
/// it would let a miner name any receipt key it likes. An otherwise-`Verified`
/// attestation whose measured compose carries no usable receipt key is
/// [`RejectReason::ReceiptKeyInvalid`] — the same fail-closed treatment the
/// policy gives every other missing measured field.
#[must_use]
pub fn verify_submission<S: NonceStore, V: QuoteVerifier + ?Sized>(
    input: &mut SubmissionInput<'_, S, V>,
) -> SubmissionVerdict {
    let Ok(parsed) = parse_tdx_quote_v4(input.quote) else {
        return rejected(RejectReason::QuoteMalformed);
    };

    let Ok(events) = events_from_json(input.event_log_json) else {
        return rejected(RejectReason::EventLogInvalid);
    };

    let Ok((compose_hash, _replay)) = replay_compose_hash(&events) else {
        return rejected(RejectReason::EventLogInvalid);
    };

    if let Some(app_compose) = input.app_compose {
        if verify_compose_preimage(app_compose, &compose_hash).is_err() {
            return rejected(RejectReason::ComposePreimageMismatch);
        }
    }

    let outcome = evaluate(&mut PolicyInput {
        measurements: input.measurements,
        td_report: &parsed.td_report,
        compose_hash: &compose_hash,
        binding: input.binding,
        quote: input.quote,
        nonces: input.nonces,
        now: input.now,
        verifier: input.verifier,
    });

    // Only a Verified attestation can vouch for a key; a park stays a park.
    let Some(app_compose) = input.app_compose.filter(|_| outcome.grants_credit()) else {
        return SubmissionVerdict {
            outcome,
            receipt_pk: None,
        };
    };
    match receipt_key_from_compose(app_compose) {
        Ok(pk) => SubmissionVerdict {
            outcome,
            receipt_pk: Some(pk),
        },
        Err(ComposeBindError::PreimageMismatch) => rejected(RejectReason::ComposePreimageMismatch),
        Err(ComposeBindError::ReceiptKeyMissing | ComposeBindError::ReceiptKeyMalformed) => {
            rejected(RejectReason::ReceiptKeyInvalid)
        }
    }
}

const fn rejected(reason: RejectReason) -> SubmissionVerdict {
    SubmissionVerdict {
        outcome: AttestOutcome::Rejected { reason },
        receipt_pk: None,
    }
}
