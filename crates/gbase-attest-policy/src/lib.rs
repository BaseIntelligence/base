//! Attestation policy engine: D10 `report_data` binding, D13 TCB table,
//! measurement allowlist (via `gbase-trustroot`), single-use nonce redeem,
//! and park-vs-reject outcomes.
//!
//! # Outcomes (D13)
//!
//! | Outcome | Attestation credit | Carry prior `Verified` |
//! |---------|--------------------|------------------------|
//! | [`AttestOutcome::Verified`] | yes | n/a (fresh) |
//! | [`AttestOutcome::Rejected`] | no | never |
//! | [`AttestOutcome::Parked`] | no | **never** |
//!
//! Cryptographic / binding failures → **Reject**. Verifier or collateral
//! outages → **Park**. Park is neutral: it grants no credit and never
//! inherits last epoch's `Verified`.
//!
//! # DCAP
//!
//! Unit tests use [`MockQuoteVerifier`]. Real `dcap-qvl >= 0.6.1` is an
//! optional `dcap` feature; full integration is tasks 35/38. `deny.toml`
//! bans `dcap-qvl < 0.6.1`.

#![forbid(unsafe_code)]

mod credit;
mod error;
mod glue;
mod pipeline;
mod policy;
mod report_data;
mod tcb;
mod verifier;

pub use credit::{AttestCreditBook, CreditKey};
pub use error::PolicyError;
pub use glue::{mr_config_id_matches, replay_compose_hash};
pub use pipeline::{verify_submission, SubmissionInput};
pub use policy::{evaluate, PolicyInput};
pub use report_data::{
    compute_report_data, report_data_preimage, verify_report_data, ReportDataBinding,
    REPORT_DATA_LEN,
};
pub use tcb::{classify_tcb, CollateralFreshness, TcbAction, TcbStatus};
pub use verifier::{
    MockQuoteVerifier, QuoteVerifier, QuoteVerifyError, QuoteVerifyOk, VerifierFailureKind,
};
// Re-export pipeline crates callers need beside policy.
pub use gbase_attest_parse as parse;
pub use gbase_attest_replay as replay;
pub use gbase_compose_hash as compose_hash;
pub use gbase_crypto as crypto;
pub use gbase_trustroot as trustroot;

/// Final attestation decision for one `(netuid, epoch, miner)` evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestOutcome {
    /// Quote, binding, nonce, allowlist, and TCB all accepted.
    Verified,
    /// Cryptographic or policy failure — no credit.
    Rejected {
        /// Stable machine reason.
        reason: RejectReason,
    },
    /// Verifier/collateral outage or soft TCB — no credit, no carry-forward.
    Parked {
        /// Stable machine reason.
        reason: ParkReason,
    },
}

impl AttestOutcome {
    /// Whether this outcome grants attestation credit for the epoch.
    #[must_use]
    pub const fn grants_credit(self) -> bool {
        matches!(self, Self::Verified)
    }

    /// Whether a prior `Verified` may be reused (always `false` under D13).
    #[must_use]
    pub const fn carries_prior_verified(self) -> bool {
        false
    }
}

/// Reject causes (cryptographic / binding / allowlist).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// Measurements body missing or empty (fail closed).
    EmptyAllowlist,
    /// Quote measurements / compose hash not on allowlist.
    MeasurementNotAllowlisted,
    /// `report_data` does not match D10 binding for claimed fields.
    ReportDataMismatch,
    /// Nonce unknown, expired, or already redeemed.
    NonceInvalid,
    /// DCAP / quote crypto verification failed.
    QuoteCryptoInvalid,
    /// TCB status is revoked.
    TcbRevoked,
    /// Quote bytes failed structural parse (truncated / wrong version).
    QuoteMalformed,
    /// Event log JSON or RTMR3 replay failed.
    EventLogInvalid,
}

/// Park causes (outage / soft TCB / stale collateral).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParkReason {
    /// PCS / collateral fetch timed out or unavailable.
    PcsTimeout,
    /// Collateral older than `max_collateral_age`.
    CollateralExpired,
    /// TCB `OutOfDate` or `OutOfDateConfigurationNeeded`.
    TcbOutOfDate,
    /// Other verifier infrastructure failure (non-crypto).
    VerifierUnavailable,
}
