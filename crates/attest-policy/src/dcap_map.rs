//! Pure mappings from `dcap-qvl` verdicts and error text onto policy types.
//!
//! These tables live outside the `dcap` feature gate on purpose: they are the
//! part of the real verifier that decides Verified / Park / Reject, so the
//! default (offline, feature-less) CI path still unit-tests them.

use crate::tcb::TcbStatus;
use crate::verifier::VerifierFailureKind;

/// Map a `dcap-qvl` TCB status name onto the policy [`TcbStatus`].
///
/// `dcap-qvl` renders its `TcbStatus` as the bare Rust variant name, so the
/// accepted inputs are `UpToDate`, `SWHardeningNeeded`, `ConfigurationNeeded`,
/// `ConfigurationAndSWHardeningNeeded`, `OutOfDate`,
/// `OutOfDateConfigurationNeeded` and `Revoked`.
///
/// Intel's `ConfigurationAndSWHardeningNeeded` has no policy variant. It maps
/// to [`TcbStatus::OutOfDate`] (Park) — the most conservative existing variant
/// — rather than widening the accept-with-warn set.
///
/// Returns `None` for a status name this build does not recognise; callers
/// treat that as an unavailable verifier (Park), never as an accept.
#[must_use]
pub fn map_dcap_tcb_status(status: &str) -> Option<TcbStatus> {
    match status {
        "UpToDate" => Some(TcbStatus::UpToDate),
        "SWHardeningNeeded" => Some(TcbStatus::SWHardeningNeeded),
        "ConfigurationNeeded" => Some(TcbStatus::ConfigurationNeeded),
        "ConfigurationAndSWHardeningNeeded" | "OutOfDate" => Some(TcbStatus::OutOfDate),
        "OutOfDateConfigurationNeeded" => Some(TcbStatus::OutOfDateConfigurationNeeded),
        "Revoked" => Some(TcbStatus::Revoked),
        _ => None,
    }
}

/// Whether a `dcap-qvl` verification error means the platform TCB is revoked.
///
/// `dcap-qvl` refuses to build a report for a revoked platform and instead
/// fails with `TCB status is invalid: Revoked`; a revoked PCK certificate
/// surfaces the same way. Both are [`TcbStatus::Revoked`] for policy.
#[must_use]
pub fn is_dcap_revoked_error(msg: &str) -> bool {
    msg.to_ascii_lowercase().contains("revoked")
}

/// Classify a `dcap-qvl` *verification* error into a [`VerifierFailureKind`].
///
/// Verification runs fully offline once collateral is in hand, so anything
/// that is not an explicit validity-window failure is treated as a
/// cryptographic rejection ([`VerifierFailureKind::CryptoInvalid`] → Reject).
/// Expiry / not-yet-valid text maps to
/// [`VerifierFailureKind::CollateralExpired`] (→ Park), because a stale
/// collateral snapshot is an operator problem, not a miner problem.
#[must_use]
pub fn classify_dcap_verify_error(msg: &str) -> VerifierFailureKind {
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("expired")
        || lowered.contains("is in the future")
        || lowered.contains("not yet valid")
    {
        VerifierFailureKind::CollateralExpired
    } else {
        VerifierFailureKind::CryptoInvalid
    }
}

/// Classify a `dcap-qvl` *collateral fetch* error into a [`VerifierFailureKind`].
///
/// Fetching hits PCCS / Intel PCS, so failures are network failures
/// ([`VerifierFailureKind::PcsTimeout`] → Park). The one exception is the
/// quote decode `dcap-qvl` performs before it can build the request, which is
/// a malformed quote and therefore a Reject.
#[must_use]
pub fn classify_dcap_collateral_error(msg: &str) -> VerifierFailureKind {
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("parse quote") || lowered.contains("decode quote") {
        VerifierFailureKind::CryptoInvalid
    } else {
        VerifierFailureKind::PcsTimeout
    }
}
