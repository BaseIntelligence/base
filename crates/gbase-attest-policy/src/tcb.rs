//! D13 TCB status → accept / warn / park / reject.

/// Intel DCAP-style TCB status values we map (subset used by policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcbStatus {
    /// Fully current — accept.
    UpToDate,
    /// Accept with warn metric.
    SWHardeningNeeded,
    /// Accept with warn metric.
    ConfigurationNeeded,
    /// Park (no credit).
    OutOfDate,
    /// Park (no credit).
    OutOfDateConfigurationNeeded,
    /// Cryptographic reject.
    Revoked,
}

/// Collateral age relative to `max_collateral_age`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollateralFreshness {
    /// Within configured max age.
    Fresh,
    /// Older than `max_collateral_age` → park.
    Expired,
}

/// Action after TCB + collateral classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcbAction {
    /// Accept (optionally with warn).
    Accept {
        /// Emit warn metric (`SWHardeningNeeded` / `ConfigurationNeeded`).
        warn: bool,
    },
    /// Park — no credit, no carry-forward.
    Park,
    /// Reject — revoked TCB.
    Reject,
}

/// Map TCB status and collateral freshness to a policy action (D13).
///
/// Expired collateral always parks (even if TCB is `UpToDate`).
#[must_use]
pub fn classify_tcb(status: TcbStatus, collateral: CollateralFreshness) -> TcbAction {
    if matches!(collateral, CollateralFreshness::Expired) {
        return TcbAction::Park;
    }
    match status {
        TcbStatus::UpToDate => TcbAction::Accept { warn: false },
        TcbStatus::SWHardeningNeeded | TcbStatus::ConfigurationNeeded => {
            TcbAction::Accept { warn: true }
        }
        TcbStatus::OutOfDate | TcbStatus::OutOfDateConfigurationNeeded => TcbAction::Park,
        TcbStatus::Revoked => TcbAction::Reject,
    }
}
