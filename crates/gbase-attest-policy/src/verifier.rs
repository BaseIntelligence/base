//! Quote cryptographic verification seam.
//!
//! Production (tasks 35/38) will implement this with `dcap-qvl >= 0.6.1`
//! behind the `dcap` feature. Unit tests use [`MockQuoteVerifier`].

use crate::tcb::{CollateralFreshness, TcbStatus};

/// Successful DCAP-style quote verification result (measurements already in quote).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteVerifyOk {
    /// TCB status from the verifier.
    pub tcb_status: TcbStatus,
    /// Collateral freshness vs configured max age.
    pub collateral: CollateralFreshness,
}

/// Verifier-side failures (mapped to Park or Reject by policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierFailureKind {
    /// Quote signature / cert chain crypto invalid → Reject.
    CryptoInvalid,
    /// PCS / collateral HTTP timeout → Park.
    PcsTimeout,
    /// Collateral present but past max age → Park.
    CollateralExpired,
    /// Other infrastructure unavailability → Park.
    Unavailable,
}

/// Error from [`QuoteVerifier::verify`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteVerifyError {
    /// Failure class.
    pub kind: VerifierFailureKind,
}

/// Abstract quote verifier (mock in tests; dcap-qvl in prod feature).
pub trait QuoteVerifier {
    /// Verify quote bytes (and optional collateral material).
    ///
    /// # Errors
    ///
    /// Returns [`QuoteVerifyError`] on crypto failure or infrastructure outage.
    fn verify(&self, quote: &[u8]) -> Result<QuoteVerifyOk, QuoteVerifyError>;
}

/// Configurable mock for unit tests (no network, no dcap-qvl).
#[derive(Debug, Clone)]
pub enum MockQuoteVerifier {
    /// Always succeed with the given TCB + collateral.
    Ok(QuoteVerifyOk),
    /// Always fail with the given kind.
    Err(VerifierFailureKind),
}

impl QuoteVerifier for MockQuoteVerifier {
    fn verify(&self, _quote: &[u8]) -> Result<QuoteVerifyOk, QuoteVerifyError> {
        match self {
            Self::Ok(ok) => Ok(ok.clone()),
            Self::Err(kind) => Err(QuoteVerifyError { kind: *kind }),
        }
    }
}

#[cfg(feature = "dcap")]
mod dcap_stub {
    //! Placeholder for task 35/38: wire `dcap-qvl` here.
    //!
    //! Do not enable `dcap` in default CI until collateral + PCS fixtures land.
    #![allow(dead_code)]

    // Re-export version floor for compile-time presence checks.
    pub use dcap_qvl as qvl;
}
