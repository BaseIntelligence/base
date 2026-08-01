//! Real Intel DCAP quote verification backed by [`dcap_qvl`] (feature `dcap`).
//!
//! [`DcapQuoteVerifier`] is the production implementation of
//! [`QuoteVerifier`]: it fetches DCAP collateral for the quote's platform from
//! a PCCS / Intel PCS endpoint, runs the full `dcap-qvl` trust-chain +
//! signature + TCB appraisal, and reports the outcome using the existing
//! policy vocabulary ([`TcbStatus`] / [`CollateralFreshness`] /
//! [`VerifierFailureKind`]). TCB appraisal itself stays in
//! [`crate::classify_tcb`]; this module only translates.
//!
//! Collateral is cached in-memory per platform (SHA-256 of the quote's PCK
//! certificate chain, which is a strict refinement of the FMSPC that keys
//! Intel's endpoints) for `max_collateral_age`, so a burst of certify calls
//! issues one PCS round-trip rather than one per miner submission.

use std::collections::BTreeMap;
use std::sync::mpsc::{sync_channel, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dcap_qvl::collateral::CollateralClient;
use dcap_qvl::quote::Quote as QvlQuote;
use dcap_qvl::verify::QuoteVerifier as QvlVerifier;
use dcap_qvl::{QuoteCollateralV3, QuotePolicy};
use sha2::{Digest, Sha256};

use crate::dcap_map::{
    classify_dcap_collateral_error, classify_dcap_verify_error, is_dcap_revoked_error,
    map_dcap_tcb_status,
};
use crate::tcb::{CollateralFreshness, TcbStatus};
use crate::verifier::{QuoteVerifier, QuoteVerifyError, QuoteVerifyOk, VerifierFailureKind};

/// Intel's production PCS base URL (the default collateral source).
pub const DEFAULT_PCS_URL: &str = dcap_qvl::collateral::INTEL_PCS_URL;

/// Phala's public PCCS mirror; useful when Intel PCS rate-limits a validator.
pub const PHALA_PCCS_URL: &str = dcap_qvl::PHALA_PCCS_URL;

/// Default maximum age of a collateral snapshot before it parks a quote.
pub const DEFAULT_COLLATERAL_MAX_AGE: Duration = Duration::from_hours(24);

/// Default wall-clock budget for one collateral fetch.
pub const DEFAULT_FETCH_TIMEOUT: Duration = Duration::from_secs(20);

/// Failure building a [`DcapQuoteVerifier`].
#[derive(Debug, thiserror::Error)]
pub enum DcapInitError {
    /// The dedicated collateral-fetch Tokio runtime could not be started.
    #[error("dcap collateral runtime: {0}")]
    Runtime(String),
    /// The HTTPS client for the PCCS / PCS endpoint could not be built.
    #[error("dcap collateral http client: {0}")]
    HttpClient(String),
}

struct CachedCollateral {
    collateral: QuoteCollateralV3,
    fetched_at: Instant,
}

/// Production [`QuoteVerifier`] using `dcap-qvl` against a PCCS / Intel PCS.
///
/// Cheap to share: it is `Send + Sync` and owns its own single-worker Tokio
/// runtime for the async collateral client, so it can live inside the shared
/// validator attestation state without leaking a runtime handle.
pub struct DcapQuoteVerifier {
    runtime: tokio::runtime::Runtime,
    client: Arc<CollateralClient>,
    qvl: QvlVerifier,
    max_collateral_age: Duration,
    fetch_timeout: Duration,
    cache: Mutex<BTreeMap<[u8; 32], CachedCollateral>>,
}

impl std::fmt::Debug for DcapQuoteVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DcapQuoteVerifier")
            .field("max_collateral_age", &self.max_collateral_age)
            .field("fetch_timeout", &self.fetch_timeout)
            .finish_non_exhaustive()
    }
}

impl DcapQuoteVerifier {
    /// Build a verifier against `pccs_url` (use [`DEFAULT_PCS_URL`] for Intel).
    ///
    /// `max_collateral_age` bounds both the in-memory collateral cache TTL and
    /// the age at which a still-valid collateral snapshot is reported as
    /// [`CollateralFreshness::Expired`] (→ Park).
    ///
    /// # Errors
    ///
    /// Returns [`DcapInitError`] when the dedicated Tokio runtime or the HTTPS
    /// client cannot be created.
    pub fn new(
        pccs_url: impl Into<String>,
        max_collateral_age: Duration,
    ) -> Result<Self, DcapInitError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("attest-dcap-pcs")
            .enable_all()
            .build()
            .map_err(|e| DcapInitError::Runtime(e.to_string()))?;
        let client = CollateralClient::with_default_http(pccs_url)
            .map_err(|e| DcapInitError::HttpClient(format!("{e:#}")))?;
        Ok(Self {
            runtime,
            client: Arc::new(client),
            qvl: QvlVerifier::new_prod(),
            max_collateral_age,
            fetch_timeout: DEFAULT_FETCH_TIMEOUT,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    /// Override the per-fetch collateral timeout (default
    /// [`DEFAULT_FETCH_TIMEOUT`]). Exceeding it parks with `PcsTimeout`.
    #[must_use]
    pub fn with_fetch_timeout(mut self, timeout: Duration) -> Self {
        self.fetch_timeout = timeout;
        self
    }

    /// Configured maximum collateral age.
    #[must_use]
    pub const fn max_collateral_age(&self) -> Duration {
        self.max_collateral_age
    }

    fn cache_guard(&self) -> MutexGuard<'_, BTreeMap<[u8; 32], CachedCollateral>> {
        // A panic in a cache critical section cannot corrupt the map (it only
        // ever holds owned collateral), so recover rather than poison-park.
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn cached(&self, key: &[u8; 32]) -> Option<QuoteCollateralV3> {
        let guard = self.cache_guard();
        let entry = guard.get(key)?;
        if entry.fetched_at.elapsed() > self.max_collateral_age {
            return None;
        }
        Some(entry.collateral.clone())
    }

    fn collateral_for(
        &self,
        key: [u8; 32],
        quote: &[u8],
    ) -> Result<QuoteCollateralV3, QuoteVerifyError> {
        if let Some(hit) = self.cached(&key) {
            return Ok(hit);
        }
        let fetched = self.fetch_collateral(quote)?;
        self.cache_guard().insert(
            key,
            CachedCollateral {
                collateral: fetched.clone(),
                fetched_at: Instant::now(),
            },
        );
        Ok(fetched)
    }

    fn fetch_collateral(&self, quote: &[u8]) -> Result<QuoteCollateralV3, QuoteVerifyError> {
        let (tx, rx) = sync_channel(1);
        let client = Arc::clone(&self.client);
        let raw = quote.to_vec();
        self.runtime.spawn(async move {
            let out = client.fetch(&raw).await.map_err(|e| format!("{e:#}"));
            let _ = tx.send(out);
        });
        match rx.recv_timeout(self.fetch_timeout) {
            Ok(Ok(collateral)) => Ok(collateral),
            Ok(Err(msg)) => Err(fail(classify_dcap_collateral_error(&msg))),
            Err(RecvTimeoutError::Timeout) => Err(fail(VerifierFailureKind::PcsTimeout)),
            Err(RecvTimeoutError::Disconnected) => Err(fail(VerifierFailureKind::Unavailable)),
        }
    }
}

impl QuoteVerifier for DcapQuoteVerifier {
    fn verify(&self, quote: &[u8]) -> Result<QuoteVerifyOk, QuoteVerifyError> {
        let now = unix_now()?;
        let key = platform_cache_key(quote)?;
        let collateral = self.collateral_for(key, quote)?;

        // `claims_only` supplies the trusted verification time and defers every
        // accept/park/reject decision to `classify_tcb`, so the DCAP appraisal
        // never silently overrides D13.
        let policy = QuotePolicy::claims_only(now);
        let claims = match self
            .qvl
            .verify_with_policy(quote, &collateral, now, &policy)
        {
            Ok(claims) => claims,
            Err(e) => {
                let msg = format!("{e:#}");
                if is_dcap_revoked_error(&msg) {
                    // Report Revoked as a verdict (Reject / TcbRevoked) rather
                    // than a generic crypto failure; `Fresh` keeps the collateral
                    // branch of `classify_tcb` from downgrading it to Park.
                    return Ok(QuoteVerifyOk {
                        tcb_status: TcbStatus::Revoked,
                        collateral: CollateralFreshness::Fresh,
                    });
                }
                return Err(fail(classify_dcap_verify_error(&msg)));
            }
        };

        let Some(tcb_status) = map_dcap_tcb_status(&claims.tcb.status.to_string()) else {
            return Err(fail(VerifierFailureKind::Unavailable));
        };
        let age = now.saturating_sub(claims.latest_issue_date);
        let collateral = if age > self.max_collateral_age.as_secs() {
            CollateralFreshness::Expired
        } else {
            CollateralFreshness::Fresh
        };
        Ok(QuoteVerifyOk {
            tcb_status,
            collateral,
        })
    }
}

const fn fail(kind: VerifierFailureKind) -> QuoteVerifyError {
    QuoteVerifyError { kind }
}

fn unix_now() -> Result<u64, QuoteVerifyError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|_| fail(VerifierFailureKind::Unavailable))
}

/// SHA-256 of the quote's PCK certificate chain — one cache slot per platform.
fn platform_cache_key(quote: &[u8]) -> Result<[u8; 32], QuoteVerifyError> {
    let parsed = QvlQuote::parse(quote).map_err(|_| fail(VerifierFailureKind::CryptoInvalid))?;
    let chain = parsed
        .raw_cert_chain()
        .map_err(|_| fail(VerifierFailureKind::CryptoInvalid))?;
    let mut hasher = Sha256::new();
    hasher.update(chain);
    Ok(hasher.finalize().into())
}
