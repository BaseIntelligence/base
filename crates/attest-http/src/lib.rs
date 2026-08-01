//! Validator-side attestation: nonce issue + quote verify (task 38).
//!
//! HTTP:
//! - `POST /v1/attest/nonce` — issue a single-use 32-byte nonce (TTL < epoch)
//! - `POST /v1/attest/submit` — parse → replay → compose-hash → policy → record
//!
//! Outcomes: `Verified` / `Rejected` / `Parked` (D13: park grants no credit and
//! never carries a prior `Verified`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use attest_policy::{
    verify_submission, AttestCreditBook, AttestOutcome, CollateralFreshness, CreditKey,
    MockQuoteVerifier, ParkReason, QuoteVerifier, QuoteVerifyOk, RejectReason, ReportDataBinding,
    SubmissionInput, TcbStatus, VerifierFailureKind,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use crypto::{register_with_ttl, MemoryNonceStore, KEY_LEN};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use trustroot::MeasurementsBody;

/// Default nonce TTL (must be strictly less than one epoch at the call site).
pub const DEFAULT_NONCE_TTL: Duration = Duration::from_mins(5);

/// Shared attestation service state (in-memory; postgres wiring is later).
#[derive(Clone)]
pub struct AttestState {
    inner: Arc<Mutex<AttestInner>>,
}

struct AttestInner {
    nonces: MemoryNonceStore,
    book: AttestCreditBook,
    measurements: MeasurementsBody,
    validator_hotkey: [u8; KEY_LEN],
    netuid: u16,
    verifier: Arc<dyn QuoteVerifier>,
    nonce_ttl: Duration,
}

impl AttestState {
    /// Build state with allowlist + validator identity + any verifier.
    #[must_use]
    pub fn new(
        measurements: MeasurementsBody,
        validator_hotkey: [u8; KEY_LEN],
        netuid: u16,
        verifier: impl QuoteVerifier + 'static,
    ) -> Self {
        Self::with_verifier(measurements, validator_hotkey, netuid, Box::new(verifier))
    }

    /// Build state with an already-boxed verifier (dynamic startup selection).
    ///
    /// Pair with [`verifier_from_env`] to pick mock vs real DCAP at runtime.
    #[must_use]
    pub fn with_verifier(
        measurements: MeasurementsBody,
        validator_hotkey: [u8; KEY_LEN],
        netuid: u16,
        verifier: Box<dyn QuoteVerifier>,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AttestInner {
                nonces: MemoryNonceStore::new(),
                book: AttestCreditBook::new(),
                measurements,
                validator_hotkey,
                netuid,
                verifier: Arc::from(verifier),
                nonce_ttl: DEFAULT_NONCE_TTL,
            })),
        }
    }

    /// Happy-path **mock** verifier (`UpToDate` + Fresh) — tests only.
    ///
    /// This accepts every structurally valid quote without running any DCAP
    /// crypto. Production must use [`verifier_from_env`] / [`with_verifier`].
    ///
    /// [`with_verifier`]: AttestState::with_verifier
    #[must_use]
    pub fn with_ok_verifier(
        measurements: MeasurementsBody,
        validator_hotkey: [u8; KEY_LEN],
        netuid: u16,
    ) -> Self {
        Self::new(
            measurements,
            validator_hotkey,
            netuid,
            MockQuoteVerifier::Ok(QuoteVerifyOk {
                tcb_status: TcbStatus::UpToDate,
                collateral: CollateralFreshness::Fresh,
            }),
        )
    }

    /// PCS-outage **mock** (`Parked` / `PcsTimeout`) — tests only.
    #[must_use]
    pub fn with_pcs_timeout(
        measurements: MeasurementsBody,
        validator_hotkey: [u8; KEY_LEN],
        netuid: u16,
    ) -> Self {
        Self::new(
            measurements,
            validator_hotkey,
            netuid,
            MockQuoteVerifier::Err(VerifierFailureKind::PcsTimeout),
        )
    }

    /// Snapshot credit for `(netuid, epoch, miner)`.
    pub async fn has_credit(&self, netuid: u16, epoch: u64, miner: [u8; KEY_LEN]) -> bool {
        let g = self.inner.lock().await;
        g.book.has_credit(&CreditKey {
            netuid,
            epoch,
            miner,
        })
    }

    /// Stored outcome if any.
    pub async fn outcome(
        &self,
        netuid: u16,
        epoch: u64,
        miner: [u8; KEY_LEN],
    ) -> Option<AttestOutcome> {
        let g = self.inner.lock().await;
        g.book
            .get(&CreditKey {
                netuid,
                epoch,
                miner,
            })
            .copied()
    }

    /// Validator hotkey bytes.
    pub async fn validator_hotkey(&self) -> [u8; KEY_LEN] {
        self.inner.lock().await.validator_hotkey
    }
}

/// Env var selecting the quote verifier backend.
pub const ENV_VERIFIER: &str = "BASE_ATTEST_VERIFIER";
/// Env var overriding the PCCS / Intel PCS collateral base URL.
pub const ENV_PCCS_URL: &str = "BASE_PCCS_URL";
/// Env var overriding the maximum collateral age, in seconds.
pub const ENV_COLLATERAL_MAX_AGE_SECS: &str = "BASE_COLLATERAL_MAX_AGE_SECS";

/// Build the quote verifier selected by the environment.
///
/// | `BASE_ATTEST_VERIFIER` | verifier |
/// |------------------------|----------|
/// | unset | `DcapQuoteVerifier` when built with `--features dcap`, else the park mock |
/// | `dcap` | `DcapQuoteVerifier` (hard park + `error!` when the feature is absent) |
/// | `pcs_timeout` / `park` | park mock (`PcsTimeout`) |
/// | `mock_ok` / `ok` | accept-everything mock — logs a loud `warn!` |
/// | anything else | park mock (`VerifierUnavailable`) |
///
/// `BASE_PCCS_URL` overrides the collateral endpoint (default Intel PCS) and
/// `BASE_COLLATERAL_MAX_AGE_SECS` the collateral cache TTL / max age.
///
/// Never fails: a verifier that cannot be constructed degrades to a park mock
/// so the validator keeps serving without ever silently granting credit.
#[must_use]
pub fn verifier_from_env() -> Box<dyn QuoteVerifier> {
    let mode = std::env::var(ENV_VERIFIER)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match mode.as_str() {
        "" | "dcap" => dcap_verifier(),
        "pcs_timeout" | "park" => park_verifier(VerifierFailureKind::PcsTimeout),
        "mock_ok" | "ok" => {
            tracing::warn!(
                target: "attest",
                verifier = "mock_ok",
                "ATTESTATION QUOTES ARE NOT VERIFIED: {ENV_VERIFIER}=mock_ok accepts every \
                 well-formed quote as TCB UpToDate. Never use this outside tests."
            );
            Box::new(MockQuoteVerifier::Ok(QuoteVerifyOk {
                tcb_status: TcbStatus::UpToDate,
                collateral: CollateralFreshness::Fresh,
            }))
        }
        other => {
            tracing::error!(
                target: "attest",
                verifier = other,
                "unknown {ENV_VERIFIER} value; parking every quote"
            );
            park_verifier(VerifierFailureKind::Unavailable)
        }
    }
}

fn park_verifier(kind: VerifierFailureKind) -> Box<dyn QuoteVerifier> {
    Box::new(MockQuoteVerifier::Err(kind))
}

#[cfg(feature = "dcap")]
fn dcap_verifier() -> Box<dyn QuoteVerifier> {
    use attest_policy::{DcapQuoteVerifier, DEFAULT_COLLATERAL_MAX_AGE, DEFAULT_PCS_URL};

    let url = std::env::var(ENV_PCCS_URL)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_PCS_URL.to_owned());
    let max_age = std::env::var(ENV_COLLATERAL_MAX_AGE_SECS)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map_or(DEFAULT_COLLATERAL_MAX_AGE, Duration::from_secs);
    match DcapQuoteVerifier::new(&url, max_age) {
        Ok(v) => {
            tracing::info!(
                target: "attest",
                verifier = "dcap",
                pccs_url = %url,
                max_collateral_age_secs = max_age.as_secs(),
                "real Intel DCAP quote verification enabled"
            );
            Box::new(v)
        }
        Err(e) => {
            tracing::error!(
                target: "attest",
                error = %e,
                "failed to build the DCAP verifier; parking every quote"
            );
            park_verifier(VerifierFailureKind::Unavailable)
        }
    }
}

#[cfg(not(feature = "dcap"))]
fn dcap_verifier() -> Box<dyn QuoteVerifier> {
    tracing::error!(
        target: "attest",
        "DCAP verification requested but this binary was built without the \
         `dcap` feature; parking every quote"
    );
    park_verifier(VerifierFailureKind::Unavailable)
}

/// Mount attestation routes on a new router (merge into the health app).
pub fn attest_router(state: AttestState) -> Router {
    Router::new()
        .route("/v1/attest/nonce", post(issue_nonce))
        .route("/v1/attest/submit", post(submit_quote))
        .with_state(state)
}

/// `POST /v1/attest/nonce` body.
#[derive(Debug, Deserialize)]
pub struct NonceRequest {
    /// Miner hotkey (64 lowercase hex).
    pub miner_hotkey_hex: String,
    /// Epoch the quote will bind.
    pub epoch: u64,
    /// Optional netuid (defaults to service netuid).
    pub netuid: Option<u16>,
}

/// `POST /v1/attest/nonce` response.
#[derive(Debug, Serialize)]
pub struct NonceResponse {
    /// Fresh 32-byte nonce as hex.
    pub nonce_hex: String,
    /// Epoch echoed.
    pub epoch: u64,
    /// Netuid bound into D10.
    pub netuid: u16,
    /// Asking validator hotkey hex.
    pub validator_hotkey_hex: String,
    /// TTL seconds from issue.
    pub ttl_secs: u64,
}

/// `POST /v1/attest/submit` body.
#[derive(Debug, Deserialize)]
pub struct SubmitRequest {
    /// Miner hotkey hex (must match D10 binding).
    pub miner_hotkey_hex: String,
    /// Epoch.
    pub epoch: u64,
    /// Netuid.
    pub netuid: u16,
    /// Nonce hex from issue.
    pub nonce_hex: String,
    /// Quote bytes hex.
    pub quote_hex: String,
    /// Event log JSON string.
    pub event_log_json: String,
    /// Optional claimed validator hotkey (defaults to this validator).
    pub validator_hotkey_hex: Option<String>,
}

/// `POST /v1/attest/submit` response.
#[derive(Debug, Serialize)]
pub struct SubmitResponse {
    /// `verified` | `rejected` | `parked`
    pub outcome: String,
    /// Machine reason when not verified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Whether attestation credit was granted (only true for verified).
    pub grants_credit: bool,
    /// Always false under D13.
    pub carries_prior_verified: bool,
}

async fn issue_nonce(
    State(st): State<AttestState>,
    Json(body): Json<NonceRequest>,
) -> Result<Json<NonceResponse>, ApiError> {
    let _miner = parse_key_hex(&body.miner_hotkey_hex)?;
    let mut g = st.inner.lock().await;
    let netuid = body.netuid.unwrap_or(g.netuid);
    let nonce = random_nonce();
    let now = Instant::now();
    let ttl = g.nonce_ttl;
    let vhk = g.validator_hotkey;
    register_with_ttl(&mut g.nonces, nonce, now, ttl)
        .map_err(|e| ApiError::bad(format!("nonce register: {e}")))?;
    Ok(Json(NonceResponse {
        nonce_hex: hex::encode(nonce),
        epoch: body.epoch,
        netuid,
        validator_hotkey_hex: hex::encode(vhk),
        ttl_secs: ttl.as_secs(),
    }))
}

async fn submit_quote(
    State(st): State<AttestState>,
    Json(body): Json<SubmitRequest>,
) -> Result<Json<SubmitResponse>, ApiError> {
    let miner = parse_key_hex(&body.miner_hotkey_hex)?;
    let nonce = parse_key_hex(&body.nonce_hex)?;
    let quote =
        hex::decode(body.quote_hex.trim()).map_err(|e| ApiError::bad(format!("quote_hex: {e}")))?;
    let event_log = body.event_log_json.into_bytes();

    let mut g = st.inner.lock().await;
    let validator_hotkey = match body.validator_hotkey_hex.as_deref() {
        Some(h) => parse_key_hex(h)?,
        None => g.validator_hotkey,
    };

    let binding = ReportDataBinding {
        netuid: body.netuid,
        epoch: body.epoch,
        miner_pubkey: miner,
        nonce,
        validator_hotkey,
    };

    let now = Instant::now();
    let verifier = Arc::clone(&g.verifier);
    let measurements = g.measurements.clone();
    let outcome = {
        let nonces = &mut g.nonces;
        verify_submission(&mut SubmissionInput {
            measurements: &measurements,
            quote: &quote,
            event_log_json: &event_log,
            binding,
            nonces,
            now,
            verifier: &verifier,
        })
    };

    let key = CreditKey {
        netuid: body.netuid,
        epoch: body.epoch,
        miner,
    };
    // Do not downgrade an existing Verified credit (resubmit / bad retry).
    let keep_verified = matches!(g.book.get(&key), Some(AttestOutcome::Verified));
    if !keep_verified {
        g.book.record(key, outcome);
    }

    Ok(Json(outcome_response(outcome)))
}

fn outcome_response(outcome: AttestOutcome) -> SubmitResponse {
    let (label, reason) = match outcome {
        AttestOutcome::Verified => ("verified".to_owned(), None),
        AttestOutcome::Rejected { reason } => {
            ("rejected".to_owned(), Some(reject_reason_str(reason)))
        }
        AttestOutcome::Parked { reason } => ("parked".to_owned(), Some(park_reason_str(reason))),
    };
    SubmitResponse {
        outcome: label,
        reason,
        grants_credit: outcome.grants_credit(),
        carries_prior_verified: outcome.carries_prior_verified(),
    }
}

fn reject_reason_str(r: RejectReason) -> String {
    match r {
        RejectReason::EmptyAllowlist => "empty_allowlist",
        RejectReason::MeasurementNotAllowlisted => "measurement_not_allowlisted",
        RejectReason::ReportDataMismatch => "report_data_mismatch",
        RejectReason::NonceInvalid => "nonce_invalid",
        RejectReason::QuoteCryptoInvalid => "quote_crypto_invalid",
        RejectReason::TcbRevoked => "tcb_revoked",
        RejectReason::QuoteMalformed => "quote_malformed",
        RejectReason::EventLogInvalid => "event_log_invalid",
    }
    .to_owned()
}

fn park_reason_str(r: ParkReason) -> String {
    match r {
        ParkReason::PcsTimeout => "pcs_timeout",
        ParkReason::CollateralExpired => "collateral_expired",
        ParkReason::TcbOutOfDate => "tcb_out_of_date",
        ParkReason::VerifierUnavailable => "verifier_unavailable",
    }
    .to_owned()
}

fn parse_key_hex(s: &str) -> Result<[u8; KEY_LEN], ApiError> {
    let bytes = hex::decode(s.trim()).map_err(|e| ApiError::bad(format!("hex: {e}")))?;
    if bytes.len() != KEY_LEN {
        return Err(ApiError::bad(format!(
            "expected {KEY_LEN} bytes, got {}",
            bytes.len()
        )));
    }
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn random_nonce() -> [u8; KEY_LEN] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut out = [0u8; KEY_LEN];
    if getrandom_fill(&mut out) {
        return out;
    }
    let mut h = DefaultHasher::new();
    Instant::now().hash(&mut h);
    std::thread::current().id().hash(&mut h);
    let a = h.finish().to_le_bytes();
    out[..8].copy_from_slice(&a);
    let mut h2 = DefaultHasher::new();
    (out[0], out[7], std::process::id()).hash(&mut h2);
    let b = h2.finish().to_le_bytes();
    out[8..16].copy_from_slice(&b);
    for i in 16..KEY_LEN {
        let ib = u8::try_from(i % 256).unwrap_or(0);
        out[i] = out[i - 16] ^ out[i - 8].wrapping_add(ib);
    }
    out
}

fn getrandom_fill(buf: &mut [u8]) -> bool {
    use std::io::Read;
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        return f.read_exact(buf).is_ok();
    }
    false
}

struct ApiError {
    status: StatusCode,
    msg: String,
}

impl ApiError {
    fn bad(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            msg: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, self.msg).into_response()
    }
}

/// Bind an attest-only HTTP server (tests / standalone).
///
/// # Errors
///
/// Bind failures.
pub async fn spawn_attest_server(
    state: AttestState,
    listen_addr: std::net::SocketAddr,
) -> Result<
    (
        std::net::SocketAddr,
        tokio::sync::watch::Sender<bool>,
        tokio::task::JoinHandle<Result<(), std::io::Error>>,
    ),
    std::io::Error,
> {
    use tokio::net::TcpListener;
    use tokio::sync::watch;

    let app = attest_router(state);
    let listener = TcpListener::bind(listen_addr).await?;
    let addr = listener.local_addr()?;
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let join = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                loop {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                    if shutdown_rx.changed().await.is_err() {
                        break;
                    }
                }
            })
            .await
    });
    Ok((addr, shutdown_tx, join))
}
