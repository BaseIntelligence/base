//! SCALE-encoded, domain-separated work receipt (R3).
//!
//! # Who verifies
//! The **challenge service** verifies receipts against the receipt public key
//! published in the measured miner compose. Validators never re-score agents
//! (D19) and do not need this key or domain tag in the trust-root ceremony.
//!
//! # Signing
//! `sr25519` over [`crypto::domain::WORK_RECEIPT`] (`gbase-agent-work-receipt-v1`)
//! and `scale(WorkReceiptBodyV1)`. Distinct from D10 `gbase-attest-v1`.

use crypto::{domain, sign_raw, verify_raw, CryptoError, DomainTag, KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::{Decode, Encode};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Domain tag for work-receipt signatures.
#[must_use]
pub const fn work_receipt_domain() -> DomainTag {
    domain::WORK_RECEIPT
}

/// Unsigned receipt body (field order is normative).
///
/// Covers: `challenge_id`, `scoring_version`, `epoch`, `miner_hotkey`,
/// `pack_id`, `sha256(model.patch)`.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
pub struct WorkReceiptBodyV1 {
    /// Challenge id bytes (e.g. `b"agent-v1"`).
    pub challenge_id: Vec<u8>,
    /// Challenge scoring version bound into the receipt.
    pub scoring_version: u16,
    /// Epoch index.
    pub epoch: u64,
    /// Miner hotkey / pubkey.
    pub miner_hotkey: [u8; KEY_LEN],
    /// Pack id UTF-8 bytes.
    pub pack_id: Vec<u8>,
    /// `sha256(model.patch)` raw bytes.
    pub patch_sha256: [u8; 32],
}

/// Body plus sr25519 signature under [`work_receipt_domain`].
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
pub struct SignedWorkReceiptV1 {
    /// Signed fields.
    pub body: WorkReceiptBodyV1,
    /// 64-byte schnorrkel signature.
    pub signature: [u8; SIGNATURE_LEN],
}

/// Receipt encode / sign / verify failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReceiptError {
    /// Cryptographic failure (bad key, bad sig encoding, or verify fail).
    #[error("receipt crypto: {0}")]
    Crypto(String),
}

impl From<CryptoError> for ReceiptError {
    fn from(err: CryptoError) -> Self {
        Self::Crypto(err.to_string())
    }
}

/// SHA-256 of `model.patch` bytes.
#[must_use]
pub fn patch_sha256(model_patch: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(model_patch);
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// SCALE bytes of the receipt body (signature payload).
#[must_use]
pub fn receipt_payload(body: &WorkReceiptBodyV1) -> Vec<u8> {
    body.encode()
}

/// Sign a receipt body with a CVM-local mini-secret.
///
/// # Errors
///
/// Propagates key / signing failures.
pub fn sign_work_receipt(
    secret: &[u8; KEY_LEN],
    body: WorkReceiptBodyV1,
) -> Result<SignedWorkReceiptV1, ReceiptError> {
    let payload = receipt_payload(&body);
    let signature = sign_raw(secret, work_receipt_domain(), &payload)?;
    Ok(SignedWorkReceiptV1 { body, signature })
}

/// Verify a signed receipt against a published CVM receipt public key.
///
/// # Errors
///
/// [`ReceiptError::Crypto`] when the signature does not verify under
/// [`work_receipt_domain`].
pub fn verify_work_receipt(
    public: &[u8; KEY_LEN],
    signed: &SignedWorkReceiptV1,
) -> Result<(), ReceiptError> {
    let payload = receipt_payload(&signed.body);
    verify_raw(
        public,
        work_receipt_domain(),
        &payload,
        &signed.signature,
    )?;
    Ok(())
}
