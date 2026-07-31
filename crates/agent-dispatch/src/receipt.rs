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
    verify_raw(public, work_receipt_domain(), &payload, &signed.signature)?;
    Ok(())
}

/// Expected bind fields for challenge-side intake (I5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedReceiptBind {
    /// Challenge id string (UTF-8 body bytes).
    pub challenge_id: String,
    /// Scoring version.
    pub scoring_version: u16,
    /// Epoch.
    pub epoch: u64,
    /// Miner hotkey.
    pub miner_hotkey: [u8; KEY_LEN],
    /// Pack id string.
    pub pack_id: String,
    /// Pinned CVM receipt public key.
    pub cvm_receipt_pk: [u8; KEY_LEN],
}

/// Patch bytes after a successful receipt bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundPatch {
    /// `model.patch` bytes (empty when absent).
    pub model_patch: Vec<u8>,
    /// `sha256(model_patch)`.
    pub patch_sha256: [u8; 32],
}

/// Receipt bind failures (pre-grading; no Harbor compute).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReceiptBindError {
    /// sr25519 verify failed under WORK_RECEIPT.
    #[error("receipt signature invalid: {0}")]
    ReceiptSigInvalid(String),
    /// Returned patch does not match `patch_sha256_hex` / receipt body.
    #[error("patch_sha256 mismatch")]
    PatchHashMismatch,
    /// Epoch does not match expected bind.
    #[error("epoch mismatch: expected {expected}, got {got}")]
    EpochMismatch {
        /// Expected epoch.
        expected: u64,
        /// Result epoch.
        got: u64,
    },
    /// Miner hotkey mismatch.
    #[error("hotkey mismatch")]
    HotkeyMismatch,
    /// Pack id mismatch.
    #[error("pack_id mismatch")]
    PackMismatch,
    /// Challenge id mismatch.
    #[error("challenge_id mismatch")]
    ChallengeMismatch,
    /// Scoring version mismatch.
    #[error("scoring_version mismatch")]
    ScoringVersionMismatch,
    /// Protocol label mismatch.
    #[error("protocol mismatch")]
    ProtocolMismatch,
    /// Hex decode / length error.
    #[error("bad hex: {0}")]
    BadHex(String),
}

fn hex_arr<const N: usize>(s: &str) -> Result<[u8; N], ReceiptBindError> {
    let v = hex::decode(s).map_err(|e| ReceiptBindError::BadHex(e.to_string()))?;
    <[u8; N]>::try_from(v).map_err(|_| ReceiptBindError::BadHex("length".into()))
}

/// Verify `TaskResultV1` receipt binds before any grading (I5).
///
/// Checks protocol, expected `(epoch, miner_hotkey, pack_id, challenge_id,
/// scoring_version)`, `patch_sha256(model_patch) == patch_sha256_hex`, then
/// sr25519 under the pinned CVM receipt pubkey.
///
/// # Errors
///
/// Typed [`ReceiptBindError`] — never invokes a Harbor verifier.
pub fn verify_task_result_bind(
    exp: &ExpectedReceiptBind,
    result: &crate::wire::TaskResultV1,
) -> Result<BoundPatch, ReceiptBindError> {
    use crate::wire::DISPATCH_PROTOCOL;
    if result.protocol != DISPATCH_PROTOCOL {
        return Err(ReceiptBindError::ProtocolMismatch);
    }
    if result.challenge_id != exp.challenge_id {
        return Err(ReceiptBindError::ChallengeMismatch);
    }
    if result.scoring_version != exp.scoring_version {
        return Err(ReceiptBindError::ScoringVersionMismatch);
    }
    if result.epoch != exp.epoch {
        return Err(ReceiptBindError::EpochMismatch {
            expected: exp.epoch,
            got: result.epoch,
        });
    }
    let hotkey = hex_arr::<KEY_LEN>(&result.miner_hotkey_hex)?;
    if hotkey != exp.miner_hotkey {
        return Err(ReceiptBindError::HotkeyMismatch);
    }
    if result.pack_id != exp.pack_id {
        return Err(ReceiptBindError::PackMismatch);
    }
    let model_patch = result
        .model_patch
        .as_deref()
        .unwrap_or("")
        .as_bytes()
        .to_vec();
    let claimed = hex_arr::<32>(&result.patch_sha256_hex)?;
    let actual = patch_sha256(&model_patch);
    if actual != claimed {
        return Err(ReceiptBindError::PatchHashMismatch);
    }
    let signed = SignedWorkReceiptV1 {
        body: WorkReceiptBodyV1 {
            challenge_id: exp.challenge_id.as_bytes().to_vec(),
            scoring_version: exp.scoring_version,
            epoch: exp.epoch,
            miner_hotkey: exp.miner_hotkey,
            pack_id: exp.pack_id.as_bytes().to_vec(),
            patch_sha256: actual,
        },
        signature: hex_arr::<SIGNATURE_LEN>(&result.receipt_sig_hex)?,
    };
    verify_work_receipt(&exp.cvm_receipt_pk, &signed)
        .map_err(|e| ReceiptBindError::ReceiptSigInvalid(e.to_string()))?;
    Ok(BoundPatch {
        model_patch,
        patch_sha256: actual,
    })
}
