//! Domain-separated sr25519 signing and single-use nonce store for base consensus.
//!
//! # Signature preimage
//!
//! Messages are signed as sr25519 (schnorrkel) over a fixed signing context
//! [`SIGNING_CONTEXT`] and the SCALE-concatenated preimage:
//!
//! ```text
//! preimage = scale(domain_tag_bytes) || scale(payload_bytes)
//! ```
//!
//! where `scale(bytes)` is the standard SCALE encoding of a byte sequence
//! (`Compact(len)` + raw bytes). Domain tags are the length-prefixed const
//! ASCII labels in [`domain`].
//!
//! Cross-domain replay is rejected because the domain tag is inside the
//! preimage; verifying under a different [`DomainTag`] yields a different
//! preimage and fails signature checks.

#![forbid(unsafe_code)]

use parity_scale_codec::Encode;
use schnorrkel::{PublicKey, SecretKey, Signature, SignatureError};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Fixed schnorrkel signing context (domain separation of the transcript itself).
///
/// Payload domain tags ([`DomainTag`]) are still bound inside the preimage.
pub const SIGNING_CONTEXT: &[u8] = b"base-sr25519-v1";

/// Length of a raw sr25519 public key / secret seed / nonce (bytes).
pub const KEY_LEN: usize = 32;

/// Length of a schnorrkel signature (bytes).
pub const SIGNATURE_LEN: usize = 64;

/// Consensus domain-separation tags (const ASCII labels).
///
/// Each tag is SCALE-encoded as a byte sequence into the signature preimage.
pub mod domain {
    use super::DomainTag;

    /// Epoch bundle body signatures.
    pub const BUNDLE: DomainTag = DomainTag::new(b"base-bundle-v1");
    /// Raw weight snapshot signatures.
    pub const RAW_WEIGHT: DomainTag = DomainTag::new(b"base-rawweight-v1");
    /// Dissent statement signatures.
    pub const DISSENT: DomainTag = DomainTag::new(b"base-dissent-v1");
    /// Peer merkle-root statement signatures.
    pub const ROOT: DomainTag = DomainTag::new(b"base-root-v1");
    /// Attestation / `report_data` binding material.
    pub const ATTEST: DomainTag = DomainTag::new(b"base-attest-v1");
    /// Owner-signed trust-root bodies.
    pub const TRUST_ROOT: DomainTag = DomainTag::new(b"base-trustroot-v1");
    /// CVM-local agent work receipt (orchestrator verifies; not D10).
    pub const WORK_RECEIPT: DomainTag = DomainTag::new(b"base-agent-work-receipt-v1");
    /// Orchestrator → runner signed dispatch (single-use nonce envelope).
    pub const DISPATCH: DomainTag = DomainTag::new(b"base-agent-dispatch-v1");
    /// Miner-signed announcement of its public CVM base URL.
    pub const MINER_ENDPOINT: DomainTag = DomainTag::new(b"base-miner-endpoint-v1");

    /// All canonical tags (stable order for tests / enumeration).
    pub const ALL: [DomainTag; 9] = [
        BUNDLE,
        RAW_WEIGHT,
        DISSENT,
        ROOT,
        ATTEST,
        TRUST_ROOT,
        WORK_RECEIPT,
        DISPATCH,
        MINER_ENDPOINT,
    ];
}

/// A length-known const domain tag used in signature preimages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainTag {
    bytes: &'static [u8],
}

impl DomainTag {
    /// Construct a domain tag from a static byte label.
    #[must_use]
    pub const fn new(bytes: &'static [u8]) -> Self {
        Self { bytes }
    }

    /// Raw ASCII label bytes (without SCALE length prefix).
    #[must_use]
    pub const fn as_bytes(self) -> &'static [u8] {
        self.bytes
    }
}

/// Build the canonical signing preimage:
/// `scale(domain_tag_bytes) || scale(payload)`.
#[must_use]
pub fn signing_preimage(domain: DomainTag, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // SCALE byte-sequence: Compact(len) ‖ bytes
    domain.as_bytes().encode_to(&mut out);
    payload.encode_to(&mut out);
    out
}

/// Constant-time equality for equal-length byte slices (bearer hashes, etc.).
///
/// `==` short-circuits on the first differing byte. Length is not secret, so
/// a length mismatch may return early.
#[must_use]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Errors from sign / verify.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// Public key bytes were not a valid sr25519 key.
    #[error("invalid public key")]
    InvalidPublicKey,
    /// Secret key bytes were not a valid sr25519 secret.
    #[error("invalid secret key")]
    InvalidSecretKey,
    /// Signature bytes were truncated, malformed, or failed verification.
    #[error("invalid signature")]
    InvalidSignature,
    /// Signature does not match the preimage under the given public key.
    #[error("signature verification failed")]
    VerificationFailed,
}

impl From<SignatureError> for CryptoError {
    fn from(_err: SignatureError) -> Self {
        Self::InvalidSignature
    }
}

/// Expand a 32-byte mini-secret into a schnorrkel [`SecretKey`].
///
/// # Errors
///
/// Returns [`CryptoError::InvalidSecretKey`] if the bytes are not a valid mini-secret.
pub fn secret_from_bytes(bytes: &[u8; KEY_LEN]) -> Result<SecretKey, CryptoError> {
    let mini =
        schnorrkel::MiniSecretKey::from_bytes(bytes).map_err(|_| CryptoError::InvalidSecretKey)?;
    Ok(mini.expand(schnorrkel::ExpansionMode::Ed25519))
}

/// Parse a 32-byte compressed public key.
///
/// # Errors
///
/// Returns [`CryptoError::InvalidPublicKey`] on malformed input.
pub fn public_from_bytes(bytes: &[u8; KEY_LEN]) -> Result<PublicKey, CryptoError> {
    PublicKey::from_bytes(bytes).map_err(|_| CryptoError::InvalidPublicKey)
}

/// Sign `payload` under `domain` with a secret key.
///
/// # Errors
///
/// Returns [`CryptoError::InvalidSecretKey`] if `secret` is malformed.
pub fn sign(
    secret: &SecretKey,
    domain: DomainTag,
    payload: &[u8],
) -> Result<[u8; SIGNATURE_LEN], CryptoError> {
    let preimage = signing_preimage(domain, payload);
    let transcript = schnorrkel::signing_context(SIGNING_CONTEXT).bytes(&preimage);
    let keypair = secret.clone().to_keypair();
    let sig = keypair.sign(transcript);
    Ok(sig.to_bytes())
}

/// Verify a signature over `payload` under `domain`.
///
/// # Errors
///
/// - [`CryptoError::InvalidPublicKey`] / [`CryptoError::InvalidSignature`] on bad encodings
/// - [`CryptoError::VerificationFailed`] if the signature does not match
pub fn verify(
    public: &PublicKey,
    domain: DomainTag,
    payload: &[u8],
    signature: &[u8],
) -> Result<(), CryptoError> {
    if signature.len() != SIGNATURE_LEN {
        return Err(CryptoError::InvalidSignature);
    }
    let sig = Signature::from_bytes(signature).map_err(|_| CryptoError::InvalidSignature)?;
    let preimage = signing_preimage(domain, payload);
    let transcript = schnorrkel::signing_context(SIGNING_CONTEXT).bytes(&preimage);
    public
        .verify(transcript, &sig)
        .map_err(|_| CryptoError::VerificationFailed)
}

/// Convenience: sign with raw 32-byte mini-secret.
///
/// # Errors
///
/// Propagates [`sign`] / key-parse errors.
pub fn sign_raw(
    secret_bytes: &[u8; KEY_LEN],
    domain: DomainTag,
    payload: &[u8],
) -> Result<[u8; SIGNATURE_LEN], CryptoError> {
    let secret = secret_from_bytes(secret_bytes)?;
    sign(&secret, domain, payload)
}

/// Convenience: verify with raw 32-byte public key and signature bytes.
///
/// # Errors
///
/// Propagates [`verify`] / key-parse errors.
pub fn verify_raw(
    public_bytes: &[u8; KEY_LEN],
    domain: DomainTag,
    payload: &[u8],
    signature: &[u8],
) -> Result<(), CryptoError> {
    let public = public_from_bytes(public_bytes)?;
    verify(&public, domain, payload, signature)
}

/// Generate a fresh 32-byte mini-secret (`OsRng`).
///
/// Suitable for CVM-local work-receipt keys and test fixtures. Never log the
/// returned bytes.
#[must_use]
pub fn generate_mini_secret() -> [u8; KEY_LEN] {
    use rand_core::OsRng;
    schnorrkel::MiniSecretKey::generate_with(OsRng).to_bytes()
}

/// Derive the 32-byte public key from a mini-secret.
///
/// # Errors
///
/// [`CryptoError::InvalidSecretKey`] if the mini-secret is malformed.
pub fn public_key_from_mini_secret(secret: &[u8; KEY_LEN]) -> Result<[u8; KEY_LEN], CryptoError> {
    let sk = secret_from_bytes(secret)?;
    Ok(sk.to_public().to_bytes())
}

/// Errors from the single-use nonce store.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NonceError {
    /// Nonce was already registered (mint collision) or already redeemed.
    #[error("nonce already present")]
    AlreadyPresent,
    /// Nonce is unknown (never issued, or already consumed and removed).
    #[error("nonce unknown")]
    Unknown,
    /// Nonce TTL has elapsed (`now >= expires_at`).
    #[error("nonce expired")]
    Expired,
}

/// Single-use 32-byte nonce store with absolute expiry.
///
/// Production backends (e.g. Postgres) implement this trait; tests use
/// [`MemoryNonceStore`]. TTL must be strictly less than one epoch at the
/// call site (this crate does not know epoch length).
pub trait NonceStore {
    /// Register `nonce` until `expires_at` (exclusive: redeem at `expires_at` fails).
    ///
    /// # Errors
    ///
    /// [`NonceError::AlreadyPresent`] if the nonce is already tracked.
    fn register(&mut self, nonce: [u8; KEY_LEN], expires_at: Instant) -> Result<(), NonceError>;

    /// Consume `nonce` at time `now`. Success removes it (not redeemable twice).
    ///
    /// # Errors
    ///
    /// - [`NonceError::Unknown`] if never registered or already redeemed
    /// - [`NonceError::Expired`] if `now >= expires_at`
    fn redeem(&mut self, nonce: &[u8; KEY_LEN], now: Instant) -> Result<(), NonceError>;
}

/// In-memory [`NonceStore`] for tests and single-process use.
///
/// Uses [`BTreeMap`] (consensus crates forbid `HashMap`).
#[derive(Debug, Default, Clone)]
pub struct MemoryNonceStore {
    /// nonce → exclusive expiry instant
    entries: BTreeMap<[u8; KEY_LEN], Instant>,
}

impl MemoryNonceStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Number of outstanding (not yet redeemed) nonces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store has no outstanding nonces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop all entries with `expires_at <= now` (housekeeping).
    pub fn purge_expired(&mut self, now: Instant) {
        self.entries.retain(|_, exp| *exp > now);
    }
}

impl NonceStore for MemoryNonceStore {
    fn register(&mut self, nonce: [u8; KEY_LEN], expires_at: Instant) -> Result<(), NonceError> {
        if self.entries.contains_key(&nonce) {
            return Err(NonceError::AlreadyPresent);
        }
        self.entries.insert(nonce, expires_at);
        Ok(())
    }

    fn redeem(&mut self, nonce: &[u8; KEY_LEN], now: Instant) -> Result<(), NonceError> {
        let Some(expires_at) = self.entries.remove(nonce) else {
            return Err(NonceError::Unknown);
        };
        if now >= expires_at {
            // Already removed; treat as expired (not re-insert — single-use).
            return Err(NonceError::Expired);
        }
        Ok(())
    }
}

/// Helper: register a nonce that lives for `ttl` from `now`.
///
/// Callers must ensure `ttl` is strictly less than one epoch.
///
/// # Errors
///
/// Propagates [`NonceStore::register`].
pub fn register_with_ttl<S: NonceStore>(
    store: &mut S,
    nonce: [u8; KEY_LEN],
    now: Instant,
    ttl: Duration,
) -> Result<(), NonceError> {
    store.register(nonce, now + ttl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::OsRng;
    use schnorrkel::Keypair;
    use std::collections::BTreeSet;

    fn fresh_keypair() -> Keypair {
        Keypair::generate_with(OsRng)
    }

    /// S1 — valid round-trip under the same domain.
    #[test]
    fn s1_sign_verify_round_trip() {
        let kp = fresh_keypair();
        let payload = b"epoch-bundle-body-v1";
        let sig = sign(&kp.secret, domain::BUNDLE, payload).expect("sign");
        verify(&kp.public, domain::BUNDLE, payload, &sig).expect("verify");
    }

    /// S1b — raw mini-secret / public key path.
    #[test]
    fn s1_sign_verify_raw_round_trip() {
        let mini = schnorrkel::MiniSecretKey::generate_with(OsRng);
        let secret_bytes = mini.to_bytes();
        let public_bytes = mini
            .expand(schnorrkel::ExpansionMode::Ed25519)
            .to_public()
            .to_bytes();
        let payload = [1u8, 2, 3, 4];
        let sig = sign_raw(&secret_bytes, domain::ROOT, &payload).expect("sign_raw");
        verify_raw(&public_bytes, domain::ROOT, &payload, &sig).expect("verify_raw");
    }

    /// S2 — cross-domain replay REJECTED.
    #[test]
    fn s2_cross_domain_replay_rejected() {
        let kp = fresh_keypair();
        let payload = b"same-payload-bytes";
        let sig = sign(&kp.secret, domain::BUNDLE, payload).expect("sign bundle");
        let err = verify(&kp.public, domain::RAW_WEIGHT, payload, &sig)
            .expect_err("cross-domain must fail");
        assert_eq!(err, CryptoError::VerificationFailed);

        // All other domains also reject a BUNDLE signature.
        for tag in domain::ALL {
            if tag == domain::BUNDLE {
                continue;
            }
            let e = verify(&kp.public, tag, payload, &sig).expect_err("cross-domain");
            assert_eq!(
                e,
                CryptoError::VerificationFailed,
                "tag {:?}",
                tag.as_bytes()
            );
        }
    }

    /// S3 — truncated / malformed signature rejected.
    #[test]
    fn s3_truncated_signature_rejected() {
        let kp = fresh_keypair();
        let payload = b"payload";
        let sig = sign(&kp.secret, domain::DISSENT, payload).expect("sign");

        let short = &sig[..SIGNATURE_LEN - 1];
        let err = verify(&kp.public, domain::DISSENT, payload, short).expect_err("truncated");
        assert_eq!(err, CryptoError::InvalidSignature);

        let empty_err = verify(&kp.public, domain::DISSENT, payload, &[]).expect_err("empty");
        assert_eq!(empty_err, CryptoError::InvalidSignature);

        // Wrong length longer than 64.
        let mut long = sig.to_vec();
        long.push(0);
        let long_err = verify(&kp.public, domain::DISSENT, payload, &long).expect_err("long");
        assert_eq!(long_err, CryptoError::InvalidSignature);

        // 64 bytes but garbage (not a valid Ristretto signature encoding / verify fail).
        let garbage = [0u8; SIGNATURE_LEN];
        let g_err = verify(&kp.public, domain::DISSENT, payload, &garbage).expect_err("garbage");
        assert!(
            matches!(
                g_err,
                CryptoError::InvalidSignature | CryptoError::VerificationFailed
            ),
            "got {g_err:?}"
        );
    }

    /// S4 — nonce not redeemable twice.
    #[test]
    fn s4_nonce_not_redeemable_twice() {
        let mut store = MemoryNonceStore::new();
        let now = Instant::now();
        let ttl = Duration::from_mins(1);
        let nonce = [7u8; KEY_LEN];

        register_with_ttl(&mut store, nonce, now, ttl).expect("register");
        store.redeem(&nonce, now).expect("first redeem");
        let err = store.redeem(&nonce, now).expect_err("second redeem");
        assert_eq!(err, NonceError::Unknown);
        assert!(store.is_empty());
    }

    /// S5 — expired nonce rejected.
    #[test]
    fn s5_expired_nonce_rejected() {
        let mut store = MemoryNonceStore::new();
        let now = Instant::now();
        let ttl = Duration::from_millis(10);
        let nonce = [9u8; KEY_LEN];

        register_with_ttl(&mut store, nonce, now, ttl).expect("register");
        // Simulate time past expiry without sleeping: redeem at expires_at boundary.
        let expires_at = now + ttl;
        let err = store
            .redeem(&nonce, expires_at)
            .expect_err("expired at boundary");
        assert_eq!(err, NonceError::Expired);

        // Already consumed by failed redeem path (removed) — second attempt Unknown.
        let err2 = store
            .redeem(&nonce, expires_at + Duration::from_secs(1))
            .expect_err("gone");
        assert_eq!(err2, NonceError::Unknown);
    }

    /// S5b — still-valid nonce before expiry succeeds.
    #[test]
    fn s5b_nonce_before_expiry_ok() {
        let mut store = MemoryNonceStore::new();
        let now = Instant::now();
        let ttl = Duration::from_hours(1);
        let nonce = [3u8; KEY_LEN];
        register_with_ttl(&mut store, nonce, now, ttl).expect("register");
        store
            .redeem(&nonce, now + Duration::from_secs(1))
            .expect("within ttl");
    }

    /// S6 — preimage construction locked (SCALE concat) + all tags distinct.
    #[test]
    fn s6_preimage_and_domain_tags() {
        let payload = b"abc";
        let pre = signing_preimage(domain::BUNDLE, payload);

        // Manual SCALE via the same Encode path as production (`&[u8]`, not `[u8; N]`).
        // Fixed-size arrays encode without a Compact length prefix.
        let tag = domain::BUNDLE.as_bytes();
        assert_eq!(tag, b"base-bundle-v1");
        let mut expected = Vec::new();
        <[u8] as Encode>::encode_to(tag, &mut expected);
        <[u8] as Encode>::encode_to(payload.as_slice(), &mut expected);
        assert_eq!(pre, expected, "preimage must be scale(tag)||scale(payload)");

        // Tag length 15 → compact single-byte prefix 15*4 = 60 = 0x3c
        let tag_len = u8::try_from(tag.len()).expect("tag len fits u8");
        assert_eq!(pre[0], tag_len << 2);
        assert_eq!(&pre[1..=tag.len()], tag);

        let mut labels = BTreeSet::new();
        for dtag in domain::ALL {
            let label = dtag.as_bytes();
            assert!(
                labels.insert(label),
                "duplicate domain tag {:?}",
                std::str::from_utf8(label)
            );
        }
        assert_eq!(labels.len(), 9);

        assert_eq!(domain::BUNDLE.as_bytes(), b"base-bundle-v1");
        assert_eq!(domain::RAW_WEIGHT.as_bytes(), b"base-rawweight-v1");
        assert_eq!(domain::DISSENT.as_bytes(), b"base-dissent-v1");
        assert_eq!(domain::ROOT.as_bytes(), b"base-root-v1");
        assert_eq!(domain::ATTEST.as_bytes(), b"base-attest-v1");
        assert_eq!(domain::TRUST_ROOT.as_bytes(), b"base-trustroot-v1");
        assert_eq!(
            domain::WORK_RECEIPT.as_bytes(),
            b"base-agent-work-receipt-v1"
        );
        assert_eq!(domain::DISPATCH.as_bytes(), b"base-agent-dispatch-v1");
        assert_eq!(domain::MINER_ENDPOINT.as_bytes(), b"base-miner-endpoint-v1");
    }

    /// S6b — different payloads under same domain do not verify.
    #[test]
    fn s6_payload_mismatch_rejected() {
        let kp = fresh_keypair();
        let sig = sign(&kp.secret, domain::ATTEST, b"one").expect("sign");
        let err = verify(&kp.public, domain::ATTEST, b"two", &sig).expect_err("payload");
        assert_eq!(err, CryptoError::VerificationFailed);
    }

    /// Register collision.
    #[test]
    fn nonce_register_duplicate_rejected() {
        let mut store = MemoryNonceStore::new();
        let now = Instant::now();
        let n = [1u8; KEY_LEN];
        store
            .register(n, now + Duration::from_secs(5))
            .expect("first");
        let err = store
            .register(n, now + Duration::from_secs(5))
            .expect_err("dup");
        assert_eq!(err, NonceError::AlreadyPresent);
    }

    /// Unknown nonce redeem.
    #[test]
    fn nonce_unknown_rejected() {
        let mut store = MemoryNonceStore::new();
        let err = store
            .redeem(&[0u8; KEY_LEN], Instant::now())
            .expect_err("unknown");
        assert_eq!(err, NonceError::Unknown);
    }

    /// `purge_expired` removes stale entries.
    #[test]
    fn nonce_purge_expired() {
        let mut store = MemoryNonceStore::new();
        let now = Instant::now();
        store
            .register([1u8; KEY_LEN], now + Duration::from_millis(1))
            .expect("r1");
        store
            .register([2u8; KEY_LEN], now + Duration::from_secs(100))
            .expect("r2");
        store.purge_expired(now + Duration::from_secs(1));
        assert_eq!(store.len(), 1);
        store
            .redeem(&[2u8; KEY_LEN], now + Duration::from_secs(1))
            .expect("still live");
    }

    /// Mini-secret expand path smoke.
    #[test]
    fn mini_secret_expand_smoke() {
        let mini = schnorrkel::MiniSecretKey::generate_with(OsRng);
        let bytes = mini.to_bytes();
        assert_eq!(bytes.len(), KEY_LEN);
        let _ = secret_from_bytes(&bytes).expect("mini expands");
    }
}
