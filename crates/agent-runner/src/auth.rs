//! Signed dispatch authentication (todo 18).
//!
//! Orchestrator signs under [`crypto::domain::DISPATCH`] with the challenge
//! hotkey. The runner verifies against a configured trusted pubkey, enforces
//! expiry (TTL shorter than one epoch), and records single-use nonces.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use agent_dispatch::TaskDescriptorV1;
use crypto::{
    domain, sign_raw, verify_raw, CryptoError, MemoryNonceStore, NonceError, NonceStore, KEY_LEN,
    SIGNATURE_LEN,
};
use parity_scale_codec::Encode;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default max dispatch-auth TTL (must stay strictly below one epoch).
///
/// Mirrors `attest-http::DEFAULT_NONCE_TTL` (5 minutes).
pub const DEFAULT_DISPATCH_NONCE_TTL: Duration = Duration::from_secs(5 * 60);

/// JSON body for authenticated `POST /v1/task`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedDispatchRequest {
    /// Task descriptor (same fields as the unauthenticated shape).
    pub descriptor: TaskDescriptorV1,
    /// Client-generated 32-byte nonce (64 hex chars).
    pub nonce_hex: String,
    /// Exclusive expiry as unix milliseconds (bound into the signature).
    pub expires_at_unix_ms: u64,
    /// Signer public key (64 hex chars). Must match the trusted challenge key.
    pub signer_pubkey_hex: String,
    /// 64-byte sr25519 signature (128 hex chars) under [`domain::DISPATCH`].
    pub signature_hex: String,
}

/// SCALE payload bound into the dispatch signature (field order normative).
#[derive(Debug, Clone, PartialEq, Eq, Encode)]
#[codec(crate = parity_scale_codec)]
struct DispatchAuthPayloadV1 {
    nonce: [u8; KEY_LEN],
    expires_at_unix_ms: u64,
    protocol: Vec<u8>,
    challenge_id: Vec<u8>,
    scoring_version: u16,
    epoch: u64,
    miner_hotkey: [u8; KEY_LEN],
    pack_id: Vec<u8>,
    deadline_unix_ms: u64,
}

/// Auth failures (safe to surface as typed API codes; never include sig bytes).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DispatchAuthError {
    /// Missing or malformed auth fields / hex.
    #[error("unauthorized")]
    Unauthorized,
    /// Signature valid shape but expired wall-clock window.
    #[error("dispatch auth expired")]
    Expired,
    /// Nonce already consumed (replay).
    #[error("dispatch nonce replay")]
    Replay,
}

impl DispatchAuthError {
    /// Stable machine code for JSON bodies.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Unauthorized => "unauthorized",
            Self::Expired => "auth_expired",
            Self::Replay => "nonce_replay",
        }
    }
}

/// Build the canonical signing payload for a descriptor + nonce + expiry.
#[must_use]
pub fn dispatch_auth_payload(
    descriptor: &TaskDescriptorV1,
    nonce: &[u8; KEY_LEN],
    expires_at_unix_ms: u64,
) -> Result<Vec<u8>, DispatchAuthError> {
    let miner_hotkey = parse_key_hex(&descriptor.miner_hotkey_hex)?;
    let body = DispatchAuthPayloadV1 {
        nonce: *nonce,
        expires_at_unix_ms,
        protocol: descriptor.protocol.as_bytes().to_vec(),
        challenge_id: descriptor.challenge_id.as_bytes().to_vec(),
        scoring_version: descriptor.scoring_version,
        epoch: descriptor.epoch,
        miner_hotkey,
        pack_id: descriptor.pack_id.as_bytes().to_vec(),
        deadline_unix_ms: descriptor.deadline_unix_ms,
    };
    Ok(body.encode())
}

/// Sign a dispatch envelope with a challenge mini-secret (orchestrator / tests).
///
/// # Errors
///
/// Hex parse failures or crypto signing errors map to [`DispatchAuthError::Unauthorized`].
pub fn sign_dispatch_request(
    secret: &[u8; KEY_LEN],
    public: &[u8; KEY_LEN],
    descriptor: TaskDescriptorV1,
    nonce: [u8; KEY_LEN],
    expires_at_unix_ms: u64,
) -> Result<SignedDispatchRequest, DispatchAuthError> {
    let payload = dispatch_auth_payload(&descriptor, &nonce, expires_at_unix_ms)?;
    let signature = sign_raw(secret, domain::DISPATCH, &payload).map_err(crypto_to_unauth)?;
    Ok(SignedDispatchRequest {
        descriptor,
        nonce_hex: hex::encode(nonce),
        expires_at_unix_ms,
        signer_pubkey_hex: hex::encode(public),
        signature_hex: hex::encode(signature),
    })
}

/// Verify envelope against `trusted_pubkey`, enforce TTL, consume nonce once.
///
/// # Errors
///
/// See [`DispatchAuthError`]. Cryptographic failures collapse to
/// [`DispatchAuthError::Unauthorized`] (no expected-signature leakage).
pub fn verify_and_consume_dispatch(
    trusted_pubkey: &[u8; KEY_LEN],
    max_ttl: Duration,
    nonces: &mut MemoryNonceStore,
    req: &SignedDispatchRequest,
    now_unix_ms: u64,
    now_instant: Instant,
) -> Result<(), DispatchAuthError> {
    let nonce = parse_key_hex(&req.nonce_hex)?;
    let signer = parse_key_hex(&req.signer_pubkey_hex)?;
    let signature = parse_sig_hex(&req.signature_hex)?;

    if signer != *trusted_pubkey {
        return Err(DispatchAuthError::Unauthorized);
    }

    if req.expires_at_unix_ms <= now_unix_ms {
        return Err(DispatchAuthError::Expired);
    }
    let remaining_ms = req.expires_at_unix_ms.saturating_sub(now_unix_ms);
    let max_ttl_ms = u64::try_from(max_ttl.as_millis()).unwrap_or(u64::MAX);
    if remaining_ms > max_ttl_ms {
        // Expiry too far in the future — treat as unauthorized (clock / abuse).
        return Err(DispatchAuthError::Unauthorized);
    }

    let payload = dispatch_auth_payload(&req.descriptor, &nonce, req.expires_at_unix_ms)?;
    verify_raw(trusted_pubkey, domain::DISPATCH, &payload, &signature).map_err(crypto_to_unauth)?;

    let expires_at = now_instant + Duration::from_millis(remaining_ms);
    match nonces.register(nonce, expires_at) {
        Ok(()) => Ok(()),
        Err(NonceError::AlreadyPresent) => Err(DispatchAuthError::Replay),
        Err(NonceError::Unknown | NonceError::Expired) => Err(DispatchAuthError::Unauthorized),
    }
}

/// Current unix time in milliseconds.
#[must_use]
pub fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn crypto_to_unauth(_err: CryptoError) -> DispatchAuthError {
    DispatchAuthError::Unauthorized
}

fn parse_key_hex(s: &str) -> Result<[u8; KEY_LEN], DispatchAuthError> {
    let bytes = hex::decode(s).map_err(|_| DispatchAuthError::Unauthorized)?;
    if bytes.len() != KEY_LEN {
        return Err(DispatchAuthError::Unauthorized);
    }
    let mut out = [0_u8; KEY_LEN];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn parse_sig_hex(s: &str) -> Result<[u8; SIGNATURE_LEN], DispatchAuthError> {
    let bytes = hex::decode(s).map_err(|_| DispatchAuthError::Unauthorized)?;
    if bytes.len() != SIGNATURE_LEN {
        return Err(DispatchAuthError::Unauthorized);
    }
    let mut out = [0_u8; SIGNATURE_LEN];
    out.copy_from_slice(&bytes);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_dispatch::DISPATCH_PROTOCOL;
    use schnorrkel::MiniSecretKey;

    fn mini_pair(seed: u8) -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
        let mini = MiniSecretKey::from_bytes(&[seed; KEY_LEN]).expect("mini");
        let secret = mini.to_bytes();
        let public = mini
            .expand(schnorrkel::ExpansionMode::Ed25519)
            .to_public()
            .to_bytes();
        (secret, public)
    }

    fn sample_desc() -> TaskDescriptorV1 {
        TaskDescriptorV1::new(
            "agent-v1",
            2,
            7,
            "aa".repeat(32),
            "pack-fixture-001",
            9_999_999_999_999,
        )
    }

    #[test]
    fn signed_dispatch_round_trip_accepts() {
        let (sk, pk) = mini_pair(0x11);
        let desc = sample_desc();
        assert_eq!(desc.protocol, DISPATCH_PROTOCOL);
        let nonce = [0x22_u8; KEY_LEN];
        let now_ms = 1_700_000_000_000_u64;
        let exp = now_ms + 60_000;
        let req = sign_dispatch_request(&sk, &pk, desc, nonce, exp).expect("sign");
        let mut store = MemoryNonceStore::new();
        verify_and_consume_dispatch(
            &pk,
            DEFAULT_DISPATCH_NONCE_TTL,
            &mut store,
            &req,
            now_ms,
            Instant::now(),
        )
        .expect("verify");
    }

    #[test]
    fn replay_nonce_rejected() {
        let (sk, pk) = mini_pair(0x33);
        let nonce = [0x44_u8; KEY_LEN];
        let now_ms = 1_700_000_000_000_u64;
        let exp = now_ms + 60_000;
        let req = sign_dispatch_request(&sk, &pk, sample_desc(), nonce, exp).expect("sign");
        let mut store = MemoryNonceStore::new();
        verify_and_consume_dispatch(
            &pk,
            DEFAULT_DISPATCH_NONCE_TTL,
            &mut store,
            &req,
            now_ms,
            Instant::now(),
        )
        .expect("first");
        let err = verify_and_consume_dispatch(
            &pk,
            DEFAULT_DISPATCH_NONCE_TTL,
            &mut store,
            &req,
            now_ms,
            Instant::now(),
        )
        .expect_err("replay");
        assert_eq!(err, DispatchAuthError::Replay);
        assert_eq!(err.code(), "nonce_replay");
    }

    #[test]
    fn foreign_signer_rejected() {
        let (sk_foreign, _pk_foreign) = mini_pair(0x55);
        let (_sk_trust, pk_trust) = mini_pair(0x66);
        let nonce = [0x77_u8; KEY_LEN];
        let now_ms = 1_700_000_000_000_u64;
        let exp = now_ms + 60_000;
        // Sign with foreign secret but claim trusted pubkey in envelope → still fails verify
        // (signer field must match trusted; we set signer to foreign public).
        let (sk_f, pk_f) = (sk_foreign, mini_pair(0x55).1);
        let req = sign_dispatch_request(&sk_f, &pk_f, sample_desc(), nonce, exp).expect("sign");
        let mut store = MemoryNonceStore::new();
        let err = verify_and_consume_dispatch(
            &pk_trust,
            DEFAULT_DISPATCH_NONCE_TTL,
            &mut store,
            &req,
            now_ms,
            Instant::now(),
        )
        .expect_err("foreign");
        assert_eq!(err, DispatchAuthError::Unauthorized);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn stale_expiry_rejected() {
        let (sk, pk) = mini_pair(0x88);
        let nonce = [0x99_u8; KEY_LEN];
        let now_ms = 1_700_000_000_000_u64;
        let exp = now_ms; // not strictly after now
        let req = sign_dispatch_request(&sk, &pk, sample_desc(), nonce, exp).expect("sign");
        let mut store = MemoryNonceStore::new();
        let err = verify_and_consume_dispatch(
            &pk,
            DEFAULT_DISPATCH_NONCE_TTL,
            &mut store,
            &req,
            now_ms,
            Instant::now(),
        )
        .expect_err("stale");
        assert_eq!(err, DispatchAuthError::Expired);
    }

    #[test]
    fn bad_signature_is_unauthorized_not_leaky() {
        let (sk, pk) = mini_pair(0xab);
        let nonce = [0xcd_u8; KEY_LEN];
        let now_ms = 1_700_000_000_000_u64;
        let exp = now_ms + 60_000;
        let mut req = sign_dispatch_request(&sk, &pk, sample_desc(), nonce, exp).expect("sign");
        // Flip one nibble of the signature.
        let mut chars: Vec<char> = req.signature_hex.chars().collect();
        chars[0] = if chars[0] == '0' { '1' } else { '0' };
        req.signature_hex = chars.into_iter().collect();
        let mut store = MemoryNonceStore::new();
        let err = verify_and_consume_dispatch(
            &pk,
            DEFAULT_DISPATCH_NONCE_TTL,
            &mut store,
            &req,
            now_ms,
            Instant::now(),
        )
        .expect_err("bad sig");
        assert_eq!(err, DispatchAuthError::Unauthorized);
        let msg = err.to_string();
        assert!(
            !msg.contains("signature"),
            "must not leak sig detail: {msg}"
        );
        assert!(!req.signature_hex.is_empty()); // request still has it; error must not
    }
}
