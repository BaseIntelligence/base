//! Detached owner signatures over `scale(version, introduced_epoch, body)`.

use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};

use crate::error::TrustRootError;
use crate::types::{ChallengesBody, MeasurementsBody};
use crypto::{domain, sign_raw, verify_raw, KEY_LEN, SIGNATURE_LEN};

/// SCALE payload signed under [`domain::TRUST_ROOT`]:
/// `scale(version: u32, introduced_epoch: u64, body_bytes: Vec<u8>)`
/// where `body_bytes` is the SCALE encoding of the typed body.
#[derive(Debug, Clone, Encode)]
struct SignPayload {
    version: u32,
    introduced_epoch: u64,
    body: Vec<u8>,
}

/// Build the exact bytes that are signed / verified.
#[must_use]
pub fn signing_payload_bytes(version: u32, introduced_epoch: u64, body_scale: &[u8]) -> Vec<u8> {
    SignPayload {
        version,
        introduced_epoch,
        body: body_scale.to_vec(),
    }
    .encode()
}

/// SCALE-encode a challenges body.
#[must_use]
pub fn encode_challenges_body(body: &ChallengesBody) -> Vec<u8> {
    body.encode()
}

/// SCALE-encode a measurements body.
#[must_use]
pub fn encode_measurements_body(body: &MeasurementsBody) -> Vec<u8> {
    body.encode()
}

/// `sha256(scale(measurements_body))` — bundle `measurements_digest`.
#[must_use]
pub fn measurements_digest(body: &MeasurementsBody) -> [u8; 32] {
    let encoded = encode_measurements_body(body);
    let hash = Sha256::digest(encoded);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hash);
    out
}

/// Sign a trust-root payload with a raw 32-byte owner mini-secret.
///
/// # Errors
///
/// Propagates [`crypto`] key errors.
pub fn sign_trust_root_raw(
    secret_bytes: &[u8; KEY_LEN],
    version: u32,
    introduced_epoch: u64,
    body_scale: &[u8],
) -> Result<[u8; SIGNATURE_LEN], TrustRootError> {
    let payload = signing_payload_bytes(version, introduced_epoch, body_scale);
    Ok(sign_raw(secret_bytes, domain::TRUST_ROOT, &payload)?)
}

/// Alias kept for ceremony docs / callers.
///
/// # Errors
///
/// Same as [`sign_trust_root_raw`].
pub fn sign_trust_root(
    secret_bytes: &[u8; KEY_LEN],
    version: u32,
    introduced_epoch: u64,
    body_scale: &[u8],
) -> Result<[u8; SIGNATURE_LEN], TrustRootError> {
    sign_trust_root_raw(secret_bytes, version, introduced_epoch, body_scale)
}

/// Verify a detached owner signature over a trust-root payload.
///
/// # Errors
///
/// [`TrustRootError::SignatureRejected`] or crypto encoding errors.
pub fn verify_trust_root_raw(
    owner_public_bytes: &[u8; KEY_LEN],
    version: u32,
    introduced_epoch: u64,
    body_scale: &[u8],
    signature: &[u8],
) -> Result<(), TrustRootError> {
    let payload = signing_payload_bytes(version, introduced_epoch, body_scale);
    verify_raw(owner_public_bytes, domain::TRUST_ROOT, &payload, signature).map_err(|e| {
        TrustRootError::SignatureRejected {
            reason: e.to_string(),
        }
    })
}

/// Alias for [`verify_trust_root_raw`].
///
/// # Errors
///
/// Same as [`verify_trust_root_raw`].
pub fn verify_trust_root(
    owner_public_bytes: &[u8; KEY_LEN],
    version: u32,
    introduced_epoch: u64,
    body_scale: &[u8],
    signature: &[u8],
) -> Result<(), TrustRootError> {
    verify_trust_root_raw(
        owner_public_bytes,
        version,
        introduced_epoch,
        body_scale,
        signature,
    )
}

/// Parse a signature from hex (optional `0x`) or raw 64 bytes.
///
/// # Errors
///
/// [`TrustRootError::InvalidEncoding`] / unsigned.
pub fn parse_signature_bytes(raw: &[u8]) -> Result<[u8; SIGNATURE_LEN], TrustRootError> {
    let trimmed = trim_ascii_whitespace(raw);
    if trimmed.is_empty() {
        return Err(TrustRootError::Unsigned {
            path: std::path::PathBuf::from("<sig>"),
        });
    }
    if trimmed.len() == SIGNATURE_LEN {
        let mut out = [0u8; SIGNATURE_LEN];
        out.copy_from_slice(trimmed);
        return Ok(out);
    }
    let text = std::str::from_utf8(trimmed)
        .map_err(|e| TrustRootError::InvalidEncoding(format!("signature not utf-8 hex: {e}")))?;
    crate::hexutil::decode_hex_array::<SIGNATURE_LEN>(text)
}

fn trim_ascii_whitespace(raw: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = raw.len();
    while start < end && raw[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && raw[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &raw[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::OsRng;
    use schnorrkel::MiniSecretKey;

    #[test]
    fn sign_verify_round_trip() {
        let mini = MiniSecretKey::generate_with(OsRng);
        let secret_bytes = mini.to_bytes();
        let public = mini
            .expand(schnorrkel::ExpansionMode::Ed25519)
            .to_public()
            .to_bytes();
        let body = ChallengesBody::default();
        let body_scale = encode_challenges_body(&body);
        let sig = sign_trust_root_raw(&secret_bytes, 1, 0, &body_scale).unwrap();
        verify_trust_root_raw(&public, 1, 0, &body_scale, &sig).unwrap();
    }
}
