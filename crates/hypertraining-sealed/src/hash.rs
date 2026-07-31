//! SHA-256 helpers (lowercase hex, no external hex crate).

use sha2::{Digest, Sha256};

use crate::error::AdmitError;

/// SHA-256 of `bytes` as lowercase hex (64 chars).
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    encode_hex(&digest)
}

/// Encode raw bytes as lowercase hex.
#[must_use]
pub fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

/// Return `Ok(())` when `expected_hex` equals `sha256_hex(bytes)` (case-insensitive hex).
///
/// # Errors
///
/// [`AdmitError::InvalidDigest`] when `expected_hex` is not 64 hex characters.
pub fn verify_hex_sha256(
    bytes: &[u8],
    expected_hex: &str,
    context: &str,
) -> Result<(), AdmitError> {
    let expected = normalize_hex(expected_hex).map_err(|detail| AdmitError::InvalidDigest {
        context: context.to_owned(),
        detail,
    })?;
    let actual = sha256_hex(bytes);
    if actual == expected {
        Ok(())
    } else {
        Err(AdmitError::DenylistHashMismatch {
            path: context.to_owned(),
        })
    }
}

fn normalize_hex(s: &str) -> Result<String, String> {
    let raw = s.trim();
    let hex = raw
        .strip_prefix("0x")
        .or_else(|| raw.strip_prefix("0X"))
        .unwrap_or(raw);
    if hex.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", hex.len()));
    }
    let mut out = String::with_capacity(64);
    for b in hex.bytes() {
        let c = match b {
            b'0'..=b'9' | b'a'..=b'f' => b as char,
            b'A'..=b'F' => (b + 32) as char,
            _ => return Err(format!("non-hex byte {b}")),
        };
        out.push(c);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn sha256_empty_known_vector() {
        // echo -n '' | sha256sum
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
