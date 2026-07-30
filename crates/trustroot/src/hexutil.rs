//! Minimal hex helpers (no external hex crate).

use crate::error::TrustRootError;

/// Decode a hex string (optional `0x` prefix) into a fixed-size array.
///
/// # Errors
///
/// Returns [`TrustRootError::InvalidEncoding`] on bad length or non-hex input.
pub fn decode_hex_array<const N: usize>(s: &str) -> Result<[u8; N], TrustRootError> {
    let raw = s.trim();
    let hex = raw
        .strip_prefix("0x")
        .or_else(|| raw.strip_prefix("0X"))
        .unwrap_or(raw);
    if hex.len() != N * 2 {
        return Err(TrustRootError::InvalidEncoding(format!(
            "expected {} hex chars, got {}",
            N * 2,
            hex.len()
        )));
    }
    let mut out = [0u8; N];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

/// Encode bytes as lowercase hex without a `0x` prefix.
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

fn hex_nibble(c: u8) -> Result<u8, TrustRootError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(TrustRootError::InvalidEncoding(format!("non-hex byte {c}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_32() {
        let b = [0xabu8; 32];
        let h = encode_hex(&b);
        assert_eq!(decode_hex_array::<32>(&h).unwrap(), b);
        assert_eq!(decode_hex_array::<32>(&format!("0x{h}")).unwrap(), b);
    }
}
