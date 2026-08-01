//! SS58 address encoding/decoding (base58 + blake2b-512 checksum).

use crate::{KeystoreError, KEY_LEN};
use blake2::{Blake2b512, Digest};

/// SS58 network prefix used by Bittensor (Substrate generic prefix 42).
pub const BITTENSOR_SS58_PREFIX: u16 = 42;

/// Bitcoin/Substrate base58 alphabet.
const ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Domain-separation prefix hashed before the SS58 payload.
const SS58_PRE: &[u8] = b"SS58PRE";

/// Number of checksum bytes appended to an SS58 account payload.
const CHECKSUM_LEN: usize = 2;

/// Encode a 32-byte public key as an SS58 address under `prefix`.
///
/// Prefixes below 64 use the one-byte form; 64..=16383 use the two-byte form.
#[must_use]
pub fn ss58_encode(public_key: &[u8; KEY_LEN], prefix: u16) -> String {
    let mut payload = prefix_bytes(prefix);
    payload.extend_from_slice(public_key);
    payload.extend_from_slice(&checksum(&payload));
    base58_encode(&payload)
}

/// Decode an SS58 address into its 32-byte public key and network prefix.
///
/// # Errors
///
/// [`KeystoreError::InvalidAddress`] if the address contains non-base58
/// characters, has a reserved prefix byte, a payload that is not 32 bytes, or
/// a bad blake2b checksum.
pub fn ss58_decode(address: &str) -> Result<([u8; KEY_LEN], u16), KeystoreError> {
    let data = base58_decode(address)?;
    let first = *data.first().ok_or(KeystoreError::InvalidAddress)?;
    let (prefix, offset) = if first < 64 {
        (u16::from(first), 1usize)
    } else if first < 128 {
        let second = u16::from(*data.get(1).ok_or(KeystoreError::InvalidAddress)?);
        let lower = (u16::from(first) & 0b0011_1111) << 2;
        (
            lower | (second >> 6) | ((second & 0b0011_1111) << 8),
            2usize,
        )
    } else {
        return Err(KeystoreError::InvalidAddress);
    };

    let body_end = data
        .len()
        .checked_sub(CHECKSUM_LEN)
        .ok_or(KeystoreError::InvalidAddress)?;
    let body = data
        .get(offset..body_end)
        .ok_or(KeystoreError::InvalidAddress)?;
    if body.len() != KEY_LEN {
        return Err(KeystoreError::InvalidAddress);
    }
    if checksum(&data[..body_end]) != data[body_end..] {
        return Err(KeystoreError::InvalidAddress);
    }

    let mut public_key = [0u8; KEY_LEN];
    public_key.copy_from_slice(body);
    Ok((public_key, prefix))
}

fn prefix_bytes(prefix: u16) -> Vec<u8> {
    if prefix < 64 {
        vec![u8::try_from(prefix).unwrap_or(0)]
    } else {
        let ident = prefix & 0x3fff;
        let first = u8::try_from((ident & 0b0000_0000_1111_1100) >> 2).unwrap_or(0) | 0b0100_0000;
        let second = u8::try_from((ident >> 8) | ((ident & 0b11) << 6)).unwrap_or(0);
        vec![first, second]
    }
}

fn checksum(payload: &[u8]) -> [u8; CHECKSUM_LEN] {
    let mut hasher = Blake2b512::new();
    hasher.update(SS58_PRE);
    hasher.update(payload);
    let digest = hasher.finalize();
    [digest[0], digest[1]]
}

fn base58_encode(input: &[u8]) -> String {
    let mut digits: Vec<u8> = Vec::with_capacity(input.len() * 138 / 100 + 1);
    for &byte in input {
        let mut carry = u32::from(byte);
        for digit in &mut digits {
            let value = u32::from(*digit) * 256 + carry;
            *digit = u8::try_from(value % 58).unwrap_or(0);
            carry = value / 58;
        }
        while carry > 0 {
            digits.push(u8::try_from(carry % 58).unwrap_or(0));
            carry /= 58;
        }
    }

    let leading_zeros = input.iter().take_while(|&&b| b == 0).count();
    let mut out = String::with_capacity(leading_zeros + digits.len());
    for _ in 0..leading_zeros {
        out.push('1');
    }
    for digit in digits.iter().rev() {
        out.push(char::from(ALPHABET[usize::from(*digit)]));
    }
    out
}

fn base58_decode(input: &str) -> Result<Vec<u8>, KeystoreError> {
    let mut bytes: Vec<u8> = Vec::with_capacity(input.len());
    for ch in input.chars() {
        let value = u8::try_from(ch)
            .ok()
            .and_then(|b| ALPHABET.iter().position(|&a| a == b))
            .ok_or(KeystoreError::InvalidAddress)?;
        let mut carry = u32::try_from(value).unwrap_or(0);
        for byte in &mut bytes {
            let acc = u32::from(*byte) * 58 + carry;
            *byte = u8::try_from(acc & 0xff).unwrap_or(0);
            carry = acc >> 8;
        }
        while carry > 0 {
            bytes.push(u8::try_from(carry & 0xff).unwrap_or(0));
            carry >>= 8;
        }
    }

    let leading_ones = input.chars().take_while(|&c| c == '1').count();
    let mut out = vec![0u8; leading_ones];
    out.extend(bytes.iter().rev());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{base58_decode, base58_encode, ss58_decode, ss58_encode, BITTENSOR_SS58_PREFIX};
    use crate::KeystoreError;

    /// Substrate `//Alice` sr25519 account.
    const ALICE_PUBLIC: &str = "d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d";
    const ALICE_SS58: &str = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";

    fn alice() -> [u8; 32] {
        let mut pk = [0u8; 32];
        hex::decode_to_slice(ALICE_PUBLIC, &mut pk).expect("valid hex");
        pk
    }

    #[test]
    fn alice_known_pair() {
        assert_eq!(ss58_encode(&alice(), BITTENSOR_SS58_PREFIX), ALICE_SS58);
        let (pk, prefix) = ss58_decode(ALICE_SS58).expect("decodes");
        assert_eq!(pk, alice());
        assert_eq!(prefix, BITTENSOR_SS58_PREFIX);
    }

    #[test]
    fn round_trips_over_many_prefixes() {
        let pk = alice();
        for prefix in [0u16, 1, 2, 42, 63, 64, 100, 252, 1000, 16383] {
            let addr = ss58_encode(&pk, prefix);
            let (back, got) = ss58_decode(&addr).expect("decodes");
            assert_eq!(back, pk, "prefix {prefix}");
            assert_eq!(got, prefix, "prefix {prefix}");
        }
    }

    #[test]
    fn round_trips_leading_zero_public_key() {
        let pk = [0u8; 32];
        let addr = ss58_encode(&pk, BITTENSOR_SS58_PREFIX);
        assert_eq!(ss58_decode(&addr).expect("decodes"), (pk, 42));
    }

    #[test]
    fn rejects_bad_checksum() {
        let mut chars: Vec<char> = ALICE_SS58.chars().collect();
        let last = chars.len() - 1;
        chars[last] = if chars[last] == 'Y' { 'Z' } else { 'Y' };
        let corrupted: String = chars.into_iter().collect();
        assert!(matches!(
            ss58_decode(&corrupted),
            Err(KeystoreError::InvalidAddress)
        ));
    }

    #[test]
    fn rejects_non_base58_and_short_input() {
        assert!(matches!(
            ss58_decode("5Grwv0EF"),
            Err(KeystoreError::InvalidAddress)
        ));
        assert!(matches!(
            ss58_decode(""),
            Err(KeystoreError::InvalidAddress)
        ));
        assert!(matches!(
            ss58_decode("5Grwva"),
            Err(KeystoreError::InvalidAddress)
        ));
    }

    #[test]
    fn base58_handles_leading_zero_bytes() {
        for raw in [
            vec![],
            vec![0u8],
            vec![0u8, 0, 1, 2, 3],
            vec![255u8; 40],
            vec![0u8, 0, 0],
        ] {
            let encoded = base58_encode(&raw);
            assert_eq!(base58_decode(&encoded).expect("decodes"), raw);
        }
    }
}
