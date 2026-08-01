//! Substrate storage key encoding (Twox128/64) and SCALE decode helpers.

use chain::{ChainError, Metagraph};
use parity_scale_codec::Decode;
use twox_hash::XxHash64;

/// Twox128: two `XxHash64` (seeds 0 and 1) LE-concatenated → 16 bytes.
fn twox128(data: &[u8]) -> [u8; 16] {
    let mut out = [0_u8; 16];
    out[..8].copy_from_slice(&XxHash64::oneshot(0, data).to_le_bytes());
    out[8..].copy_from_slice(&XxHash64::oneshot(1, data).to_le_bytes());
    out
}

/// Twox64: `XxHash64` with seed 0, LE → 8 bytes.
fn twox64(data: &[u8]) -> [u8; 8] {
    XxHash64::oneshot(0, data).to_le_bytes()
}

/// Plain storage key: `Twox128(pallet) ++ Twox128(item)`.
#[must_use]
pub fn storage_key(pallet: &str, item: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(&twox128(pallet.as_bytes()));
    key.extend_from_slice(&twox128(item.as_bytes()));
    key
}

/// Map storage key (Twox64Concat): `Twox128(pallet) ++ Twox128(item) ++ Twox64(key) ++ key`.
#[must_use]
pub fn storage_map_key(pallet: &str, item: &str, key: &[u8]) -> Vec<u8> {
    let mut k = storage_key(pallet, item);
    k.extend_from_slice(&twox64(key));
    k.extend_from_slice(key);
    k
}

/// u16-keyed map key (Twox64Concat):
/// `Twox128(pallet) ++ Twox128(item) ++ Twox64(netuid_le) ++ netuid_le`.
#[must_use]
pub fn storage_map_key_u16(pallet: &str, item: &str, netuid: u16) -> Vec<u8> {
    let encoded = netuid.to_le_bytes();
    storage_map_key(pallet, item, &encoded)
}

/// Decode a SCALE-encoded `u64`.
///
/// # Errors
/// Decode failure.
pub fn decode_u64(bytes: &[u8]) -> Result<u64, ChainError> {
    u64::decode(&mut &bytes[..]).map_err(|e| ChainError::Other(format!("decode u64: {e}")))
}

/// Decode a SCALE-encoded `u16`.
///
/// # Errors
/// Decode failure.
pub fn decode_u16(bytes: &[u8]) -> Result<u16, ChainError> {
    u16::decode(&mut &bytes[..]).map_err(|e| ChainError::Other(format!("decode u16: {e}")))
}

/// Decode a SCALE-encoded `bool`.
///
/// # Errors
/// Decode failure.
pub fn decode_bool(bytes: &[u8]) -> Result<bool, ChainError> {
    bool::decode(&mut &bytes[..]).map_err(|e| ChainError::Other(format!("decode bool: {e}")))
}

/// Decode a SCALE-encoded `Vec<Vec<u8>>` (e.g. Subtensor `Keys` storage).
///
/// # Errors
/// Decode failure.
pub fn decode_vec_vec_u8(bytes: &[u8]) -> Result<Vec<Vec<u8>>, ChainError> {
    Vec::<Vec<u8>>::decode(&mut &bytes[..])
        .map_err(|e| ChainError::Other(format!("decode Vec<Vec<u8>>: {e}")))
}

/// Decode a hotkey / `AccountId32` from raw storage bytes.
///
/// Handles raw 32-byte `AccountId32`, `Option<AccountId32>` (0x01 prefix),
/// and SCALE `Vec<u8>` fallback.
///
/// # Errors
/// Decode failure.
pub fn decode_hotkey(bytes: &[u8]) -> Result<Vec<u8>, ChainError> {
    match bytes.len() {
        32 => Ok(bytes.to_vec()),
        33 if bytes[0] == 0x01 => Ok(bytes[1..].to_vec()),
        _ => Vec::<u8>::decode(&mut &bytes[..])
            .map_err(|e| ChainError::Other(format!("decode hotkey: {e}"))),
    }
}

/// Build a [`Metagraph`] from decoded storage values.
#[must_use]
pub fn decode_metagraph(keys: Vec<Vec<u8>>, owner: Vec<u8>, netuid: u16) -> Metagraph {
    Metagraph {
        netuid,
        hotkeys: keys,
        owner_hotkey: owner,
    }
}
