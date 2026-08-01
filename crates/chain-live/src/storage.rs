//! Substrate storage key encoding and SCALE decode helpers.
//!
//! # Hashers
//!
//! Subtensor declares its per-netuid maps with the **`Identity`** hasher, so the
//! key suffix is the raw SCALE-encoded `netuid` with no hash prefix. This was
//! verified against `wss://test.finney.opentensor.ai` by enumerating
//! `state_getKeysPaged` for `SubnetOwnerHotkey`, `Tempo`, `LastEpochBlock` and
//! `CommitRevealWeightsEnabled`: every suffix is exactly 2 bytes.
//! `Keys` is a **double map** `(netuid, uid) -> AccountId32` whose suffix is 4
//! bytes (`netuid_le ++ uid_le`), not a `Vec<AccountId32>` under a single key.
//!
//! [`storage_map_key_twox64`] is retained for pallets that do use
//! `Twox64Concat`; it is not the Subtensor convention.

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

/// Map key with the `Identity` hasher: `Twox128(pallet) ++ Twox128(item) ++ key`.
#[must_use]
pub fn storage_map_key_identity(pallet: &str, item: &str, key: &[u8]) -> Vec<u8> {
    let mut k = storage_key(pallet, item);
    k.extend_from_slice(key);
    k
}

/// Map key with the `Twox64Concat` hasher (non-Subtensor pallets).
#[must_use]
pub fn storage_map_key_twox64(pallet: &str, item: &str, key: &[u8]) -> Vec<u8> {
    let mut k = storage_key(pallet, item);
    k.extend_from_slice(&twox64(key));
    k.extend_from_slice(key);
    k
}

/// Per-netuid Subtensor map key (`Identity` hasher over the LE `u16`).
#[must_use]
pub fn storage_map_key_u16(pallet: &str, item: &str, netuid: u16) -> Vec<u8> {
    storage_map_key_identity(pallet, item, &netuid.to_le_bytes())
}

/// Partial key for a `(u16, _)` double map: everything up to and including `k1`.
///
/// Used as the `state_getKeysPaged` prefix to enumerate one netuid's entries.
#[must_use]
pub fn storage_double_map_prefix_u16(pallet: &str, item: &str, k1: u16) -> Vec<u8> {
    storage_map_key_identity(pallet, item, &k1.to_le_bytes())
}

/// Full key for a `(u16, u16)` double map with `Identity` hashers on both keys.
#[must_use]
pub fn storage_double_map_key_u16_u16(pallet: &str, item: &str, k1: u16, k2: u16) -> Vec<u8> {
    let mut k = storage_double_map_prefix_u16(pallet, item, k1);
    k.extend_from_slice(&k2.to_le_bytes());
    k
}

/// Recover the trailing `u16` (e.g. `uid`) from a `(u16, u16)` double-map key.
///
/// # Errors
/// Returns [`ChainError::Other`] when the key is shorter than the 36-byte
/// `Twox128 ++ Twox128 ++ u16 ++ u16` layout.
pub fn decode_double_map_k2(key: &[u8]) -> Result<u16, ChainError> {
    // Layout: Twox128(pallet) ++ Twox128(item) ++ k1_le_u16 ++ k2_le_u16.
    let Some(tail) = key.get(34..36) else {
        return Err(ChainError::Other(format!(
            "double-map key too short: {} bytes (want 36)",
            key.len()
        )));
    };
    let mut buf = [0_u8; 2];
    buf.copy_from_slice(tail);
    Ok(u16::from_le_bytes(buf))
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

/// Decode a SCALE-encoded `Vec<Vec<u8>>`.
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
