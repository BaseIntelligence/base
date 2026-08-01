//! Environment-driven key resolution shared by the service binaries.
//!
//! Every role resolves its key the same way, from most to least preferred:
//!
//! 1. **Bittensor wallet** — `{PREFIX}_WALLET` (+ `{PREFIX}_WALLET_HOTKEY`,
//!    default `default`), falling back to the shared `BASE_WALLET_NAME` /
//!    `BASE_WALLET_HOTKEY`. This is the `btcli` on-disk format.
//! 2. **Mnemonic file** — `{PREFIX}_MNEMONIC_FILE`, mode 0600 or stricter.
//! 3. **Mini-secret file** — `{PREFIX}_SK_FILE`, 32 raw bytes or 64 hex chars.
//!
//! Mnemonics are deliberately **not** read from a plain environment variable:
//! process environments leak through `docker inspect`, `/proc/<pid>/environ`
//! and crash reporters.
//!
//! [`resolve_public_key_from_env`] additionally accepts a public-only
//! `{PREFIX}_HOTKEY` (64 hex chars or an SS58 address) for deployments that
//! only need to *name* a key rather than sign with it.

use std::path::{Path, PathBuf};

use crate::{keypair::Sr25519Keypair, ss58_decode, wallet, KeystoreError, KEY_LEN};

/// Default hotkey name inside a Bittensor wallet.
pub const DEFAULT_HOTKEY_NAME: &str = "default";
/// Shared wallet-name variable used when a role sets no override.
pub const SHARED_WALLET_ENV: &str = "BASE_WALLET_NAME";
/// Shared hotkey-name variable used when a role sets no override.
pub const SHARED_WALLET_HOTKEY_ENV: &str = "BASE_WALLET_HOTKEY";

fn var(name: &str) -> Option<String> {
    let v = std::env::var(name).ok()?;
    let t = v.trim().to_owned();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

/// Resolve the wallet/hotkey names configured for `prefix`, if any.
fn wallet_names(prefix: &str) -> Option<(String, String)> {
    let wallet = var(&format!("{prefix}_WALLET")).or_else(|| var(SHARED_WALLET_ENV))?;
    let hotkey = var(&format!("{prefix}_WALLET_HOTKEY"))
        .or_else(|| var(SHARED_WALLET_HOTKEY_ENV))
        .unwrap_or_else(|| DEFAULT_HOTKEY_NAME.to_owned());
    Some((wallet, hotkey))
}

/// Load a 32-byte mini-secret from a file of 32 raw bytes or 64 hex chars.
///
/// # Errors
/// I/O failure, or contents that are neither 32 raw bytes nor valid hex.
pub fn mini_secret_from_key_file(path: &Path) -> Result<[u8; KEY_LEN], KeystoreError> {
    let raw = std::fs::read(path).map_err(|source| KeystoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let bytes = if raw.len() == KEY_LEN {
        raw
    } else {
        let text = std::str::from_utf8(&raw).map_err(|_| KeystoreError::InvalidHex {
            field: "key file",
            expected: KEY_LEN,
        })?;
        let text = text.trim();
        let text = text.strip_prefix("0x").unwrap_or(text);
        hex::decode(text).map_err(|_| KeystoreError::InvalidHex {
            field: "key file",
            expected: KEY_LEN,
        })?
    };
    let arr: [u8; KEY_LEN] = bytes.try_into().map_err(|_| KeystoreError::InvalidHex {
        field: "key file",
        expected: KEY_LEN,
    })?;
    Ok(arr)
}

/// Resolve a signing keypair for `prefix` (e.g. `"BASE_GATEWAY"`).
///
/// Returns `Ok(None)` when the role configures no key at all, so callers can
/// distinguish "unset" from "set but broken".
///
/// # Errors
/// Propagates wallet, mnemonic and key-file failures. A configured source that
/// fails to load is an error, never a silent fallback to the next source.
pub fn resolve_keypair_from_env(prefix: &str) -> Result<Option<Sr25519Keypair>, KeystoreError> {
    if let Some((name, hotkey)) = wallet_names(prefix) {
        let dir = wallet::default_wallets_dir();
        return wallet::load_hotkey(&dir, &name, &hotkey).map(Some);
    }
    if let Some(p) = var(&format!("{prefix}_MNEMONIC_FILE")) {
        let mini = wallet::mini_secret_from_mnemonic_file(&PathBuf::from(p))?;
        return Sr25519Keypair::from_mini_secret(mini).map(Some);
    }
    if let Some(p) = var(&format!("{prefix}_SK_FILE")) {
        let mini = mini_secret_from_key_file(&PathBuf::from(p))?;
        return Sr25519Keypair::from_mini_secret(mini).map(Some);
    }
    Ok(None)
}

/// Resolve only the 32-byte public key for `prefix`.
///
/// Tries [`resolve_keypair_from_env`] first, then a public-only
/// `{PREFIX}_HOTKEY` holding 64 hex chars or an SS58 address.
///
/// # Errors
/// Propagates keypair resolution failures, or a malformed `{PREFIX}_HOTKEY`.
pub fn resolve_public_key_from_env(prefix: &str) -> Result<Option<[u8; KEY_LEN]>, KeystoreError> {
    if let Some(kp) = resolve_keypair_from_env(prefix)? {
        return Ok(Some(*kp.public_key()));
    }
    let Some(raw) = var(&format!("{prefix}_HOTKEY")) else {
        return Ok(None);
    };
    parse_public_key(&raw).map(Some)
}

/// Parse a public key given as 64 hex chars (optional `0x`) or an SS58 address.
///
/// # Errors
/// [`KeystoreError::InvalidAddress`] when the input is neither form.
pub fn parse_public_key(raw: &str) -> Result<[u8; KEY_LEN], KeystoreError> {
    let s = raw.trim();
    let hexed = s.strip_prefix("0x").unwrap_or(s);
    if hexed.len() == KEY_LEN * 2 {
        if let Ok(bytes) = hex::decode(hexed) {
            if let Ok(arr) = <[u8; KEY_LEN]>::try_from(bytes) {
                return Ok(arr);
            }
        }
    }
    ss58_decode(s).map(|(pk, _prefix)| pk)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Alice's well-known public key and its SS58 form.
    const ALICE_HEX: &str = "d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d";
    const ALICE_SS58: &str = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";

    #[test]
    fn parse_public_key_accepts_hex_and_ss58() {
        let from_hex = parse_public_key(ALICE_HEX).expect("hex");
        let from_prefixed = parse_public_key(&format!("0x{ALICE_HEX}")).expect("0x hex");
        let from_ss58 = parse_public_key(ALICE_SS58).expect("ss58");
        assert_eq!(from_hex, from_ss58);
        assert_eq!(from_hex, from_prefixed);
    }

    #[test]
    fn parse_public_key_rejects_garbage() {
        assert!(parse_public_key("not-a-key").is_err());
        assert!(parse_public_key("").is_err());
        // Right length, invalid hex characters.
        assert!(parse_public_key(&"z".repeat(64)).is_err());
    }

    #[test]
    fn key_file_accepts_hex_and_raw() {
        let dir = std::env::temp_dir().join(format!("keystore-env-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");

        let hex_path = dir.join("hex");
        std::fs::write(&hex_path, format!("0x{ALICE_HEX}\n")).expect("write hex");
        let raw_path = dir.join("raw");
        let raw_bytes = hex::decode(ALICE_HEX).expect("decode");
        std::fs::write(&raw_path, &raw_bytes).expect("write raw");

        let a = mini_secret_from_key_file(&hex_path).expect("hex file");
        let b = mini_secret_from_key_file(&raw_path).expect("raw file");
        assert_eq!(a, b);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn key_file_rejects_wrong_length() {
        let dir = std::env::temp_dir().join(format!("keystore-env-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let p = dir.join("short");
        std::fs::write(&p, "abcd").expect("write");
        assert!(mini_secret_from_key_file(&p).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
