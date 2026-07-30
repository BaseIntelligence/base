//! CVM-local work-receipt signing key (I4 / R3).
//!
//! Private half lives only as a mode-0600 file inside the measured CVM
//! (`GBASE_RECEIPT_SK_FILE`, default `/run/gbase/receipt_sk`). Never the
//! challenge signing key. Public half is published in the measured compose
//! for the challenge service to pin (D19: validators never need it).

use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use crypto::{
    generate_mini_secret, public_key_from_mini_secret, secret_from_bytes, KEY_LEN,
};
use thiserror::Error;

/// Env var naming the receipt secret file path inside the CVM.
pub const RECEIPT_SK_FILE_ENV: &str = "GBASE_RECEIPT_SK_FILE";

/// Default in-CVM path for the receipt mini-secret (matches miner template).
pub const DEFAULT_RECEIPT_SK_PATH: &str = "/run/gbase/receipt_sk";

/// Receipt key load / provision failures.
#[derive(Debug, Error)]
pub enum ReceiptKeyError {
    /// I/O failure reading or writing the secret file.
    #[error("receipt key io: {0}")]
    Io(#[from] std::io::Error),
    /// Secret file is not 32 raw bytes or 64 hex chars.
    #[error("receipt secret must be 32 raw bytes or 64 hex chars, got {0} bytes")]
    BadLength(usize),
    /// Hex decode failed.
    #[error("receipt secret hex decode: {0}")]
    Hex(String),
    /// Mini-secret expand failed.
    #[error("invalid receipt mini-secret")]
    InvalidSecret,
    /// Signing required but no key was configured / loadable.
    #[error("receipt signing key missing (set {RECEIPT_SK_FILE_ENV} or provision the mount)")]
    Missing,
}

/// Loaded CVM-local receipt key material (mini-secret + derived public key).
#[derive(Clone)]
pub struct ReceiptKey {
    secret: [u8; KEY_LEN],
    public: [u8; KEY_LEN],
    path: PathBuf,
}

impl std::fmt::Debug for ReceiptKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReceiptKey")
            .field("path", &self.path)
            .field("public_hex", &self.public_key_hex())
            .finish_non_exhaustive()
    }
}

impl ReceiptKey {
    /// Borrow the 32-byte mini-secret (never log).
    #[must_use]
    pub fn secret(&self) -> &[u8; KEY_LEN] {
        &self.secret
    }

    /// Borrow the 32-byte public key.
    #[must_use]
    pub fn public_key(&self) -> &[u8; KEY_LEN] {
        &self.public
    }

    /// Lowercase hex encoding of the public key (compose / challenge pin surface).
    #[must_use]
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.public)
    }

    /// Path the secret was loaded from / written to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Resolve the receipt secret path from env or the CVM default.
#[must_use]
pub fn receipt_sk_path_from_env() -> PathBuf {
    std::env::var_os(RECEIPT_SK_FILE_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_RECEIPT_SK_PATH))
}

/// Load a 32-byte mini-secret from `path` (raw bytes or hex text).
///
/// # Errors
///
/// See [`ReceiptKeyError`].
pub fn load_receipt_secret(path: &Path) -> Result<[u8; KEY_LEN], ReceiptKeyError> {
    let raw = fs::read(path)?;
    parse_secret_bytes(&raw)
}

fn parse_secret_bytes(raw: &[u8]) -> Result<[u8; KEY_LEN], ReceiptKeyError> {
    if raw.len() == KEY_LEN {
        let mut out = [0u8; KEY_LEN];
        out.copy_from_slice(raw);
        secret_from_bytes(&out).map_err(|_| ReceiptKeyError::InvalidSecret)?;
        return Ok(out);
    }
    let text = std::str::from_utf8(raw).map_err(|e| ReceiptKeyError::Hex(e.to_string()))?;
    let trimmed = text.trim();
    let hex_s = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    let bytes = hex::decode(hex_s).map_err(|e| ReceiptKeyError::Hex(e.to_string()))?;
    if bytes.len() != KEY_LEN {
        return Err(ReceiptKeyError::BadLength(bytes.len()));
    }
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&bytes);
    secret_from_bytes(&out).map_err(|_| ReceiptKeyError::InvalidSecret)?;
    Ok(out)
}

/// Write `secret` to `path` with mode `0600` (owner read/write only).
///
/// # Errors
///
/// I/O failures creating parent dirs or writing the file.
pub fn write_receipt_secret(path: &Path, secret: &[u8; KEY_LEN]) -> Result<(), ReceiptKeyError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true).mode(0o600);
    let mut file = opts.open(path)?;
    file.write_all(secret)?;
    file.sync_all()?;
    // Reinforce mode in case umask interfered on create.
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o600);
    fs::set_permissions(path, perms)?;
    Ok(())
}

/// Load an existing key from `path`, or generate + persist one (mode 0600).
///
/// Public key is stable across process restart when the file path is durable.
///
/// # Errors
///
/// See [`ReceiptKeyError`].
pub fn load_or_generate(path: &Path) -> Result<ReceiptKey, ReceiptKeyError> {
    let secret = if path.is_file() {
        load_receipt_secret(path)?
    } else {
        let secret = generate_mini_secret();
        write_receipt_secret(path, &secret)?;
        secret
    };
    let public =
        public_key_from_mini_secret(&secret).map_err(|_| ReceiptKeyError::InvalidSecret)?;
    Ok(ReceiptKey {
        secret,
        public,
        path: path.to_path_buf(),
    })
}

/// Load only — fail closed when the file is missing (production CVM path).
///
/// # Errors
///
/// [`ReceiptKeyError::Missing`] when the path does not exist; other load errors.
pub fn load_required(path: &Path) -> Result<ReceiptKey, ReceiptKeyError> {
    if !path.is_file() {
        return Err(ReceiptKeyError::Missing);
    }
    let secret = load_receipt_secret(path)?;
    let public =
        public_key_from_mini_secret(&secret).map_err(|_| ReceiptKeyError::InvalidSecret)?;
    Ok(ReceiptKey {
        secret,
        public,
        path: path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn load_or_generate_stable_public_key_across_reload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("receipt_sk");

        let first = load_or_generate(&path).expect("first");
        let pk1 = first.public_key_hex();
        assert_eq!(pk1.len(), 64);
        assert!(path.is_file());

        let meta = fs::metadata(&path).expect("meta");
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);

        let second = load_or_generate(&path).expect("second");
        assert_eq!(second.public_key_hex(), pk1);
        assert_eq!(second.secret(), first.secret());
    }

    #[test]
    fn load_required_fails_when_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("absent");
        let err = load_required(&path).expect_err("missing");
        assert!(matches!(err, ReceiptKeyError::Missing));
    }

    #[test]
    fn debug_fmt_does_not_contain_secret_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("receipt_sk");
        let key = load_or_generate(&path).expect("key");
        let dbg = format!("{key:?}");
        let secret_hex = hex::encode(key.secret());
        assert!(
            !dbg.contains(&secret_hex),
            "Debug must not leak secret hex: {dbg}"
        );
        assert!(dbg.contains(&key.public_key_hex()));
    }

    #[test]
    fn hex_file_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("receipt_sk.hex");
        let secret = generate_mini_secret();
        fs::write(&path, hex::encode(secret)).expect("write hex");
        let loaded = load_receipt_secret(&path).expect("load");
        assert_eq!(loaded, secret);
    }
}
