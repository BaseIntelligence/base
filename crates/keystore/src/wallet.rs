//! Reader for the standard Bittensor wallet layout.
//!
//! ```text
//! <wallets_dir>/<wallet>/hotkeys/<hotkey>        JSON, holds secretPhrase/secretSeed
//! <wallets_dir>/<wallet>/hotkeys/<hotkey>pub.txt JSON, public fields only
//! ```
//!
//! Every load re-derives the public key and compares it against the one stored
//! in the file, so a wrong derivation scheme fails loudly instead of silently
//! producing a stranger's identity.

use crate::bip39::mini_secret_from_mnemonic;
use crate::keypair::Sr25519Keypair;
use crate::ss58::ss58_decode;
use crate::{KeystoreError, KEY_LEN};
use serde::Deserialize;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

/// Environment variable overriding the wallets root directory.
pub const WALLETS_PATH_ENV: &str = "BT_WALLETS_PATH";

/// Names a hotkey inside the Bittensor wallet directory layout.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BittensorWallet {
    wallet_name: String,
    hotkey_name: String,
}

impl BittensorWallet {
    /// Name a `<wallet>/hotkeys/<hotkey>` pair.
    #[must_use]
    pub fn new(wallet_name: impl Into<String>, hotkey_name: impl Into<String>) -> Self {
        Self {
            wallet_name: wallet_name.into(),
            hotkey_name: hotkey_name.into(),
        }
    }

    /// The coldkey/wallet directory name.
    #[must_use]
    pub fn wallet_name(&self) -> &str {
        &self.wallet_name
    }

    /// The hotkey file name.
    #[must_use]
    pub fn hotkey_name(&self) -> &str {
        &self.hotkey_name
    }

    /// Absolute path of the hotkey secret file under `wallets_dir`.
    #[must_use]
    pub fn hotkey_path(&self, wallets_dir: &Path) -> PathBuf {
        wallets_dir
            .join(&self.wallet_name)
            .join("hotkeys")
            .join(&self.hotkey_name)
    }

    /// Load and verify this hotkey's sr25519 keypair.
    ///
    /// # Errors
    ///
    /// See [`load_hotkey`].
    pub fn load(&self, wallets_dir: &Path) -> Result<Sr25519Keypair, KeystoreError> {
        load_hotkey(wallets_dir, &self.wallet_name, &self.hotkey_name)
    }
}

/// Resolve the wallets root: `$BT_WALLETS_PATH`, else `$HOME/.bittensor/wallets`.
///
/// Falls back to the relative `.bittensor/wallets` when `HOME` is unset.
#[must_use]
pub fn default_wallets_dir() -> PathBuf {
    resolve_wallets_dir(
        std::env::var_os(WALLETS_PATH_ENV).as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

fn resolve_wallets_dir(env_override: Option<&OsStr>, home: Option<&OsStr>) -> PathBuf {
    if let Some(dir) = env_override {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    let home = home.map_or_else(PathBuf::new, PathBuf::from);
    home.join(".bittensor").join("wallets")
}

/// Public-only view of a hotkey, read from `<hotkey>pub.txt`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HotkeyPublic {
    /// 32-byte sr25519 public key.
    pub public_key: [u8; KEY_LEN],
    /// SS58 address recorded in the file, if present.
    pub ss58_address: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HotkeyFile {
    secret_phrase: Option<String>,
    secret_seed: Option<String>,
    public_key: Option<String>,
    account_id: Option<String>,
    ss58_address: Option<String>,
}

impl HotkeyFile {
    /// The public key recorded in the file, from whichever field carries it.
    ///
    /// `btcli` masks `publicKey` in `*pub.txt` files, so `accountId` and the
    /// SS58 address are consulted as equally authoritative fallbacks.
    fn recorded_public_key(&self, path: &Path) -> Result<[u8; KEY_LEN], KeystoreError> {
        if let Some(hex_str) = self.public_key.as_deref() {
            if let Ok(bytes) = decode_key_hex(hex_str, "publicKey") {
                return Ok(bytes);
            }
        }
        if let Some(hex_str) = self.account_id.as_deref() {
            if let Ok(bytes) = decode_key_hex(hex_str, "accountId") {
                return Ok(bytes);
            }
        }
        if let Some(address) = self.ss58_address.as_deref() {
            return ss58_decode(address).map(|(pk, _)| pk);
        }
        Err(KeystoreError::MissingPublicKey {
            path: path.to_path_buf(),
        })
    }
}

/// Load a hotkey keypair from `<wallets_dir>/<wallet>/hotkeys/<hotkey>`.
///
/// `secretPhrase` is preferred; `secretSeed` (32-byte hex) is the fallback.
/// The derived public key is checked against the file's recorded public key.
///
/// # Errors
///
/// - [`KeystoreError::Io`] if the file cannot be read
/// - [`KeystoreError::Json`] if it is not a JSON object
/// - [`KeystoreError::MissingSecret`] if it has neither `secretPhrase` nor `secretSeed`
/// - [`KeystoreError::InvalidHex`] if `secretSeed` is not 32-byte hex
/// - [`KeystoreError::MissingPublicKey`] if there is nothing to verify against
/// - [`KeystoreError::PublicKeyMismatch`] if the derived key differs from the stored one
/// - mnemonic / seed derivation errors from [`mini_secret_from_mnemonic`]
pub fn load_hotkey(
    wallets_dir: &Path,
    wallet: &str,
    hotkey: &str,
) -> Result<Sr25519Keypair, KeystoreError> {
    let path = BittensorWallet::new(wallet, hotkey).hotkey_path(wallets_dir);
    let parsed: HotkeyFile = read_json(&path)?;

    let mini_secret = if let Some(phrase) = parsed.secret_phrase.as_deref() {
        mini_secret_from_mnemonic(phrase, "")?
    } else if let Some(seed) = parsed.secret_seed.as_deref() {
        decode_key_hex(seed, "secretSeed")?
    } else {
        return Err(KeystoreError::MissingSecret { path });
    };

    let keypair = Sr25519Keypair::from_mini_secret(mini_secret)?;
    let recorded = parsed.recorded_public_key(&path)?;
    if &recorded != keypair.public_key() {
        return Err(KeystoreError::PublicKeyMismatch { path });
    }
    Ok(keypair)
}

/// Read the public-only companion file `<hotkey>pub.txt`.
///
/// # Errors
///
/// - [`KeystoreError::Io`] / [`KeystoreError::Json`] on read or parse failure
/// - [`KeystoreError::MissingPublicKey`] if no usable public field is present
pub fn load_hotkey_public(
    wallets_dir: &Path,
    wallet: &str,
    hotkey: &str,
) -> Result<HotkeyPublic, KeystoreError> {
    let path = BittensorWallet::new(wallet, format!("{hotkey}pub.txt")).hotkey_path(wallets_dir);
    let parsed: HotkeyFile = read_json(&path)?;
    Ok(HotkeyPublic {
        public_key: parsed.recorded_public_key(&path)?,
        ss58_address: parsed.ss58_address.clone(),
    })
}

/// Read a 32-byte mini-secret from a file holding only a mnemonic phrase.
///
/// Intended for container secret mounts. On Unix the file must not be group or
/// other readable. Neither the phrase nor any file content ever reaches the
/// returned error.
///
/// # Errors
///
/// - [`KeystoreError::Io`] if the file cannot be read or stat'ed
/// - [`KeystoreError::InsecurePermissions`] if mode & 0o077 is non-zero (Unix)
/// - mnemonic decoding / seed derivation errors
pub fn mini_secret_from_mnemonic_file(path: &Path) -> Result<[u8; KEY_LEN], KeystoreError> {
    ensure_owner_only(path)?;
    let contents = fs::read_to_string(path).map_err(|source| KeystoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    mini_secret_from_mnemonic(contents.trim(), "")
}

fn read_json(path: &Path) -> Result<HotkeyFile, KeystoreError> {
    let raw = fs::read_to_string(path).map_err(|source| KeystoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|_| KeystoreError::Json {
        path: path.to_path_buf(),
    })
}

fn decode_key_hex(value: &str, field: &'static str) -> Result<[u8; KEY_LEN], KeystoreError> {
    let err = KeystoreError::InvalidHex {
        field,
        expected: KEY_LEN,
    };
    let trimmed = value.strip_prefix("0x").unwrap_or(value);
    let mut out = [0u8; KEY_LEN];
    hex::decode_to_slice(trimmed, &mut out).map_err(|_| err)?;
    Ok(out)
}

#[cfg(unix)]
fn ensure_owner_only(path: &Path) -> Result<(), KeystoreError> {
    use std::os::unix::fs::PermissionsExt;

    let meta = fs::metadata(path).map_err(|source| KeystoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(KeystoreError::InsecurePermissions {
            path: path.to_path_buf(),
            mode,
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_owner_only(_path: &Path) -> Result<(), KeystoreError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        default_wallets_dir, load_hotkey, load_hotkey_public, mini_secret_from_mnemonic_file,
        resolve_wallets_dir, BittensorWallet,
    };
    use crate::{keypair_from_mnemonic, KeystoreError};
    use std::ffi::OsStr;
    use std::fs;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const DEV_PHRASE: &str =
        "bottom drive obey lake curtain smoke basket hold race lonely fit walk";
    const DEV_PUBLIC: &str = "0x46ebddef8cd9bb167dc30878d7113b7e168e6f0646beffd77d69d39bad76b47a";
    const DEV_SS58_42: &str = "5DfhGyQdFobKM8NsWvEeAKk5EQQgYe9AydgJ7rMB6E1EqRzV";

    /// Real wallets that must derive to these SS58 addresses when present.
    ///
    /// `base-*` is the current naming; `gbase-*` is the historical name kept so
    /// the check still runs on hosts that were provisioned before the rename.
    /// Absent wallets are skipped, so this stays green in CI.
    const REAL_WALLETS: [(&str, &str); 6] = [
        (
            "base-owner",
            "5CfjVGG7DaagMUuABNnqQJygLV2xtn3AQ7LnPeFoc5gVK9xo",
        ),
        (
            "base-validator",
            "5GKGF8GVsYvoMfgCu8hhNF2L6omvH2xkY8nmEcSUdScJqXno",
        ),
        (
            "base-miner",
            "5HGVirWmuYpmkA8EMCR3WM2vQkQM8tyrDYUSKzF3Kf4Vn3ro",
        ),
        (
            "gbase-owner",
            "5CfjVGG7DaagMUuABNnqQJygLV2xtn3AQ7LnPeFoc5gVK9xo",
        ),
        (
            "gbase-validator",
            "5GKGF8GVsYvoMfgCu8hhNF2L6omvH2xkY8nmEcSUdScJqXno",
        ),
        (
            "gbase-miner",
            "5HGVirWmuYpmkA8EMCR3WM2vQkQM8tyrDYUSKzF3Kf4Vn3ro",
        ),
    ];

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("keystore-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn write_hotkey(root: &Path, wallet: &str, hotkey: &str, body: &str) -> PathBuf {
        let dir = root.join(wallet).join("hotkeys");
        fs::create_dir_all(&dir).expect("create hotkeys dir");
        let path = dir.join(hotkey);
        fs::write(&path, body).expect("write hotkey");
        path
    }

    #[test]
    fn loads_hotkey_from_secret_phrase() {
        let root = scratch("phrase");
        write_hotkey(
            &root,
            "w",
            "default",
            &format!(
                r#"{{"secretPhrase":"{DEV_PHRASE}","publicKey":"{DEV_PUBLIC}","ss58Address":"{DEV_SS58_42}"}}"#
            ),
        );
        let kp = load_hotkey(&root, "w", "default").expect("loads");
        assert_eq!(kp.ss58_address(), DEV_SS58_42);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn loads_hotkey_from_secret_seed_fallback() {
        let root = scratch("seed");
        let kp = keypair_from_mnemonic(DEV_PHRASE, "").expect("derives");
        let seed = hex::encode(kp.expose_mini_secret());
        write_hotkey(
            &root,
            "w",
            "default",
            &format!(r#"{{"secretSeed":"0x{seed}","publicKey":"{DEV_PUBLIC}"}}"#),
        );
        let loaded = load_hotkey(&root, "w", "default").expect("loads");
        assert_eq!(loaded.public_key(), kp.public_key());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn verifies_against_ss58_only_files() {
        let root = scratch("ss58only");
        write_hotkey(
            &root,
            "w",
            "default",
            &format!(r#"{{"secretPhrase":"{DEV_PHRASE}","ss58Address":"{DEV_SS58_42}"}}"#),
        );
        assert!(load_hotkey(&root, "w", "default").is_ok());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_public_key_mismatch() {
        let root = scratch("mismatch");
        let wrong = "0x0000000000000000000000000000000000000000000000000000000000000001";
        write_hotkey(
            &root,
            "w",
            "default",
            &format!(r#"{{"secretPhrase":"{DEV_PHRASE}","publicKey":"{wrong}"}}"#),
        );
        assert!(matches!(
            load_hotkey(&root, "w", "default"),
            Err(KeystoreError::PublicKeyMismatch { .. })
        ));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_missing_secret_and_missing_public() {
        let root = scratch("missing");
        write_hotkey(&root, "w", "a", r#"{"publicKey":"0x00"}"#);
        assert!(matches!(
            load_hotkey(&root, "w", "a"),
            Err(KeystoreError::MissingSecret { .. })
        ));
        write_hotkey(
            &root,
            "w",
            "b",
            &format!(r#"{{"secretPhrase":"{DEV_PHRASE}"}}"#),
        );
        assert!(matches!(
            load_hotkey(&root, "w", "b"),
            Err(KeystoreError::MissingPublicKey { .. })
        ));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn reports_io_and_json_errors_without_contents() {
        let root = scratch("errors");
        assert!(matches!(
            load_hotkey(&root, "nope", "default"),
            Err(KeystoreError::Io { .. })
        ));
        write_hotkey(&root, "w", "default", "not json at all");
        let err = load_hotkey(&root, "w", "default").expect_err("json error");
        assert!(matches!(err, KeystoreError::Json { .. }));
        assert!(!format!("{err}").contains("not json at all"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn load_hotkey_public_reads_pub_file() {
        let root = scratch("pubfile");
        write_hotkey(
            &root,
            "w",
            "defaultpub.txt",
            &format!(
                r#"{{"accountId":"{DEV_PUBLIC}","ss58Address":"{DEV_SS58_42}","publicKey":"****"}}"#
            ),
        );
        let public = load_hotkey_public(&root, "w", "default").expect("loads");
        assert_eq!(format!("0x{}", hex::encode(public.public_key)), DEV_PUBLIC);
        assert_eq!(public.ss58_address.as_deref(), Some(DEV_SS58_42));
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn mnemonic_file_permissions_enforced() {
        use std::os::unix::fs::PermissionsExt;

        let root = scratch("mnemonicfile");
        let path = root.join("phrase.txt");
        let mut file = fs::File::create(&path).expect("create");
        writeln!(file, "  {DEV_PHRASE}  ").expect("write");
        drop(file);

        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).expect("chmod 644");
        assert!(matches!(
            mini_secret_from_mnemonic_file(&path),
            Err(KeystoreError::InsecurePermissions { .. })
        ));

        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).expect("chmod 600");
        let mini = mini_secret_from_mnemonic_file(&path).expect("loads");
        let expected = keypair_from_mnemonic(DEV_PHRASE, "").expect("derives");
        assert_eq!(&mini, expected.expose_mini_secret());

        let missing = root.join("absent.txt");
        assert!(matches!(
            mini_secret_from_mnemonic_file(&missing),
            Err(KeystoreError::Io { .. })
        ));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn wallets_dir_prefers_env_then_home() {
        let env = OsStr::new("/srv/wallets");
        let home = OsStr::new("/home/op");
        assert_eq!(
            resolve_wallets_dir(Some(env), Some(home)),
            PathBuf::from("/srv/wallets")
        );
        assert_eq!(
            resolve_wallets_dir(None, Some(home)),
            PathBuf::from("/home/op/.bittensor/wallets")
        );
        assert_eq!(
            resolve_wallets_dir(Some(OsStr::new("")), Some(home)),
            PathBuf::from("/home/op/.bittensor/wallets")
        );
        assert_eq!(
            resolve_wallets_dir(None, None),
            PathBuf::from(".bittensor/wallets")
        );
        assert!(default_wallets_dir().ends_with("wallets"));
    }

    #[test]
    fn wallet_paths_are_layout_correct() {
        let w = BittensorWallet::new("alpha", "default");
        assert_eq!(w.wallet_name(), "alpha");
        assert_eq!(w.hotkey_name(), "default");
        assert_eq!(
            w.hotkey_path(Path::new("/w")),
            PathBuf::from("/w/alpha/hotkeys/default")
        );
    }

    /// Env-gated: derives the real local wallets and checks the stored SS58.
    ///
    /// Skips silently when the wallets are absent so CI stays green.
    #[test]
    fn real_local_wallets_derive_correctly() {
        let root = default_wallets_dir();
        if !root.is_dir() {
            return;
        }
        let mut checked = 0usize;
        for (wallet, expected_ss58) in REAL_WALLETS {
            let path = BittensorWallet::new(wallet, "default").hotkey_path(&root);
            if !path.is_file() {
                continue;
            }
            let kp = load_hotkey(&root, wallet, "default")
                .unwrap_or_else(|e| panic!("{wallet}: load failed: {e}"));
            assert_eq!(kp.ss58_address(), expected_ss58, "{wallet}: ss58 mismatch");
            checked += 1;
        }
        println!("real_local_wallets_derive_correctly: verified {checked} wallet(s)");
    }
}
