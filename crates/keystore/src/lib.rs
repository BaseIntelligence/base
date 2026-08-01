//! BIP39 mnemonic → sr25519 key derivation using the exact Substrate/Bittensor
//! scheme, plus a reader for the standard Bittensor wallet file format.
//!
//! # Derivation scheme
//!
//! Substrate does **not** use the standard BIP39 seed derivation (PBKDF2 over
//! the normalised mnemonic *string*). It runs PBKDF2 over the decoded BIP39
//! **entropy bytes**:
//!
//! ```text
//! entropy    = bip39_decode(phrase)                  // 16..=32 bytes
//! salt       = "mnemonic" || password
//! seed64     = PBKDF2-HMAC-SHA512(entropy, salt, 2048 rounds, 64 bytes)
//! mini_secret= seed64[0..32]
//! secret_key = MiniSecretKey(mini_secret).expand(Ed25519)
//! ```
//!
//! This mirrors `substrate_bip39::seed_from_entropy` /
//! `sp_core::sr25519::Pair::from_phrase`, which is the same path taken by
//! `substrateinterface.Keypair.create_from_mnemonic` and therefore by `btcli`.
//!
//! # Secret hygiene
//!
//! [`Sr25519Keypair`] never renders its mini-secret through [`Debug`], and
//! zeroes it on drop. No error variant in [`KeystoreError`] carries key
//! material, mnemonic words, or file contents.

#![forbid(unsafe_code)]

mod bip39;
mod env;
mod keypair;
mod ss58;
mod wallet;

pub use bip39::{
    mini_secret_from_mnemonic, mnemonic_to_entropy, substrate_seed_from_entropy, wordlist,
    PBKDF2_ROUNDS, WORDLIST_LEN,
};
pub use env::{
    mini_secret_from_key_file, parse_public_key, resolve_keypair_from_env,
    resolve_public_key_from_env, DEFAULT_HOTKEY_NAME, SHARED_WALLET_ENV, SHARED_WALLET_HOTKEY_ENV,
};
pub use keypair::{keypair_from_mnemonic, Sr25519Keypair};
pub use ss58::{ss58_decode, ss58_encode, BITTENSOR_SS58_PREFIX};
pub use wallet::{
    default_wallets_dir, load_hotkey, load_hotkey_public, mini_secret_from_mnemonic_file,
    BittensorWallet, HotkeyPublic, WALLETS_PATH_ENV,
};

use std::path::PathBuf;
use thiserror::Error;

/// Length of an sr25519 public key / mini-secret in bytes.
pub const KEY_LEN: usize = 32;

/// Errors produced by mnemonic decoding, key derivation and wallet loading.
///
/// No variant ever embeds secret material (mnemonic words, seeds, private
/// keys) or raw file contents, so these are safe to log verbatim.
#[derive(Debug, Error)]
pub enum KeystoreError {
    /// Mnemonic word count is not one of 12, 15, 18, 21, 24.
    #[error("mnemonic must have 12, 15, 18, 21 or 24 words, got {0}")]
    WordCount(usize),
    /// A word in the phrase is not present in the BIP39 English wordlist.
    ///
    /// The offending word is deliberately not reported.
    #[error("mnemonic contains a word outside the BIP39 English wordlist")]
    UnknownWord,
    /// The BIP39 checksum bits do not match `SHA256(entropy)`.
    #[error("mnemonic checksum mismatch")]
    MnemonicChecksum,
    /// Entropy length is outside 16..=32 bytes or not a multiple of 4.
    #[error("entropy length {0} is not 16..=32 bytes and a multiple of 4")]
    EntropyLength(usize),
    /// HMAC-SHA512 rejected the key material length.
    #[error("hmac initialisation failed")]
    Hmac,
    /// schnorrkel rejected a mini-secret or public key.
    #[error("sr25519 key error: {0}")]
    Crypto(#[from] crypto::CryptoError),
    /// An SS58 address is malformed, has a bad checksum, or a bad payload length.
    #[error("malformed ss58 address")]
    InvalidAddress,
    /// A hex field in a wallet file is not valid hex of the expected length.
    #[error("wallet field `{field}` is not valid {expected}-byte hex")]
    InvalidHex {
        /// JSON field name that failed to parse.
        field: &'static str,
        /// Expected decoded byte length.
        expected: usize,
    },
    /// Filesystem access failed.
    #[error("io error on {path}: {source}")]
    Io {
        /// Path that was being accessed.
        path: PathBuf,
        /// Underlying OS error.
        source: std::io::Error,
    },
    /// A wallet file is not valid JSON, or is not a JSON object.
    #[error("wallet file {path} is not valid JSON")]
    Json {
        /// Path of the offending file.
        path: PathBuf,
    },
    /// A hotkey file carries neither `secretPhrase` nor `secretSeed`.
    #[error("hotkey file {path} has no secretPhrase or secretSeed")]
    MissingSecret {
        /// Path of the hotkey file.
        path: PathBuf,
    },
    /// A hotkey file carries no `publicKey`/`accountId`/`ss58Address` to verify against.
    #[error("hotkey file {path} has no publicKey, accountId or ss58Address")]
    MissingPublicKey {
        /// Path of the hotkey file.
        path: PathBuf,
    },
    /// The derived public key does not match the one recorded in the wallet file.
    ///
    /// This is the tripwire for a wrong derivation scheme.
    #[error("derived public key does not match the one stored in {path}")]
    PublicKeyMismatch {
        /// Path of the hotkey file.
        path: PathBuf,
    },
    /// A secret file is readable by group or other.
    #[error("{path} is group/other readable (mode {mode:04o}); chmod 600 it")]
    InsecurePermissions {
        /// Path of the offending file.
        path: PathBuf,
        /// Observed Unix permission bits.
        mode: u32,
    },
}
