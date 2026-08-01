//! sr25519 keypair derived from a BIP39 mnemonic under the Substrate scheme.

use crate::bip39::mini_secret_from_mnemonic;
use crate::ss58::{ss58_encode, BITTENSOR_SS58_PREFIX};
use crate::{KeystoreError, KEY_LEN};
use std::fmt;

/// An sr25519 keypair held as its 32-byte mini-secret plus derived public key.
///
/// The mini-secret is never rendered by [`Debug`] and is zeroed on drop. It is
/// only reachable through the explicitly named [`Sr25519Keypair::expose_mini_secret`].
pub struct Sr25519Keypair {
    mini_secret: [u8; KEY_LEN],
    public_key: [u8; KEY_LEN],
}

impl Sr25519Keypair {
    /// Build a keypair from a 32-byte mini-secret, deriving the public key.
    ///
    /// # Errors
    ///
    /// [`KeystoreError::Crypto`] if schnorrkel rejects the mini-secret.
    pub fn from_mini_secret(mini_secret: [u8; KEY_LEN]) -> Result<Self, KeystoreError> {
        let public_key = crypto::public_key_from_mini_secret(&mini_secret)?;
        Ok(Self {
            mini_secret,
            public_key,
        })
    }

    /// The 32-byte sr25519 public key (a.k.a. Substrate `accountId`).
    #[must_use]
    pub fn public_key(&self) -> &[u8; KEY_LEN] {
        &self.public_key
    }

    /// The 32-byte mini-secret. Never log or serialise the returned bytes.
    #[must_use]
    pub fn expose_mini_secret(&self) -> &[u8; KEY_LEN] {
        &self.mini_secret
    }

    /// SS58 address under the Bittensor prefix (42).
    #[must_use]
    pub fn ss58_address(&self) -> String {
        ss58_encode(&self.public_key, BITTENSOR_SS58_PREFIX)
    }

    /// SS58 address under an arbitrary network prefix.
    #[must_use]
    pub fn ss58_address_with_prefix(&self, prefix: u16) -> String {
        ss58_encode(&self.public_key, prefix)
    }
}

impl fmt::Debug for Sr25519Keypair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sr25519Keypair")
            .field("public_key", &hex::encode(self.public_key))
            .field("mini_secret", &"<redacted>")
            .finish()
    }
}

impl Drop for Sr25519Keypair {
    fn drop(&mut self) {
        self.mini_secret.fill(0);
        // black_box keeps the store observable so it is not elided as dead.
        let _ = std::hint::black_box(&self.mini_secret);
    }
}

/// Derive an sr25519 keypair from a BIP39 mnemonic under the Substrate scheme.
///
/// Pass `""` as `password` for the usual (Bittensor / `btcli`) case.
///
/// # Errors
///
/// Propagates mnemonic decoding, seed derivation and schnorrkel key errors.
pub fn keypair_from_mnemonic(
    phrase: &str,
    password: &str,
) -> Result<Sr25519Keypair, KeystoreError> {
    Sr25519Keypair::from_mini_secret(mini_secret_from_mnemonic(phrase, password)?)
}

#[cfg(test)]
mod tests {
    use super::{keypair_from_mnemonic, Sr25519Keypair};

    /// Substrate dev phrase.
    const DEV_PHRASE: &str =
        "bottom drive obey lake curtain smoke basket hold race lonely fit walk";

    /// sr25519 public key of the dev phrase (no password, no derivation path).
    ///
    /// Cross-checked against `substrateinterface.Keypair.create_from_mnemonic`,
    /// the library Bittensor itself uses.
    const DEV_PUBLIC: &str = "46ebddef8cd9bb167dc30878d7113b7e168e6f0646beffd77d69d39bad76b47a";

    /// SS58(42) rendering of [`DEV_PUBLIC`], also from `substrateinterface`.
    const DEV_SS58_42: &str = "5DfhGyQdFobKM8NsWvEeAKk5EQQgYe9AydgJ7rMB6E1EqRzV";

    #[test]
    fn dev_phrase_matches_substrate_vector() {
        let kp = keypair_from_mnemonic(DEV_PHRASE, "").expect("dev phrase derives");
        assert_eq!(hex::encode(kp.public_key()), DEV_PUBLIC);
        assert_eq!(kp.ss58_address(), DEV_SS58_42);
    }

    #[test]
    fn debug_redacts_the_secret() {
        let kp = keypair_from_mnemonic(DEV_PHRASE, "").expect("dev phrase derives");
        let rendered = format!("{kp:?}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
        assert!(
            !rendered.contains(&hex::encode(kp.expose_mini_secret())),
            "debug leaked the mini-secret"
        );
    }

    #[test]
    fn expanded_secret_matches_public_key() {
        let kp = keypair_from_mnemonic(DEV_PHRASE, "").expect("dev phrase derives");
        let derived =
            crypto::public_key_from_mini_secret(kp.expose_mini_secret()).expect("expands");
        assert_eq!(&derived, kp.public_key());
    }

    #[test]
    fn from_mini_secret_round_trips() {
        let kp = keypair_from_mnemonic(DEV_PHRASE, "").expect("dev phrase derives");
        let again = Sr25519Keypair::from_mini_secret(*kp.expose_mini_secret()).expect("rebuild");
        assert_eq!(again.public_key(), kp.public_key());
    }
}
