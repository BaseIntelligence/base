//! BIP39 mnemonic decoding and Substrate's entropy-based PBKDF2 seed derivation.

use crate::KeystoreError;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256, Sha512};
use std::sync::OnceLock;

/// Canonical BIP39 English wordlist, one word per line.
///
/// SHA-256 of this file is pinned in the unit tests, so a corrupted or
/// substituted list cannot ship.
const WORDLIST_RAW: &str = include_str!("../data/english.txt");

/// Number of words in the BIP39 English wordlist.
pub const WORDLIST_LEN: usize = 2048;

/// PBKDF2 iteration count used by `substrate_bip39::seed_from_entropy`.
pub const PBKDF2_ROUNDS: u32 = 2048;

/// Bits contributed by each mnemonic word.
const BITS_PER_WORD: usize = 11;

/// The BIP39 English wordlist, in canonical (ascending) order.
///
/// Indices into this slice are the 11-bit values encoded by each word.
#[must_use]
pub fn wordlist() -> &'static [&'static str] {
    static WORDS: OnceLock<Vec<&'static str>> = OnceLock::new();
    WORDS.get_or_init(|| WORDLIST_RAW.lines().filter(|l| !l.is_empty()).collect())
}

/// Decode a BIP39 mnemonic phrase into its entropy bytes.
///
/// Whitespace is normalised (any run of ASCII whitespace separates words) and
/// words are lowercased before lookup. The BIP39 checksum is verified.
///
/// # Errors
///
/// - [`KeystoreError::WordCount`] if the phrase is not 12/15/18/21/24 words
/// - [`KeystoreError::UnknownWord`] if a word is not in the English wordlist
/// - [`KeystoreError::MnemonicChecksum`] if the trailing checksum bits are wrong
pub fn mnemonic_to_entropy(phrase: &str) -> Result<Vec<u8>, KeystoreError> {
    let words: Vec<String> = phrase
        .split_ascii_whitespace()
        .map(str::to_lowercase)
        .collect();
    let count = words.len();
    if !matches!(count, 12 | 15 | 18 | 21 | 24) {
        return Err(KeystoreError::WordCount(count));
    }

    let total_bits = count * BITS_PER_WORD;
    // total_bits = ENT + CS with CS = ENT/32, hence ENT = total_bits * 32 / 33.
    let entropy_bits = total_bits / 33 * 32;
    let checksum_bits = total_bits - entropy_bits;

    let list = wordlist();
    let mut buf = vec![0u8; total_bits.div_ceil(8)];
    for (word_index, word) in words.iter().enumerate() {
        let value = list
            .binary_search(&word.as_str())
            .map_err(|_| KeystoreError::UnknownWord)?;
        for bit in 0..BITS_PER_WORD {
            if (value >> (BITS_PER_WORD - 1 - bit)) & 1 == 1 {
                let pos = word_index * BITS_PER_WORD + bit;
                buf[pos / 8] |= 1u8 << (7 - (pos % 8));
            }
        }
    }

    let entropy = buf
        .get(..entropy_bits / 8)
        .ok_or(KeystoreError::MnemonicChecksum)?
        .to_vec();
    let digest = Sha256::digest(&entropy);
    for i in 0..checksum_bits {
        let pos = entropy_bits + i;
        let got = (buf[pos / 8] >> (7 - (pos % 8))) & 1;
        let want = (digest[i / 8] >> (7 - (i % 8))) & 1;
        if got != want {
            return Err(KeystoreError::MnemonicChecksum);
        }
    }
    Ok(entropy)
}

/// Derive Substrate's 64-byte seed from BIP39 entropy bytes.
///
/// `PBKDF2-HMAC-SHA512(password = entropy, salt = "mnemonic" || password,
/// rounds = 2048, dkLen = 64)`. Because `dkLen` equals the HMAC-SHA512 output
/// length there is exactly one PBKDF2 block.
///
/// # Errors
///
/// - [`KeystoreError::EntropyLength`] if `entropy` is not 16..=32 bytes and a multiple of 4
/// - [`KeystoreError::Hmac`] if HMAC-SHA512 rejects the key length
pub fn substrate_seed_from_entropy(
    entropy: &[u8],
    password: &str,
) -> Result<[u8; 64], KeystoreError> {
    let len = entropy.len();
    if !(16..=32).contains(&len) || !len.is_multiple_of(4) {
        return Err(KeystoreError::EntropyLength(len));
    }

    let mut salted_block = Vec::with_capacity(8 + password.len() + 4);
    salted_block.extend_from_slice(b"mnemonic");
    salted_block.extend_from_slice(password.as_bytes());
    salted_block.extend_from_slice(&1u32.to_be_bytes());

    let mut u = hmac_sha512(entropy, &salted_block)?;
    let mut out = u;
    for _ in 1..PBKDF2_ROUNDS {
        u = hmac_sha512(entropy, &u)?;
        for (acc, next) in out.iter_mut().zip(u.iter()) {
            *acc ^= *next;
        }
    }
    Ok(out)
}

/// Derive the 32-byte sr25519 mini-secret from a mnemonic phrase.
///
/// Pass `""` as `password` for the usual (Bittensor / `btcli`) case.
///
/// # Errors
///
/// Propagates [`mnemonic_to_entropy`] and [`substrate_seed_from_entropy`] errors.
pub fn mini_secret_from_mnemonic(
    phrase: &str,
    password: &str,
) -> Result<[u8; crate::KEY_LEN], KeystoreError> {
    let entropy = mnemonic_to_entropy(phrase)?;
    let seed = substrate_seed_from_entropy(&entropy, password)?;
    let mut mini = [0u8; crate::KEY_LEN];
    mini.copy_from_slice(&seed[..crate::KEY_LEN]);
    Ok(mini)
}

fn hmac_sha512(key: &[u8], data: &[u8]) -> Result<[u8; 64], KeystoreError> {
    let mut mac = Hmac::<Sha512>::new_from_slice(key).map_err(|_| KeystoreError::Hmac)?;
    mac.update(data);
    let tag = mac.finalize().into_bytes();
    let mut out = [0u8; 64];
    out.copy_from_slice(&tag);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        mini_secret_from_mnemonic, mnemonic_to_entropy, substrate_seed_from_entropy, wordlist,
        WORDLIST_LEN, WORDLIST_RAW,
    };
    use crate::KeystoreError;
    use sha2::{Digest, Sha256};

    /// Canonical BIP39 `english.txt` digest (bitcoin/bips bip-0039).
    const WORDLIST_SHA256: &str =
        "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda";

    /// Substrate dev phrase; cross-checked against `substrateinterface`.
    const DEV_PHRASE: &str =
        "bottom drive obey lake curtain smoke basket hold race lonely fit walk";

    #[test]
    fn wordlist_digest_is_pinned() {
        let digest = Sha256::digest(WORDLIST_RAW.as_bytes());
        assert_eq!(hex::encode(digest), WORDLIST_SHA256);
    }

    #[test]
    fn wordlist_is_2048_sorted_lowercase_ascii() {
        let list = wordlist();
        assert_eq!(list.len(), WORDLIST_LEN);
        for w in list {
            assert!(!w.is_empty());
            assert!(
                w.bytes().all(|b| b.is_ascii_lowercase()),
                "non lowercase-ascii word: {w}"
            );
        }
        assert!(list.windows(2).all(|p| p[0] < p[1]), "wordlist not sorted");
    }

    #[test]
    fn dev_phrase_entropy_and_seed() {
        let entropy = mnemonic_to_entropy(DEV_PHRASE).expect("dev phrase decodes");
        assert_eq!(entropy.len(), 16);
        let seed = substrate_seed_from_entropy(&entropy, "").expect("seed derives");
        let mini = mini_secret_from_mnemonic(DEV_PHRASE, "").expect("mini secret");
        assert_eq!(&seed[..32], &mini[..]);
    }

    #[test]
    fn whitespace_and_case_are_normalised() {
        let a = mnemonic_to_entropy(DEV_PHRASE).expect("plain");
        let noisy = format!("  {}  ", DEV_PHRASE.to_uppercase().replace(' ', "\n\t "));
        let b = mnemonic_to_entropy(&noisy).expect("noisy");
        assert_eq!(a, b);
    }

    #[test]
    fn all_valid_word_counts_decode() {
        // 12/15/18/21/24-word phrases whose checksums are valid by construction.
        for (words, ent_len) in [(12, 16), (15, 20), (18, 24), (21, 28), (24, 32)] {
            let entropy = vec![0u8; ent_len];
            let phrase = entropy_to_phrase(&entropy);
            assert_eq!(phrase.split(' ').count(), words);
            assert_eq!(
                mnemonic_to_entropy(&phrase).expect("round trip decodes"),
                entropy
            );
        }
    }

    #[test]
    fn flipped_word_fails_checksum() {
        let mut words: Vec<&str> = DEV_PHRASE.split(' ').collect();
        // "walk" -> "wall" keeps the word count and validity of every word.
        words[11] = "wall";
        let bad = words.join(" ");
        assert!(matches!(
            mnemonic_to_entropy(&bad),
            Err(KeystoreError::MnemonicChecksum)
        ));
    }

    #[test]
    fn bad_word_count_rejected() {
        let thirteen = format!("{DEV_PHRASE} walk");
        assert!(matches!(
            mnemonic_to_entropy(&thirteen),
            Err(KeystoreError::WordCount(13))
        ));
        assert!(matches!(
            mnemonic_to_entropy(""),
            Err(KeystoreError::WordCount(0))
        ));
    }

    #[test]
    fn unknown_word_rejected() {
        let mut words: Vec<&str> = DEV_PHRASE.split(' ').collect();
        words[0] = "notaword";
        assert!(matches!(
            mnemonic_to_entropy(&words.join(" ")),
            Err(KeystoreError::UnknownWord)
        ));
    }

    #[test]
    fn entropy_length_validated() {
        for bad in [0usize, 12, 18, 36] {
            assert!(matches!(
                substrate_seed_from_entropy(&vec![0u8; bad], ""),
                Err(KeystoreError::EntropyLength(_))
            ));
        }
        for good in [16usize, 20, 24, 28, 32] {
            assert!(substrate_seed_from_entropy(&vec![7u8; good], "").is_ok());
        }
    }

    #[test]
    fn password_changes_the_seed() {
        let a = mini_secret_from_mnemonic(DEV_PHRASE, "").expect("no password");
        let b = mini_secret_from_mnemonic(DEV_PHRASE, "pw").expect("password");
        assert_ne!(a, b);
    }

    /// Encode entropy bytes as a checksummed BIP39 phrase (test helper).
    fn entropy_to_phrase(entropy: &[u8]) -> String {
        let list = wordlist();
        let digest = Sha256::digest(entropy);
        let checksum_bits = entropy.len() * 8 / 32;
        let mut bits: Vec<u8> = Vec::new();
        for byte in entropy {
            for i in 0..8 {
                bits.push((byte >> (7 - i)) & 1);
            }
        }
        for i in 0..checksum_bits {
            bits.push((digest[i / 8] >> (7 - (i % 8))) & 1);
        }
        bits.chunks(11)
            .map(|chunk| {
                let idx = chunk
                    .iter()
                    .fold(0usize, |acc, b| (acc << 1) | usize::from(*b));
                list[idx]
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}
