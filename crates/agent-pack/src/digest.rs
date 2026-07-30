//! Content digests for packs and environment Dockerfiles.

use sha2::{Digest, Sha256};

/// Width of SHA-256 digests used for pack / image identity.
pub const DIGEST_LEN: usize = 32;

/// Raw 32-byte SHA-256 digest.
pub type DigestBytes = [u8; DIGEST_LEN];

/// SHA-256 over `bytes`, returned as raw 32 bytes.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> DigestBytes {
    Sha256::digest(bytes).into()
}

/// Lowercase hex encoding of a raw digest (64 chars).
#[must_use]
pub fn digest_hex(digest: &DigestBytes) -> String {
    hex::encode(digest)
}

/// Docker-style content digest: `sha256:` + lowercase hex of `bytes`.
#[must_use]
pub fn content_digest_label(bytes: &[u8]) -> String {
    format!("sha256:{}", digest_hex(&sha256_bytes(bytes)))
}

/// Stable pack digest over a sorted list of `(relative_path, file_bytes)`.
///
/// Canonical encoding per entry (in path lexicographic order):
/// `path_utf8 || 0x00 || u64_le(content_len) || content`.
///
/// Empty path lists hash the empty byte string (still 32 bytes).
#[must_use]
pub fn pack_digest_from_entries(entries: &[(String, Vec<u8>)]) -> DigestBytes {
    let mut ordered: Vec<(&str, &[u8])> = entries
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();
    ordered.sort_by(|a, b| a.0.cmp(b.0));

    let mut hasher = Sha256::new();
    for (path, content) in ordered {
        hasher.update(path.as_bytes());
        hasher.update([0_u8]);
        let len = u64::try_from(content.len()).unwrap_or(u64::MAX);
        hasher.update(len.to_le_bytes());
        hasher.update(content);
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::{content_digest_label, digest_hex, pack_digest_from_entries, sha256_bytes};

    #[test]
    fn sha256_empty_is_known_vector() {
        let d = sha256_bytes(b"");
        assert_eq!(
            digest_hex(&d),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn pack_digest_is_order_independent() {
        let a = vec![
            ("b.txt".into(), b"B".to_vec()),
            ("a.txt".into(), b"A".to_vec()),
        ];
        let b = vec![
            ("a.txt".into(), b"A".to_vec()),
            ("b.txt".into(), b"B".to_vec()),
        ];
        assert_eq!(pack_digest_from_entries(&a), pack_digest_from_entries(&b));
    }

    #[test]
    fn content_digest_label_prefixes_sha256() {
        let label = content_digest_label(b"hello");
        assert!(label.starts_with("sha256:"));
        assert_eq!(label.len(), "sha256:".len() + 64);
    }
}
