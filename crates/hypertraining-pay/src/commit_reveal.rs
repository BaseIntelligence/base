//! Commit-reveal binding for submission content (brief §11.4).
//!
//! Miner commits `H = SHA-256(domain || nonce || payload)` before peers can
//! observe the payload; reveal must reproduce the same digest or is rejected.

use sha2::{Digest, Sha256};

use crate::error::PayError;

/// Domain separation tag for hypertraining submission commits.
pub const COMMIT_DOMAIN: &[u8] = b"base-hypertraining-commit-v1";

/// 32-byte SHA-256 commitment.
pub type CommitDigest = [u8; 32];

/// Compute the commitment digest for `payload` under `nonce`.
///
/// # Errors
///
/// [`PayError::EmptyPayload`] when `payload` is empty.
pub fn commit(payload: &[u8], nonce: &[u8]) -> Result<CommitDigest, PayError> {
    if payload.is_empty() {
        return Err(PayError::EmptyPayload);
    }
    Ok(digest(payload, nonce))
}

/// Verify a reveal against a prior commitment.
///
/// # Errors
///
/// - [`PayError::EmptyPayload`] when `payload` is empty
/// - [`PayError::CommitRevealMismatch`] when digests differ
pub fn reveal(expected: &CommitDigest, payload: &[u8], nonce: &[u8]) -> Result<(), PayError> {
    let got = commit(payload, nonce)?;
    if got != *expected {
        return Err(PayError::CommitRevealMismatch);
    }
    Ok(())
}

/// True when reveal would succeed (no alloc of error path for hot checks).
#[must_use]
pub fn reveal_matches(expected: &CommitDigest, payload: &[u8], nonce: &[u8]) -> bool {
    if payload.is_empty() {
        return false;
    }
    digest(payload, nonce) == *expected
}

fn digest(payload: &[u8], nonce: &[u8]) -> CommitDigest {
    let mut h = Sha256::new();
    h.update(COMMIT_DOMAIN);
    h.update(nonce);
    h.update(payload);
    let out = h.finalize();
    let mut arr = [0_u8; 32];
    arr.copy_from_slice(&out);
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_reveal_roundtrip() {
        let payload = b"miner-fork-tree-v1";
        let nonce = b"n-42";
        let c = commit(payload, nonce).expect("commit");
        reveal(&c, payload, nonce).expect("reveal ok");
        assert!(reveal_matches(&c, payload, nonce));
    }

    #[test]
    fn mismatch_payload_rejected() {
        let c = commit(b"alpha", b"n").expect("commit");
        let err = reveal(&c, b"beta", b"n").expect_err("mismatch");
        assert_eq!(err, PayError::CommitRevealMismatch);
        assert!(!reveal_matches(&c, b"beta", b"n"));
    }

    #[test]
    fn mismatch_nonce_rejected() {
        let c = commit(b"alpha", b"n1").expect("commit");
        let err = reveal(&c, b"alpha", b"n2").expect_err("nonce");
        assert_eq!(err, PayError::CommitRevealMismatch);
    }

    #[test]
    fn empty_payload_rejected() {
        assert_eq!(commit(b"", b"n"), Err(PayError::EmptyPayload));
        let c = [0_u8; 32];
        assert_eq!(reveal(&c, b"", b"n"), Err(PayError::EmptyPayload));
    }

    #[test]
    fn domain_changes_digest() {
        // Sanity: digest is not bare sha256(payload).
        let payload = b"x";
        let nonce = b"y";
        let c = commit(payload, nonce).expect("c");
        let mut h = Sha256::new();
        h.update(payload);
        let bare = h.finalize();
        assert_ne!(c.as_slice(), bare.as_slice());
    }
}
