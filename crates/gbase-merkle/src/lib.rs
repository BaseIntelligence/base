//! RFC 6962 Certificate Transparency Merkle tree.
//!
//! - Leaf: `SHA256(0x00 ‖ leaf_data)`
//! - Internal node: `SHA256(0x01 ‖ left ‖ right)`
//! - Odd nodes at a level are **promoted** (never duplicated) — blocks CVE-2012-2459.
//! - Empty tree root is the pinned constant [`EMPTY_ROOT`] (RFC 6962 §2.1 `SHA-256()`).
//!
//! Canonical leaf ordering is the caller's responsibility (D7); this crate hashes the
//! slice it is given in order.

#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};

/// 32-byte Merkle hash (SHA-256 output).
pub type Hash = [u8; 32];

/// Empty-tree root per RFC 6962 §2.1: `MTH({}) = SHA-256()` of the empty input.
///
/// Hex: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
///
/// Pinned as an exact constant so `BUNDLE_SPEC` and all implementers share one value
/// (no prose ambiguity about “hash of nothing”).
pub const EMPTY_ROOT: Hash = [
    0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
    0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
    0xb8, 0x55,
];

/// Errors from proof construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MerkleError {
    /// `proof` requested on an empty leaf set.
    EmptyTree,
    /// Leaf index ≥ tree size.
    IndexOutOfBounds,
}

impl core::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyTree => write!(f, "merkle proof on empty tree"),
            Self::IndexOutOfBounds => write!(f, "merkle leaf index out of bounds"),
        }
    }
}

impl std::error::Error for MerkleError {}

/// Inclusion proof for one leaf (RFC 6962 audit path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof {
    /// Raw leaf bytes (preimage of the leaf hash).
    pub leaf: Vec<u8>,
    /// Zero-based leaf index.
    pub index: u64,
    /// Number of leaves in the tree when the proof was built.
    pub tree_size: u64,
    /// Sibling hashes from the leaf level upward (RFC 6962 audit path).
    pub audit_path: Vec<Hash>,
}

/// Domain-separated leaf hash: `SHA256(0x00 ‖ leaf)`.
#[must_use]
pub fn hash_leaf(leaf: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x00_u8]);
    hasher.update(leaf);
    hasher.finalize().into()
}

/// Domain-separated node hash: `SHA256(0x01 ‖ left ‖ right)`.
#[must_use]
pub fn hash_node(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x01_u8]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Compute the Merkle root over `leaves` in the given order (RFC 6962 MTH).
///
/// Empty input returns [`EMPTY_ROOT`]. Odd trailing nodes are promoted, never duplicated.
#[must_use]
pub fn root(leaves: &[&[u8]]) -> Hash {
    if leaves.is_empty() {
        return EMPTY_ROOT;
    }
    let mut level: Vec<Hash> = leaves.iter().map(|leaf| hash_leaf(leaf)).collect();
    while level.len() > 1 {
        level = next_level(&level);
    }
    level[0]
}

/// Build an inclusion [`Proof`] for `leaves[index]`.
///
/// # Errors
///
/// Returns [`MerkleError::EmptyTree`] if `leaves` is empty, or
/// [`MerkleError::IndexOutOfBounds`] if `index >= leaves.len()`.
pub fn proof(leaves: &[&[u8]], index: usize) -> Result<Proof, MerkleError> {
    let tree_size = leaves.len();
    if tree_size == 0 {
        return Err(MerkleError::EmptyTree);
    }
    if index >= tree_size {
        return Err(MerkleError::IndexOutOfBounds);
    }

    let mut level: Vec<Hash> = leaves.iter().map(|leaf| hash_leaf(leaf)).collect();
    let mut fn_idx = index;
    let mut sn = tree_size.saturating_sub(1);
    let mut audit_path = Vec::new();

    while sn > 0 {
        if (fn_idx & 1) == 1 || fn_idx < sn {
            let sib_i = if (fn_idx & 1) == 1 {
                fn_idx.saturating_sub(1)
            } else {
                fn_idx.saturating_add(1)
            };
            audit_path.push(level[sib_i]);
        }
        level = next_level(&level);
        fn_idx >>= 1;
        sn >>= 1;
    }

    Ok(Proof {
        leaf: leaves[index].to_vec(),
        index: u64::try_from(index).unwrap_or(u64::MAX),
        tree_size: u64::try_from(tree_size).unwrap_or(u64::MAX),
        audit_path,
    })
}

/// Verify an inclusion [`Proof`] against an expected Merkle `root`.
///
/// Returns `false` if the reconstructed root differs, the path is malformed, or
/// `index`/`tree_size` are inconsistent.
#[must_use]
pub fn verify(inclusion: &Proof, expected_root: &Hash) -> bool {
    if inclusion.tree_size == 0 || inclusion.index >= inclusion.tree_size {
        return false;
    }

    let mut node = hash_leaf(&inclusion.leaf);
    let mut fn_idx = inclusion.index;
    let mut sn = inclusion.tree_size.saturating_sub(1);
    let mut path_i = 0usize;

    while sn > 0 {
        if (fn_idx & 1) == 1 || fn_idx < sn {
            let Some(sibling) = inclusion.audit_path.get(path_i) else {
                return false;
            };
            node = if (fn_idx & 1) == 1 {
                hash_node(sibling, &node)
            } else {
                hash_node(&node, sibling)
            };
            path_i = path_i.saturating_add(1);
        }
        fn_idx >>= 1;
        sn >>= 1;
    }

    path_i == inclusion.audit_path.len() && node == *expected_root
}

fn next_level(level: &[Hash]) -> Vec<Hash> {
    let mut next = Vec::with_capacity(level.len().div_ceil(2));
    let mut i = 0usize;
    while i < level.len() {
        let right = i.saturating_add(1);
        if right < level.len() {
            next.push(hash_node(&level[i], &level[right]));
            i = i.saturating_add(2);
        } else {
            // Promote odd node — never duplicate (CVE-2012-2459).
            next.push(level[i]);
            i = right;
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::{
        hash_leaf, hash_node, proof, root, verify, EMPTY_ROOT, Hash, MerkleError,
    };
    use sha2::Digest;

    /// Official leaf inputs from transparency-dev/merkle `testonly/constants.go` `LeafInputs()`.
    fn ct_leaf_inputs() -> [Vec<u8>; 8] {
        [
            vec![],
            vec![0x00],
            vec![0x10],
            vec![0x20, 0x21],
            vec![0x30, 0x31],
            vec![0x40, 0x41, 0x42, 0x43],
            vec![0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57],
            vec![
                0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d,
                0x6e, 0x6f,
            ],
        ]
    }

    fn as_refs(owned: &[Vec<u8>]) -> Vec<&[u8]> {
        owned.iter().map(Vec::as_slice).collect()
    }

    fn hex_hash(hex: &str) -> Hash {
        let mut out = [0u8; 32];
        assert_eq!(hex.len(), 64);
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let s = core::str::from_utf8(chunk).expect("utf8");
            out[i] = u8::from_str_radix(s, 16).expect("hex");
        }
        out
    }

    #[test]
    fn empty_root_is_pinned_sha256_empty() {
        assert_eq!(
            EMPTY_ROOT,
            hex_hash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        );
        assert_eq!(root(&[]), EMPTY_ROOT);
        // Independent check: SHA-256 of empty input (not hash_leaf(&[])).
        let mut h = sha2::Sha256::new();
        h.update([]);
        let digest: Hash = h.finalize().into();
        assert_eq!(digest, EMPTY_ROOT);
        assert_ne!(hash_leaf(&[]), EMPTY_ROOT);
    }

    #[test]
    fn known_answer_roots_n_0_through_8_ct_vectors() {
        // RootHashes() from transparency-dev/merkle testonly/constants.go
        let expected: &[(usize, &str)] = &[
            (
                0,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                1,
                "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            ),
            (
                2,
                "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
            ),
            (
                3,
                "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
            ),
            (
                4,
                "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
            ),
            (
                5,
                "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
            ),
            (
                7,
                "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
            ),
            (
                8,
                "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
            ),
        ];
        let all = ct_leaf_inputs();
        for &(n, hex) in expected {
            let owned = all[..n].to_vec();
            let refs = as_refs(&owned);
            assert_eq!(root(&refs), hex_hash(hex), "root mismatch n={n}");
        }
    }

    #[test]
    fn seven_leaf_every_index_verifies() {
        let owned = ct_leaf_inputs()[..7].to_vec();
        let refs = as_refs(&owned);
        let r = root(&refs);
        for (i, leaf) in owned.iter().enumerate() {
            let p = proof(&refs, i).expect("proof");
            assert!(verify(&p, &r), "index {i} must verify");
            assert_eq!(p.index, i as u64);
            assert_eq!(p.tree_size, 7);
            assert_eq!(&p.leaf, leaf);
        }
    }

    #[test]
    fn flipped_bit_in_leaf_fails_verify() {
        let owned = ct_leaf_inputs()[..7].to_vec();
        let refs = as_refs(&owned);
        let r = root(&refs);
        let mut p = proof(&refs, 3).expect("proof");
        if p.leaf.is_empty() {
            p.leaf.push(0);
        }
        p.leaf[0] ^= 0x01;
        assert!(!verify(&p, &r));
    }

    #[test]
    fn flipped_bit_in_audit_path_fails_verify() {
        let owned = ct_leaf_inputs()[..7].to_vec();
        let refs = as_refs(&owned);
        let r = root(&refs);
        let mut p = proof(&refs, 0).expect("proof");
        assert!(!p.audit_path.is_empty());
        p.audit_path[0][0] ^= 0x01;
        assert!(!verify(&p, &r));
    }

    #[test]
    fn cve_2012_2459_duplicated_node_does_not_match_promote_root() {
        // 3 leaves: RFC6962 promotes the odd third leaf.
        // Bitcoin-style duplicate would hash node(h2, h2) instead of promoting h2.
        let owned = ct_leaf_inputs()[..3].to_vec();
        let refs = as_refs(&owned);
        let rfc_root = root(&refs);

        let h0 = hash_leaf(&owned[0]);
        let h1 = hash_leaf(&owned[1]);
        let h2 = hash_leaf(&owned[2]);
        let left = hash_node(&h0, &h1);
        let dup_right = hash_node(&h2, &h2); // forbidden duplication
        let forged = hash_node(&left, &dup_right);

        assert_ne!(
            forged, rfc_root,
            "duplicated-node forgery must not equal promote-odd root"
        );
        let honest = hash_node(&left, &h2);
        assert_eq!(honest, rfc_root);
    }

    #[test]
    fn proof_errors_on_empty_and_oob() {
        assert_eq!(proof(&[], 0), Err(MerkleError::EmptyTree));
        let owned = ct_leaf_inputs()[..2].to_vec();
        let refs = as_refs(&owned);
        assert_eq!(proof(&refs, 2), Err(MerkleError::IndexOutOfBounds));
    }

    #[test]
    fn verify_rejects_inconsistent_proof_metadata() {
        let owned = ct_leaf_inputs()[..4].to_vec();
        let refs = as_refs(&owned);
        let r = root(&refs);
        let mut p = proof(&refs, 1).expect("proof");
        p.tree_size = 0;
        assert!(!verify(&p, &r));
        let mut p = proof(&refs, 1).expect("proof");
        p.index = 99;
        assert!(!verify(&p, &r));
        let mut p = proof(&refs, 1).expect("proof");
        p.audit_path.pop();
        assert!(!verify(&p, &r));
    }

    #[test]
    fn single_leaf_proof_is_empty_path() {
        let owned = ct_leaf_inputs()[..1].to_vec();
        let refs = as_refs(&owned);
        let r = root(&refs);
        let p = proof(&refs, 0).expect("proof");
        assert!(p.audit_path.is_empty());
        assert!(verify(&p, &r));
        assert_eq!(r, hash_leaf(&owned[0]));
    }

    #[test]
    fn proof_roundtrip_power_of_two_and_odd_sizes() {
        let all = ct_leaf_inputs();
        for n in [2usize, 3, 4, 5, 8] {
            let owned = all[..n].to_vec();
            let refs = as_refs(&owned);
            let r = root(&refs);
            for i in 0..n {
                let p = proof(&refs, i).expect("proof");
                assert!(verify(&p, &r), "n={n} i={i}");
            }
        }
    }

    #[test]
    fn wrong_root_fails() {
        let owned = ct_leaf_inputs()[..4].to_vec();
        let refs = as_refs(&owned);
        let p = proof(&refs, 0).expect("proof");
        let mut bad = root(&refs);
        bad[0] ^= 0xff;
        assert!(!verify(&p, &bad));
    }

    #[test]
    fn proof_is_clone_eq() {
        let owned = ct_leaf_inputs()[..2].to_vec();
        let refs = as_refs(&owned);
        let p = proof(&refs, 0).expect("proof");
        let q = p.clone();
        assert_eq!(p, q);
    }

    #[test]
    fn hasher_kats_from_transparency_dev_merkle() {
        assert_eq!(
            hash_leaf(b""),
            hex_hash("6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d")
        );
        assert_eq!(
            hash_leaf(b"L123456"),
            hex_hash("395aa064aa4c29f7010acfe3f25db9485bbd4b91897b6ad7ad547639252b4d56")
        );
        // Node KAT uses raw 4-byte chunks as child digests in the reference test
        // (not full 32-byte hashes) — HashChildren still prefixes 0x01.
        let left = *b"N123";
        let right = *b"N456";
        let mut hasher = sha2::Sha256::new();
        hasher.update([0x01_u8]);
        hasher.update(left);
        hasher.update(right);
        let got: Hash = hasher.finalize().into();
        assert_eq!(
            got,
            hex_hash("aa217fe888e47007fa15edab33c2b492a722cb106c64667fc2b044444de66bbb")
        );
    }
}
