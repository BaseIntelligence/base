//! Public hashed checkpoint lineage (brief §9.4 — atomic rollback).

use sha2::{Digest, Sha256};

use crate::state::CheckpointHash;

/// One published champion generation in the public lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageEntry {
    /// Generation index; genesis champion is 0.
    pub generation: u64,
    /// Checkpoint content hash C(n).
    pub checkpoint_hash: CheckpointHash,
    /// Prior champion hash C(n-1), if any.
    pub parent_hash: Option<CheckpointHash>,
    /// Public hash of this lineage record (binds generation + hashes).
    pub entry_hash: [u8; 32],
    /// Optional challenger id that produced this champion (audit).
    pub challenger_id: Option<u64>,
}

/// Append-only public lineage of champion checkpoints.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CheckpointLineage {
    entries: Vec<LineageEntry>,
}

impl CheckpointLineage {
    /// Empty lineage (no champion yet).
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Number of published champion generations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no champion has been published.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current tip C(n), if any.
    #[must_use]
    pub fn tip(&self) -> Option<&LineageEntry> {
        self.entries.last()
    }

    /// Current champion checkpoint hash.
    #[must_use]
    pub fn tip_hash(&self) -> Option<CheckpointHash> {
        self.tip().map(|e| e.checkpoint_hash)
    }

    /// Prior generation C(n-1), if the tip is not genesis.
    #[must_use]
    pub fn prior(&self) -> Option<&LineageEntry> {
        let n = self.entries.len();
        if n < 2 {
            None
        } else {
            self.entries.get(n - 2)
        }
    }

    /// Immutable view of the full public chain (oldest first).
    #[must_use]
    pub fn entries(&self) -> &[LineageEntry] {
        &self.entries
    }

    /// Publish a new champion checkpoint as C(n).
    ///
    /// `parent` must match the current tip when the lineage is non-empty.
    pub fn append(
        &mut self,
        checkpoint_hash: CheckpointHash,
        challenger_id: Option<u64>,
    ) -> LineageEntry {
        let parent_hash = self.tip_hash();
        let generation = u64::try_from(self.entries.len()).unwrap_or(u64::MAX);
        let entry_hash = hash_lineage_entry(generation, &checkpoint_hash, parent_hash.as_ref());
        let entry = LineageEntry {
            generation,
            checkpoint_hash,
            parent_hash,
            entry_hash,
            challenger_id,
        };
        self.entries.push(entry.clone());
        entry
    }

    /// Drop the tip and restore C(n-1). Returns the restored entry.
    ///
    /// Fails (returns `None`) when there is no prior generation.
    pub fn rollback_tip(&mut self) -> Option<LineageEntry> {
        if self.entries.len() < 2 {
            return None;
        }
        self.entries.pop();
        self.entries.last().cloned()
    }
}

/// Domain-separated SHA-256 over generation + checkpoint + optional parent.
#[must_use]
pub fn hash_lineage_entry(
    generation: u64,
    checkpoint: &CheckpointHash,
    parent: Option<&CheckpointHash>,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hypertraining-lineage-v1");
    h.update(generation.to_le_bytes());
    h.update(checkpoint);
    match parent {
        Some(p) => {
            h.update([1_u8]);
            h.update(p);
        }
        None => {
            h.update([0_u8]);
        }
    }
    let dig = h.finalize();
    let mut out = [0_u8; 32];
    out.copy_from_slice(&dig);
    out
}

/// Hex encode a 32-byte digest (lowercase).
#[must_use]
pub fn hash_hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(byte: u8) -> CheckpointHash {
        [byte; 32]
    }

    #[test]
    fn append_builds_parent_chain() {
        let mut lin = CheckpointLineage::new();
        let e0 = lin.append(h(1), Some(10));
        assert_eq!(e0.generation, 0);
        assert!(e0.parent_hash.is_none());
        let e1 = lin.append(h(2), Some(11));
        assert_eq!(e1.generation, 1);
        assert_eq!(e1.parent_hash, Some(h(1)));
        assert_eq!(lin.tip_hash(), Some(h(2)));
        assert_ne!(e0.entry_hash, e1.entry_hash);
    }

    #[test]
    fn rollback_restores_prior_hash() {
        let mut lin = CheckpointLineage::new();
        lin.append(h(1), None);
        lin.append(h(2), None);
        let restored = lin.rollback_tip().expect("prior");
        assert_eq!(restored.checkpoint_hash, h(1));
        assert_eq!(lin.tip_hash(), Some(h(1)));
        assert_eq!(lin.len(), 1);
    }

    #[test]
    fn rollback_genesis_alone_fails() {
        let mut lin = CheckpointLineage::new();
        lin.append(h(1), None);
        assert!(lin.rollback_tip().is_none());
    }

    #[test]
    fn entry_hash_is_deterministic() {
        let a = hash_lineage_entry(0, &h(9), None);
        let b = hash_lineage_entry(0, &h(9), None);
        assert_eq!(a, b);
        let c = hash_lineage_entry(0, &h(8), None);
        assert_ne!(a, c);
    }
}
