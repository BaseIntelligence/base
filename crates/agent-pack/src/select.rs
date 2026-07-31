//! Deterministic per-epoch pack selection.
//!
//! # Placement
//! This module lives in `agent-pack` because [`PackId`] is defined here and the
//! catalog is an ordered list of pack identities. Challenge replicas call the
//! same pure function with the same ordered catalog; scoring stays in
//! `agent-challenge`.
//!
//! # Algorithm (normative)
//! Let `n = catalog.len()`.
//! - `n == 0` → [`PackError::EmptyCatalog`]
//! - `digest = sha256(b"gbase-agent-pack-select-v1" ‖ miner_hotkey)`
//! - `seed = u64::from_le_bytes(digest[0..8])` (little-endian)
//! - `index = seed.wrapping_add(epoch) % (n as u64)` as `usize`
//! - return `catalog[index].clone()`
//!
//! Epoch is mixed by modular addition so that when `n > 1`, consecutive epochs
//! never share an index for a fixed miner (index advances by 1 mod `n`).
//! Single-entry catalogs always return that entry (repeat allowed by design).
//!
//! Catalog **order is significant** and must be identical across challenge
//! replicas (callers present a canonically ordered slice).
//!
//! No RNG, no clock, no I/O.

use sha2::{Digest, Sha256};

use crate::error::PackError;
use crate::PackId;

/// Domain-separation tag for pack selection (hash family, not a signing tag).
pub const PACK_SELECT_DOMAIN: &[u8] = b"gbase-agent-pack-select-v1";

/// Miner hotkey width (sr25519 public key / `crypto::KEY_LEN`).
pub const HOTKEY_LEN: usize = 32;

/// Select the pack id for `(epoch, miner_hotkey)` from an ordered catalog.
///
/// # Errors
/// [`PackError::EmptyCatalog`] when `catalog` is empty.
#[must_use = "selection result should be used"]
pub fn select_pack(
    epoch: u64,
    miner_hotkey: &[u8; HOTKEY_LEN],
    catalog: &[PackId],
) -> Result<PackId, PackError> {
    let n = catalog.len();
    if n == 0 {
        return Err(PackError::EmptyCatalog);
    }
    let index = select_index(epoch, miner_hotkey, n);
    Ok(catalog[index].clone())
}

/// Pure index into a non-empty catalog of length `n`.
#[must_use]
pub fn select_index(epoch: u64, miner_hotkey: &[u8; HOTKEY_LEN], n: usize) -> usize {
    debug_assert!(n >= 1, "caller must reject empty catalog");
    if n == 1 {
        return 0;
    }
    let digest = {
        let mut h = Sha256::new();
        h.update(PACK_SELECT_DOMAIN);
        h.update(miner_hotkey);
        h.finalize()
    };
    let seed = u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ]);
    // n is from catalog.len(); remainder is always < n, so fits usize on all targets.
    #[allow(clippy::cast_possible_truncation)]
    {
        (seed.wrapping_add(epoch) % (n as u64)) as usize
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{select_index, select_pack, HOTKEY_LEN, PACK_SELECT_DOMAIN};
    use crate::error::PackError;
    use crate::PackId;
    use std::collections::BTreeMap;

    fn catalog(ids: &[&str]) -> Vec<PackId> {
        ids.iter().map(|s| PackId::new(*s)).collect()
    }

    fn hotkey(byte: u8) -> [u8; HOTKEY_LEN] {
        [byte; HOTKEY_LEN]
    }

    /// S5 — empty catalog → typed `EmptyCatalog` (not panic).
    #[test]
    fn empty_catalog_returns_empty_catalog_error() {
        let err = select_pack(0, &hotkey(0x11), &[]).expect_err("empty");
        assert_eq!(err, PackError::EmptyCatalog);
    }

    /// S6 — single-entry catalog: same pack every epoch (repeat allowed).
    #[test]
    fn single_entry_catalog_repeats_every_epoch() {
        let cat = catalog(&["only-pack"]);
        for epoch in [0_u64, 1, 2, 99, u64::MAX] {
            let id = select_pack(epoch, &hotkey(0xAB), &cat).expect("ok");
            assert_eq!(id.as_str(), "only-pack");
        }
    }

    /// S1 — multi-pack selection returns a member of the catalog.
    #[test]
    fn multi_pack_returns_catalog_member() {
        let cat = catalog(&["a", "b", "c", "d"]);
        let id = select_pack(7, &hotkey(0x11), &cat).expect("ok");
        assert!(
            cat.iter().any(|p| p == &id),
            "selected {} not in catalog",
            id.as_str()
        );
    }

    /// S4 — no consecutive epoch repeat for same hotkey when n > 1.
    #[test]
    fn no_consecutive_repeat_when_multi_entry() {
        let cat = catalog(&["p0", "p1", "p2", "p3", "p4"]);
        let hk = hotkey(0x42);
        let mut prev: Option<PackId> = None;
        for epoch in 0..500_u64 {
            let id = select_pack(epoch, &hk, &cat).expect("ok");
            if let Some(ref p) = prev {
                assert_ne!(
                    p,
                    &id,
                    "consecutive repeat at epoch {epoch}: {}",
                    id.as_str()
                );
            }
            prev = Some(id);
        }
    }

    /// S2 — identical output across two pure invocations for 1000 synthetic pairs.
    #[test]
    fn determinism_1000_pairs_two_invocations() {
        let cat = catalog(&["alpha", "beta", "gamma", "delta", "epsilon"]);
        let mut pairs: Vec<(u64, [u8; HOTKEY_LEN])> = Vec::with_capacity(1000);
        for i in 0..1000_u64 {
            let mut hk = [0u8; HOTKEY_LEN];
            hk[0..8].copy_from_slice(&i.to_le_bytes());
            hk[8] = (i % 251) as u8;
            pairs.push((i.wrapping_mul(3) + 11, hk));
        }

        let run = |pairs: &[(u64, [u8; HOTKEY_LEN])]| -> Vec<String> {
            pairs
                .iter()
                .map(|(e, hk)| {
                    select_pack(*e, hk, &cat)
                        .expect("non-empty")
                        .as_str()
                        .to_owned()
                })
                .collect()
        };

        let a = run(&pairs);
        let b = run(&pairs);
        assert_eq!(a, b, "two pure invocations must match for all 1000 pairs");
        assert_eq!(a.len(), 1000);
    }

    /// S3 — uniform within tolerance over many (epoch, hotkey) samples.
    #[test]
    fn uniform_within_tolerance() {
        let cat = catalog(&["u0", "u1", "u2", "u3"]);
        let n = cat.len();
        #[allow(clippy::cast_precision_loss)]
        let samples = 10_000_usize;
        let mut hist: BTreeMap<String, usize> = BTreeMap::new();
        for i in 0..samples {
            let mut hk = [0u8; HOTKEY_LEN];
            let epoch = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            hk[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            hk[8..16].copy_from_slice(&epoch.to_le_bytes());
            let id = select_pack(epoch, &hk, &cat).expect("ok");
            *hist.entry(id.as_str().to_owned()).or_default() += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let expected = samples as f64 / n as f64;
        // ±15% relative tolerance (hash + modular walk is near-uniform).
        let tol = expected * 0.15;
        for (pack, count) in &hist {
            #[allow(clippy::cast_precision_loss)]
            let c = *count as f64;
            assert!(
                (c - expected).abs() <= tol,
                "pack {pack}: count {count} expected ~{expected:.0} ±{tol:.0}"
            );
        }
        assert_eq!(hist.len(), n, "every pack should appear");
    }

    /// Index helper stays in range.
    #[test]
    fn select_index_in_range() {
        let hk = hotkey(0x7F);
        for n in 1..=16_usize {
            for epoch in 0..64_u64 {
                let i = select_index(epoch, &hk, n);
                assert!(i < n, "index {i} >= n {n} at epoch {epoch}");
            }
        }
    }

    /// Domain tag is the expected ASCII label.
    #[test]
    fn domain_tag_is_gbase_agent_pack_select_v1() {
        assert_eq!(PACK_SELECT_DOMAIN, b"gbase-agent-pack-select-v1");
    }

    /// Different hotkeys can diverge at the same epoch (not a constant map).
    #[test]
    fn hotkey_sensitivity() {
        let cat = catalog(&["a", "b", "c", "d", "e", "f", "g", "h"]);
        let mut seen = std::collections::BTreeSet::new();
        for b in 0..=32_u8 {
            let id = select_pack(0, &hotkey(b), &cat).expect("ok");
            seen.insert(id.as_str().to_owned());
        }
        assert!(
            seen.len() > 1,
            "expected multiple packs across hotkeys, got {seen:?}"
        );
    }
}
