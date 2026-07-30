//! Merkle, metagraph root, sort keys, participant set.

use std::collections::BTreeSet;

use gbase_crypto::KEY_LEN;
use gbase_merkle::root as merkle_root_of;
use gbase_trustroot::ParticipantPolicy;
use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};

use crate::error::BundleError;
use crate::types::{LeafV1, MetagraphRow};

/// Canonical leaf sort key: `scale(challenge_id, miner_hotkey)`.
#[must_use]
pub fn leaf_sort_key(challenge_id: &[u8], miner_hotkey: &[u8; KEY_LEN]) -> Vec<u8> {
    (challenge_id.to_vec(), *miner_hotkey).encode()
}

/// Sort leaves by ascending [`leaf_sort_key`].
pub fn sort_leaves(leaves: &mut [LeafV1]) {
    leaves.sort_by(|a, b| {
        leaf_sort_key(&a.challenge_id, &a.miner_hotkey)
            .cmp(&leaf_sort_key(&b.challenge_id, &b.miner_hotkey))
    });
}

/// Leaf preimage bytes for merkle (`scale(LeafV1)`).
#[must_use]
pub fn leaf_preimage(leaf: &LeafV1) -> Vec<u8> {
    leaf.encode()
}

/// Compute merkle root over leaf preimages in the given order.
#[must_use]
pub fn compute_merkle_root(leaves: &[LeafV1]) -> [u8; 32] {
    let preimages: Vec<Vec<u8>> = leaves.iter().map(leaf_preimage).collect();
    let refs: Vec<&[u8]> = preimages.iter().map(Vec::as_slice).collect();
    merkle_root_of(&refs)
}

/// Project chain metagraph to sorted [`MetagraphRow`] (stake default 0).
///
/// # Errors
///
/// Bad hotkey lengths or uid overflow.
pub fn metagraph_rows_from_chain(
    hotkeys_uid_order: &[Vec<u8>],
    stakes: Option<&[u64]>,
) -> Result<Vec<MetagraphRow>, BundleError> {
    let mut rows = Vec::with_capacity(hotkeys_uid_order.len());
    for (i, hk) in hotkeys_uid_order.iter().enumerate() {
        if hk.len() != KEY_LEN {
            return Err(BundleError::InvalidMetagraph(format!(
                "hotkey len {} at uid index {i}",
                hk.len()
            )));
        }
        let mut hotkey = [0u8; KEY_LEN];
        hotkey.copy_from_slice(hk);
        let uid = u16::try_from(i)
            .map_err(|_| BundleError::InvalidMetagraph("uid does not fit u16".into()))?;
        let stake = stakes.and_then(|s| s.get(i).copied()).unwrap_or(0);
        rows.push(MetagraphRow { hotkey, uid, stake });
    }
    rows.sort_by_key(|a| a.hotkey);
    Ok(rows)
}

/// `metagraph_root = sha256(scale(rows))` (§4.3).
#[must_use]
pub fn compute_metagraph_root(rows: &[MetagraphRow]) -> [u8; 32] {
    let encoded = rows.to_vec().encode();
    let dig = Sha256::digest(encoded);
    let mut out = [0u8; 32];
    out.copy_from_slice(&dig);
    out
}

/// `uid_map` from metagraph rows (already sorted by hotkey).
#[must_use]
pub fn uid_map_from_rows(rows: &[MetagraphRow]) -> Vec<([u8; KEY_LEN], u16)> {
    rows.iter().map(|r| (r.hotkey, r.uid)).collect()
}

/// Expected participants for one challenge policy (§7.2).
#[must_use]
pub fn expected_participants(
    policy: &ParticipantPolicy,
    rows: &[MetagraphRow],
) -> BTreeSet<[u8; KEY_LEN]> {
    let meta: BTreeSet<[u8; KEY_LEN]> = rows.iter().map(|r| r.hotkey).collect();
    match policy {
        ParticipantPolicy::AllMetagraphHotkeys => meta,
        ParticipantPolicy::StakeAtLeast { min_stake } => rows
            .iter()
            .filter(|r| r.stake >= *min_stake)
            .map(|r| r.hotkey)
            .collect(),
        ParticipantPolicy::ExplicitAllowlist { hotkeys } => {
            let allow: BTreeSet<[u8; KEY_LEN]> = hotkeys.iter().copied().collect();
            allow.intersection(&meta).copied().collect()
        }
        ParticipantPolicy::AllExceptDenyList { hotkeys } => {
            let deny: BTreeSet<[u8; KEY_LEN]> = hotkeys.iter().copied().collect();
            meta.difference(&deny).copied().collect()
        }
    }
}
