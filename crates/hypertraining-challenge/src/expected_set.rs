//! Expected participant set `E` from trust-root policy + pinned metagraph.

use std::collections::BTreeSet;

use bundle::{expected_participants, metagraph_rows_from_chain};
use chain::Metagraph;
use crypto::KEY_LEN;
use thiserror::Error;
use trustroot::ParticipantPolicy;

/// Miner hotkey bytes.
pub type Hotkey = [u8; KEY_LEN];

/// Errors from expected-set derivation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExpectedSetError {
    /// Hotkey lengths / projection failed.
    #[error("invalid metagraph for expected set: {0}")]
    InvalidMetagraph(String),
}

/// Pure derivation: `E = expected_participants(policy, rows)` at a known pin.
///
/// `metagraph` MUST be the snapshot from `metagraph_at(block_hash)`.
///
/// # Errors
///
/// Invalid hotkey lengths in the metagraph projection.
pub fn expected_set_from_pinned_metagraph(
    policy: &ParticipantPolicy,
    metagraph: &Metagraph,
) -> Result<BTreeSet<Hotkey>, ExpectedSetError> {
    let rows = metagraph_rows_from_chain(&metagraph.hotkeys, None)
        .map_err(|e| ExpectedSetError::InvalidMetagraph(e.to_string()))?;
    let keys = expected_participants(policy, &rows);
    Ok(keys.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustroot::ParticipantPolicy;

    #[test]
    fn all_metagraph_hotkeys_policy() {
        let m1 = [0x11u8; 32];
        let m2 = [0x22u8; 32];
        let mg = Metagraph {
            netuid: 1,
            hotkeys: vec![m1.to_vec(), m2.to_vec()],
            owner_hotkey: vec![0u8; 32],
        };
        let e = expected_set_from_pinned_metagraph(&ParticipantPolicy::AllMetagraphHotkeys, &mg)
            .expect("e");
        assert_eq!(e.len(), 2);
        assert!(e.contains(&m1) && e.contains(&m2));
    }
}
