//! D24 exact-E leaf emission for `challenge_id = "prism"`.

use std::collections::{BTreeMap, BTreeSet};

use bundle::{LeafV1, ScoreOrAbsence};
use challenge_common::{emit_signed_leaf_set as emit_signed_leaf_set_common, Hotkey};
use crypto::KEY_LEN;
use prism_challenge_task::CHALLENGE_ID_BYTES;

use challenge_common::LeafEmitError;

/// Sign exactly one leaf per `h ∈ expected` under `prism`. Refuses subset/superset (D24).
///
/// # Errors
/// Missing/unknown hotkeys or signing failure.
pub fn emit_signed_leaf_set(
    secret: &[u8; KEY_LEN],
    epoch: u64,
    expected: &BTreeSet<Hotkey>,
    scores: &BTreeMap<Hotkey, ScoreOrAbsence>,
) -> Result<BTreeMap<Hotkey, LeafV1>, LeafEmitError> {
    emit_signed_leaf_set_common(secret, CHALLENGE_ID_BYTES, epoch, expected, scores)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use bundle::ScoreOrAbsence;
    use challenge_common::{public_key_from_secret, verify_leaf_sig};
    use crypto::KEY_LEN;

    fn sk() -> [u8; KEY_LEN] {
        let mut s = [7u8; KEY_LEN];
        s[0] = 0x42;
        s
    }

    #[test]
    fn d24_refuses_missing() {
        let expected = BTreeSet::from([[1u8; KEY_LEN]]);
        let scores = BTreeMap::new();
        let e = emit_signed_leaf_set(&sk(), 1, &expected, &scores).unwrap_err();
        assert!(matches!(e, LeafEmitError::MissingHotkeys(_)));
    }

    #[test]
    fn d24_emits_exact() {
        let h = [9u8; KEY_LEN];
        let expected = BTreeSet::from([h]);
        let mut scores = BTreeMap::new();
        scores.insert(h, ScoreOrAbsence::Score { value: 42 });
        let leaves = emit_signed_leaf_set(&sk(), 3, &expected, &scores).unwrap();
        assert_eq!(leaves.len(), 1);
        let pk = public_key_from_secret(&sk()).unwrap();
        verify_leaf_sig(leaves.get(&h).unwrap(), &pk).unwrap();
    }
}
