//! D24 exact-E leaf emission and signature helpers.

use std::collections::{BTreeMap, BTreeSet};

use bundle::{make_signed_leaf, LeafV1, ScoreOrAbsence};
use crypto::KEY_LEN;
use thiserror::Error;

use crate::expected_set::Hotkey;
use hypertraining_challenge_task::CHALLENGE_ID_BYTES;

/// Exact-E signed leaf emission errors (D24 — silence / subset is a bug).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LeafEmitError {
    /// `scores` missing one or more hotkeys from `E`.
    #[error("D24 incomplete leaf set; missing hotkey(s): {0}")]
    MissingHotkeys(String),
    /// `scores` contains hotkeys outside `E`.
    #[error("hotkey(s) not in expected set E: {0}")]
    UnknownHotkeys(String),
    /// `make_signed_leaf` failed.
    #[error("sign leaf: {0}")]
    Sign(String),
}

/// Sign exactly one leaf per `h ∈ expected`. Refuses subset/superset (D24).
///
/// # Errors
/// Missing/unknown hotkeys (named) or signing failure. Never returns a partial set.
pub fn emit_signed_leaf_set(
    secret: &[u8; KEY_LEN],
    epoch: u64,
    expected: &BTreeSet<Hotkey>,
    scores: &BTreeMap<Hotkey, ScoreOrAbsence>,
) -> Result<BTreeMap<Hotkey, LeafV1>, LeafEmitError> {
    let keys: BTreeSet<_> = scores.keys().copied().collect();
    let miss: Vec<_> = expected.difference(&keys).map(hex::encode).collect();
    if !miss.is_empty() {
        return Err(LeafEmitError::MissingHotkeys(miss.join(",")));
    }
    let extra: Vec<_> = keys.difference(expected).map(hex::encode).collect();
    if !extra.is_empty() {
        return Err(LeafEmitError::UnknownHotkeys(extra.join(",")));
    }
    scores
        .iter()
        .map(|(h, s)| {
            make_signed_leaf(secret, CHALLENGE_ID_BYTES, *h, epoch, s.clone())
                .map(|l| (*h, l))
                .map_err(|e| LeafEmitError::Sign(e.to_string()))
        })
        .collect()
}

/// Derive 32-byte public key from a mini-secret (for tests / ceremony checks).
///
/// # Errors
/// Invalid mini-secret.
pub fn public_key_from_secret(secret: &[u8; KEY_LEN]) -> Result<[u8; KEY_LEN], String> {
    let sk = crypto::secret_from_bytes(secret).map_err(|e| e.to_string())?;
    Ok(sk.to_public().to_bytes())
}

/// Verify a leaf signature against a challenge public key.
///
/// # Errors
/// Returns `Err` when verification fails.
pub fn verify_leaf_sig(leaf: &LeafV1, challenge_pk: &[u8; KEY_LEN]) -> Result<(), String> {
    use crypto::{domain, verify_raw};
    let payload = bundle::raw_weight_payload(
        &leaf.challenge_id,
        &leaf.miner_hotkey,
        leaf.epoch,
        &leaf.score_or_absence,
    );
    verify_raw(
        challenge_pk,
        domain::RAW_WEIGHT,
        &payload,
        &leaf.challenge_sig,
    )
    .map_err(|e| e.to_string())
}
