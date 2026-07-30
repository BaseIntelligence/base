//! Sign leaves and gateway bundle envelopes.

use crypto::{domain, sign_raw, KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::Encode;

use crate::error::BundleError;
use crate::merkle_util::{compute_merkle_root, sort_leaves};
use crate::types::{
    EpochBundleBodyV1, EpochBundleV1, LeafV1, RawWeightBodyV1, ScoreOrAbsence, MAX_CHALLENGE_ID_LEN,
};

/// SCALE payload for challenge leaf signature (`RawWeightBodyV1`).
#[must_use]
pub fn raw_weight_payload(
    challenge_id: &[u8],
    miner_hotkey: &[u8; KEY_LEN],
    epoch: u64,
    score_or_absence: &ScoreOrAbsence,
) -> Vec<u8> {
    RawWeightBodyV1 {
        challenge_id: challenge_id.to_vec(),
        miner_hotkey: *miner_hotkey,
        epoch,
        score_or_absence: score_or_absence.clone(),
    }
    .encode()
}

/// Sign a leaf's challenge signature with a challenge mini-secret.
///
/// # Errors
///
/// Crypto errors; challenge id too long.
pub fn sign_leaf_challenge(
    challenge_secret: &[u8; KEY_LEN],
    challenge_id: &[u8],
    miner_hotkey: &[u8; KEY_LEN],
    epoch: u64,
    score_or_absence: &ScoreOrAbsence,
) -> Result<[u8; SIGNATURE_LEN], BundleError> {
    if challenge_id.len() > MAX_CHALLENGE_ID_LEN {
        return Err(BundleError::ChallengeIdTooLong);
    }
    let payload = raw_weight_payload(challenge_id, miner_hotkey, epoch, score_or_absence);
    sign_raw(challenge_secret, domain::RAW_WEIGHT, &payload)
        .map_err(|e| BundleError::Crypto(e.to_string()))
}

/// Build a fully signed [`LeafV1`].
///
/// # Errors
///
/// Crypto / id length.
pub fn make_signed_leaf(
    challenge_secret: &[u8; KEY_LEN],
    challenge_id: &[u8],
    miner_hotkey: [u8; KEY_LEN],
    epoch: u64,
    score_or_absence: ScoreOrAbsence,
) -> Result<LeafV1, BundleError> {
    let challenge_sig = sign_leaf_challenge(
        challenge_secret,
        challenge_id,
        &miner_hotkey,
        epoch,
        &score_or_absence,
    )?;
    Ok(LeafV1 {
        challenge_id: challenge_id.to_vec(),
        miner_hotkey,
        epoch,
        score_or_absence,
        challenge_sig,
    })
}

/// Sign bundle body with gateway mini-secret → [`EpochBundleV1`].
///
/// # Errors
///
/// Crypto errors.
pub fn sign_bundle(
    gateway_secret: &[u8; KEY_LEN],
    body: EpochBundleBodyV1,
) -> Result<EpochBundleV1, BundleError> {
    let payload = body.encode();
    let gateway_sig = sign_raw(gateway_secret, domain::BUNDLE, &payload)
        .map_err(|e| BundleError::Crypto(e.to_string()))?;
    Ok(EpochBundleV1 { body, gateway_sig })
}

/// Sort leaves and fill `merkle_root` on a body.
pub fn finalize_body_merkle(body: &mut EpochBundleBodyV1) {
    sort_leaves(&mut body.leaves);
    body.merkle_root = compute_merkle_root(&body.leaves);
}
