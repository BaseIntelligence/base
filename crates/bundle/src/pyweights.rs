//! The single entry point from a bundle to the served Python-parity vector.
//!
//! Gateway seal, `verify_bundle`, the validator's independent recompute, and the
//! `/v1/weights/latest` projection all route through here, so the leaf → aggregate-input
//! conversion exists exactly once and the four sites cannot drift apart.

use aggregate::{
    aggregate_python_vector, PythonVector, ScoreOrAbsence as AggScore, VerifiedLeaf,
    ALGORITHM_VERSION as AGG_ALGO_V1,
};

use crate::error::BundleError;
use crate::types::{EpochBundleBodyV1, LeafV1, ScoreOrAbsence};

fn to_agg_score(s: &ScoreOrAbsence) -> AggScore {
    match s {
        ScoreOrAbsence::Score { value } => AggScore::Score { value: *value },
        ScoreOrAbsence::NoScore { reason } => AggScore::NoScore {
            reason: *reason as u8,
        },
    }
}

/// Compute the served vector from raw seal ingredients (before a body exists).
///
/// # Errors
///
/// [`BundleError::Aggregation`] wrapping the adapter's failure text.
pub fn python_weights_from_parts(
    leaves: &[LeafV1],
    emission_shares: &[(Vec<u8>, u16)],
    uid_map: &[([u8; crypto::KEY_LEN], u16)],
) -> Result<PythonVector, BundleError> {
    let verified: Vec<VerifiedLeaf> = leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg_score(&l.score_or_absence),
        })
        .collect();
    aggregate_python_vector(&verified, emission_shares, uid_map, AGG_ALGO_V1)
        .map_err(|e| BundleError::Aggregation(e.to_string()))
}

/// Recompute the served vector from a signed body.
///
/// The result is a pure function of the body, which is why nothing persists the floats.
///
/// # Errors
///
/// [`BundleError::Aggregation`] wrapping the adapter's failure text.
pub fn python_weights(body: &EpochBundleBodyV1) -> Result<PythonVector, BundleError> {
    python_weights_from_parts(&body.leaves, &body.emission_shares, &body.uid_map)
}
