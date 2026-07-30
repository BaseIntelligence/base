//! Gateway epoch seal helper (`BUNDLE_SPEC` §7, §13): pin chain, D24, aggregate, sign.

use std::collections::BTreeSet;

use gbase_aggregate::{
    aggregate, ScoreOrAbsence as AggScore, VerifiedLeaf, ALGORITHM_VERSION as AGG_V,
};
use gbase_chain::ChainClient;
use gbase_crypto::{secret_from_bytes, KEY_LEN};
use gbase_trustroot::{ChallengesBody, BPS_DENOM};

use crate::error::BundleError;
use crate::merkle_util::{
    compute_merkle_root, compute_metagraph_root, expected_participants, metagraph_rows_from_chain,
    sort_leaves, uid_map_from_rows,
};
use crate::sign::sign_bundle;
use crate::types::{
    EpochBundleBodyV1, EpochBundleV1, LeafV1, LocalTrustRoot, ScoreOrAbsence, ALGORITHM_VERSION,
    PROTOCOL_VERSION,
};

/// Seal inputs beyond chain + leaves + trust root.
#[derive(Debug, Clone)]
pub struct SealParams {
    /// Epoch index.
    pub epoch: u64,
    /// Subnet netuid.
    pub netuid: u16,
    /// Inclusive epoch end block.
    pub block_b: u64,
    /// Gateway mini-secret (`gbase-bundle-v1`).
    pub gateway_secret: [u8; KEY_LEN],
}

/// Build and sign an epoch bundle. Does **not** persist.
///
/// Fails closed on incomplete participant set (D24) before signing.
///
/// # Errors
///
/// [`BundleError`] on chain, completeness, aggregation, or crypto failure.
pub fn build_sealed_bundle<C: ChainClient + ?Sized>(
    chain: &C,
    trust: &LocalTrustRoot,
    mut leaves: Vec<LeafV1>,
    params: &SealParams,
) -> Result<EpochBundleV1, BundleError> {
    let block_hash = chain
        .block_hash(params.block_b)
        .map_err(|e| BundleError::Chain(e.to_string()))?;
    let mg = chain
        .metagraph_at(&block_hash)
        .map_err(|e| BundleError::Chain(e.to_string()))?;
    let rows = metagraph_rows_from_chain(&mg.hotkeys, None)?;
    let metagraph_root = compute_metagraph_root(&rows);
    let uid_map = uid_map_from_rows(&rows);

    let emission_shares = trust.challenges.emission_shares();
    validate_shares_sum(&emission_shares)?;
    assert_participant_completeness(&trust.challenges, &rows, &leaves)?;

    sort_leaves(&mut leaves);
    let merkle_root = compute_merkle_root(&leaves);

    let verified: Vec<VerifiedLeaf> = leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    let final_vector = aggregate(&verified, &emission_shares, &uid_map, AGG_V)
        .map_err(|e| BundleError::Aggregation(e.to_string()))?;

    let gateway_hotkey = secret_from_bytes(&params.gateway_secret)
        .map_err(|e| BundleError::Crypto(e.to_string()))?
        .to_public()
        .to_bytes();

    let body = EpochBundleBodyV1 {
        protocol_version: PROTOCOL_VERSION,
        epoch: params.epoch,
        netuid: params.netuid,
        block_b: params.block_b,
        block_hash,
        metagraph_root,
        algorithm_version: ALGORITHM_VERSION,
        emission_shares,
        measurements_digest: trust.measurements_digest,
        uid_map,
        leaves,
        merkle_root,
        final_vector,
        gateway_hotkey,
    };
    sign_bundle(&params.gateway_secret, body)
}

fn validate_shares_sum(shares: &[(Vec<u8>, u16)]) -> Result<(), BundleError> {
    let mut sum: u32 = 0;
    for (_, bps) in shares {
        sum = sum
            .checked_add(u32::from(*bps))
            .ok_or(BundleError::EmissionSharesSumInvalid)?;
    }
    if sum != u32::from(BPS_DENOM) {
        return Err(BundleError::EmissionSharesSumInvalid);
    }
    Ok(())
}

fn to_agg(s: &ScoreOrAbsence) -> AggScore {
    match s {
        ScoreOrAbsence::Score { value } => AggScore::Score { value: *value },
        ScoreOrAbsence::NoScore { reason } => AggScore::NoScore {
            reason: *reason as u8,
        },
    }
}

fn assert_participant_completeness(
    challenges: &ChallengesBody,
    rows: &[crate::types::MetagraphRow],
    leaves: &[LeafV1],
) -> Result<(), BundleError> {
    let mut leaf_index: BTreeSet<(Vec<u8>, [u8; KEY_LEN])> = BTreeSet::new();
    for leaf in leaves {
        let key = (leaf.challenge_id.clone(), leaf.miner_hotkey);
        if !leaf_index.insert(key) {
            return Err(BundleError::DuplicateLeaf);
        }
    }
    let mut expected_keys: BTreeSet<(Vec<u8>, [u8; KEY_LEN])> = BTreeSet::new();
    for entry in &challenges.challenges {
        if entry.emission_share_bps == 0 {
            continue;
        }
        for hk in expected_participants(&entry.policy, rows) {
            expected_keys.insert((entry.id.clone(), hk));
            if !leaf_index.contains(&(entry.id.clone(), hk)) {
                return Err(BundleError::IncompleteParticipantSet);
            }
        }
    }
    for key in &leaf_index {
        if !expected_keys.contains(key) {
            return Err(BundleError::IncompleteParticipantSet);
        }
    }
    Ok(())
}
