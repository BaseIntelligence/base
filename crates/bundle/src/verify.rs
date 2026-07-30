//! Bundle verification (`BUNDLE_SPEC` §14).

use std::collections::BTreeSet;

use aggregate::{
    aggregate, ScoreOrAbsence as AggScore, VerifiedLeaf, ALGORITHM_VERSION as AGG_ALGO_V1,
};
use chain::ChainClient;
use crypto::{domain, verify_raw, KEY_LEN};
use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};
use trustroot::BPS_DENOM;

use crate::error::BundleError;
use crate::merkle_util::{
    compute_merkle_root, compute_metagraph_root, expected_participants, leaf_sort_key,
    metagraph_rows_from_chain, uid_map_from_rows,
};
use crate::sign::raw_weight_payload;
use crate::types::{
    EpochBundleBodyV1, EpochBundleV1, LocalTrustRoot, MetagraphRow, ScoreOrAbsence, VerifiedBundle,
    ALGORITHM_VERSION, MAX_CHALLENGE_ID_LEN, PROTOCOL_VERSION,
};

/// Dual equality: pairwise + sha256(scale) (§8).
#[must_use]
pub fn final_vectors_equal(a: &[(u16, u16)], b: &[(u16, u16)]) -> bool {
    if a != b {
        return false;
    }
    let ha = Sha256::digest(a.encode());
    let hb = Sha256::digest(b.encode());
    ha == hb
}

fn to_agg_score(s: &ScoreOrAbsence) -> AggScore {
    match s {
        ScoreOrAbsence::Score { value } => AggScore::Score { value: *value },
        ScoreOrAbsence::NoScore { reason } => AggScore::NoScore {
            reason: *reason as u8,
        },
    }
}

/// Verify bundle against chain + local trust root.
///
/// # Errors
///
/// See [`BundleError`].
pub fn verify_bundle<C: ChainClient>(
    bundle: &EpochBundleV1,
    chain: &C,
    trust: &LocalTrustRoot,
) -> Result<VerifiedBundle, BundleError> {
    let body = &bundle.body;

    if body.protocol_version != PROTOCOL_VERSION {
        return Err(BundleError::ProtocolVersionUnsupported(
            body.protocol_version,
        ));
    }

    let body_bytes = body.encode();
    verify_raw(
        &body.gateway_hotkey,
        domain::BUNDLE,
        &body_bytes,
        &bundle.gateway_sig,
    )
    .map_err(|_| BundleError::BundleSignatureInvalid)?;

    let chain_hash = chain
        .block_hash(body.block_b)
        .map_err(|e| BundleError::Chain(e.to_string()))?;
    if chain_hash != body.block_hash {
        return Err(BundleError::BlockHashMismatch);
    }

    let mg = chain
        .metagraph_at(&body.block_hash)
        .map_err(|e| BundleError::Chain(e.to_string()))?;
    let rows = metagraph_rows_from_chain(&mg.hotkeys, None)?;
    let expected_meta_root = compute_metagraph_root(&rows);
    if expected_meta_root != body.metagraph_root {
        return Err(BundleError::MetagraphRootMismatch);
    }
    let expected_uid_map = uid_map_from_rows(&rows);
    if expected_uid_map != body.uid_map {
        return Err(BundleError::UidMapMismatch);
    }

    let local_shares = trust.challenges.emission_shares();
    if body.emission_shares != local_shares {
        return Err(BundleError::EmissionShareMismatch);
    }
    let mut sum_bps: u32 = 0;
    for (_, bps) in &body.emission_shares {
        sum_bps = sum_bps
            .checked_add(u32::from(*bps))
            .ok_or(BundleError::EmissionSharesSumInvalid)?;
    }
    if sum_bps != u32::from(BPS_DENOM) {
        return Err(BundleError::EmissionSharesSumInvalid);
    }

    if body.measurements_digest != trust.measurements_digest {
        return Err(BundleError::MeasurementsDigestMismatch);
    }

    check_leaves(body, trust, &rows)?;

    if body.algorithm_version != ALGORITHM_VERSION {
        return Err(BundleError::FinalVectorMismatch);
    }
    let verified_leaves = body
        .leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg_score(&l.score_or_absence),
        })
        .collect::<Vec<_>>();
    let local_vector = aggregate(
        &verified_leaves,
        &body.emission_shares,
        &body.uid_map,
        AGG_ALGO_V1,
    )
    .map_err(|e| BundleError::Aggregation(e.to_string()))?;
    if !final_vectors_equal(&local_vector, &body.final_vector) {
        return Err(BundleError::FinalVectorMismatch);
    }

    Ok(VerifiedBundle { body: body.clone() })
}

fn check_leaves(
    body: &EpochBundleBodyV1,
    trust: &LocalTrustRoot,
    rows: &[MetagraphRow],
) -> Result<(), BundleError> {
    let mut prev_key: Option<Vec<u8>> = None;
    let mut seen: BTreeSet<(Vec<u8>, [u8; KEY_LEN])> = BTreeSet::new();
    for leaf in &body.leaves {
        if leaf.challenge_id.len() > MAX_CHALLENGE_ID_LEN {
            return Err(BundleError::ChallengeIdTooLong);
        }
        let key = leaf_sort_key(&leaf.challenge_id, &leaf.miner_hotkey);
        if let Some(ref p) = prev_key {
            if key < *p {
                return Err(BundleError::LeafSortOrder);
            }
            if key == *p {
                return Err(BundleError::DuplicateLeaf);
            }
        }
        prev_key = Some(key);
        if !seen.insert((leaf.challenge_id.clone(), leaf.miner_hotkey)) {
            return Err(BundleError::DuplicateLeaf);
        }

        let entry = trust
            .challenges
            .get(&leaf.challenge_id)
            .ok_or(BundleError::LeafChallengeKeyUnknown)?;

        let payload = raw_weight_payload(
            &leaf.challenge_id,
            &leaf.miner_hotkey,
            leaf.epoch,
            &leaf.score_or_absence,
        );
        verify_raw(
            &entry.public_key,
            domain::RAW_WEIGHT,
            &payload,
            &leaf.challenge_sig,
        )
        .map_err(|_| BundleError::LeafSignatureInvalid)?;

        if leaf.epoch != body.epoch {
            return Err(BundleError::LeafSignatureInvalid);
        }
    }

    let recomputed = compute_merkle_root(&body.leaves);
    if recomputed != body.merkle_root {
        return Err(BundleError::MerkleRootMismatch);
    }

    for ch in &trust.challenges.challenges {
        if ch.emission_share_bps == 0 {
            continue;
        }
        let expected = expected_participants(&ch.policy, rows);
        let mut present: BTreeSet<[u8; KEY_LEN]> = BTreeSet::new();
        for leaf in &body.leaves {
            if leaf.challenge_id == ch.id {
                present.insert(leaf.miner_hotkey);
            }
        }
        if present != expected {
            return Err(BundleError::IncompleteParticipantSet);
        }
    }

    Ok(())
}
