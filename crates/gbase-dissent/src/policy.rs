//! Three-outcome policy + quarantine (`BUNDLE_SPEC` §11.2, task 32).
//!
//! - **Class A**: inputs ok, peers agree, gateway vector ≠ local → submit local + dissent.
//! - **Quarantine**: drop one **or more** bad challenges, renormalize, submit if mass ok.
//! - **Class B**: hard failure / peer conflict / mass floor → **no** submission + dissent.
//!
//! Does **not** call CRV4 / chain extrinsics (task 33). Emits [`SubmissionIntent`] only.

use std::collections::BTreeSet;

use gbase_aggregate::{
    aggregate, renormalize_after_quarantine, ScoreOrAbsence as AggScore, VerifiedLeaf,
    ALGORITHM_VERSION as AGG_ALGO_V1, BPS_DENOM,
};
use gbase_bundle::{
    compute_metagraph_root, expected_participants, metagraph_rows_from_chain, raw_weight_payload,
    uid_map_from_rows, BundleError, EpochBundleBodyV1, EpochBundleV1, LocalTrustRoot,
    ScoreOrAbsence, MAX_CHALLENGE_ID_LEN, PROTOCOL_VERSION,
};
use gbase_chain::ChainClient;
use gbase_crosscheck::{CrossCheckFailKind, CrossCheckOutcome};
use gbase_crypto::{domain, verify_raw, KEY_LEN};
use gbase_trustroot::BPS_DENOM as TRUST_BPS;
use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};

use crate::{
    record_challenges_quarantined, record_class_b, DissentBodyV1, DissentReasonCode, DissentStore,
    DissentV1,
};

/// `sha256(scale(V))` for a final weight vector (`BUNDLE_SPEC` §8).
#[must_use]
pub fn vector_sha256(vector: &[(u16, u16)]) -> [u8; 32] {
    let digest = Sha256::digest(vector.encode());
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Recompute / compare view consumed by the policy (mapped from validator task 29/31).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecomputeView {
    /// Inputs verify; local == gateway.
    Match {
        /// Epoch.
        epoch: u64,
        /// Local vector.
        local_vector: Vec<(u16, u16)>,
        /// Bundle merkle root.
        merkle_root: [u8; 32],
        /// Vector hash.
        vector_hash: [u8; 32],
    },
    /// Class A candidate: inputs ok, vectors differ.
    VectorMismatch {
        /// Epoch.
        epoch: u64,
        /// Local recompute.
        local_vector: Vec<(u16, u16)>,
        /// Gateway vector.
        gateway_vector: Vec<(u16, u16)>,
        /// Local hash.
        local_vector_hash: [u8; 32],
        /// Gateway hash.
        gateway_vector_hash: [u8; 32],
        /// Merkle root.
        merkle_root: [u8; 32],
    },
    /// Bundle verify / aggregate input failure.
    InputInvalid {
        /// Bundle error.
        error: BundleError,
    },
    /// No bundle path (fetch / peer sample / disagreement).
    NoSubmission {
        /// Structured reason.
        reason: NoSubmitReason,
    },
}

/// Why there is no bundle-derived submission path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoSubmitReason {
    /// Peer roots disagree (class B).
    PeerRootDisagreement {
        /// Epoch.
        epoch: u64,
        /// Candidate root.
        candidate: [u8; 32],
    },
    /// Peer sample insufficient (D26).
    PeerSampleInsufficient {
        /// Epoch if known.
        epoch: u64,
    },
    /// Other fetch / transport failure.
    FetchFailed {
        /// Epoch if known.
        epoch: u64,
    },
}

/// Hook for task 33: the vector the validator **would** submit (no extrinsic here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionIntent {
    /// Epoch.
    pub epoch: u64,
    /// Weight vector (uid, weight), sorted.
    pub vector: Vec<(u16, u16)>,
    /// How the vector was produced.
    pub source: SubmissionSource,
}

/// Provenance of a submission intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionSource {
    /// Full independent recompute matched or class-A local vector.
    Recompute,
    /// After dropping quarantined challenges.
    Quarantine,
}

/// Final epoch decision after recompute + cross-check + quarantine policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpochDecision {
    /// Honest match — submit, no dissent required.
    Match {
        /// Submission hook.
        intent: SubmissionIntent,
    },
    /// Class A — submit local recompute + signed dissent.
    ClassA {
        /// Submission hook (local vector).
        intent: SubmissionIntent,
        /// Signed dissent (`VectorMismatch`).
        dissent: DissentV1,
    },
    /// Quarantine success — submit renormalized aggregate; metric emitted.
    Quarantine {
        /// Submission hook.
        intent: SubmissionIntent,
        /// Challenge ids dropped.
        dropped: Vec<Vec<u8>>,
        /// Surviving share mass (bps) before renormalize.
        surviving_mass_bps: u16,
    },
    /// Class B — **no** submission; signed dissent; alarm metric.
    ClassB {
        /// Signed dissent.
        dissent: DissentV1,
        /// Reason (mirrors dissent body).
        reason: DissentReasonCode,
    },
}

impl EpochDecision {
    /// Zero or one submission intents (class B → none).
    #[must_use]
    pub fn submissions(&self) -> Vec<&SubmissionIntent> {
        match self {
            Self::Match { intent }
            | Self::ClassA { intent, .. }
            | Self::Quarantine { intent, .. } => vec![intent],
            Self::ClassB { .. } => vec![],
        }
    }

    /// Optional dissent produced for this epoch.
    #[must_use]
    pub fn dissent(&self) -> Option<&DissentV1> {
        match self {
            Self::ClassA { dissent, .. } | Self::ClassB { dissent, .. } => Some(dissent),
            Self::Match { .. } | Self::Quarantine { .. } => None,
        }
    }
}

/// Signing material for dissent statements.
#[derive(Debug, Clone, Copy)]
pub struct DissentSigner<'a> {
    /// 32-byte mini-secret.
    pub secret: &'a [u8; 32],
    /// Corresponding public hotkey.
    pub hotkey: [u8; 32],
}

/// Apply the three-outcome policy.
///
/// When `view` is [`RecomputeView::InputInvalid`] and `bundle` is provided,
/// attempts multi-challenge quarantine before class B.
///
/// # Errors
///
/// Crypto failures while signing dissent.
#[allow(clippy::too_many_arguments)]
pub fn apply_three_outcome_policy<C: ChainClient>(
    view: &RecomputeView,
    crosscheck: &CrossCheckOutcome,
    bundle: Option<&EpochBundleV1>,
    chain: &C,
    trust: &LocalTrustRoot,
    min_share_mass_bps: u16,
    signer: DissentSigner<'_>,
    store: Option<&DissentStore>,
) -> Result<EpochDecision, crate::DissentError> {
    if let Some(decision) = class_b_from_crosscheck(crosscheck, view, signer, store)? {
        return Ok(decision);
    }

    match view {
        RecomputeView::Match {
            epoch,
            local_vector,
            ..
        } => Ok(EpochDecision::Match {
            intent: SubmissionIntent {
                epoch: *epoch,
                vector: local_vector.clone(),
                source: SubmissionSource::Recompute,
            },
        }),
        RecomputeView::VectorMismatch {
            epoch,
            local_vector,
            local_vector_hash,
            gateway_vector_hash,
            merkle_root,
            ..
        } => {
            let body = DissentBodyV1::new(
                *epoch,
                *merkle_root,
                *local_vector_hash,
                *gateway_vector_hash,
                DissentReasonCode::VectorMismatch,
            );
            let dissent = DissentV1::sign(signer.secret, signer.hotkey, body)?;
            dissent.verify()?;
            if let Some(s) = store {
                s.put(dissent.clone());
            }
            Ok(EpochDecision::ClassA {
                intent: SubmissionIntent {
                    epoch: *epoch,
                    vector: local_vector.clone(),
                    source: SubmissionSource::Recompute,
                },
                dissent,
            })
        }
        RecomputeView::NoSubmission { reason } => {
            class_b_no_submission(reason, view, signer, store)
        }
        RecomputeView::InputInvalid { error } => {
            if let Some(b) = bundle {
                if let Some(q) = try_quarantine(b, chain, trust, min_share_mass_bps, signer, store)?
                {
                    return Ok(q);
                }
            }
            class_b_from_bundle_error(error, bundle, signer, store)
        }
    }
}

fn class_b_from_crosscheck(
    crosscheck: &CrossCheckOutcome,
    view: &RecomputeView,
    signer: DissentSigner<'_>,
    store: Option<&DissentStore>,
) -> Result<Option<EpochDecision>, crate::DissentError> {
    match crosscheck {
        CrossCheckOutcome::Agreed { .. } | CrossCheckOutcome::SingleValidatorExempt { .. } => {
            Ok(None)
        }
        CrossCheckOutcome::Failed { epoch, kind, .. } => {
            let (reason, root, exp_h, act_h) = match kind {
                CrossCheckFailKind::RootDisagreement { candidate, .. } => {
                    let (eh, ah) = hashes_from_view(view);
                    (DissentReasonCode::PeerRootConflict, *candidate, eh, ah)
                }
                CrossCheckFailKind::InsufficientPeers { .. } => {
                    let (root, eh, ah) = roots_hashes_from_view(view);
                    (DissentReasonCode::PeerSampleInsufficient, root, eh, ah)
                }
            };
            Ok(Some(finish_class_b(
                *epoch, root, exp_h, act_h, reason, signer, store,
            )?))
        }
    }
}

fn class_b_no_submission(
    reason: &NoSubmitReason,
    view: &RecomputeView,
    signer: DissentSigner<'_>,
    store: Option<&DissentStore>,
) -> Result<EpochDecision, crate::DissentError> {
    let (code, epoch, root) = match reason {
        NoSubmitReason::PeerRootDisagreement { epoch, candidate } => {
            (DissentReasonCode::PeerRootConflict, *epoch, *candidate)
        }
        NoSubmitReason::PeerSampleInsufficient { epoch } => {
            let (root, _, _) = roots_hashes_from_view(view);
            (DissentReasonCode::PeerSampleInsufficient, *epoch, root)
        }
        NoSubmitReason::FetchFailed { epoch } => {
            (DissentReasonCode::BundleSignatureInvalid, *epoch, [0u8; 32])
        }
    };
    let (eh, ah) = hashes_from_view(view);
    finish_class_b(epoch, root, eh, ah, code, signer, store)
}

fn class_b_from_bundle_error(
    error: &BundleError,
    bundle: Option<&EpochBundleV1>,
    signer: DissentSigner<'_>,
    store: Option<&DissentStore>,
) -> Result<EpochDecision, crate::DissentError> {
    let reason = reason_from_bundle_error(error);
    let (epoch, root) = bundle.map_or((0, [0u8; 32]), |b| (b.body.epoch, b.body.merkle_root));
    let (eh, ah) = bundle.map_or(([0u8; 32], [0u8; 32]), |b| {
        let act = vector_sha256(&b.body.final_vector);
        ([0u8; 32], act)
    });
    finish_class_b(epoch, root, eh, ah, reason, signer, store)
}

fn finish_class_b(
    epoch: u64,
    bundle_root: [u8; 32],
    expected_vector_hash: [u8; 32],
    actual_vector_hash: [u8; 32],
    reason: DissentReasonCode,
    signer: DissentSigner<'_>,
    store: Option<&DissentStore>,
) -> Result<EpochDecision, crate::DissentError> {
    let body = DissentBodyV1::new(
        epoch,
        bundle_root,
        expected_vector_hash,
        actual_vector_hash,
        reason,
    );
    let dissent = DissentV1::sign(signer.secret, signer.hotkey, body)?;
    dissent.verify()?;
    if let Some(s) = store {
        s.put(dissent.clone());
    }
    record_class_b();
    Ok(EpochDecision::ClassB { dissent, reason })
}

fn reason_from_bundle_error(e: &BundleError) -> DissentReasonCode {
    match e {
        BundleError::ProtocolVersionUnsupported(_) => DissentReasonCode::ProtocolVersionUnsupported,
        BundleError::BundleSignatureInvalid
        | BundleError::ChallengeIdTooLong
        | BundleError::Chain(_)
        | BundleError::Codec(_)
        | BundleError::Crypto(_)
        | BundleError::InvalidMetagraph(_) => DissentReasonCode::BundleSignatureInvalid,
        BundleError::LeafSignatureInvalid => DissentReasonCode::LeafSignatureInvalid,
        BundleError::LeafChallengeKeyUnknown => DissentReasonCode::LeafChallengeKeyUnknown,
        BundleError::IncompleteParticipantSet => DissentReasonCode::IncompleteParticipantSet,
        BundleError::MerkleRootMismatch => DissentReasonCode::MerkleRootMismatch,
        BundleError::EmissionShareMismatch | BundleError::EmissionSharesSumInvalid => {
            DissentReasonCode::EmissionShareMismatch
        }
        BundleError::UidMapMismatch => DissentReasonCode::UidMapMismatch,
        BundleError::MetagraphRootMismatch => DissentReasonCode::MetagraphRootMismatch,
        BundleError::BlockHashMismatch => DissentReasonCode::BlockHashMismatch,
        BundleError::MeasurementsDigestMismatch => DissentReasonCode::MeasurementsDigestMismatch,
        BundleError::DuplicateLeaf | BundleError::LeafSortOrder => DissentReasonCode::DuplicateLeaf,
        BundleError::Aggregation(_) => DissentReasonCode::AggregationOverflow,
        BundleError::FinalVectorMismatch => DissentReasonCode::VectorMismatch,
    }
}

fn hashes_from_view(c: &RecomputeView) -> ([u8; 32], [u8; 32]) {
    match c {
        RecomputeView::Match { vector_hash, .. } => (*vector_hash, *vector_hash),
        RecomputeView::VectorMismatch {
            local_vector_hash,
            gateway_vector_hash,
            ..
        } => (*local_vector_hash, *gateway_vector_hash),
        _ => ([0u8; 32], [0u8; 32]),
    }
}

fn roots_hashes_from_view(c: &RecomputeView) -> ([u8; 32], [u8; 32], [u8; 32]) {
    match c {
        RecomputeView::Match {
            merkle_root,
            vector_hash,
            ..
        } => (*merkle_root, *vector_hash, *vector_hash),
        RecomputeView::VectorMismatch {
            merkle_root,
            local_vector_hash,
            gateway_vector_hash,
            ..
        } => (*merkle_root, *local_vector_hash, *gateway_vector_hash),
        _ => ([0u8; 32], [0u8; 32], [0u8; 32]),
    }
}

#[allow(clippy::too_many_lines)]
fn try_quarantine<C: ChainClient>(
    bundle: &EpochBundleV1,
    chain: &C,
    trust: &LocalTrustRoot,
    min_share_mass_bps: u16,
    signer: DissentSigner<'_>,
    store: Option<&DissentStore>,
) -> Result<Option<EpochDecision>, crate::DissentError> {
    let body = &bundle.body;

    if structural_hard_fail(body, chain, trust).is_err() {
        return Ok(None);
    }

    let Ok(mg) = chain.metagraph_at(&body.block_hash) else {
        return Ok(None);
    };
    let Ok(rows) = metagraph_rows_from_chain(&mg.hotkeys, None) else {
        return Ok(None);
    };

    let mut dropped: Vec<Vec<u8>> = Vec::new();
    for ch in &trust.challenges.challenges {
        if ch.emission_share_bps == 0 {
            continue;
        }
        if challenge_is_bad(body, trust, &rows, &ch.id) {
            dropped.push(ch.id.clone());
        }
    }

    if dropped.is_empty() {
        return Ok(None);
    }

    let mut surv: u32 = 0;
    for (id, bps) in &body.emission_shares {
        if dropped.iter().any(|d| d == id) {
            continue;
        }
        surv = surv.saturating_add(u32::from(*bps));
    }
    let house = u16::try_from(BPS_DENOM).unwrap_or(10_000);
    let surviving_mass_bps = u16::try_from(surv.min(u32::from(house))).unwrap_or(house);

    if surviving_mass_bps == 0 {
        return Ok(Some(finish_class_b(
            body.epoch,
            body.merkle_root,
            [0u8; 32],
            vector_sha256(&body.final_vector),
            DissentReasonCode::QuarantineExhausted,
            signer,
            store,
        )?));
    }

    if surviving_mass_bps < min_share_mass_bps {
        return Ok(Some(finish_class_b(
            body.epoch,
            body.merkle_root,
            [0u8; 32],
            vector_sha256(&body.final_vector),
            DissentReasonCode::ShareMassBelowThreshold,
            signer,
            store,
        )?));
    }

    let Ok(new_shares) = renormalize_after_quarantine(&body.emission_shares, &dropped) else {
        return Ok(Some(finish_class_b(
            body.epoch,
            body.merkle_root,
            [0u8; 32],
            vector_sha256(&body.final_vector),
            DissentReasonCode::QuarantineExhausted,
            signer,
            store,
        )?));
    };

    let dropped_set: BTreeSet<&[u8]> = dropped.iter().map(Vec::as_slice).collect();
    let verified: Vec<VerifiedLeaf> = body
        .leaves
        .iter()
        .filter(|l| !dropped_set.contains(l.challenge_id.as_slice()))
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: match &l.score_or_absence {
                ScoreOrAbsence::Score { value } => AggScore::Score { value: *value },
                ScoreOrAbsence::NoScore { reason } => AggScore::NoScore {
                    reason: *reason as u8,
                },
            },
        })
        .collect();

    let Ok(vector) = aggregate(&verified, &new_shares, &body.uid_map, AGG_ALGO_V1) else {
        return Ok(Some(finish_class_b(
            body.epoch,
            body.merkle_root,
            [0u8; 32],
            vector_sha256(&body.final_vector),
            DissentReasonCode::AggregationOverflow,
            signer,
            store,
        )?));
    };

    if vector.is_empty() {
        return Ok(Some(finish_class_b(
            body.epoch,
            body.merkle_root,
            [0u8; 32],
            vector_sha256(&body.final_vector),
            DissentReasonCode::EmptyScoreVectorNoSubmit,
            signer,
            store,
        )?));
    }

    let n = u64::try_from(dropped.len()).unwrap_or(u64::MAX);
    record_challenges_quarantined(n);

    Ok(Some(EpochDecision::Quarantine {
        intent: SubmissionIntent {
            epoch: body.epoch,
            vector,
            source: SubmissionSource::Quarantine,
        },
        dropped,
        surviving_mass_bps,
    }))
}

fn structural_hard_fail<C: ChainClient>(
    body: &EpochBundleBodyV1,
    chain: &C,
    trust: &LocalTrustRoot,
) -> Result<(), ()> {
    if body.protocol_version != PROTOCOL_VERSION {
        return Err(());
    }
    let chain_hash = chain.block_hash(body.block_b).map_err(|_| ())?;
    if chain_hash != body.block_hash {
        return Err(());
    }
    let mg = chain.metagraph_at(&body.block_hash).map_err(|_| ())?;
    let rows = metagraph_rows_from_chain(&mg.hotkeys, None).map_err(|_| ())?;
    if compute_metagraph_root(&rows) != body.metagraph_root {
        return Err(());
    }
    if uid_map_from_rows(&rows) != body.uid_map {
        return Err(());
    }
    let local_shares = trust.challenges.emission_shares();
    if body.emission_shares != local_shares {
        return Err(());
    }
    let mut sum: u32 = 0;
    for (_, bps) in &body.emission_shares {
        sum = sum.checked_add(u32::from(*bps)).ok_or(())?;
    }
    if sum != u32::from(TRUST_BPS) {
        return Err(());
    }
    if body.measurements_digest != trust.measurements_digest {
        return Err(());
    }
    Ok(())
}

fn challenge_is_bad(
    body: &EpochBundleBodyV1,
    trust: &LocalTrustRoot,
    rows: &[gbase_bundle::MetagraphRow],
    challenge_id: &[u8],
) -> bool {
    let Some(entry) = trust.challenges.get(challenge_id) else {
        return true;
    };

    let mut present: BTreeSet<[u8; KEY_LEN]> = BTreeSet::new();
    for leaf in &body.leaves {
        if leaf.challenge_id.as_slice() != challenge_id {
            continue;
        }
        if leaf.challenge_id.len() > MAX_CHALLENGE_ID_LEN {
            return true;
        }
        let payload = raw_weight_payload(
            &leaf.challenge_id,
            &leaf.miner_hotkey,
            leaf.epoch,
            &leaf.score_or_absence,
        );
        if verify_raw(
            &entry.public_key,
            domain::RAW_WEIGHT,
            &payload,
            &leaf.challenge_sig,
        )
        .is_err()
        {
            return true;
        }
        if leaf.epoch != body.epoch {
            return true;
        }
        present.insert(leaf.miner_hotkey);
    }

    let expected = expected_participants(&entry.policy, rows);
    present != expected
}
