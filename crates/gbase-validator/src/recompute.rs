//! Bundle fetch, independent recomputation, and final-vector comparison (task 29/30).
//!
//! Flow (`BUNDLE_SPEC` §8, §9, §11.2, §14):
//! 1. Fetch sealed bundle bytes from the gateway (allowlisted path only).
//! 2. On gateway failure, fetch by content-addressed root from metagraph peers
//!    (task 30), bound to the expected `(epoch, root)`.
//! 3. Verify against the validator's **own** local trust root + chain.
//! 4. Independently `gbase_aggregate` the verified leaves.
//! 5. Dual-compare gateway `final_vector` vs local: full equality **and**
//!    `sha256(scale(V))`.
//! 6. Persist verified bundles into the local mirror store for peer serve.
//!
//! **No last-known-good:** a failed fetch/verify never returns a prior epoch's
//! vector as success. This module never submits extrinsics (task 33 owns that).

use gbase_aggregate::{
    aggregate, ScoreOrAbsence as AggScore, VerifiedLeaf, ALGORITHM_VERSION as AGG_ALGO_V1,
};
use gbase_bundle::{
    decode_bundle, final_vectors_equal, verify_bundle, BundleError, EpochBundleBodyV1,
    EpochBundleV1, LocalTrustRoot, ALGORITHM_VERSION,
};
use gbase_chain::ChainClient;
use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::coordination::{CoordinationClient, CoordinationError};
use crate::mirror::SharedMirrorStore;
use crate::peers::PeerBook;

/// Why the validator must not submit weights for this epoch (no class-A path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoSubmissionReason {
    /// No gateway base URL configured.
    NoGatewayConfigured,
    /// Transport / DNS / connection failure talking to the gateway.
    GatewayUnreachable(String),
    /// Gateway returned a non-success HTTP status (including 404).
    GatewayHttp {
        /// HTTP status code.
        status: u16,
        /// Path attempted.
        path: String,
    },
    /// Allowlist / client misconfiguration (should not happen for bundle paths).
    PathNotAllowed {
        /// Path that was refused.
        path: String,
    },
    /// Gateway failed and no `(epoch, root)` binding was supplied for peer fetch.
    NoExpectedRootForPeerFetch,
    /// No metagraph peers available (or peer book empty after discovery).
    NoPeersAvailable,
    /// All peers failed or returned unbound/wrong content.
    PeerFetchExhausted {
        /// Last peer error detail.
        detail: String,
    },
    /// A peer returned bytes whose merkle root ≠ requested root.
    PeerRootMismatch,
    /// A peer returned a well-formed bundle for a different epoch.
    PeerEpochMismatch {
        /// Epoch inside the peer body.
        got: u64,
        /// Epoch the caller bound the fetch to.
        expected: u64,
    },
    /// Peer root sample below `min_peer_sample` while other validators exist (D26).
    PeerSampleInsufficient {
        /// Verified agreeing peer statements collected.
        reachable: usize,
        /// Configured minimum sample size.
        required: u32,
    },
    /// Verified peers disagree on the merkle root for this epoch (class B hook).
    PeerRootDisagreement {
        /// Epoch under check.
        epoch: u64,
        /// This validator's candidate root.
        candidate: [u8; 32],
        /// Number of verified peer statements retained as evidence.
        sample_len: usize,
    },
}

impl std::fmt::Display for NoSubmissionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoGatewayConfigured => write!(f, "gateway endpoint not configured"),
            Self::GatewayUnreachable(e) => write!(f, "gateway unreachable: {e}"),
            Self::GatewayHttp { status, path } => {
                write!(f, "gateway HTTP {status} for {path}")
            }
            Self::PathNotAllowed { path } => {
                write!(f, "gateway path not allowed: {path}")
            }
            Self::NoExpectedRootForPeerFetch => {
                write!(f, "gateway failed and no expected root for peer fetch")
            }
            Self::NoPeersAvailable => write!(f, "no peers available for bundle fetch"),
            Self::PeerFetchExhausted { detail } => {
                write!(f, "peer fetch exhausted: {detail}")
            }
            Self::PeerRootMismatch => {
                write!(f, "peer bundle merkle root does not match request")
            }
            Self::PeerEpochMismatch { got, expected } => {
                write!(f, "peer bundle epoch {got} != expected {expected}")
            }
            Self::PeerSampleInsufficient {
                reachable,
                required,
            } => write!(
                f,
                "peer sample insufficient: reachable={reachable} required={required} (Degraded)"
            ),
            Self::PeerRootDisagreement {
                epoch,
                candidate,
                sample_len,
            } => write!(
                f,
                "peer root disagreement epoch={epoch} sample_len={sample_len} candidate={}",
                hex::encode(candidate)
            ),
        }
    }
}

/// Outcome of fetch + verify + independent recompute + dual comparison.
///
/// Downstream tasks (dissent / CRV4) consume these variants; this module does
/// **not** submit weights or dissent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonOutcome {
    /// Inputs verify; local recompute equals gateway `final_vector` under §8.
    Match {
        /// Epoch index from the bundle body.
        epoch: u64,
        /// Independently recomputed weight vector (uid, weight), sorted by uid.
        local_vector: Vec<(u16, u16)>,
        /// Gateway-published `final_vector` (equal to `local_vector`).
        gateway_vector: Vec<(u16, u16)>,
        /// `sha256(scale(local_vector))` (== gateway hash when Match).
        vector_hash: [u8; 32],
        /// Bundle merkle root (for peer/dissent binding later).
        merkle_root: [u8; 32],
    },
    /// Class A (`BUNDLE_SPEC` §11.2): inputs ok; gateway vector ≠ local recompute.
    VectorMismatch {
        /// Epoch index.
        epoch: u64,
        /// Local independent recompute (what class-A submit would use).
        local_vector: Vec<(u16, u16)>,
        /// Gateway-published vector that failed dual equality.
        gateway_vector: Vec<(u16, u16)>,
        /// `sha256(scale(local_vector))`.
        local_vector_hash: [u8; 32],
        /// `sha256(scale(gateway_vector))`.
        gateway_vector_hash: [u8; 32],
        /// Bundle merkle root.
        merkle_root: [u8; 32],
    },
    /// Class B input failure: signatures, trust root, metagraph, participants, etc.
    InputInvalid {
        /// Underlying bundle verify / aggregate failure.
        error: BundleError,
    },
    /// No weight submission path: fetch failed. **No extrinsic attempted.**
    NoSubmission {
        /// Why fetch could not produce a bundle.
        reason: NoSubmissionReason,
    },
}

/// Errors internal to recompute helpers (mapped into [`ComparisonOutcome`]).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RecomputeError {
    /// Aggregation failed after inputs were accepted.
    #[error("independent aggregate: {0}")]
    Aggregate(String),
}

/// `sha256(scale(V))` for a final weight vector (`BUNDLE_SPEC` §8).
#[must_use]
pub fn vector_sha256(vector: &[(u16, u16)]) -> [u8; 32] {
    let digest = Sha256::digest(vector.encode());
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Map leaves on a body into aggregate inputs and run `gbase_aggregate`.
///
/// Call only after leaf signatures / participant checks have passed (or when
/// `verify_bundle` failed solely on final-vector equality).
///
/// # Errors
///
/// [`RecomputeError::Aggregate`] on aggregation failure.
pub fn independent_aggregate(body: &EpochBundleBodyV1) -> Result<Vec<(u16, u16)>, RecomputeError> {
    let verified: Vec<VerifiedLeaf> = body
        .leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: match &l.score_or_absence {
                gbase_bundle::ScoreOrAbsence::Score { value } => AggScore::Score { value: *value },
                gbase_bundle::ScoreOrAbsence::NoScore { reason } => AggScore::NoScore {
                    reason: *reason as u8,
                },
            },
        })
        .collect();
    aggregate(&verified, &body.emission_shares, &body.uid_map, AGG_ALGO_V1)
        .map_err(|e| RecomputeError::Aggregate(e.to_string()))
}

/// Verify + independently recompute + dual-compare a decoded bundle.
///
/// Does **not** touch the chain write path. Uses the caller's local trust root
/// only (never HTTP-sourced challenges/measurements).
#[must_use]
pub fn compare_bundle<C: ChainClient>(
    bundle: &EpochBundleV1,
    chain: &C,
    trust: &LocalTrustRoot,
) -> ComparisonOutcome {
    match verify_bundle(bundle, chain, trust) {
        Ok(verified) => match finish_match(&verified.body) {
            Ok(outcome) => outcome,
            Err(e) => ComparisonOutcome::InputInvalid {
                error: BundleError::Aggregation(e.to_string()),
            },
        },
        Err(BundleError::FinalVectorMismatch) => {
            // Inputs (through leaf checks) passed inside verify; only vector/algo failed.
            if bundle.body.algorithm_version != ALGORITHM_VERSION {
                return ComparisonOutcome::InputInvalid {
                    error: BundleError::FinalVectorMismatch,
                };
            }
            match independent_aggregate(&bundle.body) {
                Ok(local_vector) => {
                    let gateway_vector = bundle.body.final_vector.clone();
                    if final_vectors_equal(&local_vector, &gateway_vector) {
                        // Defensive: verify said mismatch but dual check agrees.
                        ComparisonOutcome::Match {
                            epoch: bundle.body.epoch,
                            vector_hash: vector_sha256(&local_vector),
                            local_vector,
                            gateway_vector,
                            merkle_root: bundle.body.merkle_root,
                        }
                    } else {
                        ComparisonOutcome::VectorMismatch {
                            epoch: bundle.body.epoch,
                            local_vector_hash: vector_sha256(&local_vector),
                            gateway_vector_hash: vector_sha256(&gateway_vector),
                            local_vector,
                            gateway_vector,
                            merkle_root: bundle.body.merkle_root,
                        }
                    }
                }
                Err(RecomputeError::Aggregate(msg)) => ComparisonOutcome::InputInvalid {
                    error: BundleError::Aggregation(msg),
                },
            }
        }
        Err(error) => ComparisonOutcome::InputInvalid { error },
    }
}

fn finish_match(body: &EpochBundleBodyV1) -> Result<ComparisonOutcome, RecomputeError> {
    let local_vector = independent_aggregate(body)?;
    let gateway_vector = body.final_vector.clone();
    if !final_vectors_equal(&local_vector, &gateway_vector) {
        return Ok(ComparisonOutcome::VectorMismatch {
            epoch: body.epoch,
            local_vector_hash: vector_sha256(&local_vector),
            gateway_vector_hash: vector_sha256(&gateway_vector),
            local_vector,
            gateway_vector,
            merkle_root: body.merkle_root,
        });
    }
    Ok(ComparisonOutcome::Match {
        epoch: body.epoch,
        vector_hash: vector_sha256(&local_vector),
        local_vector,
        gateway_vector,
        merkle_root: body.merkle_root,
    })
}

/// Decode SCALE bytes then [`compare_bundle`].
#[must_use]
pub fn compare_bundle_bytes<C: ChainClient>(
    bytes: &[u8],
    chain: &C,
    trust: &LocalTrustRoot,
) -> ComparisonOutcome {
    match decode_bundle(bytes) {
        Ok(bundle) => compare_bundle(&bundle, chain, trust),
        Err(error) => ComparisonOutcome::InputInvalid { error },
    }
}

fn map_coord_err(err: CoordinationError) -> ComparisonOutcome {
    let reason = match err {
        CoordinationError::NoGateway => NoSubmissionReason::NoGatewayConfigured,
        CoordinationError::Transport(e) => NoSubmissionReason::GatewayUnreachable(e),
        CoordinationError::HttpStatus { status, path } => {
            NoSubmissionReason::GatewayHttp { status, path }
        }
        CoordinationError::PathNotAllowed { path } => NoSubmissionReason::PathNotAllowed { path },
    };
    ComparisonOutcome::NoSubmission { reason }
}

/// Expected content-addressed binding for peer fetch (task 30).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedBundle {
    /// Epoch the caller intends to evaluate.
    pub epoch: u64,
    /// Merkle root the caller intends to fetch (`body.merkle_root`).
    pub root: [u8; 32],
}

/// Persist SCALE bytes into the mirror when the outcome proves inputs verified.
///
/// Stores on [`ComparisonOutcome::Match`] and [`ComparisonOutcome::VectorMismatch`]
/// only (class A still has a trustworthy merkle root). Never stores `InputInvalid`.
pub fn maybe_persist_verified(
    store: &SharedMirrorStore,
    bytes: &[u8],
    outcome: &ComparisonOutcome,
) {
    let (epoch, root) = match outcome {
        ComparisonOutcome::Match {
            epoch,
            merkle_root,
            ..
        }
        | ComparisonOutcome::VectorMismatch {
            epoch,
            merkle_root,
            ..
        } => (*epoch, *merkle_root),
        ComparisonOutcome::InputInvalid { .. } | ComparisonOutcome::NoSubmission { .. } => return,
    };
    // Defence-in-depth: refuse to index under a root that does not match the body.
    match EpochBundleV1::decode_bytes(bytes) {
        Ok(b) if b.body.merkle_root == root && b.body.epoch == epoch => {
            store.put_verified(epoch, root, bytes.to_vec());
        }
        _ => {}
    }
}

/// Fetch `GET /v1/bundle/{epoch}` and run independent recompute/compare.
///
/// On any fetch failure returns [`ComparisonOutcome::NoSubmission`] and never
/// attempts a chain extrinsic. Does **not** fall back to another epoch (no LKG).
/// Does **not** attempt peer fetch — use [`fetch_and_compare_with_mirror`].
pub async fn fetch_and_compare<C: ChainClient>(
    client: &CoordinationClient,
    epoch: u64,
    chain: &C,
    trust: &LocalTrustRoot,
) -> ComparisonOutcome {
    match client.fetch_bundle(epoch).await {
        Ok(bytes) => compare_bundle_bytes(&bytes, chain, trust),
        Err(e) => map_coord_err(e),
    }
}

/// Gateway-first fetch with verified-mirror persist and peer root fallback (task 30).
///
/// 1. Try gateway `GET /v1/bundle/{epoch}`.
/// 2. On success: compare; if Match/VectorMismatch, persist into `store`.
/// 3. On gateway failure: if `expected` is set, discover peers from the metagraph
///    via `peers` and `GET /v1/bundle/root/{root}` from each until one binds to
///    `(expected.epoch, expected.root)` and compares; reject wrong root/epoch.
/// 4. Never falls back to another epoch (no LKG). Never submits extrinsics.
pub async fn fetch_and_compare_with_mirror<C: ChainClient>(
    client: &CoordinationClient,
    epoch: u64,
    chain: &C,
    trust: &LocalTrustRoot,
    store: &SharedMirrorStore,
    peers: &PeerBook,
    expected: Option<ExpectedBundle>,
) -> ComparisonOutcome {
    match client.fetch_bundle(epoch).await {
        Ok(bytes) => {
            let outcome = compare_bundle_bytes(&bytes, chain, trust);
            maybe_persist_verified(store, &bytes, &outcome);
            outcome
        }
        Err(gw_err) => {
            // Local mirror hit (same process already verified this root).
            if let Some(exp) = expected {
                if exp.epoch == epoch {
                    if let Some(bytes) = store.get_by_root(&exp.root) {
                        if let Some(out) = accept_bound_bytes(&bytes, exp, chain, trust, store) {
                            return out;
                        }
                    }
                }
            }
            match expected {
                None => map_coord_err(gw_err),
                Some(exp) if exp.epoch != epoch => ComparisonOutcome::NoSubmission {
                    reason: NoSubmissionReason::PeerEpochMismatch {
                        got: exp.epoch,
                        expected: epoch,
                    },
                },
                Some(exp) => fetch_from_peers(client, chain, trust, store, peers, exp, gw_err).await,
            }
        }
    }
}

/// Validate peer/local bytes against the expected `(epoch, root)` binding.
fn accept_bound_bytes<C: ChainClient>(
    bytes: &[u8],
    expected: ExpectedBundle,
    chain: &C,
    trust: &LocalTrustRoot,
    store: &SharedMirrorStore,
) -> Option<ComparisonOutcome> {
    let Ok(bundle) = decode_bundle(bytes) else {
        return None;
    };
    if bundle.body.merkle_root != expected.root {
        return Some(ComparisonOutcome::NoSubmission {
            reason: NoSubmissionReason::PeerRootMismatch,
        });
    }
    if bundle.body.epoch != expected.epoch {
        return Some(ComparisonOutcome::NoSubmission {
            reason: NoSubmissionReason::PeerEpochMismatch {
                got: bundle.body.epoch,
                expected: expected.epoch,
            },
        });
    }
    let outcome = compare_bundle(&bundle, chain, trust);
    maybe_persist_verified(store, bytes, &outcome);
    Some(outcome)
}

async fn fetch_from_peers<C: ChainClient>(
    client: &CoordinationClient,
    chain: &C,
    trust: &LocalTrustRoot,
    store: &SharedMirrorStore,
    peers: &PeerBook,
    expected: ExpectedBundle,
    gw_err: CoordinationError,
) -> ComparisonOutcome {
    let discovered = match peers.discover_from_chain(chain) {
        Ok(p) => p,
        Err(e) => {
            return ComparisonOutcome::NoSubmission {
                reason: NoSubmissionReason::PeerFetchExhausted {
                    detail: format!("metagraph discover: {e}; gateway: {gw_err}"),
                },
            };
        }
    };
    if discovered.is_empty() {
        // Preserve gateway failure flavour when no peers exist.
        return match map_coord_err(gw_err) {
            ComparisonOutcome::NoSubmission { reason } => ComparisonOutcome::NoSubmission {
                reason: match reason {
                    NoSubmissionReason::GatewayUnreachable(_)
                    | NoSubmissionReason::GatewayHttp { .. }
                    | NoSubmissionReason::NoGatewayConfigured => NoSubmissionReason::NoPeersAvailable,
                    other => other,
                },
            },
            other => other,
        };
    }

    let mut last_detail = format!("gateway: {gw_err}");
    let mut saw_root_mismatch = false;
    let mut saw_epoch_mismatch: Option<(u64, u64)> = None;

    for peer in &discovered {
        match client
            .fetch_bundle_by_root_from(&peer.base_url, &expected.root)
            .await
        {
            Ok(bytes) => match accept_bound_bytes(&bytes, expected, chain, trust, store) {
                Some(ComparisonOutcome::NoSubmission {
                    reason: NoSubmissionReason::PeerRootMismatch,
                }) => {
                    saw_root_mismatch = true;
                    last_detail = format!("peer {} root mismatch", peer.base_url);
                }
                Some(ComparisonOutcome::NoSubmission {
                    reason: NoSubmissionReason::PeerEpochMismatch { got, expected },
                }) => {
                    saw_epoch_mismatch = Some((got, expected));
                    last_detail = format!(
                        "peer {} epoch mismatch got={got} expected={expected}",
                        peer.base_url
                    );
                }
                Some(outcome) => return outcome,
                None => {
                    last_detail = format!("peer {} decode failed", peer.base_url);
                }
            },
            Err(e) => {
                last_detail = format!("peer {} => {e}", peer.base_url);
            }
        }
    }

    if saw_root_mismatch && saw_epoch_mismatch.is_none() {
        return ComparisonOutcome::NoSubmission {
            reason: NoSubmissionReason::PeerRootMismatch,
        };
    }
    if let Some((got, expected)) = saw_epoch_mismatch {
        return ComparisonOutcome::NoSubmission {
            reason: NoSubmissionReason::PeerEpochMismatch { got, expected },
        };
    }
    ComparisonOutcome::NoSubmission {
        reason: NoSubmissionReason::PeerFetchExhausted {
            detail: last_detail,
        },
    }
}


/// Fetch + compare (with mirror) then peer-root cross-check (task 31).
///
/// On `Match` / `VectorMismatch`: record local root, sample peers, gate fail-closed.
/// Returns gated [`ComparisonOutcome`] plus raw [`crate::crosscheck::CrossCheckOutcome`].
#[allow(clippy::too_many_arguments)]
pub async fn fetch_compare_and_crosscheck<C: ChainClient>(
    client: &CoordinationClient,
    epoch: u64,
    chain: &C,
    trust: &LocalTrustRoot,
    store: &SharedMirrorStore,
    peers: &PeerBook,
    expected: Option<ExpectedBundle>,
    run: &crate::crosscheck::CrossCheckRun<'_>,
) -> (ComparisonOutcome, crate::crosscheck::CrossCheckOutcome) {
    let comparison =
        fetch_and_compare_with_mirror(client, epoch, chain, trust, store, peers, expected).await;

    let Some((cand_epoch, cand_root)) = crate::crosscheck::candidate_from_comparison(&comparison)
    else {
        return (
            comparison,
            crate::crosscheck::CrossCheckOutcome::Failed {
                epoch,
                kind: crate::crosscheck::CrossCheckFailKind::InsufficientPeers {
                    reachable: 0,
                    required: run.cfg.min_peer_sample,
                    others_on_metagraph: 0,
                },
                statements: vec![],
            },
        );
    };

    if let (Some(secret), Some(hk)) = (run.signing_secret, run.own_hotkey_pk) {
        if let Err(e) =
            crate::crosscheck::record_local_root(run.root_store, secret, hk, cand_epoch, cand_root)
        {
            tracing::warn!(error = %e, "failed to sign local root statement");
        }
    }

    let discovered = peers.discover_from_chain(chain).unwrap_or_default();
    let peer_refs: Vec<gbase_crosscheck::PeerRef> = discovered
        .into_iter()
        .map(|p| gbase_crosscheck::PeerRef {
            hotkey: p.hotkey,
            base_url: p.base_url,
        })
        .collect();

    let cross = crate::crosscheck::crosscheck_peer_roots(
        client.http_client(),
        chain,
        &peer_refs,
        run.root_store,
        run.cfg,
        cand_epoch,
        cand_root,
    )
    .await;

    let gated = crate::crosscheck::apply_crosscheck_gate(comparison, &cross);
    (gated, cross)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
mod tests {
    use super::*;
    use gbase_aggregate::{
        aggregate, ScoreOrAbsence as AggScore, VerifiedLeaf, ALGORITHM_VERSION as AGG_V,
    };
    use gbase_bundle::{
        compute_merkle_root, compute_metagraph_root, finalize_body_merkle, make_signed_leaf,
        metagraph_rows_from_chain, sign_bundle, sort_leaves, uid_map_from_rows, EpochBundleBodyV1,
        EpochBundleV1, ScoreOrAbsence, ALGORITHM_VERSION, PROTOCOL_VERSION,
    };
    use gbase_chain::{FakeChain, FakeChainConfig};
    use gbase_crypto::secret_from_bytes;
    use gbase_trustroot::{
        measurements_digest, ChallengeEntry, ChallengesBody, MeasurementsBody, ParticipantPolicy,
    };
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sk(tag: u8) -> [u8; 32] {
        let dig = Sha256::digest([0x5A, tag, 0xA5, tag]);
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&dig);
        seed
    }

    fn pk_of(secret: &[u8; 32]) -> [u8; 32] {
        secret_from_bytes(secret)
            .expect("sk")
            .to_public()
            .to_bytes()
    }

    fn hk(tag: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0] = tag;
        h
    }

    fn trust_one(cid: &[u8], challenge_pk: [u8; 32], policy: ParticipantPolicy) -> LocalTrustRoot {
        LocalTrustRoot {
            challenges: ChallengesBody {
                challenges: vec![ChallengeEntry {
                    id: cid.to_vec(),
                    public_key: challenge_pk,
                    emission_share_bps: 10_000,
                    policy,
                }],
            },
            measurements_digest: measurements_digest(&MeasurementsBody::default()),
        }
    }

    fn chain_with(hotkeys: Vec<[u8; 32]>, block_b: u64) -> FakeChain {
        FakeChain::new(FakeChainConfig {
            current_block: block_b.max(10),
            hotkeys: hotkeys.into_iter().map(|h| h.to_vec()).collect(),
            owner_hotkey: vec![0xA1; 32],
            ..FakeChainConfig::default()
        })
    }

    fn to_agg(s: &ScoreOrAbsence) -> AggScore {
        match s {
            ScoreOrAbsence::Score { value } => AggScore::Score { value: *value },
            ScoreOrAbsence::NoScore { reason } => AggScore::NoScore {
                reason: *reason as u8,
            },
        }
    }

    fn valid_bundle(
        csk: &[u8; 32],
        gsk: &[u8; 32],
        cid: &[u8],
        miners: &[(u8, u64)],
        block_b: u64,
    ) -> (EpochBundleV1, LocalTrustRoot, FakeChain) {
        let cpk = pk_of(csk);
        let gpk = pk_of(gsk);
        let trust = trust_one(cid, cpk, ParticipantPolicy::AllMetagraphHotkeys);
        let hotkeys: Vec<[u8; 32]> = miners.iter().map(|(t, _)| hk(*t)).collect();
        let chain = chain_with(hotkeys.clone(), block_b);
        let block_hash = chain.block_hash(block_b).expect("hash");
        let rows = metagraph_rows_from_chain(
            &hotkeys.iter().map(|h| h.to_vec()).collect::<Vec<_>>(),
            None,
        )
        .expect("rows");
        let epoch = 7u64;
        let mut leaves = Vec::new();
        for (tag, score) in miners {
            leaves.push(
                make_signed_leaf(
                    csk,
                    cid,
                    hk(*tag),
                    epoch,
                    ScoreOrAbsence::Score { value: *score },
                )
                .expect("leaf"),
            );
        }
        sort_leaves(&mut leaves);
        let merkle_root = compute_merkle_root(&leaves);
        let uid_map = uid_map_from_rows(&rows);
        let shares = trust.challenges.emission_shares();
        let verified: Vec<VerifiedLeaf> = leaves
            .iter()
            .map(|l| VerifiedLeaf {
                challenge_id: l.challenge_id.clone(),
                miner_hotkey: l.miner_hotkey,
                score_or_absence: to_agg(&l.score_or_absence),
            })
            .collect();
        let final_vector = aggregate(&verified, &shares, &uid_map, AGG_V).expect("agg");
        let body = EpochBundleBodyV1 {
            protocol_version: PROTOCOL_VERSION,
            epoch,
            netuid: 1,
            block_b,
            block_hash,
            metagraph_root: compute_metagraph_root(&rows),
            algorithm_version: ALGORITHM_VERSION,
            emission_shares: shares,
            measurements_digest: trust.measurements_digest,
            uid_map,
            leaves,
            merkle_root,
            final_vector,
            gateway_hotkey: gpk,
        };
        (sign_bundle(gsk, body).expect("sign"), trust, chain)
    }

    fn assert_no_extrinsic(chain: &FakeChain) {
        assert!(
            chain.submissions().is_empty(),
            "task 29 must not submit extrinsics; got {:?}",
            chain.submissions()
        );
    }

    /// S1: honest bundle → Match; dual equality; no extrinsic.
    #[test]
    fn s1_honest_match() {
        let (bundle, trust, chain) = valid_bundle(
            &sk(1),
            &sk(2),
            b"dummy",
            &[(0xA1, 50), (0xB2, 30), (0xC3, 20)],
            100,
        );
        let outcome = compare_bundle(&bundle, &chain, &trust);
        match outcome {
            ComparisonOutcome::Match {
                epoch,
                local_vector,
                gateway_vector,
                vector_hash,
                merkle_root,
            } => {
                assert_eq!(epoch, 7);
                assert_eq!(local_vector, gateway_vector);
                assert_eq!(local_vector, bundle.body.final_vector);
                assert!(final_vectors_equal(&local_vector, &gateway_vector));
                assert_eq!(vector_hash, vector_sha256(&local_vector));
                assert_eq!(vector_hash, vector_sha256(&gateway_vector));
                assert_eq!(merkle_root, bundle.body.merkle_root);
            }
            other => panic!("expected Match, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    /// S2: tampered vector, intact inputs → VectorMismatch (class A).
    #[test]
    fn s2_tampered_vector_class_a() {
        let gsk = sk(2);
        let (mut bundle, trust, chain) =
            valid_bundle(&sk(1), &gsk, b"dummy", &[(0xA1, 50), (0xB2, 30)], 100);
        let honest = bundle.body.final_vector.clone();
        // Flip weights while keeping length; re-sign gateway envelope only.
        assert!(!honest.is_empty());
        bundle.body.final_vector[0].1 = bundle.body.final_vector[0].1.wrapping_add(1);
        if bundle.body.final_vector[0].1 == honest[0].1 {
            bundle.body.final_vector[0].1 = honest[0].1.wrapping_sub(1);
        }
        bundle = sign_bundle(&gsk, bundle.body).expect("resign");

        let outcome = compare_bundle(&bundle, &chain, &trust);
        match outcome {
            ComparisonOutcome::VectorMismatch {
                epoch,
                local_vector,
                gateway_vector,
                local_vector_hash,
                gateway_vector_hash,
                ..
            } => {
                assert_eq!(epoch, 7);
                assert_eq!(local_vector, honest);
                assert_eq!(gateway_vector, bundle.body.final_vector);
                assert_ne!(local_vector, gateway_vector);
                assert!(!final_vectors_equal(&local_vector, &gateway_vector));
                assert_eq!(local_vector_hash, vector_sha256(&local_vector));
                assert_eq!(gateway_vector_hash, vector_sha256(&gateway_vector));
                assert_ne!(local_vector_hash, gateway_vector_hash);
            }
            other => panic!("expected VectorMismatch, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    /// S3: corrupted leaf signature → InputInvalid.
    #[test]
    fn s3_corrupted_leaf_sig_input_invalid() {
        let gsk = sk(2);
        let (mut bundle, trust, chain) = valid_bundle(&sk(1), &gsk, b"dummy", &[(0xA1, 10)], 50);
        bundle.body.leaves[0].challenge_sig[0] ^= 0xff;
        finalize_body_merkle(&mut bundle.body);
        bundle = sign_bundle(&gsk, bundle.body).unwrap();
        let outcome = compare_bundle(&bundle, &chain, &trust);
        match outcome {
            ComparisonOutcome::InputInvalid { error } => {
                assert_eq!(error, BundleError::LeafSignatureInvalid);
            }
            other => panic!("expected InputInvalid, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    /// S4: uid_map disagreeing with metagraph_at(block_hash) → InputInvalid.
    #[test]
    fn s4_uid_map_mismatch_input_invalid() {
        let gsk = sk(2);
        let (mut bundle, trust, chain) = valid_bundle(&sk(1), &gsk, b"dummy", &[(0xA1, 10)], 50);
        bundle.body.uid_map[0].1 = bundle.body.uid_map[0].1.wrapping_add(99);
        bundle = sign_bundle(&gsk, bundle.body).unwrap();
        let outcome = compare_bundle(&bundle, &chain, &trust);
        match outcome {
            ComparisonOutcome::InputInvalid { error } => {
                assert_eq!(error, BundleError::UidMapMismatch);
            }
            other => panic!("expected InputInvalid, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    /// S5: unknown challenge key / id on a leaf (D18) → InputInvalid.
    #[test]
    fn s5_unknown_challenge_key_input_invalid() {
        let csk = sk(1);
        let gsk = sk(2);
        let (mut bundle, trust, chain) = valid_bundle(&csk, &gsk, b"dummy", &[(0xA1, 10)], 50);
        // Leaf for challenge id not present in local trust root (shares still match).
        let leaf = make_signed_leaf(
            &csk,
            b"nope",
            hk(0xA1),
            bundle.body.epoch,
            ScoreOrAbsence::Score { value: 1 },
        )
        .unwrap();
        bundle.body.leaves = vec![leaf];
        finalize_body_merkle(&mut bundle.body);
        bundle = sign_bundle(&gsk, bundle.body).unwrap();

        let outcome = compare_bundle(&bundle, &chain, &trust);
        match outcome {
            ComparisonOutcome::InputInvalid { error } => {
                assert_eq!(error, BundleError::LeafChallengeKeyUnknown);
            }
            other => panic!("expected InputInvalid, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    /// S6: emission shares differ from local trust root → InputInvalid.
    #[test]
    fn s6_shares_differ_input_invalid() {
        let gsk = sk(2);
        let (mut bundle, trust, chain) = valid_bundle(&sk(1), &gsk, b"dummy", &[(0xA1, 10)], 50);
        bundle.body.emission_shares = vec![(b"other".to_vec(), 10_000)];
        bundle = sign_bundle(&gsk, bundle.body).unwrap();
        let outcome = compare_bundle(&bundle, &chain, &trust);
        match outcome {
            ComparisonOutcome::InputInvalid { error } => {
                assert_eq!(error, BundleError::EmissionShareMismatch);
            }
            other => panic!("expected InputInvalid, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    /// S7: missing declared participant → InputInvalid.
    #[test]
    fn s7_missing_participant_input_invalid() {
        let csk = sk(1);
        let gsk = sk(2);
        let cid = b"dummy";
        // Metagraph has two miners; bundle only covers one (no NoScore for the other).
        let (mut bundle, trust, chain) =
            valid_bundle(&csk, &gsk, cid, &[(0xA1, 10), (0xB2, 20)], 80);
        // Drop second leaf → incomplete participant set.
        bundle.body.leaves.pop();
        finalize_body_merkle(&mut bundle.body);
        // Re-aggregate remaining leaf so we fail on D24 not vector.
        let verified: Vec<VerifiedLeaf> = bundle
            .body
            .leaves
            .iter()
            .map(|l| VerifiedLeaf {
                challenge_id: l.challenge_id.clone(),
                miner_hotkey: l.miner_hotkey,
                score_or_absence: to_agg(&l.score_or_absence),
            })
            .collect();
        bundle.body.final_vector = aggregate(
            &verified,
            &bundle.body.emission_shares,
            &bundle.body.uid_map,
            AGG_V,
        )
        .expect("agg");
        bundle = sign_bundle(&gsk, bundle.body).unwrap();

        let outcome = compare_bundle(&bundle, &chain, &trust);
        match outcome {
            ComparisonOutcome::InputInvalid { error } => {
                assert_eq!(error, BundleError::IncompleteParticipantSet);
            }
            other => panic!("expected InputInvalid, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    /// S8: gateway unreachable → NoSubmission; no extrinsic.
    #[tokio::test]
    async fn s8_gateway_unreachable_no_submission() {
        let (bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 10)], 50);
        let _ = bundle; // unused; fetch never succeeds
                        // Nothing listening.
        let client = CoordinationClient::new(Some("http://127.0.0.1:1".into())).expect("client");
        let outcome = fetch_and_compare(&client, 7, &chain, &trust).await;
        match outcome {
            ComparisonOutcome::NoSubmission { reason } => {
                assert!(
                    matches!(reason, NoSubmissionReason::GatewayUnreachable(_)),
                    "got {reason:?}"
                );
            }
            other => panic!("expected NoSubmission, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    /// S8b: no gateway configured → NoSubmission.
    #[tokio::test]
    async fn s8b_no_gateway_configured_no_submission() {
        let (_bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 10)], 50);
        let client = CoordinationClient::new(None).expect("client");
        let outcome = fetch_and_compare(&client, 7, &chain, &trust).await;
        assert_eq!(
            outcome,
            ComparisonOutcome::NoSubmission {
                reason: NoSubmissionReason::NoGatewayConfigured
            }
        );
        assert_no_extrinsic(&chain);
    }

    /// S1 fetch path: wiremock serves honest SCALE bytes → Match.
    #[tokio::test]
    async fn s1_fetch_wiremock_match() {
        let (bundle, trust, chain) =
            valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 50), (0xB2, 30)], 100);
        let bytes = bundle.encode_bytes();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/bundle/7"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/octet-stream")
                    .set_body_bytes(bytes),
            )
            .mount(&server)
            .await;

        let client = CoordinationClient::new(Some(server.uri())).expect("client");
        let outcome = fetch_and_compare(&client, 7, &chain, &trust).await;
        assert!(
            matches!(outcome, ComparisonOutcome::Match { epoch: 7, .. }),
            "got {outcome:?}"
        );
        assert_no_extrinsic(&chain);

        let received = server.received_requests().await.expect("log");
        assert_eq!(received.len(), 1);
        assert!(received[0].url.path().ends_with("/v1/bundle/7"));
    }

    /// S9: no last-known-good — failed fetch is NoSubmission, not a prior Match.
    #[tokio::test]
    async fn s9_no_last_known_good_on_fetch_fail() {
        let (honest, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 50)], 100);
        // Prior epoch would Match if we cheated with LKG — we must not.
        let prior = compare_bundle(&honest, &chain, &trust);
        assert!(matches!(prior, ComparisonOutcome::Match { .. }));

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/bundle/8"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        let client = CoordinationClient::new(Some(server.uri())).expect("client");
        let outcome = fetch_and_compare(&client, 8, &chain, &trust).await;
        match outcome {
            ComparisonOutcome::NoSubmission {
                reason: NoSubmissionReason::GatewayHttp { status: 404, .. },
            } => {}
            other => panic!("expected NoSubmission 404 without LKG, got {other:?}"),
        }
        assert_no_extrinsic(&chain);
    }

    #[test]
    fn dual_equality_requires_both_pairwise_and_digest() {
        let a = vec![(0u16, 1u16), (1, 2)];
        let b = vec![(0u16, 1u16), (1, 2)];
        assert!(final_vectors_equal(&a, &b));
        assert_eq!(vector_sha256(&a), vector_sha256(&b));
        let c = vec![(0u16, 1u16), (1, 3)];
        assert!(!final_vectors_equal(&a, &c));
        assert_ne!(vector_sha256(&a), vector_sha256(&c));
    }
}
