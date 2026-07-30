//! Challenge trait and `agent-v1` implementation (`AGENT_CHALLENGE` §10).

use std::collections::{BTreeMap, BTreeSet};

use bundle::{make_signed_leaf, LeafV1, NoScoreReasonCode, ScoreOrAbsence};
use chain::Metagraph;
use crypto::KEY_LEN;
use thiserror::Error;
use trustroot::{ChallengeEntry, ChallengesBody, ParticipantPolicy};

use crate::leaf_map::{cover_expected_verify_leaves, score_from_verify_result};
use crate::score::{score_from_outcome, AttestationStatus, CallOutcome, ScoreInputs};
use crate::verify::{Reward, VerifyError};
use crate::submit::{GatewayClient, SubmitError, SubmitOutcome};
use crate::task_gen::{
    answer_digest_v2, task_id_v2, CHALLENGE_ID, CHALLENGE_ID_BYTES, FIXTURE_MODEL_PATCH,
    FIXTURE_PACK_ID, SCORING_VERSION,
};

/// Miner hotkey type alias.
pub type Hotkey = [u8; KEY_LEN];

/// Epoch context for scoring one challenge epoch.
#[derive(Debug, Clone)]
pub struct EpochCtx {
    /// Subnet netuid.
    pub netuid: u16,
    /// Epoch index.
    pub epoch: u64,
    /// Hash of `block_B` — E is sealed at this pin (I7). Never a moving tip.
    pub block_hash: [u8; 32],
    /// Metagraph snapshot from `metagraph_at(block_hash)`.
    pub metagraph: Metagraph,
    /// Local owner-signed challenges body (for policy + public key).
    pub challenges: ChallengesBody,
    /// Pack id selected for this miner/epoch (bound into v2 task identity).
    pub pack_id: Vec<u8>,
}

/// Look up attestation status for a miner this epoch.
pub trait AttestationLookup: Send + Sync {
    /// Status for `(netuid, epoch, miner)`.
    fn status(&self, netuid: u16, epoch: u64, miner: &Hotkey) -> AttestationStatus;
}

/// Fixed map of attestation outcomes (tests / injected control plane).
#[derive(Debug, Clone, Default)]
pub struct MapAttestationLookup {
    /// Keyed by miner hotkey.
    pub by_miner: BTreeMap<Hotkey, AttestationStatus>,
    /// Default when miner missing from map.
    pub default: AttestationStatus,
}

impl AttestationLookup for MapAttestationLookup {
    fn status(&self, _netuid: u16, _epoch: u64, miner: &Hotkey) -> AttestationStatus {
        self.by_miner.get(miner).copied().unwrap_or(self.default)
    }
}

/// Outcome of contacting one miner (or a synthetic fixture outcome).
#[derive(Debug, Clone)]
pub enum MinerCallOutcome {
    /// Pure scoring inputs already resolved (offline / fixture path).
    Resolved(ScoreInputs),
    /// Use precomputed `CallOutcome` + duration with live attestation lookup.
    Observed {
        /// Challenge-side wall ms.
        duration_ms: u64,
        /// Terminal call outcome.
        outcome: CallOutcome,
        /// Oracle / fixture expected `model.patch` for v2 answer check.
        expected_model_patch: Vec<u8>,
    },
}

/// Challenge trait shaped for Prism accommodation later (`AGENT_CHALLENGE` §10).
pub trait Challenge {
    /// Challenge id string.
    fn challenge_id(&self) -> &str;
    /// Scoring version.
    fn scoring_version(&self) -> u16;
    /// Expected participant set `E` for this epoch.
    ///
    /// # Errors
    ///
    /// When the challenge is absent from the local trust root or metagraph is invalid.
    fn expected_set(&self, ctx: &EpochCtx) -> Result<BTreeSet<Hotkey>, ChallengeError>;
    /// Score one miner (must not be called for `h ∉ E` in production paths).
    fn score_one(
        &self,
        ctx: &EpochCtx,
        miner: Hotkey,
        call: &MinerCallOutcome,
        attest: &dyn AttestationLookup,
    ) -> ScoreOrAbsence;
    /// Full cover of `expected_set` — **silence is a bug** (D24).
    ///
    /// # Errors
    ///
    /// Propagates [`expected_set`] failures; every `h ∈ E` still gets an entry.
    fn score_epoch(
        &self,
        ctx: &EpochCtx,
        calls: &BTreeMap<Hotkey, MinerCallOutcome>,
        attest: &dyn AttestationLookup,
    ) -> Result<BTreeMap<Hotkey, ScoreOrAbsence>, ChallengeError>;
    /// Sign one leaf with the challenge mini-secret.
    ///
    /// # Errors
    ///
    /// Crypto / id length.
    fn sign_leaf(
        &self,
        secret: &[u8; KEY_LEN],
        miner: Hotkey,
        epoch: u64,
        score_or_absence: ScoreOrAbsence,
    ) -> Result<LeafV1, ChallengeError>;
    /// Sign all scores and POST to gateway.
    ///
    /// # Errors
    ///
    /// Sign or submit failures.
    fn submit_all(
        &self,
        secret: &[u8; KEY_LEN],
        epoch: u64,
        scores: &BTreeMap<Hotkey, ScoreOrAbsence>,
        gateway: &GatewayClient,
    ) -> impl std::future::Future<Output = Result<Vec<SubmitOutcome>, ChallengeError>> + Send;
}

/// Errors from challenge orchestration.
#[derive(Debug, Error)]
pub enum ChallengeError {
    /// Challenge id missing from local trust root.
    #[error("challenge {0} not in local trust root")]
    UnknownChallenge(String),
    /// Metagraph shape invalid.
    #[error("metagraph: {0}")]
    Metagraph(String),
    /// Leaf signing failed.
    #[error("sign leaf: {0}")]
    Sign(String),
    /// Gateway submit failed.
    #[error("submit: {0}")]
    Submit(#[from] SubmitError),
}

/// Concrete `agent-v1` challenge.
#[derive(Debug, Clone, Default)]
pub struct AgentV1Challenge;

impl AgentV1Challenge {
    /// Construct.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Trust-root entry for `agent-v1`.
    fn entry(body: &ChallengesBody) -> Result<&ChallengeEntry, ChallengeError> {
        body.get(CHALLENGE_ID_BYTES)
            .ok_or_else(|| ChallengeError::UnknownChallenge(CHALLENGE_ID.into()))
    }

    /// Policy for participant derivation.
    fn policy(body: &ChallengesBody) -> Result<&ParticipantPolicy, ChallengeError> {
        Ok(&Self::entry(body)?.policy)
    }
}

impl Challenge for AgentV1Challenge {
    fn challenge_id(&self) -> &str {
        CHALLENGE_ID
    }

    fn scoring_version(&self) -> u16 {
        SCORING_VERSION
    }

    fn expected_set(&self, ctx: &EpochCtx) -> Result<BTreeSet<Hotkey>, ChallengeError> {
        let policy = Self::policy(&ctx.challenges)?;
        let sealed = crate::expected_set::expected_set_from_pinned_metagraph(
            policy,
            crate::expected_set::PinnedBlockHash::new(ctx.block_hash),
            &ctx.metagraph,
        )
        .map_err(|e| ChallengeError::Metagraph(e.to_string()))?;
        Ok(sealed.hotkeys())
    }

    fn score_one(
        &self,
        ctx: &EpochCtx,
        miner: Hotkey,
        call: &MinerCallOutcome,
        attest: &dyn AttestationLookup,
    ) -> ScoreOrAbsence {
        match call {
            MinerCallOutcome::Resolved(inputs) => score_from_outcome(inputs),
            MinerCallOutcome::Observed {
                duration_ms,
                outcome,
                expected_model_patch,
            } => {
                let attestation = attest.status(ctx.netuid, ctx.epoch, &miner);
                let inputs = ScoreInputs {
                    netuid: ctx.netuid,
                    epoch: ctx.epoch,
                    miner_hotkey: miner,
                    pack_id: ctx.pack_id.clone(),
                    expected_model_patch: expected_model_patch.clone(),
                    attestation,
                    duration_ms: *duration_ms,
                    outcome: outcome.clone(),
                };
                score_from_outcome(&inputs)
            }
        }
    }

    fn score_epoch(
        &self,
        ctx: &EpochCtx,
        calls: &BTreeMap<Hotkey, MinerCallOutcome>,
        attest: &dyn AttestationLookup,
    ) -> Result<BTreeMap<Hotkey, ScoreOrAbsence>, ChallengeError> {
        let expected = self.expected_set(ctx)?;
        let mut out = BTreeMap::new();
        for h in &expected {
            let call = calls.get(h).cloned().unwrap_or(
                // Silence is a bug — emit ChallengeInternal NoScore, never omit.
                MinerCallOutcome::Observed {
                    duration_ms: 0,
                    outcome: CallOutcome::ChallengeInternal,
                    expected_model_patch: Vec::new(),
                },
            );
            let soa = self.score_one(ctx, *h, &call, attest);
            out.insert(*h, soa);
        }
        // Do not emit leaves for h ∉ E.
        Ok(out)
    }

    fn sign_leaf(
        &self,
        secret: &[u8; KEY_LEN],
        miner: Hotkey,
        epoch: u64,
        score_or_absence: ScoreOrAbsence,
    ) -> Result<LeafV1, ChallengeError> {
        make_signed_leaf(secret, CHALLENGE_ID_BYTES, miner, epoch, score_or_absence)
            .map_err(|e| ChallengeError::Sign(e.to_string()))
    }

    async fn submit_all(
        &self,
        secret: &[u8; KEY_LEN],
        epoch: u64,
        scores: &BTreeMap<Hotkey, ScoreOrAbsence>,
        gateway: &GatewayClient,
    ) -> Result<Vec<SubmitOutcome>, ChallengeError> {
        let mut leaves = Vec::with_capacity(scores.len());
        for (miner, soa) in scores {
            leaves.push(self.sign_leaf(secret, *miner, epoch, soa.clone())?);
        }
        Ok(gateway.submit_all(&leaves).await?)
    }
}

/// Build a correct `Http200` outcome for fixtures (v2 pack-bound identity + model.patch digest).
#[must_use]
pub fn correct_http200(
    netuid: u16,
    epoch: u64,
    miner: &Hotkey,
    pack_id: &[u8],
    model_patch: &[u8],
) -> CallOutcome {
    let tid = task_id_v2(netuid, epoch, miner, pack_id, SCORING_VERSION);
    let ans = answer_digest_v2(model_patch);
    CallOutcome::Http200 {
        challenge_id: CHALLENGE_ID.to_owned(),
        epoch,
        task_id: tid,
        answer_digest: ans,
        agent_version: "1".into(),
    }
}

/// Fixture helper: correct HTTP 200 with default pack/patch placeholders.
#[must_use]
pub fn correct_http200_fixture(netuid: u16, epoch: u64, miner: &Hotkey) -> CallOutcome {
    correct_http200(netuid, epoch, miner, FIXTURE_PACK_ID, FIXTURE_MODEL_PATCH)
}

/// Helper: `NoScore` reason for missing call coverage (D24).
#[must_use]
pub fn silence_is_bug_leaf() -> ScoreOrAbsence {
    ScoreOrAbsence::NoScore {
        reason: NoScoreReasonCode::ChallengeInternal,
    }
}


/// Cover `E` from operator-side verify results (Harbor grade).
///
/// Every `h ∈ E` gets exactly one leaf. Missing results → `ChallengeInternal`.
/// See [`crate::leaf_map::map_verify_error`] for fault attribution.
///
/// # Errors
///
/// Propagates [`Challenge::expected_set`] failures.
pub fn score_epoch_from_verify(
    challenge: &impl Challenge,
    ctx: &EpochCtx,
    results: &BTreeMap<Hotkey, Result<Reward, VerifyError>>,
) -> Result<BTreeMap<Hotkey, ScoreOrAbsence>, ChallengeError> {
    let expected = challenge.expected_set(ctx)?;
    Ok(cover_expected_verify_leaves(&expected, results))
}

/// Map one verify grade into a leaf (no retries). Prefer
/// [`crate::leaf_map::grade_to_score_or_absence`] when calling the verifier.
#[must_use]
pub fn leaf_from_verify_result(result: &Result<Reward, VerifyError>) -> ScoreOrAbsence {
    score_from_verify_result(result)
}


#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::keys::public_key_from_secret;
    use rand_core::OsRng;
    use trustroot::{ChallengeEntry, ParticipantPolicy, BPS_DENOM};

    fn mini() -> ([u8; 32], [u8; 32]) {
        let m = schnorrkel::MiniSecretKey::generate_with(OsRng);
        let sk = m.to_bytes();
        let pk = m
            .expand(schnorrkel::ExpansionMode::Ed25519)
            .to_public()
            .to_bytes();
        (sk, pk)
    }

    fn ctx(pk: [u8; 32], hotkeys: Vec<Vec<u8>>) -> EpochCtx {
        EpochCtx {
            netuid: 1,
            epoch: 7,
            block_hash: {
                let mut h = [0u8; 32];
                h[0] = 0xB0;
                h[1] = 0xB1;
                h
            },
            metagraph: Metagraph {
                netuid: 1,
                hotkeys,
                owner_hotkey: vec![0u8; 32],
            },
            challenges: ChallengesBody {
                challenges: vec![ChallengeEntry {
                    id: CHALLENGE_ID_BYTES.to_vec(),
                    public_key: pk,
                    emission_share_bps: BPS_DENOM,
                    policy: ParticipantPolicy::AllMetagraphHotkeys,
                }],
            },
            pack_id: FIXTURE_PACK_ID.to_vec(),
        }
    }

    #[test]
    fn d24_missing_call_emits_noscore_not_silence() {
        let (sk, pk) = mini();
        let m1 = [0x11u8; 32];
        let m2 = [0x22u8; 32];
        let ch = AgentV1Challenge::new();
        let ctx = ctx(pk, vec![m1.to_vec(), m2.to_vec()]);
        let mut calls = BTreeMap::new();
        // Only score m1; m2 is in E but has no call.
        calls.insert(
            m1,
            MinerCallOutcome::Observed {
                duration_ms: 2000,
                outcome: correct_http200_fixture(1, 7, &m1),
                expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
            },
        );
        let attest = MapAttestationLookup {
            default: AttestationStatus::Verified,
            ..Default::default()
        };
        let scores = ch.score_epoch(&ctx, &calls, &attest).expect("epoch");
        assert_eq!(scores.len(), 2, "full cover of E");
        assert_eq!(
            scores.get(&m1),
            Some(&ScoreOrAbsence::Score {
                value: crate::SCORE_MAX
            })
        );
        assert_eq!(
            scores.get(&m2),
            Some(&ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::ChallengeInternal
            }),
            "declared miner with no score → signed NoScore, never silence"
        );
        let leaf = ch
            .sign_leaf(&sk, m2, 7, scores[&m2].clone())
            .expect("sign noscore");
        assert!(matches!(
            leaf.score_or_absence,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::ChallengeInternal
            }
        ));
        let derived = public_key_from_secret(&sk).expect("pk");
        assert_eq!(derived, pk);
    }

    #[test]
    fn extra_miner_not_in_e_not_emitted() {
        let (_sk, pk) = mini();
        let m1 = [0x11u8; 32];
        let outsider = [0x99u8; 32];
        let ch = AgentV1Challenge::new();
        let ctx = ctx(pk, vec![m1.to_vec()]);
        let mut calls = BTreeMap::new();
        calls.insert(
            outsider,
            MinerCallOutcome::Observed {
                duration_ms: 0,
                outcome: correct_http200_fixture(1, 7, &outsider),
                expected_model_patch: FIXTURE_MODEL_PATCH.to_vec(),
            },
        );
        let attest = MapAttestationLookup {
            default: AttestationStatus::Verified,
            ..Default::default()
        };
        let scores = ch.score_epoch(&ctx, &calls, &attest).expect("epoch");
        assert_eq!(scores.len(), 1);
        assert!(scores.contains_key(&m1));
        assert!(!scores.contains_key(&outsider));
    }

    #[test]
    fn scoring_version_is_two() {
        assert_eq!(AgentV1Challenge::new().scoring_version(), 2);
        assert_eq!(SCORING_VERSION, 2);
    }

    #[test]
    fn verify_outage_covers_all_e_with_challenge_internal_leaves() {
        let (sk, pk) = mini();
        let m1 = [0x11u8; 32];
        let m2 = [0x22u8; 32];
        let m3 = [0x33u8; 32];
        let ch = AgentV1Challenge::new();
        let ctx = ctx(pk, vec![m1.to_vec(), m2.to_vec(), m3.to_vec()]);
        let results = BTreeMap::new();
        let scores = score_epoch_from_verify(&ch, &ctx, &results).expect("epoch");
        assert_eq!(scores.len(), 3, "leaf count == |E|");
        for h in [m1, m2, m3] {
            assert_eq!(
                scores.get(&h),
                Some(&ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::ChallengeInternal
                }),
                "hotkey must not be silent"
            );
            let leaf = ch
                .sign_leaf(&sk, h, 7, scores[&h].clone())
                .expect("sign");
            assert!(matches!(
                leaf.score_or_absence,
                ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::ChallengeInternal
                }
            ));
        }
    }

    #[test]
    fn verify_malformed_not_park_and_apply_failed_miner_zero() {
        let (_sk, pk) = mini();
        let m1 = [0x11u8; 32];
        let m2 = [0x22u8; 32];
        let ch = AgentV1Challenge::new();
        let ctx = ctx(pk, vec![m1.to_vec(), m2.to_vec()]);
        let mut results = BTreeMap::new();
        results.insert(
            m1,
            Err(VerifyError::MalformedOutput {
                message: "junit broken".into(),
            }),
        );
        results.insert(
            m2,
            Err(VerifyError::ApplyFailed {
                message: "patch reject".into(),
            }),
        );
        let scores = score_epoch_from_verify(&ch, &ctx, &results).expect("epoch");
        assert_eq!(
            scores[&m1],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::ChallengeInternal
            },
            "malformed junit is operator ChallengeInternal, never Park"
        );
        assert_eq!(
            scores[&m2],
            ScoreOrAbsence::Score { value: 0 },
            "ApplyFailed is miner-attributable Score(0)"
        );
        assert!(!matches!(
            scores[&m1],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::AttestationNotVerified
            }
        ));
    }
}
