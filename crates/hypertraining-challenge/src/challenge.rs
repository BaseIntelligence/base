//! `HypertrainingChallenge`: expected set, D24 score cover, leaf sign, gateway submit.

use std::collections::{BTreeMap, BTreeSet};

use bundle::{make_signed_leaf, LeafV1, ScoreOrAbsence};
use chain::Metagraph;
use crypto::KEY_LEN;
use thiserror::Error;
use trustroot::{ChallengeEntry, ChallengesBody, ParticipantPolicy};

use crate::config::HypertrainingConfig;
use crate::expected_set::{expected_set_from_pinned_metagraph, Hotkey};
use crate::score::{
    missing_call_noscore, score_from_pipeline, AttestationStatus, PipelineOutcome,
};
use crate::leaf_emit::emit_signed_leaf_set;
use crate::submit::{submit_signed_leaf_set, GatewayClient, SubmitError, SubmitOutcome};
use hypertraining_challenge_task::{CHALLENGE_ID, CHALLENGE_ID_BYTES, SCORING_VERSION};

/// Epoch context for scoring one hypertraining epoch.
#[derive(Debug, Clone)]
pub struct EpochCtx {
    /// Subnet netuid.
    pub netuid: u16,
    /// Epoch index.
    pub epoch: u64,
    /// Hash of `block_B` — E is sealed at this pin.
    pub block_hash: [u8; 32],
    /// Metagraph snapshot from `metagraph_at(block_hash)`.
    pub metagraph: Metagraph,
    /// Local owner-signed challenges body (policy + public key).
    pub challenges: ChallengesBody,
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

/// Concrete hypertraining challenge orchestrator.
#[derive(Debug, Clone)]
pub struct HypertrainingChallenge {
    /// Runtime config (`require_attestation`, …).
    pub config: HypertrainingConfig,
}

impl Default for HypertrainingChallenge {
    fn default() -> Self {
        Self::new(HypertrainingConfig::default())
    }
}

impl HypertrainingChallenge {
    /// Construct with explicit config.
    #[must_use]
    pub const fn new(config: HypertrainingConfig) -> Self {
        Self { config }
    }

    /// Sim / test profile (`require_attestation = false`).
    #[must_use]
    pub const fn sim() -> Self {
        Self::new(HypertrainingConfig::sim())
    }

    /// Challenge id string (`hypertraining`).
    #[must_use]
    pub fn challenge_id(&self) -> &'static str {
        CHALLENGE_ID
    }

    /// Scoring version (`1`).
    #[must_use]
    pub fn scoring_version(&self) -> u16 {
        SCORING_VERSION
    }

    fn entry(body: &ChallengesBody) -> Result<&ChallengeEntry, ChallengeError> {
        body.get(CHALLENGE_ID_BYTES)
            .ok_or_else(|| ChallengeError::UnknownChallenge(CHALLENGE_ID.into()))
    }

    fn policy(body: &ChallengesBody) -> Result<&ParticipantPolicy, ChallengeError> {
        Ok(&Self::entry(body)?.policy)
    }

    /// Expected participant set `E` for this epoch.
    ///
    /// # Errors
    /// When the challenge is absent from the local trust root or metagraph is invalid.
    pub fn expected_set(&self, ctx: &EpochCtx) -> Result<BTreeSet<Hotkey>, ChallengeError> {
        let policy = Self::policy(&ctx.challenges)?;
        expected_set_from_pinned_metagraph(policy, &ctx.metagraph)
            .map_err(|e| ChallengeError::Metagraph(e.to_string()))
    }

    /// Score one miner from a pipeline outcome.
    #[must_use]
    pub fn score_one(
        &self,
        ctx: &EpochCtx,
        miner: Hotkey,
        outcome: &PipelineOutcome,
        attest: &dyn AttestationLookup,
    ) -> ScoreOrAbsence {
        let st = attest.status(ctx.netuid, ctx.epoch, &miner);
        score_from_pipeline(outcome, st, self.config.require_attestation)
    }

    /// Full cover of `expected_set` — **silence is a bug** (D24).
    ///
    /// Missing map entries become `NoScore(ChallengeInternal)`.
    ///
    /// # Errors
    /// Propagates [`expected_set`] failures; every `h ∈ E` still gets an entry on success.
    pub fn score_epoch(
        &self,
        ctx: &EpochCtx,
        outcomes: &BTreeMap<Hotkey, PipelineOutcome>,
        attest: &dyn AttestationLookup,
    ) -> Result<BTreeMap<Hotkey, ScoreOrAbsence>, ChallengeError> {
        let expected = self.expected_set(ctx)?;
        Ok(expected
            .iter()
            .map(|h| {
                let outcome = outcomes
                    .get(h)
                    .cloned()
                    .unwrap_or(PipelineOutcome::ChallengeInternal);
                // Map ChallengeInternal variant through score_one for attestation gate.
                let scored = match &outcome {
                    PipelineOutcome::ChallengeInternal if !outcomes.contains_key(h) => {
                        // Missing call: always ChallengeInternal leaf (even if attest fails first
                        // would also NoScore — prefer explicit missing-call reason).
                        if self.config.require_attestation
                            && !attest.status(ctx.netuid, ctx.epoch, h).allows_score()
                        {
                            // Attestation still gates Score; missing call is not Score.
                            // Prefer AttestationNotVerified when required and not verified.
                            score_from_pipeline(
                                &PipelineOutcome::ChallengeInternal,
                                attest.status(ctx.netuid, ctx.epoch, h),
                                true,
                            )
                        } else {
                            missing_call_noscore()
                        }
                    }
                    _ => self.score_one(ctx, *h, &outcome, attest),
                };
                (*h, scored)
            })
            .collect())
    }

    /// Sign one leaf with the challenge mini-secret (`challenge_id = hypertraining`).
    ///
    /// # Errors
    /// Crypto / id length.
    pub fn sign_leaf(
        &self,
        secret: &[u8; KEY_LEN],
        miner: Hotkey,
        epoch: u64,
        score_or_absence: ScoreOrAbsence,
    ) -> Result<LeafV1, ChallengeError> {
        make_signed_leaf(secret, CHALLENGE_ID_BYTES, miner, epoch, score_or_absence)
            .map_err(|e| ChallengeError::Sign(e.to_string()))
    }

    /// Emit exact-E leaves then POST to gateway.
    ///
    /// # Errors
    /// Emit (D24) or submit failures.
    pub async fn submit_all(
        &self,
        secret: &[u8; KEY_LEN],
        epoch: u64,
        expected: &BTreeSet<Hotkey>,
        scores: &BTreeMap<Hotkey, ScoreOrAbsence>,
        gateway: &GatewayClient,
    ) -> Result<Vec<SubmitOutcome>, ChallengeError> {
        let leaves = emit_signed_leaf_set(secret, epoch, expected, scores)
            .map_err(|e| ChallengeError::Sign(e.to_string()))?;
        Ok(submit_signed_leaf_set(gateway, &leaves).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaf_emit::{emit_signed_leaf_set, public_key_from_secret, verify_leaf_sig, LeafEmitError};
    use bundle::NoScoreReasonCode;
    use trustroot::BPS_DENOM;

    fn mini() -> ([u8; 32], [u8; 32]) {
        let m = schnorrkel::MiniSecretKey::generate_with(rand_core::OsRng);
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
                    emission_share_bps: 0,
                    policy: ParticipantPolicy::AllMetagraphHotkeys,
                }],
            },
        }
    }

    #[test]
    fn challenge_id_is_hypertraining_not_agent_v1() {
        let ch = HypertrainingChallenge::sim();
        assert_eq!(ch.challenge_id(), "hypertraining");
        assert_ne!(ch.challenge_id(), "agent-v1");
        assert_eq!(ch.scoring_version(), 1);
        let _ = BPS_DENOM;
    }

    #[test]
    fn d24_missing_call_emits_noscore_not_silence() {
        let (sk, pk) = mini();
        let m1 = [0x11u8; 32];
        let m2 = [0x22u8; 32];
        let ch = HypertrainingChallenge::sim();
        let ctx = ctx(pk, vec![m1.to_vec(), m2.to_vec()]);
        let mut outcomes = BTreeMap::new();
        outcomes.insert(
            m1,
            PipelineOutcome::Measured {
                t_champ_ms: 10_000,
                t_cand_ms: 8_000,
                guards_passed: true,
            },
        );
        let attest = MapAttestationLookup {
            default: AttestationStatus::Verified,
            ..Default::default()
        };
        let scores = ch.score_epoch(&ctx, &outcomes, &attest).expect("epoch");
        assert_eq!(scores.len(), 2, "full cover of E");
        assert!(matches!(
            scores.get(&m1),
            Some(ScoreOrAbsence::Score { value }) if *value > 0
        ));
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
        assert_eq!(leaf.challenge_id, CHALLENGE_ID_BYTES);
        verify_leaf_sig(&leaf, &pk).expect("verify");
        let derived = public_key_from_secret(&sk).expect("pk");
        assert_eq!(derived, pk);
    }

    #[test]
    fn d24_two_miners_both_get_leaves() {
        let (sk, pk) = mini();
        let m1 = [0xAAu8; 32];
        let m2 = [0xBBu8; 32];
        let ch = HypertrainingChallenge::sim();
        let ctx = ctx(pk, vec![m1.to_vec(), m2.to_vec()]);
        let mut outcomes = BTreeMap::new();
        outcomes.insert(
            m1,
            PipelineOutcome::Measured {
                t_champ_ms: 5_000,
                t_cand_ms: 4_000,
                guards_passed: true,
            },
        );
        outcomes.insert(
            m2,
            PipelineOutcome::Measured {
                t_champ_ms: 5_000,
                t_cand_ms: 4_500,
                guards_passed: true,
            },
        );
        let attest = MapAttestationLookup {
            default: AttestationStatus::Verified,
            ..Default::default()
        };
        let scores = ch.score_epoch(&ctx, &outcomes, &attest).expect("epoch");
        let expected = ch.expected_set(&ctx).expect("E");
        let leaves = emit_signed_leaf_set(&sk, 7, &expected, &scores).expect("emit");
        assert_eq!(leaves.len(), 2);
        for h in [m1, m2] {
            let leaf = leaves.get(&h).expect("leaf");
            assert_eq!(leaf.challenge_id, b"hypertraining");
            verify_leaf_sig(leaf, &pk).expect("sig");
            assert!(matches!(leaf.score_or_absence, ScoreOrAbsence::Score { value } if value > 0));
        }
    }

    #[test]
    fn emit_subset_refused() {
        let (sk, _pk) = mini();
        let m1 = [0x11u8; 32];
        let m2 = [0x22u8; 32];
        let expected = BTreeSet::from([m1, m2]);
        let mut scores = BTreeMap::new();
        scores.insert(m1, ScoreOrAbsence::Score { value: 1 });
        let err = emit_signed_leaf_set(&sk, 1, &expected, &scores).expect_err("subset");
        assert!(matches!(err, LeafEmitError::MissingHotkeys(_)));
    }
}
