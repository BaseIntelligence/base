//! PRISM epoch-close leaf emission (D24-complete sets, exactly-once outbox).
//!
//! Semantics (normative: `docs/PRISM.md` § Leaf emission):
//!
//! - **One D24-complete leaf set per chain epoch**, emitted by the first
//!   emitter tick that observes the epoch. The competition input is the
//!   union of (a) every submission finalized since the previously emitted
//!   epoch — the outbox batch (`kind IS NOT NULL AND emitted_epoch IS NULL`)
//!   — and (b) every still-active positive lattice score
//!   (`kind = 'score' AND score > 0`), competition-aggregated
//!   (`prism-registry`), plus `NoScore(NotAttempted)` for every other
//!   expected participant so the epoch is sealable on its own.
//! - A row's acceptance epoch (`prism_submission.epoch`) is **intake
//!   metadata only**: a submission that finalizes after an epoch boundary
//!   (prod trains up to 6h ≫ 72-min epochs) first enters the outbox at the
//!   next boundary, never in its acceptance epoch.
//! - **Exactly-once outbox assignment per scoring run**: the fresh batch is
//!   assigned (`emitted_epoch = E`, sticky) before the submit, and the
//!   per-netuid emit cursor advances only after the full set landed. A crash
//!   between submit and cursor advance replays the sticky assigned set on
//!   the next tick; gateway leaves are first-write-wins with identical
//!   values under replay, so the retry converges. Retried/re-scored rows
//!   re-enter the outbox (`reset_for_retry` clears the watermark).
//! - **Positive scores carry forward**: after outbox assignment, a
//!   `Score(v>0)` row keeps participating in every later epoch's
//!   competition set until a better/valid score supersedes it via `max`.
//!   Leaf emission then applies **WTA** ([`prism_registry::apply_wta`]) so
//!   only the single best hotkey receives a positive Score leaf. Architecture-
//!   owner credit is temporarily off ([`prism_registry::OWNER_ARCH_CREDIT_ENABLED`])
//!   so WTA follows own BPB-derived scores. Empty or reject-only fresh batches
//!   therefore do not burn the prism share.
//! - Epochs during a master outage carry no *new* outbox rows; the first
//!   epoch after recovery still includes active positive scores plus any
//!   backlog (the seal always pins fresh epochs — stale ones can never
//!   Match on-chain anyway).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bundle::{LeafV1, NoScoreReasonCode, ScoreOrAbsence};
use challenge_common::{ExpectedSet, GatewayClient, Hotkey};
use crypto::KEY_LEN;
use prism_challenge_task::CHALLENGE_ID_BYTES;
use prism_store::{EpochScoreRow, FinalScore, PrismStore, StoreError};
use thiserror::Error;
use tracing::info;

/// Emission failures.
#[derive(Debug, Error)]
pub enum EmitError {
    /// Store fault.
    #[error("store: {0}")]
    Store(#[from] StoreError),
    /// Leaf signing / D24 shape.
    #[error("emit: {0}")]
    Sign(String),
    /// Gateway submit.
    #[error("submit: {0}")]
    Submit(String),
}

/// What one emitted leaf set carried.
#[derive(Debug, Clone)]
pub struct EmitSummary {
    /// Leaf epoch.
    pub epoch: u64,
    /// Full D24 set size (== `|expected|`).
    pub leaves: usize,
    /// Fresh outbox rows assigned to this epoch (0 = carry-only or empty).
    pub batch: usize,
    /// The submitted signed leaves (test introspection).
    pub signed: BTreeMap<Hotkey, LeafV1>,
}

/// Epoch-close emitter: assigns outbox batches, signs + submits D24 sets,
/// tracks the completion cursor. Construct once per service; `tick` is the
/// only driver and is safe to call concurrently with finalizes.
#[derive(Debug)]
pub struct EpochEmitter {
    store: Arc<dyn PrismStore>,
    gateway: GatewayClient,
    sk: [u8; KEY_LEN],
    netuid: u16,
}

impl EpochEmitter {
    /// Construct. `gateway` may be dry-run (local sim).
    #[must_use]
    pub fn new(
        store: Arc<dyn PrismStore>,
        sk: [u8; KEY_LEN],
        netuid: u16,
        gateway: GatewayClient,
    ) -> Self {
        Self {
            store,
            gateway,
            sk,
            netuid,
        }
    }

    /// Drive emission up to `current_epoch`: replay any assigned-but-
    /// incomplete epoch (crash recovery), then emit the set for
    /// `current_epoch` unless it already landed. Returns the summary when a
    /// new set was emitted.
    ///
    /// # Errors
    /// Store / sign / submit failures (the caller retries next tick; the
    /// outbox makes every retry safe).
    pub async fn tick(
        &self,
        current_epoch: u64,
        expected: &ExpectedSet,
    ) -> Result<Option<EmitSummary>, EmitError> {
        for pending in self.store.pending_emit_epochs(self.netuid).await? {
            info!(
                epoch = pending,
                "replaying assigned leaf set (emit recovery)"
            );
            self.emit_assigned(pending, expected).await?;
        }
        let done = self.store.emit_cursor(self.netuid).await?;
        match done {
            Some(d) if d >= current_epoch => Ok(None),
            // First boot or catch-up after downtime: emit once at the live
            // epoch with the whole backlog; skipped gap epochs get no set
            // (seals always pin fresh epochs).
            _ => self.emit_new(current_epoch, expected).await.map(Some),
        }
    }

    /// Assign the pending batch to `epoch` and emit its set.
    ///
    /// # Errors
    /// See [`EmitError`]; on submit failure the assignment stays sticky and
    /// the next [`Self::tick`] replays it via recovery.
    pub async fn emit_new(
        &self,
        epoch: u64,
        expected: &ExpectedSet,
    ) -> Result<EmitSummary, EmitError> {
        let batch = self.store.assign_emit_batch(self.netuid, epoch).await?;
        self.emit_rows(epoch, batch, expected).await
    }

    /// Re-emit the sticky assigned batch of `epoch` (recovery path).
    ///
    /// # Errors
    /// See [`EmitError`].
    pub async fn emit_assigned(
        &self,
        epoch: u64,
        expected: &ExpectedSet,
    ) -> Result<EmitSummary, EmitError> {
        let batch = self.store.emit_batch(self.netuid, epoch).await?;
        self.emit_rows(epoch, batch, expected).await
    }

    async fn emit_rows(
        &self,
        epoch: u64,
        batch: Vec<EpochScoreRow>,
        expected: &ExpectedSet,
    ) -> Result<EmitSummary, EmitError> {
        let n = batch.len();
        let active = self.store.active_score_rows(self.netuid).await?;
        let competition = merge_competition_rows(&batch, &active);
        let owners: BTreeMap<String, String> =
            self.store.arch_owners().await?.into_iter().collect();
        let signed = build_epoch_leaves(&self.sk, epoch, expected, &competition, &owners)?;
        challenge_common::submit_signed_leaf_set(&self.gateway, &signed)
            .await
            .map_err(|e| EmitError::Submit(e.to_string()))?;
        // Cursor advances only after the full set landed.
        self.store.set_emit_cursor(self.netuid, epoch).await?;
        Ok(EmitSummary {
            epoch,
            leaves: signed.len(),
            batch: n,
            signed,
        })
    }
}

/// Union of the fresh outbox batch and active positive scores for competition.
///
/// Duplicates (a just-assigned `Score(v>0)` row also appears in `active`) are
/// harmless: [`prism_registry::competition_scores`] takes per-hotkey/`arch`
/// maxima.
fn merge_competition_rows(batch: &[EpochScoreRow], active: &[EpochScoreRow]) -> Vec<EpochScoreRow> {
    let mut out = Vec::with_capacity(batch.len().saturating_add(active.len()));
    out.extend_from_slice(batch);
    out.extend_from_slice(active);
    out
}

/// Build the signed D24 set for `epoch`: the batch competition-aggregated
/// (owner + challenger credits, max lattice) plus `NoScore(NotAttempted)`
/// for every expected participant without a score this epoch. Batch rows
/// whose hotkey left the metagraph are dropped from the set (D24 forbids
/// extra leaves) but stay assigned.
pub fn build_epoch_leaves(
    secret: &[u8; KEY_LEN],
    epoch: u64,
    expected: &ExpectedSet,
    batch: &[EpochScoreRow],
    arch_owners: &BTreeMap<String, String>,
) -> Result<BTreeMap<Hotkey, LeafV1>, EmitError> {
    let by_miner =
        prism_registry::apply_wta(prism_registry::competition_scores(batch, arch_owners));
    let mut scores: BTreeMap<Hotkey, ScoreOrAbsence> = BTreeMap::new();
    let mut expected_set: BTreeSet<Hotkey> = BTreeSet::new();
    for p in &expected.participants {
        expected_set.insert(p.hotkey);
        let soa = match by_miner.get(&hex::encode(p.hotkey)) {
            Some(FinalScore::Score(v)) => ScoreOrAbsence::Score { value: *v },
            Some(FinalScore::NoScore(r)) => ScoreOrAbsence::NoScore {
                reason: reason_from_u8(*r),
            },
            None => ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::NotAttempted,
            },
        };
        scores.insert(p.hotkey, soa);
    }
    challenge_common::emit_signed_leaf_set(
        secret,
        CHALLENGE_ID_BYTES,
        epoch,
        &expected_set,
        &scores,
    )
    .map_err(|e| EmitError::Sign(e.to_string()))
}

fn reason_from_u8(v: u8) -> NoScoreReasonCode {
    match v {
        0 => NoScoreReasonCode::NotAttempted,
        1 => NoScoreReasonCode::Timeout,
        2 => NoScoreReasonCode::InvalidResponse,
        3 => NoScoreReasonCode::AttestationNotVerified,
        4 => NoScoreReasonCode::MinerError,
        5 => NoScoreReasonCode::RateLimited,
        7 => NoScoreReasonCode::PolicySkip,
        _ => NoScoreReasonCode::ChallengeInternal,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn sk() -> [u8; KEY_LEN] {
        let mut s = [7u8; KEY_LEN];
        s[0] = 0x42;
        s
    }

    fn expected(hks: &[[u8; KEY_LEN]]) -> ExpectedSet {
        ExpectedSet {
            block_hash: [0x77u8; 32],
            participants: hks
                .iter()
                .enumerate()
                .map(|(i, h)| challenge_common::ExpectedParticipant {
                    hotkey: *h,
                    uid: u16::try_from(i).unwrap_or(0),
                })
                .collect(),
        }
    }

    fn row(hk: [u8; KEY_LEN], arch: Option<&str>, score: u64) -> EpochScoreRow {
        EpochScoreRow {
            miner_hotkey: hex::encode(hk),
            arch_id: arch.map(str::to_owned),
            final_score: FinalScore::Score(score),
        }
    }

    fn soa_of(leaves: &BTreeMap<Hotkey, LeafV1>, hk: &[u8; KEY_LEN]) -> ScoreOrAbsence {
        leaves.get(hk).map(|l| l.score_or_absence.clone()).unwrap()
    }

    #[test]
    fn batch_scores_land_and_rest_is_not_attempted() {
        let (a, b, c) = ([0xAA; 32], [0xBB; 32], [0xCC; 32]);
        let leaves = build_epoch_leaves(
            &sk(),
            9,
            &expected(&[a, b, c]),
            &[row(a, None, 100), row(b, None, 200)],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(leaves.len(), 3);
        // WTA: only the argmax (b=200) keeps a positive Score leaf.
        assert!(matches!(
            soa_of(&leaves, &a),
            ScoreOrAbsence::Score { value: 0 }
        ));
        assert!(matches!(
            soa_of(&leaves, &b),
            ScoreOrAbsence::Score { value: 200 }
        ));
        assert!(matches!(
            soa_of(&leaves, &c),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::NotAttempted
            }
        ));
        assert!(leaves.values().all(|l| l.epoch == 9));
    }

    #[test]
    fn absence_reason_is_preserved_for_scoreless_rows() {
        let a = [0xAA; 32];
        let mut r = row(a, None, 0);
        r.final_score = FinalScore::NoScore(6);
        let leaves = build_epoch_leaves(&sk(), 9, &expected(&[a]), &[r], &BTreeMap::new()).unwrap();
        assert!(matches!(
            soa_of(&leaves, &a),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::ChallengeInternal
            }
        ));
    }

    #[test]
    fn batch_hotkey_outside_metagraph_is_dropped_from_set() {
        let (a, outsider) = ([0xAA; 32], [0xDD; 32]);
        let leaves = build_epoch_leaves(
            &sk(),
            9,
            &expected(&[a]),
            &[row(outsider, None, 500)],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(leaves.len(), 1);
        assert!(matches!(
            soa_of(&leaves, &a),
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::NotAttempted
            }
        ));
    }

    #[test]
    fn empty_batch_emits_full_no_score_set() {
        let (a, b) = ([0xAA; 32], [0xBB; 32]);
        let leaves =
            build_epoch_leaves(&sk(), 9, &expected(&[a, b]), &[], &BTreeMap::new()).unwrap();
        assert_eq!(leaves.len(), 2);
        assert!(leaves
            .values()
            .all(|l| matches!(l.score_or_absence, ScoreOrAbsence::NoScore { .. })));
    }
}
