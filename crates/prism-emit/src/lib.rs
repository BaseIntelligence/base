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
//!   the next tick; gateway leaves tip-supersede on digest change and treat
//!   identical digests as 409-as-ok, so the retry converges. Retried/re-scored
//!   rows re-enter the outbox (`reset_for_retry` clears the watermark).
//! - **Positive scores carry forward**: after outbox assignment, a
//!   `Score(v>0)` row keeps participating in every later epoch's
//!   competition set until a better/valid score supersedes it via `max`.
//!   Leaf emission then applies **WTA** ([`prism_registry::apply_wta`]) so
//!   only the single best **submitter** receives a positive Score leaf.
//!   Architecture-owner credit stays off
//!   ([`prism_registry::OWNER_ARCH_CREDIT_ENABLED`] = `false`) — WTA follows
//!   the miner hotkey that posted the best-BPB run. Empty or reject-only
//!   fresh batches therefore do not burn the prism share (active carry).
//! - **Tip refresh**: once the cursor reaches the live epoch, later ticks
//!   re-submit the current WTA projection so a mid-epoch champion change
//!   tip-supersedes gateway leaves (then `prod-real-seal` reseals). The
//!   cursor does not advance again on tip refresh.
//! - Epochs during a master outage carry no *new* outbox rows; the first
//!   epoch after recovery still includes active positive scores plus any
//!   backlog (the seal always pins fresh epochs — stale ones can never
//!   Match on-chain anyway).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

pub mod leaf_emit;
pub mod submit;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bundle::{LeafV1, NoScoreReasonCode, ScoreOrAbsence};

/// Aggregator burn sink (`BUNDLE_SPEC` §6.5 `ZERO_MINER_BURN_UID`): a leaf
/// mapping to this uid is dropped and its mass burns.
const BURN_UID: u16 = 0;
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
    /// `current_epoch` (first land) or tip-refresh when the cursor already
    /// covers the tip. Returns the summary when leaves were submitted.
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
            Some(d) if d >= current_epoch => {
                // Tip tracking: re-submit WTA so mid-epoch champion changes
                // supersede gateway leaves. Cursor stays put.
                self.refresh_tip(current_epoch, expected).await.map(Some)
            }
            // First boot or catch-up after downtime: emit once at the live
            // epoch with the whole backlog; skipped gap epochs get no set
            // (seals always pin fresh epochs).
            _ => self.emit_new(current_epoch, expected).await.map(Some),
        }
    }

    /// Re-submit the tip epoch's current WTA projection without re-assigning
    /// the outbox or advancing the emit cursor.
    ///
    /// # Errors
    /// See [`EmitError`].
    pub async fn refresh_tip(
        &self,
        epoch: u64,
        expected: &ExpectedSet,
    ) -> Result<EmitSummary, EmitError> {
        let batch = self.store.emit_batch(self.netuid, epoch).await?;
        self.submit_rows(epoch, batch, expected, false).await
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
        self.submit_rows(epoch, batch, expected, true).await
    }

    async fn submit_rows(
        &self,
        epoch: u64,
        batch: Vec<EpochScoreRow>,
        expected: &ExpectedSet,
        advance_cursor: bool,
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
        if advance_cursor {
            // Cursor advances only after the first full set for the epoch landed.
            self.store.set_emit_cursor(self.netuid, epoch).await?;
        }
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

/// Build the signed D24 set for `epoch`: competition-aggregated submitter
/// credits (own BPB lattice → WTA; arch-owner credit disabled) plus
/// `NoScore(NotAttempted)` for every expected participant without a score
/// this epoch. Batch rows whose hotkey left the metagraph are dropped from
/// the set (D24 forbids extra leaves) but stay assigned.
pub fn build_epoch_leaves(
    secret: &[u8; KEY_LEN],
    epoch: u64,
    expected: &ExpectedSet,
    batch: &[EpochScoreRow],
    arch_owners: &BTreeMap<String, String>,
) -> Result<BTreeMap<Hotkey, LeafV1>, EmitError> {
    build_epoch_leaves_with(secret, epoch, expected, batch, arch_owners, None)
}

/// [`build_epoch_leaves`] with optional significance evidence for
/// `PRISM_EMISSION_MODE=sig`. `None` (every caller today) keeps the
/// historical projection exactly.
pub fn build_epoch_leaves_with(
    secret: &[u8; KEY_LEN],
    epoch: u64,
    expected: &ExpectedSet,
    batch: &[EpochScoreRow],
    arch_owners: &BTreeMap<String, String>,
    sig_ctx: Option<&prism_registry::SigContext>,
) -> Result<BTreeMap<Hotkey, LeafV1>, EmitError> {
    // Emission knobs (all default to the historical bit-identical WTA path):
    // `PRISM_EMISSION_MODE` (`wta` | `top3` | `sig`) and
    // `PRISM_OWNER_ARCH_CREDIT_BPS` (0..=5000, owner split of the winner).
    let mode = prism_registry::EmissionMode::from_env();
    let by_miner = prism_registry::emission_leaves_with(
        batch,
        arch_owners,
        mode,
        prism_registry::owner_split_bps_from_env(),
        sig_ctx,
    );
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
    if mode == prism_registry::EmissionMode::Significance {
        apply_burn_leaf(&mut scores, expected, batch, arch_owners, sig_ctx);
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

/// Route the significance rule's unallocated remainder to the burn sink.
///
/// **Why a leaf is needed at all.** `BUNDLE_SPEC` §6.4 normalizes each
/// challenge's positive leaves to sum to 1 before scaling by the
/// challenge's emission share, so a "60 % champion" leaf set with nothing
/// else in it would still deliver 100 % of Prism's share to the champion —
/// the remainder would be silently redistributed, not burned. The only way
/// to make burn real is to allocate the remainder to a hotkey the
/// aggregator drops. Per §6.5 a hotkey mapping to `ZERO_MINER_BURN_UID`
/// (uid 0) is dropped **and its mass burns**, which is exactly the sink
/// this needs.
///
/// Fail-safe: if uid 0 is absent from the expected set, or already carries
/// a positive competition credit, the burn leaf is skipped rather than
/// overwriting a real miner's score. In that case the remainder
/// redistributes (documented in `docs/PRISM.md` as the one case where burn
/// degrades to dilution).
fn apply_burn_leaf(
    scores: &mut BTreeMap<Hotkey, ScoreOrAbsence>,
    expected: &ExpectedSet,
    batch: &[EpochScoreRow],
    arch_owners: &BTreeMap<String, String>,
    sig_ctx: Option<&prism_registry::SigContext>,
) {
    let fallback = prism_registry::SigContext::default();
    let ctx = sig_ctx.unwrap_or(&fallback);
    let credits = prism_registry::competition_scores(batch, arch_owners);
    let plan = prism_registry::plan_emission(&credits, ctx);
    if plan.burn_bps == 0 {
        return;
    }
    let Some(sink) = expected.participants.iter().find(|p| p.uid == BURN_UID) else {
        return;
    };
    // Never clobber a positive score: a real miner at uid 0 keeps its leaf.
    if matches!(
        scores.get(&sink.hotkey),
        Some(ScoreOrAbsence::Score { value }) if *value > 0
    ) {
        return;
    }
    let value = prism_registry::sig::bps_to_lattice(plan.burn_bps);
    scores.insert(sink.hotkey, ScoreOrAbsence::Score { value });
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
            weight_eligible: true,
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

    // ---- v3: the default emission path must not move ----

    #[test]
    fn default_env_leaf_set_is_bit_identical_to_legacy_wta() {
        // The whole v3 change is gated on `PRISM_EMISSION_MODE`. With the
        // env unset (the live configuration), `build_epoch_leaves` must
        // reproduce `apply_wta(competition_scores(..))` exactly — leaf
        // values, absence reasons, set membership, and signatures.
        let (a, b, c) = ([0xAA; 32], [0xBB; 32], [0xCC; 32]);
        let owners: BTreeMap<String, String> = [("arch_x".to_owned(), hex::encode(c))]
            .into_iter()
            .collect();
        let batch = vec![
            row(a, Some("arch_x"), 177_155),
            row(b, Some("arch_x"), 900_000),
        ];
        let exp = expected(&[a, b, c]);

        let got = build_epoch_leaves(&sk(), 42, &exp, &batch, &owners).unwrap();

        // Independent reference: the historical two-call projection.
        let reference =
            prism_registry::apply_wta(prism_registry::competition_scores(&batch, &owners));
        let mut want: BTreeMap<Hotkey, ScoreOrAbsence> = BTreeMap::new();
        for p in &exp.participants {
            let soa = match reference.get(&hex::encode(p.hotkey)) {
                Some(FinalScore::Score(v)) => ScoreOrAbsence::Score { value: *v },
                Some(FinalScore::NoScore(r)) => ScoreOrAbsence::NoScore {
                    reason: reason_from_u8(*r),
                },
                None => ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::NotAttempted,
                },
            };
            want.insert(p.hotkey, soa);
        }
        // sr25519 signatures are randomized, so compare the signed payload
        // (challenge/hotkey/epoch/score) rather than the signature bytes —
        // that payload is what the bundle and the chain vector consume.
        let payload: BTreeMap<Hotkey, (Vec<u8>, u64, ScoreOrAbsence)> = got
            .iter()
            .map(|(hk, l)| {
                (
                    *hk,
                    (l.challenge_id.clone(), l.epoch, l.score_or_absence.clone()),
                )
            })
            .collect();
        let want_payload: BTreeMap<Hotkey, (Vec<u8>, u64, ScoreOrAbsence)> = want
            .iter()
            .map(|(hk, soa)| (*hk, (CHALLENGE_ID_BYTES.to_vec(), 42, soa.clone())))
            .collect();
        assert_eq!(
            payload, want_payload,
            "default path must be identical to legacy WTA"
        );
        // Every leaf must still verify under the challenge key.
        let pk = challenge_common::public_key_from_secret(&sk()).unwrap();
        for leaf in got.values() {
            challenge_common::verify_leaf_sig(leaf, &pk).unwrap();
        }
        // And spelled out: WTA semantics, no burn leaf.
        assert!(matches!(
            soa_of(&got, &b),
            ScoreOrAbsence::Score { value: 900_000 }
        ));
        assert!(matches!(
            soa_of(&got, &a),
            ScoreOrAbsence::Score { value: 0 }
        ));
    }

    #[test]
    fn default_mode_never_emits_a_burn_leaf() {
        // uid 0 is in the expected set; under the default mode it must stay
        // NotAttempted rather than carrying a burn share.
        let (sink, a) = ([0x01; 32], [0xAA; 32]);
        let exp = expected(&[sink, a]);
        assert_eq!(exp.participants[0].uid, 0, "uid 0 present");
        let leaves =
            build_epoch_leaves(&sk(), 9, &exp, &[row(a, None, 500_000)], &BTreeMap::new()).unwrap();
        let sink_hk = exp.participants[0].hotkey;
        assert!(
            matches!(soa_of(&leaves, &sink_hk), ScoreOrAbsence::NoScore { .. }),
            "no burn leaf outside sig mode"
        );
    }

    #[test]
    fn sig_context_is_ignored_by_the_default_mode() {
        // Passing evidence must not change anything while the env selects
        // WTA — the knob, not the argument, switches behavior.
        let (a, b) = ([0xAA; 32], [0xBB; 32]);
        let exp = expected(&[a, b]);
        let batch = vec![row(a, None, 400_000), row(b, None, 900_000)];
        let plain = build_epoch_leaves(&sk(), 7, &exp, &batch, &BTreeMap::new()).unwrap();
        let ctx = prism_registry::SigContext {
            incumbent: Some(hex::encode(a)),
            tenure_days: 30,
            ..prism_registry::SigContext::default()
        };
        let with_ctx =
            build_epoch_leaves_with(&sk(), 7, &exp, &batch, &BTreeMap::new(), Some(&ctx)).unwrap();
        // Signatures are randomized; the scored payload must be unchanged.
        let strip = |m: &BTreeMap<Hotkey, LeafV1>| -> BTreeMap<Hotkey, ScoreOrAbsence> {
            m.iter()
                .map(|(hk, l)| (*hk, l.score_or_absence.clone()))
                .collect()
        };
        assert_eq!(strip(&plain), strip(&with_ctx));
    }
}
