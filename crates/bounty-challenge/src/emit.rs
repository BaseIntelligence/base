//! D24 exact-E leaf emission for `challenge_id = "bounty"`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bounty_challenge_task::{score_epoch, EpochScoreInput, CHALLENGE_ID_BYTES, SCORE_MAX};
use bounty_store::{BountyStore, EpochScoreRow, FinalScore};
use bundle::{LeafV1, NoScoreReasonCode, ScoreOrAbsence};
use challenge_common::{
    emit_signed_leaf_set, hotkey_hex, submit_signed_leaf_set, ExpectedSet, GatewayClient, Hotkey,
    LeafEmitError,
};
use crypto::KEY_LEN;
use thiserror::Error;
use tracing::info;
use trustroot::decode_hex_array;

/// Emission failures.
#[derive(Debug, Error)]
pub enum EmitError {
    /// Store fault.
    #[error("store: {0}")]
    Store(String),
    /// Leaf signing / D24 shape.
    #[error("emit: {0}")]
    Sign(#[from] LeafEmitError),
    /// Gateway submit.
    #[error("submit: {0}")]
    Submit(String),
    /// Missing uid=0 participant in expected set.
    #[error("expected set has no uid=0 hotkey")]
    NoUidZero,
}

/// One emitted leaf set summary.
#[derive(Debug, Clone)]
pub struct EmitSummary {
    /// Leaf epoch.
    pub epoch: u64,
    /// Full D24 set size.
    pub leaves: usize,
    /// Burn units assigned to uid=0.
    pub burn_units: u64,
    /// Miner pool before burn.
    pub miner_pool: u64,
    /// Signed leaves (test introspection).
    pub signed: BTreeMap<Hotkey, LeafV1>,
}

/// Sign exactly one leaf per `h ∈ expected` under `bounty`.
///
/// # Errors
/// Missing/unknown hotkeys or signing failure.
pub fn emit_signed_bounty_leaf_set(
    secret: &[u8; KEY_LEN],
    epoch: u64,
    expected: &BTreeSet<Hotkey>,
    scores: &BTreeMap<Hotkey, ScoreOrAbsence>,
) -> Result<BTreeMap<Hotkey, LeafV1>, LeafEmitError> {
    emit_signed_leaf_set(secret, CHALLENGE_ID_BYTES, epoch, expected, scores)
}

/// Build D24 scores from epoch approved points + burn sink on uid=0.
///
/// # Errors
/// [`EmitError::NoUidZero`] when the expected set has no uid 0.
pub fn build_epoch_scores(
    expected: &ExpectedSet,
    approved_points: &BTreeMap<String, u32>,
) -> Result<(BTreeMap<Hotkey, ScoreOrAbsence>, u64, u64), EmitError> {
    let uid0 = expected
        .participants
        .iter()
        .find(|p| p.uid == 0)
        .map(|p| p.hotkey)
        .ok_or(EmitError::NoUidZero)?;

    let outcome = score_epoch(&EpochScoreInput {
        approved_points: approved_points.clone(),
    });

    let mut scores: BTreeMap<Hotkey, ScoreOrAbsence> = BTreeMap::new();
    for p in &expected.participants {
        let hex = hotkey_hex(&p.hotkey);
        let miner_val = outcome.miner_scores.get(&hex).copied().unwrap_or(0);
        if p.hotkey == uid0 {
            // UID0 receives burn mass (+ any approved points if they also mine).
            let total = miner_val.saturating_add(outcome.burn_units);
            scores.insert(
                p.hotkey,
                if total > 0 {
                    ScoreOrAbsence::Score {
                        value: total.min(SCORE_MAX),
                    }
                } else {
                    ScoreOrAbsence::NoScore {
                        reason: NoScoreReasonCode::NotAttempted,
                    }
                },
            );
        } else if miner_val > 0 {
            scores.insert(p.hotkey, ScoreOrAbsence::Score { value: miner_val });
        } else {
            scores.insert(
                p.hotkey,
                ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::NotAttempted,
                },
            );
        }
    }
    Ok((scores, outcome.burn_units, outcome.miner_pool))
}

/// Persist score rows for audit after a successful emit.
async fn persist_scores(
    store: &dyn BountyStore,
    epoch: u64,
    approved_points: &BTreeMap<String, u32>,
    scores: &BTreeMap<Hotkey, ScoreOrAbsence>,
) -> Result<(), EmitError> {
    for (hk, soa) in scores {
        let hex = hotkey_hex(hk);
        let points = approved_points.get(&hex).copied().unwrap_or(0);
        let final_score = match soa {
            ScoreOrAbsence::Score { value } => Some(FinalScore::Score(*value)),
            ScoreOrAbsence::NoScore { reason } => Some(FinalScore::NoScore(*reason as u8)),
        };
        store
            .upsert_epoch_score(&EpochScoreRow {
                epoch,
                miner_hotkey: hex,
                points,
                final_score,
            })
            .await
            .map_err(|e| EmitError::Store(e.to_string()))?;
    }
    Ok(())
}

/// Emit one epoch's bounty leaf set (idempotent when scores already exist).
///
/// # Errors
/// Store / sign / submit failures.
pub async fn emit_epoch(
    store: Arc<dyn BountyStore>,
    gateway: &GatewayClient,
    secret: &[u8; KEY_LEN],
    epoch: u64,
    expected: &ExpectedSet,
) -> Result<Option<EmitSummary>, EmitError> {
    let existing = store
        .list_epoch_scores(epoch)
        .await
        .map_err(|e| EmitError::Store(e.to_string()))?;
    if existing.iter().any(|r| r.final_score.is_some()) {
        return Ok(None);
    }

    let approved = store
        .approved_points_for_epoch(epoch)
        .await
        .map_err(|e| EmitError::Store(e.to_string()))?;
    let (scores, burn_units, miner_pool) = build_epoch_scores(expected, &approved)?;
    let hotkeys = expected.hotkeys();
    let signed = emit_signed_bounty_leaf_set(secret, epoch, &hotkeys, &scores)?;
    submit_signed_leaf_set(gateway, &signed)
        .await
        .map_err(|e| EmitError::Submit(e.to_string()))?;
    persist_scores(store.as_ref(), epoch, &approved, &scores).await?;
    info!(
        epoch,
        leaves = signed.len(),
        burn_units,
        miner_pool,
        "bounty epoch leaf set emitted"
    );
    Ok(Some(EmitSummary {
        epoch,
        leaves: signed.len(),
        burn_units,
        miner_pool,
        signed,
    }))
}

/// Decode lowercase 64-hex hotkey.
///
/// # Errors
/// Bad hex.
pub fn hotkey_from_hex(s: &str) -> Result<Hotkey, String> {
    decode_hex_array::<KEY_LEN>(s).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use bounty_store::MemoryBountyStore;
    use challenge_common::{
        public_key_from_secret, verify_leaf_sig, ExpectedParticipant, GatewayClientConfig,
        DRY_RUN_BASE_URL,
    };

    fn sk() -> [u8; KEY_LEN] {
        let mut s = [7u8; KEY_LEN];
        s[0] = 0x42;
        s
    }

    fn expected_two() -> ExpectedSet {
        let uid0 = [1u8; KEY_LEN];
        let miner = [2u8; KEY_LEN];
        ExpectedSet {
            block_hash: [9u8; 32],
            participants: vec![
                ExpectedParticipant {
                    hotkey: uid0,
                    uid: 0,
                },
                ExpectedParticipant {
                    hotkey: miner,
                    uid: 1,
                },
            ],
        }
    }

    #[test]
    fn burn_sink_on_uid0_when_under_target() {
        let exp = expected_two();
        let miner_hex = hotkey_hex(&[2u8; KEY_LEN]);
        let mut pts = BTreeMap::new();
        pts.insert(miner_hex, 25);
        let (scores, burn, pool) = build_epoch_scores(&exp, &pts).unwrap();
        assert_eq!(pool, SCORE_MAX / 2);
        assert_eq!(burn, SCORE_MAX / 2);
        match scores.get(&[1u8; KEY_LEN]).unwrap() {
            ScoreOrAbsence::Score { value } => assert_eq!(*value, SCORE_MAX / 2),
            ScoreOrAbsence::NoScore { .. } => panic!("expected burn score"),
        }
        match scores.get(&[2u8; KEY_LEN]).unwrap() {
            ScoreOrAbsence::Score { value } => assert_eq!(*value, SCORE_MAX / 2),
            ScoreOrAbsence::NoScore { .. } => panic!("expected miner score"),
        }
    }

    #[tokio::test]
    async fn emit_epoch_dry_run_and_idempotent() {
        let store: Arc<dyn BountyStore> = Arc::new(MemoryBountyStore::new());
        let gw = GatewayClient::new(GatewayClientConfig {
            base_url: DRY_RUN_BASE_URL.into(),
            ..Default::default()
        })
        .unwrap();
        let exp = expected_two();
        let first = emit_epoch(Arc::clone(&store), &gw, &sk(), 7, &exp)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first.leaves, 2);
        assert_eq!(first.burn_units, SCORE_MAX); // zero approved → full burn
        let pk = public_key_from_secret(&sk()).unwrap();
        for leaf in first.signed.values() {
            verify_leaf_sig(leaf, &pk).unwrap();
        }
        let second = emit_epoch(store, &gw, &sk(), 7, &exp).await.unwrap();
        assert!(second.is_none());
    }
}
