//! Epoch scoring: TARGET_BUGS burn-sink formula.
//!
//! ```text
//! capped = min(total, TARGET)
//! miner_pool = SCORE_MAX * capped / TARGET
//! burn_units = SCORE_MAX - miner_pool
//! miner m: floor(miner_pool * points_m / total)   when total > 0
//! ```

use std::collections::BTreeMap;

use crate::{SCORE_MAX, TARGET_BUGS};

/// Inputs for one epoch's bounty leaf projection.
#[derive(Debug, Clone, Default)]
pub struct EpochScoreInput {
    /// Miner hotkey → approved bug count this epoch.
    pub approved_points: BTreeMap<String, u32>,
}

/// Pure scoring outcome (emit maps `burn_units` onto metagraph uid=0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochScoreOutcome {
    /// Per-miner lattice scores (`Score(n)` values; absent miners omitted).
    pub miner_scores: BTreeMap<String, u64>,
    /// Units for the UID-0 burn sink leaf (`0` when `total >= TARGET_BUGS`).
    pub burn_units: u64,
    /// Pool allocated to miners before per-miner floor split.
    pub miner_pool: u64,
    /// Sum of approved points (uncapped).
    pub total_approved: u64,
    /// `min(total_approved, TARGET_BUGS)`.
    pub capped: u64,
}

/// Compute miner scores + burn sink for one epoch.
///
/// Rules:
/// - `capped = min(total, TARGET_BUGS)`
/// - `miner_pool = SCORE_MAX * capped / TARGET_BUGS` (integer)
/// - `burn_units = SCORE_MAX - miner_pool`
/// - When `total > 0`, each miner with points gets
///   `floor(miner_pool * points_m / total)`
/// - When `total == 0`, `miner_scores` is empty and `burn_units == SCORE_MAX`
/// - Above target: `burn_units == 0`, full pool diluted across all points
#[must_use]
pub fn score_epoch(input: &EpochScoreInput) -> EpochScoreOutcome {
    let total_approved: u64 = input.approved_points.values().map(|p| u64::from(*p)).sum();
    let capped = total_approved.min(TARGET_BUGS);
    let miner_pool = if TARGET_BUGS == 0 {
        0
    } else {
        u64::try_from(
            u128::from(SCORE_MAX).saturating_mul(u128::from(capped)) / u128::from(TARGET_BUGS),
        )
        .unwrap_or(0)
    };
    let burn_units = SCORE_MAX.saturating_sub(miner_pool);

    let mut miner_scores = BTreeMap::new();
    if total_approved > 0 {
        for (hk, pts) in &input.approved_points {
            if *pts == 0 {
                miner_scores.insert(hk.clone(), 0);
                continue;
            }
            let share = u64::try_from(
                u128::from(miner_pool).saturating_mul(u128::from(*pts))
                    / u128::from(total_approved),
            )
            .unwrap_or(0);
            miner_scores.insert(hk.clone(), share);
        }
    }

    EpochScoreOutcome {
        miner_scores,
        burn_units,
        miner_pool,
        total_approved,
        capped,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn input(points: &[(&str, u32)]) -> EpochScoreInput {
        EpochScoreInput {
            approved_points: points
                .iter()
                .map(|(hk, n)| ((*hk).to_owned(), *n))
                .collect(),
        }
    }

    #[test]
    fn zero_bugs_full_burn() {
        let out = score_epoch(&input(&[]));
        assert_eq!(out.total_approved, 0);
        assert_eq!(out.capped, 0);
        assert_eq!(out.miner_pool, 0);
        assert_eq!(out.burn_units, SCORE_MAX);
        assert!(out.miner_scores.is_empty());
    }

    #[test]
    fn twenty_five_bugs_half_pool_half_burn() {
        // 25 / 50 → miner_pool = SCORE_MAX / 2, burn = SCORE_MAX / 2.
        let out = score_epoch(&input(&[("aa", 25)]));
        assert_eq!(out.total_approved, 25);
        assert_eq!(out.capped, 25);
        assert_eq!(out.miner_pool, SCORE_MAX / 2);
        assert_eq!(out.burn_units, SCORE_MAX / 2);
        assert_eq!(out.miner_scores.get("aa"), Some(&(SCORE_MAX / 2)));
    }

    #[test]
    fn fifty_bugs_full_pool_no_burn() {
        let out = score_epoch(&input(&[("aa", 30), ("bb", 20)]));
        assert_eq!(out.total_approved, 50);
        assert_eq!(out.capped, 50);
        assert_eq!(out.miner_pool, SCORE_MAX);
        assert_eq!(out.burn_units, 0);
        // 30/50 and 20/50 of SCORE_MAX.
        assert_eq!(out.miner_scores.get("aa"), Some(&(SCORE_MAX / 5 * 3)));
        assert_eq!(out.miner_scores.get("bb"), Some(&(SCORE_MAX / 5 * 2)));
    }

    #[test]
    fn seventy_five_bugs_dilution_no_burn() {
        // Above target: capped = 50, full pool, burn 0, share by uncapped total.
        let out = score_epoch(&input(&[("aa", 50), ("bb", 25)]));
        assert_eq!(out.total_approved, 75);
        assert_eq!(out.capped, TARGET_BUGS);
        assert_eq!(out.miner_pool, SCORE_MAX);
        assert_eq!(out.burn_units, 0);
        assert_eq!(out.miner_scores.get("aa"), Some(&(SCORE_MAX * 50 / 75)));
        assert_eq!(out.miner_scores.get("bb"), Some(&(SCORE_MAX * 25 / 75)));
    }

    #[test]
    fn single_miner_at_target_takes_all() {
        let out = score_epoch(&input(&[("aa", 50)]));
        assert_eq!(out.burn_units, 0);
        assert_eq!(out.miner_scores.get("aa"), Some(&SCORE_MAX));
    }
}
