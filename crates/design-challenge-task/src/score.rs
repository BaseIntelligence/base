//! Round + window scoring: admin wins → rolling [`SCORING_WINDOW_ROUNDS`] share.
//! `challenge_scoring_version = 3`: proportional `SCORE_MAX` by window points.

use std::collections::{BTreeMap, BTreeSet};

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use design_store::FinalScore;

use crate::{SCORE_MAX, SCORING_WINDOW_ROUNDS};

/// Inputs for closing / awarding a single round (win accounting).
#[derive(Debug, Clone)]
pub struct ScorePlan {
    /// Miner hotkeys that submitted a harness (any status) this round.
    pub miners_with_harness: Vec<String>,
    /// Miners with at least one clean (`AwaitingAdmin` / scored-clean) run.
    pub miners_clean: Vec<String>,
    /// Winner harness → miner hotkey (len 0–2). Empty = timeout, no new wins.
    pub winner_miners: Vec<String>,
    /// Miners already zeroed by agentic cheat/suspicious (still Score(0)).
    pub cheat_miners: Vec<String>,
}

/// Rolling-window reward projection after aggregating round wins.
#[derive(Debug, Clone)]
pub struct WindowScorePlan {
    /// Miner → round wins accumulated over the scoring window (post-award).
    pub win_counts: BTreeMap<String, u32>,
    /// Miners who participated (harness) on any round of the window.
    pub miners_with_harness: BTreeSet<String>,
    /// Cheat miners (forced Score(0); their wins are excluded from the pool).
    pub cheat_miners: BTreeSet<String>,
}

/// Inclusive first round of the scoring window that ends at `round_id`.
#[must_use]
pub fn window_start(round_id: u64) -> u64 {
    round_id.saturating_sub(SCORING_WINDOW_ROUNDS.saturating_sub(1))
}

/// Per-round win delta from an admin award (or empty on timeout).
///
/// Does **not** assign lattice scores — see [`score_window`].
#[must_use]
pub fn round_win_delta(plan: &ScorePlan) -> BTreeMap<String, u32> {
    let cheat: BTreeSet<_> = plan.cheat_miners.iter().cloned().collect();
    let mut wins = BTreeMap::new();
    for hk in &plan.winner_miners {
        if cheat.contains(hk) {
            continue;
        }
        *wins.entry(hk.clone()).or_insert(0u32) += 1;
    }
    wins
}

/// Compute per-miner final scores over the rolling window (leaf projection).
///
/// Rules (`challenge_scoring_version = 3`):
/// - Each round win is one point; a miner's share is
///   `SCORE_MAX * points / total_points` (integer floor; remainder unused).
/// - Cheat miners' wins are excluded from the pool; cheat → `Score(0)`.
/// - Participants without wins → `Score(0)`.
/// - No eligible wins in the window → all participants `Score(0)`.
#[must_use]
pub fn score_window(plan: &WindowScorePlan) -> BTreeMap<String, FinalScore> {
    let mut out: BTreeMap<String, FinalScore> = BTreeMap::new();

    let mut eligible: BTreeMap<String, u64> = BTreeMap::new();
    for (hk, wins) in &plan.win_counts {
        if plan.cheat_miners.contains(hk) || *wins == 0 {
            continue;
        }
        eligible.insert(hk.clone(), u64::from(*wins));
    }
    let total: u64 = eligible.values().sum();

    if total > 0 {
        for (hk, wins) in &eligible {
            let share = u64::try_from(
                u128::from(SCORE_MAX).saturating_mul(u128::from(*wins)) / u128::from(total),
            )
            .unwrap_or(0);
            out.insert(hk.clone(), FinalScore::Score(share));
        }
    }

    for hk in &plan.miners_with_harness {
        out.entry(hk.clone()).or_insert(FinalScore::Score(0));
    }
    for hk in plan.win_counts.keys() {
        out.entry(hk.clone()).or_insert(FinalScore::Score(0));
    }
    for hk in &plan.cheat_miners {
        out.insert(hk.clone(), FinalScore::Score(0));
    }

    out
}

/// Convert store final score to leaf payload.
#[must_use]
pub fn to_leaf(fs: &FinalScore) -> ScoreOrAbsence {
    match fs {
        FinalScore::Score(v) => ScoreOrAbsence::Score { value: *v },
        FinalScore::NoScore(code) => ScoreOrAbsence::NoScore {
            reason: absence_from_u8(*code),
        },
    }
}

fn absence_from_u8(c: u8) -> NoScoreReasonCode {
    match c {
        0 => NoScoreReasonCode::NotAttempted,
        1 => NoScoreReasonCode::Timeout,
        2 => NoScoreReasonCode::InvalidResponse,
        3 => NoScoreReasonCode::AttestationNotVerified,
        4 => NoScoreReasonCode::MinerError,
        5 => NoScoreReasonCode::RateLimited,
        6 => NoScoreReasonCode::ChallengeInternal,
        _ => NoScoreReasonCode::PolicySkip,
    }
}

/// NotAttempted leaf for miners without a harness.
#[must_use]
pub fn not_attempted() -> FinalScore {
    FinalScore::NoScore(NoScoreReasonCode::NotAttempted as u8)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn plan(wins: &[(&str, u32)], participants: &[&str], cheat: &[&str]) -> WindowScorePlan {
        WindowScorePlan {
            win_counts: wins.iter().map(|(hk, n)| ((*hk).to_owned(), *n)).collect(),
            miners_with_harness: participants.iter().map(|s| (*s).to_owned()).collect(),
            cheat_miners: cheat.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    #[test]
    fn window_zero_rounds_all_zero() {
        let s = score_window(&plan(&[], &["aa", "bb"], &[]));
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(0)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn window_single_winner_full_share() {
        // One win in the window already qualifies (no ≥2 threshold in v3).
        let s = score_window(&plan(&[("aa", 1)], &["aa", "bb"], &[]));
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(SCORE_MAX)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn window_points_share_proportional() {
        // 3 wins vs 1 win → 3/4 and 1/4 of SCORE_MAX.
        let s = score_window(&plan(&[("aa", 3), ("bb", 1)], &["aa", "bb"], &[]));
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(SCORE_MAX / 4 * 3)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(SCORE_MAX / 4)));
    }

    #[test]
    fn window_equal_wins_equal_share() {
        let s = score_window(&plan(&[("aa", 2), ("bb", 2)], &["aa", "bb"], &[]));
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(SCORE_MAX / 2)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(SCORE_MAX / 2)));
    }

    #[test]
    fn window_cheat_excluded_and_zeroed() {
        // Cheater's 5 wins leave the pool; honest miner takes the full share.
        let s = score_window(&plan(&[("aa", 1), ("cc", 5)], &["aa", "cc"], &["cc"]));
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(SCORE_MAX)));
        assert_eq!(s.get("cc"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn window_start_math() {
        assert_eq!(window_start(0), 0);
        assert_eq!(window_start(9), 0);
        assert_eq!(window_start(10), 1);
        assert_eq!(window_start(25), 16);
    }

    #[test]
    fn round_delta_skips_cheat() {
        let plan = ScorePlan {
            miners_with_harness: vec!["aa".into(), "bb".into()],
            miners_clean: vec!["aa".into()],
            winner_miners: vec!["aa".into(), "bb".into()],
            cheat_miners: vec!["bb".into()],
        };
        let d = round_win_delta(&plan);
        assert_eq!(d.get("aa"), Some(&1));
        assert!(!d.contains_key("bb"));
    }
}
