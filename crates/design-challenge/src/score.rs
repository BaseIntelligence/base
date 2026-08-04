//! Round + daily scoring: admin round wins → daily share (≥2 wins) → lattice.
//!
//! Elo / pairwise annotation is no longer on the on-chain leaf path.
//! `challenge_scoring_version = 2`: not WTA — day's qualified winners share
//! `SCORE_MAX` equally. Qualification requires [`MIN_DAILY_WINS`] round wins
//! in the UTC day.

use std::collections::{BTreeMap, BTreeSet};

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use design_challenge_task::{MIN_DAILY_WINS, SCORE_MAX};
use design_store::FinalScore;

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

/// Daily reward projection after aggregating round wins.
#[derive(Debug, Clone)]
pub struct DayScorePlan {
    /// Miner → round wins accumulated for the UTC day (post-award).
    pub win_counts: BTreeMap<String, u32>,
    /// Miners who participated (harness) on any round of the day.
    pub miners_with_harness: BTreeSet<String>,
    /// Cheat miners (forced Score(0), never qualify).
    pub cheat_miners: BTreeSet<String>,
}

/// Per-round win delta from an admin award (or empty on timeout).
///
/// Does **not** assign lattice scores — see [`score_day`].
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

/// Compute per-miner final scores for the UTC day (leaf projection).
///
/// Rules (`challenge_scoring_version = 2`):
/// - Miners with `wins >= MIN_DAILY_WINS` (and not cheat) share `SCORE_MAX`
///   equally (integer division; remainder unused).
/// - Other participants → `Score(0)`.
/// - Cheat → `Score(0)`.
/// - No qualified winners → all participants `Score(0)`.
#[must_use]
pub fn score_day(plan: &DayScorePlan) -> BTreeMap<String, FinalScore> {
    let mut out: BTreeMap<String, FinalScore> = BTreeMap::new();

    let mut qualified: Vec<String> = plan
        .win_counts
        .iter()
        .filter(|(hk, n)| **n >= MIN_DAILY_WINS && !plan.cheat_miners.contains(*hk))
        .map(|(hk, _)| hk.clone())
        .collect();
    qualified.sort();
    qualified.dedup();

    let share = if qualified.is_empty() {
        0
    } else {
        SCORE_MAX / u64::try_from(qualified.len()).unwrap_or(1)
    };

    for hk in &qualified {
        out.insert(hk.clone(), FinalScore::Score(share));
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

/// Legacy helper kept for tests that only exercise a single round as a day.
///
/// Maps one round's winners through [`score_day`] (1 win ⇒ not yet qualified).
#[must_use]
pub fn score_round(plan: &ScorePlan) -> BTreeMap<String, FinalScore> {
    let mut win_counts = BTreeMap::new();
    for (hk, n) in round_win_delta(plan) {
        win_counts.insert(hk, n);
    }
    let mut miners_with_harness: BTreeSet<_> = plan.miners_with_harness.iter().cloned().collect();
    miners_with_harness.extend(plan.miners_clean.iter().cloned());
    let cheat_miners: BTreeSet<_> = plan.cheat_miners.iter().cloned().collect();
    score_day(&DayScorePlan {
        win_counts,
        miners_with_harness,
        cheat_miners,
    })
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

    #[test]
    fn single_round_win_not_qualified() {
        let plan = ScorePlan {
            miners_with_harness: vec!["aa".into(), "bb".into()],
            miners_clean: vec!["aa".into(), "bb".into()],
            winner_miners: vec!["aa".into()],
            cheat_miners: vec![],
        };
        let s = score_round(&plan);
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(0)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn two_daily_wins_full_share() {
        let mut win_counts = BTreeMap::new();
        win_counts.insert("aa".into(), 2);
        win_counts.insert("bb".into(), 1);
        let s = score_day(&DayScorePlan {
            win_counts,
            miners_with_harness: ["aa".into(), "bb".into(), "cc".into()]
                .into_iter()
                .collect(),
            cheat_miners: BTreeSet::new(),
        });
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(SCORE_MAX)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(0)));
        assert_eq!(s.get("cc"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn two_qualified_share_half() {
        let mut win_counts = BTreeMap::new();
        win_counts.insert("aa".into(), 3);
        win_counts.insert("bb".into(), 2);
        let s = score_day(&DayScorePlan {
            win_counts,
            miners_with_harness: ["aa".into(), "bb".into()].into_iter().collect(),
            cheat_miners: BTreeSet::new(),
        });
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(SCORE_MAX / 2)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(SCORE_MAX / 2)));
    }

    #[test]
    fn cheat_never_qualifies() {
        let mut win_counts = BTreeMap::new();
        win_counts.insert("aa".into(), 5);
        let s = score_day(&DayScorePlan {
            win_counts,
            miners_with_harness: ["aa".into()].into_iter().collect(),
            cheat_miners: ["aa".into()].into_iter().collect(),
        });
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn no_winners_all_zero() {
        let plan = ScorePlan {
            miners_with_harness: vec!["aa".into()],
            miners_clean: vec!["aa".into()],
            winner_miners: vec![],
            cheat_miners: vec![],
        };
        let s = score_round(&plan);
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(0)));
    }
}
