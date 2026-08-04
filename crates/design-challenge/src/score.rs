//! Round scoring: admin winners → lattice + absence reasons.
//!
//! Elo / pairwise annotation is no longer on the on-chain leaf path.

use std::collections::{BTreeMap, BTreeSet};

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use design_challenge_task::SCORE_MAX;
use design_store::FinalScore;

/// Inputs for closing / awarding a round.
#[derive(Debug, Clone)]
pub struct ScorePlan {
    /// Miner hotkeys that submitted a harness (any status) this round.
    pub miners_with_harness: Vec<String>,
    /// Miners with at least one clean (`AwaitingAdmin` / scored-clean) run.
    pub miners_clean: Vec<String>,
    /// Winner harness → miner hotkey (len 0–2). Empty = timeout, all Score(0).
    pub winner_miners: Vec<String>,
    /// Miners already zeroed by agentic cheat/suspicious (still Score(0)).
    pub cheat_miners: Vec<String>,
}

/// Compute per-miner final scores for leaf emission.
///
/// Rules:
/// - 1 winner → `Score(SCORE_MAX)`
/// - 2 winners → each `Score(SCORE_MAX / 2)`
/// - other clean / cheat → `Score(0)`
/// - harness but no clean attempt still listed in `miners_with_harness` → `Score(0)`
/// - no harness is handled by the caller as `NotAttempted` on the expected set
#[must_use]
pub fn score_round(plan: &ScorePlan) -> BTreeMap<String, FinalScore> {
    let mut out: BTreeMap<String, FinalScore> = BTreeMap::new();

    let winners: BTreeSet<_> = plan.winner_miners.iter().cloned().collect();
    let winner_score = match winners.len() {
        1 => SCORE_MAX,
        2 => SCORE_MAX / 2,
        _ => 0,
    };

    for hk in &winners {
        out.insert(hk.clone(), FinalScore::Score(winner_score));
    }

    for hk in &plan.miners_clean {
        out.entry(hk.clone()).or_insert(FinalScore::Score(0));
    }

    for hk in &plan.miners_with_harness {
        out.entry(hk.clone()).or_insert(FinalScore::Score(0));
    }

    // Cheat / suspicious always Score(0), even if mis-nominated as winner.
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

    #[test]
    fn two_winners_half_score() {
        let plan = ScorePlan {
            miners_with_harness: vec!["aa".into(), "bb".into(), "cc".into()],
            miners_clean: vec!["aa".into(), "bb".into(), "cc".into()],
            winner_miners: vec!["aa".into(), "bb".into()],
            cheat_miners: vec![],
        };
        let s = score_round(&plan);
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(SCORE_MAX / 2)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(SCORE_MAX / 2)));
        assert_eq!(s.get("cc"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn one_winner_full_score() {
        let plan = ScorePlan {
            miners_with_harness: vec!["aa".into(), "bb".into()],
            miners_clean: vec!["aa".into(), "bb".into()],
            winner_miners: vec!["aa".into()],
            cheat_miners: vec![],
        };
        let s = score_round(&plan);
        assert_eq!(s.get("aa"), Some(&FinalScore::Score(SCORE_MAX)));
        assert_eq!(s.get("bb"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn cheat_zero_even_if_listed_winner() {
        // Award path should not nominate cheat; defensive Score(0) if listed in cheat.
        let plan = ScorePlan {
            miners_with_harness: vec!["aa".into()],
            miners_clean: vec![],
            winner_miners: vec![],
            cheat_miners: vec!["aa".into()],
        };
        let s = score_round(&plan);
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
