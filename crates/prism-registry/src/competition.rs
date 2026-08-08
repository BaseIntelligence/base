//! Architecture competition emission math (epoch-local, lattice-preserving).
//!
//! Exact rule (mirrors `docs/PRISM.md` § Architecture competition):
//!
//! - Input: every scored submission row of the epoch (multiple rows per
//!   hotkey are possible: 1 architecture submission + 1 training-only entry
//!   per published arch) plus the registry ownership map.
//! - **Challenger credit**: a hotkey's own rows — its best lattice score,
//!   i.e. `max(Score)` over its submissions this epoch (own best training
//!   result per arch, then across archs).
//! - **Architecture-owner credit**: for each registered arch, the arch's
//!   best epoch result (`max(Score)` over all rows linked to that arch, any
//!   trainer) is credited to the arch's owner — the owner is rewarded when
//!   *anyone* trains well on their architecture.
//! - **Per-hotkey credit**: `max(own credits, owner credits)` — never
//!   summed, so the SCORE_MAX lattice bound and the no-double-count property
//!   hold by construction. Hotkeys whose rows are all `NoScore` keep their
//!   absence; `Score(0)` rows (cheat / copy-gate reject) emit 0 and never
//!   set an arch's best.
//! - **WTA leaf emission**: [`apply_wta`] collapses the credit map to a
//!   single positive `Score` (argmax; lexicographically smallest hotkey on
//!   ties). Prism's emission share goes to that one winner.

use std::collections::BTreeMap;

use prism_store::{EpochScoreRow, FinalScore};

/// Compute per-hotkey emission for one epoch.
#[must_use]
pub fn competition_scores(
    rows: &[EpochScoreRow],
    arch_owners: &BTreeMap<String, String>,
) -> BTreeMap<String, FinalScore> {
    // Arch epoch best (max lattice score among linked rows, any trainer).
    let mut arch_best: BTreeMap<&str, u64> = BTreeMap::new();
    // Own credits per hotkey.
    let mut own: BTreeMap<String, u64> = BTreeMap::new();
    // Absence fallback for hotkeys with no score rows at all.
    let mut absence: BTreeMap<String, u8> = BTreeMap::new();

    for r in rows {
        match &r.final_score {
            FinalScore::Score(v) => {
                let e = own.entry(r.miner_hotkey.clone()).or_insert(0);
                *e = (*e).max(*v);
                if let Some(a) = r.arch_id.as_deref() {
                    let e = arch_best.entry(a).or_insert(0);
                    *e = (*e).max(*v);
                }
            }
            FinalScore::NoScore(reason) => {
                absence.entry(r.miner_hotkey.clone()).or_insert(*reason);
            }
        }
    }

    // Owner credits: arch epoch best → arch owner.
    let mut owner_credit: BTreeMap<String, u64> = BTreeMap::new();
    for (arch_id, best) in &arch_best {
        if let Some(owner) = arch_owners.get(*arch_id) {
            let e = owner_credit.entry(owner.clone()).or_insert(0);
            *e = (*e).max(*best);
        }
    }

    let mut out: BTreeMap<String, FinalScore> = BTreeMap::new();
    for hk in own.keys().chain(owner_credit.keys()).chain(absence.keys()) {
        if out.contains_key(hk) {
            continue;
        }
        let score = own
            .get(hk)
            .copied()
            .unwrap_or(0)
            .max(owner_credit.get(hk).copied().unwrap_or(0));
        if score > 0 || own.contains_key(hk) {
            out.insert(hk.clone(), FinalScore::Score(score));
        } else if let Some(reason) = absence.get(hk) {
            out.insert(hk.clone(), FinalScore::NoScore(*reason));
        }
    }
    out
}

/// Winner-take-all collapse: keep only the single highest positive score.
///
/// Ties break by lexicographically smallest hotkey (stable, hex-encoded).
/// Non-positive `Score(0)` and `NoScore` rows are preserved unchanged;
/// every other positive `Score` is zeroed so the aggregator cannot soft-
/// allocate Prism's share across multiple hotkeys.
#[must_use]
pub fn apply_wta(scores: BTreeMap<String, FinalScore>) -> BTreeMap<String, FinalScore> {
    let winner = scores
        .iter()
        .filter_map(|(hk, s)| match s {
            FinalScore::Score(v) if *v > 0 => Some((hk.as_str(), *v)),
            _ => None,
        })
        // Higher score wins; on equal score prefer the smaller hotkey.
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(hk, _)| hk.to_owned());
    let Some(winner) = winner else {
        return scores;
    };
    scores
        .into_iter()
        .map(|(hk, s)| match &s {
            FinalScore::Score(v) if *v > 0 && hk != winner => (hk, FinalScore::Score(0)),
            _ => (hk, s),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn row(hk: &str, arch: Option<&str>, score: u64) -> EpochScoreRow {
        EpochScoreRow {
            miner_hotkey: hk.into(),
            arch_id: arch.map(str::to_owned),
            final_score: FinalScore::Score(score),
        }
    }

    fn owners(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(a, o)| ((*a).to_owned(), (*o).to_owned()))
            .collect()
    }

    #[test]
    fn owner_credited_for_challenger_result_on_their_arch() {
        // Owner A published arch X (scored 400k on their own submission).
        // Challenger B trains on X and scores 900k. A's emission = 900k
        // (arch best), B's emission = 900k (own best) — max, not summed.
        let rows = vec![
            row("aa", Some("arch_x"), 400_000),
            row("bb", Some("arch_x"), 900_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(900_000)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(900_000)));
    }

    #[test]
    fn owner_emission_is_max_across_archs_not_sum() {
        let rows = vec![
            row("bb", Some("arch_x"), 300_000),
            row("cc", Some("arch_y"), 500_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa"), ("arch_y", "aa")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(500_000)));
    }

    #[test]
    fn zero_scores_never_set_arch_best() {
        let rows = vec![row("bb", Some("arch_x"), 0), row("aa", Some("arch_x"), 0)];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(0)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn unlinked_rows_only_own_credit() {
        let rows = vec![row("aa", None, 700_000), row("bb", None, 200_000)];
        let out = competition_scores(&rows, &BTreeMap::new());
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(700_000)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(200_000)));
    }

    #[test]
    fn absence_preserved_for_scoreless_hotkeys() {
        let rows = vec![
            EpochScoreRow {
                miner_hotkey: "cc".into(),
                arch_id: None,
                final_score: FinalScore::NoScore(6),
            },
            row("aa", None, 100_000),
        ];
        let out = competition_scores(&rows, &BTreeMap::new());
        assert_eq!(out.get("cc"), Some(&FinalScore::NoScore(6)));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(100_000)));
    }

    #[test]
    fn own_and_owner_credits_take_max() {
        // A owns arch X (epoch best 300k by challenger) and also scores
        // 800k on their own training-only entry on arch Y.
        let rows = vec![
            row("bb", Some("arch_x"), 300_000),
            row("aa", Some("arch_y"), 800_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa"), ("arch_y", "cc")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(800_000)));
        // Y's epoch best is 800k (by A) → credited to Y's owner C.
        assert_eq!(out.get("cc"), Some(&FinalScore::Score(800_000)));
    }

    #[test]
    fn wta_keeps_only_the_argmax_score() {
        let rows = vec![
            row("aa", None, 177_155),
            row("bb", Some("arch_x"), 111_595),
        ];
        let credits = competition_scores(&rows, &owners(&[("arch_x", "cc")]));
        // Credits: aa=177155, bb=111595, cc=111595 (owner).
        let wta = apply_wta(credits);
        assert_eq!(wta.get("aa"), Some(&FinalScore::Score(177_155)));
        assert_eq!(wta.get("bb"), Some(&FinalScore::Score(0)));
        assert_eq!(wta.get("cc"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn wta_tie_breaks_by_lexicographically_smallest_hotkey() {
        let mut credits = BTreeMap::new();
        credits.insert("bb".into(), FinalScore::Score(900_000));
        credits.insert("aa".into(), FinalScore::Score(900_000));
        let wta = apply_wta(credits);
        assert_eq!(wta.get("aa"), Some(&FinalScore::Score(900_000)));
        assert_eq!(wta.get("bb"), Some(&FinalScore::Score(0)));
    }
}
