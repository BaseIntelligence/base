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
//! - **Architecture-owner credit** *(temporarily disabled — see
//!   [`OWNER_ARCH_CREDIT_ENABLED`])*: for each registered arch, the arch's
//!   best epoch result (`max(Score)` over all rows linked to that arch, any
//!   trainer) is credited to the arch's owner — the owner is rewarded when
//!   *anyone* trains well on their architecture.
//! - **Per-hotkey credit**: while owner credit is disabled, this is own
//!   credit only. When re-enabled: `max(own credits, owner credits)` —
//!   never summed, so the SCORE_MAX lattice bound and the no-double-count
//!   property hold by construction. Hotkeys whose rows are all `NoScore`
//!   keep their absence; `Score(0)` rows (cheat / copy-gate reject) emit 0
//!   and never set an arch's best.
//! - **WTA leaf emission**: [`apply_wta`] collapses the credit map to a
//!   single positive `Score` (argmax; lexicographically smallest hotkey on
//!   ties). Prism's emission share goes to that one winner.

use std::collections::BTreeMap;

use prism_store::{EpochScoreRow, FinalScore};

/// Temporary kill-switch for architecture-owner emission credit.
///
/// When `false` (current prod intent): leaf emission uses **own BPB-derived
/// lattice scores only**. An arch owner is not credited when a challenger
/// trains well on their architecture, so the BPB winner keeps the WTA leaf
/// (and Prism's weight share) instead of losing a lex tie to a non-training
/// owner — including owners who have left the metagraph.
///
/// Flip back to `true` to restore the architecture-competition owner credit
/// documented in `docs/PRISM.md`.
pub const OWNER_ARCH_CREDIT_ENABLED: bool = false;

/// Compute per-hotkey emission for one epoch.
///
/// `arch_owners` is ignored while [`OWNER_ARCH_CREDIT_ENABLED`] is `false`.
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
                if OWNER_ARCH_CREDIT_ENABLED {
                    if let Some(a) = r.arch_id.as_deref() {
                        let e = arch_best.entry(a).or_insert(0);
                        *e = (*e).max(*v);
                    }
                }
            }
            FinalScore::NoScore(reason) => {
                absence.entry(r.miner_hotkey.clone()).or_insert(*reason);
            }
        }
    }

    // Owner credits: arch epoch best → arch owner (gated).
    let mut owner_credit: BTreeMap<String, u64> = BTreeMap::new();
    if OWNER_ARCH_CREDIT_ENABLED {
        for (arch_id, best) in &arch_best {
            if let Some(owner) = arch_owners.get(*arch_id) {
                let e = owner_credit.entry(owner.clone()).or_insert(0);
                *e = (*e).max(*best);
            }
        }
    } else {
        // Silence unused-param lint while the kill-switch is off; callers
        // still pass the registry map so re-enable is a one-line flip.
        let _ = arch_owners;
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
    fn owner_arch_credit_temporarily_disabled() {
        assert!(
            !OWNER_ARCH_CREDIT_ENABLED,
            "flip this test when restoring owner-arch credit"
        );
        // Challenger BB trains owner AA's arch to 900k. With owner credit
        // off, AA keeps only their own 400k — BB (best BPB/score) wins WTA.
        let rows = vec![
            row("aa", Some("arch_x"), 400_000),
            row("bb", Some("arch_x"), 900_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(400_000)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(900_000)));
        let wta = apply_wta(out);
        assert_eq!(wta.get("bb"), Some(&FinalScore::Score(900_000)));
        assert_eq!(wta.get("aa"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn owner_not_credited_across_archs_while_disabled() {
        let rows = vec![
            row("bb", Some("arch_x"), 300_000),
            row("cc", Some("arch_y"), 500_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa"), ("arch_y", "aa")]));
        assert!(out.get("aa").is_none());
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(300_000)));
        assert_eq!(out.get("cc"), Some(&FinalScore::Score(500_000)));
    }

    #[test]
    fn zero_scores_do_not_invent_owner_rows() {
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
    fn challenger_keeps_own_score_without_owner_boost() {
        // A owns arch X but owner credit is off — only own rows count.
        let rows = vec![
            row("bb", Some("arch_x"), 300_000),
            row("aa", Some("arch_y"), 800_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa"), ("arch_y", "cc")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(800_000)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(300_000)));
        assert!(out.get("cc").is_none());
    }

    #[test]
    fn wta_keeps_only_the_argmax_score() {
        let rows = vec![row("aa", None, 177_155), row("bb", Some("arch_x"), 111_595)];
        let credits = competition_scores(&rows, &owners(&[("arch_x", "cc")]));
        // Own-only: aa=177155, bb=111595 — owner cc gets nothing.
        let wta = apply_wta(credits);
        assert_eq!(wta.get("aa"), Some(&FinalScore::Score(177_155)));
        assert_eq!(wta.get("bb"), Some(&FinalScore::Score(0)));
        assert!(wta.get("cc").is_none());
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
