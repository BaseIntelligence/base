//! Per-axis elite archive (Prism v3, default-off).
//!
//! **What it is.** For each scored axis (`g1`..`g8`) the archive records
//! which submission holds the best measured value. A submission that is
//! third on the composite but **first on G3** (associative recall) or
//! **first on G7** (inference cost) has produced transferable information,
//! and the exploration pool pays for exactly that.
//!
//! **Why this shape and not a novelty score.** Nobody has made "pay for
//! measured difference" work: Numerai marketed *being different pays* and
//! implemented *marginal contribution* instead, and the component that got
//! exploited was the rank-shaped bonus. Measured-novelty terms are
//! maximizable by renaming variables and reordering statements — the
//! plagiarism-detector literature measures obfuscation defeating MOSS and
//! Dolos outright. A per-axis frontier cannot be faked that way: being
//! best at algorithmic reasoning requires being best at algorithmic
//! reasoning.
//!
//! **The cells are operator-owned.** The descriptor space is the fixed
//! group set `g1..g8`. A miner cannot invent a ninth axis to be best at,
//! which is the structural difference from a novelty-distance score where
//! the participant chooses the direction of "difference".
//!
//! **Derivable, not trusted state.** The archive is recomputed from the
//! same stored per-group values the composite already persists. It holds
//! no state a miner can influence and nothing carries over except the
//! measurements themselves.

use std::collections::{BTreeMap, BTreeSet};

/// Maximum exploration-pool slots (design cap: a false-positive frontier
/// claim costs at most one slot's share).
pub const MAX_EXPLORE_SLOTS: usize = 5;

/// One submission's measured value on one axis.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisScore {
    /// Group key (`g1`..`g8`).
    pub axis: String,
    /// Submitting hotkey (hex).
    pub hotkey: String,
    /// Normalized per-group value; higher is better on every axis after
    /// the composite's per-group normalization.
    pub value: f64,
    /// Whether the submission passed every lexicographic gate. Only
    /// gate-passing entries may hold a frontier.
    pub gates_ok: bool,
}

/// Who holds each axis frontier, and the value that holds it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EliteArchive {
    /// Axis -> (holder hotkey, value).
    pub holders: BTreeMap<String, (String, f64)>,
}

impl EliteArchive {
    /// Recompute the archive from the round's per-axis measurements.
    ///
    /// Ties resolve to the lexicographically smallest hotkey, matching the
    /// emission tie convention elsewhere in this crate. Non-finite values
    /// and gate failures never hold a frontier.
    #[must_use]
    pub fn build(scores: &[AxisScore]) -> Self {
        let mut holders: BTreeMap<String, (String, f64)> = BTreeMap::new();
        for s in scores {
            if !s.gates_ok || !s.value.is_finite() {
                continue;
            }
            match holders.get(&s.axis) {
                Some((held_by, held)) => {
                    // Strictly better wins; equal value breaks to lex-min.
                    let better = s.value > *held;
                    let tie_wins = (s.value - *held).abs() < f64::EPSILON && s.hotkey < *held_by;
                    if better || tie_wins {
                        holders.insert(s.axis.clone(), (s.hotkey.clone(), s.value));
                    }
                }
                None => {
                    holders.insert(s.axis.clone(), (s.hotkey.clone(), s.value));
                }
            }
        }
        Self { holders }
    }

    /// Axes held by `hotkey`.
    #[must_use]
    pub fn axes_held_by(&self, hotkey: &str) -> Vec<String> {
        self.holders
            .iter()
            .filter(|(_, (hk, _))| hk == hotkey)
            .map(|(axis, _)| axis.clone())
            .collect()
    }

    /// Distinct hotkeys holding at least one axis frontier — the
    /// "archive occupancy" diversity statistic.
    #[must_use]
    pub fn occupancy(&self) -> usize {
        self.holders
            .values()
            .map(|(hk, _)| hk.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Exploration-pool recipients: hotkeys that hold ≥1 axis frontier,
    /// excluding those already paid by the champion/band tiers, capped at
    /// [`MAX_EXPLORE_SLOTS`].
    ///
    /// Ordering is deterministic: most axes held first, then lex-smallest
    /// hotkey, so the same inputs always select the same slots.
    #[must_use]
    pub fn explore_slots(&self, already_paid: &BTreeSet<String>) -> Vec<String> {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for (hk, _) in self.holders.values() {
            if already_paid.contains(hk.as_str()) {
                continue;
            }
            *counts.entry(hk.as_str()).or_insert(0) += 1;
        }
        let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        ranked
            .into_iter()
            .take(MAX_EXPLORE_SLOTS)
            .map(|(hk, _)| hk.to_owned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn axis(axis: &str, hotkey: &str, value: f64) -> AxisScore {
        AxisScore {
            axis: axis.into(),
            hotkey: hotkey.into(),
            value,
            gates_ok: true,
        }
    }

    fn paid(hks: &[&str]) -> BTreeSet<String> {
        hks.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn best_value_per_axis_holds_the_frontier() {
        let a = EliteArchive::build(&[
            axis("g1", "aa", 0.50),
            axis("g1", "bb", 0.70),
            axis("g3", "aa", 0.90),
            axis("g3", "bb", 0.10),
        ]);
        assert_eq!(a.holders.get("g1").unwrap().0, "bb");
        assert_eq!(a.holders.get("g3").unwrap().0, "aa");
        assert_eq!(a.occupancy(), 2);
    }

    #[test]
    fn third_on_composite_but_first_on_one_axis_earns_a_slot() {
        // The concrete case the design names: a looped architecture wins
        // G3/G4/G5, loses G7, and under WTA earns exactly nothing.
        let a = EliteArchive::build(&[
            axis("g1", "champ", 0.90),
            axis("g7", "champ", 0.90),
            axis("g3", "looped", 0.95),
            axis("g4", "looped", 0.93),
            axis("g5", "looped", 0.91),
            axis("g7", "looped", 0.10),
        ]);
        assert_eq!(a.axes_held_by("looped"), vec!["g3", "g4", "g5"]);
        let slots = a.explore_slots(&paid(&["champ"]));
        assert_eq!(slots, vec!["looped".to_owned()]);
    }

    #[test]
    fn gate_failures_never_hold_a_frontier() {
        let mut cheat = axis("g3", "cheat", 1.0);
        cheat.gates_ok = false;
        let a = EliteArchive::build(&[cheat, axis("g3", "honest", 0.2)]);
        assert_eq!(a.holders.get("g3").unwrap().0, "honest");
    }

    #[test]
    fn non_finite_values_are_ignored() {
        let a = EliteArchive::build(&[
            axis("g6", "nan", f64::NAN),
            axis("g6", "inf", f64::INFINITY),
            axis("g6", "real", 0.4),
        ]);
        assert_eq!(a.holders.get("g6").unwrap().0, "real");
    }

    #[test]
    fn ties_break_to_lexicographically_smallest_hotkey() {
        let a = EliteArchive::build(&[axis("g1", "bb", 0.5), axis("g1", "aa", 0.5)]);
        assert_eq!(a.holders.get("g1").unwrap().0, "aa");
        // Input order must not matter.
        let b = EliteArchive::build(&[axis("g1", "aa", 0.5), axis("g1", "bb", 0.5)]);
        assert_eq!(a, b);
    }

    #[test]
    fn already_paid_hotkeys_are_excluded_from_the_pool() {
        let a = EliteArchive::build(&[axis("g1", "champ", 0.9), axis("g3", "runner", 0.8)]);
        assert!(a.explore_slots(&paid(&["champ", "runner"])).is_empty());
    }

    #[test]
    fn pool_is_capped_and_ordered_by_axes_held() {
        let mut scores = Vec::new();
        // 7 distinct holders, one axis each, plus one holding two.
        for (i, hk) in ["a", "b", "c", "d", "e", "f", "g"].iter().enumerate() {
            scores.push(axis(&format!("g{}", i + 1), hk, 0.5));
        }
        scores.push(axis("g8", "b", 0.99));
        let slots = EliteArchive::build(&scores).explore_slots(&BTreeSet::new());
        assert_eq!(slots.len(), MAX_EXPLORE_SLOTS, "capped at 5 slots");
        assert_eq!(slots[0], "b", "two frontiers ranks first");
    }

    #[test]
    fn empty_input_yields_empty_archive() {
        let a = EliteArchive::build(&[]);
        assert!(a.holders.is_empty());
        assert_eq!(a.occupancy(), 0);
        assert!(a.explore_slots(&BTreeSet::new()).is_empty());
    }
}
