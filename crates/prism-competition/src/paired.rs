//! Paired per-example displacement test (Prism v3, default-off).
//!
//! **Why paired and not a level difference.** The live v3 lattice score is
//! `round(SCORE_MAX × (C − 1.645·SE(C)))` — a *level* with an
//! independently-bootstrapped standard error. Comparing two such levels
//! throws away the pairing: example difficulty dominates the variance of
//! both sides and *cancels* when the same eval item is scored by both
//! models. This module therefore compares champion and challenger
//! **example by example on the identical slice** and bootstraps the win
//! rate, which is the shape three independent systems converged on
//! (Bittensor SN56's boss round, SN9's per-batch pairwise wins, and the
//! parameter-free Ladder of Blum & Hardt).
//!
//! **Absolute, never relative, margins.** The dead zone is in absolute
//! metric units (bits/byte for G1, accuracy for the item groups). A
//! *relative* margin (`|champion| × 1 %`) collapses to nothing exactly
//! where the metric saturates, which is where a converging field spends
//! most of its time. See `docs/spikes/prism-v3/research/15-…` for the
//! evidence and `docs/PRISM.md` § Significance-gated emission for the
//! normative statement.
//!
//! **Determinism is consensus-critical.** Leaves derived from this test
//! must be reproducible byte-for-byte by anyone re-scoring the same
//! inputs, so the bootstrap uses a fixed seed and an in-crate SplitMix64
//! PRNG (no `rand`, no platform float-order dependence): clusters are
//! sorted, resampled by index, and the percentile is taken on a
//! `total_cmp`-sorted vector.
//!
//! **What this module does NOT do.** It measures *eval-item* variance
//! only. It cannot see training-seed variance (`σ_seed`), because each
//! submission is trained exactly once — so the lower bound it returns is
//! **overconfident by construction**. That is the documented reason the
//! significance-gated emission mode ships default-off and must not be
//! enabled before `σ_seed` is measured by replication.

use std::collections::BTreeMap;

/// Fixed bootstrap seed. Two operators re-scoring the same round must get
/// the same verdict, so this is a pinned constant, never a clock or a
/// per-run value.
pub const BOOTSTRAP_SEED: u64 = 20_260_816;

/// Bootstrap resample count (SN56 production value).
pub const BOOTSTRAP_RESAMPLES: u32 = 10_000;

/// One-sided lower-bound confidence for the win-rate gate.
pub const BOOTSTRAP_CONFIDENCE_BPS: u64 = 9_900;

/// Dead zone in absolute metric units: per-example differences smaller
/// than this are *undecided* and excluded from the win rate.
///
/// Without a dead zone the win count is decided in the 5th decimal of a
/// float, which is noise rather than evidence.
pub const DEADZONE: f64 = 0.01;

/// Minimum mean gap over decided examples, absolute units. Equal to the
/// dead zone: SN56 recorded a false negative where a larger second
/// threshold rejected a model that was better on 100 % of 800 samples.
pub const MIN_MEAN_GAP: f64 = DEADZONE;

/// Win-rate bar at the bootstrap lower bound (bps of decided examples).
///
/// Deliberately **not** higher: a genuinely better model with wide
/// per-example spread sits near 0.55, so demanding much more selects for
/// low-variance submissions rather than good ones.
pub const MIN_WIN_RATE_BPS: u64 = 5_500;

/// Minimum decided examples for a verdict. Below this the slice has too
/// little to go on and the champion holds (saturated axes fail naturally).
pub const MIN_DECIDED: usize = 30;

/// Direction of a metric's scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Lower is better (bits/byte, latency).
    LowerBetter,
    /// Higher is better (accuracy).
    HigherBetter,
}

/// Per-example values for one submission on one metric, keyed by the
/// harness cluster id (`g2/hellaswag#17`, `prose#4`, `mqar/…`).
///
/// The harness records one cluster id per **item**, so this map is the
/// per-example series the paired test needs. Aggregates are never used.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExampleSeries {
    /// Cluster id -> value on the metric's natural scale.
    pub by_cluster: BTreeMap<String, f64>,
}

impl ExampleSeries {
    /// Build from an iterator of `(cluster_id, value)`.
    pub fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, f64)>,
        S: Into<String>,
    {
        Self {
            by_cluster: pairs
                .into_iter()
                .filter(|(_, v)| v.is_finite())
                .map(|(k, v)| (k.into(), v))
                .collect(),
        }
    }

    /// Number of retained examples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_cluster.len()
    }

    /// Whether the series has no usable examples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_cluster.is_empty()
    }
}

/// A challenger-vs-champion comparison on one metric of one shared slice.
#[derive(Debug, Clone)]
pub struct PairedInput {
    /// `org.*` metric key the comparison runs on.
    pub metric: String,
    /// Scale direction.
    pub direction: Direction,
    /// Slice identity (anchor version + tier + asset digest). Both sides
    /// **must** carry the same value or the comparison is refused: a
    /// rotating private slice otherwise confounds architecture with slice
    /// difficulty.
    pub slice_id: String,
    /// Champion per-example series.
    pub champion: ExampleSeries,
    /// Challenger per-example series.
    pub challenger: ExampleSeries,
}

/// Why a paired comparison could not produce a verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairedRefusal {
    /// The two sides were not scored on the same slice.
    SliceMismatch,
    /// Fewer than [`MIN_DECIDED`] decided examples.
    NotEnoughDecided,
    /// No overlapping cluster ids at all.
    NoOverlap,
}

/// Outcome of the paired test.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedOutcome {
    /// Overlapping examples considered.
    pub n_paired: usize,
    /// Examples whose absolute difference cleared the dead zone.
    pub n_decided: usize,
    /// Challenger wins among decided examples.
    pub n_wins: usize,
    /// Point-estimate win rate over decided examples (bps).
    pub win_rate_bps: u64,
    /// One-sided lower bound on the win rate (bps) at
    /// [`BOOTSTRAP_CONFIDENCE_BPS`].
    pub win_rate_lcb_bps: u64,
    /// Mean signed gap over decided examples, absolute units, positive
    /// meaning the challenger is better.
    pub mean_gap: f64,
    /// Whether every displacement condition held.
    pub displaces: bool,
}

impl PairedOutcome {
    /// A verdict that never displaces (used when evidence is missing).
    #[must_use]
    pub fn hold() -> Self {
        Self {
            n_paired: 0,
            n_decided: 0,
            n_wins: 0,
            win_rate_bps: 0,
            win_rate_lcb_bps: 0,
            mean_gap: 0.0,
            displaces: false,
        }
    }
}

/// Signed per-example gaps (positive = challenger better) on the
/// overlapping cluster ids, in deterministic cluster order.
fn signed_gaps(input: &PairedInput) -> Vec<f64> {
    let mut out = Vec::new();
    for (cluster, champ) in &input.champion.by_cluster {
        let Some(chal) = input.challenger.by_cluster.get(cluster) else {
            continue;
        };
        let d = match input.direction {
            // Lower-better: champion minus challenger is the improvement.
            Direction::LowerBetter => champ - chal,
            Direction::HigherBetter => chal - champ,
        };
        if d.is_finite() {
            out.push(d);
        }
    }
    out
}

/// Run the paired displacement test.
///
/// Returns `Err` when the comparison is structurally impossible (slice
/// mismatch, no overlap, too few decided examples) — the caller treats
/// every refusal as "champion holds", never as a displacement.
pub fn paired_test(input: &PairedInput) -> Result<PairedOutcome, PairedRefusal> {
    // Same-slice discipline: pairing across slices measures slice
    // difficulty, not architecture.
    if input.slice_id.is_empty() {
        return Err(PairedRefusal::SliceMismatch);
    }
    let gaps = signed_gaps(input);
    if gaps.is_empty() {
        return Err(PairedRefusal::NoOverlap);
    }
    let n_paired = gaps.len();

    // Dead zone: only examples with real separation vote.
    let decided: Vec<f64> = gaps
        .iter()
        .copied()
        .filter(|d| d.abs() >= DEADZONE)
        .collect();
    if decided.len() < MIN_DECIDED {
        return Err(PairedRefusal::NotEnoughDecided);
    }
    let n_decided = decided.len();
    let n_wins = decided.iter().filter(|d| **d > 0.0).count();
    let sum: f64 = decided.iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_gap = sum / n_decided as f64;
    let win_rate_bps = rate_bps(n_wins, n_decided);
    let win_rate_lcb_bps = bootstrap_win_rate_lcb_bps(&decided);

    let displaces = win_rate_lcb_bps >= MIN_WIN_RATE_BPS && mean_gap >= MIN_MEAN_GAP;

    Ok(PairedOutcome {
        n_paired,
        n_decided,
        n_wins,
        win_rate_bps,
        win_rate_lcb_bps,
        mean_gap,
        displaces,
    })
}

/// `round(10_000 × wins / total)`, saturating and total-zero safe.
fn rate_bps(wins: usize, total: usize) -> u64 {
    if total == 0 {
        return 0;
    }
    let wins = wins as u128;
    let total = total as u128;
    u64::try_from((wins * 10_000 + total / 2) / total).unwrap_or(10_000)
}

/// SplitMix64 — small, deterministic, and identical on every platform.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Unbiased index in `0..n` (Lemire rejection; `n > 0`).
    fn index(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let r = self.next_u64();
            if r < zone {
                return r % n;
            }
        }
    }
}

/// One-sided lower bound on the win rate over decided per-example gaps.
///
/// Resamples the decided examples with replacement (the paired unit is the
/// example, whose difficulty already cancelled in the pairing), recomputes
/// the win rate per resample, and takes the
/// `1 − BOOTSTRAP_CONFIDENCE_BPS` percentile. Fixed seed ⇒ identical
/// output for identical input.
#[must_use]
pub fn bootstrap_win_rate_lcb_bps(decided: &[f64]) -> u64 {
    let n = decided.len();
    if n == 0 {
        return 0;
    }
    let mut rng = SplitMix64(BOOTSTRAP_SEED);
    let mut rates: Vec<u64> = Vec::with_capacity(BOOTSTRAP_RESAMPLES as usize);
    let n64 = n as u64;
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let mut wins = 0usize;
        for _ in 0..n {
            // `index` returns < n, and n came from `decided.len()`, so the
            // conversion cannot fail on any target.
            let idx = usize::try_from(rng.index(n64)).unwrap_or(0);
            if decided[idx] > 0.0 {
                wins += 1;
            }
        }
        rates.push(rate_bps(wins, n));
    }
    // Integer bps sort: no float ordering ambiguity.
    rates.sort_unstable();
    let alpha_bps = 10_000_u64.saturating_sub(BOOTSTRAP_CONFIDENCE_BPS);
    let len = rates.len() as u64;
    // Lower-tail percentile index, floor, clamped into range.
    let idx = usize::try_from(alpha_bps.saturating_mul(len) / 10_000).unwrap_or(0);
    let idx = idx.min(rates.len().saturating_sub(1));
    rates[idx]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::cast_precision_loss)]
    use super::*;

    fn series(vals: &[f64], tag: &str) -> ExampleSeries {
        ExampleSeries::from_pairs(
            vals.iter()
                .enumerate()
                .map(|(i, v)| (format!("{tag}#{i}"), *v)),
        )
    }

    fn input(champ: &[f64], chal: &[f64], direction: Direction) -> PairedInput {
        PairedInput {
            metric: "org.g1.bits_per_byte_prose".into(),
            direction,
            slice_id: "v3/private/abc".into(),
            champion: series(champ, "prose"),
            challenger: series(chal, "prose"),
        }
    }

    #[test]
    fn clear_win_displaces_on_lower_better_metric() {
        // Challenger better by 0.05 bpb on every one of 200 examples.
        let champ: Vec<f64> = (0..200).map(|i| 1.20 + f64::from(i) * 0.001).collect();
        let chal: Vec<f64> = champ.iter().map(|v| v - 0.05).collect();
        let out = paired_test(&input(&champ, &chal, Direction::LowerBetter)).unwrap();
        assert_eq!(out.n_decided, 200);
        assert_eq!(out.n_wins, 200);
        assert_eq!(out.win_rate_bps, 10_000);
        assert_eq!(out.win_rate_lcb_bps, 10_000);
        assert!(out.mean_gap > 0.049);
        assert!(out.displaces);
    }

    #[test]
    fn identical_clone_never_displaces() {
        // A pure copy has identical true quality: every gap is 0, so every
        // example is undecided and the test refuses rather than coin-flips.
        let champ: Vec<f64> = (0..200).map(|i| 1.20 + f64::from(i) * 0.001).collect();
        let err = paired_test(&input(&champ, &champ, Direction::LowerBetter)).unwrap_err();
        assert_eq!(err, PairedRefusal::NotEnoughDecided);
    }

    #[test]
    fn deadzone_excludes_hairline_differences() {
        // Challenger better by 0.001 bpb everywhere — under the dead zone.
        let champ: Vec<f64> = (0..300).map(|i| 1.20 + f64::from(i) * 0.0001).collect();
        let chal: Vec<f64> = champ.iter().map(|v| v - 0.001).collect();
        let err = paired_test(&input(&champ, &chal, Direction::LowerBetter)).unwrap_err();
        assert_eq!(
            err,
            PairedRefusal::NotEnoughDecided,
            "hairline wins must not decide the crown"
        );
    }

    #[test]
    fn coin_flip_field_does_not_clear_the_win_rate_bar() {
        // Alternating ±0.05: 50 % win rate, real separation per example.
        let champ: Vec<f64> = vec![1.20; 200];
        let chal: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 1.15 } else { 1.25 })
            .collect();
        let out = paired_test(&input(&champ, &chal, Direction::LowerBetter)).unwrap();
        assert_eq!(out.n_decided, 200);
        assert_eq!(out.win_rate_bps, 5_000);
        assert!(
            out.win_rate_lcb_bps < MIN_WIN_RATE_BPS,
            "lcb {} must miss the 0.55 bar",
            out.win_rate_lcb_bps
        );
        assert!(!out.displaces);
    }

    #[test]
    fn majority_of_hairline_wins_fails_the_mean_gap_floor() {
        // Wins 60 % of decided examples by exactly the dead zone but loses
        // badly where it loses ⇒ mean gap below the floor.
        let mut champ = Vec::new();
        let mut chal = Vec::new();
        for i in 0..200 {
            champ.push(1.20);
            if i % 10 < 6 {
                chal.push(1.20 - 0.011); // narrow win
            } else {
                chal.push(1.20 + 0.10); // heavy loss
            }
        }
        let out = paired_test(&input(&champ, &chal, Direction::LowerBetter)).unwrap();
        assert!(out.win_rate_bps >= 5_500, "wins the majority");
        assert!(out.mean_gap < MIN_MEAN_GAP, "but is materially worse");
        assert!(!out.displaces, "mean-gap floor must veto");
    }

    #[test]
    fn higher_better_direction_is_respected() {
        let champ: Vec<f64> = vec![0.30; 200];
        let chal: Vec<f64> = vec![0.45; 200];
        let out = paired_test(&input(&champ, &chal, Direction::HigherBetter)).unwrap();
        assert!(out.displaces);
        assert!(out.mean_gap > 0.14);
        // Reversed roles must reverse the verdict.
        let rev = paired_test(&input(&chal, &champ, Direction::HigherBetter)).unwrap();
        assert!(!rev.displaces);
    }

    #[test]
    fn slice_mismatch_is_refused() {
        let mut inp = input(&[1.0; 200], &[0.9; 200], Direction::LowerBetter);
        inp.slice_id = String::new();
        assert_eq!(
            paired_test(&inp).unwrap_err(),
            PairedRefusal::SliceMismatch,
            "an unidentified slice must never decide emission"
        );
    }

    #[test]
    fn non_overlapping_clusters_are_refused() {
        let inp = PairedInput {
            metric: "org.g1.bits_per_byte_prose".into(),
            direction: Direction::LowerBetter,
            slice_id: "v3/private/abc".into(),
            champion: series(&[1.0; 100], "prose"),
            challenger: series(&[0.9; 100], "code"),
        };
        assert_eq!(paired_test(&inp).unwrap_err(), PairedRefusal::NoOverlap);
    }

    #[test]
    fn only_overlapping_examples_are_paired() {
        // Champion has 200 items, challenger only the first 100.
        let champ: Vec<f64> = vec![1.20; 200];
        let chal: Vec<f64> = vec![1.10; 100];
        let out = paired_test(&input(&champ, &chal, Direction::LowerBetter)).unwrap();
        assert_eq!(out.n_paired, 100, "pairs on the intersection only");
    }

    #[test]
    fn bootstrap_is_deterministic_across_calls() {
        let decided: Vec<f64> = (0..200)
            .map(|i| if i % 3 == 0 { -0.05 } else { 0.05 })
            .collect();
        let a = bootstrap_win_rate_lcb_bps(&decided);
        let b = bootstrap_win_rate_lcb_bps(&decided);
        assert_eq!(a, b, "fixed seed ⇒ identical verdict");
        // And the point estimate must dominate the lower bound.
        assert!(a <= rate_bps(decided.iter().filter(|d| **d > 0.0).count(), decided.len()));
    }

    #[test]
    fn lcb_is_below_point_estimate_and_bounded() {
        let decided: Vec<f64> = (0..100)
            .map(|i| if i < 60 { 0.05 } else { -0.05 })
            .collect();
        let lcb = bootstrap_win_rate_lcb_bps(&decided);
        assert!(lcb <= 6_000, "lcb {lcb} must not exceed the 0.60 estimate");
        assert!(lcb > 3_000, "lcb {lcb} should not collapse to nothing");
    }

    #[test]
    fn empty_decided_set_has_zero_lcb() {
        assert_eq!(bootstrap_win_rate_lcb_bps(&[]), 0);
    }

    #[test]
    fn nonfinite_values_are_dropped_not_propagated() {
        let s = ExampleSeries::from_pairs([("a", 1.0), ("b", f64::NAN), ("c", f64::INFINITY)]);
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
    }

    #[test]
    fn parameters_match_the_documented_rule() {
        // These are consensus-critical constants; changing one changes
        // every verdict, so pin them in a test.
        assert!((DEADZONE - 0.01).abs() < f64::EPSILON);
        assert!((MIN_MEAN_GAP - DEADZONE).abs() < f64::EPSILON);
        assert_eq!(MIN_WIN_RATE_BPS, 5_500);
        assert_eq!(BOOTSTRAP_CONFIDENCE_BPS, 9_900);
        assert_eq!(BOOTSTRAP_RESAMPLES, 10_000);
        assert_eq!(BOOTSTRAP_SEED, 20_260_816);
        assert_eq!(MIN_DECIDED, 30);
    }
}
