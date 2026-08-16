//! Significance-gated emission collapse (Prism v3, **default-off**).
//!
//! ## The rule
//!
//! ```text
//! champion  60 %   held until displaced by the paired test (paired.rs);
//!                  scaled toward a 50 % floor when the win is real but
//!                  marginal, with the difference BURNED
//! band      15 / 10 / 5 %   ranks 2–4 among gate-passing entries
//! explore   10 %   ≤5 gate-passing entries advancing ≥1 per-axis frontier
//! remainder        BURNS
//! ```
//!
//! **Two distinct bars.** Clearing the paired test *transfers the crown*;
//! a strictly larger mean gap (`PREMIUM_GAP`) is required to earn *above*
//! the champion floor. Separating "who is champion" from "how much the
//! champion is paid" means a marginal-but-real win does not automatically
//! unlock the full share.
//!
//! **The statistical term never decays; only the economic floor does.**
//! The win-rate bar and the dead zone are truth conditions, so decaying
//! them would knowingly crown champions on noise. Tenure instead erodes
//! the *economic* floor, which is what stops a hoarder squatting on a
//! stale title.
//!
//! ## Why this replaces winner-take-all — by arithmetic, not by fairness
//!
//! A functional clone of the champion has *identical true quality*. Under
//! WTA on point estimates it therefore wins the whole pot with probability
//! ≈ 0.5 by symmetry of the noise: expected value ≈ **50 % of emissions
//! for the price of one pod**. That makes an evadable copy detector
//! load-bearing. Under a one-sided significance test a true-Δ-zero clone
//! passes at most at the test's false-positive rate, so the same attack is
//! worth **< 5 %** — a >10× reduction with no detector involved.
//!
//! The honest qualification: significance gating protects the **champion
//! share**, not the graded band, where a statistical tie lands high by
//! construction. SN9 measured exactly that outcome under its epsilon rule
//! ("direct copying with practically zero amendments"). The band is
//! therefore capped at 30 % and the copy gate stays necessary.
//!
//! ## Sequencing constraint — why this ships off
//!
//! The bootstrap in [`crate::paired`] measures **eval-item** variance
//! only. Training-seed variance (`σ_seed`) is absent from the model
//! because each submission is trained exactly once, and a seed change
//! alone re-ranks NAS architectures at Kendall τ = 0.48. The lower bound
//! is therefore **overconfident**, and a significance test computed on a
//! provably wrong standard error is *worse* than honest WTA: it lends
//! false statistical authority to a biased ranking. This mode must not be
//! enabled before `σ_seed` is measured by replication.

use std::collections::{BTreeMap, BTreeSet};

use prism_challenge_task::SCORE_MAX;
use prism_store::FinalScore;

use crate::frontier::EliteArchive;
use crate::paired::PairedOutcome;

/// Champion share when the premium bar is cleared (bps).
pub const CHAMPION_BPS: u64 = 6_000;

/// Champion floor when the win is real but sub-premium (bps). The
/// difference between this and [`CHAMPION_BPS`] burns.
pub const CHAMPION_FLOOR_BPS: u64 = 5_000;

/// Runner-up band shares for ranks 2/3/4 (bps).
pub const BAND_BPS: [u64; 3] = [1_500, 1_000, 500];

/// Exploration pool total (bps), split equally across the selected slots.
pub const EXPLORE_POOL_BPS: u64 = 1_000;

/// Mean gap (absolute units) above which the champion earns the premium
/// share instead of the floor.
pub const PREMIUM_GAP: f64 = 0.02;

/// Tail floor: an allocation at or below this (bps) is zeroed rather than
/// paid, so unresolvable rank differences are not paid at all.
pub const TAIL_FLOOR_BPS: u64 = 100;

/// Weight-EMA smoothing factor (bps of the new vector). `alpha = 0.5`
/// matches SN9's validator EMA; handover is smoothed over rounds so one
/// anomalous round cannot swing emission.
pub const EMA_ALPHA_BPS: u64 = 5_000;

/// Economic-floor decay per day of champion tenure (bps of the base
/// floor). ~0.15 %/day, interpolating three live implementations, and
/// **linear to a nonzero floor** — never exponential to zero, so a
/// minimum copy deterrent stays permanently in place.
pub const TENURE_DECAY_BPS_PER_DAY: u64 = 15;

/// Lower bound the tenure decay can reach (bps of the base floor).
pub const TENURE_DECAY_MIN_BPS: u64 = 8_000;

/// Economic floor multiplier after `tenure_days` (bps of the base value).
///
/// Linear decay to [`TENURE_DECAY_MIN_BPS`]: a champion that has held the
/// title a long time is progressively easier to displace on the *policy*
/// bar, while the statistical bar is untouched.
#[must_use]
pub fn tenure_multiplier_bps(tenure_days: u64) -> u64 {
    let decayed = 10_000_u64.saturating_sub(TENURE_DECAY_BPS_PER_DAY.saturating_mul(tenure_days));
    decayed.max(TENURE_DECAY_MIN_BPS)
}

/// Everything the significance-gated collapse needs beyond the credits.
///
/// All of it is derived from stored measurements; none of it is state a
/// miner can write.
#[derive(Debug, Clone, Default)]
pub struct SigContext {
    /// Current champion hotkey (hex), if the subnet has one.
    pub incumbent: Option<String>,
    /// Days the incumbent has held the title (economic-floor decay only).
    pub tenure_days: u64,
    /// Paired verdict of the leading challenger against the incumbent.
    /// `None` ⇒ no admissible paired evidence ⇒ the champion holds.
    pub challenger: Option<(String, PairedOutcome)>,
    /// Per-axis elite archive for the exploration pool.
    pub archive: EliteArchive,
    /// Previous round's emitted share vector (bps) for the weight EMA.
    pub previous_bps: BTreeMap<String, u64>,
    /// Whether the round's runs produced live contamination evidence
    /// ([`crate::contamination`]). **`false` fail-closes the protected
    /// champion share**: the mirror-gap defence is inert by construction in
    /// the `public_dev` tier, so an unchecked round's "no contamination
    /// penalty" is the absence of a measurement, not a clean result. This
    /// rule pays a protected 60 % on measured evidence, so it must not do so
    /// on a number whose contamination detector was switched off. Default
    /// `false` — silence is not evidence.
    pub contamination_checked: bool,
}

/// A resolved allocation: who is paid what share, and what burns.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmissionPlan {
    /// Hotkey -> allocated share (bps). Sums with [`Self::burn_bps`] to
    /// exactly 10 000 whenever anything was allocated.
    pub shares: BTreeMap<String, u64>,
    /// Unallocated share that burns (bps).
    pub burn_bps: u64,
    /// Champion after the displacement test, if any.
    pub champion: Option<String>,
    /// Whether the challenger displaced the incumbent this round.
    pub displaced: bool,
}

impl EmissionPlan {
    /// Total allocated share (bps).
    #[must_use]
    pub fn allocated_bps(&self) -> u64 {
        self.shares.values().copied().sum()
    }

    /// Integer conservation: allocation + burn is exactly 10 000.
    #[must_use]
    pub fn conserves(&self) -> bool {
        self.allocated_bps().saturating_add(self.burn_bps) == 10_000
    }
}

/// Rank positive credits under the emission tie convention (higher score
/// first, then lexicographically smaller hotkey).
fn ranked_positive(scores: &BTreeMap<String, FinalScore>) -> Vec<(String, u64)> {
    let mut ranked: Vec<(&str, u64)> = scores
        .iter()
        .filter_map(|(hk, s)| match s {
            FinalScore::Score(v) if *v > 0 => Some((hk.as_str(), *v)),
            _ => None,
        })
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked
        .into_iter()
        .map(|(hk, v)| (hk.to_owned(), v))
        .collect()
}

/// Resolve the champion for this round.
///
/// The incumbent holds unless the challenger cleared every paired
/// condition. With no incumbent the top-ranked gate-passing credit takes
/// the crown (cold start), which is the only path that does not require a
/// paired test — there is nothing to be paired against.
fn resolve_champion(
    ctx: &SigContext,
    ranked: &[(String, u64)],
) -> (Option<String>, bool, Option<PairedOutcome>) {
    match &ctx.incumbent {
        None => (ranked.first().map(|(hk, _)| hk.clone()), false, None),
        Some(inc) => match &ctx.challenger {
            Some((challenger, outcome)) if outcome.displaces => {
                (Some(challenger.clone()), true, Some(outcome.clone()))
            }
            Some((_, outcome)) => (Some(inc.clone()), false, Some(outcome.clone())),
            None => (Some(inc.clone()), false, None),
        },
    }
}

/// Champion share in bps, applying the premium bar and the tenure-decayed
/// economic floor. The shortfall against [`CHAMPION_BPS`] burns.
fn champion_share_bps(ctx: &SigContext, outcome: Option<&PairedOutcome>, displaced: bool) -> u64 {
    // A fresh displacement with a large gap earns the premium; a marginal
    // win, or a champion merely holding, earns the floor.
    let premium = outcome.is_some_and(|o| displaced && o.mean_gap >= PREMIUM_GAP);
    if premium {
        return CHAMPION_BPS;
    }
    let floor = CHAMPION_FLOOR_BPS.saturating_mul(tenure_multiplier_bps(ctx.tenure_days)) / 10_000;
    floor.min(CHAMPION_BPS)
}

/// Build the round's allocation plan.
///
/// Deterministic: identical inputs always produce an identical plan.
#[must_use]
pub fn plan_emission(scores: &BTreeMap<String, FinalScore>, ctx: &SigContext) -> EmissionPlan {
    let ranked = ranked_positive(scores);
    if ranked.is_empty() {
        // Nothing eligible: the whole share burns (fail-closed, matching
        // the existing all-ineligible behavior).
        return burn_everything();
    }
    // Fail-closed on contamination evidence. The mirror-gap defence is inert
    // by construction in `public_dev`, so an unchecked round cannot
    // distinguish "clean" from "not measured" — and this rule's whole
    // premise is that the protected champion share is granted on *measured*
    // evidence. Paying it on an unchecked round would be exactly the failure
    // the significance test exists to prevent, dressed in statistics. The
    // safe direction is a visible full burn, the same posture the gateway
    // takes when it has no sealed bundle: no allocation is strictly better
    // than a confidently wrong one.
    if !ctx.contamination_checked {
        return burn_everything();
    }

    let (champion, displaced, outcome) = resolve_champion(ctx, &ranked);
    let mut shares: BTreeMap<String, u64> = BTreeMap::new();
    let mut paid: BTreeSet<String> = BTreeSet::new();

    if let Some(champ) = &champion {
        // A champion must still hold a positive credit this round; a
        // champion whose score vanished cannot be paid.
        if ranked.iter().any(|(hk, _)| hk == champ) {
            let bps = champion_share_bps(ctx, outcome.as_ref(), displaced);
            shares.insert(champ.clone(), bps);
            paid.insert(champ.clone());
        }
    }

    // Band: ranks 2–4 among gate-passing credits, skipping the champion.
    let mut band_iter = BAND_BPS.iter();
    for (hk, _) in &ranked {
        if paid.contains(hk) {
            continue;
        }
        let Some(bps) = band_iter.next() else {
            break;
        };
        shares.insert(hk.clone(), *bps);
        paid.insert(hk.clone());
    }

    // Exploration pool: equal split across axis-frontier holders. Only the
    // champion is excluded — it already takes the largest share and holds
    // the top composite by construction. Band members ARE eligible: the
    // case the rule exists for is "3rd on the composite but 1st on G3",
    // which is a rank-3 band entry that produced real information.
    // Unfilled slots burn rather than concentrate.
    let champion_only: BTreeSet<String> = champion.iter().cloned().collect();
    let slots = ctx.archive.explore_slots(&champion_only);
    if !slots.is_empty() {
        let per = EXPLORE_POOL_BPS / (slots.len() as u64);
        if per > 0 {
            for hk in &slots {
                // A pool recipient must have a positive credit: the pool
                // pays measured frontier advances, not absent submissions.
                if ranked.iter().any(|(r, _)| r == hk) {
                    *shares.entry(hk.clone()).or_insert(0) += per;
                    paid.insert(hk.clone());
                }
            }
        }
    }

    // Weight EMA, then the tail floor, then conservation.
    let mut shares = apply_ema(&shares, &ctx.previous_bps);
    // Only a hotkey with a positive credit **this round** can be paid.
    //
    // Without this, the EMA's phase-out term resurrects hotkeys that are in
    // `previous_bps` but not in this round's credits. Their share would then
    // count toward `allocated_bps` — so it would not be burned — while
    // `apply_significance` maps over `scores` and emits no leaf for them, so
    // it would not be paid either. The mass would silently redistribute to
    // the champion at `BUNDLE_SPEC` §6.4 normalization. Dropping them here
    // makes the phase-out honest: an absent hotkey's decayed share burns.
    let live: BTreeSet<&str> = ranked.iter().map(|(hk, _)| hk.as_str()).collect();
    shares.retain(|hk, bps| *bps > TAIL_FLOOR_BPS && live.contains(hk.as_str()));
    let allocated: u64 = shares.values().copied().sum();
    // Over-allocation is structurally impossible (60+15+10+5+10 = 100),
    // but clamp rather than underflow if a future parameter edit breaks it.
    let allocated = allocated.min(10_000);
    let burn_bps = 10_000_u64.saturating_sub(allocated);

    EmissionPlan {
        shares,
        burn_bps,
        champion,
        displaced,
    }
}

/// A plan that allocates nothing and burns the entire share.
///
/// `burn_bps = 10_000` rather than 0 so [`EmissionPlan::conserves`] holds
/// unconditionally and the burn leaf in `prism-emit` carries the full
/// remainder — "allocated nothing" and "burned everything" are the same
/// statement and the plan should say so.
fn burn_everything() -> EmissionPlan {
    EmissionPlan {
        shares: BTreeMap::new(),
        burn_bps: 10_000,
        champion: None,
        displaced: false,
    }
}

/// Weight EMA on the emitted share vector: `α·new + (1−α)·previous`.
///
/// A *temporal* smoother that composes with the statistical test — it
/// stops a single anomalous round from swinging emission, while the paired
/// test decides whether the round means anything at all. Hotkeys present
/// only in the previous vector decay toward zero rather than vanishing.
fn apply_ema(
    fresh: &BTreeMap<String, u64>,
    previous: &BTreeMap<String, u64>,
) -> BTreeMap<String, u64> {
    if previous.is_empty() {
        return fresh.clone();
    }
    let alpha = EMA_ALPHA_BPS;
    let beta = 10_000_u64.saturating_sub(alpha);
    let mut out: BTreeMap<String, u64> = BTreeMap::new();
    let keys: BTreeSet<&str> = fresh
        .keys()
        .chain(previous.keys())
        .map(String::as_str)
        .collect();
    for hk in keys {
        let new = fresh.get(hk).copied().unwrap_or(0);
        let old = previous.get(hk).copied().unwrap_or(0);
        // Integer EMA with round-half-up; both terms are ≤10 000 so the
        // product cannot overflow u64.
        let blended = (new.saturating_mul(alpha) + old.saturating_mul(beta) + 5_000) / 10_000;
        if blended > 0 {
            out.insert(hk.to_owned(), blended);
        }
    }
    out
}

/// Project a plan onto lattice leaf values.
///
/// Leaves are share-proportional (`bps × SCORE_MAX / 10 000`) because the
/// aggregator normalizes each challenge's positive leaves to sum to 1 —
/// so the *ratios* are what reach the chain. A positive share never rounds
/// to 0 (same guarantee as the `top3` path): a hotkey the rule intends to
/// pay is never silently dropped.
#[must_use]
pub fn apply_significance(
    scores: BTreeMap<String, FinalScore>,
    ctx: &SigContext,
) -> BTreeMap<String, FinalScore> {
    let plan = plan_emission(&scores, ctx);
    scores
        .into_iter()
        .map(|(hk, s)| match &s {
            FinalScore::Score(v) if *v > 0 => {
                let bps = plan.shares.get(&hk).copied().unwrap_or(0);
                (hk, FinalScore::Score(bps_to_lattice(bps)))
            }
            // `Score(0)` and `NoScore` rows pass through untouched.
            _ => (hk, s),
        })
        .collect()
}

/// `bps` share -> lattice value, never rounding a positive share to 0.
#[must_use]
pub fn bps_to_lattice(bps: u64) -> u64 {
    if bps == 0 {
        return 0;
    }
    (bps.saturating_mul(SCORE_MAX) / 10_000).max(1)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::frontier::AxisScore;

    fn credits(pairs: &[(&str, u64)]) -> BTreeMap<String, FinalScore> {
        pairs
            .iter()
            .map(|(hk, v)| ((*hk).to_owned(), FinalScore::Score(*v)))
            .collect()
    }

    /// A context with contamination evidence present — the precondition for
    /// any allocation at all. `SigContext::default()` deliberately has it
    /// `false` (silence is not evidence), so tests about the *allocation*
    /// rule start from here and the fail-closed path is tested on its own.
    fn checked() -> SigContext {
        SigContext {
            contamination_checked: true,
            ..SigContext::default()
        }
    }

    fn win(mean_gap: f64) -> PairedOutcome {
        PairedOutcome {
            n_paired: 200,
            n_decided: 180,
            n_wins: 120,
            win_rate_bps: 6_667,
            win_rate_lcb_bps: 6_000,
            mean_gap,
            displaces: true,
        }
    }

    fn loss() -> PairedOutcome {
        PairedOutcome {
            n_paired: 200,
            n_decided: 180,
            n_wins: 90,
            win_rate_bps: 5_000,
            win_rate_lcb_bps: 4_200,
            mean_gap: 0.001,
            displaces: false,
        }
    }

    fn archive(entries: &[(&str, &str, f64)]) -> EliteArchive {
        EliteArchive::build(
            &entries
                .iter()
                .map(|(axis, hk, v)| AxisScore {
                    axis: (*axis).to_owned(),
                    hotkey: (*hk).to_owned(),
                    value: *v,
                    gates_ok: true,
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn cold_start_crowns_the_top_credit() {
        let plan = plan_emission(&credits(&[("aa", 900), ("bb", 500)]), &checked());
        assert_eq!(plan.champion.as_deref(), Some("aa"));
        assert!(!plan.displaced);
        // No premium evidence on a cold start ⇒ floor, remainder burns.
        assert_eq!(plan.shares.get("aa"), Some(&CHAMPION_FLOOR_BPS));
        assert_eq!(plan.shares.get("bb"), Some(&BAND_BPS[0]));
        assert!(plan.conserves());
    }

    #[test]
    fn incumbent_holds_when_the_challenger_fails_the_test() {
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            challenger: Some(("chal".into(), loss())),
            ..checked()
        };
        // Challenger has the HIGHER raw credit but did not clear the bar.
        let plan = plan_emission(&credits(&[("champ", 500), ("chal", 900)]), &ctx);
        assert_eq!(plan.champion.as_deref(), Some("champ"));
        assert!(!plan.displaced);
        assert_eq!(plan.shares.get("champ"), Some(&CHAMPION_FLOOR_BPS));
        // The challenger is still paid as the leading band entry.
        assert_eq!(plan.shares.get("chal"), Some(&BAND_BPS[0]));
    }

    #[test]
    fn clone_with_identical_quality_cannot_take_the_crown() {
        // The load-bearing property: a copy's paired outcome never
        // displaces, so its EV collapses from ~50 % of the pot to a band
        // slot — without consulting any copy detector.
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            challenger: Some(("clone".into(), PairedOutcome::hold())),
            ..checked()
        };
        let plan = plan_emission(&credits(&[("champ", 900_000), ("clone", 900_001)]), &ctx);
        assert_eq!(plan.champion.as_deref(), Some("champ"));
        assert!(plan.shares.get("clone").unwrap() <= &BAND_BPS[0]);
    }

    #[test]
    fn significant_challenger_with_premium_gap_takes_the_full_share() {
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            challenger: Some(("chal".into(), win(0.05))),
            ..checked()
        };
        let plan = plan_emission(&credits(&[("champ", 800), ("chal", 900)]), &ctx);
        assert_eq!(plan.champion.as_deref(), Some("chal"));
        assert!(plan.displaced);
        assert_eq!(plan.shares.get("chal"), Some(&CHAMPION_BPS));
        assert_eq!(plan.shares.get("champ"), Some(&BAND_BPS[0]));
        assert!(plan.conserves());
    }

    #[test]
    fn marginal_win_takes_the_crown_but_not_the_premium() {
        // Two bars: displacement is cleared, premium is not, so the
        // difference burns instead of paying for a hairline gain.
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            challenger: Some(("chal".into(), win(0.011))),
            ..checked()
        };
        let plan = plan_emission(&credits(&[("champ", 800), ("chal", 900)]), &ctx);
        assert!(plan.displaced);
        assert_eq!(plan.shares.get("chal"), Some(&CHAMPION_FLOOR_BPS));
        assert!(
            plan.burn_bps >= CHAMPION_BPS - CHAMPION_FLOOR_BPS,
            "sub-premium remainder must burn, got {}",
            plan.burn_bps
        );
    }

    #[test]
    fn tenure_decays_the_economic_floor_only() {
        assert_eq!(tenure_multiplier_bps(0), 10_000);
        assert_eq!(tenure_multiplier_bps(10), 9_850);
        // Linear to a nonzero floor, never to zero.
        assert_eq!(tenure_multiplier_bps(10_000), TENURE_DECAY_MIN_BPS);
        let fresh = SigContext {
            incumbent: Some("champ".into()),
            tenure_days: 0,
            ..checked()
        };
        let stale = SigContext {
            incumbent: Some("champ".into()),
            tenure_days: 200,
            ..checked()
        };
        let a = plan_emission(&credits(&[("champ", 900)]), &fresh);
        let b = plan_emission(&credits(&[("champ", 900)]), &stale);
        assert!(
            b.shares.get("champ") < a.shares.get("champ"),
            "a long tenure must be cheaper to displace"
        );
        // The statistical bar is untouched by tenure — asserted in
        // `paired.rs`, which has no tenure input at all.
    }

    #[test]
    fn explore_pool_pays_axis_frontier_holders_equally() {
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            archive: archive(&[("g3", "looped", 0.95), ("g7", "sparse", 0.90)]),
            ..checked()
        };
        let plan = plan_emission(
            &credits(&[("champ", 900), ("looped", 300), ("sparse", 200)]),
            &ctx,
        );
        // Both hold a frontier; each takes half the pool on top of any
        // band share they earned.
        let per = EXPLORE_POOL_BPS / 2;
        assert_eq!(plan.shares.get("looped"), Some(&(BAND_BPS[0] + per)));
        assert_eq!(plan.shares.get("sparse"), Some(&(BAND_BPS[1] + per)));
        assert!(plan.conserves());
    }

    #[test]
    fn explore_pool_ignores_holders_without_a_positive_credit() {
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            archive: archive(&[("g3", "ghost", 0.99)]),
            ..checked()
        };
        let plan = plan_emission(&credits(&[("champ", 900)]), &ctx);
        assert!(!plan.shares.contains_key("ghost"));
        assert!(plan.conserves());
    }

    #[test]
    fn unallocated_share_burns_rather_than_concentrating() {
        // Single entrant: champion floor only, everything else burns.
        let plan = plan_emission(&credits(&[("solo", 900)]), &checked());
        assert_eq!(plan.shares.get("solo"), Some(&CHAMPION_FLOOR_BPS));
        assert_eq!(plan.burn_bps, 10_000 - CHAMPION_FLOOR_BPS);
        assert!(plan.conserves());
    }

    #[test]
    fn integer_conservation_holds_across_field_sizes() {
        for n in 1..12_usize {
            let rows: Vec<(String, u64)> = (0..n)
                .map(|i| (format!("hk{i:02}"), 1_000 - (i as u64)))
                .collect();
            let credits: BTreeMap<String, FinalScore> = rows
                .iter()
                .map(|(hk, v)| (hk.clone(), FinalScore::Score(*v)))
                .collect();
            let plan = plan_emission(&credits, &checked());
            assert!(plan.conserves(), "n={n} must conserve: {plan:?}");
            assert!(plan.allocated_bps() <= 10_000, "n={n} over-allocated");
        }
    }

    #[test]
    fn an_unchecked_contamination_round_pays_nobody() {
        // Fail-closed: the mirror defence is inert by construction in
        // `public_dev`, so an unchecked round cannot tell "clean" from "not
        // measured". A protected 60 % share must not be granted on it.
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            challenger: Some(("chal".into(), win(0.05))),
            contamination_checked: false,
            ..SigContext::default()
        };
        let plan = plan_emission(&credits(&[("champ", 900), ("chal", 950)]), &ctx);
        assert!(plan.shares.is_empty(), "nothing may be paid: {plan:?}");
        assert_eq!(plan.burn_bps, 10_000, "the whole share burns");
        assert_eq!(plan.champion, None);
        assert!(!plan.displaced);
        assert!(plan.conserves());
        // The identical round WITH evidence pays normally — so the burn is
        // attributable to the missing check, not to anything else.
        let with_evidence = SigContext {
            contamination_checked: true,
            ..ctx
        };
        let ok = plan_emission(&credits(&[("champ", 900), ("chal", 950)]), &with_evidence);
        assert_eq!(ok.shares.get("chal"), Some(&CHAMPION_BPS));
    }

    #[test]
    fn default_context_is_fail_closed() {
        // Silence is not evidence: the zero value of the context must not
        // authorize payment.
        assert!(!SigContext::default().contamination_checked);
        let plan = plan_emission(&credits(&[("aa", 900)]), &SigContext::default());
        assert_eq!(plan.burn_bps, 10_000);
        assert!(plan.shares.is_empty());
    }

    #[test]
    fn an_empty_field_burns_the_whole_share_not_zero() {
        // `burn_bps` must be the full share, not 0: "allocated nothing" and
        // "burned everything" are the same statement, and `prism-emit`'s
        // burn leaf reads `burn_bps` directly.
        for c in [BTreeMap::new(), credits(&[("aa", 0), ("bb", 0)])] {
            let plan = plan_emission(&c, &checked());
            assert!(plan.shares.is_empty());
            assert_eq!(plan.burn_bps, 10_000, "empty field must burn everything");
            assert!(plan.conserves());
        }
    }

    #[test]
    fn ema_ghosts_never_consume_allocation_they_cannot_be_paid() {
        // Regression: a hotkey in `previous_bps` but absent from this
        // round's credits used to keep a decayed share. That share counted
        // as allocated (so it was not burned) while `apply_significance`
        // emitted no leaf for it (so it was not paid) — the mass silently
        // redistributed to the champion at BUNDLE_SPEC §6.4 normalization.
        let previous: BTreeMap<String, u64> =
            [("ghost".to_owned(), 6_000_u64)].into_iter().collect();
        let ctx = SigContext {
            previous_bps: previous,
            ..checked()
        };
        let plan = plan_emission(&credits(&[("aa", 900)]), &ctx);
        assert!(
            !plan.shares.contains_key("ghost"),
            "a hotkey with no credit this round must not be allocated: {plan:?}"
        );
        // Every allocated hotkey must be one `apply_significance` can emit.
        let out = apply_significance(credits(&[("aa", 900)]), &ctx);
        for hk in plan.shares.keys() {
            assert!(
                matches!(out.get(hk), Some(FinalScore::Score(v)) if *v > 0),
                "allocated {hk} has no positive leaf"
            );
        }
        assert!(plan.conserves());
    }

    #[test]
    fn every_allocated_share_reaches_a_leaf() {
        // The general form of the property above, across field shapes and
        // a populated previous vector.
        let previous: BTreeMap<String, u64> =
            [("old1".to_owned(), 3_000_u64), ("old2".to_owned(), 200)]
                .into_iter()
                .collect();
        for n in 1..8_usize {
            let rows: Vec<(String, u64)> = (0..n)
                .map(|i| (format!("hk{i:02}"), 1_000 - (i as u64)))
                .collect();
            let c: BTreeMap<String, FinalScore> = rows
                .iter()
                .map(|(hk, v)| (hk.clone(), FinalScore::Score(*v)))
                .collect();
            let ctx = SigContext {
                previous_bps: previous.clone(),
                archive: archive(&[("g3", "hk01", 0.9)]),
                ..checked()
            };
            let plan = plan_emission(&c, &ctx);
            let out = apply_significance(c.clone(), &ctx);
            assert!(plan.conserves(), "n={n}");
            for hk in plan.shares.keys() {
                assert!(
                    matches!(out.get(hk), Some(FinalScore::Score(v)) if *v > 0),
                    "n={n}: allocated {hk} has no positive leaf"
                );
            }
        }
    }

    #[test]
    fn tail_floor_zeroes_unresolvable_shares() {
        // A 5-way explore split is 200 bps each, above the floor; force a
        // sub-floor share via the EMA against an empty fresh allocation.
        let previous: BTreeMap<String, u64> = [("ghost".to_owned(), 100_u64)].into_iter().collect();
        let ctx = SigContext {
            previous_bps: previous,
            ..checked()
        };
        let plan = plan_emission(&credits(&[("aa", 900)]), &ctx);
        assert!(
            !plan.shares.contains_key("ghost"),
            "a decayed tail must be zeroed, not paid: {plan:?}"
        );
        assert!(plan.conserves());
    }

    #[test]
    fn ema_smooths_handover_between_rounds() {
        let previous: BTreeMap<String, u64> = [("old".to_owned(), 6_000_u64)].into_iter().collect();
        let ctx = SigContext {
            incumbent: Some("new".into()),
            previous_bps: previous,
            ..checked()
        };
        let plan = plan_emission(&credits(&[("new", 900), ("old", 100)]), &ctx);
        let new_share = *plan.shares.get("new").unwrap();
        let old_share = *plan.shares.get("old").unwrap();
        // Neither jumps straight to its target: the outgoing champion
        // phases out and the incoming one phases in.
        assert!(new_share < CHAMPION_FLOOR_BPS, "incoming phases in");
        assert!(old_share > BAND_BPS[0], "outgoing phases out");
        assert!(plan.conserves());
    }

    #[test]
    fn empty_and_all_ineligible_fields_allocate_nothing() {
        let plan = plan_emission(&BTreeMap::new(), &checked());
        assert!(plan.shares.is_empty());
        assert_eq!(plan.champion, None);
        let zeros = credits(&[("aa", 0), ("bb", 0)]);
        let plan = plan_emission(&zeros, &checked());
        assert!(plan.shares.is_empty());
    }

    #[test]
    fn plan_is_deterministic() {
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            challenger: Some(("chal".into(), win(0.05))),
            archive: archive(&[("g3", "x", 0.9), ("g7", "y", 0.8)]),
            tenure_days: 7,
            ..checked()
        };
        let c = credits(&[("champ", 900), ("chal", 950), ("x", 400), ("y", 300)]);
        let a = plan_emission(&c, &ctx);
        for _ in 0..8 {
            assert_eq!(
                plan_emission(&c, &ctx),
                a,
                "identical inputs ⇒ identical plan"
            );
        }
    }

    #[test]
    fn lattice_projection_never_rounds_a_paid_share_to_zero() {
        assert_eq!(bps_to_lattice(0), 0);
        assert_eq!(bps_to_lattice(6_000), 600_000);
        assert_eq!(bps_to_lattice(1_500), 150_000);
        // Even an absurdly small share keeps a positive leaf.
        assert_eq!(bps_to_lattice(1), 100);
        for bps in 1..10_000_u64 {
            assert!(bps_to_lattice(bps) > 0, "bps {bps} rounded to zero");
        }
    }

    #[test]
    fn apply_significance_preserves_absence_and_zero_rows() {
        let mut c = credits(&[("aa", 900), ("bb", 500)]);
        c.insert("cc".into(), FinalScore::NoScore(6));
        c.insert("dd".into(), FinalScore::Score(0));
        let out = apply_significance(c, &checked());
        assert_eq!(out.get("cc"), Some(&FinalScore::NoScore(6)));
        assert_eq!(out.get("dd"), Some(&FinalScore::Score(0)));
        assert_eq!(
            out.get("aa"),
            Some(&FinalScore::Score(bps_to_lattice(CHAMPION_FLOOR_BPS)))
        );
    }

    #[test]
    fn share_ratios_reach_the_chain_as_intended() {
        // The aggregator normalizes positive leaves within the challenge,
        // so the ratio champion:band must match the rule.
        let ctx = SigContext {
            incumbent: Some("champ".into()),
            challenger: Some(("chal".into(), win(0.05))),
            ..checked()
        };
        let out = apply_significance(credits(&[("champ", 800), ("chal", 900)]), &ctx);
        let champ = match out.get("chal") {
            Some(FinalScore::Score(v)) => *v,
            other => panic!("expected score, got {other:?}"),
        };
        let band = match out.get("champ") {
            Some(FinalScore::Score(v)) => *v,
            other => panic!("expected score, got {other:?}"),
        };
        assert_eq!(champ, bps_to_lattice(CHAMPION_BPS));
        assert_eq!(band, bps_to_lattice(BAND_BPS[0]));
        assert_eq!(champ / band, 4, "60 % : 15 % = 4:1");
    }

    #[test]
    fn parameters_match_the_documented_rule() {
        assert_eq!(CHAMPION_BPS, 6_000);
        assert_eq!(CHAMPION_FLOOR_BPS, 5_000);
        assert_eq!(BAND_BPS, [1_500, 1_000, 500]);
        assert_eq!(EXPLORE_POOL_BPS, 1_000);
        // 60 + 15 + 10 + 5 + 10 = 100 % of the share.
        let total: u64 = CHAMPION_BPS + BAND_BPS.iter().sum::<u64>() + EXPLORE_POOL_BPS;
        assert_eq!(total, 10_000);
    }
}
