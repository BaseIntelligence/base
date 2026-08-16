//! Unannounced champion re-runs ("prove it again") — v3, default-off.
//!
//! **What it defends.** `prism-emit` carries a champion's positive score
//! forward until superseded, and by the winner's curse that carried number
//! is an *optimistic* draw. So an incumbent is defended by an inflated
//! figure that becomes harder to displace over time, and any overfit to the
//! public anchors is never re-tested after it has been paid. Re-measuring
//! the champion on a freshly generated private slice is a direct, causal
//! test of exactly that failure mode — the only defence in the design that
//! can catch anchor-overfit *after* it has already been rewarded.
//!
//! **Eval-only.** The champion's checkpoint is already parked with a
//! verified receipt, so this is inference, not retraining (~1.2 h rather
//! than 6 h+). The operator funds it: a champion has no incentive to fund
//! its own audit, and a trustworthy ranking is a public good for the subnet.
//!
//! **Why the timing must be unannounced.** IOTA states the reason
//! plainly — miners must not know when they are being monitored, or they
//! behave correctly only during observed intervals. A re-run every round on
//! a published schedule is a known quantity to design against; one that
//! fires unpredictably is not, at the same expected cost.
//!
//! **Unpredictable, yet verifiable after the fact.** The decision is a
//! keyed hash of `(epoch, slice_id, champion)`. A miner cannot predict it
//! without the slice id, which is not published until the round closes;
//! once it is published, anyone can recompute the same verdict and confirm
//! the operator neither skipped nor fabricated an audit. That is the same
//! "public rule, private slice contents" split the rest of the design uses.

/// Default audit probability (bps). ~1 round in 2, so the expected cost is
/// half a re-run per round while the *timing* stays unpredictable.
pub const AUDIT_PROBABILITY_BPS: u64 = 5_000;

/// Domain separator, so the audit draw cannot collide with any other
/// derivation that hashes the same tuple.
const AUDIT_DOMAIN: &[u8] = b"base-prism-champion-audit-v1";

/// Whether the champion is audited this round.
///
/// Deterministic in its inputs and uniform in `0..10_000`. `slice_id` must
/// be the round's private-slice identity (unpublished until the round
/// closes), which is what makes the draw unpredictable in advance and
/// checkable afterwards.
#[must_use]
pub fn audit_due(epoch: u64, slice_id: &str, champion: &str, probability_bps: u64) -> bool {
    if probability_bps == 0 {
        return false;
    }
    if probability_bps >= 10_000 {
        return true;
    }
    audit_draw_bps(epoch, slice_id, champion) < probability_bps
}

/// The audit draw in `0..10_000` (FNV-1a over the domain-separated tuple —
/// no external dependency, identical on every platform).
#[must_use]
pub fn audit_draw_bps(epoch: u64, slice_id: &str, champion: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    eat(AUDIT_DOMAIN);
    eat(&epoch.to_be_bytes());
    eat(slice_id.as_bytes());
    eat(champion.as_bytes());
    // Fold the high bits down so the low-order modulo is well mixed.
    (h ^ (h >> 32)) % 10_000
}

/// Verdict of a completed champion re-run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerunVerdict {
    /// Re-measured within tolerance: tenure continues.
    Holds,
    /// Re-measured materially worse on the fresh slice: the champion loses
    /// tenure and the next significant submission may be promoted.
    Regressed,
}

/// Judge a champion re-run.
///
/// `prior` and `remeasured` are composite values on the same scale;
/// `se_paired` is the paired standard error of their difference. The
/// champion loses tenure when the drop exceeds `z · SE` — the same
/// one-sided z the lattice already uses (`lcb_z = 1.645`), so the rule is
/// no stricter than the score it audits.
///
/// A non-finite input returns [`RerunVerdict::Holds`]: a failed measurement
/// is not evidence of overfit, and must not be able to unseat a champion.
#[must_use]
pub fn judge_rerun(prior: f64, remeasured: f64, se_paired: f64, z: f64) -> RerunVerdict {
    if !prior.is_finite() || !remeasured.is_finite() || !se_paired.is_finite() || se_paired < 0.0 {
        return RerunVerdict::Holds;
    }
    let drop = prior - remeasured;
    if drop > z * se_paired {
        RerunVerdict::Regressed
    } else {
        RerunVerdict::Holds
    }
}

/// One-sided z matching the composite's `lcb_z`.
pub const RERUN_Z: f64 = 1.645;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_is_deterministic_and_in_range() {
        for epoch in 0..200_u64 {
            let d = audit_draw_bps(epoch, "v3/private/abc", "champ");
            assert!(d < 10_000, "draw {d} out of range");
            assert_eq!(d, audit_draw_bps(epoch, "v3/private/abc", "champ"));
        }
    }

    #[test]
    fn draw_changes_with_every_input() {
        let base = audit_draw_bps(7, "slice", "champ");
        assert_ne!(base, audit_draw_bps(8, "slice", "champ"), "epoch matters");
        assert_ne!(base, audit_draw_bps(7, "slice2", "champ"), "slice matters");
        assert_ne!(base, audit_draw_bps(7, "slice", "other"), "champ matters");
    }

    #[test]
    fn audit_frequency_is_near_the_configured_rate() {
        // Unpredictable per round, but the expected cost must be the
        // budgeted one: ~half of rounds at the default probability.
        let n = 4_000_u64;
        let hits = (0..n)
            .filter(|e| audit_due(*e, "v3/private/round", "champ", AUDIT_PROBABILITY_BPS))
            .count();
        #[allow(clippy::cast_precision_loss)]
        let rate = hits as f64 / n as f64;
        assert!(
            (0.45..0.55).contains(&rate),
            "audit rate {rate} is not near 0.50"
        );
    }

    #[test]
    fn probability_bounds_are_absolute() {
        for epoch in 0..50 {
            assert!(!audit_due(epoch, "s", "c", 0), "0 bps never audits");
            assert!(audit_due(epoch, "s", "c", 10_000), "10 000 bps always does");
        }
    }

    #[test]
    fn a_material_drop_costs_tenure() {
        // Prior 0.80, re-measured 0.70, SE 0.02 ⇒ drop 0.10 ≫ 1.645·0.02.
        assert_eq!(
            judge_rerun(0.80, 0.70, 0.02, RERUN_Z),
            RerunVerdict::Regressed
        );
    }

    #[test]
    fn noise_sized_drops_do_not_unseat_the_champion() {
        // Drop 0.01 against SE 0.02 is well inside the noise band.
        assert_eq!(judge_rerun(0.80, 0.79, 0.02, RERUN_Z), RerunVerdict::Holds);
        // Just inside the band holds; just outside regresses. (The exact
        // boundary is a float coin-flip and is not a property worth pinning.)
        let band = RERUN_Z * 0.02;
        assert_eq!(
            judge_rerun(0.80, 0.80 - band * 0.99, 0.02, RERUN_Z),
            RerunVerdict::Holds
        );
        assert_eq!(
            judge_rerun(0.80, 0.80 - band * 1.01, 0.02, RERUN_Z),
            RerunVerdict::Regressed
        );
    }

    #[test]
    fn improvement_never_regresses() {
        assert_eq!(judge_rerun(0.70, 0.90, 0.01, RERUN_Z), RerunVerdict::Holds);
    }

    #[test]
    fn a_failed_measurement_cannot_unseat_a_champion() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(judge_rerun(bad, 0.5, 0.01, RERUN_Z), RerunVerdict::Holds);
            assert_eq!(judge_rerun(0.5, bad, 0.01, RERUN_Z), RerunVerdict::Holds);
            assert_eq!(judge_rerun(0.9, 0.5, bad, RERUN_Z), RerunVerdict::Holds);
        }
        assert_eq!(judge_rerun(0.9, 0.5, -1.0, RERUN_Z), RerunVerdict::Holds);
    }

    #[test]
    fn zero_variance_makes_any_real_drop_material() {
        assert_eq!(
            judge_rerun(0.80, 0.7999, 0.0, RERUN_Z),
            RerunVerdict::Regressed
        );
        assert_eq!(judge_rerun(0.80, 0.80, 0.0, RERUN_Z), RerunVerdict::Holds);
    }
}
