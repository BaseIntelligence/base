//! Integer-only map from vested marginal reward (ms) to leaf `Score` u64.
//!
//! Normative: `SCORE_MAX = 1_000_000`, range `0 ..= SCORE_MAX`, no floating-point
//! on the public score path (HYPERTRAINING.md §7.2 / agent-challenge pattern).

use crate::delta::{payable_delta_ms, PayInputs};

/// Maximum leaf score value (shared lattice with agent-challenge).
pub const SCORE_MAX: u64 = 1_000_000;

/// Default reference window (ms) when mapping absolute Δ to the score lattice.
///
/// A candidate that saves `DEFAULT_REFERENCE_MS` against the champion maps to
/// [`SCORE_MAX`] (clamped). Smaller savings scale linearly via integer math.
pub const DEFAULT_REFERENCE_MS: u64 = 1_000;

/// Map a non-negative vested reward in milliseconds to `0 ..= SCORE_MAX`.
///
/// ```text
/// score = min(SCORE_MAX, floor(reward_ms * SCORE_MAX / reference_ms))
/// ```
///
/// - `reward_ms == 0` → `0`
/// - `reference_ms == 0` → `0` (degenerate scale; refuse to invent a score)
/// - monotone non-decreasing in `reward_ms` for fixed `reference_ms > 0`
/// - saturates at [`SCORE_MAX`]
///
/// Integer arithmetic only (`u128` intermediate). No floating point.
#[must_use]
pub fn score_from_reward_ms(reward_ms: u64, reference_ms: u64) -> u64 {
    if reward_ms == 0 || reference_ms == 0 {
        return 0;
    }
    let num = u128::from(reward_ms).saturating_mul(u128::from(SCORE_MAX));
    let den = u128::from(reference_ms);
    let q = num / den;
    if q >= u128::from(SCORE_MAX) {
        SCORE_MAX
    } else {
        // q < SCORE_MAX ≤ u64::MAX
        u64::try_from(q).unwrap_or(SCORE_MAX)
    }
}

/// Map payable marginal Δ (guards + max(Δ,0)) to a leaf score using
/// [`DEFAULT_REFERENCE_MS`].
#[must_use]
pub fn score_from_pay_inputs(inputs: &PayInputs) -> u64 {
    score_from_pay_inputs_with_reference(inputs, DEFAULT_REFERENCE_MS)
}

/// Map payable marginal Δ to a leaf score with an explicit reference window.
#[must_use]
pub fn score_from_pay_inputs_with_reference(inputs: &PayInputs, reference_ms: u64) -> u64 {
    let delta = payable_delta_ms(inputs);
    score_from_reward_ms(delta, reference_ms)
}

/// Map already-vested reward ms to score (post-ledger path).
#[must_use]
pub fn score_from_vested_ms(vested_ms: u64) -> u64 {
    score_from_reward_ms(vested_ms, DEFAULT_REFERENCE_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_max_is_one_million() {
        assert_eq!(SCORE_MAX, 1_000_000);
    }

    #[test]
    fn delta_zero_maps_to_score_zero() {
        let p = PayInputs {
            t_champ_ms: 10_000,
            t_cand_ms: 10_000,
            guards_passed: true,
        };
        assert_eq!(score_from_pay_inputs(&p), 0);
        assert_eq!(score_from_reward_ms(0, DEFAULT_REFERENCE_MS), 0);
    }

    #[test]
    fn positive_delta_maps_to_positive_score() {
        let p = PayInputs {
            t_champ_ms: 10_000,
            t_cand_ms: 9_000,
            guards_passed: true,
        };
        let s = score_from_pay_inputs(&p);
        assert!(s > 0, "expected positive score, got {s}");
        assert!(s <= SCORE_MAX);
    }

    #[test]
    fn slower_candidate_score_zero() {
        let p = PayInputs {
            t_champ_ms: 10_000,
            t_cand_ms: 12_000,
            guards_passed: true,
        };
        assert_eq!(score_from_pay_inputs(&p), 0);
    }

    #[test]
    fn guards_fail_score_zero() {
        let p = PayInputs {
            t_champ_ms: 10_000,
            t_cand_ms: 1_000,
            guards_passed: false,
        };
        assert_eq!(score_from_pay_inputs(&p), 0);
    }

    #[test]
    fn full_reference_window_hits_score_max() {
        assert_eq!(
            score_from_reward_ms(DEFAULT_REFERENCE_MS, DEFAULT_REFERENCE_MS),
            SCORE_MAX
        );
        assert_eq!(
            score_from_reward_ms(DEFAULT_REFERENCE_MS * 2, DEFAULT_REFERENCE_MS),
            SCORE_MAX
        );
    }

    #[test]
    fn monotone_in_reward() {
        let a = score_from_reward_ms(100, 1_000);
        let b = score_from_reward_ms(200, 1_000);
        let c = score_from_reward_ms(500, 1_000);
        assert!(a < b && b < c);
    }

    #[test]
    fn zero_reference_yields_zero() {
        assert_eq!(score_from_reward_ms(500, 0), 0);
    }

    #[test]
    fn vested_path_matches_reward_path() {
        assert_eq!(
            score_from_vested_ms(250),
            score_from_reward_ms(250, DEFAULT_REFERENCE_MS)
        );
    }

    /// Public score module production code must stay integer-only.
    #[test]
    fn score_source_no_float() {
        let src = include_str!("score.rs");
        let prod = src
            .split("#[cfg(test)]")
            .next()
            .expect("production half of score.rs");
        assert!(
            !prod.contains("f32") && !prod.contains("f64"),
            "score production code must not use f32/f64"
        );
    }
}
