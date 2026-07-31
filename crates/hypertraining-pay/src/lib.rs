//! Hypertraining marginal payment, vesting/clawback, and commit-reveal.
//!
//! # Score meaning (brief §11 / HYPERTRAINING.md §7)
//!
//! ```text
//! Δ(candidate) = T_champion − T_candidate
//! pay ∝ max(Δ, 0)  iff guards passed
//! release = 1/V per segment; clawback unvested on regression
//! leaf score ∈ [0, SCORE_MAX] via integer-only map
//! ```
//!
//! Public `score_from_*` paths use only integer arithmetic (no floating-point types).

#![forbid(unsafe_code)]

mod commit_reveal;
mod delta;
mod error;
mod score;
mod vesting;

pub use commit_reveal::{commit, reveal, reveal_matches, CommitDigest, COMMIT_DOMAIN};
pub use delta::{payable_delta_ms, raw_delta_ms, PayInputs};
pub use error::PayError;
pub use score::{
    score_from_pay_inputs, score_from_pay_inputs_with_reference, score_from_reward_ms,
    score_from_vested_ms, DEFAULT_REFERENCE_MS, SCORE_MAX,
};
pub use vesting::{
    segment_release, GrantId, VestingGrant, VestingLedger, DEFAULT_VESTING_SEGMENTS,
};

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "hypertraining-pay"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_is_hypertraining_pay() {
        assert_eq!(crate_name(), "hypertraining-pay");
    }

    #[test]
    fn public_api_score_path_integer_only_surface() {
        // Compile-time surface: these symbols exist and return u64.
        let p = PayInputs {
            t_champ_ms: 2_000,
            t_cand_ms: 1_500,
            guards_passed: true,
        };
        let s: u64 = score_from_pay_inputs(&p);
        assert!(s > 0 && s <= SCORE_MAX);
        let z: u64 = score_from_reward_ms(0, DEFAULT_REFERENCE_MS);
        assert_eq!(z, 0);
        let v: u64 = score_from_vested_ms(0);
        assert_eq!(v, 0);
    }

    /// Entire crate production sources must not introduce f32/f64 on score path modules.
    #[test]
    fn production_score_and_delta_no_float() {
        for (name, src) in [
            ("score.rs", include_str!("score.rs")),
            ("delta.rs", include_str!("delta.rs")),
            ("lib.rs", include_str!("lib.rs")),
        ] {
            let prod = src.split("#[cfg(test)]").next().unwrap_or(src);
            assert!(
                !prod.contains("f32") && !prod.contains("f64"),
                "{name} production code must not use f32/f64"
            );
        }
    }
}
