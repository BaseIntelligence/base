//! Hypertraining promotion state machine (brief §9.4).
//!
//! ```text
//! ADMITTED → SCREENED (K=3) → DUELLED (K=5) → CONFIRMED (holdout) → CHAMPION
//!                 │ fail              │ fail           │ disagree
//!                 └──────────────────┴────────────────┴──► REJECTED
//! CHAMPION ── later regression ──► ROLLBACK to prior hashed checkpoint C(n-1)
//! ```
//!
//! - Screen K = [`SCREEN_K`] (3), promotion K = [`PROMOTION_K`] (5)
//! - α = [`ALPHA`] (0.05) with Benjamini–Hochberg across duel cohort
//! - Public hashed checkpoint lineage enables atomic rollback

#![forbid(unsafe_code)]
// Result errors are documented on PromoError variants.
#![allow(clippy::missing_errors_doc)]

mod bh;
mod error;
mod lineage;
mod machine;
mod state;

pub use bh::{benjamini_hochberg, benjamini_hochberg_default, DEFAULT_ALPHA};
pub use error::PromoError;
pub use lineage::{hash_hex, hash_lineage_entry, CheckpointLineage, LineageEntry};
pub use machine::{Challenger, DuelEvidence, HoldoutEvidence, PromotionMachine, ScreenEvidence};
pub use state::{
    ChallengerId, CheckpointHash, PromoState, RejectReason, ALPHA, CALIBRATION_K, PROMOTION_K,
    SCREEN_K,
};

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "hypertraining-promo"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_is_hypertraining_promo() {
        assert_eq!(crate_name(), "hypertraining-promo");
    }

    #[test]
    fn pins_match_brief() {
        assert_eq!(SCREEN_K, 3);
        assert_eq!(PROMOTION_K, 5);
        assert_eq!(CALIBRATION_K, 10);
        assert!((ALPHA - 0.05).abs() < f64::EPSILON);
    }
}
