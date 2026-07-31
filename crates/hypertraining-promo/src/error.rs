//! Typed errors for the promotion state machine.

use crate::state::PromoState;
use thiserror::Error;

/// Failures when advancing or rolling back a challenger / champion.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PromoError {
    /// Transition is illegal from the current state.
    #[error("invalid transition from {from:?} via {action}")]
    InvalidTransition {
        /// State before the attempted action.
        from: PromoState,
        /// Action name (`screen`, `duel`, `holdout`, `promote`, `rollback`).
        action: &'static str,
    },
    /// Screen / duel / holdout evidence failed the stage rule.
    #[error("stage rejected: {reason}")]
    StageRejected {
        /// Human-readable reject reason (stable for tests).
        reason: &'static str,
    },
    /// No champion is installed (cannot duel against empty throne / rollback).
    #[error("no champion installed")]
    NoChampion,
    /// Lineage has no prior generation to restore.
    #[error("no prior champion in lineage to roll back to")]
    NoPriorChampion,
    /// p-value outside [0, 1].
    #[error("p-value out of range: {0}")]
    InvalidPValue(String),
    /// Cohort size mismatch for BH (ids vs p-values).
    #[error("BH cohort length mismatch")]
    CohortMismatch,
}
