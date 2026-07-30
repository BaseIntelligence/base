//! Harbor pack contract surface for agent-v1.
//!
//! # Scope (this crate)
//! Pack identity, manifest projection, and loader traits used by the challenge
//! orchestrator and runners. Real Harbor archive parsing lands in a later task.
//!
//! # What stays in `agent-challenge`
//! Scoring, NoScore / D24 completeness gates, sr25519 signing of weight
//! payloads, and the signed raw-weight submit HTTP path remain in
//! `agent-challenge`. This crate must not grow scoring or submit logic.
//!
//! Skeletons only — no parser implementation yet.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Stable pack identity (content-addressed id string; format fixed later).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackId(String);

impl PackId {
    /// Construct from an already-validated id string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the raw id bytes as `str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stripped pack projection safe to hand to a runner (no full archive bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackProjection {
    /// Pack identity.
    pub id: PackId,
    /// Human-readable task name from the pack manifest (when present).
    pub name: String,
}

/// Failures while resolving or projecting a pack.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PackError {
    /// Pack id is unknown or not registered.
    #[error("pack not found: {0}")]
    NotFound(String),
    /// Pack bytes / manifest failed validation (parser lands later).
    #[error("pack invalid: {0}")]
    Invalid(String),
}

/// Resolve a pack id into a runner-safe projection.
pub trait PackStore: Send + Sync {
    /// Look up a pack by id and return its stripped projection.
    ///
    /// # Errors
    /// Returns [`PackError`] when the pack is missing or invalid.
    fn project(&self, id: &PackId) -> Result<PackProjection, PackError>;
}

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "agent-pack"
}

#[cfg(test)]
mod tests {
    use super::{crate_name, PackError, PackId, PackProjection, PackStore};

    struct EmptyStore;

    impl PackStore for EmptyStore {
        fn project(&self, id: &PackId) -> Result<PackProjection, PackError> {
            Err(PackError::NotFound(id.as_str().to_owned()))
        }
    }

    #[test]
    fn crate_name_is_agent_pack() {
        assert_eq!(crate_name(), "agent-pack");
    }

    #[test]
    fn empty_store_returns_not_found() {
        let store = EmptyStore;
        let id = PackId::new("missing");
        let err = store.project(&id).expect_err("empty store");
        assert_eq!(err, PackError::NotFound("missing".into()));
    }
}
