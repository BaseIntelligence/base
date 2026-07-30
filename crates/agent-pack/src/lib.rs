//! Harbor pack contract surface for agent-v1.
//!
//! # Scope (this crate)
//! Parse Harbor pack directories (`task.toml` schema 1.1), compute a stable
//! [`HarborPack::pack_digest`], and project a **total** [`StrippedDescriptor`]
//! that cannot carry solution or held-out test bytes.
//!
//! # What stays in `agent-challenge`
//! Scoring, `NoScore` / D24 completeness gates, sr25519 signing of weight
//! payloads, and the signed raw-weight submit HTTP path remain in
//! `agent-challenge`. This crate must not grow scoring or submit logic.

#![forbid(unsafe_code)]

mod digest;
mod error;
mod load;
mod model;
mod select;

pub use digest::{
    content_digest_label, digest_hex, pack_digest_from_entries, sha256_bytes, DigestBytes,
    DIGEST_LEN,
};
pub use error::PackError;
pub use load::load_pack;
pub use model::{
    HarborPack, HeldOutMaterials, StrippedDescriptor, PACK_DIGEST_LEN, SCHEMA_VERSION_1_1,
    STRIPPED_FIELD_NAMES,
};
pub use select::{select_index, select_pack, HOTKEY_LEN, PACK_SELECT_DOMAIN};

/// Stable pack identity (content-addressed id string; typically `metadata.task_id`).
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

/// Runner-safe pack projection (alias of the total stripped descriptor).
pub type PackProjection = StrippedDescriptor;

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
    use super::{crate_name, PackError, PackId, PackStore};

    struct EmptyStore;

    impl PackStore for EmptyStore {
        fn project(&self, id: &PackId) -> Result<super::PackProjection, PackError> {
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

    /// Property: [`super::StrippedDescriptor`] serde shape has no channel for
    /// solution / held-out test payloads (structural key allowlist).
    #[test]
    fn stripped_descriptor_has_no_solution_or_test_fields_structurally() {
        use super::{StrippedDescriptor, STRIPPED_FIELD_NAMES};
        use serde_json::Value;

        let d = StrippedDescriptor {
            task_id: "x".into(),
            instruction: "do the thing".into(),
            environment_image_digest: "sha256:00".into(),
            repository_url: "https://example.com/r.git".into(),
            base_commit_hash: "deadbeef".into(),
            deadline_sec: 60,
        };
        d.assert_total_keys().expect("allowlist");

        let Value::Object(map) = serde_json::to_value(&d).expect("ser") else {
            panic!("expected object");
        };
        for forbidden in [
            "solution",
            "solution_patch",
            "test_patch",
            "tests",
            "grader",
            "grader_py",
            "held_out",
            "files",
            "dockerfile",
        ] {
            assert!(
                !map.contains_key(forbidden),
                "forbidden key present: {forbidden}"
            );
        }
        assert_eq!(map.len(), STRIPPED_FIELD_NAMES.len());

        // Type-level: every value is a JSON string or number — never a byte array.
        for (k, v) in &map {
            assert!(
                v.is_string() || v.is_number(),
                "field {k} must be string|number, got {v}"
            );
        }
    }
}
