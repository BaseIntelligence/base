//! Harbor pack domain model and total stripped agent projection.

use serde::{Deserialize, Serialize};

use crate::digest::{
    content_digest_label, digest_hex, pack_digest_from_entries, DigestBytes, DIGEST_LEN,
};
use crate::error::PackError;
use crate::PackId;

/// Accepted Harbor `task.toml` schema version.
pub const SCHEMA_VERSION_1_1: &str = "1.1";

/// JSON / struct field names allowed on [`StrippedDescriptor`] (total projection).
pub const STRIPPED_FIELD_NAMES: &[&str] = &[
    "task_id",
    "instruction",
    "environment_image_digest",
    "repository_url",
    "base_commit_hash",
    "deadline_sec",
];

/// Agent-safe pack projection: **only** identity + instruction + env pin + deadline.
///
/// This type is **total** w.r.t. held-out material: it has no field whose type can
/// carry `solution/` bytes, `tests/test.patch`, or grader sources. Adding such a
/// field is a contract break and fails [`StrippedDescriptor::assert_total_keys`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrippedDescriptor {
    /// `metadata.task_id`.
    pub task_id: String,
    /// Full `instruction.md` text.
    pub instruction: String,
    /// Content digest of `environment/Dockerfile` (`sha256:<hex>`).
    pub environment_image_digest: String,
    /// Upstream git URL from metadata.
    pub repository_url: String,
    /// Pinned base commit SHA from metadata.
    pub base_commit_hash: String,
    /// Agent wall-clock budget in whole seconds (`agent.timeout_sec`).
    pub deadline_sec: u64,
}

impl StrippedDescriptor {
    /// Structural allowlist check on serde field names (property / contract test hook).
    ///
    /// # Errors
    /// [`PackError::Invalid`] when serialized keys differ from [`STRIPPED_FIELD_NAMES`].
    pub fn assert_total_keys(&self) -> Result<(), PackError> {
        let value = serde_json::to_value(self).map_err(|e| PackError::Invalid(e.to_string()))?;
        let obj = value
            .as_object()
            .ok_or_else(|| PackError::Invalid("stripped descriptor is not a JSON object".into()))?;
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut expected = STRIPPED_FIELD_NAMES.to_vec();
        expected.sort_unstable();
        if keys != expected {
            return Err(PackError::Invalid(format!(
                "stripped keys {keys:?} != allowlist {expected:?}"
            )));
        }
        Ok(())
    }

    /// Pack identity derived from `task_id`.
    #[must_use]
    pub fn pack_id(&self) -> PackId {
        PackId::new(self.task_id.clone())
    }
}

/// Held-out verifier / oracle material — never projected to agents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldOutMaterials {
    /// `solution/solution.patch` when present.
    pub solution_patch: Option<Vec<u8>>,
    /// `tests/test.patch` when present.
    pub test_patch: Option<Vec<u8>>,
    /// `tests/grader.py` when present.
    pub grader_py: Option<Vec<u8>>,
}

/// Full Harbor pack after directory load (includes held-out bytes for operator use).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarborPack {
    /// Directory basename / `metadata.task_id` (must match).
    pub task_id: String,
    /// `schema_version` from `task.toml`.
    pub schema_version: String,
    /// Upstream repository URL.
    pub repository_url: String,
    /// Pinned base commit.
    pub base_commit_hash: String,
    /// Instruction body.
    pub instruction: String,
    /// Raw `environment/Dockerfile` bytes.
    pub dockerfile: Vec<u8>,
    /// Agent timeout in whole seconds.
    pub agent_timeout_sec: u64,
    /// Verifier timeout in whole seconds (when present).
    pub verifier_timeout_sec: Option<u64>,
    /// Held-out material (operator / verifier only).
    pub held_out: HeldOutMaterials,
    /// All regular files under the pack root, relative path → bytes (for digest).
    pub files: Vec<(String, Vec<u8>)>,
}

impl HarborPack {
    /// Content-addressed pack digest (stable across load order).
    #[must_use]
    pub fn pack_digest(&self) -> DigestBytes {
        pack_digest_from_entries(&self.files)
    }

    /// Lowercase hex of [`Self::pack_digest`].
    #[must_use]
    pub fn pack_digest_hex(&self) -> String {
        digest_hex(&self.pack_digest())
    }

    /// Environment image content digest from Dockerfile bytes.
    #[must_use]
    pub fn environment_image_digest(&self) -> String {
        content_digest_label(&self.dockerfile)
    }

    /// Total stripped projection safe to hand to a miner / runner.
    #[must_use]
    pub fn strip(&self) -> StrippedDescriptor {
        StrippedDescriptor {
            task_id: self.task_id.clone(),
            instruction: self.instruction.clone(),
            environment_image_digest: self.environment_image_digest(),
            repository_url: self.repository_url.clone(),
            base_commit_hash: self.base_commit_hash.clone(),
            deadline_sec: self.agent_timeout_sec,
        }
    }

    /// Pack id string.
    #[must_use]
    pub fn id(&self) -> PackId {
        PackId::new(self.task_id.clone())
    }
}

/// Compile-time width of digest (documents no variable-length secret channel on digest type).
pub const PACK_DIGEST_LEN: usize = DIGEST_LEN;

#[cfg(test)]
mod tests {
    use super::{StrippedDescriptor, STRIPPED_FIELD_NAMES};

    #[test]
    fn stripped_allowlist_has_six_fields() {
        assert_eq!(STRIPPED_FIELD_NAMES.len(), 6);
    }

    #[test]
    fn stripped_descriptor_assert_total_keys_ok() {
        let d = StrippedDescriptor {
            task_id: "t".into(),
            instruction: "i".into(),
            environment_image_digest: "sha256:aa".into(),
            repository_url: "https://example.com/r.git".into(),
            base_commit_hash: "abc".into(),
            deadline_sec: 1,
        };
        d.assert_total_keys().expect("total");
    }
}
