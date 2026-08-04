//! Miner harness bundle contract + embedded Python harness / SDK.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;

use design_challenge_task::SUBMISSION_DOMAIN;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Embedded operator harness entrypoint.
pub const HARNESS_PY: &str = include_str!("../python/design_harness.py");

/// Embedded `base_design` SDK package init.
pub const BASE_DESIGN_INIT_PY: &str = include_str!("../python/base_design/__init__.py");

/// Max extra files in a miner bundle.
pub const MAX_EXTRA_FILES: usize = 16;

/// Max bytes per extra file.
pub const MAX_FILE_BYTES: usize = 256 * 1024;

/// Max total bundle bytes (agent + pyproject + extras).
pub const MAX_BUNDLE_BYTES: usize = 1024 * 1024;

/// Required output pages under `/out/pages/`.
pub const REQUIRED_PAGES: &[&str] = &["index.html", "pricing.html", "components.html"];

/// Miner-submitted harness bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessBundle {
    /// Miner hotkey (lowercase 64 hex).
    pub miner_hotkey: String,
    /// `agent.py` source.
    pub agent_py: String,
    /// `pyproject.toml` contents.
    pub pyproject_toml: String,
    /// Additional files (relative path → utf-8 or base64 text).
    #[serde(default)]
    pub extra_files: BTreeMap<String, String>,
}

/// Validation / digest errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HarnessError {
    /// Structural validation failed.
    #[error("invalid harness: {0}")]
    Invalid(String),
}

/// Validate size/shape limits and required entrypoints.
pub fn validate_bundle(b: &HarnessBundle) -> Result<(), HarnessError> {
    let hk = b.miner_hotkey.trim();
    if hk.len() != 64 || !hk.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(HarnessError::Invalid("miner_hotkey must be 64 hex".into()));
    }
    if !b.agent_py.contains("def run") {
        return Err(HarnessError::Invalid(
            "agent.py must define run(...)".into(),
        ));
    }
    if b.pyproject_toml.trim().is_empty() {
        return Err(HarnessError::Invalid("pyproject.toml required".into()));
    }
    if b.extra_files.len() > MAX_EXTRA_FILES {
        return Err(HarnessError::Invalid(format!(
            "too many extra files (max {MAX_EXTRA_FILES})"
        )));
    }
    let mut total = b.agent_py.len() + b.pyproject_toml.len();
    for (path, body) in &b.extra_files {
        if path.contains("..") || path.starts_with('/') || path.contains('\0') {
            return Err(HarnessError::Invalid(format!("bad extra path: {path}")));
        }
        if body.len() > MAX_FILE_BYTES {
            return Err(HarnessError::Invalid(format!("file too large: {path}")));
        }
        total = total.saturating_add(body.len());
    }
    if total > MAX_BUNDLE_BYTES {
        return Err(HarnessError::Invalid("bundle exceeds 1 MiB".into()));
    }
    Ok(())
}

/// Content digest id: `sha256(domain || hotkey || agent || pyproject || extras)`.
#[must_use]
pub fn harness_id(b: &HarnessBundle) -> String {
    let mut h = Sha256::new();
    h.update(SUBMISSION_DOMAIN);
    h.update(b.miner_hotkey.trim().as_bytes());
    h.update(b.agent_py.as_bytes());
    h.update(b.pyproject_toml.as_bytes());
    for (k, v) in &b.extra_files {
        h.update(k.as_bytes());
        h.update(v.as_bytes());
    }
    hex::encode(h.finalize())
}

/// Materialize bundle + injected SDK into a work directory layout description.
#[derive(Debug, Clone)]
pub struct StagedFiles {
    /// Relative path → contents.
    pub files: BTreeMap<String, String>,
}

/// Stage miner sources + operator harness/SDK for the sandbox work root.
#[must_use]
pub fn stage_work_files(b: &HarnessBundle) -> StagedFiles {
    let mut files = BTreeMap::new();
    files.insert("agent.py".into(), b.agent_py.clone());
    files.insert("pyproject.toml".into(), b.pyproject_toml.clone());
    for (k, v) in &b.extra_files {
        files.insert(k.clone(), v.clone());
    }
    files.insert("design_harness.py".into(), HARNESS_PY.into());
    files.insert("base_design/__init__.py".into(), BASE_DESIGN_INIT_PY.into());
    StagedFiles { files }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn sample() -> HarnessBundle {
        HarnessBundle {
            miner_hotkey: "ab".repeat(32),
            agent_py:
                "def run(task, llm, out):\n    out.write_page('index.html', '<html></html>')\n"
                    .into(),
            pyproject_toml: "[project]\nname='x'\nversion='0.1.0'\n".into(),
            extra_files: BTreeMap::new(),
        }
    }

    #[test]
    fn validate_and_id_stable() {
        let b = sample();
        validate_bundle(&b).unwrap();
        let id = harness_id(&b);
        assert_eq!(id.len(), 64);
        assert_eq!(id, harness_id(&b));
        let staged = stage_work_files(&b);
        assert!(staged.files.contains_key("design_harness.py"));
        assert!(staged.files.contains_key("base_design/__init__.py"));
    }

    #[test]
    fn rejects_missing_run() {
        let mut b = sample();
        b.agent_py = "x = 1\n".into();
        assert!(validate_bundle(&b).is_err());
    }
}
