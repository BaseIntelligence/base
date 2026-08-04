//! Miner harness bundle contract + embedded Python harness / SDK.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

mod env;
mod zip_bundle;

use std::collections::BTreeMap;

use design_challenge_task::SUBMISSION_DOMAIN;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub use env::{
    decode_env_from_extras, encode_env_into_extras, peek_env_from_extras, validate_env_vars,
    EnvError, ENV_EXTRA_KEY, MAX_ENV_KEYS,
};
pub use zip_bundle::{harness_from_zip, ZipBundleError};

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
    /// Miner launch env for sandbox run phase (API keys, etc.). Never logged.
    #[serde(default)]
    pub env_vars: BTreeMap<String, String>,
}

/// Validation / digest errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HarnessError {
    /// Structural validation failed.
    #[error("invalid harness: {0}")]
    Invalid(String),
}

impl From<EnvError> for HarnessError {
    fn from(e: EnvError) -> Self {
        Self::Invalid(e.to_string())
    }
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
    if b.extra_files.contains_key(ENV_EXTRA_KEY) {
        return Err(HarnessError::Invalid(format!(
            "reserved extra path: {ENV_EXTRA_KEY}"
        )));
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
    validate_env_vars(&b.env_vars)?;
    Ok(())
}

/// Content digest id: `sha256(domain || hotkey || agent || pyproject || extras || env_keys)`.
///
/// Env **values** are included so distinct secrets yield distinct digests, but
/// callers must never log the digest inputs.
#[must_use]
pub fn harness_id(b: &HarnessBundle) -> String {
    let mut h = Sha256::new();
    h.update(SUBMISSION_DOMAIN);
    h.update(b.miner_hotkey.trim().as_bytes());
    h.update(b.agent_py.as_bytes());
    h.update(b.pyproject_toml.as_bytes());
    for (k, v) in &b.extra_files {
        if k == ENV_EXTRA_KEY {
            continue;
        }
        h.update(k.as_bytes());
        h.update(v.as_bytes());
    }
    for (k, v) in &b.env_vars {
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
        if k == ENV_EXTRA_KEY {
            continue;
        }
        files.insert(k.clone(), v.clone());
    }
    files.insert("design_harness.py".into(), HARNESS_PY.into());
    files.insert("base_design/__init__.py".into(), BASE_DESIGN_INIT_PY.into());
    StagedFiles { files }
}

/// Persistable extras map including reserved env slot.
#[must_use]
pub fn extras_for_store(b: &HarnessBundle) -> BTreeMap<String, String> {
    let mut extras = b.extra_files.clone();
    encode_env_into_extras(&mut extras, &b.env_vars);
    extras
}

/// Reconstruct bundle fields from a stored harness row.
#[must_use]
pub fn bundle_from_stored(
    miner_hotkey: String,
    agent_py: String,
    pyproject_toml: String,
    mut extra_files: BTreeMap<String, String>,
) -> HarnessBundle {
    let env_vars = decode_env_from_extras(&mut extra_files);
    HarnessBundle {
        miner_hotkey,
        agent_py,
        pyproject_toml,
        extra_files,
        env_vars,
    }
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
            env_vars: BTreeMap::new(),
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

    #[test]
    fn env_changes_digest() {
        let mut a = sample();
        let mut b = sample();
        b.env_vars.insert("FOO".into(), "1".into());
        validate_bundle(&b).unwrap();
        assert_ne!(harness_id(&a), harness_id(&b));
        a.env_vars.insert("FOO".into(), "1".into());
        assert_eq!(harness_id(&a), harness_id(&b));
    }
}
