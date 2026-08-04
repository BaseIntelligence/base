//! ZIP → [`HarnessBundle`] unpack (path-safe).

use std::collections::BTreeMap;
use std::io::{Cursor, Read};

use thiserror::Error;
use zip::ZipArchive;

use crate::{
    validate_bundle, HarnessBundle, HarnessError, MAX_BUNDLE_BYTES, MAX_EXTRA_FILES, MAX_FILE_BYTES,
};

/// ZIP unpack errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ZipBundleError {
    /// Archive / structure failure.
    #[error("invalid zip harness: {0}")]
    Invalid(String),
}

impl From<HarnessError> for ZipBundleError {
    fn from(e: HarnessError) -> Self {
        Self::Invalid(e.to_string())
    }
}

/// Unpack a ZIP into a validated [`HarnessBundle`].
///
/// Required members: `agent.py`, `pyproject.toml` (flat or one top-level folder).
/// Other regular files become `extra_files` (≤ limits). Symlinks / absolute /
/// `..` paths are rejected.
///
/// # Errors
/// Archive or contract validation failures.
pub fn harness_from_zip(
    miner_hotkey: &str,
    zip_bytes: &[u8],
    env_vars: BTreeMap<String, String>,
) -> Result<HarnessBundle, ZipBundleError> {
    if zip_bytes.len() > MAX_BUNDLE_BYTES.saturating_mul(2) {
        return Err(ZipBundleError::Invalid("zip exceeds size budget".into()));
    }
    let cursor = Cursor::new(zip_bytes);
    let mut archive =
        ZipArchive::new(cursor).map_err(|e| ZipBundleError::Invalid(format!("zip open: {e}")))?;

    let mut agent_py = None;
    let mut pyproject_toml = None;
    let mut extra_files = BTreeMap::new();
    let mut total = 0usize;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| ZipBundleError::Invalid(format!("zip entry: {e}")))?;
        if file.is_dir() {
            continue;
        }
        #[allow(deprecated)]
        if file.unix_mode().is_some_and(|m| m & 0o170_000 == 0o120_000) {
            return Err(ZipBundleError::Invalid("symlinks forbidden".into()));
        }
        let name = file
            .enclosed_name()
            .ok_or_else(|| ZipBundleError::Invalid("unsafe zip path".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        let name = name.trim_start_matches("./");
        if name.is_empty() || name.contains("..") || name.starts_with('/') || name.contains('\0') {
            return Err(ZipBundleError::Invalid(format!("bad path: {name}")));
        }
        let rel = normalize_zip_rel(name);
        if rel.is_empty() {
            continue;
        }
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| ZipBundleError::Invalid(format!("zip read: {e}")))?;
        if data.len() > MAX_FILE_BYTES {
            return Err(ZipBundleError::Invalid(format!("file too large: {rel}")));
        }
        total = total.saturating_add(data.len());
        if total > MAX_BUNDLE_BYTES {
            return Err(ZipBundleError::Invalid("bundle exceeds 1 MiB".into()));
        }
        let text = String::from_utf8(data)
            .map_err(|_| ZipBundleError::Invalid(format!("non-utf8 file: {rel}")))?;
        match rel.as_str() {
            "agent.py" => agent_py = Some(text),
            "pyproject.toml" => pyproject_toml = Some(text),
            other => {
                if extra_files.len() >= MAX_EXTRA_FILES {
                    return Err(ZipBundleError::Invalid("too many extra files".into()));
                }
                extra_files.insert(other.to_owned(), text);
            }
        }
    }

    let bundle = HarnessBundle {
        miner_hotkey: miner_hotkey.trim().to_ascii_lowercase(),
        agent_py: agent_py
            .ok_or_else(|| ZipBundleError::Invalid("agent.py missing from zip".into()))?,
        pyproject_toml: pyproject_toml
            .ok_or_else(|| ZipBundleError::Invalid("pyproject.toml missing from zip".into()))?,
        extra_files,
        env_vars,
    };
    validate_bundle(&bundle)?;
    Ok(bundle)
}

/// Peel a single wrapper directory when present (`pkg/agent.py` → `agent.py`).
fn normalize_zip_rel(name: &str) -> String {
    let parts: Vec<&str> = name.split('/').filter(|p| !p.is_empty()).collect();
    match parts.as_slice() {
        [] => String::new(),
        [one] => (*one).to_owned(),
        [root, rest @ ..]
            if root
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') =>
        {
            rest.join("/")
        }
        _ => name.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    fn make_zip(files: &[(&str, &str)]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = ZipWriter::new(&mut buf);
            let opts = SimpleFileOptions::default();
            for (name, body) in files {
                w.start_file(*name, opts).unwrap();
                w.write_all(body.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn unpacks_flat_zip() {
        let z = make_zip(&[
            (
                "agent.py",
                "def run(task, llm, out):\n    out.write_page('index.html', '<html></html>')\n",
            ),
            ("pyproject.toml", "[project]\nname='t'\nversion='0'\n"),
            ("readme.txt", "hi"),
        ]);
        let b = harness_from_zip(&"ab".repeat(32), &z, BTreeMap::new()).unwrap();
        assert!(b.agent_py.contains("def run"));
        assert_eq!(
            b.extra_files.get("readme.txt").map(String::as_str),
            Some("hi")
        );
    }

    #[test]
    fn unpacks_nested_zip() {
        let z = make_zip(&[
            ("bundle/agent.py", "def run(task, llm, out):\n    pass\n"),
            (
                "bundle/pyproject.toml",
                "[project]\nname='t'\nversion='0'\n",
            ),
        ]);
        let b = harness_from_zip(&"cd".repeat(32), &z, BTreeMap::new()).unwrap();
        assert!(b.agent_py.contains("def run"));
    }
}
