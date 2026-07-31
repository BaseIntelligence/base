//! Export agent-safe stripped Harbor pack trees (no solution/tests).
//!
//! Used by agent-challenge pack serve to deliver miner-facing pack content
//! without held-out grader material.

use std::fs;
use std::io::{Cursor, Write};
use std::path::Path;

use flate2::write::GzEncoder;
use flate2::Compression;
use tar::{Builder, Header};

use crate::error::PackError;
use crate::model::HarborPack;

/// Relative path prefixes / names never delivered to miners.
const FORBIDDEN_PREFIXES: &[&str] = &["solution/", "tests/"];
const FORBIDDEN_EXACT: &[&str] = &[
    "solution",
    "tests",
    "solution.patch",
    "test.patch",
    "grader.py",
];

/// Whether a relative pack path is agent-safe (stripped).
#[must_use]
pub fn is_stripped_rel_path(rel: &str) -> bool {
    let rel = rel.trim_start_matches("./");
    if rel.is_empty() || rel.contains("..") {
        return false;
    }
    let lower = rel.to_ascii_lowercase();
    for exact in FORBIDDEN_EXACT {
        if lower == *exact {
            return false;
        }
    }
    for prefix in FORBIDDEN_PREFIXES {
        if lower.starts_with(prefix) {
            return false;
        }
    }
    // Also reject nested held-out style names at any depth
    if lower.ends_with("/solution.patch")
        || lower.ends_with("/test.patch")
        || lower.ends_with("/grader.py")
    {
        return false;
    }
    true
}

/// Collect agent-safe file entries from a loaded pack.
#[must_use]
pub fn stripped_file_entries(pack: &HarborPack) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = pack
        .files
        .iter()
        .filter(|(rel, _)| is_stripped_rel_path(rel))
        .map(|(rel, bytes)| (rel.clone(), bytes.clone()))
        .collect();

    // Guarantee required layout even if files map was incomplete.
    ensure_required(&mut out, "instruction.md", pack.instruction.as_bytes());
    ensure_required(&mut out, "environment/Dockerfile", &pack.dockerfile);
    if !out.iter().any(|(p, _)| p == "task.toml") {
        // Reconstruct a minimal task.toml sufficient for load_pack.
        let toml = reconstruct_minimal_task_toml(pack);
        out.push(("task.toml".into(), toml.into_bytes()));
    }

    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.dedup_by(|a, b| a.0 == b.0);
    out
}

fn ensure_required(out: &mut Vec<(String, Vec<u8>)>, rel: &str, bytes: &[u8]) {
    if !out.iter().any(|(p, _)| p == rel) {
        out.push((rel.into(), bytes.to_vec()));
    }
}

fn reconstruct_minimal_task_toml(pack: &HarborPack) -> String {
    let ver = pack
        .verifier_timeout_sec
        .map(|s| format!("timeout_sec = {s}.0"))
        .unwrap_or_default();
    let verifier_block = if ver.is_empty() {
        String::new()
    } else {
        format!("[verifier]\n{ver}\n")
    };
    format!(
        r#"schema_version = "1.1"

[metadata]
task_id = "{task_id}"
repository_url = "{repo}"
base_commit_hash = "{base}"

[agent]
timeout_sec = {agent}.0
{verifier}
"#,
        task_id = escape_toml_basic(&pack.task_id),
        repo = escape_toml_basic(&pack.repository_url),
        base = escape_toml_basic(&pack.base_commit_hash),
        agent = pack.agent_timeout_sec,
        verifier = verifier_block,
    )
}

fn escape_toml_basic(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Write a stripped pack tree to `dest` (creates dirs as needed).
///
/// # Errors
/// I/O failures or empty stripped set.
pub fn write_stripped_tree(pack: &HarborPack, dest: &Path) -> Result<(), PackError> {
    let entries = stripped_file_entries(pack);
    if entries.is_empty() {
        return Err(PackError::Invalid("stripped pack has no files".into()));
    }
    fs::create_dir_all(dest).map_err(|e| PackError::Io {
        path: dest.to_path_buf(),
        message: e.to_string(),
    })?;
    for (rel, bytes) in &entries {
        if !is_stripped_rel_path(rel) {
            return Err(PackError::Invalid(format!(
                "refusing to write held-out path {rel}"
            )));
        }
        let target = dest.join(rel);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|e| PackError::Io {
                path: parent.to_path_buf(),
                message: e.to_string(),
            })?;
        }
        fs::write(&target, bytes).map_err(|e| PackError::Io {
            path: target,
            message: e.to_string(),
        })?;
    }
    // Fail closed: never leave solution/ or tests/ under dest.
    for forbidden in ["solution", "tests"] {
        let p = dest.join(forbidden);
        if p.exists() {
            return Err(PackError::Invalid(format!(
                "stripped tree still contains {forbidden}"
            )));
        }
    }
    Ok(())
}

/// Export stripped pack as gzipped ustar bytes.
///
/// # Errors
/// I/O or archive failures.
pub fn export_stripped_tar_gz(pack: &HarborPack) -> Result<Vec<u8>, PackError> {
    let entries = stripped_file_entries(pack);
    if entries.is_empty() {
        return Err(PackError::Invalid("stripped pack has no files".into()));
    }
    let mut raw = Vec::new();
    {
        let mut builder = Builder::new(Cursor::new(&mut raw));
        for (rel, bytes) in &entries {
            if !is_stripped_rel_path(rel) {
                return Err(PackError::Invalid(format!(
                    "refusing to archive held-out path {rel}"
                )));
            }
            let mut header = Header::new_gnu();
            header
                .set_path(rel)
                .map_err(|e| PackError::Invalid(format!("tar path {rel}: {e}")))?;
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append(&header, bytes.as_slice())
                .map_err(|e| PackError::Invalid(format!("tar append {rel}: {e}")))?;
        }
        builder
            .finish()
            .map_err(|e| PackError::Invalid(format!("tar finish: {e}")))?;
    }
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&raw)
        .map_err(|e| PackError::Invalid(format!("gzip write: {e}")))?;
    enc.finish()
        .map_err(|e| PackError::Invalid(format!("gzip finish: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::load_pack;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn is_stripped_rejects_held_out_paths() {
        assert!(is_stripped_rel_path("instruction.md"));
        assert!(is_stripped_rel_path("task.toml"));
        assert!(is_stripped_rel_path("environment/Dockerfile"));
        assert!(!is_stripped_rel_path("solution/solution.patch"));
        assert!(!is_stripped_rel_path("tests/test.patch"));
        assert!(!is_stripped_rel_path("tests/grader.py"));
        assert!(!is_stripped_rel_path("solution"));
    }

    #[test]
    fn stripped_tree_from_fixture_has_no_solution() {
        let pack = load_pack(fixture("minimal-ok")).expect("load");
        let dir = tempfile::tempdir().expect("tmp");
        write_stripped_tree(&pack, dir.path()).expect("write");
        assert!(dir.path().join("instruction.md").is_file());
        assert!(dir.path().join("task.toml").is_file());
        assert!(dir.path().join("environment/Dockerfile").is_file());
        assert!(!dir.path().join("solution").exists());
        assert!(!dir.path().join("tests").exists());
        let reloaded = load_pack(dir.path()).expect("reload stripped");
        reloaded.strip().assert_total_keys().expect("keys");
        assert!(reloaded.held_out.solution_patch.is_none());
        assert!(reloaded.held_out.test_patch.is_none());
        assert!(reloaded.held_out.grader_py.is_none());
        let tar = export_stripped_tar_gz(&pack).expect("tar");
        assert!(tar.len() > 32);
        // gzip magic
        assert_eq!(&tar[0..2], &[0x1f, 0x8b]);
    }
}
