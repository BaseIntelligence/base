//! Load a Harbor pack directory (`tasks/<id>/`).

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::PackError;
use crate::model::{
    HarborPack, HeldOutMaterials, SCHEMA_VERSION_1_1,
};

/// Files that must exist for a valid pack layout.
const REQUIRED_REL_PATHS: &[&str] = &[
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
];

#[derive(Debug, Deserialize)]
struct RawTaskToml {
    schema_version: String,
    metadata: RawMetadata,
    agent: RawAgent,
    #[serde(default)]
    verifier: Option<RawVerifier>,
}

#[derive(Debug, Deserialize)]
struct RawMetadata {
    task_id: String,
    repository_url: String,
    /// Optional so we can emit [`PackError::MissingField`] ourselves.
    #[serde(default)]
    base_commit_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawAgent {
    timeout_sec: f64,
}

#[derive(Debug, Deserialize)]
struct RawVerifier {
    #[serde(default)]
    timeout_sec: Option<f64>,
}

/// Parse a Harbor pack directory into a [`HarborPack`].
///
/// # Errors
/// Missing layout files, unsupported schema, missing `base_commit_hash`, I/O, or TOML errors.
pub fn load_pack(dir: impl AsRef<Path>) -> Result<HarborPack, PackError> {
    let root = dir.as_ref();
    if !root.is_dir() {
        return Err(PackError::NotFound(root.display().to_string()));
    }

    for rel in REQUIRED_REL_PATHS {
        let p = root.join(rel);
        if !p.is_file() {
            return Err(PackError::Invalid(format!(
                "missing required pack file `{rel}` under {}",
                root.display()
            )));
        }
    }

    let task_toml_path = root.join("task.toml");
    let task_toml_text = read_to_string(&task_toml_path)?;
    let raw: RawTaskToml = toml::from_str(&task_toml_text)
        .map_err(|e| PackError::Toml(e.to_string()))?;

    if raw.schema_version != SCHEMA_VERSION_1_1 {
        return Err(PackError::UnsupportedSchema {
            found: raw.schema_version,
            expected: SCHEMA_VERSION_1_1,
        });
    }

    let base_commit_hash = require_nonempty(
        raw.metadata.base_commit_hash.as_deref(),
        "base_commit_hash",
    )?;
    let task_id = require_nonempty(Some(raw.metadata.task_id.as_str()), "task_id")?;
    let repository_url =
        require_nonempty(Some(raw.metadata.repository_url.as_str()), "repository_url")?;

    let agent_timeout_sec = finite_timeout_sec(raw.agent.timeout_sec, "agent.timeout_sec")?;

    let verifier_timeout_sec = match raw.verifier.as_ref().and_then(|v| v.timeout_sec) {
        Some(t) => Some(finite_timeout_sec(t, "verifier.timeout_sec")?),
        None => None,
    };

    let instruction_path = root.join("instruction.md");
    let instruction = read_to_string(&instruction_path)?;
    if instruction.trim().is_empty() {
        return Err(PackError::Invalid("instruction.md is empty".into()));
    }

    let dockerfile_path = root.join("environment/Dockerfile");
    let dockerfile = read_to_bytes(&dockerfile_path)?;
    if dockerfile.is_empty() {
        return Err(PackError::Invalid("environment/Dockerfile is empty".into()));
    }

    let files = collect_files(root)?;
    let held_out = HeldOutMaterials {
        solution_patch: read_optional_bytes(&root.join("solution/solution.patch"))?,
        test_patch: read_optional_bytes(&root.join("tests/test.patch"))?,
        grader_py: read_optional_bytes(&root.join("tests/grader.py"))?,
    };

    Ok(HarborPack {
        task_id,
        schema_version: raw.schema_version,
        repository_url,
        base_commit_hash,
        instruction,
        dockerfile,
        agent_timeout_sec,
        verifier_timeout_sec,
        held_out,
        files,
    })
}

fn require_nonempty(value: Option<&str>, field: &'static str) -> Result<String, PackError> {
    match value {
        Some(s) if !s.trim().is_empty() => Ok(s.trim().to_owned()),
        _ => Err(PackError::MissingField { field }),
    }
}

fn finite_timeout_sec(value: f64, field: &'static str) -> Result<u64, PackError> {
    // Harbor timeouts are wall-clock seconds; reject absurd magnitudes.
    const MAX_SEC: f64 = 86_400_000.0; // 1000 days
    if !value.is_finite() || value < 0.0 {
        return Err(PackError::Invalid(format!(
            "{field} must be a finite non-negative number"
        )));
    }
    if value > MAX_SEC {
        return Err(PackError::Invalid(format!(
            "{field} exceeds maximum supported timeout"
        )));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(value.floor() as u64)
}

fn read_to_string(path: &Path) -> Result<String, PackError> {
    fs::read_to_string(path).map_err(|e| PackError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

fn read_to_bytes(path: &Path) -> Result<Vec<u8>, PackError> {
    fs::read(path).map_err(|e| PackError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

fn read_optional_bytes(path: &Path) -> Result<Option<Vec<u8>>, PackError> {
    if !path.exists() {
        return Ok(None);
    }
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(read_to_bytes(path)?))
}

/// Walk pack root; skip hidden paths (`.git`, …). Paths use `/` separators.
fn collect_files(root: &Path) -> Result<Vec<(String, Vec<u8>)>, PackError> {
    let mut out = Vec::new();
    collect_files_rec(root, root, &mut out)?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn collect_files_rec(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, Vec<u8>)>,
) -> Result<(), PackError> {
    let entries = fs::read_dir(dir).map_err(|e| PackError::Io {
        path: dir.to_path_buf(),
        message: e.to_string(),
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| PackError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?;
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue;
        }
        let ft = entry.file_type().map_err(|e| PackError::Io {
            path: path.clone(),
            message: e.to_string(),
        })?;
        if ft.is_dir() {
            collect_files_rec(root, &path, out)?;
        } else if ft.is_file() {
            let rel = path.strip_prefix(root).map_err(|_| PackError::Invalid(format!(
                "path {} not under pack root",
                path.display()
            )))?;
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            let bytes = read_to_bytes(&path)?;
            out.push((rel_str, bytes));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::load_pack;
    use crate::error::PackError;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn load_minimal_ok_strips_without_held_out_keys() {
        let pack = load_pack(fixture("minimal-ok")).expect("load");
        assert_eq!(pack.task_id, "minimal-ok");
        assert_eq!(
            pack.base_commit_hash,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        assert!(pack.held_out.solution_patch.is_some());
        assert!(pack.held_out.test_patch.is_some());
        assert!(pack.held_out.grader_py.is_some());

        let stripped = pack.strip();
        stripped.assert_total_keys().expect("total keys");
        assert_eq!(stripped.task_id, "minimal-ok");
        assert!(stripped.instruction.contains("Fix the bug"));
        assert!(stripped.environment_image_digest.starts_with("sha256:"));
        assert_eq!(stripped.deadline_sec, 300);
        assert_eq!(
            stripped.repository_url,
            "https://github.com/example/repo.git"
        );

        // Held-out markers must not appear in any stripped string field.
        let json = serde_json::to_string(&stripped).expect("json");
        assert!(!json.contains("SECRET_SOLUTION"));
        assert!(!json.contains("SECRET_TEST"));
        assert!(!json.contains("grader"));
        assert!(!json.contains("solution"));
        assert!(!json.contains("test_patch"));
    }

    #[test]
    fn pack_digest_stable_across_two_loads() {
        let a = load_pack(fixture("minimal-ok")).expect("a").pack_digest_hex();
        let b = load_pack(fixture("minimal-ok")).expect("b").pack_digest_hex();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn missing_base_commit_hash_is_typed_error() {
        let err = load_pack(fixture("missing-base")).expect_err("must fail");
        assert_eq!(
            err,
            PackError::MissingField {
                field: "base_commit_hash"
            }
        );
        let msg = err.to_string();
        assert!(
            msg.contains("base_commit_hash"),
            "error must name the field: {msg}"
        );
    }
}
