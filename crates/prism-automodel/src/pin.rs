//! `AutoModel` pin metadata (recipe 2.0).
//!
//! Production pin identity is a frozen upstream tag/commit plus the
//! content SHA-256 of the **operator-staged** pin tree (same ceremony as
//! the `FineWeb` dataset pin). CI uses [`FIXTURE_PIN`] — a tiny fake tree
//! under `fixtures/automodel-pin/` — never a full NVIDIA clone.

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Upstream `AutoModel` repository URL.
pub const AUTOMODEL_GIT_URL: &str = "https://github.com/NVIDIA-NeMo/Automodel";

/// Frozen upstream release tag for recipe 2.0.
pub const AUTOMODEL_GIT_TAG: &str = "v0.5.0";

/// Commit SHA for [`AUTOMODEL_GIT_TAG`] (annotated tag object → commit).
pub const AUTOMODEL_GIT_COMMIT: &str = "d02f49cb314554715aabb97e8dba6599c9f6e9e0";

/// Pin id miners must echo in `automodel.base`.
pub const AUTOMODEL_PIN_ID: &str = "automodel@v0.5.0";

/// Recipe contract version that requires this pin layout.
pub const AUTOMODEL_RECIPE_VERSION: &str = "2.0.0";

/// Content SHA-256 of the operator-staged production pin tree.
///
/// Computed with [`tree_content_sha256`] over a clean checkout of
/// [`AUTOMODEL_GIT_COMMIT`] (`.git` and dotfiles excluded). Operators must
/// stage that tree at `PRISM_AUTOMODEL_PIN_DIR` and verify before live intake.
/// Fixture tests use [`FIXTURE_CONTENT_SHA256`].
pub const AUTOMODEL_CONTENT_SHA256: &str =
    "f8af64ef572e2e3634dcbae7b351fdcd3c8d458caf2fe974aff26d301a11d838";

/// CI fixture pin id (`automodel.base` for local/Sim tests).
pub const FIXTURE_PIN_ID: &str = "automodel@fixture-v1";

/// Content SHA-256 of `fixtures/automodel-pin/` (see [`tree_content_sha256`]).
pub const FIXTURE_CONTENT_SHA256: &str =
    "2d6df7881ab17b810c67c2cc0ad338315701446f7a59a4f0ec361060aafd1950";

/// Frozen pin descriptor shared by production and fixture pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutomodelPin {
    /// Miner-facing pin id (`automodel.base`).
    pub id: &'static str,
    /// Upstream git URL (empty for the pure fixture pin).
    pub git_url: &'static str,
    /// Upstream tag (empty for the fixture pin).
    pub git_tag: &'static str,
    /// Upstream commit SHA (empty for the fixture pin).
    pub git_commit: &'static str,
    /// SHA-256 hex of the staged pin tree, or empty if not yet staged.
    pub content_sha256: &'static str,
    /// Recipe version that freezes this pin.
    pub recipe_version: &'static str,
}

impl AutomodelPin {
    /// Whether [`Self::content_sha256`] is populated for fail-closed verify.
    #[must_use]
    pub fn has_content_sha(&self) -> bool {
        !self.content_sha256.is_empty()
    }
}

/// Production `AutoModel` pin (recipe 2.0). Content SHA filled at pin ceremony.
pub const AUTOMODEL_PIN: AutomodelPin = AutomodelPin {
    id: AUTOMODEL_PIN_ID,
    git_url: AUTOMODEL_GIT_URL,
    git_tag: AUTOMODEL_GIT_TAG,
    git_commit: AUTOMODEL_GIT_COMMIT,
    content_sha256: AUTOMODEL_CONTENT_SHA256,
    recipe_version: AUTOMODEL_RECIPE_VERSION,
};

/// Tiny CI fixture pin (minimal AutoModel-like layout).
pub const FIXTURE_PIN: AutomodelPin = AutomodelPin {
    id: FIXTURE_PIN_ID,
    git_url: "",
    git_tag: "",
    git_commit: "",
    content_sha256: FIXTURE_CONTENT_SHA256,
    recipe_version: AUTOMODEL_RECIPE_VERSION,
};

/// Absolute path to the vendored CI fixture pin directory.
#[must_use]
pub fn fixture_pin_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/automodel-pin")
}

/// Absolute path to the vendored happy-path fixture patch.
#[must_use]
pub fn fixture_happy_patch_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/patches/happy.patch")
}

/// Canonical content SHA-256 of every regular file under `pin_dir`.
///
/// For each relative path (POSIX `/`, sorted), hash:
/// `path || 0x00 || decimal_len || 0x00 || bytes || 0x00`.
///
/// Skips `.git` and any path component starting with `.`.
pub fn tree_content_sha256(pin_dir: &Path) -> Result<String, std::io::Error> {
    let mut files = Vec::new();
    collect_files(pin_dir, pin_dir, &mut files)?;
    files.sort();
    let mut hasher = Sha256::new();
    for rel in &files {
        let abs = pin_dir.join(rel);
        let data = fs::read(&abs)?;
        hasher.update(rel.as_bytes());
        hasher.update([0]);
        hasher.update(data.len().to_string().as_bytes());
        hasher.update([0]);
        hasher.update(&data);
        hasher.update([0]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Verify `pin_dir` matches `pin.content_sha256` (fail-closed when set).
pub fn verify_pin_tree(pin: &AutomodelPin, pin_dir: &Path) -> Result<(), String> {
    if !pin.has_content_sha() {
        return Err(format!(
            "pin {} has empty content_sha256 (not staged)",
            pin.id
        ));
    }
    let got = tree_content_sha256(pin_dir).map_err(|e| e.to_string())?;
    if got != pin.content_sha256 {
        return Err(format!(
            "pin {} content sha mismatch: got {got}, want {}",
            pin.id, pin.content_sha256
        ));
    }
    Ok(())
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == ".git" || name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
            continue;
        }
        if !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .map_err(|_| std::io::Error::other("path outside pin root"))?
            .to_string_lossy()
            .replace('\\', "/");
        out.push(rel);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_content_sha_matches_constant() {
        let got = tree_content_sha256(&fixture_pin_dir()).expect("hash fixture");
        assert_eq!(got, FIXTURE_CONTENT_SHA256);
        verify_pin_tree(&FIXTURE_PIN, &fixture_pin_dir()).expect("verify");
    }

    #[test]
    fn production_pin_identity_frozen() {
        assert_eq!(AUTOMODEL_PIN.git_tag, "v0.5.0");
        assert_eq!(
            AUTOMODEL_PIN.git_commit,
            "d02f49cb314554715aabb97e8dba6599c9f6e9e0"
        );
        assert_eq!(AUTOMODEL_PIN.id, "automodel@v0.5.0");
        assert_eq!(AUTOMODEL_PIN.recipe_version, "2.0.0");
        assert!(AUTOMODEL_PIN.has_content_sha());
        assert_eq!(AUTOMODEL_PIN.content_sha256, AUTOMODEL_CONTENT_SHA256);
        assert_eq!(AUTOMODEL_CONTENT_SHA256.len(), 64);
    }
}
