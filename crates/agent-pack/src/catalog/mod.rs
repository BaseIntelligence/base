//! Pinned pack catalog: materialize, manifest integrity, catalog digest.
//!
//! # Split-brain safety
//! Challenge replicas pin the same deepagent revision ([`DEEPAGENT_PIN`]) and the
//! same [`CatalogManifest::catalog_digest`]. Divergent pins or tampered cache
//! entries fail closed on [`load_catalog`].
//!
//! # Catalog order
//! Entries are always sorted by `pack_id` so [`crate::select_pack`] callers get a
//! stable ordered slice via [`Catalog::pack_ids`].

mod manifest;
mod materialize;

use std::fs;
use std::path::{Path, PathBuf};

pub use manifest::{
    build_manifest, compute_catalog_digest, manifest_to_bytes, write_manifest, CatalogEntry,
    CatalogManifest, CATALOG_DIGEST_DOMAIN, DEEPAGENT_PIN, MANIFEST_FILE_NAME, PACKS_DIR_NAME,
};
pub use materialize::{materialize_catalog, materialize_catalog_with_pin};

use crate::error::{CatalogError, PackError};
use crate::load::load_pack;
use crate::PackId;

use self::manifest::{compute_catalog_digest as recompute_digest, MANIFEST_FILE_NAME as MANIFEST};

/// Verified in-memory catalog after [`load_catalog`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Catalog {
    manifest: CatalogManifest,
    /// Cache root that was verified.
    cache_dir: PathBuf,
}

impl Catalog {
    /// Borrow the verified manifest.
    #[must_use]
    pub fn manifest(&self) -> &CatalogManifest {
        &self.manifest
    }

    /// Catalog digest (same as `manifest.catalog_digest`).
    #[must_use]
    pub fn catalog_digest(&self) -> &str {
        &self.manifest.catalog_digest
    }

    /// Source pin recorded at materialization.
    #[must_use]
    pub fn pin(&self) -> &str {
        &self.manifest.pin
    }

    /// Ordered pack ids (sorted) for [`crate::select_pack`].
    #[must_use]
    pub fn pack_ids(&self) -> Vec<PackId> {
        self.manifest
            .entries
            .iter()
            .map(|e| PackId::new(e.pack_id.clone()))
            .collect()
    }

    /// Number of packs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.manifest.entries.len()
    }

    /// Whether the catalog has zero packs (should not occur after successful load).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifest.entries.is_empty()
    }

    /// Cache directory path.
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Path to a pack directory inside the cache.
    #[must_use]
    pub fn pack_dir(&self, pack_id: &str) -> PathBuf {
        self.cache_dir.join(PACKS_DIR_NAME).join(pack_id)
    }
}

/// Load and verify a catalog from `cache_dir` (manifest + on-disk pack digests).
///
/// # Errors
/// Missing/invalid manifest, empty catalog, digest mismatch (names `pack_id`), I/O.
pub fn load_catalog(cache_dir: impl AsRef<Path>) -> Result<Catalog, CatalogError> {
    let cache = cache_dir.as_ref();
    let manifest_path = cache.join(MANIFEST);
    if !manifest_path.is_file() {
        return Err(CatalogError::ManifestMissing {
            path: manifest_path,
        });
    }

    let text = fs::read_to_string(&manifest_path).map_err(|e| CatalogError::Io {
        path: manifest_path.clone(),
        message: e.to_string(),
    })?;
    let mut manifest: CatalogManifest =
        serde_json::from_str(&text).map_err(|e| CatalogError::ManifestInvalid(e.to_string()))?;

    if manifest.pin.trim().is_empty() || manifest.pin == "latest" {
        return Err(CatalogError::FloatingPin(manifest.pin.clone()));
    }

    if manifest.entries.is_empty() {
        return Err(CatalogError::Empty);
    }

    manifest
        .entries
        .sort_by(|a, b| a.pack_id.cmp(&b.pack_id));

    let expected_digest = recompute_digest(&manifest.pin, &manifest.entries);
    if expected_digest != manifest.catalog_digest {
        return Err(CatalogError::CatalogDigestMismatch {
            expected: expected_digest,
            found: manifest.catalog_digest.clone(),
        });
    }

    for entry in &manifest.entries {
        let pack_path = cache.join(PACKS_DIR_NAME).join(&entry.pack_id);
        let pack = load_pack(&pack_path).map_err(|e| match e {
            PackError::NotFound(_) | PackError::Io { .. } => CatalogError::Integrity {
                pack_id: entry.pack_id.clone(),
                message: format!(
                    "pack directory missing or unreadable at {}",
                    pack_path.display()
                ),
            },
            other => CatalogError::from(other),
        })?;

        let actual_digest = pack.pack_digest_hex();
        if actual_digest != entry.pack_digest {
            return Err(CatalogError::Integrity {
                pack_id: entry.pack_id.clone(),
                message: format!(
                    "pack_digest mismatch: manifest={} on_disk={}",
                    entry.pack_digest, actual_digest
                ),
            });
        }

        let actual_env = pack.environment_image_digest();
        if actual_env != entry.environment_image_digest {
            return Err(CatalogError::Integrity {
                pack_id: entry.pack_id.clone(),
                message: format!(
                    "environment_image_digest mismatch: manifest={} on_disk={}",
                    entry.environment_image_digest, actual_env
                ),
            });
        }
    }

    Ok(Catalog {
        manifest,
        cache_dir: cache.to_path_buf(),
    })
}
