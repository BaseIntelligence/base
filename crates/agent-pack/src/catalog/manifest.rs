//! Catalog entry / manifest types and stable digest.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::digest::digest_hex;
use crate::error::CatalogError;

/// Domain-separation tag for catalog digest (hash family, not a signing tag).
pub const CATALOG_DIGEST_DOMAIN: &[u8] = b"base-agent-pack-catalog-v1";

/// Pinned deepagent git revision used as the sole floating-ref replacement.
///
/// Recorded in every manifest so two processes with different pins are
/// immediately detectable via [`CatalogManifest::catalog_digest`] / pin field.
pub const DEEPAGENT_PIN: &str = "4a16f063c83032ad4db2bb5a3099608bfdcb5fe2";

/// Relative path of the manifest file under a cache root.
pub const MANIFEST_FILE_NAME: &str = "manifest.json";

/// Relative directory under the cache root that holds pack trees.
pub const PACKS_DIR_NAME: &str = "packs";

/// One pack identity row in a pinned catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Pack identity (`metadata.task_id` / directory name).
    pub pack_id: String,
    /// Lowercase hex of the pack content digest (64 chars).
    pub pack_digest: String,
    /// Dockerfile content digest label (`sha256:<hex>`).
    pub environment_image_digest: String,
}

/// On-disk catalog manifest (byte-stable when entries + pin match).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogManifest {
    /// Deepagent (or source) pin string — never `"latest"`.
    pub pin: String,
    /// Sorted by `pack_id` ascending.
    pub entries: Vec<CatalogEntry>,
    /// Digest over pin + sorted entries ([`compute_catalog_digest`]).
    pub catalog_digest: String,
}

/// Stable catalog digest: SHA-256 over domain ‖ pin ‖ sorted entries.
///
/// Encoding (after sorting entries by `pack_id`):
/// `domain || 0x00 || pin_utf8 || 0x00 || Σ (pack_id || 0x00 || pack_digest || 0x00 || env_digest || 0x00)`.
#[must_use]
pub fn compute_catalog_digest(pin: &str, entries: &[CatalogEntry]) -> String {
    let mut ordered: Vec<&CatalogEntry> = entries.iter().collect();
    ordered.sort_by(|a, b| a.pack_id.cmp(&b.pack_id));

    let mut hasher = Sha256::new();
    hasher.update(CATALOG_DIGEST_DOMAIN);
    hasher.update([0_u8]);
    hasher.update(pin.as_bytes());
    hasher.update([0_u8]);
    for e in ordered {
        hasher.update(e.pack_id.as_bytes());
        hasher.update([0_u8]);
        hasher.update(e.pack_digest.as_bytes());
        hasher.update([0_u8]);
        hasher.update(e.environment_image_digest.as_bytes());
        hasher.update([0_u8]);
    }
    digest_hex(&hasher.finalize().into())
}

/// Build a manifest from entries + pin (sorts entries, fills `catalog_digest`).
#[must_use]
pub fn build_manifest(pin: impl Into<String>, mut entries: Vec<CatalogEntry>) -> CatalogManifest {
    let pin = pin.into();
    entries.sort_by(|a, b| a.pack_id.cmp(&b.pack_id));
    let catalog_digest = compute_catalog_digest(&pin, &entries);
    CatalogManifest {
        pin,
        entries,
        catalog_digest,
    }
}

/// Canonical JSON bytes for a manifest (stable across processes).
///
/// # Errors
/// [`CatalogError::Serialize`] when JSON encoding fails.
pub fn manifest_to_bytes(manifest: &CatalogManifest) -> Result<Vec<u8>, CatalogError> {
    let mut bytes =
        serde_json::to_vec_pretty(manifest).map_err(|e| CatalogError::Serialize(e.to_string()))?;
    if !bytes.ends_with(b"\n") {
        bytes.push(b'\n');
    }
    Ok(bytes)
}

/// Write `manifest.json` under `cache_dir`.
///
/// # Errors
/// I/O or serialize failures.
pub fn write_manifest(
    cache_dir: impl AsRef<std::path::Path>,
    manifest: &CatalogManifest,
) -> Result<(), CatalogError> {
    let path = cache_dir.as_ref().join(MANIFEST_FILE_NAME);
    let bytes = manifest_to_bytes(manifest)?;
    std::fs::write(&path, bytes).map_err(|e| CatalogError::Io {
        path,
        message: e.to_string(),
    })
}
