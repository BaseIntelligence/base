//! Materialize Harbor task packs into a content-addressed cache.

use std::fs;
use std::path::Path;

use super::manifest::{
    build_manifest, write_manifest, CatalogEntry, CatalogManifest, DEEPAGENT_PIN, PACKS_DIR_NAME,
};
use crate::error::CatalogError;
use crate::load::load_pack;

/// Materialize every pack under `source_dir` into `cache_dir` and write the manifest.
///
/// `source_dir` is a Harbor `tasks/` directory (immediate children are pack roots).
/// Packs are copied to `cache_dir/packs/<pack_id>/`. Manifest is written to
/// `cache_dir/manifest.json`. Pin defaults to [`DEEPAGENT_PIN`].
///
/// # Errors
/// Empty source, pack load failures, I/O, or serialize errors.
pub fn materialize_catalog(
    source_dir: impl AsRef<Path>,
    cache_dir: impl AsRef<Path>,
) -> Result<CatalogManifest, CatalogError> {
    materialize_catalog_with_pin(source_dir, cache_dir, DEEPAGENT_PIN)
}

/// Same as [`materialize_catalog`] with an explicit pin string.
///
/// # Errors
/// Empty source, pack load failures, I/O, or serialize errors.
pub fn materialize_catalog_with_pin(
    source_dir: impl AsRef<Path>,
    cache_dir: impl AsRef<Path>,
    pin: &str,
) -> Result<CatalogManifest, CatalogError> {
    let source = source_dir.as_ref();
    let cache = cache_dir.as_ref();

    if pin.trim().is_empty() || pin == "latest" {
        return Err(CatalogError::FloatingPin(pin.to_owned()));
    }

    if !source.is_dir() {
        return Err(CatalogError::Io {
            path: source.to_path_buf(),
            message: "source_dir is not a directory".into(),
        });
    }

    fs::create_dir_all(cache).map_err(|e| CatalogError::Io {
        path: cache.to_path_buf(),
        message: e.to_string(),
    })?;
    let packs_root = cache.join(PACKS_DIR_NAME);
    fs::create_dir_all(&packs_root).map_err(|e| CatalogError::Io {
        path: packs_root.clone(),
        message: e.to_string(),
    })?;

    let mut entries = Vec::new();
    let children = fs::read_dir(source).map_err(|e| CatalogError::Io {
        path: source.to_path_buf(),
        message: e.to_string(),
    })?;

    for child in children {
        let child = child.map_err(|e| CatalogError::Io {
            path: source.to_path_buf(),
            message: e.to_string(),
        })?;
        let path = child.path();
        if !path.is_dir() {
            continue;
        }
        let name = child.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue;
        }
        if !path.join("task.toml").is_file() {
            continue;
        }

        let pack = load_pack(&path).map_err(CatalogError::from)?;
        let pack_id = pack.task_id.clone();
        let dest = packs_root.join(&pack_id);
        if dest.exists() {
            fs::remove_dir_all(&dest).map_err(|e| CatalogError::Io {
                path: dest.clone(),
                message: e.to_string(),
            })?;
        }
        copy_dir_recursive(&path, &dest)?;

        // Re-load from cache so digests match what load_catalog will see.
        let cached = load_pack(&dest).map_err(CatalogError::from)?;
        entries.push(CatalogEntry {
            pack_id: cached.task_id.clone(),
            pack_digest: cached.pack_digest_hex(),
            environment_image_digest: cached.environment_image_digest(),
        });
    }

    if entries.is_empty() {
        return Err(CatalogError::Empty);
    }

    let manifest = build_manifest(pin, entries);
    write_manifest(cache, &manifest)?;
    Ok(manifest)
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), CatalogError> {
    fs::create_dir_all(dst).map_err(|e| CatalogError::Io {
        path: dst.to_path_buf(),
        message: e.to_string(),
    })?;
    let entries = fs::read_dir(src).map_err(|e| CatalogError::Io {
        path: src.to_path_buf(),
        message: e.to_string(),
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| CatalogError::Io {
            path: src.to_path_buf(),
            message: e.to_string(),
        })?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        let ft = entry.file_type().map_err(|e| CatalogError::Io {
            path: from.clone(),
            message: e.to_string(),
        })?;
        if ft.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if ft.is_file() {
            fs::copy(&from, &to).map_err(|e| CatalogError::Io {
                path: from,
                message: e.to_string(),
            })?;
        }
    }
    Ok(())
}
