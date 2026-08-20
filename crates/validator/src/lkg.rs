//! Last sealed bundle on disk so a validator restart (or a gateway burn
//! fallback after master bounce) can still Match and submit weights.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Persisted last-known-good sealed epoch bundle (SCALE bytes).
#[derive(Debug, Clone)]
pub struct SealedBundleLkg {
    path: Option<PathBuf>,
}

impl SealedBundleLkg {
    /// Disabled (tests / no disk).
    #[must_use]
    pub const fn disabled() -> Self {
        Self { path: None }
    }

    /// Write/read `path`.
    #[must_use]
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
        }
    }

    /// `BASE_VALIDATOR_LKG_PATH`, else `/var/lib/base/last-sealed.bundle`.
    #[must_use]
    pub fn from_env() -> Self {
        let path = std::env::var("BASE_VALIDATOR_LKG_PATH")
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "/var/lib/base/last-sealed.bundle".into());
        Self {
            path: Some(PathBuf::from(path)),
        }
    }

    /// Persist SCALE bundle bytes (atomic replace).
    pub fn save(&self, bytes: &[u8]) {
        let Some(path) = self.path.as_deref() else {
            return;
        };
        if bytes.is_empty() {
            return;
        }
        if let Some(dir) = path.parent() {
            if let Err(e) = fs::create_dir_all(dir) {
                tracing::warn!(error = %e, path = %path.display(), "lkg mkdir failed");
                return;
            }
        }
        let tmp = path.with_extension("bundle.tmp");
        if let Err(e) = write_atomic(&tmp, path, bytes) {
            tracing::warn!(error = %e, path = %path.display(), "lkg save failed");
            return;
        }
        tracing::info!(
            event = "validator_lkg_saved",
            path = %path.display(),
            bytes = bytes.len(),
            "persisted last sealed bundle"
        );
    }

    /// Load last SCALE bytes, if any.
    #[must_use]
    pub fn load(&self) -> Option<Vec<u8>> {
        let path = self.path.as_deref()?;
        let bytes = fs::read(path).ok()?;
        if bytes.is_empty() {
            return None;
        }
        Some(bytes)
    }
}

fn write_atomic(tmp: &Path, dest: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    let mut f = fs::File::create(tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    fs::rename(tmp, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn disabled_is_noop() {
        let lkg = SealedBundleLkg::disabled();
        lkg.save(&[1, 2, 3]);
        assert!(lkg.load().is_none());
    }

    #[test]
    fn round_trip_file() {
        let dir = std::env::temp_dir().join(format!("base-lkg-{}", std::process::id()));
        let path = dir.join("last-sealed.bundle");
        let lkg = SealedBundleLkg::at(&path);
        lkg.save(b"bundle-bytes");
        assert_eq!(lkg.load().as_deref(), Some(b"bundle-bytes".as_slice()));
        let _ = fs::remove_dir_all(dir);
    }
}
