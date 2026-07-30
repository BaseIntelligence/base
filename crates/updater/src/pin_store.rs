//! Durable `current.json` / `previous.json` pin files.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::UpdaterError;

/// One pin record written to disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinRecord {
    /// Compose service name.
    pub service: String,
    /// Full pinned image `repo@sha256:…`.
    pub image: String,
    /// Canonical digest `sha256:…`.
    pub digest: String,
    /// RFC3339 timestamp when written.
    pub updated_at: String,
}

/// Paths for current/previous pins under a state directory.
#[derive(Debug, Clone)]
pub struct PinStore {
    /// Directory containing the JSON files.
    pub dir: PathBuf,
}

impl PinStore {
    /// Create a store rooted at `dir`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Path to `current.json`.
    #[must_use]
    pub fn current_path(&self) -> PathBuf {
        self.dir.join("current.json")
    }

    /// Path to `previous.json`.
    #[must_use]
    pub fn previous_path(&self) -> PathBuf {
        self.dir.join("previous.json")
    }
}

/// Load both pins (missing file → `None`).
///
/// # Errors
/// [`UpdaterError::PinStore`] on I/O or JSON errors for an existing file.
pub fn load_pins(store: &PinStore) -> Result<(Option<PinRecord>, Option<PinRecord>), UpdaterError> {
    Ok((
        read_optional(&store.current_path())?,
        read_optional(&store.previous_path())?,
    ))
}

/// Atomically write `current.json`.
///
/// # Errors
/// [`UpdaterError::PinStore`] on failure.
pub fn save_current(store: &PinStore, record: &PinRecord) -> Result<(), UpdaterError> {
    atomic_write(&store.current_path(), record)
}

/// Atomically write `previous.json`.
///
/// # Errors
/// [`UpdaterError::PinStore`] on failure.
pub fn save_previous(store: &PinStore, record: &PinRecord) -> Result<(), UpdaterError> {
    atomic_write(&store.previous_path(), record)
}

/// Commit a successful rollout: previous ← old current, current ← new.
///
/// # Errors
/// [`UpdaterError::PinStore`] on failure.
pub fn commit_pins(
    store: &PinStore,
    old_current: Option<&PinRecord>,
    new_current: &PinRecord,
) -> Result<(), UpdaterError> {
    fs::create_dir_all(&store.dir).map_err(|e| UpdaterError::PinStore(e.to_string()))?;
    if let Some(old) = old_current {
        save_previous(store, old)?;
    }
    save_current(store, new_current)?;
    Ok(())
}

fn read_optional(path: &Path) -> Result<Option<PinRecord>, UpdaterError> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|e| UpdaterError::PinStore(e.to_string()))?;
    let rec = serde_json::from_str(&text).map_err(|e| UpdaterError::PinStore(e.to_string()))?;
    Ok(Some(rec))
}

fn atomic_write(path: &Path, record: &PinRecord) -> Result<(), UpdaterError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| UpdaterError::PinStore(e.to_string()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let text =
        serde_json::to_string_pretty(record).map_err(|e| UpdaterError::PinStore(e.to_string()))?;
    fs::write(&tmp, text).map_err(|e| UpdaterError::PinStore(e.to_string()))?;
    fs::rename(&tmp, path).map_err(|e| UpdaterError::PinStore(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_current_previous() {
        let dir = tempdir().expect("tmp");
        let store = PinStore::new(dir.path());
        let cur = PinRecord {
            service: "validator".into(),
            image: "img@sha256:".to_owned() + &"aa".repeat(32),
            digest: "sha256:".to_owned() + &"aa".repeat(32),
            updated_at: "2026-07-30T00:00:00Z".into(),
        };
        save_current(&store, &cur).expect("save");
        let (c, p) = load_pins(&store).expect("load");
        assert_eq!(c.as_ref(), Some(&cur));
        assert!(p.is_none());
        commit_pins(
            &store,
            Some(&cur),
            &PinRecord {
                service: "validator".into(),
                image: "img@sha256:".to_owned() + &"bb".repeat(32),
                digest: "sha256:".to_owned() + &"bb".repeat(32),
                updated_at: "2026-07-30T00:01:00Z".into(),
            },
        )
        .expect("commit");
        let (c2, p2) = load_pins(&store).expect("load2");
        assert_eq!(
            p2.as_ref().map(|r| r.digest.as_str()),
            Some(cur.digest.as_str())
        );
        assert_eq!(
            c2.as_ref().map(|r| r.digest.as_str()),
            Some(&*format!("sha256:{}", "bb".repeat(32)))
        );
    }
}
