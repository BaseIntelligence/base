//! Diffstat + path classification for `AutoModel`-like trees.
//!
//! # Allowlist roots (`AutoModel` layout)
//!
//! Classification is prefix-based on paths *inside* the pin (after stripping
//! a leading `a/` / `b/` from unified-diff headers). Roots mirror
//! [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel):
//!
//! | Class     | Allowlist prefixes |
//! |-----------|--------------------|
//! | `arch`    | `nemo_automodel/components/models/`, `nemo_automodel/_transformers/`, `nemo_automodel/components/attention/`, `nemo_automodel/components/moe/`, `nemo_automodel/components/_peft/` |
//! | `trainer` | `nemo_automodel/components/training/`, `nemo_automodel/components/optim/`, `nemo_automodel/components/loss/`, `nemo_automodel/components/checkpoint/`, `nemo_automodel/recipes/`, `nemo_automodel/cli/` |
//! | `data`    | `nemo_automodel/components/datasets/` |
//! | `other`   | everything else under the pin (config, utils, examples, …) |
//!
//! Apply still accepts `other` paths mechanically; intake/anti-cheat may
//! hard-reject outside a narrower policy later. This module only labels.

use serde::{Deserialize, Serialize};

use crate::apply::{parse_file_hunks, ApplyError, ParsedFile};

/// Path classification for a touched `AutoModel` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathClass {
    /// Model / architecture surface.
    Arch,
    /// Trainer, optim, loss, checkpoint, recipes, CLI.
    Trainer,
    /// Dataset / dataloader surface.
    Data,
    /// Anything else under the pin.
    Other,
}

impl PathClass {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Arch => "arch",
            Self::Trainer => "trainer",
            Self::Data => "data",
            Self::Other => "other",
        }
    }
}

/// Prefixes classified as architecture (see module docs).
pub const ARCH_ROOTS: &[&str] = &[
    "nemo_automodel/components/models/",
    "nemo_automodel/_transformers/",
    "nemo_automodel/components/attention/",
    "nemo_automodel/components/moe/",
    "nemo_automodel/components/_peft/",
];

/// Prefixes classified as trainer / train-entry surface.
pub const TRAINER_ROOTS: &[&str] = &[
    "nemo_automodel/components/training/",
    "nemo_automodel/components/optim/",
    "nemo_automodel/components/loss/",
    "nemo_automodel/components/checkpoint/",
    "nemo_automodel/recipes/",
    "nemo_automodel/cli/",
];

/// Prefixes classified as data / dataloader surface.
pub const DATA_ROOTS: &[&str] = &["nemo_automodel/components/datasets/"];

/// Classify a pin-relative path into [`PathClass`].
#[must_use]
pub fn classify_path(path: &str) -> PathClass {
    let path = normalize_path(path);
    if ARCH_ROOTS.iter().any(|r| path.starts_with(r)) {
        return PathClass::Arch;
    }
    if TRAINER_ROOTS.iter().any(|r| path.starts_with(r)) {
        return PathClass::Trainer;
    }
    if DATA_ROOTS.iter().any(|r| path.starts_with(r)) {
        return PathClass::Data;
    }
    PathClass::Other
}

/// One file in a diffstat report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffStatEntry {
    pub path: String,
    pub added: usize,
    pub deleted: usize,
    pub class: PathClass,
}

/// Aggregate diffstat over a unified diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffStat {
    pub files: Vec<DiffStatEntry>,
    pub total_added: usize,
    pub total_deleted: usize,
}

/// Parse `patch` bytes into a classified diffstat (no filesystem I/O).
pub fn diffstat(patch: &[u8]) -> Result<DiffStat, ApplyError> {
    let files = parse_file_hunks(patch)?;
    Ok(diffstat_from_parsed(&files))
}

pub(crate) fn diffstat_from_parsed(files: &[ParsedFile]) -> DiffStat {
    let mut entries = Vec::with_capacity(files.len());
    let mut total_added = 0usize;
    let mut total_deleted = 0usize;
    for f in files {
        let mut added = 0usize;
        let mut deleted = 0usize;
        for hunk in &f.hunks {
            for line in &hunk.lines {
                match line.kind {
                    crate::apply::HunkLineKind::Add => added = added.saturating_add(1),
                    crate::apply::HunkLineKind::Delete => deleted = deleted.saturating_add(1),
                    crate::apply::HunkLineKind::Context => {}
                }
            }
        }
        total_added = total_added.saturating_add(added);
        total_deleted = total_deleted.saturating_add(deleted);
        entries.push(DiffStatEntry {
            path: f.path.clone(),
            added,
            deleted,
            class: classify_path(&f.path),
        });
    }
    DiffStat {
        files: entries,
        total_added,
        total_deleted,
    }
}

fn normalize_path(path: &str) -> String {
    let p = path.trim().trim_start_matches("./");
    let p = p
        .strip_prefix("a/")
        .or_else(|| p.strip_prefix("b/"))
        .unwrap_or(p);
    p.replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_automodel_roots() {
        assert_eq!(
            classify_path("nemo_automodel/components/models/toy/model.py"),
            PathClass::Arch
        );
        assert_eq!(
            classify_path("nemo_automodel/components/training/loop.py"),
            PathClass::Trainer
        );
        assert_eq!(
            classify_path("nemo_automodel/recipes/llm/train_ft.py"),
            PathClass::Trainer
        );
        assert_eq!(
            classify_path("nemo_automodel/components/datasets/loader.py"),
            PathClass::Data
        );
        assert_eq!(
            classify_path("nemo_automodel/components/utils/seed.py"),
            PathClass::Other
        );
    }

    #[test]
    fn diffstat_happy_fixture_patch() {
        let bytes = std::fs::read(crate::pin::fixture_happy_patch_path()).expect("read patch");
        let stat = diffstat(&bytes).expect("diffstat");
        assert_eq!(stat.files.len(), 2);
        assert!(stat.total_added > 0);
        assert!(stat.files.iter().all(|e| e.class == PathClass::Arch));
    }
}
