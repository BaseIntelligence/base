//! Diff-aware review surface for recipe 2.0 (`AutoModel` miner deltas).
//!
//! Agentic / copy / static screens should focus on the **unified diff** and
//! **touched files**, not the whole pin tree. Trainer / data path edits are
//! flagged for higher scrutiny (policy); hard harness gates stay elsewhere.

use std::collections::BTreeMap;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::classify::{classify_path, DiffStat, PathClass};
use crate::intake::{diff_from_files, META_DIFFSTAT, META_PATCH};

/// Relpath written into the agentic workdir (primary).
pub const REVIEW_PATCH_REL: &str = ".prism/automodel.patch";
/// Classified diffstat JSON in the agentic workdir.
pub const REVIEW_DIFFSTAT_REL: &str = ".prism/diffstat.json";
/// Operator briefing (scrutiny flags + cheat emphasis).
pub const REVIEW_BRIEF_REL: &str = ".prism/review_brief.md";

/// Copy / similarity / agentic surface derived from a materialized delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaReviewSurface {
    /// Raw unified-diff text.
    pub patch_text: String,
    /// Classified diffstat.
    pub diffstat: DiffStat,
    /// SHA-256 (hex) of patch bytes — patch fingerprint.
    pub patch_sha256: String,
    /// Concatenated post-apply touched `.py` bodies (AST / copy-gate source).
    pub touched_py: String,
    /// Store field: patch fingerprint header + touched Python (copy / similarity).
    pub architecture_py: String,
    /// Trainer/data touched Python only (empty when none); not used for telemetry.
    pub training_py: String,
    /// True when any touched path is [`PathClass::Trainer`] or [`PathClass::Data`].
    pub trainer_data_scrutiny: bool,
    /// Human briefing for the agentic workdir.
    pub brief_markdown: String,
}

/// Build a review surface from a packed-tree file map (includes `.prism/` meta).
#[must_use]
pub fn delta_surface_from_files(files: &BTreeMap<String, Vec<u8>>) -> Option<DeltaReviewSurface> {
    let (patch_text, diffstat) = diff_from_files(files)?;
    Some(build_surface(&patch_text, &diffstat, files))
}

/// Build a review surface from an already-materialized apply result.
#[must_use]
pub fn delta_surface_from_materialized(
    patch_text: &str,
    diffstat: &DiffStat,
    files: &BTreeMap<String, Vec<u8>>,
) -> DeltaReviewSurface {
    build_surface(patch_text, diffstat, files)
}

/// SHA-256 hex of raw patch bytes (patch fingerprint).
#[must_use]
pub fn patch_sha256_hex(patch: &[u8]) -> String {
    hex::encode(Sha256::digest(patch))
}

/// True when `files` carry recipe 2.0 diff artifacts.
#[must_use]
pub fn files_have_automodel_diff(files: &BTreeMap<String, Vec<u8>>) -> bool {
    files.contains_key(META_PATCH) && files.contains_key(META_DIFFSTAT)
}

/// Extract unified-diff text from a packed `tree_blob`, if present.
#[must_use]
pub fn patch_text_from_tree_blob(tree_blob: &[u8]) -> Option<String> {
    let tree = prism_tree::StagedTree::unpack(tree_blob).ok()?;
    let (patch, _) = diff_from_files(tree.files())?;
    Some(patch)
}

/// Materialize **delta-only** review inputs into `workdir`.
///
/// Writes patch + diffstat + brief + touched files (not the full `AutoModel` tree).
/// Returns primary relative paths for the agentic gate (patch first).
///
/// # Errors
/// Filesystem failures.
pub fn materialize_delta_review(
    workdir: &Path,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<String>, String> {
    let surface = delta_surface_from_files(files)
        .ok_or_else(|| "tree lacks .prism/automodel.patch + diffstat".to_owned())?;
    std::fs::create_dir_all(workdir.join(".prism")).map_err(|e| format!("mkdir .prism: {e}"))?;
    std::fs::write(
        workdir.join(REVIEW_PATCH_REL),
        surface.patch_text.as_bytes(),
    )
    .map_err(|e| format!("write patch: {e}"))?;
    let diffstat_json =
        serde_json::to_vec_pretty(&surface.diffstat).map_err(|e| format!("diffstat json: {e}"))?;
    std::fs::write(workdir.join(REVIEW_DIFFSTAT_REL), diffstat_json)
        .map_err(|e| format!("write diffstat: {e}"))?;
    std::fs::write(
        workdir.join(REVIEW_BRIEF_REL),
        surface.brief_markdown.as_bytes(),
    )
    .map_err(|e| format!("write brief: {e}"))?;

    let mut primaries = vec![
        REVIEW_PATCH_REL.to_owned(),
        REVIEW_DIFFSTAT_REL.to_owned(),
        REVIEW_BRIEF_REL.to_owned(),
    ];
    for entry in &surface.diffstat.files {
        let path = entry.path.as_str();
        let Some(bytes) = files.get(path) else {
            continue;
        };
        let dest = workdir.join(path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        std::fs::write(&dest, bytes).map_err(|e| format!("write {path}: {e}"))?;
        primaries.push(path.to_owned());
    }
    Ok(primaries)
}

fn build_surface(
    patch_text: &str,
    diffstat: &DiffStat,
    files: &BTreeMap<String, Vec<u8>>,
) -> DeltaReviewSurface {
    let patch_sha256 = patch_sha256_hex(patch_text.as_bytes());
    let touched_py = concat_touched_py(files, diffstat, None);
    let training_py = concat_touched_py(
        files,
        diffstat,
        Some(&[PathClass::Trainer, PathClass::Data]),
    );
    let trainer_data_scrutiny = diffstat.files.iter().any(|e| {
        matches!(e.class, PathClass::Trainer | PathClass::Data)
            || matches!(classify_path(&e.path), PathClass::Trainer | PathClass::Data)
    });
    let architecture_py = if touched_py.trim().is_empty() {
        format!(
            "# prism-automodel-copy patch_sha256={patch_sha256}\n# (no touched .py — patch fingerprint only)\n{patch_text}"
        )
    } else {
        format!("# prism-automodel-copy patch_sha256={patch_sha256}\n{touched_py}")
    };
    let brief_markdown = build_brief(diffstat, &patch_sha256, trainer_data_scrutiny);
    DeltaReviewSurface {
        patch_text: patch_text.to_owned(),
        diffstat: diffstat.clone(),
        patch_sha256,
        touched_py,
        architecture_py,
        training_py,
        trainer_data_scrutiny,
        brief_markdown,
    }
}

fn concat_touched_py(
    files: &BTreeMap<String, Vec<u8>>,
    diffstat: &DiffStat,
    class_filter: Option<&[PathClass]>,
) -> String {
    let mut out = String::new();
    for entry in &diffstat.files {
        if !std::path::Path::new(&entry.path)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("py"))
        {
            continue;
        }
        if let Some(filter) = class_filter {
            if !filter.contains(&entry.class) {
                continue;
            }
        }
        let Some(bytes) = files.get(&entry.path) else {
            continue;
        };
        let Ok(text) = std::str::from_utf8(bytes) else {
            continue;
        };
        out.push_str("# --- ");
        out.push_str(&entry.path);
        out.push_str(" (");
        out.push_str(entry.class.as_str());
        out.push_str(") ---\n");
        out.push_str(text);
        if !text.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

fn build_brief(diffstat: &DiffStat, patch_sha256: &str, trainer_data_scrutiny: bool) -> String {
    let mut lines = vec![
        "# Prism AutoModel delta review".to_owned(),
        String::new(),
        format!("- patch_sha256: `{patch_sha256}`"),
        format!(
            "- files touched: {} (+{}/-{})",
            diffstat.files.len(),
            diffstat.total_added,
            diffstat.total_deleted
        ),
        String::new(),
        "## Path classes".to_owned(),
    ];
    for e in &diffstat.files {
        lines.push(format!(
            "- `{}` — **{}** (+{}/-{})",
            e.path,
            e.class.as_str(),
            e.added,
            e.deleted
        ));
    }
    lines.push(String::new());
    if trainer_data_scrutiny {
        lines.push("## Scrutiny: trainer / data paths edited".to_owned());
        lines.push(
            "Edits under trainer / dataloader / loss / checkpoint / recipes require **higher scrutiny**."
                .into(),
        );
        lines.push(
            "Look for eval-set leakage, telemetry disable/mock, network/exfil, causal-label leaks in the **delta**."
                .into(),
        );
    } else {
        lines.push("## Scrutiny".to_owned());
        lines.push(
            "No trainer/data-class paths in this delta; still scan the patch for telemetry/network/eval leaks."
                .into(),
        );
    }
    lines.push(String::new());
    lines.push("## Primary surface".to_owned());
    lines.push(
        "Review **diff hunks** + touched files listed above — not the whole AutoModel pin tree."
            .into(),
    );
    lines.push(
        "Hard harness gates (netns, telemetry hooks, budget, causal screen) remain enforced outside this brief."
            .into(),
    );
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::env::with_fixture_env;
    use crate::intake::{fixture_automodel_zip, intake_automodel_zip};

    #[test]
    fn happy_fixture_surface_is_arch_focused() {
        with_fixture_env(true, || {
            let zip = fixture_automodel_zip().unwrap();
            let mat = intake_automodel_zip(&zip).unwrap();
            let surface =
                delta_surface_from_materialized(&mat.patch_text, &mat.diffstat, &mat.files);
            assert!(!surface.patch_sha256.is_empty());
            assert!(surface.architecture_py.contains("patch_sha256="));
            assert!(
                surface.architecture_py.contains("ToyModel")
                    || surface.touched_py.contains("scale")
            );
            assert!(!surface.trainer_data_scrutiny);
            assert!(surface.brief_markdown.contains("Primary surface"));
        });
    }

    #[test]
    fn materialize_delta_writes_patch_not_full_tree() {
        with_fixture_env(true, || {
            let zip = fixture_automodel_zip().unwrap();
            let mat = intake_automodel_zip(&zip).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let primaries = materialize_delta_review(dir.path(), &mat.files).unwrap();
            assert!(primaries.iter().any(|p| p == REVIEW_PATCH_REL));
            assert!(dir.path().join(REVIEW_PATCH_REL).is_file());
            assert!(dir.path().join(REVIEW_BRIEF_REL).is_file());
            // Touched model files present; unrelated pin files absent.
            assert!(dir
                .path()
                .join("nemo_automodel/components/models/toy/model.py")
                .is_file());
            assert!(!dir
                .path()
                .join("nemo_automodel/components/datasets/loader.py")
                .is_file());
        });
    }

    #[test]
    fn trainer_touch_sets_scrutiny() {
        let mut files = BTreeMap::new();
        let patch = "\
diff --git a/nemo_automodel/components/training/loop.py b/nemo_automodel/components/training/loop.py
--- a/nemo_automodel/components/training/loop.py
+++ b/nemo_automodel/components/training/loop.py
@@ -1,1 +1,2 @@
 # loop
+print('x')
";
        let diffstat = crate::classify::diffstat(patch.as_bytes()).unwrap();
        files.insert(META_PATCH.into(), patch.as_bytes().to_vec());
        files.insert(META_DIFFSTAT.into(), serde_json::to_vec(&diffstat).unwrap());
        files.insert(
            "nemo_automodel/components/training/loop.py".into(),
            b"# loop\nprint('x')\n".to_vec(),
        );
        let surface = delta_surface_from_files(&files).unwrap();
        assert!(surface.trainer_data_scrutiny);
        assert!(surface.brief_markdown.contains("trainer / data"));
        assert!(surface.training_py.contains("print('x')"));
    }
}
