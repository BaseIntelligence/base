//! Materialize miner sources for the agentic reviewer.

use std::fs;
use std::path::Path;

/// Write review inputs into `workdir` and return the primary relative paths
/// the agentic gate should read.
///
/// - Recipe **2.0** (`tree_blob` with `.prism/automodel.patch`): delta-only —
///   unified diff, diffstat, scrutiny brief, and **touched** files (not the
///   whole `AutoModel` pin tree).
/// - Legacy source-tree `tree_blob`: full validated tree.
/// - Else: two seam files (`architecture.py` / `training.py`).
///
/// # Errors
/// Unpack / filesystem failures.
pub fn materialize_review_sources(
    workdir: &Path,
    architecture_py: &str,
    training_py: &str,
    tree_blob: Option<&[u8]>,
) -> Result<Vec<String>, String> {
    if let Some(blob) = tree_blob {
        let tree = prism_tree::StagedTree::unpack(blob).map_err(|e| e.to_string())?;
        if prism_automodel::files_have_automodel_diff(tree.files()) {
            return prism_automodel::materialize_delta_review(workdir, tree.files());
        }
        let mut paths = tree.materialize(workdir).map_err(|e| e.to_string())?;
        let manifest =
            serde_json::to_vec_pretty(&tree.manifest()).map_err(|e| format!("manifest: {e}"))?;
        fs::write(workdir.join("tree_manifest.json"), manifest)
            .map_err(|e| format!("write tree_manifest.json: {e}"))?;
        paths.push("tree_manifest.json".into());
        return Ok(paths);
    }
    fs::write(workdir.join("architecture.py"), architecture_py)
        .map_err(|e| format!("write architecture.py: {e}"))?;
    fs::write(workdir.join("training.py"), training_py)
        .map_err(|e| format!("write training.py: {e}"))?;
    Ok(vec!["architecture.py".into(), "training.py".into()])
}
