//! Materialize miner sources for the agentic reviewer.

use std::fs;
use std::path::Path;

/// Write review inputs into `workdir` and return the primary relative paths
/// the agentic gate should read.
///
/// When `tree_blob` is present the whole validated tree is materialized
/// (helpers, kernels, `tokenizer/`, …). Otherwise the legacy two seam files
/// are written.
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
