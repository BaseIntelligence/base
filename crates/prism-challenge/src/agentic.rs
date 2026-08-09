//! Workdir helpers for the Prism agentic anti-cheat gate.
//!
//! Corpus builders live in [`prism_pipeline::precheck`] (LOC split); this
//! module only materializes the review workdir.

use std::fs;
use std::path::Path;

use challenge_agentic::{CorpusEntry, ReviewRequest, PRISM_DOMAIN_RULES};
use prism_lium::{EvalReceipt, RemoteExecResult};
use prism_store::SubmissionState;

pub use prism_pipeline::{corpus_from_rows, gate_corpus_from_rows, same_miner};

/// Build a temp workdir + [`ReviewRequest`] for one Prism submission.
///
/// # Errors
/// IO / materialize failures.
pub fn build_review_request(
    workdir: &Path,
    row: &SubmissionState,
    metrics: Option<&RemoteExecResult>,
    receipt: Option<&EvalReceipt>,
    corpus: Vec<CorpusEntry>,
) -> Result<ReviewRequest, String> {
    let primary_relpaths = prism_recipe::materialize_review_sources(
        workdir,
        &row.architecture_py,
        &row.training_py,
        row.tree_blob.as_deref(),
    )?;
    let metrics_relpath = metrics
        .map(|m| {
            fs::write(
                workdir.join("metrics.json"),
                serde_json::to_vec(m).map_err(|e| format!("metrics json: {e}"))?,
            )
            .map_err(|e| format!("write metrics.json: {e}"))?;
            Ok::<_, String>("metrics.json".into())
        })
        .transpose()?;
    if let Some(r) = receipt {
        fs::write(
            workdir.join("receipt.json"),
            serde_json::to_vec(r).map_err(|e| format!("receipt json: {e}"))?,
        )
        .map_err(|e| format!("write receipt.json: {e}"))?;
    }
    Ok(ReviewRequest {
        workdir: workdir.to_path_buf(),
        primary_relpaths,
        corpus,
        metrics_relpath,
        pages_relpath: None,
        sanitize_report_relpath: None,
        domain_rules: PRISM_DOMAIN_RULES.into(),
    })
}
