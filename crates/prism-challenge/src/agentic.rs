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
/// IO failures writing sources / metrics / receipt.
pub fn build_review_request(
    workdir: &Path,
    row: &SubmissionState,
    metrics: Option<&RemoteExecResult>,
    receipt: Option<&EvalReceipt>,
    corpus: Vec<CorpusEntry>,
) -> Result<ReviewRequest, String> {
    fs::write(workdir.join("architecture.py"), &row.architecture_py)
        .map_err(|e| format!("write architecture.py: {e}"))?;
    fs::write(workdir.join("training.py"), &row.training_py)
        .map_err(|e| format!("write training.py: {e}"))?;
    let metrics_relpath = if let Some(m) = metrics {
        let path = workdir.join("metrics.json");
        let body = serde_json::to_vec(m).map_err(|e| format!("metrics json: {e}"))?;
        fs::write(&path, body).map_err(|e| format!("write metrics.json: {e}"))?;
        Some("metrics.json".into())
    } else {
        None
    };
    if let Some(r) = receipt {
        let body = serde_json::to_vec(r).map_err(|e| format!("receipt json: {e}"))?;
        fs::write(workdir.join("receipt.json"), body)
            .map_err(|e| format!("write receipt.json: {e}"))?;
    }
    Ok(ReviewRequest {
        workdir: workdir.to_path_buf(),
        primary_relpaths: vec!["architecture.py".into(), "training.py".into()],
        corpus,
        metrics_relpath,
        pages_relpath: None,
        sanitize_report_relpath: None,
        domain_rules: PRISM_DOMAIN_RULES.into(),
    })
}
