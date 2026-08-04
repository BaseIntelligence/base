//! Workdir + corpus helpers for the Prism agentic anti-cheat gate.

use std::fs;
use std::path::Path;

use challenge_agentic::{CorpusEntry, ReviewRequest, PRISM_DOMAIN_RULES};
use prism_lium::{EvalReceipt, RemoteExecResult};
use prism_recipe::{BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY};
use prism_store::SubmissionState;

/// Join architecture + training the same way [`challenge_agentic::SimAgent`] fingerprints.
#[must_use]
pub fn join_sources(architecture_py: &str, training_py: &str) -> String {
    format!("{architecture_py}\n#--\n{training_py}")
}

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

/// Baseline + recent terminated submissions as agentic corpus entries.
#[must_use]
pub fn corpus_from_rows(
    current_id: &str,
    recent: &[prism_store::SubmissionState],
) -> Vec<CorpusEntry> {
    let mut v = vec![CorpusEntry {
        id: "baseline".into(),
        source: join_sources(BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY),
    }];
    for r in recent {
        if r.id == current_id {
            continue;
        }
        let label = if r.id.len() >= 8 {
            format!("subm:{}", &r.id[..8])
        } else {
            format!("subm:{}", r.id)
        };
        v.push(CorpusEntry {
            id: label,
            source: join_sources(&r.architecture_py, &r.training_py),
        });
    }
    v
}
