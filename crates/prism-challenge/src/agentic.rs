//! Workdir + corpus helpers for the Prism agentic anti-cheat gate.

use std::fs;
use std::path::Path;

use challenge_agentic::{CorpusEntry, ReviewRequest, PRISM_DOMAIN_RULES};
use prism_lium::{EvalReceipt, RemoteExecResult};
use prism_recipe::BASELINE_ARCHITECTURE_PY;
use prism_store::SubmissionState;

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

/// Architecture-only corpus (similarity v2); `exempt_arch` drops identities.
#[must_use]
pub fn corpus_from_rows(
    current_id: &str,
    recent: &[prism_store::SubmissionState],
    exempt_arch: Option<&str>,
) -> Vec<CorpusEntry> {
    let mut v = vec![CorpusEntry {
        id: "baseline".into(),
        source: BASELINE_ARCHITECTURE_PY.into(),
    }];
    for r in recent {
        if r.id == current_id || Some(r.architecture_py.as_str()) == exempt_arch {
            continue;
        }
        v.push(CorpusEntry {
            id: format!("subm:{}", &r.id[..r.id.len().min(8)]),
            source: r.architecture_py.clone(),
        });
    }
    v
}
