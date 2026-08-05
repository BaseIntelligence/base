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
///
/// Corpus entries are **architecture.py only** (similarity v2): `training.py`
/// is exempt from every copy/similarity comparison — the same training
/// script on two different architectures is legitimate competition behavior.
/// `exempt_arch` drops entries byte-equal to that source (training-only
/// submissions on a registry architecture: the identity is by design).
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
        let label = if r.id.len() >= 8 {
            format!("subm:{}", &r.id[..8])
        } else {
            format!("subm:{}", r.id)
        };
        v.push(CorpusEntry {
            id: label,
            source: r.architecture_py.clone(),
        });
    }
    v
}
