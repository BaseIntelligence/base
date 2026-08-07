//! Workdir + corpus helpers for the Prism agentic anti-cheat gate.

use std::fs;
use std::path::Path;

use challenge_agentic::{
    same_miner_identity, CorpusEntry, GateCorpusEntry, ReviewRequest, PRISM_DOMAIN_RULES,
};
use prism_lium::{EvalReceipt, RemoteExecResult};
use prism_recipe::BASELINE_ARCHITECTURE_PY;
use prism_store::SubmissionState;

/// True when `other` is the same economic miner as `candidate` (hotkey or coldkey).
#[must_use]
pub fn same_miner(candidate: &SubmissionState, other: &SubmissionState) -> bool {
    same_miner_identity(
        &candidate.miner_hotkey,
        candidate.miner_coldkey.as_deref(),
        &other.miner_hotkey,
        other.miner_coldkey.as_deref(),
    )
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

/// Pre-LLM copy-gate corpus: other miners' prior art only (hotkey + coldkey).
#[must_use]
pub fn gate_corpus_from_rows(
    candidate: &SubmissionState,
    recent: &[SubmissionState],
) -> Vec<GateCorpusEntry> {
    recent
        .iter()
        .filter(|r| r.id != candidate.id && !same_miner(candidate, r))
        .map(|r| GateCorpusEntry {
            id: format!("subm:{}", r.id),
            source: r.architecture_py.clone(),
            created_at_ms: r.created_at_ms,
        })
        .collect()
}

/// Baseline + recent terminated submissions as agentic corpus entries.
///
/// Corpus entries are **architecture.py only** (similarity v2): `training.py`
/// is exempt from every copy/similarity comparison — the same training
/// script on two different architectures is legitimate competition behavior.
/// `exempt_arch` drops entries byte-equal to that source (training-only
/// submissions on a registry architecture: the identity is by design).
/// Same-hotkey and same-coldkey prior art are excluded.
#[must_use]
pub fn corpus_from_rows(
    candidate: &SubmissionState,
    recent: &[SubmissionState],
    exempt_arch: Option<&str>,
) -> Vec<CorpusEntry> {
    let mut v = vec![CorpusEntry {
        id: "baseline".into(),
        source: BASELINE_ARCHITECTURE_PY.into(),
    }];
    for r in recent {
        if r.id == candidate.id || same_miner(candidate, r) {
            continue;
        }
        if Some(r.architecture_py.as_str()) == exempt_arch {
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

#[cfg(test)]
mod tests {
    use super::*;
    use prism_store::{FinalScore, Stage};

    fn row(
        id: &str,
        hotkey: &str,
        coldkey: Option<&str>,
        arch: &str,
        created_at_ms: u64,
    ) -> SubmissionState {
        SubmissionState {
            id: id.into(),
            miner_hotkey: hotkey.into(),
            miner_coldkey: coldkey.map(str::to_owned),
            epoch: 1,
            netuid: 1,
            status: Stage::Terminated,
            architecture_py: arch.into(),
            training_py: "train".into(),
            label: None,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: None,
            bpb: Some(1.0),
            arch_id: None,
            review: None,
            similarity: None,
            final_score: Some(FinalScore::Score(1)),
            retry_count: 0,
            error_detail: None,
            created_at_ms,
            updated_at_ms: created_at_ms,
        }
    }

    #[test]
    fn same_hotkey_prior_art_excluded() {
        let prior = row("aaaaaaaa", "aa", Some("11"), "arch_a", 1_000);
        let next = row("bbbbbbbb", "aa", Some("11"), "arch_b", 2_000);
        let recent = vec![prior, next.clone()];
        assert!(gate_corpus_from_rows(&next, &recent).is_empty());
        assert_eq!(
            corpus_from_rows(&next, &recent, None)
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>(),
            vec!["baseline"]
        );
    }

    #[test]
    fn same_coldkey_different_hotkey_excluded() {
        let prior = row("aaaaaaaa", "aa", Some("11"), "arch_a", 1_000);
        let next = row("bbbbbbbb", "bb", Some("11"), "arch_b", 2_000);
        let recent = vec![prior, next.clone()];
        assert!(gate_corpus_from_rows(&next, &recent).is_empty());
        assert_eq!(
            corpus_from_rows(&next, &recent, None)
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>(),
            vec!["baseline"]
        );
    }

    #[test]
    fn different_coldkey_stays_in_corpus() {
        let victim = row("aaaaaaaa", "aa", Some("11"), "arch_a", 1_000);
        let copier = row("bbbbbbbb", "bb", Some("22"), "arch_b", 2_000);
        let recent = vec![victim, copier.clone()];
        assert_eq!(
            gate_corpus_from_rows(&copier, &recent)
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>(),
            vec!["subm:aaaaaaaa"]
        );
        assert_eq!(
            corpus_from_rows(&copier, &recent, None)
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>(),
            vec!["baseline", "subm:aaaaaaaa"]
        );
    }
}
