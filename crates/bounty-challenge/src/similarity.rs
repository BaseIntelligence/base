//! Agentic similar-24h review (OpenRouter / Sim).

use std::path::Path;

use async_trait::async_trait;
use challenge_agentic::{
    AgenticBackend, AgenticError, AgenticVerdict, CheatCode, CorpusEntry, ReviewRequest,
    VerdictKind,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use bounty_store::BugRow;

/// Default OpenRouter model (`BOUNTY_OPENROUTER_MODEL` override).
pub const DEFAULT_OPENROUTER_MODEL: &str = "deepseek/deepseek-v4-flash";

/// Domain rules appended to the agentic system prompt.
pub const BOUNTY_DOMAIN_RULES: &str = r"
Bounty domain (video bug reports):
- Primaries are text reports (title/description/steps), not Python harnesses.
- Verdict Clean means the report is NOVEL vs the 24h corpus → pending_admin.
- Verdict Cheat or Suspicious with near_identical / high similarity means DUPLICATE
  → reject (Score 0). Prefer cheat_code near_identical_harness_copy for copies.
- Compare semantic content of the bug report, not video pixels.
- Fail closed: if unsure whether novel, prefer suspicious/cheat over clean.
";

/// Structured similarity verdict for the bounty pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityKind {
    /// Novel relative to 24h corpus.
    Novel,
    /// Duplicate / near-duplicate of `nearest_id`.
    Duplicate,
}

/// Verdict after agentic review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityVerdict {
    /// novel | duplicate.
    pub kind: SimilarityKind,
    /// Nearest corpus bug id when duplicate.
    pub nearest_id: Option<String>,
    /// Similarity in basis points (0..=10000).
    pub similarity_bps: u16,
    /// Short rationale.
    pub rationale: String,
}

/// Similarity failures (fail-closed → retry / failed, never invent novel).
#[derive(Debug, Error)]
pub enum SimilarityError {
    /// Agentic backend fault.
    #[error("agentic: {0}")]
    Agentic(#[from] AgenticError),
    /// Staging IO.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Canonical report text used for hash / agentic primary.
#[must_use]
pub fn report_text(bug: &BugRow) -> String {
    format!(
        "app_id: {}\ntitle: {}\ndescription: {}\nsteps: {}\n",
        bug.app_id,
        bug.title,
        bug.description,
        bug.steps.as_deref().unwrap_or("")
    )
}

fn report_hash(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    hex::encode(h.finalize())
}

/// Deterministic offline similarity (CI / `BOUNTY_FORCE_SIM`).
///
/// Exact report-text hash match → duplicate; otherwise novel. Does **not** use
/// Python AST heuristics (those false-positive on prose bug reports).
#[derive(Debug, Default)]
pub struct BountySimAgent;

impl BountySimAgent {
    /// New sim backend.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AgenticBackend for BountySimAgent {
    async fn review(&self, req: &ReviewRequest) -> Result<AgenticVerdict, AgenticError> {
        let primary = req
            .primary_relpaths
            .first()
            .ok_or_else(|| AgenticError::NoVerdict("no primary".into()))?;
        let path = req.workdir.join(primary);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| AgenticError::Tool(format!("read primary: {e}")))?;
        let cand = report_hash(text.trim());
        for entry in &req.corpus {
            if report_hash(entry.source.trim()) == cand {
                return Ok(AgenticVerdict {
                    verdict: VerdictKind::Cheat,
                    cheat_codes: vec![CheatCode::NearIdenticalHarnessCopy],
                    nearest_id: Some(entry.id.clone()),
                    similarity_bps: 10_000,
                    rationale: "sim: exact report hash match".into(),
                });
            }
        }
        Ok(AgenticVerdict {
            verdict: VerdictKind::Clean,
            cheat_codes: vec![],
            nearest_id: None,
            similarity_bps: 0,
            rationale: "sim: no exact report match in 24h corpus".into(),
        })
    }
}

/// Map agentic anti-cheat verdict → bounty novel/duplicate.
#[must_use]
pub fn map_agentic_verdict(v: &AgenticVerdict) -> SimilarityVerdict {
    match v.verdict {
        VerdictKind::Clean => SimilarityVerdict {
            kind: SimilarityKind::Novel,
            nearest_id: v.nearest_id.clone(),
            similarity_bps: v.similarity_bps,
            rationale: v.rationale.clone(),
        },
        VerdictKind::Suspicious | VerdictKind::Cheat => SimilarityVerdict {
            kind: SimilarityKind::Duplicate,
            nearest_id: v.nearest_id.clone(),
            similarity_bps: v.similarity_bps,
            rationale: v.rationale.clone(),
        },
    }
}

/// Stage candidate + corpus text under `workdir` and run agentic review.
///
/// # Errors
/// IO / agentic fail-closed errors.
pub async fn review_similarity(
    agent: &dyn AgenticBackend,
    bug: &BugRow,
    corpus: &[BugRow],
    workdir: &Path,
) -> Result<SimilarityVerdict, SimilarityError> {
    tokio::fs::create_dir_all(workdir).await?;
    let primary = "report.txt";
    tokio::fs::write(workdir.join(primary), report_text(bug)).await?;
    let corpus_entries: Vec<CorpusEntry> = corpus
        .iter()
        .map(|c| CorpusEntry {
            id: c.id.clone(),
            source: report_text(c),
        })
        .collect();
    let req = ReviewRequest {
        workdir: workdir.to_path_buf(),
        primary_relpaths: vec![primary.into()],
        corpus: corpus_entries,
        metrics_relpath: None,
        pages_relpath: None,
        sanitize_report_relpath: None,
        domain_rules: BOUNTY_DOMAIN_RULES.into(),
    };
    let verdict = agent.review(&req).await?;
    Ok(map_agentic_verdict(&verdict))
}

/// Resolve OpenRouter model from env (default [`DEFAULT_OPENROUTER_MODEL`]).
#[must_use]
pub fn openrouter_model() -> String {
    std::env::var("BOUNTY_OPENROUTER_MODEL")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_OPENROUTER_MODEL.to_owned())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use bounty_store::BugStatus;
    fn bug(id: &str, title: &str, desc: &str) -> BugRow {
        BugRow {
            id: id.into(),
            miner_hotkey: "aa".repeat(32),
            miner_coldkey: None,
            app_id: "demo".into(),
            title: title.into(),
            description: desc.into(),
            steps: None,
            status: BugStatus::PendingAdmin,
            agentic_verdict: None,
            nearest_id: None,
            video_sha256: None,
            video_bytes: None,
            video_path: None,
            reject_reason: None,
            epoch: 1,
            created_at_ms: 1,
            updated_at_ms: 1,
        }
    }

    #[tokio::test]
    async fn sim_detects_duplicate_report() {
        let dir = tempfile::tempdir().unwrap();
        let cand = bug("c1", "crash", "null deref on save");
        let prior = bug("p1", "crash", "null deref on save");
        let v = review_similarity(&BountySimAgent::new(), &cand, &[prior], dir.path())
            .await
            .unwrap();
        assert_eq!(v.kind, SimilarityKind::Duplicate);
        assert_eq!(v.nearest_id.as_deref(), Some("p1"));
    }

    #[tokio::test]
    async fn sim_novel_when_distinct() {
        let dir = tempfile::tempdir().unwrap();
        let cand = bug("c1", "crash A", "unique bug one");
        let prior = bug("p1", "crash B", "totally different");
        let v = review_similarity(&BountySimAgent::new(), &cand, &[prior], dir.path())
            .await
            .unwrap();
        assert_eq!(v.kind, SimilarityKind::Novel);
    }
}
