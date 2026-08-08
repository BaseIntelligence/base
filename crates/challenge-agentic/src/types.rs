//! Shared request / verdict types.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// `OpenRouter` chat API base.
pub const OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";

/// Failures — all fail closed for scoring (no invented clean).
#[derive(Debug, thiserror::Error)]
pub enum AgenticError {
    /// HTTP / timeout.
    #[error("transport: {0}")]
    Transport(String),
    /// Provider non-success.
    #[error("provider: {0}")]
    Provider(String),
    /// Schema / JSON parse failure.
    #[error("parse: {0}")]
    Parse(String),
    /// Loop ended without a parseable `submit_verdict`.
    #[error("no verdict: {0}")]
    NoVerdict(String),
    /// Tool execution / sandbox violation.
    #[error("tool: {0}")]
    Tool(String),
    /// Missing API key.
    #[error("no llm key configured")]
    NoKey,
}

/// Final anti-cheat class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictKind {
    /// Eligible for scoring / admin selection.
    Clean,
    /// Policy: treat as Score(0); rationale kept for admin.
    Suspicious,
    /// Hard Score(0).
    Cheat,
}

/// Normative cheat taxonomy (Design §4 + Prism §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheatCode {
    /// Near-identical harness / submission vs corpus.
    NearIdenticalHarnessCopy,
    /// Thin wrapper republishing another miner's HTML/output.
    TrivialRepublishWrapper,
    /// Fetch + republish identifiable real site with no substance.
    ScrapedSiteClone,
    /// Sanitize bypass / JS exfil / phishing reinjection.
    SanitizeBypass,
    /// Obfuscation whose only purpose is hiding a copy.
    ObfuscationToHideCopy,
    /// Prism metrics inconsistent with tokens/time/receipt.
    InconsistentMetrics,
    /// Harness short-circuits eval / hardcodes `METRICS_JSON`.
    EvalShortCircuit,
    /// AST copy of another miner's architecture/training.
    AstArchitectureCopy,
    /// Prism `training.py` missing `prism_telemetry.report` / `finish_evaluation`.
    MissingTelemetryHooks,
    /// Non-causal time-axis mix (`TokenMix` / dense `Linear`) ⇒ label leak.
    NonCausalLabelLeak,
}

/// One prior corpus submission for AST nearest-neighbor tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusEntry {
    /// Stable id (`baseline`, `harness:<hex>`, …).
    pub id: String,
    /// Full Python source text (caller loads from store).
    pub source: String,
}

/// Input for one agentic review.
#[derive(Debug, Clone)]
pub struct ReviewRequest {
    /// Absolute workdir root; tools refuse path escapes.
    pub workdir: PathBuf,
    /// Relative Python paths under `workdir` to prioritize (e.g. `agent.py`).
    pub primary_relpaths: Vec<String>,
    /// Historical / baseline corpus for AST nearest.
    pub corpus: Vec<CorpusEntry>,
    /// Optional relative metrics JSON (Prism).
    pub metrics_relpath: Option<String>,
    /// Optional relative pages directory (Design sanitized output).
    pub pages_relpath: Option<String>,
    /// Optional relative sanitize report JSON.
    pub sanitize_report_relpath: Option<String>,
    /// Extra domain rules appended to the system prompt (Design vs Prism).
    pub domain_rules: String,
}

/// Structured final verdict from `submit_verdict`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgenticVerdict {
    /// clean | suspicious | cheat.
    pub verdict: VerdictKind,
    /// Zero or more cheat codes (empty when clean).
    pub cheat_codes: Vec<CheatCode>,
    /// Nearest corpus id when applicable.
    pub nearest_id: Option<String>,
    /// Integer similarity basis points 0..=10000.
    pub similarity_bps: u16,
    /// Audit rationale (truncated on ingest).
    pub rationale: String,
}

/// Serializable request mirror for the containerized reviewer
/// (`challenge-review` bin reads this from `/work/_review_request.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerReviewRequest {
    /// Relative Python paths under `/work` to prioritize.
    pub primary_relpaths: Vec<String>,
    /// Historical / baseline corpus for AST nearest.
    pub corpus: Vec<CorpusEntry>,
    /// Optional relative metrics JSON (Prism).
    pub metrics_relpath: Option<String>,
    /// Optional relative pages directory (Design sanitized output).
    pub pages_relpath: Option<String>,
    /// Optional relative sanitize report JSON.
    pub sanitize_report_relpath: Option<String>,
    /// Extra domain rules appended to the system prompt.
    pub domain_rules: String,
}

impl ContainerReviewRequest {
    /// Rebuild the full request with the in-container workdir (`/work`).
    #[must_use]
    pub fn into_request(self, workdir: PathBuf) -> ReviewRequest {
        ReviewRequest {
            workdir,
            primary_relpaths: self.primary_relpaths,
            corpus: self.corpus,
            metrics_relpath: self.metrics_relpath,
            pages_relpath: self.pages_relpath,
            sanitize_report_relpath: self.sanitize_report_relpath,
            domain_rules: self.domain_rules,
        }
    }
}

impl ReviewRequest {
    /// Strip the host workdir for container transport.
    #[must_use]
    pub fn to_container(&self) -> ContainerReviewRequest {
        ContainerReviewRequest {
            primary_relpaths: self.primary_relpaths.clone(),
            corpus: self.corpus.clone(),
            metrics_relpath: self.metrics_relpath.clone(),
            pages_relpath: self.pages_relpath.clone(),
            sanitize_report_relpath: self.sanitize_report_relpath.clone(),
            domain_rules: self.domain_rules.clone(),
        }
    }
}

/// One code review pass producing a fail-closed verdict.
#[async_trait]
pub trait AgenticBackend: Send + Sync {
    /// Run tools (+ LLM or sim heuristics) and return a verdict.
    ///
    /// # Errors
    /// Transport / parse / missing verdict — never invents `clean`.
    async fn review(&self, req: &ReviewRequest) -> Result<AgenticVerdict, AgenticError>;
}
