//! Shared review types + errors.

/// `OpenRouter` chat API base.
pub const OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";

/// Cost cap hint per LLM call group (documented, not enforced server-side).
pub const MAX_REVIEW_TOKENS: u32 = 1_200;
/// Max corpus snippets injected into a similarity prompt.
pub const MAX_CORPUS_SNIPPETS: usize = 6;
/// Per-source byte cap injected into prompts (truncate, never drop mid-token
/// boundary beyond this).
pub const MAX_SOURCE_BYTES_IN_PROMPT: usize = 24 * 1024;

/// Errors: transport, provider, parse. All fail closed for scoring.
#[derive(Debug, thiserror::Error)]
pub enum ReviewError {
    /// HTTP/timeout.
    #[error("transport: {0}")]
    Transport(String),
    /// Provider returned error / non-200.
    #[error("provider: {0}")]
    Provider(String),
    /// Answer did not conform (parse failed / schema mismatch).
    #[error("parse: {0}")]
    Parse(String),
    /// Key missing in sim-required-live mode.
    #[error("no llm key configured")]
    NoKey,
}

/// Structured quality verdict (`prompts/review_v1.md`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewVerdict {
    /// Quality score 0..1000 (multiplied into lattice part).
    pub quality_score: u16,
    /// Freeform issue list (≤ 5).
    pub issues: Vec<String>,
    /// Prompt version served.
    pub prompt_version: &'static str,
}

/// Similarity class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityKind {
    /// Novel relative to corpus.
    Original,
    /// Overlapping style/structure; warn-only.
    Suspicious,
    /// Near-copy of a corpus snippet (hard zero).
    Copied,
}

/// Structured similarity verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityVerdict {
    /// Class.
    pub kind: SimilarityKind,
    /// 0.0..1.0 similarity estimate vs closest corpus entry.
    pub score: f64,
    /// Which corpus snippet was closest (submission id / baseline label).
    pub closest: Option<String>,
    /// Freeform evidence (≤ 3).
    pub evidence: Vec<String>,
    /// Prompt version served.
    pub prompt_version: &'static str,
}

/// One corpus snippet for similarity compare (baseline or past submission).
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSnippet {
    /// Label: `baseline` or `subm:<id-prefix>`.
    pub label: String,
    /// architecture.py source (truncated to cap).
    pub architecture_py: String,
    /// training.py source (truncated to cap).
    pub training_py: String,
}

/// Truncate a source to the prompt cap (byte-safe on char boundaries).
#[must_use]
pub fn truncate_source(src: &str) -> String {
    if src.len() <= MAX_SOURCE_BYTES_IN_PROMPT {
        return src.to_owned();
    }
    let mut n = MAX_SOURCE_BYTES_IN_PROMPT;
    while n > 0 && !src.is_char_boundary(n) {
        n -= 1;
    }
    format!("{}…", &src[..n])
}
