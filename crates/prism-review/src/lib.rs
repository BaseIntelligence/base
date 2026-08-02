//! LLM review + LLM similarity review for PRISM submissions.
//!
//! **Master-side**: the master calls `OpenRouter` directly (never the pod —
//! pods never see an API key). The wire path targets the chat-completions
//! endpoint with prompts **versioned in this repo** (`prompts/`). Parsers are
//! strict: any non-conforming answer fails closed (`ReviewError`), which the
//! challenge maps to `ChallengeInternal` — no score is ever invented.
//!
//! Two backends implement [`ReviewBackend`]:
//!
//! - [`OpenRouterClient`] — real provider, key read from a file at startup.
//! - [`SimReviewer`] — deterministic offline reviewer for CI / local sim.
//!
//! Secrets discipline mirrors `prism-lium`: no key in env vars (read file),
//! no key in logs/Debug/Display.

#![forbid(unsafe_code)]

mod llm;
mod prompts;
mod sim;
mod types;

pub use llm::{load_api_key_file, OpenRouterClient};
pub use prompts::{REVIEW_PROMPT_VERSION, SIMILARITY_PROMPT_VERSION};
pub use sim::SimReviewer;
pub use types::{
    ReviewError, ReviewVerdict, SimilarityKind, SimilarityVerdict, SourceSnippet,
    OPENROUTER_API_BASE,
};

use async_trait::async_trait;

/// One code review (quality) + one similarity judgment per submission.
#[async_trait]
pub trait ReviewBackend: Send + Sync {
    /// Code quality review of the two-script submission.
    ///
    /// # Errors
    /// Transport / quota / parse failure (caller maps to `ChallengeInternal`).
    async fn review(
        &self,
        architecture_py: &str,
        training_py: &str,
    ) -> Result<ReviewVerdict, ReviewError>;

    /// Compare the submission against the baseline + known historical
    /// submissions. `corpus` is pre-selected by the caller (top-k).
    ///
    /// # Errors
    /// Transport / quota / parse failure.
    async fn similarity(
        &self,
        architecture_py: &str,
        training_py: &str,
        corpus: &[SourceSnippet],
    ) -> Result<SimilarityVerdict, ReviewError>;
}

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "prism-review"
}
