//! Versioned prompt templates. Edit = bump the version + test fixture.

/// Coherence / anti-cheat review prompt (`{ARCH}` / `{TRAIN}` placeholders).
///
/// v2 role change: the reviewer is a gatekeeper investigating honesty and
/// coherence, **never** a grader (operator decision: LLM votes must not move
/// the integer bpb-only score). v1 (quality grading) is retired from the
/// live path; the historical prompt stays in git history only.
pub const REVIEW_PROMPT_V2: &str = include_str!("../prompts/review_v2.md");

/// Similarity prompt template (`{ARCH}` / `{TRAIN}` / `{CORPUS}` placeholders).
pub const SIMILARITY_PROMPT_V1: &str = include_str!("../prompts/similarity_v1.md");

/// Version string for the review prompt.
pub const REVIEW_PROMPT_VERSION: &str = "review-v2";
/// Version string for the similarity prompt.
pub const SIMILARITY_PROMPT_VERSION: &str = "similarity-v1";
