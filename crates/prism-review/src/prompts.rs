//! Versioned prompt templates. Edit = bump the version + test fixture.

/// Quality review prompt template (`{ARCH}` / `{TRAIN}` placeholders).
pub const REVIEW_PROMPT_V1: &str = include_str!("../prompts/review_v1.md");

/// Similarity prompt template (`{ARCH}` / `{TRAIN}` / `{CORPUS}` placeholders).
pub const SIMILARITY_PROMPT_V1: &str = include_str!("../prompts/similarity_v1.md");

/// Version string for the review prompt.
pub const REVIEW_PROMPT_VERSION: &str = "review-v1";
/// Version string for the similarity prompt.
pub const SIMILARITY_PROMPT_VERSION: &str = "similarity-v1";
