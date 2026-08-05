//! Versioned prompt templates. Edit = bump the version + test fixture.

/// Coherence / anti-cheat review prompt (`{ARCH}` / `{TRAIN}` placeholders).
///
/// v2 role change: the reviewer is a gatekeeper investigating honesty and
/// coherence, **never** a grader (operator decision: LLM votes must not move
/// the integer bpb-only score). v3 adds the telemetry-hook contract
/// (`prism_telemetry.report` + `finish_evaluation`) as a hard gate item.
pub const REVIEW_PROMPT_V3: &str = include_str!("../prompts/review_v3.md");

/// Architecture-only similarity prompt (`{ARCH}` / `{CORPUS}` placeholders).
///
/// v2 scope change: similarity judges `architecture.py` ONLY — `training.py`
/// is exempt from both the candidate and the corpus (the same training
/// script on two different architectures is legitimate).
pub const SIMILARITY_PROMPT_V2: &str = include_str!("../prompts/similarity_v2.md");

/// Version string for the review prompt.
pub const REVIEW_PROMPT_VERSION: &str = "review-v3";
/// Version string for the similarity prompt.
pub const SIMILARITY_PROMPT_VERSION: &str = "similarity-v2";
