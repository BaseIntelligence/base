//! `challenge-agentic-types` — the agentic anti-cheat review contract, split
//! out of `challenge-agentic` for the per-crate non-test LOC cap (the repo's
//! standard crate-split pattern, same as `prism-lium-types` and
//! `prism-store-types`).
//!
//! A pure leaf: the review request shapes (source tree or container), the
//! corpus entry a reviewer compares against, the verdict lattice
//! ([`VerdictKind`], [`CheatCode`], [`AgenticVerdict`]), and the
//! [`AgenticBackend`] trait both the real and sim reviewers implement. No
//! HTTP, no filesystem, no prompts — those stay in `challenge-agentic`, which
//! re-exports every name here.

#![forbid(unsafe_code)]

mod types;

pub use types::{
    AgenticBackend, AgenticError, AgenticVerdict, CheatCode, ContainerReviewRequest, CorpusEntry,
    ReviewRequest, VerdictKind, OPENROUTER_API_BASE,
};
