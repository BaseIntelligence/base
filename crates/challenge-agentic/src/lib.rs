//! Agentic anti-cheat verifier for Design / Prism (master-side).
//!
//! Runs an OpenRouter-compatible **tool-calling** loop over a read-only
//! workdir, then requires a final `submit_verdict` function call. Fail-closed
//! when no parseable verdict is produced (caller maps to `ChallengeInternal`).
//!
//! Backends:
//! - [`OpenRouterAgent`] — live multi-step tool use (key from file only).
//! - [`SimAgent`] — deterministic AST + hash heuristics for CI (no network).
//!
//! The containerized backend (`DockerAgent`, `design-review` image) lives in
//! the `review-docker` crate; the pre-LLM copy gate lives in `challenge-ast`
//! and is re-exported here.

#![forbid(unsafe_code)]

mod agent;
mod llm;
mod prompts;
mod sim;
mod tools;
mod types;

pub use agent::{AgentConfig, OpenRouterAgent};
pub use challenge_ast::{
    arch_has_noncausal_seq_mix, copy_gate, static_source_cheat, training_has_telemetry_hooks,
    CopyGateHit, GateCorpusEntry, SourceCheatHit, SourceCheatKind,
};
pub use llm::{load_api_key_file, DEFAULT_MODEL};
pub use prompts::{AGENTIC_PROMPT_VERSION, DESIGN_DOMAIN_RULES, PRISM_DOMAIN_RULES};
pub use sim::{SimAgent, SIM_CHEAT_BPS, SIM_SUSPICIOUS_BPS};
pub use types::{
    AgenticBackend, AgenticError, AgenticVerdict, CheatCode, ContainerReviewRequest, CorpusEntry,
    ReviewRequest, VerdictKind, OPENROUTER_API_BASE,
};

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "challenge-agentic"
}

/// Same economic miner for copy/similarity corpora: matching hotkey, or both
/// coldkeys known and equal (case-insensitive). Used when 1-max gating forces
/// hotkey rotation under one coldkey.
#[must_use]
pub fn same_miner_identity(
    hotkey_a: &str,
    coldkey_a: Option<&str>,
    hotkey_b: &str,
    coldkey_b: Option<&str>,
) -> bool {
    if hotkey_a.eq_ignore_ascii_case(hotkey_b) {
        return true;
    }
    match (coldkey_a, coldkey_b) {
        (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => a.eq_ignore_ascii_case(b),
        _ => false,
    }
}
