//! Agentic anti-cheat verifier for Design / Prism (master-side).
//!
//! Runs an OpenRouter-compatible **tool-calling** loop over a read-only
//! workdir, then requires a final `submit_verdict` function call. Fail-closed
//! when no parseable verdict is produced (caller maps to `ChallengeInternal`).
//!
//! Backends:
//! - [`OpenRouterAgent`] — live multi-step tool use (key from file only).
//! - [`SimAgent`] — deterministic AST + hash heuristics for CI (no network).

#![forbid(unsafe_code)]

mod agent;
mod llm;
mod prompts;
mod sim;
mod tools;
mod types;

pub use agent::{AgentConfig, OpenRouterAgent};
pub use llm::{load_api_key_file, DEFAULT_MODEL};
pub use prompts::{AGENTIC_PROMPT_VERSION, DESIGN_DOMAIN_RULES, PRISM_DOMAIN_RULES};
pub use sim::{SimAgent, SIM_CHEAT_BPS, SIM_SUSPICIOUS_BPS};
pub use types::{
    AgenticBackend, AgenticError, AgenticVerdict, CheatCode, CorpusEntry, ReviewRequest,
    VerdictKind, OPENROUTER_API_BASE,
};

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "challenge-agentic"
}
