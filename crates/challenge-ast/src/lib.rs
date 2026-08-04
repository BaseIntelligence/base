//! Python AST structural fingerprint + integer similarity (basis points).
//!
//! Feeds the agentic anti-cheat verifier (`challenge-agentic`). This crate
//! **never decides** cheat vs clean on its own — it only ranks corpus neighbors
//! and summarizes structural overlap for tool results.

#![forbid(unsafe_code)]

mod fingerprint;
mod similarity;
mod walk;

pub use fingerprint::{fingerprint_source, AstError, Fingerprint};
pub use similarity::{
    similarity_bps, structural_diff_summary, summarize_fingerprint, top_k_nearest, Neighbor,
};

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "challenge-ast"
}
