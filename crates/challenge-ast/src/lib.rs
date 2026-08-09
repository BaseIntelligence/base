//! Python AST structural fingerprint + integer similarity (basis points).
//!
//! Feeds the agentic anti-cheat verifier (`challenge-agentic`). This crate
//! **never decides** cheat vs clean on its own — it only ranks corpus neighbors
//! and summarizes structural overlap for tool results.

#![forbid(unsafe_code)]

mod fingerprint;
mod gate;
mod similarity;
mod source_cheats;
mod walk;

pub use fingerprint::{fingerprint_source, AstError, Fingerprint};
pub use gate::{
    copy_gate, CopyGateHit, GateCorpusEntry, AST_CHEAT_BPS, AST_SUSPICIOUS_BPS,
    BASELINE_CORPUS_PREFIX,
};
pub use similarity::{
    similarity_bps, structural_diff_summary, summarize_fingerprint, top_k_nearest, Neighbor,
};
pub use source_cheats::{
    arch_has_noncausal_seq_mix, static_source_cheat, training_has_telemetry_hooks, SourceCheatHit,
    SourceCheatKind,
};

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "challenge-ast"
}
