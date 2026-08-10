//! Bounty challenge orchestrator on Base.
//!
//! Miner video bug report → ffmpeg compress → agentic similar-24h → admin
//! approve → TARGET_BUGS burn-sink epoch leaves under `challenge_id = "bounty"`.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc, clippy::doc_markdown)]
#![allow(clippy::too_many_lines)]

pub mod compress;
pub mod emit;
pub mod orchestrator;
pub mod similarity;

pub use bounty_challenge_task::{
    score_epoch, EpochScoreInput, EpochScoreOutcome, BUG_ID_DOMAIN, CHALLENGE_ID,
    CHALLENGE_ID_BYTES, SCORE_MAX, SCORING_VERSION, SUBMISSION_DOMAIN, TARGET_BUGS,
};
pub use bounty_http::{bounty_router, AppState};
pub use bounty_store::{
    BountyStore, BugPatch, BugRow, BugStatus, EpochScoreRow, FinalScore, MemoryBountyStore,
    StageEvent, StoreError,
};
pub use bounty_store_pg::DbBountyStore;
pub use challenge_common::{
    public_key_from_secret, submit_signed_leaf_set, verify_leaf_sig, GatewayClient,
    GatewayClientConfig, LeafEmitError, SubmitError, SubmitOutcome, DRY_RUN_BASE_URL,
};
pub use compress::{compress_video, CompressError, CompressResult};
pub use emit::{
    build_epoch_scores, emit_epoch, emit_signed_bounty_leaf_set, EmitError, EmitSummary,
};
pub use orchestrator::{Orchestrator, OrchestratorConfig};
pub use similarity::{
    map_agentic_verdict, openrouter_model, report_text, review_similarity, BountySimAgent,
    SimilarityError, SimilarityKind, SimilarityVerdict, BOUNTY_DOMAIN_RULES,
    DEFAULT_OPENROUTER_MODEL,
};

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "bounty-challenge"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        assert_eq!(crate_name(), "bounty-challenge");
        assert_eq!(CHALLENGE_ID, "bounty");
        assert_eq!(SCORING_VERSION, 1);
        assert_eq!(TARGET_BUGS, 50);
    }
}
