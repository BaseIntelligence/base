//! PRISM challenge orchestrator on Base.
//!
//! Master-centralized GPU eval via Lium (or Sim). **No Phala CVM.**
//! Miner submit API mirrors agent/hypertraining shape; scores emit D24 leaves
//! under `challenge_id = "prism"`.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]

mod api;
pub mod eval_finalize;
mod leaf_emit;
pub mod orchestrator;

pub use api::{record_epoch, submission_router, AppState};
pub use challenge_common::{
    public_key_from_secret, submit_signed_leaf_set, verify_leaf_sig, GatewayClient,
    GatewayClientConfig, LeafEmitError, SubmitError, SubmitOutcome,
};
pub use leaf_emit::emit_signed_leaf_set;
pub use orchestrator::{Orchestrator, OrchestratorConfig};
pub use prism_challenge_task::{
    CHALLENGE_ID, CHALLENGE_ID_BYTES, SCORE_MAX, SCORING_VERSION, TASK_ID_DOMAIN,
};
pub use prism_emit::{EmitError, EmitSummary, EpochEmitter};
pub use prism_eval_store::{DbEvalStore, MemoryEvalStore};
pub use prism_store::eval::EvalStore;
// 2×2 attribution matrix (module lives in prism-recipe beside SourceTree for
// the per-crate LOC cap; conceptually prism_challenge::attribution).
pub use prism_final::{combine_final, FinalOutcome};
pub use prism_pipeline::{
    example_automodel_request, example_valid_request, expand_zip_fields, run_eval_pipeline,
    run_sim_pipeline, score_from_bpb, score_from_pipeline, submission_id, validate, PipelineError,
    PipelineInput, PipelineOutcome, PipelineResult, PrismConfig, QueuedSubmission,
    SubmissionAccepted, SubmissionError, SubmissionId, SubmissionRequest, SubmissionService,
};
pub use prism_recipe::attribution;
pub use prism_store::{
    DbPrismStore, FinalScore, MemoryPrismStore, PrismStore, Stage, StageEvent, StatePatch,
    StoreError, SubmissionState,
};

pub use bundle::{LeafV1, NoScoreReasonCode, ScoreOrAbsence};
pub use crypto::KEY_LEN;
pub use prism_lium::{
    EvalJobBackend, EvalReceipt, LiumClient, LiumSshConfig, SimLiumBackend, LIUM_API_BASE_URL,
};

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "prism-challenge"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        assert_eq!(crate_name(), "prism-challenge");
        assert_eq!(CHALLENGE_ID, "prism");
        assert_eq!(SCORING_VERSION, 2);
        assert_ne!(CHALLENGE_ID, "agent-v1");
        assert_ne!(CHALLENGE_ID, "hypertraining");
    }
}
