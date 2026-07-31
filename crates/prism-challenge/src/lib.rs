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


mod config;
mod leaf_emit;
mod pipeline;
mod routes;
mod score;
mod submission;
mod submit;

pub use config::PrismConfig;
pub use leaf_emit::{
    emit_signed_leaf_set, public_key_from_secret, verify_leaf_sig, LeafEmitError,
};
pub use pipeline::{run_eval_pipeline, run_sim_pipeline, PipelineError, PipelineInput, PipelineResult};
pub use prism_challenge_task::{
    CHALLENGE_ID, CHALLENGE_ID_BYTES, SCORE_MAX, SCORING_VERSION, TASK_ID_DOMAIN,
};
pub use routes::{submission_router, AppState};
pub use score::{score_from_bpb, score_from_pipeline, PipelineOutcome};
pub use submission::{
    example_valid_request, QueuedSubmission, SubmissionAccepted, SubmissionError,
    SubmissionId, SubmissionRequest, SubmissionService,
};
pub use submit::{
    submit_signed_leaf_set, GatewayClient, GatewayClientConfig, SubmitError, SubmitOutcome,
};

pub use bundle::{LeafV1, NoScoreReasonCode, ScoreOrAbsence};
pub use crypto::KEY_LEN;
pub use prism_lium::{EvalJobBackend, EvalReceipt, LiumClient, LiumSshConfig, SimLiumBackend, LIUM_API_BASE_URL};

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
        assert_eq!(SCORING_VERSION, 1);
        assert_ne!(CHALLENGE_ID, "agent-v1");
        assert_ne!(CHALLENGE_ID, "hypertraining");
    }
}
