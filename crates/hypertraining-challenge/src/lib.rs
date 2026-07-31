//! Hypertraining challenge orchestrator: pipeline, D24 leaves, gateway submit,
//! miner HTTP submit API (brief §7).
//!
//! Wires sealed → build → kernel → cluster Sim → eval → promo hooks → antinois → pay
//! into epoch scoring and signed `LeafV1` emission under `challenge_id = hypertraining`.
//!
//! # D24
//!
//! Every hotkey in expected set `E` receives a `Score` or `NoScore` leaf. Silence is a bug.
//!
//! # Attestation
//!
//! [`HypertrainingConfig::require_attestation`] defaults to `true` (prod). Sim tests use
//! [`HypertrainingChallenge::sim`] (`false`).
//!
//! # Miner submit
//!
//! [`routes::submission_router`] exposes `POST /v1/submissions` and `GET /health`.

#![forbid(unsafe_code)]

mod challenge;
mod leaf_emit;
mod config;
mod expected_set;
mod pipeline;
mod pipeline_types;
mod score;
mod sim_search;
mod submit;
mod submission;
mod routes;

pub use challenge::{
    AttestationLookup, ChallengeError, EpochCtx, HypertrainingChallenge, MapAttestationLookup,
};
pub use leaf_emit::{
    emit_signed_leaf_set, public_key_from_secret, verify_leaf_sig, LeafEmitError,
};
pub use config::HypertrainingConfig;
pub use expected_set::{expected_set_from_pinned_metagraph, ExpectedSetError, Hotkey};
pub use hypertraining_challenge_task::{
    CHALLENGE_ID, CHALLENGE_ID_BYTES, SCORING_VERSION, TASK_ID_DOMAIN,
};
pub use hypertraining_pay::SCORE_MAX;
pub use pipeline::{default_dedupe, run_sim_pipeline};
pub use pipeline_types::{
    PipelineError, SimPipelineInput, SimPipelineResult,
};
pub use sim_search::{code_fingerprint, find_faster_compiled};
pub use sim_search::find_faster_compiled as search_faster_compiled;
pub use score::{
    missing_call_noscore, score_from_pipeline, AttestationStatus, PipelineOutcome, HT_SCORE_MAX,
};
pub use submit::{
    submit_signed_leaf_set, GatewayClient, GatewayClientConfig, SubmitError, SubmitOutcome,
    DEFAULT_MAX_RETRIES,
};
pub use submission::{
    example_valid_request, PrecisionAttestationWire, QueuedSubmission, SubmissionAccepted,
    SubmissionError, SubmissionId, SubmissionRequest, SubmissionService, TopologyWire,
};
pub use routes::{submission_router, AppState};

pub use bundle::{LeafV1, NoScoreReasonCode, ScoreOrAbsence};
pub use crypto::KEY_LEN;

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "hypertraining-challenge"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_and_challenge_id() {
        assert_eq!(crate_name(), "hypertraining-challenge");
        assert_eq!(CHALLENGE_ID, "hypertraining");
        assert_ne!(CHALLENGE_ID, "agent-v1");
        assert_eq!(SCORING_VERSION, 1);
    }
}
