//! PRISM eval pipeline (provision → exec → receipt → score → terminate) plus
//! the miner submission intake contract. Split out of `prism-challenge` for
//! the per-crate LOC cap; `prism-challenge` re-exports everything.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]

pub mod config;
pub mod pipeline;
pub mod score;
pub mod submission;

pub use config::PrismConfig;
pub use pipeline::{
    run_eval_pipeline, run_sim_pipeline, PipelineError, PipelineInput, PipelineResult,
};
pub use score::{score_from_bpb, score_from_pipeline, PipelineOutcome};
pub use submission::{
    arch_digest, arch_id_for, example_valid_request, expand_zip_fields, gating_key, is_arch_id,
    submission_id, validate, QueuedSubmission, SubmissionAccepted, SubmissionError, SubmissionId,
    SubmissionRequest, SubmissionService,
};
