//! PRISM eval pipeline (provision → exec → receipt → score → terminate) plus
//! the miner submission intake contract. Split out of `prism-challenge` for
//! the per-crate LOC cap; `prism-challenge` re-exports everything.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]

pub mod composite;
pub mod config;
pub mod pipeline;
pub mod precheck;
pub mod score;
pub mod submission;

pub use composite::{
    evaluate, BootstrapInfo, BudgetFacts, CompositeAnchors, CompositeError, CompositeOutcome,
    CompositeScore, GateFailure, GateReport, GateThresholds, GroupScore, Ineligible, MetricSeries,
    MirrorPair, NormDesc, SubmissionMetrics, GROUP_KEYS, N_GROUPS,
};
pub use config::PrismConfig;
pub use pipeline::{
    measurement_patch, mid_pod_resume, resume_measurement, run_eval_pipeline, run_sim_pipeline,
    PipelineError, PipelineInput, PipelineResult,
};
pub use precheck::{
    corpus_from_rows, ephemeral_candidate, evaluate_copy_precheck, gate_corpus_from_rows,
    precheck_json, precheck_quota_exceeded_json, precheck_skipped, quota_identity, quota_view,
    same_miner, utc_day, PrecheckIdentityKind, PrecheckQuotaView, PrecheckResult,
    PRECHECK_DAILY_LIMIT,
};
pub use score::{final_lattice, score_from_bpb, score_from_pipeline, PipelineOutcome, ScoringMode};
pub use submission::{
    arch_digest, arch_id_for, example_automodel_request, example_legacy_request,
    example_valid_request, expand_zip_fields, gating_key, is_arch_id, is_automodel_request,
    queued_row, source_tree, submission_id, tree_blob_for, validate, QueuedSubmission,
    SubmissionAccepted, SubmissionError, SubmissionId, SubmissionRequest, SubmissionService,
};
