//! Measurement→composite→storage glue (E4).
//!
//! The implementation lives in `prism-eval-store` (split out for the
//! per-crate non-test LOC cap — the same reason `prism-pipeline` exists);
//! this module pins the orchestrator-facing path
//! `prism_challenge::eval_finalize::finalize_composite`.
//!
//! Wiring (integration pass): call [`finalize_composite`] from the
//! orchestrator scoring step with the run's METRICS_JSON v2 blob and
//! [`AnchorInput::v0_placeholder`], then attach the returned outcome to
//! `FinalOutcome::Measured.composite`.

pub use prism_eval_store::{finalize_composite, AnchorInput, FinalizeError};
