//! gbase-validator library: independent weight recomputation process.
//!
//! Task 28: registration stub, epoch clock, telemetry, coordination allowlist,
//! graceful shutdown.
//! Task 29: bundle fetch, verify against **local** trust root, independent
//! aggregate, dual final-vector comparison (`Match` / `VectorMismatch` /
//! `InputInvalid` / `NoSubmission`). No last-known-good; no extrinsic submit.

#![forbid(unsafe_code)]

pub mod app;
pub mod coordination;
pub mod epoch;
pub mod error;
pub mod recompute;
pub mod registration;
pub mod sync_chain;

pub use app::{
    build_health_router, chain_ready_check, db_ready_from_fn, db_ready_ok, spawn_validator,
    spawn_validator_with_ok_db, RunningValidator, ValidatorRuntime,
};
pub use coordination::{
    is_allowed_gateway_path, is_master_only_path, CoordinationClient, CoordinationError,
    ALLOWED_GATEWAY_PATHS, MASTER_ONLY_PATHS,
};
pub use epoch::{epoch_from_block, epoch_from_chain, EpochSnapshot};
pub use error::ValidatorError;
pub use recompute::{
    compare_bundle, compare_bundle_bytes, fetch_and_compare, independent_aggregate, vector_sha256,
    ComparisonOutcome, NoSubmissionReason, RecomputeError,
};
pub use registration::{RegistrationStatus, RegistrationStub};
pub use sync_chain::SyncChain;

#[cfg(test)]
mod skeleton_tests;
