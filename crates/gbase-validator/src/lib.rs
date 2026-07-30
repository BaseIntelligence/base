//! gbase-validator library: skeleton process for independent weight recomputation.
//!
//! Task 28 scope: registration stub, epoch clock from `current_block`, telemetry
//! (`/healthz`, `/readyz`, `/metrics`), coordination client that never calls
//! master-only gateway endpoints, graceful shutdown. Full bundle fetch/recompute
//! is task 29.

#![forbid(unsafe_code)]

pub mod app;
pub mod coordination;
pub mod epoch;
pub mod error;
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
pub use registration::{RegistrationStatus, RegistrationStub};
pub use sync_chain::SyncChain;

#[cfg(test)]
mod skeleton_tests;
