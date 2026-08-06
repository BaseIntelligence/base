//! Gateway raw-weights submit client (thin wrap over `challenge-common`).
//!
//! The implementation lives in `prism-emit::submit` (moved for the
//! per-crate non-test LOC cap); this shim keeps the
//! `prism_challenge::submit::*` path stable for the orchestrator and
//! integration tests.

pub use prism_emit::submit::{
    submit_signed_leaf_set, GatewayClient, GatewayClientConfig, SubmitError, SubmitOutcome,
};
