//! Data-plane modules extracted from the `gateway` crate so both stay under
//! the workspace LOC cap. `gateway` re-exports everything here; external
//! callers should keep importing `gateway::*`.
//!
//! - [`admin_attest`]: master-only owner credit for non-TEE runtimes.
//! - [`weights_store`]: raw-weight leaf row + in-memory store + ingress errors.

#![forbid(unsafe_code)]

pub mod admin_attest;
pub mod weights_store;
