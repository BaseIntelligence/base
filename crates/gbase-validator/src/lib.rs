//! gbase-validator library: independent weight recomputation process.
//!
//! Task 28: registration stub, epoch clock, telemetry, coordination allowlist,
//! graceful shutdown.
//! Task 29: bundle fetch, verify against **local** trust root, independent
//! aggregate, dual final-vector comparison (`Match` / `VectorMismatch` /
//! `InputInvalid` / `NoSubmission`). No last-known-good; no extrinsic submit.
//! Task 30: verified-bundle mirror store, `GET /v1/bundle/root/{root}`, peer
//! fetch by root when the gateway is unreachable (content-addressed).
//! Task 31: peer root cross-check — `GET /v1/consensus/root/{epoch}` signed
//! under `gbase-root-v1`, `min_peer_sample` gate (D26), fail closed on
//! disagreement / insufficient sample.

#![forbid(unsafe_code)]

pub mod app;
pub mod attest;
pub mod coordination;
pub mod crosscheck;
pub mod epoch;
pub mod error;
pub mod mirror;
pub mod peers;
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
pub use crosscheck::{
    apply_crosscheck_gate, candidate_from_comparison, consensus_root_router, crosscheck_peer_roots,
    evaluate_sample, fetch_peer_root, no_submission_for_crosscheck, other_validators_on_metagraph,
    peer_refs, record_local_root, root_statement_payload, CrossCheckConfig, CrossCheckFailKind,
    CrossCheckOutcome, CrossCheckRun, RootStatementJson, RootStatementStore, SharedRootStore,
    SignedRootStatement, DEFAULT_MIN_SAMPLE,
};
pub use epoch::{epoch_from_block, epoch_from_chain, EpochSnapshot};
pub use error::ValidatorError;
pub use mirror::{
    bundle_identity, mirror_router, parse_root_hex, root_hex, MemoryMirrorStore, SharedMirrorStore,
};
pub use peers::{PeerBook, PeerEndpoint};
pub use recompute::{
    compare_bundle, compare_bundle_bytes, fetch_and_compare, fetch_and_compare_with_mirror,
    fetch_compare_and_crosscheck, independent_aggregate, maybe_persist_verified, vector_sha256,
    ComparisonOutcome, ExpectedBundle, NoSubmissionReason, RecomputeError,
};
pub use registration::{RegistrationStatus, RegistrationStub};
pub use sync_chain::SyncChain;

#[cfg(test)]
mod skeleton_tests;
#[cfg(test)]
mod mirror_peer_tests;
#[cfg(test)]
mod crosscheck_tests;

pub use attest::{
    attest_router, spawn_attest_server, AttestState, NonceRequest, NonceResponse, SubmitRequest,
    SubmitResponse, DEFAULT_NONCE_TTL,
};
