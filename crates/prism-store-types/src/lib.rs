//! `prism-store-types` — the PRISM persistence data contract, split out of
//! `prism-store` for the per-crate non-test LOC cap (the repo's standard
//! crate-split pattern, same as `prism-lium-types`).
//!
//! A pure leaf: the submission row and its stage lattice, the patch applied
//! by the orchestrator, the store error taxonomy, and the registry / epoch /
//! top-model records. No trait, no SQL, no I/O — the `PrismStore` contract,
//! the memory impl, and the Postgres impl stay in `prism-store`, which
//! re-exports every name here so `prism_store::…` paths keep working.

#![forbid(unsafe_code)]

mod types;

pub use types::{
    ArchitectureRecord, EpochScoreRow, FinalScore, PublishArchOutcome, Stage, StageEvent,
    StatePatch, StoreError, SubmissionId, SubmissionState, TopModelPublication,
};
