//! `prism-store` — persistence surface for the PRISM orchestrator.
//!
//! Single source of truth for submission rows + stage events + the
//! architecture registry / top-model journal (migration 0010). Two impls:
//! [`MemoryPrismStore`] (offline tests/sim) and [`DbPrismStore`] (production
//! SQL via `db`).

#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod arch;
mod dbprism;
mod emit;
pub mod eval;
mod store;
mod telemetry;

pub use dbprism::DbPrismStore;
pub use eval::EvalStore;
pub use prism_lium_types::TelemetryPoint;
pub use store::{MemoryPrismStore, PrismStore};
// The data contract lives in `prism-store-types` (per-crate LOC cap); it is
// re-exported wholesale so `prism_store::…` stays the single import path.
pub use prism_store_types::{
    ArchitectureRecord, EpochScoreRow, FinalScore, PublishArchOutcome, Stage, StageEvent,
    StatePatch, StoreError, SubmissionId, SubmissionState, TopModelPublication,
};
