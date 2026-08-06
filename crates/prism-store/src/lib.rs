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
pub use prism_lium::TelemetryPoint;
pub use store::{
    ArchitectureRecord, EpochScoreRow, FinalScore, MemoryPrismStore, PrismStore,
    PublishArchOutcome, Stage, StageEvent, StatePatch, StoreError, SubmissionState,
    TopModelPublication,
};
