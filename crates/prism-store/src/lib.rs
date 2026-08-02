//! `prism-store` — persistence surface for the PRISM orchestrator.
//!
//! Single source of truth for submission rows + stage events. Two impls:
//! [`MemoryPrismStore`] (offline tests/sim) and [`DbPrismStore`] (production
//! SQL via `db`).

#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod dbprism;
mod store;

pub use dbprism::DbPrismStore;
pub use store::{
    FinalScore, MemoryPrismStore, PrismStore, Stage, StageEvent, StatePatch, StoreError,
    SubmissionState,
};
