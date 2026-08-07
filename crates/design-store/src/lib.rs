//! Design challenge persistence surface.
//!
//! [`MemoryDesignStore`] for CI/sim; [`DbDesignStore`] for production SQL.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::assigning_clones)]

mod dbstore;
mod store;

pub use dbstore::DbDesignStore;
pub use store::{
    ArtifactPage, DesignStore, FinalScore, HarnessRow, MemoryDesignStore, PairRow, QuotaUsage,
    RatingRow, RoundAward, RoundRow, RunOrigin, RunStage, RunState, StageEvent, StoreError,
    StorePatch,
};
