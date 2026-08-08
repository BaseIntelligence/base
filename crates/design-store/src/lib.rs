//! Design challenge persistence surface.
//!
//! [`MemoryDesignStore`] for CI/sim; Postgres adapter lives in `design-store-pg`.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::assigning_clones)]

mod store;

pub use store::{
    ArtifactPage, DesignStore, FinalScore, HarnessRow, MemoryDesignStore, PairRow, QuotaUsage,
    RatingRow, RoundAward, RoundRow, RunOrigin, RunStage, RunState, StageEvent, StoreError,
    StorePatch,
};
