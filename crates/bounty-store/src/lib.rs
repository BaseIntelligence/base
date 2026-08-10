//! Bounty challenge persistence surface.
//!
//! [`MemoryBountyStore`] for CI/sim; Postgres adapter lives in `bounty-store-pg`.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]

mod store;

pub use store::{
    BountyStore, BugPatch, BugRow, BugStatus, EpochScoreRow, FinalScore, MemoryBountyStore,
    StageEvent, StoreError,
};
