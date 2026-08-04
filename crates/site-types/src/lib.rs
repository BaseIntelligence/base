//! Marketing site JSON contract (types + static arena frames).
//!
//! Shared by `site-api` so the aggregator crate stays under the LOC cap.

#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown)]

mod frames;
mod paginate;
mod types;

pub use frames::{coding_arena, design_frame, prism_frame};
pub use paginate::page_slice;
pub use types::*;
