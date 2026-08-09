//! D24 exact-E leaf emission for `challenge_id = "prism"`.
//!
//! The implementation lives in `prism-emit::leaf_emit` (moved for the
//! per-crate non-test LOC cap); this shim keeps the
//! `prism_challenge::leaf_emit::*` path stable.

pub use prism_emit::leaf_emit::emit_signed_leaf_set;
