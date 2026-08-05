//! validator-verify: gateway coordination client, independent bundle
//! recompute, verified-bundle mirror, peer book, and peer root cross-check
//! glue. Extracted from `validator` (per-crate LOC cap); the `validator`
//! crate re-exports these modules unchanged.

#![forbid(unsafe_code)]

pub mod coordination;
pub mod crosscheck;
pub mod mirror;
pub mod peers;
pub mod recompute;
