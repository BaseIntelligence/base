//! Bounty challenge HTTP surface (miner / admin / public).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::result_large_err)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::unnested_or_patterns)]
#![allow(clippy::manual_let_else)]

mod api;

pub use api::{bounty_router, AppState};
pub use bounty_challenge_task::CHALLENGE_ID;
