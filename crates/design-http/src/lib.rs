//! Design challenge HTTP surface (miner submit, viewer, annotate, stats).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::result_large_err)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::struct_field_names)]

mod api;
mod stats;

pub use api::{
    design_router, mark_awaiting, mark_awaiting_admin, record_epoch, AdminAwardHook, AppState,
};
pub use design_challenge_task::CHALLENGE_ID;
