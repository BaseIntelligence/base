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
    design_router, enqueue_active_harnesses_for_round, mark_awaiting, mark_awaiting_admin,
    record_epoch, schedule_harness_for_round, AdminAwardHook, AppState, EnqueueRoundResult,
};
pub use design_challenge_task::CHALLENGE_ID;
pub use design_challenge_task::{
    awaiting_admin_unscored_expired, reject_awaiting_admin_run, sanitize_reject_reason,
};
