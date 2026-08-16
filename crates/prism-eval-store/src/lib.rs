//! `prism-eval-store` — PRISM v3 eval persistence + finalize glue (E4).
//!
//! Implements [`prism_store::eval::EvalStore`] twice ([`MemoryEvalStore`]
//! for offline tests/sim, [`DbEvalStore`] for production Postgres over
//! migration 0017) and hosts the measurement→composite→storage glue
//! ([`finalize_composite`]), the legacy `train_metrics` Zone B lift, and the
//! API JSON views. Split out of `prism-store` / `prism-challenge` for the
//! per-crate non-test LOC cap (the repo's standard crate-split pattern, same
//! as `prism-pipeline` vs `prism-challenge`).

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]

mod db;
mod finalize;
mod memory;
mod views;

pub use db::DbEvalStore;
pub use finalize::{
    contamination_checked, finalize_composite, finalize_for_submission, from_train_metrics,
    ground_truth, require_contamination_check, AnchorInput, FinalizeError,
};
pub use memory::MemoryEvalStore;
pub use views::{
    anchors_view, detail_view, eval_detail, eval_json, list_view, prepare_cohort, prereg_view,
    zone_a_view, zone_b_view,
};
