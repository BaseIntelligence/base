//! Design challenge orchestrator on Base.
//!
//! Miner harness submit → sandboxed Python runs → sanitized viewer →
//! agentic anti-cheat → admin winners → D24 exact-E leaves under
//! `challenge_id = "design"`.

#![forbid(unsafe_code)]
#![allow(
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#![allow(clippy::duration_suboptimal_units, clippy::map_unwrap_or)]
#![allow(clippy::cast_possible_wrap, clippy::result_large_err)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::struct_field_names)]

pub mod backfill;
pub mod corpus;
pub mod host_sim;
mod orchestrator;
pub mod screenshot;

pub use challenge_common::{
    emit_signed_leaf_set, public_key_from_secret, submit_signed_leaf_set, verify_leaf_sig,
    GatewayClient, GatewayClientConfig, LeafEmitError,
};
pub use design_challenge_task::{
    agent_run_timeout_secs, awaiting_admin_unscored_expired, daily_run_quota,
    manual_daily_run_quota, prompts_per_round, reject_awaiting_admin_run, round_id_at, round_secs,
    round_win_delta, rounds_per_day_effective, scheduled_daily_run_cap, scheduled_runs_per_day,
    score_window, unscored_epochs_elapsed, window_start, ScorePlan, WindowScorePlan, CHALLENGE_ID,
    CHALLENGE_ID_BYTES, MANUAL_DAILY_RUN_QUOTA, PROMPTS_PER_ROUND, ROUNDS_PER_DAY, ROUND_SECS,
    SCORE_MAX, SCORING_VERSION, SCORING_WINDOW_ROUNDS, UNSCORED_EPOCH_LIMIT,
};
pub use design_http::{
    design_router, mark_awaiting, mark_awaiting_admin, record_epoch, AdminAwardHook, AppState,
};
pub use design_store::{
    DesignStore, FinalScore, MemoryDesignStore, RoundAward, RunStage, StoreError,
};
pub use design_store_pg::DbDesignStore;
pub use host_sim::{
    force_sim_refusal_reason, host_sim_allowed, is_prod_env, require_host_sim_for_force,
};
pub use orchestrator::{ErrorClass, Orchestrator, OrchestratorConfig};

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "design-challenge"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        assert_eq!(crate_name(), "design-challenge");
        assert_eq!(CHALLENGE_ID, "design");
        assert_eq!(SCORING_VERSION, 3);
    }
}
