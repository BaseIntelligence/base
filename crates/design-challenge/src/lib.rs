//! Design challenge orchestrator on Base.
//!
//! Miner harness submit → sandboxed Python runs → sanitized viewer →
//! agentic anti-cheat → admin winners → D24 exact-E leaves under
//! `challenge_id = "design"`.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::duration_suboptimal_units)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::result_large_err)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::struct_field_names)]

pub mod host_sim;
mod orchestrator;
pub mod score;
mod screenshot;

pub use challenge_common::{
    emit_signed_leaf_set, public_key_from_secret, submit_signed_leaf_set, verify_leaf_sig,
    GatewayClient, GatewayClientConfig, LeafEmitError,
};
pub use design_challenge_task::{
    agent_run_timeout_secs, prompts_per_round, round_id_at, round_secs, CHALLENGE_ID,
    CHALLENGE_ID_BYTES, DAILY_RUN_QUOTA, PROMPTS_PER_ROUND, ROUND_SECS, SCORE_MAX, SCORING_VERSION,
    SCORING_WINDOW_ROUNDS,
};
pub use design_http::{
    design_router, mark_awaiting, mark_awaiting_admin, record_epoch, AdminAwardHook, AppState,
};
pub use design_store::{
    DbDesignStore, DesignStore, FinalScore, MemoryDesignStore, RoundAward, RunStage, StoreError,
};
pub use host_sim::{
    force_sim_refusal_reason, host_sim_allowed, is_prod_env, require_host_sim_for_force,
};
pub use orchestrator::{ErrorClass, Orchestrator, OrchestratorConfig};
pub use score::{round_win_delta, score_window, window_start, ScorePlan, WindowScorePlan};

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
