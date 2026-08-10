//! Design challenge identity constants.
//!
//! ```text
//! challenge_id     = "design"
//! scoring_version  = 3
//! round_id domain  = b"base-design-round-id-v1"
//! submission domain = b"base-design-submission-v1"
//! pair_id domain   = b"base-design-pair-id-v1"
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc, clippy::doc_markdown)]

mod emit;
mod score;
pub use emit::{design_emit_plan, DesignEmitPlan, DESIGN_EMIT_LATE_BLOCKS};
pub use score::{
    not_attempted, round_win_delta, score_window, to_leaf, window_start, ScorePlan, WindowScorePlan,
};

use design_store::{
    DesignStore, FinalScore, RunStage, RunState, StageEvent, StoreError, StorePatch,
};
use serde_json::json;
use submission_gating::{GatingState, GatingStore};

/// Normative challenge id (trust-root / leaf `challenge_id` string).
pub const CHALLENGE_ID: &str = "design";

/// UTF-8 bytes of [`CHALLENGE_ID`].
pub const CHALLENGE_ID_BYTES: &[u8] = b"design";

/// Live `challenge_scoring_version` for design (rolling 10-round points share).
pub const SCORING_VERSION: u16 = 3;

/// Domain tag for round id digests / prompt selection.
pub const ROUND_ID_DOMAIN: &[u8] = b"base-design-round-id-v1";

/// Domain tag for harness / submission digests.
pub const SUBMISSION_DOMAIN: &[u8] = b"base-design-submission-v1";

/// Domain tag for pairwise comparison ids.
pub const PAIR_ID_DOMAIN: &[u8] = b"base-design-pair-id-v1";

/// Domain tag for run ids.
pub const RUN_ID_DOMAIN: &[u8] = b"base-design-run-id-v1";

/// Integer score lattice max (same scale as other challenges).
pub const SCORE_MAX: u64 = 1_000_000;

/// Round duration in seconds (10 equal UTC-day slots → 86400/10).
///
/// Default only — runtime code must use [`round_secs`] so staging can compress
/// rounds via `DESIGN_ROUND_SECS`.
pub const ROUND_SECS: u64 = 8_640;

/// Rounds per UTC calendar day.
pub const ROUNDS_PER_DAY: u64 = 10;

/// Sandbox agent run wall-clock timeout (30 minutes). Distinct from round length.
///
/// Default only — runtime code must use [`agent_run_timeout_secs`]
/// (`DESIGN_AGENT_RUN_TIMEOUT_SECS` override).
pub const AGENT_RUN_TIMEOUT_SECS: u64 = 1_800;

/// Prompts selected per round (each becomes one organizer-scheduled sandbox run).
///
/// Prod pin is **1**: every harness in a round executes the identical shared
/// prompt. Default only — runtime code must use [`prompts_per_round`]
/// (`DESIGN_PROMPTS_PER_ROUND` override; staging may raise this).
pub const PROMPTS_PER_ROUND: usize = 1;

/// Epochs a clean `awaiting_admin` run may wait before auto-reject.
pub const UNSCORED_EPOCH_LIMIT: u64 = 5;

/// `true` when `current_epoch - start_epoch >= UNSCORED_EPOCH_LIMIT`.
#[must_use]
pub const fn unscored_epochs_elapsed(start_epoch: u64, current_epoch: u64) -> bool {
    current_epoch.saturating_sub(start_epoch) >= UNSCORED_EPOCH_LIMIT
}

#[must_use]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Miner-visible reject reason: trim, strip controls, cap length.
#[must_use]
pub fn sanitize_reject_reason(raw: Option<&str>, default: &str) -> String {
    const MAX: usize = 500;
    let fallback = if default.is_empty() {
        "rejected"
    } else {
        default
    };
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return fallback.to_owned();
    };
    let cleaned: String = s
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .take(MAX)
        .collect();
    if cleaned.trim().is_empty() {
        fallback.to_owned()
    } else {
        cleaned
    }
}

/// `awaiting_admin` past the unscored epoch budget.
#[must_use]
pub fn awaiting_admin_unscored_expired(run: &RunState, current_epoch: u64) -> bool {
    let start = run
        .awaiting_admin_epoch
        .or(run.attempt_epoch)
        .unwrap_or(current_epoch);
    run.status == RunStage::AwaitingAdmin && unscored_epochs_elapsed(start, current_epoch)
}

/// Reject clean `awaiting_admin` run; gate hotkey `Rejected` until UID leave.
pub async fn reject_awaiting_admin_run(
    store: &dyn DesignStore,
    gating: Option<&dyn GatingStore>,
    run: &RunState,
    reason: &str,
    reason_code: &str,
) -> Result<(), StoreError> {
    if run.status == RunStage::Rejected {
        return Ok(());
    }
    if run.status != RunStage::AwaitingAdmin {
        return Err(StoreError::Backend(format!(
            "not_candidate: run {} status={}",
            run.id,
            run.status.as_str()
        )));
    }
    let reason = sanitize_reject_reason(Some(reason), reason_code);
    store
        .apply_run(
            &run.id,
            &StorePatch {
                status: Some(RunStage::Rejected),
                reject_reason: Some(reason.clone()),
                final_score: Some(FinalScore::Score(0)),
                ..StorePatch::default()
            },
            Some(&StageEvent {
                stage: "rejected".into(),
                detail: Some(json!({
                    "reason": reason_code,
                    "reject_reason": reason,
                })),
                at_ms: now_ms(),
            }),
        )
        .await?;
    if let Some(g) = gating {
        if let Ok(Some(h)) = store.get_harness(&run.harness_id).await {
            let _ = g
                .set_terminal(
                    CHALLENGE_ID,
                    &h.miner_hotkey,
                    GatingState::Rejected,
                    Some(reason_code),
                )
                .await;
        }
    }
    Ok(())
}

/// Rolling scoring window in rounds (`challenge_scoring_version = 3`): the leaf
/// projection shares `SCORE_MAX` by round-win points over the last
/// [`SCORING_WINDOW_ROUNDS`] rounds, cheat excluded.
pub const SCORING_WINDOW_ROUNDS: u64 = 10;

/// Anti-spam ceiling on **miner-initiated** sandbox runs per hotkey per UTC
/// day — runs created by `POST /v1/harness`. Organizer-scheduled round runs
/// draw on [`scheduled_daily_run_cap`] instead, so a harness that participates
/// in every round of the day is never blocked from submitting.
///
/// Default only — runtime code must use [`manual_daily_run_quota`]
/// (`DESIGN_MANUAL_DAILY_RUN_QUOTA` override).
pub const MANUAL_DAILY_RUN_QUOTA: u32 = 10;

/// Headroom multiplier on the derived organizer-scheduled daily run cap. The
/// cap is a runaway-scheduler guard, not a participation limit, so it must
/// stay comfortably above the full-day schedule volume.
pub const SCHEDULED_DAILY_RUN_HEADROOM: u32 = 2;

/// Minimum annotations required per pair before Elo consume.
pub const MIN_ANNOTATIONS_PER_PAIR: u32 = 3;

/// Bottom fraction eliminated each round (20%).
pub const ELIMINATION_BOTTOM_BPS: u32 = 2000;

/// Elimination cooldown in rounds (10 rounds = 1 day at 10 rounds/day).
pub const ELIMINATION_COOLDOWN_ROUNDS: u64 = 10;

/// Read a `u64` env knob, falling back to `default` when unset/unparseable and
/// clamping to `min`.
fn env_u64(name: &str, default: u64, min: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(default, |v| v.max(min))
}

/// Effective round length in seconds (`DESIGN_ROUND_SECS` override; default
/// [`ROUND_SECS`]). Read on every call so tests/staging can adjust per process.
#[must_use]
pub fn round_secs() -> u64 {
    env_u64("DESIGN_ROUND_SECS", ROUND_SECS, 1)
}

/// Effective sandbox agent run timeout (`DESIGN_AGENT_RUN_TIMEOUT_SECS`
/// override; default [`AGENT_RUN_TIMEOUT_SECS`]).
#[must_use]
pub fn agent_run_timeout_secs() -> u64 {
    env_u64("DESIGN_AGENT_RUN_TIMEOUT_SECS", AGENT_RUN_TIMEOUT_SECS, 1)
}

/// Effective prompts per round (`DESIGN_PROMPTS_PER_ROUND` override; default
/// [`PROMPTS_PER_ROUND`]).
#[must_use]
pub fn prompts_per_round() -> usize {
    usize::try_from(env_u64(
        "DESIGN_PROMPTS_PER_ROUND",
        PROMPTS_PER_ROUND as u64,
        1,
    ))
    .unwrap_or(PROMPTS_PER_ROUND)
}

/// Rounds in a UTC day under the effective [`round_secs`]. Equals
/// [`ROUNDS_PER_DAY`] in production; staging compresses rounds, so scheduling
/// limits must derive from this and never from the constant.
#[must_use]
pub fn rounds_per_day_effective() -> u64 {
    (86_400 / round_secs()).max(1)
}

/// Sandbox runs the organizer dispatches to one harness across a full UTC day
/// (`rounds/day × prompts/round`). This is the volume an honest, fully
/// participating harness must be allowed to execute.
#[must_use]
pub fn scheduled_runs_per_day() -> u32 {
    let per_round = u64::try_from(prompts_per_round()).unwrap_or(PROMPTS_PER_ROUND as u64);
    u32::try_from(rounds_per_day_effective().saturating_mul(per_round)).unwrap_or(u32::MAX)
}

/// Effective anti-spam ceiling on miner-initiated runs per hotkey per UTC day
/// (`DESIGN_MANUAL_DAILY_RUN_QUOTA` override; default
/// [`MANUAL_DAILY_RUN_QUOTA`]).
#[must_use]
pub fn manual_daily_run_quota() -> u32 {
    u32::try_from(env_u64(
        "DESIGN_MANUAL_DAILY_RUN_QUOTA",
        u64::from(MANUAL_DAILY_RUN_QUOTA),
        1,
    ))
    .unwrap_or(MANUAL_DAILY_RUN_QUOTA)
}

/// Effective ceiling on organizer-scheduled runs per hotkey per UTC day
/// (`DESIGN_SCHEDULED_DAILY_RUN_CAP` override; default
/// [`scheduled_runs_per_day`] × [`SCHEDULED_DAILY_RUN_HEADROOM`]).
///
/// The floor is [`scheduled_runs_per_day`]: an operator override can never sit
/// below the day's own schedule, which is the bug this cap replaced.
#[must_use]
pub fn scheduled_daily_run_cap() -> u32 {
    let floor = scheduled_runs_per_day();
    let default = floor.saturating_mul(SCHEDULED_DAILY_RUN_HEADROOM);
    u32::try_from(env_u64(
        "DESIGN_SCHEDULED_DAILY_RUN_CAP",
        u64::from(default),
        u64::from(floor),
    ))
    .unwrap_or(default)
}

/// Total sandbox runs one hotkey may accumulate in a UTC day across both
/// origins (display / dashboard value; enforcement is per-origin).
#[must_use]
pub fn daily_run_quota() -> u32 {
    manual_daily_run_quota().saturating_add(scheduled_daily_run_cap())
}

/// Compute round id from unix seconds under an explicit round length.
#[must_use]
pub const fn round_id_at_with(unix_secs: u64, round_secs: u64) -> u64 {
    unix_secs / round_secs
}

/// Compute round id from unix seconds under the effective ([`round_secs`])
/// round length.
#[must_use]
pub fn round_id_at(unix_secs: u64) -> u64 {
    round_id_at_with(unix_secs, round_secs())
}

/// UTC day index (`floor(unix / 86400)`).
#[must_use]
pub const fn day_index_at(unix_secs: u64) -> u64 {
    unix_secs / 86_400
}

/// UTC day index that owns `round_id` (aligned with [`ROUND_SECS`] × [`ROUNDS_PER_DAY`]).
#[must_use]
pub const fn day_index_for_round(round_id: u64) -> u64 {
    round_id / ROUNDS_PER_DAY
}

/// Inclusive `(first_round, last_round)` for a UTC day index.
#[must_use]
pub const fn rounds_for_day(day_index: u64) -> (u64, u64) {
    let start = day_index * ROUNDS_PER_DAY;
    (start, start + ROUNDS_PER_DAY - 1)
}

/// Cap harness log payload stored in stage-event detail (JSON).
pub const MAX_LOG_CHARS: usize = 65_536;

#[must_use]
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[must_use]
pub fn clip_logs(text: &str) -> String {
    if text.len() <= MAX_LOG_CHARS {
        return text.to_owned();
    }
    format!("...[truncated]\n{}", &text[text.len() - MAX_LOG_CHARS..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        assert_eq!(CHALLENGE_ID, "design");
        assert_eq!(CHALLENGE_ID_BYTES, b"design");
        assert_eq!(SCORING_VERSION, 3);
        assert_eq!(SCORE_MAX, 1_000_000);
        assert_eq!(ROUND_SECS * ROUNDS_PER_DAY, 86_400);
        assert_eq!(AGENT_RUN_TIMEOUT_SECS, 1_800);
        assert_eq!(SCORING_WINDOW_ROUNDS, 10);
    }

    #[test]
    fn env_knobs_default_to_prod_consts() {
        // CI runs with no DESIGN_* overrides: getters must equal the prod pins.
        assert_eq!(round_secs(), ROUND_SECS);
        assert_eq!(agent_run_timeout_secs(), AGENT_RUN_TIMEOUT_SECS);
        assert_eq!(prompts_per_round(), PROMPTS_PER_ROUND);
        assert_eq!(manual_daily_run_quota(), MANUAL_DAILY_RUN_QUOTA);
    }

    #[test]
    fn unscored_epoch_clock_rejects_at_limit() {
        assert!(!unscored_epochs_elapsed(100, 104));
        assert!(unscored_epochs_elapsed(100, 105));
        assert!(unscored_epochs_elapsed(100, 200));
        assert!(!unscored_epochs_elapsed(100, 100));
    }

    #[test]
    fn scheduled_cap_covers_a_full_day_of_rounds() {
        // One shared prompt per round → 10 scheduled runs/day; the scheduled
        // cap must still sit above that full-day volume.
        assert_eq!(rounds_per_day_effective(), ROUNDS_PER_DAY);
        assert_eq!(PROMPTS_PER_ROUND, 1);
        assert_eq!(UNSCORED_EPOCH_LIMIT, 5);
        assert_eq!(
            scheduled_runs_per_day(),
            u32::try_from(ROUNDS_PER_DAY).unwrap() * u32::try_from(PROMPTS_PER_ROUND).unwrap()
        );
        assert_eq!(scheduled_runs_per_day(), 10);
        assert!(scheduled_daily_run_cap() >= scheduled_runs_per_day());
        assert_eq!(
            scheduled_daily_run_cap(),
            scheduled_runs_per_day() * SCHEDULED_DAILY_RUN_HEADROOM
        );
        assert!(daily_run_quota() > scheduled_runs_per_day());
        // An operator override below the schedule clamps back up to it, so no
        // env value can re-create the lockout. (Own var name: env is global.)
        let floor = u64::from(scheduled_runs_per_day());
        std::env::set_var("BASE_SCHEDULED_CAP_CLAMP_TEST", "1");
        assert_eq!(env_u64("BASE_SCHEDULED_CAP_CLAMP_TEST", 60, floor), floor);
        std::env::remove_var("BASE_SCHEDULED_CAP_CLAMP_TEST");
    }

    #[test]
    fn env_knob_parse_and_clamp() {
        assert_eq!(env_u64("DEFINITELY_UNSET_BASE_KNOB", 42, 1), 42);
        std::env::set_var("BASE_KNOB_TEST_PARSE", "7");
        assert_eq!(env_u64("BASE_KNOB_TEST_PARSE", 42, 1), 7);
        std::env::set_var("BASE_KNOB_TEST_PARSE", "0");
        assert_eq!(env_u64("BASE_KNOB_TEST_PARSE", 42, 1), 1);
        std::env::set_var("BASE_KNOB_TEST_PARSE", "junk");
        assert_eq!(env_u64("BASE_KNOB_TEST_PARSE", 42, 1), 42);
        std::env::remove_var("BASE_KNOB_TEST_PARSE");
    }

    #[test]
    fn round_id_aligned() {
        assert_eq!(round_id_at_with(0, ROUND_SECS), 0);
        assert_eq!(round_id_at_with(ROUND_SECS - 1, ROUND_SECS), 0);
        assert_eq!(round_id_at_with(ROUND_SECS, ROUND_SECS), 1);
        assert_eq!(round_id_at_with(86_399, ROUND_SECS), 9);
        assert_eq!(round_id_at_with(86_400, ROUND_SECS), 10);
        // Compressed staging rounds: 15-minute slots.
        assert_eq!(round_id_at_with(86_400, 900), 96);
    }

    #[test]
    fn day_round_alignment() {
        assert_eq!(day_index_for_round(0), 0);
        assert_eq!(day_index_for_round(9), 0);
        assert_eq!(day_index_for_round(10), 1);
        assert_eq!(rounds_for_day(0), (0, 9));
        assert_eq!(rounds_for_day(1), (10, 19));
        assert_eq!(
            day_index_at(86_400),
            day_index_for_round(round_id_at(86_400))
        );
    }
}
