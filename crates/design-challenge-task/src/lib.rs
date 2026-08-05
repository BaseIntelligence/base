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

/// Prompts selected per round (~2–3 × harness under daily quota).
///
/// Default only — runtime code must use [`prompts_per_round`]
/// (`DESIGN_PROMPTS_PER_ROUND` override).
pub const PROMPTS_PER_ROUND: usize = 3;

/// Rolling scoring window in rounds (`challenge_scoring_version = 3`): the leaf
/// projection shares `SCORE_MAX` by round-win points over the last
/// [`SCORING_WINDOW_ROUNDS`] rounds, cheat excluded.
pub const SCORING_WINDOW_ROUNDS: u64 = 10;

/// Max sandboxed runs per hotkey per UTC day.
pub const DAILY_RUN_QUOTA: u32 = 10;

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
