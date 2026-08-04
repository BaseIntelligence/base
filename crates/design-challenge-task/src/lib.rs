//! Design challenge identity constants.
//!
//! ```text
//! challenge_id     = "design"
//! scoring_version  = 2
//! round_id domain  = b"base-design-round-id-v1"
//! submission domain = b"base-design-submission-v1"
//! pair_id domain   = b"base-design-pair-id-v1"
//! ```

#![forbid(unsafe_code)]

/// Normative challenge id (trust-root / leaf `challenge_id` string).
pub const CHALLENGE_ID: &str = "design";

/// UTF-8 bytes of [`CHALLENGE_ID`].
pub const CHALLENGE_ID_BYTES: &[u8] = b"design";

/// Live `challenge_scoring_version` for design (daily share ≥2 wins).
pub const SCORING_VERSION: u16 = 2;

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
pub const ROUND_SECS: u64 = 8_640;

/// Rounds per UTC calendar day.
pub const ROUNDS_PER_DAY: u64 = 10;

/// Sandbox agent run wall-clock timeout (30 minutes). Distinct from round length.
pub const AGENT_RUN_TIMEOUT_SECS: u64 = 1_800;

/// Minimum round wins in a UTC day before a miner shares that day's reward pool.
pub const MIN_DAILY_WINS: u32 = 2;

/// Prompts selected per round (~2–3 × harness under daily quota).
pub const PROMPTS_PER_ROUND: usize = 3;

/// Max sandboxed runs per hotkey per UTC day.
pub const DAILY_RUN_QUOTA: u32 = 10;

/// Minimum annotations required per pair before Elo consume.
pub const MIN_ANNOTATIONS_PER_PAIR: u32 = 3;

/// Bottom fraction eliminated each round (20%).
pub const ELIMINATION_BOTTOM_BPS: u32 = 2000;

/// Elimination cooldown in rounds (10 rounds = 1 day at 10 rounds/day).
pub const ELIMINATION_COOLDOWN_ROUNDS: u64 = 10;

/// Compute round id from unix seconds.
#[must_use]
pub const fn round_id_at(unix_secs: u64) -> u64 {
    unix_secs / ROUND_SECS
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
        assert_eq!(SCORING_VERSION, 2);
        assert_eq!(SCORE_MAX, 1_000_000);
        assert_eq!(ROUND_SECS * ROUNDS_PER_DAY, 86_400);
        assert_eq!(AGENT_RUN_TIMEOUT_SECS, 1_800);
        assert_eq!(MIN_DAILY_WINS, 2);
    }

    #[test]
    fn round_id_aligned() {
        assert_eq!(round_id_at(0), 0);
        assert_eq!(round_id_at(ROUND_SECS - 1), 0);
        assert_eq!(round_id_at(ROUND_SECS), 1);
        assert_eq!(round_id_at(86_399), 9);
        assert_eq!(round_id_at(86_400), 10);
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
