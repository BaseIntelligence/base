//! Design challenge identity constants.
//!
//! ```text
//! challenge_id     = "design"
//! scoring_version  = 1
//! round_id domain  = b"base-design-round-id-v1"
//! submission domain = b"base-design-submission-v1"
//! pair_id domain   = b"base-design-pair-id-v1"
//! ```

#![forbid(unsafe_code)]

/// Normative challenge id (trust-root / leaf `challenge_id` string).
pub const CHALLENGE_ID: &str = "design";

/// UTF-8 bytes of [`CHALLENGE_ID`].
pub const CHALLENGE_ID_BYTES: &[u8] = b"design";

/// Live `challenge_scoring_version` for design (integer Elo → lattice).
pub const SCORING_VERSION: u16 = 1;

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

/// Round duration in seconds (6h).
pub const ROUND_SECS: u64 = 21_600;

/// Prompts selected per round (~2–3 × harness under daily quota).
pub const PROMPTS_PER_ROUND: usize = 3;

/// Max sandboxed runs per hotkey per UTC day.
pub const DAILY_RUN_QUOTA: u32 = 10;

/// Minimum annotations required per pair before Elo consume.
pub const MIN_ANNOTATIONS_PER_PAIR: u32 = 3;

/// Bottom fraction eliminated each round (20%).
pub const ELIMINATION_BOTTOM_BPS: u32 = 2000;

/// Elimination cooldown in rounds (4 rounds = 1 day at 6h).
pub const ELIMINATION_COOLDOWN_ROUNDS: u64 = 4;

/// Compute round id from unix seconds.
#[must_use]
pub const fn round_id_at(unix_secs: u64) -> u64 {
    unix_secs / ROUND_SECS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        assert_eq!(CHALLENGE_ID, "design");
        assert_eq!(CHALLENGE_ID_BYTES, b"design");
        assert_eq!(SCORING_VERSION, 1);
        assert_eq!(SCORE_MAX, 1_000_000);
    }

    #[test]
    fn round_id_aligned() {
        assert_eq!(round_id_at(0), 0);
        assert_eq!(round_id_at(ROUND_SECS - 1), 0);
        assert_eq!(round_id_at(ROUND_SECS), 1);
    }
}
