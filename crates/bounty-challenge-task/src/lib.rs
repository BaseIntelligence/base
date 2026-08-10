//! Bounty challenge identity constants and pure epoch scoring.
//!
//! ```text
//! challenge_id     = "bounty"
//! scoring_version  = 1
//! bug_id domain    = b"base-bounty-bug-id-v1"
//! submission domain = b"base-bounty-submission-v1"
//! ```
//!
//! Epoch scoring: target [`TARGET_BUGS`] approved bugs = full [`SCORE_MAX`]
//! miner pool; shortfall burns via UID-0 leaf mass (see [`score_epoch`]).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc, clippy::doc_markdown)]

mod score;
pub use score::{score_epoch, EpochScoreInput, EpochScoreOutcome};

/// Normative challenge id (trust-root / leaf `challenge_id` string).
pub const CHALLENGE_ID: &str = "bounty";

/// UTF-8 bytes of [`CHALLENGE_ID`].
pub const CHALLENGE_ID_BYTES: &[u8] = b"bounty";

/// Live `challenge_scoring_version` for bounty (TARGET_BUGS burn-sink).
pub const SCORING_VERSION: u16 = 1;

/// Domain tag for bug id digests.
pub const BUG_ID_DOMAIN: &[u8] = b"base-bounty-bug-id-v1";

/// Domain tag for submission / intake digests.
pub const SUBMISSION_DOMAIN: &[u8] = b"base-bounty-submission-v1";

/// Integer score lattice max (same scale as other challenges).
pub const SCORE_MAX: u64 = 1_000_000;

/// Approved bugs that fully claim the bounty emission share this epoch.
pub const TARGET_BUGS: u64 = 50;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        assert_eq!(CHALLENGE_ID, "bounty");
        assert_eq!(CHALLENGE_ID_BYTES, b"bounty");
        assert_eq!(SCORING_VERSION, 1);
        assert_eq!(SCORE_MAX, 1_000_000);
        assert_eq!(TARGET_BUGS, 50);
        assert_eq!(BUG_ID_DOMAIN, b"base-bounty-bug-id-v1");
        assert_eq!(SUBMISSION_DOMAIN, b"base-bounty-submission-v1");
    }
}
