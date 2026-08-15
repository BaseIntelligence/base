//! PRISM challenge identity constants.
//!
//! Normative pins (`scoring_version = 4` live):
//!
//! ```text
//! challenge_id     = "prism"
//! scoring_version  = 4   # G2 public-suite benchmarks (default)
//! task_id domain   = b"base-prism-task-id-v1"
//! task_blob domain = b"base-prism-task-blob-v1"
//! answer domain    = b"base-prism-answer-v1"
//! receipt domain   = b"base-prism-receipt-v1"
//! ```
//!
//! Distinct from `agent-v1` and `hypertraining` so digests never collide.
//! No Phala/CVM domains — master-centralized Lium eval path.

#![forbid(unsafe_code)]

/// Normative challenge id (trust-root / leaf `challenge_id` string).
pub const CHALLENGE_ID: &str = "prism";

/// UTF-8 bytes of [`CHALLENGE_ID`].
pub const CHALLENGE_ID_BYTES: &[u8] = b"prism";

/// Legacy v2 `challenge_scoring_version` (pure bits/token bpb leaf).
///
/// Retained for `PRISM_SCORING_MODE=shadow` rows and historical leaves.
pub const SCORING_VERSION: u16 = 2;

/// v3 composite scoring version (G1–G8 weighted geometric mean with
/// lexicographic gates, clustered-bootstrap LCB lattice).
///
/// Opt-in via `PRISM_SCORING_MODE=composite` after anchors are calibrated.
pub const SCORING_VERSION_V3: u16 = 3;

/// Live default: equal-weight G2 public-suite accuracies → lattice.
///
/// `PRISM_SCORING_MODE=benchmarks` (default). Tokenizer length no longer
/// farms the emission leaf; see `prism_final::score_from_g2_benchmarks`.
pub const SCORING_VERSION_V4: u16 = 4;

/// Domain tag for PRISM task id digests.
pub const TASK_ID_DOMAIN: &[u8] = b"base-prism-task-id-v1";

/// Domain tag for PRISM task blob digests.
pub const TASK_BLOB_DOMAIN: &[u8] = b"base-prism-task-blob-v1";

/// Domain tag for PRISM answer / score digests.
pub const ANSWER_DOMAIN: &[u8] = b"base-prism-answer-v1";

/// Domain tag for PRISM eval-receipt digests (pod id, image digest, metrics hash).
pub const RECEIPT_DOMAIN: &[u8] = b"base-prism-receipt-v1";

/// Integer score lattice max (same scale as hypertraining / bundle leaves).
pub const SCORE_MAX: u64 = 1_000_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge_id_is_prism() {
        assert_eq!(CHALLENGE_ID, "prism");
        assert_eq!(CHALLENGE_ID_BYTES, b"prism");
        assert_ne!(CHALLENGE_ID, "agent-v1");
        assert_ne!(CHALLENGE_ID, "hypertraining");
    }

    #[test]
    fn scoring_version_and_score_max() {
        assert_eq!(SCORING_VERSION, 2);
        assert_eq!(SCORING_VERSION_V3, 3);
        assert_eq!(SCORING_VERSION_V4, 4);
        assert_eq!(SCORE_MAX, 1_000_000);
    }

    #[test]
    fn domain_tags_distinct_from_siblings() {
        assert_eq!(TASK_ID_DOMAIN, b"base-prism-task-id-v1");
        assert!(!std::str::from_utf8(TASK_ID_DOMAIN)
            .unwrap_or("")
            .contains("agent"));
        assert!(!std::str::from_utf8(TASK_ID_DOMAIN)
            .unwrap_or("")
            .contains("hypertraining"));
    }
}
