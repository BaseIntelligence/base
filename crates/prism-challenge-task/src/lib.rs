//! PRISM challenge identity constants.
//!
//! Normative pins (`scoring_version = 2`):
//!
//! ```text
//! challenge_id     = "prism"
//! scoring_version  = 2
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

/// Live `challenge_scoring_version` for PRISM (integer score map).
///
/// v1 blended measured bpb with an LLM quality vote (0.7/0.3). v2 drops the
/// LLM vote from the number entirely: the score is **pure bpb**, and the
/// LLM/coherence review is an anti-cheat GATE recording an audit event
/// (`Copied`/`Suspicious` still hard-zero). Older v1 leaves stay attributable
/// under their version tag.
pub const SCORING_VERSION: u16 = 2;

/// v3 composite scoring version (G1–G8 weighted geometric mean with
/// lexicographic gates, clustered-bootstrap LCB lattice).
///
/// **Not live**: v2 remains the live default until the anchor set is
/// calibrated on the reference baselines and governance flips
/// `PRISM_SCORING_MODE` from `shadow` to `composite`. Only rows scored under
/// composite mode carry this version tag; the bundle `protocol_version`
/// stays 1.
pub const SCORING_VERSION_V3: u16 = 3;

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
