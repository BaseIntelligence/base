//! Hypertraining challenge identity constants (`challenge_id`, scoring version, domain tags).
//!
//! Domain tags are distinct from agent-v1 (`base-agent-*`) so digests never collide across
//! challenges. Digest helpers land in later todos; this crate freezes the normative strings.
//!
//! # Normative pins (`scoring_version = 1`)
//!
//! ```text
//! challenge_id        = "hypertraining"
//! scoring_version     = 1
//! task_id domain      = b"base-hypertraining-task-id-v1"
//! task_blob domain    = b"base-hypertraining-task-blob-v1"
//! answer domain       = b"base-hypertraining-answer-v1"
//! receipt domain      = b"base-hypertraining-receipt-v1"
//! ```

#![forbid(unsafe_code)]

/// Normative challenge id (trust-root / leaf `challenge_id` string).
pub const CHALLENGE_ID: &str = "hypertraining";

/// UTF-8 bytes of [`CHALLENGE_ID`].
pub const CHALLENGE_ID_BYTES: &[u8] = b"hypertraining";

/// Live `challenge_scoring_version` for hypertraining (integer score map v1).
pub const SCORING_VERSION: u16 = 1;

/// Domain tag for hypertraining task id digests (distinct from `base-agent-task-id-v*`).
pub const TASK_ID_DOMAIN: &[u8] = b"base-hypertraining-task-id-v1";

/// Domain tag for hypertraining task blob digests.
pub const TASK_BLOB_DOMAIN: &[u8] = b"base-hypertraining-task-blob-v1";

/// Domain tag for hypertraining answer / score digests.
pub const ANSWER_DOMAIN: &[u8] = b"base-hypertraining-answer-v1";

/// Domain tag for hypertraining work-receipt style digests (later orchestration).
pub const RECEIPT_DOMAIN: &[u8] = b"base-hypertraining-receipt-v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge_id_is_hypertraining() {
        assert_eq!(CHALLENGE_ID, "hypertraining");
        assert_eq!(CHALLENGE_ID_BYTES, b"hypertraining");
        assert_eq!(CHALLENGE_ID.as_bytes(), CHALLENGE_ID_BYTES);
    }

    #[test]
    fn scoring_version_is_one() {
        assert_eq!(SCORING_VERSION, 1);
    }

    #[test]
    fn domain_tags_match_normative_bytes() {
        assert_eq!(TASK_ID_DOMAIN, b"base-hypertraining-task-id-v1");
        assert_eq!(TASK_BLOB_DOMAIN, b"base-hypertraining-task-blob-v1");
        assert_eq!(ANSWER_DOMAIN, b"base-hypertraining-answer-v1");
        assert_eq!(RECEIPT_DOMAIN, b"base-hypertraining-receipt-v1");
    }

    #[test]
    fn domain_tags_distinct_from_each_other() {
        let tags = [
            TASK_ID_DOMAIN,
            TASK_BLOB_DOMAIN,
            ANSWER_DOMAIN,
            RECEIPT_DOMAIN,
        ];
        for (i, a) in tags.iter().enumerate() {
            for (j, b) in tags.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "domain tags at {i} and {j} must differ");
                }
            }
        }
    }

    #[test]
    fn domain_tags_distinct_from_base_agent() {
        const AGENT_TASK_ID_V1: &[u8] = b"base-agent-task-id-v1";
        const AGENT_TASK_ID_V2: &[u8] = b"base-agent-task-id-v2";
        const AGENT_TASK_BLOB_V1: &[u8] = b"base-agent-task-blob-v1";
        const AGENT_TASK_BLOB_V2: &[u8] = b"base-agent-task-blob-v2";
        const AGENT_ANSWER_V1: &[u8] = b"base-agent-answer-v1";
        const AGENT_ANSWER_V2: &[u8] = b"base-agent-answer-v2";
        const AGENT_RECEIPT: &[u8] = b"base-agent-work-receipt-v1";
        const ATTEST: &[u8] = b"base-attest-v1";

        let ours = [
            TASK_ID_DOMAIN,
            TASK_BLOB_DOMAIN,
            ANSWER_DOMAIN,
            RECEIPT_DOMAIN,
        ];
        let agent = [
            AGENT_TASK_ID_V1,
            AGENT_TASK_ID_V2,
            AGENT_TASK_BLOB_V1,
            AGENT_TASK_BLOB_V2,
            AGENT_ANSWER_V1,
            AGENT_ANSWER_V2,
            AGENT_RECEIPT,
            ATTEST,
        ];
        for o in ours {
            for a in agent {
                assert_ne!(o, a, "hypertraining domain must not equal agent/attest tag");
            }
            assert!(
                !o.starts_with(b"base-agent-"),
                "hypertraining domain must not use base-agent- prefix: {o:?}"
            );
        }
    }

    #[test]
    fn challenge_id_distinct_from_agent_v1() {
        assert_ne!(CHALLENGE_ID, "agent-v1");
        assert_ne!(CHALLENGE_ID_BYTES, b"agent-v1");
    }
}
