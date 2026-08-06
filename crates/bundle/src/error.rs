//! Bundle error types.

use thiserror::Error;

/// Bundle encode / verify failures (map to dissent reason codes in callers).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BundleError {
    /// `protocol_version` not supported.
    #[error("protocol version unsupported: {0}")]
    ProtocolVersionUnsupported(u16),
    /// Gateway signature invalid.
    #[error("bundle gateway signature invalid")]
    BundleSignatureInvalid,
    /// Leaf challenge signature invalid.
    #[error("leaf challenge signature invalid")]
    LeafSignatureInvalid,
    /// Challenge id / key unknown to local trust root (D18).
    #[error("leaf challenge key unknown (D18)")]
    LeafChallengeKeyUnknown,
    /// Participant set incomplete or extra leaf (D24).
    #[error("incomplete participant set (D24)")]
    IncompleteParticipantSet,
    /// Merkle root does not recompute.
    #[error("merkle root mismatch")]
    MerkleRootMismatch,
    /// Emission shares ≠ local trust root (D23).
    #[error("emission share mismatch (D23)")]
    EmissionShareMismatch,
    /// Share bps do not sum to `10_000`.
    #[error("emission shares do not sum to 10000")]
    EmissionSharesSumInvalid,
    /// `uid_map` mismatch.
    #[error("uid_map mismatch")]
    UidMapMismatch,
    /// `metagraph_root` mismatched.
    #[error("metagraph root mismatch")]
    MetagraphRootMismatch,
    /// `block_hash` ≠ chain.
    #[error("block hash mismatch: block_b={block_b} chain={chain} bundle={bundle}")]
    BlockHashMismatch {
        /// `block_b` the bundle pinned.
        block_b: u64,
        /// Hash the chain reports for `block_b` (hex).
        chain: String,
        /// Hash inside the bundle body (hex).
        bundle: String,
    },
    /// Measurements digest mismatch.
    #[error("measurements digest mismatch")]
    MeasurementsDigestMismatch,
    /// Duplicate leaf `(challenge_id, miner_hotkey)`.
    #[error("duplicate leaf")]
    DuplicateLeaf,
    /// Leaves not in canonical sort order.
    #[error("leaves not canonically sorted")]
    LeafSortOrder,
    /// `algorithm_version` unsupported or final vector mismatch.
    #[error("final vector or algorithm_version mismatch")]
    FinalVectorMismatch,
    /// Challenge id longer than max.
    #[error("challenge id too long")]
    ChallengeIdTooLong,
    /// Chain client error.
    #[error("chain error: {0}")]
    Chain(String),
    /// Aggregation overflow / error.
    #[error("aggregation error: {0}")]
    Aggregation(String),
    /// SCALE codec failure.
    #[error("scale codec error: {0}")]
    Codec(String),
    /// Crypto failure.
    #[error("crypto error: {0}")]
    Crypto(String),
    /// Hotkey length / metagraph shape invalid.
    #[error("invalid metagraph shape: {0}")]
    InvalidMetagraph(String),
}
