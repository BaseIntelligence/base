//! SCALE wire types for epoch bundles (`BUNDLE_SPEC` §3–§4). Field order is normative.

use gbase_crypto::{KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::{Decode, Encode};

/// Current bundle protocol version (`BUNDLE_SPEC` §2).
pub const PROTOCOL_VERSION: u16 = 1;

/// Aggregation algorithm version required in the body.
pub const ALGORITHM_VERSION: u16 = 1;

/// Maximum challenge id length in bytes.
pub const MAX_CHALLENGE_ID_LEN: usize = 64;

/// `NoScoreReasonCode` (`BUNDLE_SPEC` §3.3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
#[allow(clippy::cast_possible_truncation)] // SCALE enum index is u8
#[repr(u8)]
pub enum NoScoreReasonCode {
    /// Challenge did not invoke the miner this epoch.
    NotAttempted = 0,
    /// Miner failed to respond in time.
    Timeout = 1,
    /// Response failed schema/scoring preconditions.
    InvalidResponse = 2,
    /// Required attestation missing or not `Verified`.
    AttestationNotVerified = 3,
    /// Miner returned an explicit error.
    MinerError = 4,
    /// Challenge rate-limited the miner.
    RateLimited = 5,
    /// Challenge-side fault; still must cover the participant.
    ChallengeInternal = 6,
    /// Reserved; MUST NOT shrink the expected set (D24).
    PolicySkip = 7,
}

/// Score present or explicit absence (`BUNDLE_SPEC` §3.3).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
#[allow(clippy::cast_possible_truncation)] // SCALE enum index is u8
pub enum ScoreOrAbsence {
    /// Miner produced a score.
    Score {
        /// Raw score value.
        value: u64,
    },
    /// Explicit absence signed by the challenge key.
    NoScore {
        /// Reason discriminant.
        reason: NoScoreReasonCode,
    },
}

/// Merkle leaf preimage (`LeafV1`, §3.3).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
pub struct LeafV1 {
    /// Challenge id (`Bytes` / `Vec<u8>`).
    pub challenge_id: Vec<u8>,
    /// Miner hotkey.
    pub miner_hotkey: [u8; KEY_LEN],
    /// Epoch number.
    pub epoch: u64,
    /// Score or absence.
    pub score_or_absence: ScoreOrAbsence,
    /// sr25519 signature over raw-weight body (§3.4).
    pub challenge_sig: [u8; SIGNATURE_LEN],
}

/// Unsigned bundle body (`EpochBundleBodyV1`, §4.1).
///
/// **Field order is normative** — do not reorder without bumping `protocol_version`.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
pub struct EpochBundleBodyV1 {
    /// Must be [`PROTOCOL_VERSION`].
    pub protocol_version: u16,
    /// Epoch index.
    pub epoch: u64,
    /// Subnet netuid.
    pub netuid: u16,
    /// Inclusive epoch end block.
    pub block_b: u64,
    /// Hash of `block_b`.
    pub block_hash: [u8; 32],
    /// `sha256(scale(metagraph rows at block_hash))` (§4.3).
    pub metagraph_root: [u8; 32],
    /// Aggregation algorithm version (must be 1).
    pub algorithm_version: u16,
    /// Emission shares sorted by challenge id.
    pub emission_shares: Vec<(Vec<u8>, u16)>,
    /// Local measurements trust-root body digest.
    pub measurements_digest: [u8; 32],
    /// Hotkey → uid, sorted by hotkey.
    pub uid_map: Vec<([u8; KEY_LEN], u16)>,
    /// Leaves sorted by `scale(challenge_id, miner_hotkey)`.
    pub leaves: Vec<LeafV1>,
    /// Merkle root over leaf preimages.
    pub merkle_root: [u8; 32],
    /// Final weight vector sorted by uid.
    pub final_vector: Vec<(u16, u16)>,
    /// Gateway hotkey that signs the envelope.
    pub gateway_hotkey: [u8; KEY_LEN],
}

/// Signed epoch bundle envelope (`EpochBundleV1`, §4.1).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
pub struct EpochBundleV1 {
    /// Bundle body.
    pub body: EpochBundleBodyV1,
    /// sr25519 over tag `gbase-bundle-v1` ‖ `scale(body)`.
    pub gateway_sig: [u8; SIGNATURE_LEN],
}

/// Metagraph row for `metagraph_root` preimage (§4.3).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
pub struct MetagraphRow {
    /// Hotkey.
    pub hotkey: [u8; KEY_LEN],
    /// UID.
    pub uid: u16,
    /// Stake at `block_hash`.
    pub stake: u64,
}

/// Raw-weight body under challenge signature (§3.4).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[codec(crate = parity_scale_codec)]
pub struct RawWeightBodyV1 {
    /// Challenge id.
    pub challenge_id: Vec<u8>,
    /// Miner hotkey.
    pub miner_hotkey: [u8; KEY_LEN],
    /// Epoch.
    pub epoch: u64,
    /// Score or absence.
    pub score_or_absence: ScoreOrAbsence,
}

/// Local trust material for verify (D18/D23).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalTrustRoot {
    /// Owner-signed challenges body (local disk).
    pub challenges: gbase_trustroot::ChallengesBody,
    /// `sha256(scale(measurements body))` from local measurements root.
    pub measurements_digest: [u8; 32],
}

/// Bundle that passed all verification checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBundle {
    /// Verified body.
    pub body: EpochBundleBodyV1,
}

/// Normative field names of [`EpochBundleBodyV1`] in SCALE encode order (§4.1).
pub const BODY_FIELD_ORDER: &[&str] = &[
    "protocol_version",
    "epoch",
    "netuid",
    "block_B",
    "block_hash",
    "metagraph_root",
    "algorithm_version",
    "emission_shares",
    "measurements_digest",
    "uid_map",
    "leaves",
    "merkle_root",
    "final_vector",
    "gateway_hotkey",
];

/// Minimal empty-leaf body used as a field-order canary.
#[must_use]
pub fn field_order_canary_body() -> EpochBundleBodyV1 {
    EpochBundleBodyV1 {
        protocol_version: 1,
        epoch: 0,
        netuid: 0,
        block_b: 0,
        block_hash: [0u8; 32],
        metagraph_root: [0u8; 32],
        algorithm_version: 1,
        emission_shares: Vec::new(),
        measurements_digest: [0u8; 32],
        uid_map: Vec::new(),
        leaves: Vec::new(),
        merkle_root: gbase_merkle::EMPTY_ROOT,
        final_vector: Vec::new(),
        gateway_hotkey: [0u8; KEY_LEN],
    }
}

/// First bytes of the canary body encoding: `protocol_version` LE u16 = 1, then epoch u64 = 0.
#[must_use]
pub fn field_order_canary_prefix() -> [u8; 10] {
    [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
}
