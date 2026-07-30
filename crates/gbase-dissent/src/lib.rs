//! [`DissentV1`] statements (`BUNDLE_SPEC` §10) under [`gbase_crypto::domain::DISSENT`].
//!
//! Off-chain HTTP evidence only (D5): no merkle root on the weight payload.
//! Task 32: three-outcome policy consumers sign and serve dissents here.

#![forbid(unsafe_code)]

mod policy;

pub use policy::{
    apply_three_outcome_policy, vector_sha256, DissentSigner, EpochDecision, NoSubmitReason,
    RecomputeView, SubmissionIntent, SubmissionSource,
};

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use gbase_crypto::{domain, sign_raw, verify_raw, KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::{Decode, Encode};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Protocol version embedded in dissent bodies (matches bundle protocol).
pub const DISSENT_PROTOCOL_VERSION: u16 = 1;

/// Fully enumerated `DissentReasonCode` (`BUNDLE_SPEC` §10.1).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub enum DissentReasonCode {
    /// Class A — gateway vector ≠ local recompute.
    VectorMismatch = 0,
    /// Leaf challenge signature invalid.
    LeafSignatureInvalid = 1,
    /// Challenge key unknown (D18).
    LeafChallengeKeyUnknown = 2,
    /// Incomplete participant set (D24).
    IncompleteParticipantSet = 3,
    /// Merkle root mismatch.
    MerkleRootMismatch = 4,
    /// Emission shares ≠ local trust root (D23).
    EmissionShareMismatch = 5,
    /// Metagraph root mismatch.
    MetagraphRootMismatch = 6,
    /// Block hash mismatch.
    BlockHashMismatch = 7,
    /// Unsupported protocol version.
    ProtocolVersionUnsupported = 8,
    /// Peer roots conflict (class B).
    PeerRootConflict = 9,
    /// Peer sample below threshold (D26).
    PeerSampleInsufficient = 10,
    /// Surviving mass after quarantine below floor.
    ShareMassBelowThreshold = 11,
    /// Gateway bundle signature invalid.
    BundleSignatureInvalid = 12,
    /// Aggregation overflow.
    AggregationOverflow = 13,
    /// Empty score vector → no-submit.
    EmptyScoreVectorNoSubmit = 14,
    /// UID map mismatch.
    UidMapMismatch = 15,
    /// Measurements digest mismatch.
    MeasurementsDigestMismatch = 16,
    /// Duplicate leaf.
    DuplicateLeaf = 17,
    /// Quarantine left nothing usable / exhausted.
    QuarantineExhausted = 18,
}

impl DissentReasonCode {
    /// All defined codes (0..=18).
    pub const ALL: [Self; 19] = [
        Self::VectorMismatch,
        Self::LeafSignatureInvalid,
        Self::LeafChallengeKeyUnknown,
        Self::IncompleteParticipantSet,
        Self::MerkleRootMismatch,
        Self::EmissionShareMismatch,
        Self::MetagraphRootMismatch,
        Self::BlockHashMismatch,
        Self::ProtocolVersionUnsupported,
        Self::PeerRootConflict,
        Self::PeerSampleInsufficient,
        Self::ShareMassBelowThreshold,
        Self::BundleSignatureInvalid,
        Self::AggregationOverflow,
        Self::EmptyScoreVectorNoSubmit,
        Self::UidMapMismatch,
        Self::MeasurementsDigestMismatch,
        Self::DuplicateLeaf,
        Self::QuarantineExhausted,
    ];

    /// Parse a wire `u8` if it is a known enum variant.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::VectorMismatch),
            1 => Some(Self::LeafSignatureInvalid),
            2 => Some(Self::LeafChallengeKeyUnknown),
            3 => Some(Self::IncompleteParticipantSet),
            4 => Some(Self::MerkleRootMismatch),
            5 => Some(Self::EmissionShareMismatch),
            6 => Some(Self::MetagraphRootMismatch),
            7 => Some(Self::BlockHashMismatch),
            8 => Some(Self::ProtocolVersionUnsupported),
            9 => Some(Self::PeerRootConflict),
            10 => Some(Self::PeerSampleInsufficient),
            11 => Some(Self::ShareMassBelowThreshold),
            12 => Some(Self::BundleSignatureInvalid),
            13 => Some(Self::AggregationOverflow),
            14 => Some(Self::EmptyScoreVectorNoSubmit),
            15 => Some(Self::UidMapMismatch),
            16 => Some(Self::MeasurementsDigestMismatch),
            17 => Some(Self::DuplicateLeaf),
            18 => Some(Self::QuarantineExhausted),
            _ => None,
        }
    }

    /// Wire discriminant.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Whether this code is in the normative enum (always true for typed values).
    #[must_use]
    pub const fn is_spec_enum(self) -> bool {
        self.as_u8() <= 18
    }
}

/// SCALE body under [`domain::DISSENT`].
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct DissentBodyV1 {
    /// Bundle protocol version the validator understands.
    pub protocol_version: u16,
    /// Epoch under dispute.
    pub epoch: u64,
    /// Disputed bundle merkle root, or zeros if none.
    pub bundle_root: [u8; 32],
    /// `sha256(scale(local final_vector))`.
    pub expected_vector_hash: [u8; 32],
    /// `sha256(scale(gateway final_vector))` or zeros if absent.
    pub actual_vector_hash: [u8; 32],
    /// Reason discriminant.
    pub reason_code: u8,
}

impl DissentBodyV1 {
    /// Build a body with typed reason.
    #[must_use]
    pub fn new(
        epoch: u64,
        bundle_root: [u8; 32],
        expected_vector_hash: [u8; 32],
        actual_vector_hash: [u8; 32],
        reason: DissentReasonCode,
    ) -> Self {
        Self {
            protocol_version: DISSENT_PROTOCOL_VERSION,
            epoch,
            bundle_root,
            expected_vector_hash,
            actual_vector_hash,
            reason_code: reason.as_u8(),
        }
    }

    /// Typed reason if the wire code is known.
    #[must_use]
    pub fn reason(&self) -> Option<DissentReasonCode> {
        DissentReasonCode::from_u8(self.reason_code)
    }
}

/// Signed dissent statement.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct DissentV1 {
    /// Body.
    pub body: DissentBodyV1,
    /// Validator hotkey (sr25519 public).
    pub validator_hotkey: [u8; 32],
    /// Signature over `scale(body)` under [`domain::DISSENT`].
    pub signature: [u8; SIGNATURE_LEN],
}

/// Sign / verify errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DissentError {
    /// Crypto layer failure.
    #[error("crypto: {0}")]
    Crypto(String),
    /// Reason code outside the spec enum.
    #[error("unknown dissent reason_code={0}")]
    UnknownReasonCode(u8),
}

impl DissentV1 {
    /// Sign a body with a 32-byte mini-secret.
    ///
    /// # Errors
    ///
    /// Invalid secret or unknown reason code.
    pub fn sign(
        secret: &[u8; KEY_LEN],
        hotkey: [u8; KEY_LEN],
        body: DissentBodyV1,
    ) -> Result<Self, DissentError> {
        if DissentReasonCode::from_u8(body.reason_code).is_none() {
            return Err(DissentError::UnknownReasonCode(body.reason_code));
        }
        let payload = body.encode();
        let signature = sign_raw(secret, domain::DISSENT, &payload)
            .map_err(|e| DissentError::Crypto(e.to_string()))?;
        Ok(Self {
            body,
            validator_hotkey: hotkey,
            signature,
        })
    }

    /// Verify under [`domain::DISSENT`] for `validator_hotkey`.
    ///
    /// # Errors
    ///
    /// Crypto failure or unknown reason code.
    pub fn verify(&self) -> Result<(), DissentError> {
        if DissentReasonCode::from_u8(self.body.reason_code).is_none() {
            return Err(DissentError::UnknownReasonCode(self.body.reason_code));
        }
        let payload = self.body.encode();
        verify_raw(
            &self.validator_hotkey,
            domain::DISSENT,
            &payload,
            &self.signature,
        )
        .map_err(|e| DissentError::Crypto(e.to_string()))
    }

    /// Typed reason (after verify, always known).
    #[must_use]
    pub fn reason_code(&self) -> Option<DissentReasonCode> {
        self.body.reason()
    }
}

/// Wire JSON for `GET /v1/dissent/{epoch}` (hex fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissentJson {
    /// Epoch.
    pub epoch: u64,
    /// Protocol version.
    pub protocol_version: u16,
    /// Bundle root hex.
    pub bundle_root: String,
    /// Expected vector hash hex.
    pub expected_vector_hash: String,
    /// Actual vector hash hex.
    pub actual_vector_hash: String,
    /// Reason code u8.
    pub reason_code: u8,
    /// Hotkey hex.
    pub validator_hotkey: String,
    /// Signature hex.
    pub signature: String,
}

impl DissentJson {
    /// From a signed statement.
    #[must_use]
    pub fn from_dissent(d: &DissentV1) -> Self {
        Self {
            epoch: d.body.epoch,
            protocol_version: d.body.protocol_version,
            bundle_root: hex::encode(d.body.bundle_root),
            expected_vector_hash: hex::encode(d.body.expected_vector_hash),
            actual_vector_hash: hex::encode(d.body.actual_vector_hash),
            reason_code: d.body.reason_code,
            validator_hotkey: hex::encode(d.validator_hotkey),
            signature: hex::encode(d.signature),
        }
    }
}

/// In-memory dissent store (latest per epoch + history).
#[derive(Debug, Default)]
pub struct DissentStore {
    by_epoch: RwLock<BTreeMap<u64, Vec<DissentV1>>>,
}

/// Shared store handle.
pub type SharedDissentStore = Arc<DissentStore>;

impl DissentStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Shared empty store.
    #[must_use]
    pub fn shared() -> SharedDissentStore {
        Arc::new(Self::new())
    }

    /// Persist a verified (or freshly signed) dissent for an epoch.
    pub fn put(&self, dissent: DissentV1) {
        let epoch = dissent.body.epoch;
        let mut map = self
            .by_epoch
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = map.entry(epoch).or_default();
        if !entry.iter().any(|d| d == &dissent) {
            entry.push(dissent);
        }
    }

    /// All dissents for `epoch`.
    #[must_use]
    pub fn get_epoch(&self, epoch: u64) -> Vec<DissentV1> {
        self.by_epoch
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&epoch)
            .cloned()
            .unwrap_or_default()
    }

    /// Count of stored statements for `epoch`.
    #[must_use]
    pub fn count_epoch(&self, epoch: u64) -> usize {
        self.get_epoch(epoch).len()
    }
}

/// Increment `gbase_challenge_quarantined_total` by `n` (number of dropped challenges).
pub fn record_challenges_quarantined(n: u64) {
    if n == 0 {
        return;
    }
    metrics::describe_counter!(
        "gbase_challenge_quarantined_total",
        "Challenges dropped by validator quarantine (D6)"
    );
    metrics::counter!("gbase_challenge_quarantined_total").increment(n);
}

/// Increment class-B no-submit counter (alarm path).
pub fn record_class_b() {
    metrics::describe_counter!(
        "gbase_class_b_total",
        "Epochs where the validator took class B (no weight submission)"
    );
    metrics::counter!("gbase_class_b_total").increment(1);
}

/// Axum router: `GET /v1/dissent/{epoch}`.
pub fn dissent_router(store: SharedDissentStore) -> Router {
    Router::new()
        .route("/v1/dissent/{epoch}", get(get_dissent_epoch))
        .with_state(store)
}

async fn get_dissent_epoch(
    State(store): State<SharedDissentStore>,
    Path(epoch): Path<u64>,
) -> Response {
    let list = store.get_epoch(epoch);
    if list.is_empty() {
        return StatusCode::NOT_FOUND.into_response();
    }
    let json: Vec<DissentJson> = list.iter().map(DissentJson::from_dissent).collect();
    (StatusCode::OK, Json(json)).into_response()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use gbase_crypto::secret_from_bytes;
    use sha2::{Digest, Sha256};

    fn sk(tag: u8) -> [u8; 32] {
        let dig = Sha256::digest([0xD1, tag, 0x55, tag]);
        let mut s = [0u8; 32];
        s.copy_from_slice(&dig);
        s
    }

    fn pk(secret: &[u8; 32]) -> [u8; 32] {
        secret_from_bytes(secret)
            .expect("sk")
            .to_public()
            .to_bytes()
    }

    #[test]
    fn s1_sign_verify_roundtrip_domain_dissent() {
        let secret = sk(1);
        let hotkey = pk(&secret);
        let body = DissentBodyV1::new(
            7,
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            DissentReasonCode::VectorMismatch,
        );
        let d = DissentV1::sign(&secret, hotkey, body).expect("sign");
        d.verify().expect("verify");
        assert_eq!(d.reason_code(), Some(DissentReasonCode::VectorMismatch));
    }

    #[test]
    fn s2_cross_domain_rejected() {
        let secret = sk(2);
        let hotkey = pk(&secret);
        let body = DissentBodyV1::new(
            1,
            [0u8; 32],
            [0u8; 32],
            [0u8; 32],
            DissentReasonCode::PeerRootConflict,
        );
        let payload = body.encode();
        let sig = sign_raw(&secret, domain::ROOT, &payload).expect("sign root");
        let d = DissentV1 {
            body,
            validator_hotkey: hotkey,
            signature: sig,
        };
        assert!(d.verify().is_err());
    }

    #[test]
    fn s3_all_reason_codes_in_spec_enum() {
        for code in DissentReasonCode::ALL {
            assert!(code.is_spec_enum());
            assert_eq!(DissentReasonCode::from_u8(code.as_u8()), Some(code));
        }
        assert!(DissentReasonCode::from_u8(19).is_none());
        assert!(DissentReasonCode::from_u8(255).is_none());
    }

    #[test]
    fn s4_unknown_reason_rejected_on_sign() {
        let secret = sk(3);
        let hotkey = pk(&secret);
        let mut body = DissentBodyV1::new(
            1,
            [0u8; 32],
            [0u8; 32],
            [0u8; 32],
            DissentReasonCode::VectorMismatch,
        );
        body.reason_code = 99;
        let err = DissentV1::sign(&secret, hotkey, body).expect_err("unknown");
        assert!(matches!(err, DissentError::UnknownReasonCode(99)));
    }

    #[test]
    fn s5_store_and_json() {
        let secret = sk(4);
        let hotkey = pk(&secret);
        let body = DissentBodyV1::new(
            42,
            [9u8; 32],
            [8u8; 32],
            [7u8; 32],
            DissentReasonCode::ShareMassBelowThreshold,
        );
        let d = DissentV1::sign(&secret, hotkey, body).expect("sign");
        let store = DissentStore::new();
        store.put(d.clone());
        store.put(d.clone()); // idempotent
        assert_eq!(store.count_epoch(42), 1);
        let j = DissentJson::from_dissent(&d);
        assert_eq!(j.epoch, 42);
        assert_eq!(j.reason_code, 11);
    }
}
