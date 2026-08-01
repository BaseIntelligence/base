//! Signed raw-weight ingress (`POST /v1/weights/raw`) — task 26 / `BUNDLE_SPEC` §3.4.
//!
//! Challenge leaves are verified against the **local** owner-signed trust root
//! (D18 defence in depth) under domain tag `base-rawweight-v1`, then appended
//! to an append-only store. Unique key: `(challenge_id, epoch, miner_hotkey)`.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use crypto::{domain, verify_raw, KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::Encode;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use trustroot::ChallengesBody;
use uuid::Uuid;

use crate::api::GatewayState;

/// One persisted raw-weight leaf (memory or DB-shaped).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RawWeightRow {
    /// Stable row id.
    pub id: Uuid,
    /// Challenge id (UTF-8).
    pub challenge_id: String,
    /// Epoch index.
    pub epoch: u64,
    /// Miner hotkey hex (64 chars, lowercase).
    pub miner_hotkey: String,
    /// `"score"` or `"no_score"`.
    pub kind: String,
    /// Present when `kind == "score"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<u64>,
    /// Present when `kind == "no_score"` (reason code as decimal string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absence_reason: Option<String>,
    /// SCALE-encoded `RawWeightBodyV1`.
    #[serde(skip)]
    pub payload: Vec<u8>,
    /// SHA-256 of `payload`.
    #[serde(skip)]
    pub payload_digest: [u8; 32],
    /// Challenge sr25519 signature (64 bytes).
    #[serde(skip)]
    pub challenge_sig: Vec<u8>,
}

/// Append-only raw-weight persistence.
pub trait RawWeightStore: Send + Sync {
    /// Insert a new row. Fails with [`StoreError::Conflict`] if the unique key exists.
    ///
    /// # Errors
    ///
    /// [`StoreError::Conflict`] when `(challenge_id, epoch, miner_hotkey)` already stored.
    fn insert(&self, row: RawWeightRow) -> Result<RawWeightRow, StoreError>;

    /// Lookup by unique key.
    fn get(&self, challenge_id: &str, epoch: u64, miner_hotkey: &str) -> Option<RawWeightRow>;

    /// Number of stored rows.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All rows for a given epoch (any challenge / miner).
    fn list_for_epoch(&self, epoch: u64) -> Vec<RawWeightRow>;
}

/// Store insert failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// Unique key already present; original row is returned for 409 bodies.
    #[error("raw weight already present for challenge/epoch/miner")]
    Conflict {
        /// Unchanged original row.
        original: Box<RawWeightRow>,
    },
    /// Backend failure; the row was **not** persisted.
    #[error("raw weight store backend failure: {0}")]
    Backend(String),
}

/// In-memory append-only store (tests + default runtime until DB hydrate).
#[derive(Debug, Default)]
pub struct MemoryRawWeightStore {
    rows: RwLock<BTreeMap<(String, u64, String), RawWeightRow>>,
}

impl MemoryRawWeightStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl RawWeightStore for MemoryRawWeightStore {
    fn insert(&self, row: RawWeightRow) -> Result<RawWeightRow, StoreError> {
        let key = (
            row.challenge_id.clone(),
            row.epoch,
            row.miner_hotkey.clone(),
        );
        let mut guard = self.rows.write();
        if let Some(existing) = guard.get(&key) {
            return Err(StoreError::Conflict {
                original: Box::new(existing.clone()),
            });
        }
        guard.insert(key, row.clone());
        Ok(row)
    }

    fn get(&self, challenge_id: &str, epoch: u64, miner_hotkey: &str) -> Option<RawWeightRow> {
        let key = (challenge_id.to_owned(), epoch, miner_hotkey.to_owned());
        self.rows.read().get(&key).cloned()
    }

    fn len(&self) -> usize {
        self.rows.read().len()
    }

    fn list_for_epoch(&self, epoch: u64) -> Vec<RawWeightRow> {
        self.rows
            .read()
            .values()
            .filter(|r| r.epoch == epoch)
            .cloned()
            .collect()
    }
}

/// JSON body for `POST /v1/weights/raw`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawWeightRequest {
    /// Challenge id string.
    pub challenge_id: String,
    /// Miner hotkey hex (32 bytes).
    pub miner_hotkey: String,
    /// Epoch.
    pub epoch: u64,
    /// Score or signed absence.
    pub score_or_absence: ScoreOrAbsenceWire,
    /// sr25519 signature hex (64 bytes) over `base-rawweight-v1` ‖ scale(body).
    pub challenge_sig: String,
}

/// Wire form of `ScoreOrAbsence`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoreOrAbsenceWire {
    /// Score present.
    Score {
        /// Raw score value.
        value: u64,
    },
    /// Explicit absence.
    NoScore {
        /// `NoScoreReasonCode` (u8).
        reason: u8,
    },
}

/// SCALE enum matching `BUNDLE_SPEC` §3.3 (`0 = Score`, `1 = NoScore`).
#[derive(Debug, Clone, PartialEq, Eq, Encode)]
#[allow(clippy::cast_possible_truncation)] // SCALE enum index is u8
enum ScoreOrAbsenceScale {
    Score { value: u64 },
    NoScore { reason: u8 },
}

/// SCALE body matching `BUNDLE_SPEC` §3.4 `RawWeightBodyV1`.
#[derive(Debug, Clone, PartialEq, Eq, Encode)]
struct RawWeightBodyV1 {
    challenge_id: Vec<u8>,
    miner_hotkey: [u8; KEY_LEN],
    epoch: u64,
    score_or_absence: ScoreOrAbsenceScale,
}

/// 202 acknowledgement.
#[derive(Debug, Clone, Serialize)]
pub struct RawWeightAccepted {
    /// Stored row id.
    pub id: Uuid,
    /// Challenge id.
    pub challenge_id: String,
    /// Epoch.
    pub epoch: u64,
    /// Miner hotkey hex.
    pub miner_hotkey: String,
    /// `"score"` / `"no_score"`.
    pub kind: String,
    /// Score when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<u64>,
    /// Absence reason when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absence_reason: Option<String>,
}

impl From<&RawWeightRow> for RawWeightAccepted {
    fn from(row: &RawWeightRow) -> Self {
        Self {
            id: row.id,
            challenge_id: row.challenge_id.clone(),
            epoch: row.epoch,
            miner_hotkey: row.miner_hotkey.clone(),
            kind: row.kind.clone(),
            score: row.score,
            absence_reason: row.absence_reason.clone(),
        }
    }
}

/// Ingress errors → HTTP.
#[derive(Debug, Error)]
pub enum IngressError {
    /// Malformed JSON fields / hex.
    #[error("bad request: {0}")]
    BadRequest(String),
    /// Signature does not verify under the trust-root challenge key.
    #[error("unauthorized: invalid challenge signature")]
    Unauthorized,
    /// Challenge id absent from local trust root.
    #[error("challenge not registered")]
    UnknownChallenge,
    /// Unique key already present.
    #[error("conflict: raw weight already stored")]
    Conflict {
        /// Original row (unchanged).
        original: Box<RawWeightRow>,
    },
    /// Store backend refused or failed the write; nothing was persisted.
    #[error("storage backend unavailable: {0}")]
    Backend(String),
}

impl IntoResponse for IngressError {
    fn into_response(self) -> Response {
        let plain = || serde_json::json!({ "error": self.to_string() });
        let (status, body) = match &self {
            Self::BadRequest(msg) => (StatusCode::BAD_REQUEST, serde_json::json!({ "error": msg })),
            Self::Unauthorized => (StatusCode::UNAUTHORIZED, plain()),
            Self::UnknownChallenge => (StatusCode::NOT_FOUND, plain()),
            Self::Backend(_) => (StatusCode::SERVICE_UNAVAILABLE, plain()),
            Self::Conflict { original } => (
                StatusCode::CONFLICT,
                serde_json::json!({
                    "error": self.to_string(),
                    "original": RawWeightAccepted::from(original.as_ref()),
                }),
            ),
        };
        (status, Json(body)).into_response()
    }
}

/// Mount `POST /v1/weights/raw`.
pub fn weights_router(state: GatewayState) -> Router {
    Router::new()
        .route("/v1/weights/raw", post(post_raw_weight))
        .with_state(state)
}

async fn post_raw_weight(
    State(st): State<GatewayState>,
    Json(req): Json<RawWeightRequest>,
) -> Result<(StatusCode, Json<RawWeightAccepted>), IngressError> {
    let row = accept_raw_weight(st.challenges.as_ref(), st.weights.as_ref(), &req)?;
    Ok((StatusCode::ACCEPTED, Json(RawWeightAccepted::from(&row))))
}

/// Verify + append a single raw-weight leaf.
///
/// # Errors
///
/// See [`IngressError`].
pub fn accept_raw_weight(
    challenges: &ChallengesBody,
    store: &dyn RawWeightStore,
    req: &RawWeightRequest,
) -> Result<RawWeightRow, IngressError> {
    if req.challenge_id.is_empty() {
        return Err(IngressError::BadRequest(
            "challenge_id must be non-empty".into(),
        ));
    }
    let challenge_id_bytes = req.challenge_id.as_bytes();
    let entry = challenges
        .get(challenge_id_bytes)
        .ok_or(IngressError::UnknownChallenge)?;

    let miner = parse_hotkey_hex(&req.miner_hotkey)
        .map_err(|e| IngressError::BadRequest(format!("miner_hotkey: {e}")))?;
    let sig = parse_sig_hex(&req.challenge_sig)
        .map_err(|e| IngressError::BadRequest(format!("challenge_sig: {e}")))?;

    let (soa_scale, kind, score_value, absence_reason) = match &req.score_or_absence {
        ScoreOrAbsenceWire::Score { value } => (
            ScoreOrAbsenceScale::Score { value: *value },
            "score".to_owned(),
            Some(*value),
            None,
        ),
        ScoreOrAbsenceWire::NoScore { reason } => (
            ScoreOrAbsenceScale::NoScore { reason: *reason },
            "no_score".to_owned(),
            None,
            Some(reason.to_string()),
        ),
    };

    let body = RawWeightBodyV1 {
        challenge_id: challenge_id_bytes.to_vec(),
        miner_hotkey: miner,
        epoch: req.epoch,
        score_or_absence: soa_scale,
    };
    let payload = body.encode();

    verify_raw(&entry.public_key, domain::RAW_WEIGHT, &payload, &sig)
        .map_err(|_| IngressError::Unauthorized)?;

    let payload_digest = Sha256::digest(&payload);
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&payload_digest);

    let row = RawWeightRow {
        id: Uuid::new_v4(),
        challenge_id: req.challenge_id.clone(),
        epoch: req.epoch,
        miner_hotkey: hex::encode(miner),
        kind,
        score: score_value,
        absence_reason,
        payload,
        payload_digest: digest,
        challenge_sig: sig.to_vec(),
    };

    match store.insert(row) {
        Ok(stored) => Ok(stored),
        Err(StoreError::Conflict { original }) => Err(IngressError::Conflict { original }),
        Err(StoreError::Backend(msg)) => Err(IngressError::Backend(msg)),
    }
}

fn parse_hotkey_hex(s: &str) -> Result<[u8; KEY_LEN], String> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let bytes = hex::decode(s).map_err(|e| e.to_string())?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("expected 32 bytes, got {}", v.len()))
}

fn parse_sig_hex(s: &str) -> Result<[u8; SIGNATURE_LEN], String> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let bytes = hex::decode(s).map_err(|e| e.to_string())?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("expected 64 bytes, got {}", v.len()))
}

/// Shared handle type for the weight store.
pub type SharedWeightStore = Arc<dyn RawWeightStore>;

#[cfg(test)]
mod unit_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crypto::sign_raw;
    use rand_core::OsRng;
    use trustroot::{ChallengeEntry, ParticipantPolicy, BPS_DENOM};

    fn kp() -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
        let mini = schnorrkel::MiniSecretKey::generate_with(OsRng);
        let sk = mini.to_bytes();
        let pk = mini
            .expand(schnorrkel::ExpansionMode::Ed25519)
            .to_public()
            .to_bytes();
        (sk, pk)
    }

    fn body(pk: [u8; KEY_LEN]) -> ChallengesBody {
        ChallengesBody {
            challenges: vec![ChallengeEntry {
                id: b"c1".to_vec(),
                public_key: pk,
                emission_share_bps: BPS_DENOM,
                policy: ParticipantPolicy::AllMetagraphHotkeys,
            }],
        }
    }

    #[test]
    fn unit_accept_score_round_trip() {
        let (sk, pk) = kp();
        let miner = [9u8; 32];
        let scale = RawWeightBodyV1 {
            challenge_id: b"c1".to_vec(),
            miner_hotkey: miner,
            epoch: 1,
            score_or_absence: ScoreOrAbsenceScale::Score { value: 7 },
        };
        let payload = scale.encode();
        let sig = sign_raw(&sk, domain::RAW_WEIGHT, &payload).unwrap();
        let store = MemoryRawWeightStore::new();
        let req = RawWeightRequest {
            challenge_id: "c1".into(),
            miner_hotkey: hex::encode(miner),
            epoch: 1,
            score_or_absence: ScoreOrAbsenceWire::Score { value: 7 },
            challenge_sig: hex::encode(sig),
        };
        let row = accept_raw_weight(&body(pk), &store, &req).unwrap();
        assert_eq!(row.score, Some(7));
        assert_eq!(store.len(), 1);
    }
}
