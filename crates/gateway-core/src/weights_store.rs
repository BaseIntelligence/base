//! Raw-weight store plane: persisted leaf shape, store trait, in-memory
//! implementation and error type. The HTTP router and the verification
//! pipeline stay in [`crate::weights`].

use std::collections::BTreeMap;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

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

/// Raw-weight persistence with tip supersede.
///
/// Unique key: `(challenge_id, epoch, miner_hotkey)`. A later leaf with a
/// **different** `payload_digest` replaces the stored row (tip tracking). An
/// identical digest is a conflict (idempotent replay).
pub trait RawWeightStore: Send + Sync {
    /// Insert a new row, or replace when the digest changes for the same key.
    ///
    /// # Errors
    ///
    /// [`StoreError::Conflict`] when the key exists with the **same**
    /// `payload_digest`.
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
    /// Unique key already present with the same digest; original for 409 bodies.
    #[error("raw weight already present for challenge/epoch/miner")]
    Conflict {
        /// Unchanged original row.
        original: Box<RawWeightRow>,
    },
    /// Backend failure; the row was **not** persisted.
    #[error("raw weight store backend failure: {0}")]
    Backend(String),
}

/// In-memory store with tip supersede (tests + default runtime until DB hydrate).
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
            if existing.payload_digest == row.payload_digest {
                return Err(StoreError::Conflict {
                    original: Box::new(existing.clone()),
                });
            }
            // Tip supersede: digest changed → replace in place.
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
    /// True when an earlier leaf for the same key was replaced (digest change).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub superseded: bool,
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
            superseded: false,
        }
    }
}

impl RawWeightAccepted {
    /// Build an ack, marking tip supersede when requested.
    #[must_use]
    pub fn from_row(row: &RawWeightRow, superseded: bool) -> Self {
        let mut ack = Self::from(row);
        ack.superseded = superseded;
        ack
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
    /// Unique key already present with the same digest.
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
