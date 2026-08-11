//!
//! Raw-weight HTTP router, verification pipeline and request/response shapes.
//! Challenge leaves are verified against the **local** owner-signed trust root
//! (D18 defence in depth) under domain tag `base-rawweight-v1`, then stored
//! under unique key `(challenge_id, epoch, miner_hotkey)`. A later leaf with a
//! different `payload_digest` tip-supersedes the prior row (202 +
//! `superseded: true`); identical digest remains 409. The store plane itself
//! lives in [`crate::weights_store`].

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use crypto::{domain, verify_raw, KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};
use trustroot::ChallengesBody;
use uuid::Uuid;

use crate::api::GatewayState;
pub use gateway_core::weights_store::{
    IngressError, MemoryRawWeightStore, RawWeightAccepted, RawWeightRequest, RawWeightRow,
    RawWeightStore, ScoreOrAbsenceWire, StoreError,
};

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
    let (row, superseded) = accept_raw_weight(st.challenges.as_ref(), st.weights.as_ref(), &req)?;
    Ok((
        StatusCode::ACCEPTED,
        Json(RawWeightAccepted::from_row(&row, superseded)),
    ))
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

/// Verify + store a single raw-weight leaf (insert or tip supersede).
///
/// Returns `(row, superseded)` where `superseded` is true when an earlier
/// leaf for the same key was replaced because `payload_digest` changed.
///
/// # Errors
///
/// See [`IngressError`].
pub fn accept_raw_weight(
    challenges: &ChallengesBody,
    store: &dyn RawWeightStore,
    req: &RawWeightRequest,
) -> Result<(RawWeightRow, bool), IngressError> {
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

    let miner_hex = hex::encode(miner);
    let superseded = store
        .get(&req.challenge_id, req.epoch, &miner_hex)
        .is_some_and(|prev| prev.payload_digest != digest);

    let row = RawWeightRow {
        id: Uuid::new_v4(),
        challenge_id: req.challenge_id.clone(),
        epoch: req.epoch,
        miner_hotkey: miner_hex,
        kind,
        score: score_value,
        absence_reason,
        payload,
        payload_digest: digest,
        challenge_sig: sig.to_vec(),
    };

    match store.insert(row) {
        Ok(stored) => Ok((stored, superseded)),
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
        let (row, superseded) = accept_raw_weight(&body(pk), &store, &req).unwrap();
        assert_eq!(row.score, Some(7));
        assert!(!superseded);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn unit_accept_supersedes_when_digest_changes() {
        let (sk, pk) = kp();
        let miner = [9u8; 32];
        let store = MemoryRawWeightStore::new();
        let challenges = body(pk);
        let mk = |value: u64| {
            let scale = RawWeightBodyV1 {
                challenge_id: b"c1".to_vec(),
                miner_hotkey: miner,
                epoch: 1,
                score_or_absence: ScoreOrAbsenceScale::Score { value },
            };
            let payload = scale.encode();
            let sig = sign_raw(&sk, domain::RAW_WEIGHT, &payload).unwrap();
            RawWeightRequest {
                challenge_id: "c1".into(),
                miner_hotkey: hex::encode(miner),
                epoch: 1,
                score_or_absence: ScoreOrAbsenceWire::Score { value },
                challenge_sig: hex::encode(sig),
            }
        };
        let (first, _) = accept_raw_weight(&challenges, &store, &mk(7)).unwrap();
        assert_eq!(first.score, Some(7));
        let (second, superseded) = accept_raw_weight(&challenges, &store, &mk(99)).unwrap();
        assert!(superseded);
        assert_eq!(second.score, Some(99));
        assert_eq!(
            store.get("c1", 1, &hex::encode(miner)).unwrap().score,
            Some(99)
        );
        // Identical digest replay → conflict.
        let err = accept_raw_weight(&challenges, &store, &mk(99)).unwrap_err();
        assert!(matches!(err, IngressError::Conflict { .. }));
    }
}
